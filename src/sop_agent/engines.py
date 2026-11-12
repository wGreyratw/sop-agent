"""引擎接口：LangGraph 图只通过这里的方法使用流程引擎。

两种流程形状：
- 排障（kind: troubleshooting，电信）：症状 → 按路径逐项检查和修复，见 flow.py
- 请求清单（kind: requests，零售）：一通电话里的多个请求，每个都要查状态、收集细节、顾客确认后执行，见 requests.py

每轮的处理顺序（graph.py 的 extract 和 guard 节点）：
1. observe：用代码读工具调用和结果，更新账本（可靠，不经过模型）
2. extract_messages / apply_extraction：用一次模型调用，把对话里的新信息写进账本
3. next_step / summary：算出下一步，生成给回复模型的提示
4. check_reply：守卫检查回复草稿
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import yaml
from tau2.data_model.message import SystemMessage, UserMessage

from .extract import extract_prompt, extract_request, parse_extraction
from .flow import FlowEngine, FlowSpec, Ledger, Step
from .guard import check_reply


class Engine(Protocol):
    extract_thinking: bool
    """提取调用是否开启模型的思考模式。"""

    def new_ledger(self) -> dict[str, Any]: ...

    def observe(self, ledger: dict[str, Any], calls: list[Any], results: list[Any]) -> dict[str, Any]:
        """calls：Agent 上一条消息里的工具调用；results：本轮新到的工具结果。"""

    def extract_messages(self, ledger: dict[str, Any], recent: list[Any]) -> list[Any]: ...

    def apply_extraction(self, ledger: dict[str, Any], content: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
        """返回新账本和写入轨迹的提取事件。"""

    def next_step(self, ledger: dict[str, Any]) -> Step: ...

    def summary(self, ledger: dict[str, Any]) -> str: ...

    def check_reply(self, reply: Any, ledger: dict[str, Any], transferred: bool) -> str | None: ...


def succeeded_calls(calls: list[Any], results: list[Any]) -> list[tuple[Any, Any]]:
    """配对工具调用和它成功的结果。"""
    ok = {getattr(r, "id", None): r for r in results if not getattr(r, "error", False)}
    return [(call, ok[call.id]) for call in calls if call.id in ok]


class TroubleshootingEngine:
    """电信排障：包装 FlowEngine，行为和重构前完全相同。"""

    extract_thinking = True  # 和冻结的评测设置保持一致

    def __init__(self, spec: FlowSpec) -> None:
        self.flow = FlowEngine(spec)
        self.spec = spec
        self._extract_system = SystemMessage(role="system", content=extract_prompt(spec))

    def new_ledger(self) -> dict[str, Any]:
        return Ledger().model_dump()

    def observe(self, ledger: dict[str, Any], calls: list[Any], results: list[Any]) -> dict[str, Any]:
        # Agent 上一步调用了修复工具且成功：清空症状，要求重新检查
        current = Ledger.model_validate(ledger)
        for call, _ in succeeded_calls(calls, results):
            current = self.flow.apply_tool_fix(current, call.name)
        return current.model_dump()

    def extract_messages(self, ledger: dict[str, Any], recent: list[Any]) -> list[Any]:
        request = extract_request(Ledger.model_validate(ledger).facts, recent)
        return [self._extract_system, UserMessage(role="user", content=request)]

    def apply_extraction(self, ledger: dict[str, Any], content: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
        found = parse_extraction(content)
        before = Ledger.model_validate(ledger)
        updated = self.flow.update(before, found)
        event = {"event": "extract", "found": found, "facts": updated.facts, "fixes": updated.fixes[len(before.fixes):]}
        return updated.model_dump(), event

    def next_step(self, ledger: dict[str, Any]) -> Step:
        return self.flow.next_step(Ledger.model_validate(ledger))

    def summary(self, ledger: dict[str, Any]) -> str:
        return self.flow.summary(Ledger.model_validate(ledger))

    def check_reply(self, reply: Any, ledger: dict[str, Any], transferred: bool) -> str | None:
        return check_reply(reply, self.flow, Ledger.model_validate(ledger), transferred)


def make_engine(spec: Any) -> Engine:
    from .requests import RequestEngine, RequestSpec

    if isinstance(spec, FlowSpec):
        return TroubleshootingEngine(spec)
    if isinstance(spec, RequestSpec):
        return RequestEngine(spec)
    raise TypeError(f"Unsupported flow spec: {type(spec).__name__}")


def load_spec(path: str | Path) -> Any:
    """按 YAML 里的 kind 选择流程形状，默认是排障。"""
    from .requests import RequestSpec

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data.get("kind", "troubleshooting") == "requests":
        return RequestSpec.model_validate(data)
    return FlowSpec.model_validate({k: v for k, v in data.items() if k != "kind"})


__all__ = ["Engine", "TroubleshootingEngine", "make_engine", "load_spec", "succeeded_calls"]
