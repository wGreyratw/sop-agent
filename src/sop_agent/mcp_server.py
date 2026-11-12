"""把 SOP 流程引擎作为 MCP Server 提供，让任何支持 MCP 的 Agent 都能用它做“流程导航”。

接入方式（stdio）：python -m sop_agent.mcp_server
一个工单（case）对应一段客服对话，引擎在服务端为每个工单保存排查账本：

1. get_flow：拿到流程定义，知道有哪些事实可以记录、取值是什么
2. start_case：开一个工单，得到第一步
3. 对话中每得到新信息，就 record_facts；自己用工具修好了问题，就 record_fix_tool
4. 每次准备调用工具或回复前，用 check_action 问守卫是否允许
5. get_case 查看完整进度

和评测时的区别：评测时由 Agent 内部的提取步骤调用模型得到事实；这里由接入的 Agent 自己判断事实，
引擎只负责记录进度、决定下一步和执行守卫规则，不调用任何模型。
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .engines import load_spec
from .flow import FlowEngine, FlowSpec, Ledger, Step, StepKind, load_flow
from .guard import TRANSFER_TOOL, check_reply
from .replay import progress

FLOWS_DIR = Path(__file__).resolve().parents[2] / "flows"

server = MCPServer(
    name="sop-engine",
    instructions=(
        "SOP workflow engine for customer-support agents. Call get_flow once, start_case per conversation, "
        "record_facts whenever the customer or a tool reveals something, and check_action before every tool "
        "call or reply. Always follow next_step."
    ),
)

_engines: dict[str, FlowEngine] = {}
_cases: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def available_flows() -> list[str]:
    """只提供排障类流程：这里的工具（记录事实、修复工具、路径进度）是按排障流程设计的。"""
    return sorted(path.stem for path in FLOWS_DIR.glob("*.yaml") if isinstance(load_spec(path), FlowSpec))


def _engine(flow: str) -> FlowEngine:
    if flow not in available_flows():
        # ToolError 的原因会传给客户端；普通异常只显示“执行出错”，接入的 Agent 无法纠正
        raise ToolError(f"Unknown flow '{flow}'. Available: {', '.join(available_flows())}")
    if flow not in _engines:
        _engines[flow] = FlowEngine(load_flow(FLOWS_DIR / f"{flow}.yaml"))
    return _engines[flow]


def _case(case_id: str) -> dict[str, Any]:
    case = _cases.get(case_id)
    if case is None:
        raise ToolError(f"Unknown case_id '{case_id}'. Call start_case first.")
    return case


def _step(step: Step) -> dict[str, Any]:
    return {"kind": step.kind.value, "instruction": step.instruction, "path": step.path, "item": step.item, "fact": step.fact}


@server.tool()
def get_flow(flow: str = "telecom") -> dict[str, Any]:
    """Return the workflow definition: the facts you can record (with allowed values), problem types and their paths."""
    spec = _engine(flow).spec
    return {
        "name": spec.name,
        "problem_fact": spec.problem_fact,
        "problems": spec.problems,
        "paths": {path_id: {"title": path.title, "symptom": path.symptom, "items": [item.id for item in path.items]} for path_id, path in spec.paths.items()},
        "facts": {name: {"values": fact.values, "description": fact.description, "source": fact.source} for name, fact in spec.facts.items()},
        "fix_tools": spec.fix_tools,
    }


@server.tool()
def start_case(flow: str = "telecom") -> dict[str, Any]:
    """Open a case for one customer conversation. Returns the case_id and the first required step."""
    engine = _engine(flow)
    case_id = uuid.uuid4().hex[:12]
    with _lock:
        _cases[case_id] = {"flow": flow, "ledger": Ledger(), "transferred": False}
    return {"case_id": case_id, "next_step": _step(engine.next_step(Ledger()))}


@server.tool()
def record_facts(case_id: str, facts: dict[str, str]) -> dict[str, Any]:
    """Record facts learned from the customer or tool results, e.g. {"airplane_mode": "on"}.

    Unknown fact names or values are ignored and reported back. If a fact moves from a faulty value to a normal one,
    it counts as a fix and all symptoms must be checked again. Returns the next required step.
    """
    with _lock:
        case = _case(case_id)
        engine = _engine(case["flow"])
        spec = engine.spec
        ignored = {
            name: value
            for name, value in facts.items()
            if name not in spec.facts or value not in spec.facts[name].values
        }
        before = case["ledger"]
        case["ledger"] = engine.update(before, facts)
        ledger = case["ledger"]
    return {
        "accepted": {name: value for name, value in facts.items() if name not in ignored},
        "ignored": ignored,
        "new_fixes": ledger.fixes[len(before.fixes):],
        "next_step": _step(engine.next_step(ledger)),
    }


@server.tool()
def record_fix_tool(case_id: str, tool_name: str, succeeded: bool = True) -> dict[str, Any]:
    """Report that you called one of the flow's fix tools (e.g. enable_roaming). A successful fix means symptoms must be rechecked."""
    with _lock:
        case = _case(case_id)
        engine = _engine(case["flow"])
        if succeeded:
            case["ledger"] = engine.apply_tool_fix(case["ledger"], tool_name)
        ledger = case["ledger"]
    return {"counted_as_fix": succeeded and tool_name in engine.spec.fix_tools, "next_step": _step(engine.next_step(ledger))}


@server.tool()
def check_action(case_id: str, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Ask the guard whether your next action is allowed, before you send it.

    tool_calls: the tool calls you intend to make, each {"name": ..., "arguments": {...}}; empty means a plain text reply.
    If allowed and it includes the transfer tool, the case is marked as transferred.
    """
    calls = [SimpleNamespace(name=call.get("name", ""), arguments=call.get("arguments") or {}) for call in tool_calls or []]
    with _lock:
        case = _case(case_id)
        engine = _engine(case["flow"])
        reason = check_reply(SimpleNamespace(tool_calls=calls), engine, case["ledger"], transferred=case["transferred"])
        if reason is None and any(call.name == TRANSFER_TOOL for call in calls):
            case["transferred"] = True
    return {"allowed": reason is None, "reason": reason}


@server.tool()
def get_case(case_id: str) -> dict[str, Any]:
    """Return the case's known facts, completed fixes, per-path progress and next step."""
    with _lock:
        case = _case(case_id)
        engine = _engine(case["flow"])
        ledger = case["ledger"]
    intake, paths = progress(engine, ledger.facts)
    step = engine.next_step(ledger)
    return {
        "facts": ledger.facts,
        "fixes": ledger.fixes,
        "transferred": case["transferred"],
        "intake": [vars(item) for item in intake],
        "paths": [{**vars(path), "items": [vars(item) for item in path.items]} for path in paths],
        "next_step": _step(step),
        "resolved": step.kind == StepKind.RESOLVED,
    }


@server.resource("sop://flows/{flow}", mime_type="application/yaml", description="The workflow's YAML source.")
def flow_source(flow: str) -> str:
    _engine(flow)
    return (FLOWS_DIR / f"{flow}.yaml").read_text(encoding="utf-8")


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
