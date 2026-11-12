"""接入 τ²-bench 的 Agent。

τ²-bench 要求实现 get_init_state 和 generate_next_message 两个方法。
这里把它们翻译成对 LangGraph 图的“启动”和“继续”，每段对话用一个独立的 thread_id。

两种模式，便于做对照实验：
- 不给流程（flow=None）：骨架，行为和 τ²-bench 自带的简单 Agent 相同。
- 给流程：提取事实 → 账本 → 下一步提示 → 回复 → 守卫（可关闭）。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from langgraph.types import Command
from pydantic import BaseModel
from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, SystemMessage
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

from .engines import load_spec, make_engine
from .graph import ChatCall, LlmCall, build_flow_graph, build_graph
from .usage import usage_from_raw

FLOWS_DIR = Path(__file__).resolve().parents[2] / "flows"
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens")
_TRACE_LOCK = threading.Lock()


class SopAgentState(BaseModel):
    thread_id: str
    started: bool = False
    turn: int = 0
    initial_history: list[Any] = []


def merge_usage(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """把一轮里多次模型调用的用量加起来，llm_calls 记录调用次数。"""
    total: dict[str, Any] = {field: sum(int(e.get(field) or 0) for e in entries) for field in USAGE_FIELDS}
    total["llm_calls"] = len(entries)
    return total


def write_trace(record: dict[str, Any]) -> None:
    """设置了 SOP_TRACE_PATH 时，把每轮的提取结果、下一步和守卫决定写入 JSONL。"""
    path = os.environ.get("SOP_TRACE_PATH")
    if not path:
        return
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _TRACE_LOCK:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")


class SopAgent(LLMConfigMixin, HalfDuplexAgent[SopAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        additional_stop_tool_names: frozenset[str] = frozenset(),
        call_llm: LlmCall | None = None,
        chat: ChatCall | None = None,
        flow: Any | None = None,
        use_guard: bool = True,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.additional_stop_tool_names = additional_stop_tool_names
        # 回复用的系统提示和 τ²-bench 简单 Agent 相同，差别只在流程和守卫
        self.system_message = SystemMessage(
            role="system", content=SYSTEM_PROMPT.format(domain_policy=domain_policy, agent_instruction=AGENT_INSTRUCTION)
        )
        self.flow = flow
        if flow is None:
            self.graph = build_graph(call_llm or self._call_llm)
        else:
            self.graph = build_flow_graph(chat or self._chat, self.system_message, make_engine(flow), use_guard=use_guard)

    def _chat(self, messages: list[Any], use_tools: bool = True, json_mode: bool = False, thinking: bool = True) -> AssistantMessage:
        kwargs = dict(self.llm_args)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if not thinking:
            # 零售提取规则复杂，开着思考时一次提取会想上万 token，JSON 本身只有几百 token
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        reply = generate(
            model=self.llm,
            tools=self.tools if use_tools else None,
            messages=messages,
            call_name="sop_agent_extract" if json_mode else "sop_agent_respond",
            **kwargs,
        )
        usage = usage_from_raw(reply.raw_data)
        if usage is not None:
            # 补上缓存命中数，结果文件里就能算出准确费用
            reply.usage = {**(reply.usage or {}), **usage}
        # 原始响应很大，算完用量就不再需要
        reply.raw_data = None
        return reply

    def _call_llm(self, history: list[Any]) -> AssistantMessage:
        return self._chat([self.system_message, *history])

    def is_stop(self, message: AssistantMessage) -> bool:
        return any(call.name in self.additional_stop_tool_names for call in message.tool_calls or [])

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> SopAgentState:
        return SopAgentState(thread_id=uuid.uuid4().hex, initial_history=list(message_history or []))

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: SopAgentState
    ) -> tuple[AssistantMessage, SopAgentState]:
        incoming = list(message.tool_messages) if isinstance(message, MultiToolMessage) else [message]
        config = {"configurable": {"thread_id": state.thread_id}}
        if not state.started:
            start: dict[str, Any] = {"messages": [*state.initial_history, *incoming]}
            if self.flow is not None:
                start["fresh"] = incoming
            result = self.graph.invoke(start, config)
            state.started = True
        else:
            result = self.graph.invoke(Command(resume=incoming), config)
        interrupts = result.get("__interrupt__") or []
        if len(interrupts) != 1:
            raise RuntimeError(f"图应当停在等待下一条输入的位置，实际中断数为 {len(interrupts)}")
        reply: AssistantMessage = interrupts[0].value
        state.turn += 1
        if self.flow is not None:
            values = self.graph.get_state(config).values
            # 交给 τ²-bench 的回复带上本轮所有调用（提取、回复、被拦截的重试）的用量
            reply.usage = merge_usage(values.get("turn_usage") or [])
            write_trace(
                {
                    "thread_id": state.thread_id,
                    "turn": state.turn,
                    "events": values.get("events") or [],
                    "facts": (values.get("ledger") or {}).get("facts", {}),
                    "fixes": (values.get("ledger") or {}).get("fixes", []),
                    # 请求清单（零售）的账本没有 facts，整份记下
                    **({} if "facts" in (values.get("ledger") or {}) else {"ledger": values.get("ledger") or {}}),
                    # 回放时按回复内容把这条记录和 τ²-bench 结果里的对话对上
                    "reply": {
                        "content": reply.content,
                        "tool_calls": [{"name": call.name, "arguments": call.arguments} for call in reply.tool_calls or []],
                    },
                }
            )
        return reply, state


def create_sop_agent(tools, domain_policy, flow: str | None = None, use_guard: bool = True, **kwargs):
    return SopAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        additional_stop_tool_names=kwargs.get("additional_stop_tool_names", frozenset()),
        flow=load_spec(FLOWS_DIR / f"{flow}.yaml") if flow else None,
        use_guard=use_guard,
    )
