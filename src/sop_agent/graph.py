"""Agent 的 LangGraph 状态图。

τ²-bench 每次只要 Agent 的一条消息，对话循环在 τ²-bench 那边。所以图在需要说话或调用工具时
用 interrupt() 暂停，把这条消息交出去；收到顾客回复或工具结果后，用 Command(resume=...) 继续。
状态由检查点保存，所以每次继续时历史和账本都在。

等待必须单独成一个节点：继续执行时 LangGraph 会从被中断的节点开头重跑，
如果调用模型和 interrupt() 在同一个节点里，每次继续都会多调用一次模型。

有两种图：
- build_graph：骨架，respond → wait，行为和 τ²-bench 自带的简单 Agent 相同，用作对照。
- build_flow_graph：完整版，extract → respond → guard → wait，见 build_flow_graph 的说明。
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import interrupt
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage

from .engines import Engine
from .flow import StepKind

LlmCall = Callable[[list[Any]], Any]
"""输入对话历史（不含系统提示），返回一条 AssistantMessage。"""

ChatCall = Callable[..., AssistantMessage]
"""chat(messages, use_tools=..., json_mode=..., [thinking=False])：消息需自带系统提示。"""

MAX_GUARD_RETRIES = 2


def add_or_reset(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """追加；传入 None 时清空。用于“每轮重新累计”的字段。"""
    if right is None:
        return []
    return [*(left or []), *right]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], operator.add]
    """对话历史：顾客消息、Agent 消息和工具结果，按顺序追加。"""


def build_graph(call_llm: LlmCall, checkpointer: Any | None = None):
    def respond(state: AgentState) -> dict[str, Any]:
        reply = call_llm(list(state["messages"]))
        return {"messages": [reply]}

    def wait(state: AgentState) -> dict[str, Any]:
        incoming = interrupt(state["messages"][-1])
        return {"messages": list(incoming)}

    graph = StateGraph(AgentState)
    graph.add_node("respond", respond)
    graph.add_node("wait", wait)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", "wait")
    graph.add_edge("wait", "respond")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


class FlowState(TypedDict, total=False):
    messages: Annotated[list[Any], operator.add]
    fresh: list[Any]
    """本轮新到的消息（顾客回复或工具结果），提取事实只看这些和 Agent 上一句话。"""
    ledger: dict[str, Any]
    draft: Any
    """待守卫检查的回复。通过后才进入对话历史。"""
    feedback: str | None
    """守卫的拦截原因，下一次 respond 会看到。"""
    retries: int
    turn_usage: Annotated[list[dict[str, Any]], add_or_reset]
    """本轮所有模型调用的用量，交出回复时汇总。"""
    events: Annotated[list[dict[str, Any]], add_or_reset]
    """轨迹：每轮的提取结果、下一步和守卫决定。"""


STEP_NOTE = """[SOP tracker - internal note, never mention it to the customer]
Known facts:
{facts}

Next required step ({kind}): {instruction}

Do the next required step now with one tool call or one message. If the customer raised something else, address it briefly and then continue with this step. Do not transfer to a human agent unless the next required step says to escalate."""


def build_flow_graph(
    chat: ChatCall,
    system_message: SystemMessage,
    engine: Engine,
    use_guard: bool = True,
    checkpointer: Any | None = None,
):
    """完整版：

        extract（模型提取事实）→ respond（附上下一步的提示后调用模型）→ guard（程序检查）
          → 通过：wait（交出回复，等待下一条输入）→ extract ...
          → 拦截：带着原因回到 respond，最多重试 MAX_GUARD_RETRIES 次

    提取和回复用两次独立的调用：提取的系统提示只有事实清单，回复的系统提示和基线相同，
    两者前缀都保持不变，可以命中缓存。下一步的提示附在对话最后，不进入历史。

    流程形状（排障或请求清单）由 engine 决定，图本身不含领域知识，见 engines.py。
    """

    def extract(state: FlowState) -> dict[str, Any]:
        ledger = state.get("ledger") or engine.new_ledger()
        fresh = list(state.get("fresh") or [])
        previous = [m for m in state["messages"][: len(state["messages"]) - len(fresh)] if isinstance(m, AssistantMessage)][-1:]
        # 先用代码读工具调用和结果（可靠），再用模型读对话
        calls = list(previous[0].tool_calls or []) if previous else []
        ledger = engine.observe(ledger, calls, [m for m in fresh if isinstance(m, ToolMessage)])
        options = {} if engine.extract_thinking else {"thinking": False}
        reply = chat(engine.extract_messages(ledger, [*previous, *fresh]), use_tools=False, json_mode=True, **options)
        updated, event = engine.apply_extraction(ledger, reply.content)
        event["usage"] = reply.usage or {}
        return {"ledger": updated, "turn_usage": [reply.usage or {}], "events": [event]}

    def respond(state: FlowState) -> dict[str, Any]:
        ledger = state.get("ledger") or engine.new_ledger()
        step = engine.next_step(ledger)
        note = STEP_NOTE.format(facts=engine.summary(ledger), kind=step.kind.value, instruction=step.instruction)
        if state.get("feedback"):
            note += f"\n\nYour previous reply was rejected. {state['feedback']}"
        reply = chat([system_message, *state["messages"], SystemMessage(role="system", content=note)], use_tools=True, json_mode=False)
        event = {"event": "respond", "step": step.kind.value, "item": step.item, "path": step.path, "usage": reply.usage or {}}
        return {"draft": reply, "turn_usage": [reply.usage or {}], "events": [event]}

    def guard(state: FlowState) -> dict[str, Any]:
        ledger = state.get("ledger") or engine.new_ledger()
        transferred = any(call.name == "transfer_to_human_agents" for m in state["messages"] for call in (getattr(m, "tool_calls", None) or []))
        reason = engine.check_reply(state["draft"], ledger, transferred) if use_guard else None
        retries = state.get("retries") or 0
        if reason and retries < MAX_GUARD_RETRIES:
            return {"feedback": reason, "retries": retries + 1, "events": [{"event": "guard_block", "reason": reason}]}
        events = [{"event": "guard_override", "reason": reason}] if reason else []
        return {"messages": [state["draft"]], "feedback": None, "retries": 0, "events": events}

    def after_guard(state: FlowState) -> Literal["respond", "wait"]:
        return "respond" if state.get("feedback") else "wait"

    def wait(state: FlowState) -> dict[str, Any]:
        incoming = interrupt(state["messages"][-1])
        return {"messages": list(incoming), "fresh": list(incoming), "turn_usage": None, "events": None}

    graph = StateGraph(FlowState)
    graph.add_node("extract", extract)
    graph.add_node("respond", respond)
    graph.add_node("guard", guard)
    graph.add_node("wait", wait)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "respond")
    graph.add_edge("respond", "guard")
    graph.add_conditional_edges("guard", after_guard, ["respond", "wait"])
    graph.add_edge("wait", "extract")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


__all__ = ["AgentState", "FlowState", "build_graph", "build_flow_graph", "StepKind"]
