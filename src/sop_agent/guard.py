"""守卫：在 Agent 的回复发出前，用程序检查关键规则。

规则写在提示词里，模型仍可能违反；这里的检查由代码执行，违反时把原因告诉模型，让它重新回复。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .flow import FlowEngine, Ledger, StepKind

if TYPE_CHECKING:  # 只用到 tool_calls 的 name 和 arguments，运行时不依赖 τ²-bench
    from tau2.data_model.message import AssistantMessage

TRANSFER_TOOL = "transfer_to_human_agents"
MAX_REFUEL_GB = 2.0


def check_reply(reply: AssistantMessage, engine: FlowEngine | None, ledger: Ledger, transferred: bool = False) -> str | None:
    """返回拦截原因；没有问题时返回 None。transferred 表示这段对话里已经调用过转人工工具。"""
    calls = {call.name for call in reply.tool_calls or []}
    # 回归：流程已要求转人工，Agent 却先向顾客解释，顾客随即结束了对话
    if engine is not None and not transferred and TRANSFER_TOOL not in calls and engine.next_step(ledger).kind == StepKind.ESCALATE:
        return (
            "The workflow requires a transfer now. Your next action must be the transfer_to_human_agents tool call; "
            "you can explain the reason in the message you send after it."
        )
    for call in reply.tool_calls or []:
        if call.name == TRANSFER_TOOL and engine is not None and not engine.transfer_allowed(ledger):
            step = engine.next_step(ledger)
            return (
                "Transfer blocked: the troubleshooting workflow is not finished, so this issue may still be "
                f"solvable. Do the next required step instead: {step.instruction}"
            )
        if call.name == "refuel_data":
            amount = call.arguments.get("gb_amount", call.arguments.get("amount"))
            try:
                if amount is not None and float(amount) > MAX_REFUEL_GB:
                    return f"Refuel blocked: the policy allows at most {MAX_REFUEL_GB:g} GB per refuel."
            except (TypeError, ValueError):
                return "Refuel blocked: the amount must be a number of GB, at most 2."
    return None
