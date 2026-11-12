"""骨架阶段的离线测试：不调用真实模型。

检查 LangGraph 的“暂停 → 继续”和 τ²-bench 的一问一答能正确配合。
"""

from __future__ import annotations

import threading

import pytest
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage

from sop_agent.agent import SopAgent
from sop_agent.usage import estimate_cost_usd, usage_from_raw


class FakeLlm:
    """按顺序返回预设的回复，并记录每次调用时看到的对话历史。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen: list[list] = []
        self.lock = threading.Lock()

    def __call__(self, history):
        with self.lock:
            self.seen.append(list(history))
            return self.replies.pop(0)


def say(text):
    return AssistantMessage(role="assistant", content=text)


def call_tool(name, **arguments):
    return AssistantMessage(role="assistant", tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)])


def user(text):
    return UserMessage(role="user", content=text)


def tool_result(name, content):
    return ToolMessage(id=f"call_{name}", role="tool", content=content, requestor="assistant")


def make_agent(llm):
    return SopAgent(tools=[], domain_policy="policy", llm="fake", call_llm=llm)


def test_each_turn_returns_one_reply_and_keeps_history():
    llm = FakeLlm([say("Hi"), call_tool("get_customer_by_phone", phone_number="555"), say("Found you")])
    agent = make_agent(llm)
    state = agent.get_init_state()

    reply, state = agent.generate_next_message(user("My phone is broken"), state)
    assert reply.content == "Hi"
    reply, state = agent.generate_next_message(user("555"), state)
    assert reply.tool_calls[0].name == "get_customer_by_phone"
    reply, state = agent.generate_next_message(tool_result("get_customer_by_phone", "C1001"), state)
    assert reply.content == "Found you"

    # 第三次调用模型时，应看到完整的对话：顾客、回复、顾客、工具调用、工具结果
    last = llm.seen[-1]
    assert [m.role for m in last] == ["user", "assistant", "user", "assistant", "tool"]
    assert last[-1].content == "C1001"


def test_resume_does_not_call_the_model_twice():
    """回归保护：调用模型和 interrupt() 不能放在同一个节点，否则继续执行时会重复调用模型。"""
    llm = FakeLlm([say("one"), say("two"), say("three")])
    agent = make_agent(llm)
    state = agent.get_init_state()
    for text in ("a", "b", "c"):
        _, state = agent.generate_next_message(user(text), state)
    assert len(llm.seen) == 3


def test_multiple_tool_results_are_all_added():
    llm = FakeLlm([call_tool("x"), say("done")])
    agent = make_agent(llm)
    state = agent.get_init_state()
    _, state = agent.generate_next_message(user("go"), state)
    both = MultiToolMessage(role="tool", tool_messages=[tool_result("x", "1"), tool_result("y", "2")])
    agent.generate_next_message(both, state)
    assert [m.content for m in llm.seen[-1][-2:]] == ["1", "2"]


def test_prior_history_is_included_in_the_first_turn():
    llm = FakeLlm([say("ok")])
    agent = make_agent(llm)
    state = agent.get_init_state([user("earlier"), say("earlier reply")])
    agent.generate_next_message(user("now"), state)
    assert [m.content for m in llm.seen[0]] == ["earlier", "earlier reply", "now"]


def test_conversations_do_not_share_history():
    llm = FakeLlm([say("to A"), say("to B"), say("to A again")])
    agent = make_agent(llm)
    a, b = agent.get_init_state(), agent.get_init_state()
    _, a = agent.generate_next_message(user("A1"), a)
    _, b = agent.generate_next_message(user("B1"), b)
    agent.generate_next_message(user("A2"), a)
    assert [m.content for m in llm.seen[-1]] == ["A1", "to A", "A2"]


def test_stop_tools_are_recognized():
    agent = SopAgent(tools=[], domain_policy="p", llm="fake", call_llm=FakeLlm([]), additional_stop_tool_names=frozenset({"end"}))
    assert agent.is_stop(call_tool("end"))
    assert not agent.is_stop(call_tool("other"))


@pytest.mark.parametrize(
    "raw, expected",
    [
        # DeepSeek 原生字段
        ({"usage": {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_cache_hit_tokens": 900, "prompt_cache_miss_tokens": 100}}, (900, 100)),
        # OpenAI 风格字段，没有 miss 时用 prompt - hit 计算
        ({"usage": {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 600}}}, (600, 400)),
        # 完全没有缓存信息时，全部算作未命中
        ({"usage": {"prompt_tokens": 1000, "completion_tokens": 50}}, (0, 1000)),
    ],
)
def test_usage_reads_cache_hits(raw, expected):
    usage = usage_from_raw(raw)
    assert (usage["cache_hit_tokens"], usage["cache_miss_tokens"]) == expected


def test_cost_uses_cache_prices():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "cache_hit_tokens": 1_000_000, "cache_miss_tokens": 0}
    assert estimate_cost_usd(usage) == pytest.approx(0.006 + 1.2)
    assert usage_from_raw(None) is None
