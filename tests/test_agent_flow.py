"""完整模式（提取事实 → 账本 → 下一步提示 → 回复 → 守卫）的离线测试。不调用真实模型。"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage

from sop_agent.agent import SopAgent, merge_usage
from sop_agent.extract import parse_extraction, render_messages
from sop_agent.flow import load_flow
from sop_agent.guard import check_reply

FLOW = load_flow(Path(__file__).resolve().parents[1] / "flows" / "telecom.yaml")
USAGE = {"prompt_tokens": 100, "completion_tokens": 10, "cache_hit_tokens": 90, "cache_miss_tokens": 10}


class FakeChat:
    """json_mode=True 时扮演提取，否则扮演回复。记录每次看到的消息。"""

    def __init__(self, extractions, replies):
        self.extractions = list(extractions)
        self.replies = list(replies)
        self.extract_calls: list[list] = []
        self.respond_calls: list[list] = []
        self.lock = threading.Lock()

    def __call__(self, messages, use_tools=True, json_mode=False):
        with self.lock:
            if json_mode:
                self.extract_calls.append(list(messages))
                facts = self.extractions.pop(0)
                content = facts if isinstance(facts, str) else json.dumps({"facts": facts})
                return AssistantMessage(role="assistant", content=content, usage=dict(USAGE))
            self.respond_calls.append(list(messages))
            reply = self.replies.pop(0)
            reply.usage = dict(USAGE)
            return reply


def say(text):
    return AssistantMessage(role="assistant", content=text)


def call_tool(name, **arguments):
    return AssistantMessage(role="assistant", tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)])


def user(text):
    return UserMessage(role="user", content=text)


def make_agent(chat, use_guard=True):
    return SopAgent(tools=[], domain_policy="policy", llm="fake", chat=chat, flow=FLOW, use_guard=use_guard)


def note_of(messages) -> str:
    """回复调用的最后一条是内部提示。"""
    assert isinstance(messages[-1], SystemMessage)
    return messages[-1].content


# 已核实身份、问题是彩信、有信号、能上网、彩信发不出去
MMS_READY = {"customer_identified": "yes", "problem": "mms", "service_available": "yes", "speed_test": "fair", "mms_works": "no"}


def test_extracted_facts_drive_the_next_step_note():
    chat = FakeChat([MMS_READY], [say("Please check your network mode.")])
    agent = make_agent(chat)
    reply, _ = agent.generate_next_message(user("I can't send pictures"), agent.get_init_state())
    assert reply.content == "Please check your network mode."
    note = note_of(chat.respond_calls[0])
    assert "Next required step (check)" in note and "network mode" in note
    assert "mms_works = no" in note


def test_note_is_not_kept_in_history():
    chat = FakeChat([MMS_READY, {"network_mode": "4g"}], [say("a"), say("b")])
    agent = make_agent(chat)
    state = agent.get_init_state()
    _, state = agent.generate_next_message(user("hi"), state)
    agent.generate_next_message(user("it says 4G"), state)
    second = chat.respond_calls[1]
    # 系统提示 + 历史（顾客、回复、顾客）+ 本轮提示，历史里没有上一轮的提示
    assert [type(m).__name__ for m in second] == ["SystemMessage", "UserMessage", "AssistantMessage", "UserMessage", "SystemMessage"]
    assert "wi-fi calling" in note_of(second).lower()


def test_extraction_sees_the_question_it_answers():
    chat = FakeChat([MMS_READY, {"network_mode": "4g"}], [say("Which network mode is set?"), say("ok")])
    agent = make_agent(chat)
    state = agent.get_init_state()
    _, state = agent.generate_next_message(user("hi"), state)
    agent.generate_next_message(user("4G"), state)
    request = json.loads(chat.extract_calls[1][-1].content)
    assert "AGENT: Which network mode is set?" in request["newest_messages"]
    assert "CUSTOMER: 4G" in request["newest_messages"]
    assert request["known_facts"]["mms_works"] == "no"


def test_guard_blocks_early_transfer_and_retries():
    """回归：基线在流程没走完时就转人工。"""
    chat = FakeChat([MMS_READY], [call_tool("transfer_to_human_agents", summary="x"), say("Please check your network mode.")])
    agent = make_agent(chat)
    reply, _ = agent.generate_next_message(user("still broken"), agent.get_init_state())
    assert reply.content == "Please check your network mode."
    assert "Transfer blocked" in note_of(chat.respond_calls[1])
    # 用量包含提取 1 次 + 回复 2 次
    assert reply.usage["llm_calls"] == 3 and reply.usage["prompt_tokens"] == 300


def test_blocked_draft_does_not_enter_history():
    chat = FakeChat([MMS_READY, {}], [call_tool("transfer_to_human_agents", summary="x"), say("check mode"), say("next")])
    agent = make_agent(chat)
    state = agent.get_init_state()
    _, state = agent.generate_next_message(user("hi"), state)
    agent.generate_next_message(user("ok"), state)
    history = chat.respond_calls[-1][1:-1]
    assert not any(getattr(m, "tool_calls", None) for m in history)


def test_guard_gives_up_after_two_retries():
    transfer = [call_tool("transfer_to_human_agents", summary="x") for _ in range(3)]
    chat = FakeChat([MMS_READY], transfer)
    agent = make_agent(chat)
    reply, _ = agent.generate_next_message(user("transfer me"), agent.get_init_state())
    assert reply.tool_calls[0].name == "transfer_to_human_agents"
    assert len(chat.respond_calls) == 3


def test_transfer_allowed_when_workflow_says_escalate():
    exhausted = {**MMS_READY, "network_mode": "4g", "wifi_calling": "off", "messaging_permissions": "granted", "mmsc_url": "present"}
    chat = FakeChat([exhausted], [call_tool("transfer_to_human_agents", summary="x")])
    agent = make_agent(chat)
    reply, _ = agent.generate_next_message(user("still broken"), agent.get_init_state())
    assert reply.tool_calls[0].name == "transfer_to_human_agents" and len(chat.respond_calls) == 1


def test_guard_can_be_disabled():
    chat = FakeChat([MMS_READY], [call_tool("transfer_to_human_agents", summary="x")])
    agent = make_agent(chat, use_guard=False)
    reply, _ = agent.generate_next_message(user("hi"), agent.get_init_state())
    assert reply.tool_calls[0].name == "transfer_to_human_agents"


def test_refuel_over_limit_is_blocked():
    from sop_agent.flow import FlowEngine, Ledger

    engine = FlowEngine(FLOW)
    assert "at most 2" in check_reply(call_tool("refuel_data", customer_id="C1", line_id="L1", gb_amount=5), engine, Ledger())
    assert check_reply(call_tool("refuel_data", customer_id="C1", line_id="L1", gb_amount=2), engine, Ledger()) is None


def test_bad_extraction_output_does_not_break_the_turn():
    chat = FakeChat(["not json"], [say("Could you describe the issue?")])
    agent = make_agent(chat)
    reply, _ = agent.generate_next_message(user("hi"), agent.get_init_state())
    assert reply.content == "Could you describe the issue?"
    assert "identify" in note_of(chat.respond_calls[0]).lower()


def test_parse_extraction_handles_code_fences_and_junk():
    assert parse_extraction('```json\n{"facts": {"vpn": "on"}}\n```') == {"vpn": "on"}
    assert parse_extraction("") == {} and parse_extraction("{oops") == {}


def test_render_messages_includes_tool_calls_and_results():
    text = render_messages([call_tool("get_data_usage", line_id="L1"), ToolMessage(id="call_get_data_usage", role="tool", content='{"used": 15}', requestor="assistant")])
    assert 'AGENT TOOL CALL get_data_usage({"line_id": "L1"})' in text and 'TOOL RESULT: {"used": 15}' in text


def test_trace_records_each_turn(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SOP_TRACE_PATH", str(trace))
    chat = FakeChat([MMS_READY], [call_tool("transfer_to_human_agents", summary="x"), say("check")])
    agent = make_agent(chat)
    agent.generate_next_message(user("hi"), agent.get_init_state())
    record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert [e["event"] for e in record["events"]] == ["extract", "respond", "guard_block", "respond"]
    assert record["facts"]["problem"] == "mms"
    # 回放靠 reply 把 trace 和对话对上，被拦截的草稿不应出现在这里
    assert record["reply"] == {"content": "check", "tool_calls": []}


ROAMING_OFF_ON_LINE = {
    "customer_identified": "yes",
    "problem": "mms",
    "service_available": "yes",
    "speed_test": "no_connection",
    "traveling": "yes",
    "device_roaming": "on",
    "line_roaming": "disabled",
}


def roaming_result(error=False):
    content = "Error: failed" if error else "Roaming enabled successfully"
    return ToolMessage(id="call_enable_roaming", role="tool", content=content, requestor="assistant", error=error)


def test_successful_fix_tool_requires_retest():
    """回归：开通漫游后应先让顾客重测网速，而不是转人工。"""
    chat = FakeChat([ROAMING_OFF_ON_LINE, {"line_roaming": "enabled"}], [call_tool("enable_roaming", customer_id="C1", line_id="L2"), say("Please run a speed test again.")])
    agent = make_agent(chat)
    state = agent.get_init_state()
    _, state = agent.generate_next_message(user("I'm abroad and data is dead"), state)
    assert "(fix)" in note_of(chat.respond_calls[0])
    agent.generate_next_message(roaming_result(), state)
    assert "Next required step (check_symptom)" in note_of(chat.respond_calls[1])


def test_failed_fix_tool_does_not_clear_symptoms():
    chat = FakeChat([ROAMING_OFF_ON_LINE, {}], [call_tool("enable_roaming", customer_id="C1", line_id="L2"), say("Let me retry.")])
    agent = make_agent(chat)
    state = agent.get_init_state()
    _, state = agent.generate_next_message(user("I'm abroad"), state)
    agent.generate_next_message(roaming_result(error=True), state)
    assert "speed_test = no_connection" in note_of(chat.respond_calls[1])


CONTRACT_END = {
    "customer_identified": "yes",
    "problem": "no_service",
    "service_available": "no",
    "airplane_mode": "off",
    "sim_status": "active",
    "apn_reset_done": "yes",
    "line_suspension": "contract_end",
}


def test_escalate_step_requires_transfer_call():
    """回归：流程要求转人工，Agent 却先解释，顾客随即结束了对话。"""
    chat = FakeChat([CONTRACT_END], [say("Your contract has ended, so I cannot restore service."), call_tool("transfer_to_human_agents", summary="x")])
    agent = make_agent(chat)
    reply, _ = agent.generate_next_message(user("why no service?"), agent.get_init_state())
    assert reply.tool_calls[0].name == "transfer_to_human_agents"
    assert "next action must be the transfer" in note_of(chat.respond_calls[1])


def test_message_after_transfer_is_allowed():
    chat = FakeChat([CONTRACT_END, {}], [call_tool("transfer_to_human_agents", summary="x"), say("YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.")])
    agent = make_agent(chat)
    state = agent.get_init_state()
    _, state = agent.generate_next_message(user("no service"), state)
    reply, _ = agent.generate_next_message(ToolMessage(id="call_transfer_to_human_agents", role="tool", content="Transfer successful", requestor="assistant"), state)
    assert reply.content.startswith("YOU ARE BEING TRANSFERRED") and len(chat.respond_calls) == 2


def test_merge_usage_counts_calls():
    total = merge_usage([USAGE, USAGE])
    assert total["llm_calls"] == 2 and total["cache_hit_tokens"] == 180
