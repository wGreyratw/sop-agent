"""请求清单引擎（零售）的离线测试。不调用真实模型。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from sop_agent.agent import SopAgent
from sop_agent.engines import TroubleshootingEngine, load_spec, make_engine
from sop_agent.flow import StepKind
from sop_agent.requests import RequestEngine

FLOWS = Path(__file__).resolve().parents[1] / "flows"
SPEC = load_spec(FLOWS / "retail.yaml")


def engine() -> RequestEngine:
    return make_engine(SPEC)


def call(name, **arguments):
    return ToolCall(id=f"call_{name}_{len(arguments)}", name=name, arguments=arguments)


def result(tool_call, content, error=False):
    return ToolMessage(id=tool_call.id, role="tool", content=content, requestor="assistant", error=error)


def order_json(order_id, status):
    return json.dumps({"order_id": order_id, "user_id": "u1", "status": status, "items": []})


def extraction(*requests):
    return json.dumps({"requests": list(requests)})


def req(type_, order_id=None, id_=None, details=False, confirmed=False, state="open"):
    return {"id": id_, "type": type_, "order_id": order_id, "details_complete": details, "confirmed": confirmed, "state": state}


def authenticated(e: RequestEngine, orders: dict[str, str] | None = None):
    find = call("find_user_id_by_email", email="a@b.c")
    ledger = e.observe(e.new_ledger(), [find], [result(find, "user_1")])
    for order_id, status in (orders or {}).items():
        look = call("get_order_details", order_id=order_id)
        ledger = e.observe(ledger, [look], [result(look, order_json(order_id, status))])
    return ledger


def test_spec_kind_selects_the_engine():
    assert isinstance(make_engine(load_spec(FLOWS / "telecom.yaml")), TroubleshootingEngine)
    assert isinstance(engine(), RequestEngine)


def test_identity_and_order_status_come_from_tool_results():
    e = engine()
    assert e.next_step(e.new_ledger()).item == "identify_customer"
    failed = call("find_user_id_by_email", email="x")
    ledger = e.observe(e.new_ledger(), [failed], [result(failed, "Error: User not found", error=True)])
    assert ledger["user_id"] is None
    ledger = authenticated(e, {"#W1": "pending"})
    assert ledger["user_id"] == "user_1" and ledger["orders"] == {"#W1": "pending"}
    assert e.next_step(ledger).item == "ask_request"


def test_request_goes_through_collect_confirm_execute_then_done():
    e = engine()
    ledger = authenticated(e)
    ledger, _ = e.apply_extraction(ledger, extraction(req("cancel_order", "#W1")))
    assert e.next_step(ledger).kind == StepKind.CHECK  # 状态未知，先查订单
    look = call("get_order_details", order_id="#W1")
    ledger = e.observe(ledger, [look], [result(look, order_json("#W1", "pending"))])
    assert e.next_step(ledger).kind == StepKind.COLLECT
    ledger, _ = e.apply_extraction(ledger, extraction(req("cancel_order", "#W1", "r1", details=True)))
    assert e.next_step(ledger).kind == StepKind.CONFIRM
    ledger, _ = e.apply_extraction(ledger, extraction(req("cancel_order", "#W1", "r1", details=True, confirmed=True)))
    assert e.next_step(ledger).kind == StepKind.EXECUTE
    cancel = call("cancel_pending_order", order_id="#W1", reason="no longer needed")
    ledger = e.observe(ledger, [cancel], [result(cancel, order_json("#W1", "cancelled"))])
    assert ledger["requests"][0]["state"] == "done" and ledger["orders"]["#W1"] == "cancelled"
    assert e.next_step(ledger).kind == StepKind.RESOLVED


def test_wrong_status_means_deny_and_move_on():
    e = engine()
    ledger = authenticated(e, {"#W1": "processed", "#W2": "delivered"})
    ledger, _ = e.apply_extraction(ledger, extraction(req("cancel_order", "#W1"), req("return_items", "#W2")))
    step = e.next_step(ledger)
    assert step.kind == StepKind.DENY and "'processed'" in step.instruction
    ledger, _ = e.apply_extraction(ledger, extraction(req("cancel_order", "#W1", "r1", state="denied")))
    assert e.next_step(ledger).path == "r2"


def test_extraction_merge_rules():
    e = engine()
    ledger = authenticated(e, {"#W1": "delivered"})
    ledger, _ = e.apply_extraction(ledger, extraction(req("return_items", "#W1"), req("information")))
    # 不带 id 但同订单同类型：更新已有请求，不新增；模型不能把需要写操作的请求标成 done；没收齐细节不能算确认
    ledger, _ = e.apply_extraction(ledger, extraction(req("return_items", "#W1", confirmed=True, state="done"), req("bogus_type")))
    requests = ledger["requests"]
    assert [r["id"] for r in requests] == ["r1", "r2"]
    assert requests[0]["state"] == "open" and requests[0]["confirmed"] is False
    # 回答类请求可以由模型标为完成；漏掉的请求保留
    ledger, _ = e.apply_extraction(ledger, extraction(req("information", None, "r2", state="done")))
    assert [r["state"] for r in ledger["requests"]] == ["open", "done"]
    # 输出不是 JSON 时账本不变
    same, event = e.apply_extraction(ledger, "oops")
    assert same == ledger and event["found"] == []


def test_item_changes_wait_until_other_changes_on_the_order_are_done():
    e = engine()
    ledger = authenticated(e, {"#W1": "pending"})
    ledger, _ = e.apply_extraction(ledger, extraction(req("modify_order_items", "#W1"), req("modify_order_address", "#W1")))
    assert e.next_step(ledger).item == "modify_order_address"


def test_request_type_follows_the_order_status():
    """顾客说“换货”，但订单还没发货：实际要办的是修改商品，反之亦然。"""
    e = engine()
    ledger = authenticated(e, {"#W1": "pending", "#W2": "delivered"})
    ledger, _ = e.apply_extraction(ledger, extraction(req("exchange_items", "#W1"), req("modify_order_items", "#W2")))
    assert [(r["type"], r["order_id"]) for r in ledger["requests"]] == [("modify_order_items", "#W1"), ("exchange_items", "#W2")]
    # 模型下次不带 id 又说“换货”，不会再新增一个请求
    ledger, _ = e.apply_extraction(ledger, extraction(req("exchange_items", "#W1", details=True)))
    assert len(ledger["requests"]) == 2 and ledger["requests"][0]["details_complete"]
    # 订单号来自提取、状态稍后由工具查到时也会纠正
    ledger, _ = e.apply_extraction(ledger, extraction(req("exchange_items", "#W3")))
    look = call("get_order_details", order_id="#W3")
    ledger = e.observe(ledger, [look], [result(look, order_json("#W3", "pending"))])
    assert ledger["requests"][2]["type"] == "modify_order_items"


def write_reply(name, **arguments):
    return AssistantMessage(role="assistant", tool_calls=[call(name, **arguments)])


def test_guard_requires_confirmation_status_and_single_use():
    e = engine()
    ledger = authenticated(e, {"#W1": "delivered", "#W2": "pending"})
    exchange = write_reply("exchange_delivered_order_items", order_id="#W1", item_ids=["1"], new_item_ids=["2"], payment_method_id="p")
    ledger, _ = e.apply_extraction(ledger, extraction(req("exchange_items", "#W1", details=True)))
    assert "explicit yes" in e.check_reply(exchange, ledger, False)
    ledger, _ = e.apply_extraction(ledger, extraction(req("exchange_items", "#W1", "r1", details=True, confirmed=True)))
    assert e.check_reply(exchange, ledger, False) is None
    assert "needs delivered" in e.check_reply(write_reply("return_delivered_order_items", order_id="#W2"), ledger, False)
    assert "get_order_details first" in e.check_reply(write_reply("cancel_pending_order", order_id="#W9"), ledger, False)
    done = call("exchange_delivered_order_items", order_id="#W1")
    ledger = e.observe(ledger, [done], [result(done, order_json("#W1", "delivered"))])
    assert "only be used once" in e.check_reply(exchange, ledger, False)


def test_guard_blocks_writes_before_authentication_and_unneeded_transfers():
    e = engine()
    assert "authenticate" in e.check_reply(write_reply("modify_user_address", user_id="u"), e.new_ledger(), False)
    ledger = authenticated(e, {"#W1": "pending"})
    ledger, _ = e.apply_extraction(ledger, extraction(req("cancel_order", "#W1")))
    transfer = write_reply("transfer_to_human_agents", summary="x")
    assert "Transfer blocked" in e.check_reply(transfer, ledger, False)
    ledger, _ = e.apply_extraction(ledger, extraction(req("cancel_order", "#W1", "r1", state="withdrawn"), req("needs_human")))
    assert e.check_reply(transfer, ledger, False) is None
    assert "must be the transfer_to_human_agents" in e.check_reply(AssistantMessage(role="assistant", content="Sorry"), ledger, False)


class FakeChat:
    def __init__(self, extractions, replies):
        self.extractions, self.replies, self.respond_calls, self.extract_options = list(extractions), list(replies), [], []

    def __call__(self, messages, use_tools=True, json_mode=False, **options):
        if json_mode:
            self.extract_options.append(options)
            return AssistantMessage(role="assistant", content=self.extractions.pop(0), usage={})
        self.respond_calls.append(list(messages))
        reply = self.replies.pop(0)
        reply.usage = {}
        return reply


def test_agent_runs_the_retail_flow_and_traces_the_ledger(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SOP_TRACE_PATH", str(trace))
    find = ToolCall(id="c1", name="find_user_id_by_email", arguments={"email": "a@b.c"})
    chat = FakeChat(
        [extraction(req("return_items")), extraction(req("return_items", None, "r1"))],
        [
            # 还没核实身份就想退货：守卫拦下，重写为查用户
            AssistantMessage(role="assistant", tool_calls=[ToolCall(id="c0", name="return_delivered_order_items", arguments={"order_id": "#W1"})]),
            AssistantMessage(role="assistant", tool_calls=[find]),
            AssistantMessage(role="assistant", content="Which order?"),
        ],
    )
    agent = SopAgent(tools=[], domain_policy="policy", llm="fake", chat=chat, flow=SPEC)
    state = agent.get_init_state()
    reply, state = agent.generate_next_message(UserMessage(role="user", content="I want to return a lamp, a@b.c"), state)
    assert reply.tool_calls[0].name == "find_user_id_by_email"
    reply, _ = agent.generate_next_message(ToolMessage(id="c1", role="tool", content="user_1", requestor="assistant"), state)
    assert "Find which order" in chat.respond_calls[-1][-1].content
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in records[0]["events"]] == ["extract", "respond", "guard_block", "respond"]
    assert records[1]["ledger"]["user_id"] == "user_1"
    # 零售的提取关闭思考，省下大量输出 token
    assert chat.extract_options == [{"thinking": False}, {"thinking": False}]


@pytest.mark.parametrize("bad", ["cancel_pending_order", "exchange_delivered_order_items"])
def test_each_write_tool_belongs_to_one_type(bad):
    assert SPEC.type_of_tool(bad) is not None
