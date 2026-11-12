"""流程引擎和电信流程配置的离线测试。不调用模型。

用一串“已知事实”模拟排查过程，检查引擎给出的下一步是否符合官方流程图。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sop_agent.flow import FlowEngine, FlowSpec, Ledger, StepKind, load_flow

FLOW_PATH = Path(__file__).resolve().parents[1] / "flows" / "telecom.yaml"
SPEC = load_flow(FLOW_PATH)
ENGINE = FlowEngine(SPEC)

# 已核实身份、确认问题类型后的起点
START = {"customer_identified": "yes"}
SERVICE_OK = {"service_available": "yes"}
DATA_CONNECTED = {"speed_test": "fair"}


def ledger(**facts) -> Ledger:
    return ENGINE.update(Ledger(), facts)


def step_of(**facts):
    return ENGINE.next_step(ledger(**facts))


# ---------- 配置本身 ----------


def test_all_values_are_strings():
    """回归：YAML 会把不加引号的 yes/no/on/off 解析成布尔值。"""
    raw = yaml.safe_load(FLOW_PATH.read_text(encoding="utf-8"))
    for name, fact in raw["facts"].items():
        assert all(isinstance(v, str) for v in fact["values"]), name


def test_config_references_are_checked():
    raw = yaml.safe_load(FLOW_PATH.read_text(encoding="utf-8"))
    raw["paths"]["mms"]["items"][0]["fact"] = "no_such_fact"
    with pytest.raises(ValueError, match="未定义的事实"):
        FlowSpec.model_validate(raw)


def test_problem_types_match_paths():
    assert set(SPEC.problems) == {"no_service", "mobile_data", "mms"}


# ---------- 开场 ----------


def test_identify_customer_first():
    step = step_of()
    assert step.kind == StepKind.CHECK and step.item == "identify_customer"


def test_then_classify_problem():
    step = step_of(**START)
    assert step.item == "classify_problem"


# ---------- 症状正常就跳过整条路径 ----------


def test_service_symptom_is_checked_first_for_every_problem():
    for problem in ("no_service", "mobile_data", "mms"):
        step = step_of(**START, problem=problem)
        assert (step.kind, step.path) == (StepKind.CHECK_SYMPTOM, "service")


def test_working_service_skips_to_next_path():
    step = step_of(**START, problem="mms", **SERVICE_OK)
    assert (step.kind, step.path) == (StepKind.CHECK_SYMPTOM, "data_connectivity")


def test_mms_resolved_when_all_symptoms_ok():
    step = step_of(**START, problem="mms", **SERVICE_OK, **DATA_CONNECTED, mms_works="yes")
    assert step.kind == StepKind.RESOLVED


# ---------- 基线漏掉的步骤：彩信题必须先查数据连通，包括流量用量 ----------


def test_mms_with_no_data_goes_through_data_usage_check():
    """回归：基线 8 道失败的彩信题都没查流量用量。"""
    facts = dict(**START, problem="mms", **SERVICE_OK, speed_test="no_connection")
    assert step_of(**facts).item == "traveling"
    facts["traveling"] = "no"
    assert step_of(**facts).item == "mobile_data"
    facts["mobile_data"] = "on"
    step = step_of(**facts)
    assert (step.kind, step.item) == (StepKind.CHECK, "data_usage")
    assert "get_data_usage" in step.instruction
    facts["data_usage"] = "exceeded"
    step = step_of(**facts)
    assert (step.kind, step.item) == (StepKind.FIX, "data_usage")
    assert "2 GB" in step.instruction


def test_mms_items_wait_until_data_connects():
    """数据没通时不应跳到彩信本身的检查。"""
    step = step_of(**START, problem="mms", **SERVICE_OK, speed_test="no_connection", traveling="no", mobile_data="on", data_usage="within_limit")
    assert step.kind == StepKind.ESCALATE and step.path == "data_connectivity"


def test_mms_path_checks_items_in_order():
    facts = dict(**START, problem="mms", **SERVICE_OK, **DATA_CONNECTED, mms_works="no")
    order = []
    for fact, ok_value in [("network_mode", "4g"), ("wifi_calling", "off"), ("messaging_permissions", "granted"), ("mmsc_url", "present")]:
        step = step_of(**facts)
        order.append(step.item)
        facts[fact] = ok_value
    assert order == ["network_mode_mms", "wifi_calling", "messaging_permissions", "mmsc_url"]
    assert step_of(**facts).kind == StepKind.ESCALATE


def test_3g_is_enough_for_mms_but_not_for_speed():
    mms = step_of(**START, problem="mms", **SERVICE_OK, **DATA_CONNECTED, mms_works="no", network_mode="3g")
    assert mms.item == "wifi_calling"
    speed = step_of(**START, problem="mobile_data", **SERVICE_OK, speed_test="fair", data_saver="off", network_mode="3g")
    assert (speed.kind, speed.item) == (StepKind.FIX, "network_mode_speed")


# ---------- 修好后必须重新检查症状 ----------


def test_fix_clears_symptoms_and_requires_recheck():
    book = ledger(**START, problem="no_service", service_available="no", airplane_mode="on")
    assert ENGINE.next_step(book).kind == StepKind.FIX
    book = ENGINE.update(book, {"airplane_mode": "off"})
    assert "service_available" not in book.facts
    assert book.fixes == ["airplane_mode: on -> off"]
    assert ENGINE.next_step(book).kind == StepKind.CHECK_SYMPTOM


def test_fix_reported_together_with_new_symptom_keeps_it():
    book = ledger(**START, problem="no_service", service_available="no", airplane_mode="on")
    book = ENGINE.update(book, {"airplane_mode": "off", "service_available": "yes"})
    assert ENGINE.next_step(book).kind == StepKind.RESOLVED


def test_checked_items_are_not_asked_again_after_recheck():
    facts = dict(**START, problem="no_service", service_available="no", airplane_mode="on")
    book = ENGINE.update(ledger(**facts), {"airplane_mode": "off"})
    book = ENGINE.update(book, {"service_available": "no"})
    assert ENGINE.next_step(book).item == "sim_card"


def test_switching_from_one_ok_value_to_another_is_not_a_fix():
    book = ledger(**START, problem="mms", **SERVICE_OK, **DATA_CONNECTED, mms_works="no", network_mode="4g")
    book = ENGINE.update(book, {"network_mode": "5g"})
    assert book.fixes == [] and book.facts["mms_works"] == "no"


def test_unknown_facts_and_values_are_ignored():
    book = ENGINE.update(Ledger(), {"no_such_fact": "x", "airplane_mode": "maybe", "mobile_data": "on"})
    assert book.facts == {"mobile_data": "on"}


# ---------- 什么时候允许转人工 ----------


def test_apn_reset_must_be_done_before_checking_suspension():
    step = step_of(**START, problem="no_service", service_available="no", airplane_mode="off", sim_status="active")
    assert (step.kind, step.item) == (StepKind.FIX, "reset_apn")


def test_locked_sim_allows_transfer():
    book = ledger(**START, problem="no_service", service_available="no", airplane_mode="off", sim_status="locked")
    assert ENGINE.transfer_allowed(book)


def test_contract_end_suspension_allows_transfer_but_overdue_bill_must_be_handled():
    base = dict(**START, problem="no_service", service_available="no", airplane_mode="off", sim_status="active", apn_reset_done="yes")
    assert ENGINE.transfer_allowed(ledger(**base, line_suspension="contract_end"))
    step = step_of(**base, line_suspension="overdue_bill")
    assert step.kind == StepKind.FIX and "payment request" in step.instruction
    assert not ENGINE.transfer_allowed(ledger(**base, line_suspension="overdue_bill"))


def test_declined_refuel_allows_transfer():
    facts = dict(**START, problem="mobile_data", **SERVICE_OK, speed_test="no_connection", traveling="no", mobile_data="on")
    assert not ENGINE.transfer_allowed(ledger(**facts, data_usage="exceeded"))
    assert ENGINE.transfer_allowed(ledger(**facts, data_usage="refuel_declined"))


def test_transfer_blocked_while_steps_remain():
    """回归：基线说“都查过了”就转人工，但其实还有步骤没做。"""
    book = ledger(**START, problem="mms", **SERVICE_OK, **DATA_CONNECTED, mms_works="no", network_mode="4g", wifi_calling="off")
    assert not ENGINE.transfer_allowed(book)
    assert ENGINE.next_step(book).item == "messaging_permissions"


def test_transfer_not_allowed_once_resolved():
    assert not ENGINE.transfer_allowed(ledger(**START, problem="no_service", service_available="yes"))


# ---------- 漫游只在出行时检查 ----------


def test_roaming_checked_only_when_traveling():
    base = dict(**START, problem="mobile_data", **SERVICE_OK, speed_test="no_connection")
    assert step_of(**base, traveling="no").item == "mobile_data"
    assert step_of(**base, traveling="yes").item == "device_roaming"
    step = step_of(**base, traveling="yes", device_roaming="on")
    assert step.item == "line_roaming" and "get_details_by_id" in step.instruction
    step = step_of(**base, traveling="yes", device_roaming="on", line_roaming="disabled")
    assert step.kind == StepKind.FIX and "enable_roaming" in step.instruction


def test_slow_data_goes_through_speed_path_after_connecting():
    facts = dict(**START, problem="mobile_data", **SERVICE_OK, speed_test="poor")
    order = []
    for fact, ok_value in [("data_saver", "off"), ("network_mode", "5g"), ("vpn", "off")]:
        order.append(step_of(**facts).item)
        facts[fact] = ok_value
    assert order == ["data_saver", "network_mode_speed", "vpn"]
    assert step_of(**facts).kind == StepKind.ESCALATE


def test_escalate_step_tells_the_agent_to_transfer_now():
    """回归：Agent 只问“要不要转人工”而不调用工具，顾客随即结束了对话。"""
    base = dict(**START, problem="no_service", service_available="no", airplane_mode="off", sim_status="active", apn_reset_done="yes")
    step = step_of(**base, line_suspension="contract_end")
    assert step.kind == StepKind.ESCALATE
    assert "call transfer_to_human_agents" in step.instruction and "Do not ask" in step.instruction


def test_agent_tool_fix_clears_symptoms():
    """回归：Agent 开通漫游后没有让顾客重测网速，直接转了人工。"""
    book = ledger(**START, problem="mms", **SERVICE_OK, speed_test="no_connection", traveling="yes", device_roaming="on", line_roaming="enabled")
    fixed = ENGINE.apply_tool_fix(book, "enable_roaming")
    assert "speed_test" not in fixed.facts and "service_available" not in fixed.facts
    assert fixed.fixes[-1] == "tool enable_roaming"
    assert ENGINE.next_step(fixed).kind == StepKind.CHECK_SYMPTOM


def test_read_only_tools_do_not_clear_symptoms():
    book = ledger(**START, problem="mms", **SERVICE_OK)
    assert ENGINE.apply_tool_fix(book, "get_data_usage") == book


def test_account_checks_name_the_reported_line():
    """回归：Agent 查了顾客另一条线路的用量。"""
    items = {item.id: item for path in SPEC.paths.values() for item in path.items}
    for item_id in ("data_usage", "line_roaming", "line_suspension"):
        assert "phone number is the one the customer reported" in items[item_id].check


def test_known_blocking_fault_escalates_without_checking_symptom():
    """回归：SIM 卡已知被锁，引擎却坚持先检查状态栏。"""
    step = step_of(**START, problem="no_service", airplane_mode="off", sim_status="locked")
    assert step.kind == StepKind.ESCALATE and step.item == "sim_card"


def test_known_fixable_fault_still_checks_symptom_first():
    step = step_of(**START, problem="no_service", airplane_mode="on")
    assert step.kind == StepKind.CHECK_SYMPTOM


def test_speed_test_description_treats_any_speed_as_connected():
    assert "even 0.00 Mbps" in SPEC.facts["speed_test"].description


def test_contract_end_takes_priority_over_overdue_bill():
    """回归：同时有欠费和合同到期时被记成欠费，Agent 走了不允许的恢复流程。"""
    description = SPEC.facts["line_suspension"].description
    assert "even if there is also an overdue bill" in description


def test_resolved_instruction_is_english():
    assert step_of(**START, problem="no_service", service_available="yes").instruction.isascii()


def test_summary_lists_facts_and_fixes():
    book = ENGINE.update(ledger(**START, airplane_mode="on"), {"airplane_mode": "off"})
    text = ENGINE.summary(book)
    assert "airplane_mode = off" in text and "已修复" in text
