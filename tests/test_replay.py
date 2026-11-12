"""回放逻辑的离线测试：对话和 trace 的配对，以及 SOP 进度的计算。"""

from __future__ import annotations

import json
from pathlib import Path

from sop_agent.flow import FlowEngine, load_flow
from sop_agent.replay import load_run, progress

ENGINE = FlowEngine(load_flow(Path(__file__).resolve().parents[1] / "flows" / "telecom.yaml"))


def message(role, content="", tool_calls=None):
    return {"role": role, "content": content, "tool_calls": tool_calls}


def reply(content="", tools=()):
    return {"content": content, "tool_calls": [{"name": name, "arguments": {}} for name in tools]}


def write_run(tmp_path, simulations, traces):
    (tmp_path / "results.json").write_text(json.dumps({"simulations": simulations}), encoding="utf-8")
    lines = [json.dumps(record) for record in traces]
    (tmp_path / "trace.jsonl").write_text("\n".join(lines), encoding="utf-8")


def sim(sim_id, task_id, texts):
    messages = [message("assistant", "Hi! How can I help you today?")]
    for text in texts:
        messages.append(message("user", "..."))
        if text.startswith("tool:"):
            messages.append(message("assistant", None, [{"id": "c1", "name": text[5:], "arguments": {}}]))
        else:
            messages.append(message("assistant", text))
    return {"id": sim_id, "task_id": task_id, "trial": 0, "reward_info": {"reward": 1.0}, "messages": messages}


def record(thread, turn, content="", tools=(), events=()):
    return {"thread_id": thread, "turn": turn, "events": list(events), "facts": {}, "fixes": [], "reply": reply(content, tools)}


def test_traces_are_matched_by_reply_sequence(tmp_path):
    simulations = [sim("s1", "t1", ["Who are you?", "tool:get_customer_by_phone"]), sim("s2", "t2", ["Who are you?", "Thanks"])]
    # trace 的写入顺序和对话不同（并发运行），配对只看回复内容
    traces = [
        record("b", 2, "Thanks"),
        record("a", 1, "Who are you?"),
        record("b", 1, "Who are you?"),
        record("a", 2, tools=["get_customer_by_phone"], events=[{"event": "guard_block", "reason": "x"}]),
    ]
    write_run(tmp_path, simulations, traces)
    first, second = load_run(tmp_path)
    assert [r["thread_id"] for r in first.turns] == ["a", "a"] and [r["thread_id"] for r in second.turns] == ["b", "b"]
    # 第 1 轮的对话到 Agent 的第一条回复为止，不含开场问候以外的后续消息
    assert first.messages_until(1)[-1]["content"] == "Who are you?"
    assert first.guard_blocks() == 1 and second.guard_blocks() == 0


def test_old_traces_without_replies_are_left_unmatched(tmp_path):
    old = {"thread_id": "a", "turn": 1, "events": [], "facts": {}}
    write_run(tmp_path, [sim("s1", "t1", ["Hello"])], [old])
    (only,) = load_run(tmp_path)
    assert not only.has_trace


def test_progress_marks_skipped_unknown_ok_and_fault():
    facts = {"customer_identified": "yes", "problem": "mobile_data", "service_available": "yes", "speed_test": "no_connection", "traveling": "no", "mobile_data": "off"}
    intake, paths = progress(ENGINE, facts)
    assert [i.status for i in intake] == ["ok", "ok"]
    assert [p.id for p in paths] == ["service", "data_connectivity", "data_speed"]
    service, data, speed = paths
    assert service.symptom_ok is True
    statuses = {i.id: i.status for i in data.items}
    assert statuses == {"traveling": "ok", "device_roaming": "skipped", "line_roaming": "skipped", "mobile_data": "fault", "data_usage": "unknown"}
    assert data.symptom_ok is False and speed.symptom_ok is False


def test_progress_without_problem_has_no_paths():
    intake, paths = progress(ENGINE, {})
    assert [i.status for i in intake] == ["unknown", "unknown"] and paths == []
