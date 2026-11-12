"""汇总脚本的离线测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_runs import pass_hat_k, summarize  # noqa: E402


def test_pass_hat_k_follows_the_tau_bench_definition():
    outcomes = {"a": [True, True, True], "b": [True, True, False], "c": [False, False, False]}
    assert pass_hat_k(outcomes, 1) == pytest.approx((1 + 2 / 3 + 0) / 3)
    # b 题 3 次中 2 次成功：随机抽 2 次都成功的概率是 C(2,2)/C(3,2) = 1/3
    assert pass_hat_k(outcomes, 2) == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert pass_hat_k(outcomes, 3) == pytest.approx(1 / 3)


def test_tasks_with_fewer_runs_are_left_out_of_larger_k():
    assert pass_hat_k({"a": [True, True], "b": [True]}, 2) == 1.0


def sim(task, reward, transfer=False):
    calls = [{"name": "transfer_to_human_agents", "arguments": {}}] if transfer else []
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "cache_hit_tokens": 90, "cache_miss_tokens": 10}
    messages = [{"role": "assistant", "content": None, "tool_calls": calls, "usage": usage}]
    return {"task_id": task, "reward_info": {"reward": reward}, "messages": messages, "duration": 10.0, "termination_reason": "user_stop", "user_cost": 0.001}


def test_summary_counts_always_and_never_passing_tasks():
    summary = summarize([sim("a", 1.0), sim("a", 1.0), sim("b", 0.0, transfer=True), sim("b", 1.0)])
    assert summary["always_pass"] == 1 and summary["never_pass"] == 0
    assert summary["transfer_rate"] == 0.25 and summary["agent_cost_usd_per_run"] is not None
