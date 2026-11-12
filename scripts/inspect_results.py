"""查看 τ²-bench 一次运行的结果：汇总成绩、费用、耗时，并展开失败的题目。不调用模型。

用法：python scripts/inspect_results.py <results.json> [--show-failures]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CACHE_MISS, CACHE_HIT, OUTPUT = 0.3e-6, 0.006e-6, 1.2e-6
# 会改变数据库的工具。只读工具（get_* / find_* / list_* / calculate）不在这里
WRITE_TOOLS = {
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
}


def agent_cost(sim: dict) -> tuple[float, float, int]:
    usage = [m["usage"] for m in sim["messages"] if m.get("role") == "assistant" and m.get("usage")]
    prompt = sum(u["prompt_tokens"] for u in usage)
    completion = sum(u["completion_tokens"] for u in usage)
    return prompt * CACHE_HIT + completion * OUTPUT, prompt * CACHE_MISS + completion * OUTPUT, len(usage)


def tool_calls(sim: dict) -> list[dict]:
    calls = []
    for message in sim["messages"]:
        for call in message.get("tool_calls") or []:
            calls.append({"name": call["name"], "arguments": call.get("arguments")})
    return calls


def show_failure(sim: dict, task: dict) -> None:
    info = sim.get("reward_info") or {}
    criteria = task["evaluation_criteria"]
    print(f"\n========== task {sim['task_id']}  reward={info.get('reward')}  终止={sim['termination_reason']}")
    print("分项:", info.get("reward_breakdown"))
    print("顾客任务:", json.dumps(task["user_scenario"]["instructions"], ensure_ascii=False)[:900])
    print("参考写操作:", [(a["name"], a.get("arguments")) for a in criteria.get("actions") or [] if a["name"] in WRITE_TOOLS])
    print("Agent 写操作:", [(c["name"], c["arguments"]) for c in tool_calls(sim) if c["name"] in WRITE_TOOLS])
    for check in info.get("nl_assertions") or []:
        if not check.get("met"):
            print("未满足的断言:", check.get("nl_assertion"), "|", check.get("justification"))
    db = info.get("db_check") or {}
    if db and not db.get("db_match", True):
        print("数据库最终状态和参考不一致")
    print("--- 对话（截断）")
    for message in sim["messages"]:
        role = message.get("role")
        if role == "tool":
            print(f"  [tool] {str(message.get('content'))[:160]}")
        elif message.get("tool_calls"):
            for call in message["tool_calls"]:
                print(f"  [agent→{call['name']}] {json.dumps(call.get('arguments'), ensure_ascii=False)[:200]}")
        else:
            print(f"  [{role}] {str(message.get('content'))[:260]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    tasks = {task["id"]: task for task in data["tasks"]}
    sims = data["simulations"]
    rewards = [(sim.get("reward_info") or {}).get("reward", 0.0) for sim in sims]
    costs = [agent_cost(sim) for sim in sims]
    user = [sim.get("user_cost") or 0.0 for sim in sims]
    durations = [sim["duration"] for sim in sims]
    failed = [sim for sim, reward in zip(sims, rewards) if reward < 1.0]
    print(f"题目 {len(sims)}，通过 {len(sims) - len(failed)}，成功率 {sum(r >= 1.0 for r in rewards) / len(sims):.3f}")
    print(f"失败题: {sorted((sim['task_id'] for sim in failed), key=lambda x: int(x) if x.isdigit() else x)}")
    print(f"Agent 费用 ${sum(c[0] for c in costs):.3f}-${sum(c[1] for c in costs):.3f}，模拟顾客 ${sum(user):.3f}")
    print(f"Agent 调用次数 中位 {statistics.median(c[2] for c in costs)}，最多 {max(c[2] for c in costs)}")
    print(f"单题耗时 中位 {statistics.median(durations):.0f}s，最长 {max(durations):.0f}s")
    print("终止原因:", {reason: sum(sim['termination_reason'] == reason for sim in sims) for reason in {sim['termination_reason'] for sim in sims}})
    if args.show_failures:
        for sim in failed:
            show_failure(sim, tasks[sim["task_id"]])


if __name__ == "__main__":
    main()
