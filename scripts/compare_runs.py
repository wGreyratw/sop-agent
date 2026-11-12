"""对比两次 τ²-bench 运行（通常是基线和本项目的 Agent），输出最终成绩。不调用模型。

pass^k 按 τ-bench 论文的定义计算：每道题跑 n 次、成功 c 次，pass^k = C(c, k) / C(n, k)，
再对所有题取平均。它表示“随机抽 k 次运行都成功”的概率，k 越大越能反映稳定性。
因基础设施错误（例如余额不足）中断的运行不计入。

用法：python scripts/compare_runs.py <基线 results.json> <新 Agent results.json> [--json 输出路径]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path

from run_cost import conversation_cost

TRANSFER_TOOL = "transfer_to_human_agents"


def load(path: str) -> list[dict]:
    sims = json.loads(Path(path).read_text(encoding="utf-8"))["simulations"]
    return [sim for sim in sims if sim["termination_reason"] != "infrastructure_error"]


def passed(sim: dict) -> bool:
    return ((sim.get("reward_info") or {}).get("reward") or 0.0) >= 1.0


def transferred(sim: dict) -> bool:
    return any(call["name"] == TRANSFER_TOOL for message in sim["messages"] for call in message.get("tool_calls") or [])


def by_task(sims: list[dict]) -> dict[str, list[bool]]:
    result: dict[str, list[bool]] = defaultdict(list)
    for sim in sims:
        result[sim["task_id"]].append(passed(sim))
    return dict(result)


def pass_hat_k(outcomes: dict[str, list[bool]], k: int) -> float | None:
    usable = {task: runs for task, runs in outcomes.items() if len(runs) >= k}
    if not usable:
        return None
    return sum(comb(sum(runs), k) / comb(len(runs), k) for runs in usable.values()) / len(usable)


def summarize(sims: list[dict]) -> dict:
    outcomes = by_task(sims)
    trials = max((len(runs) for runs in outcomes.values()), default=0)
    costs = [conversation_cost(sim) for sim in sims]
    exact = all(cost[4] for cost in costs)
    return {
        "tasks": len(outcomes),
        "runs": len(sims),
        "trials_per_task": trials,
        **{f"pass^{k}": pass_hat_k(outcomes, k) for k in range(1, trials + 1)},
        "always_pass": sum(all(runs) for runs in outcomes.values()),
        "never_pass": sum(not any(runs) for runs in outcomes.values()),
        "transfer_rate": sum(transferred(sim) for sim in sims) / len(sims),
        "median_messages": statistics.median(len(sim["messages"]) for sim in sims),
        "median_seconds": statistics.median(sim["duration"] for sim in sims),
        "agent_cost_usd_per_run": sum(cost[0] for cost in costs) / len(sims) if exact else None,
        "agent_cost_usd_per_run_range": None if exact else [sum(c[0] for c in costs) / len(sims), sum(c[1] for c in costs) / len(sims)],
        "user_cost_usd_per_run": sum(sim.get("user_cost") or 0.0 for sim in sims) / len(sims),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--json", help="把汇总写入这个文件")
    args = parser.parse_args()

    base_sims, cand_sims = load(args.baseline), load(args.candidate)
    base, cand = summarize(base_sims), summarize(cand_sims)
    print(f"{'指标':<28}{'基线':>12}{'新 Agent':>12}")
    for key in base:
        if key.endswith("_range"):
            continue
        left, right = base[key], cand.get(key)
        fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"{key:<28}{fmt(left):>12}{fmt(right):>12}")
    if base.get("agent_cost_usd_per_run_range"):
        low, high = base["agent_cost_usd_per_run_range"]
        print(f"基线 Agent 费用没有缓存数据，区间 ${low:.4f}-${high:.4f}/次")

    base_tasks, cand_tasks = by_task(base_sims), by_task(cand_sims)
    common = sorted(set(base_tasks) & set(cand_tasks))
    print("\n逐题（成功次数/运行次数），只列两者不同的题：")
    changed = 0
    for task in common:
        b, c = base_tasks[task], cand_tasks[task]
        if sum(b) / len(b) != sum(c) / len(c):
            changed += 1
            print(f"  基线 {sum(b)}/{len(b)}  新 {sum(c)}/{len(c)}  {task}")
    print(f"共 {len(common)} 题，其中 {changed} 题结果不同")

    if args.json:
        payload = {"baseline": base, "candidate": cand, "per_task": {t: {"baseline": base_tasks[t], "candidate": cand_tasks[t]} for t in common}}
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
