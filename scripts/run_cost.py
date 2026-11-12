"""从 τ²-bench 的结果文件计算每道题 Agent 的费用。

本项目的 Agent 会在消息的 usage 里记录缓存命中数，可以算出准确费用。
τ²-bench 自带的 Agent 没有这项记录，只能给出区间：上限按全部未命中算，下限按全部命中算。
价格是 DeepSeek 峰时价（美元/百万 token），和 Deep Research 项目一致。真实花费以账户余额变化为准。

用法：python scripts/run_cost.py <results.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sop_agent.usage import PRICE_CACHE_HIT, PRICE_CACHE_MISS, PRICE_OUTPUT

HIT, MISS, OUT = PRICE_CACHE_HIT / 1e6, PRICE_CACHE_MISS / 1e6, PRICE_OUTPUT / 1e6


def conversation_cost(sim: dict) -> tuple[float, float, int, int, bool]:
    """返回 (下限, 上限, 调用次数, 输入 token, 是否有缓存数据)。有缓存数据时上下限相同。"""
    usage = [m["usage"] for m in sim["messages"] if m.get("role") == "assistant" and m.get("usage")]
    exact = bool(usage) and all("cache_hit_tokens" in u for u in usage)
    prompt = sum(u["prompt_tokens"] for u in usage)
    completion = sum(u["completion_tokens"] for u in usage)
    if exact:
        cost = sum(u["cache_hit_tokens"] * HIT + u["cache_miss_tokens"] * MISS for u in usage) + completion * OUT
        return cost, cost, len(usage), prompt, True
    return prompt * HIT + completion * OUT, prompt * MISS + completion * OUT, len(usage), prompt, False


def main(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    total_low = total_high = hits = prompts = 0.0
    for sim in data["simulations"]:
        low, high, calls, prompt, exact = conversation_cost(sim)
        total_low += low
        total_high += high
        reward = (sim.get("reward_info") or {}).get("reward")
        cost = f"${low:.4f}" if exact else f"${low:.4f}-${high:.4f}"
        print(f"task {sim['task_id'][:70]}: reward {reward}, {sim['duration']:.0f}s, Agent 调用 {calls} 次, Agent {cost}, 模拟顾客 ${sim.get('user_cost') or 0:.4f}")
        if exact:
            usage = [m["usage"] for m in sim["messages"] if m.get("role") == "assistant" and m.get("usage")]
            hits += sum(u["cache_hit_tokens"] for u in usage)
            prompts += prompt
    count = len(data["simulations"])
    print(f"Agent 合计 ${total_low:.4f}" + ("" if total_low == total_high else f"-${total_high:.4f}") + f"，平均 ${total_low / count:.4f}/题")
    if prompts:
        print(f"缓存命中率 {hits / prompts:.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
