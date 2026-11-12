"""从模型的原始响应里读出 token 用量，包括缓存命中数。

τ²-bench 只记录输入和输出 token，不记录缓存命中，所以只能估出一个很宽的费用区间。
DeepSeek 的原始响应里有缓存命中数，这里把它补进消息的 usage，结果文件里就能算出准确费用。
"""

from __future__ import annotations

from typing import Any

# DeepSeek 峰时价，美元/百万 token（与 Deep Research 项目一致）。用于估算，以账户余额变化为准。
PRICE_CACHE_HIT = 0.006
PRICE_CACHE_MISS = 0.3
PRICE_OUTPUT = 1.2


def usage_from_raw(raw: dict[str, Any] | None) -> dict[str, int] | None:
    """返回 prompt、completion、cache_hit、cache_miss 四个字段。没有用量信息时返回 None。"""
    if not raw:
        return None
    usage = raw.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        # LiteLLM 有时把命中数放在 OpenAI 风格的字段里
        hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    hit = int(hit or 0)
    miss = usage.get("prompt_cache_miss_tokens")
    miss = int(miss) if miss is not None else max(prompt - hit, 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "cache_hit_tokens": hit, "cache_miss_tokens": miss}


def estimate_cost_usd(usage: dict[str, int]) -> float:
    return (
        usage["cache_hit_tokens"] * PRICE_CACHE_HIT
        + usage["cache_miss_tokens"] * PRICE_CACHE_MISS
        + usage["completion_tokens"] * PRICE_OUTPUT
    ) / 1e6
