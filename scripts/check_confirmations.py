"""检查每次写操作之前，顾客最近一句话是否明确同意。不调用模型。

店规要求：改数据库前，必须列出操作细节并得到用户明确的 yes。τ²-bench 只看最终数据库和断言，
不单独统计这一条，所以这里用简单规则粗查。规则可能误判，结果要人工抽查。

用法：python scripts/check_confirmations.py <results.json>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from inspect_results import WRITE_TOOLS

AGREE = re.compile(r"\b(yes|yeah|yep|sure|go ahead|proceed|confirm|correct|that'?s right|do it|sounds (good|right)|please do)\b", re.IGNORECASE)


def main(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    writes = flagged = 0
    for sim in data["simulations"]:
        last_user = ""
        for message in sim["messages"]:
            if message.get("role") == "user":
                last_user = str(message.get("content") or "")
            for call in message.get("tool_calls") or []:
                if call["name"] not in WRITE_TOOLS:
                    continue
                writes += 1
                if not AGREE.search(last_user):
                    flagged += 1
                    print(f"task {sim['task_id']} {call['name']}: 顾客最近一句是“{last_user[:150]}”")
    print(f"写操作 {writes} 次，前一句没有明确同意的 {flagged} 次")


if __name__ == "__main__":
    main(sys.argv[1])
