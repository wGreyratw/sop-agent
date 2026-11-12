"""查询 DeepSeek 账户余额。在跑真实模型前使用，避免运行到一半余额耗尽。

只打印余额，不打印密钥。密钥从项目根目录的 .env 读取。

用法：python scripts/check_balance.py [预计花费（元）]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> int:
    # 中文 Windows 控制台默认 GBK，打印“¥”会报错
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("没有找到 DEEPSEEK_API_KEY，请检查 .env", file=sys.stderr)
        return 1
    request = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    total = 0.0
    for info in data.get("balance_infos", []):
        print(f"余额 {info['total_balance']} {info['currency']}（充值 {info['topped_up_balance']}，赠送 {info['granted_balance']}）")
        if info["currency"] == "CNY":
            total = float(info["total_balance"])
    if not data.get("is_available"):
        print("账户当前不可用")
        return 2
    if len(sys.argv) > 1:
        needed = float(sys.argv[1])
        if total < needed * 1.5:
            print(f"余额不足以覆盖预计花费 ¥{needed} 的 1.5 倍，建议先充值")
            return 3
        print(f"余额足够（预计花费 ¥{needed}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
