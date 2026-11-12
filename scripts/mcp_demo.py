"""演示：像一个接入 MCP 的客服 Agent 那样，通过 stdio 使用 SOP 引擎走一遍彩信排障。

不调用模型、不产生费用。每一步的“顾客说了什么”是写死的剧本，真实场景里由 Agent 从对话中判断。
运行：.venv/Scripts/python scripts/mcp_demo.py
"""

from __future__ import annotations

import sys

import anyio
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

TRANSFER = [{"name": "transfer_to_human_agents", "arguments": {"summary": "MMS still failing"}}]

# (这一步发生了什么, 记录的事实)
SCRIPT = [
    ("查到顾客账户；顾客说彩信发不出去", {"customer_identified": "yes", "problem": "mms"}),
    ("顾客看了状态栏：有信号", {"service_available": "yes"}),
    ("顾客测网速：Fair", {"speed_test": "fair", "mms_works": "no"}),
    ("网络制式是 2G", {"network_mode": "2g"}),
    ("顾客把制式改成 4G", {"network_mode": "4g"}),
]


async def call(client: Client, tool: str, **arguments):
    result = await client.call_tool(tool, arguments)
    if result.is_error:
        raise RuntimeError(result.content[0].text)
    return result.structured_content


def show_step(step: dict) -> str:
    target = step["item"] or step["path"] or ""
    return f"{step['kind']} {target}".strip()


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "sop_agent.mcp_server"])
    async with Client(params) as client:
        tools = [tool.name for tool in (await client.list_tools()).tools]
        print(f"已连接 sop-engine，工具：{', '.join(tools)}\n")
        case = await call(client, "start_case", flow="telecom")
        case_id = case["case_id"]
        print(f"开工单 {case_id}，下一步：{show_step(case['next_step'])}")

        for what, facts in SCRIPT:
            result = await call(client, "record_facts", case_id=case_id, facts=facts)
            fixes = f"，完成修复：{result['new_fixes']}" if result["new_fixes"] else ""
            print(f"- {what}{fixes}\n  下一步：{show_step(result['next_step'])}")

        verdict = await call(client, "check_action", case_id=case_id, tool_calls=TRANSFER)
        print(f"\nAgent 想现在转人工 → 守卫{'放行' if verdict['allowed'] else '拦截'}：{verdict['reason']}")

        status = await call(client, "get_case", case_id=case_id)
        print("\n当前进度：")
        for path in status["paths"]:
            marks = " ".join(f"{item['id']}={item['status']}" for item in path["items"])
            print(f"  {path['id']}: 症状 {path['symptom_value']} | {marks}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    anyio.run(main)
