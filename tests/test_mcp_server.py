"""SOP 引擎 MCP Server 的离线测试：通过 MCP 协议调用，不调用模型。"""

from __future__ import annotations

import sys

import anyio
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

from sop_agent.mcp_server import server

MMS_READY = {"customer_identified": "yes", "problem": "mms", "service_available": "yes", "speed_test": "fair", "mms_works": "no"}
MMS_EXHAUSTED = {"network_mode": "4g", "wifi_calling": "off", "messaging_permissions": "granted", "mmsc_url": "present"}
TRANSFER = [{"name": "transfer_to_human_agents", "arguments": {"summary": "x"}}]


def run(steps):
    """在一个 MCP 会话里依次执行 steps(client)，返回它的结果。"""

    async def main():
        async with Client(server) as client:
            return await steps(client)

    return anyio.run(main)


async def call(client, tool, **arguments):
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    return result.structured_content


def test_tools_and_flow_definition_are_exposed():
    async def steps(client):
        tools = {tool.name for tool in (await client.list_tools()).tools}
        flow = await call(client, "get_flow")
        source = await client.read_resource("sop://flows/telecom")
        return tools, flow, source.contents[0].text

    tools, flow, source = run(steps)
    assert tools == {"get_flow", "start_case", "record_facts", "record_fix_tool", "check_action", "get_case"}
    assert flow["problems"]["mms"] == ["service", "data_connectivity", "mms"]
    assert flow["facts"]["airplane_mode"]["values"] == ["on", "off"]
    assert source.startswith("# 电信技术支持流程")


def test_guard_blocks_early_transfer_and_requires_it_when_exhausted():
    async def steps(client):
        case = (await call(client, "start_case"))["case_id"]
        first = await call(client, "record_facts", case_id=case, facts=MMS_READY)
        early = await call(client, "check_action", case_id=case, tool_calls=TRANSFER)
        last = await call(client, "record_facts", case_id=case, facts=MMS_EXHAUSTED)
        talk = await call(client, "check_action", case_id=case, tool_calls=[])
        transfer = await call(client, "check_action", case_id=case, tool_calls=TRANSFER)
        after = await call(client, "check_action", case_id=case)
        return first, early, last, talk, transfer, after

    first, early, last, talk, transfer, after = run(steps)
    assert first["next_step"]["item"] == "network_mode_mms"
    assert not early["allowed"] and "Transfer blocked" in early["reason"]
    assert last["next_step"]["kind"] == "escalate"
    # 该转人工时，只回复文字也会被拦下
    assert not talk["allowed"] and "must be the transfer" in talk["reason"]
    assert transfer["allowed"] and after["allowed"]


def test_fix_clears_symptoms_and_unknown_values_are_ignored():
    async def steps(client):
        case = (await call(client, "start_case"))["case_id"]
        await call(client, "record_facts", case_id=case, facts={"customer_identified": "yes", "problem": "no_service", "service_available": "no", "airplane_mode": "on"})
        fixed = await call(client, "record_facts", case_id=case, facts={"airplane_mode": "off", "vpn": "maybe", "colour": "red"})
        status = await call(client, "get_case", case_id=case)
        return fixed, status

    fixed, status = run(steps)
    assert fixed["new_fixes"] == ["airplane_mode: on -> off"]
    assert fixed["ignored"] == {"vpn": "maybe", "colour": "red"}
    assert fixed["next_step"]["kind"] == "check_symptom"
    assert "service_available" not in status["facts"] and status["paths"][0]["symptom_ok"] is None


def test_fix_tool_requires_retest():
    roaming = {"customer_identified": "yes", "problem": "mobile_data", "service_available": "yes", "speed_test": "no_connection", "traveling": "yes", "device_roaming": "on", "line_roaming": "disabled"}

    async def steps(client):
        case = (await call(client, "start_case"))["case_id"]
        before = await call(client, "record_facts", case_id=case, facts=roaming)
        failed = await call(client, "record_fix_tool", case_id=case, tool_name="enable_roaming", succeeded=False)
        done = await call(client, "record_fix_tool", case_id=case, tool_name="enable_roaming")
        return before, failed, done

    before, failed, done = run(steps)
    assert before["next_step"]["kind"] == "fix" and before["next_step"]["item"] == "line_roaming"
    assert not failed["counted_as_fix"] and failed["next_step"]["kind"] == "fix"
    assert done["counted_as_fix"] and done["next_step"]["kind"] == "check_symptom"


def test_mistakes_are_reported_to_the_client():
    async def steps(client):
        return await client.call_tool("record_facts", {"case_id": "missing", "facts": {}}), await client.call_tool("start_case", {"flow": "nope"})

    missing, unknown_flow = run(steps)
    assert missing.is_error and "Call start_case first" in missing.content[0].text
    assert unknown_flow.is_error and "Available: telecom" in unknown_flow.content[0].text


def test_server_runs_over_stdio():
    """和真实接入一样，以子进程启动服务端，通过 stdio 通信。"""
    params = StdioServerParameters(command=sys.executable, args=["-m", "sop_agent.mcp_server"])

    async def main():
        async with Client(params) as client:
            return await call(client, "start_case")

    started = anyio.run(main)
    assert started["next_step"]["item"] == "identify_customer"
