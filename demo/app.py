"""SOP-Agent 回放界面：逐轮查看对话，以及程序记录的事实、SOP 进度和守卫拦截。

运行：.venv/Scripts/streamlit run demo/app.py
只读取评测结果和 trace，不调用模型，不产生费用。
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from sop_agent.flow import FlowEngine, load_flow
from sop_agent.replay import Simulation, ledger_of, load_run, progress

ROOT = Path(__file__).resolve().parents[1]
ENGINE = FlowEngine(load_flow(ROOT / "flows" / "telecom.yaml"))

STEP_NAMES = {"check": "检查", "fix": "修复", "check_symptom": "重新确认症状", "escalate": "转人工", "resolved": "已解决，结束对话"}
PATH_NAMES = {"service": "基础信号", "data_connectivity": "数据连通", "data_speed": "网速", "mms": "彩信设置"}
ITEM_NAMES = {
    "identify_customer": "核实顾客身份",
    "classify_problem": "判断问题类型",
    "airplane_mode": "飞行模式",
    "sim_card": "SIM 卡状态",
    "reset_apn": "重置 APN",
    "line_suspension": "线路是否停机",
    "traveling": "是否在国外",
    "device_roaming": "手机漫游开关",
    "line_roaming": "线路漫游权限",
    "mobile_data": "移动数据开关",
    "data_usage": "流量是否用超",
    "data_saver": "省流模式",
    "network_mode_speed": "网络制式",
    "vpn": "VPN",
    "network_mode_mms": "网络制式",
    "wifi_calling": "WiFi 通话",
    "messaging_permissions": "短信应用权限",
    "mmsc_url": "彩信服务器地址",
}
STATUS_ICONS = {"ok": "✅", "fault": "❌", "unknown": "⬜", "skipped": "➖"}


def find_runs() -> list[Path]:
    """能回放的运行：有结果文件，也有带回复内容的 trace。"""
    candidates = [ROOT / "demo" / "sample_run", *sorted((ROOT / "outputs").glob("*"))]
    runs = []
    for path in candidates:
        trace = path / "trace.jsonl"
        if (path / "results.json").exists() and trace.exists():
            first = trace.read_text(encoding="utf-8").split("\n", 1)[0]
            if first and "reply" in json.loads(first):
                runs.append(path)
    return runs


@st.cache_data
def cached_run(path: str) -> list[Simulation]:
    sims = [sim for sim in load_run(path) if sim.has_trace]
    # 默认展示通过的长题：最能看出逐项排查
    return sorted(sims, key=lambda sim: (sim.reward < 1, -len(sim.turns)))


def show_message(msg: dict) -> None:
    role, content, calls = msg.get("role"), msg.get("content"), msg.get("tool_calls") or []
    if role == "tool":
        who = "Agent 工具" if msg.get("requestor") == "assistant" else "顾客手机"
        with st.expander(f"{who}返回结果", expanded=False):
            st.code(content or "", language="json")
        return
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message("user" if role == "user" else "assistant", avatar=avatar):
        if content:
            st.markdown(content)
        for call in calls:
            args = json.dumps(call.get("arguments") or {}, ensure_ascii=False)
            label = "在手机上操作" if role == "user" else "调用工具"
            st.caption(f"{label}：`{call.get('name')}({args})`")


def show_progress(facts: dict, current_item: str | None, current_path: str | None) -> None:
    intake, paths = progress(ENGINE, facts)
    lines = ["**开场**"]
    for item in intake:
        mark = "　◀ 当前" if item.id == current_item and current_path is None else ""
        lines.append(f"- {STATUS_ICONS[item.status]} {ITEM_NAMES.get(item.id, item.id)}{mark}")
    for path in paths:
        if path.symptom_ok is None:
            symptom = "症状未确认"
        else:
            symptom = f"症状 {path.symptom} = {path.symptom_value}，" + ("正常，跳过" if path.symptom_ok else "异常，逐项排查")
        here = "　◀ 当前" if path.id == current_path and current_item is None else ""
        lines.append(f"\n**{PATH_NAMES.get(path.id, path.id)}**（{symptom}）{here}")
        for item in path.items:
            value = f"：{item.value}" if item.value else ""
            mark = "　◀ 当前" if item.id == current_item and path.id == current_path else ""
            # 症状正常时整条路径跳过，没查的项不算遗漏
            status = "skipped" if path.symptom_ok and item.status == "unknown" else item.status
            lines.append(f"- {STATUS_ICONS[status]} {ITEM_NAMES.get(item.id, item.id)}{value}{mark}")
    st.markdown("\n".join(lines))
    st.caption("✅ 正常　❌ 故障　⬜ 未检查　➖ 不适用")


def main() -> None:
    st.set_page_config(page_title="SOP-Agent 回放", layout="wide")
    st.title("SOP-Agent 回放")
    st.caption("左边是这一轮的对话，右边是程序记录的进度：模型负责说话，流程由程序掌控。")

    runs = find_runs()
    if not runs:
        st.warning("没有可回放的运行。先按 README 录制一次评测（需要带回复内容的 trace）。")
        return
    with st.sidebar:
        run = st.selectbox("评测运行", runs, format_func=lambda p: p.name)
        sims = cached_run(str(run))
        if not sims:
            st.warning("这次运行的对话和 trace 配不上。")
            return
        sim = st.selectbox(
            "题目",
            sims,
            format_func=lambda s: f"{'通过' if s.reward >= 1 else '失败'} · 第{s.trial + 1}次 · {s.task_id[:60]}",
        )
        st.caption(sim.task_id)
        turn = st.slider("第几轮", 1, len(sim.turns), len(sim.turns), key=f"turn-{sim.id}")

    record = sim.turns[turn - 1]
    events = record.get("events", [])
    blocks = [e for e in events if e.get("event") == "guard_block"]

    cols = st.columns(4)
    cols[0].metric("结果", "通过" if sim.reward >= 1 else "失败")
    cols[1].metric("Agent 回复轮数", len(sim.turns))
    cols[2].metric("守卫拦截（整题）", sim.guard_blocks())
    cols[3].metric("已完成修复（到本轮）", len(record.get("fixes") or []))

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader(f"第 {turn} 轮")
        end = sim.reply_indices[turn - 1]
        start = sim.reply_indices[turn - 2] + 1 if turn > 1 else 0
        for msg in sim.messages[start : end + 1]:
            show_message(msg)
        if start > 0:
            with st.expander("之前的对话"):
                for msg in sim.messages[:start]:
                    show_message(msg)

    with right:
        responds = [e for e in events if e.get("event") == "respond"]
        step = responds[0] if responds else {}
        st.subheader("程序给出的下一步")
        if step:
            target = ITEM_NAMES.get(step.get("item"), step.get("item")) or PATH_NAMES.get(step.get("path"), "")
            st.info(f"**{STEP_NAMES.get(step['step'], step['step'])}**　{target or ''}")
        found = next((e.get("found") for e in events if e.get("event") == "extract"), None)
        if found:
            st.markdown("本轮提取到的事实：" + "，".join(f"`{k} = {v}`" for k, v in found.items()))
        for block in blocks:
            st.error(f"守卫拦截了草稿，要求重写：{block.get('reason')}")
        for override in (e for e in events if e.get("event") == "guard_override"):
            st.warning(f"重写 2 次仍违规，按原样放行：{override.get('reason')}")

        st.subheader("SOP 进度")
        show_progress(ledger_of(record).facts, step.get("item"), step.get("path"))
        if record.get("fixes"):
            st.markdown("**已完成的修复**\n" + "\n".join(f"- {fix}" for fix in record["fixes"]))


main()
