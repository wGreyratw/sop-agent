"""回放一次评测：把 τ²-bench 的对话和 Agent 每轮的 trace 对上，算出每一轮的 SOP 进度。

只读文件，不调用模型。界面在 demo/app.py。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .flow import FlowEngine, ItemSpec, Ledger


def _signature(content: Any, tool_calls: Any) -> tuple[str, tuple[str, ...]]:
    """用回复文本和工具名识别一条回复。结果文件和 trace 都能算出同样的值。"""
    names = tuple(sorted(call.get("name", "") for call in tool_calls or []))
    return ((content or "").strip(), names)


@dataclass
class Simulation:
    id: str
    task_id: str
    trial: int
    reward: float
    messages: list[dict[str, Any]]
    turns: list[dict[str, Any]] = field(default_factory=list)  # 按轮次排好的 trace 记录
    reply_indices: list[int] = field(default_factory=list)  # 每轮 Agent 回复在 messages 里的位置

    @property
    def has_trace(self) -> bool:
        return bool(self.turns)

    def messages_until(self, turn: int) -> list[dict[str, Any]]:
        """到第 turn 轮（从 1 开始）Agent 回复为止的对话。"""
        return self.messages[: self.reply_indices[turn - 1] + 1]

    def guard_blocks(self) -> int:
        return sum(1 for record in self.turns for event in record.get("events", []) if event.get("event") == "guard_block")


def load_traces(path: Path) -> dict[str, list[dict[str, Any]]]:
    threads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            threads[record["thread_id"]].append(record)
    for records in threads.values():
        records.sort(key=lambda record: record["turn"])
    return dict(threads)


def match_traces(simulations: list[Simulation], threads: dict[str, list[dict[str, Any]]]) -> None:
    """按回复序列把 trace 线程配给对话。没有 reply 字段的旧 trace 无法配对，会被跳过。"""
    by_replies: dict[tuple, list[list[dict[str, Any]]]] = defaultdict(list)
    for records in threads.values():
        if all("reply" in record for record in records):
            key = tuple(_signature(r["reply"]["content"], r["reply"]["tool_calls"]) for r in records)
            by_replies[key].append(records)
    for sim in simulations:
        assistant = [i for i, m in enumerate(sim.messages) if m.get("role") == "assistant"]
        signatures = [_signature(sim.messages[i].get("content"), sim.messages[i].get("tool_calls")) for i in assistant]
        # 开场问候由 τ²-bench 发出，不经过 Agent，所以 trace 只对应后面的回复
        for skip in range(len(signatures) + 1):
            candidates = by_replies.get(tuple(signatures[skip:]))
            if candidates:
                sim.turns = candidates.pop(0)
                sim.reply_indices = assistant[skip:]
                break


def load_run(run_dir: str | Path) -> list[Simulation]:
    run_dir = Path(run_dir)
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    simulations = [
        Simulation(
            id=sim["id"],
            task_id=sim["task_id"],
            trial=sim.get("trial") or 0,
            reward=(sim.get("reward_info") or {}).get("reward") or 0.0,
            messages=sim["messages"],
        )
        for sim in data["simulations"]
    ]
    match_traces(simulations, load_traces(run_dir / "trace.jsonl"))
    return simulations


@dataclass
class ItemStatus:
    id: str
    fact: str
    value: str | None
    status: str  # ok / fault / unknown / skipped


@dataclass
class PathStatus:
    id: str
    title: str
    symptom: str
    symptom_value: str | None
    symptom_ok: bool | None
    items: list[ItemStatus]


def _item_status(item: ItemSpec, facts: dict[str, str]) -> ItemStatus:
    value = facts.get(item.fact)
    if not FlowEngine._applicable(item, facts):
        status = "skipped"
    elif value is None:
        status = "unknown"
    else:
        status = "ok" if value in item.ok else "fault"
    return ItemStatus(id=item.id, fact=item.fact, value=value, status=status)


def progress(engine: FlowEngine, facts: dict[str, str]) -> tuple[list[ItemStatus], list[PathStatus]]:
    """开场项目的状态，以及当前问题类型要走的各条路径的状态。"""
    spec = engine.spec
    intake = [_item_status(item, facts) for item in spec.intake]
    paths = []
    for path_id in spec.problems.get(facts.get(spec.problem_fact) or "", []):
        path = spec.paths[path_id]
        value = facts.get(path.symptom)
        paths.append(
            PathStatus(
                id=path_id,
                title=path.title,
                symptom=path.symptom,
                symptom_value=value,
                symptom_ok=None if value is None else value in path.symptom_ok,
                items=[_item_status(item, facts) for item in path.items],
            )
        )
    return intake, paths


def ledger_of(record: dict[str, Any]) -> Ledger:
    return Ledger(facts=record.get("facts") or {}, fixes=record.get("fixes") or [])
