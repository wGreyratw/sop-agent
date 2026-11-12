"""通用的 SOP 流程引擎。不调用模型，也不依赖具体领域。

流程由配置文件描述（见 flows/telecom.yaml），引擎只做三件事：
1. 维护“账本”：已经知道的事实，例如飞行模式=开、流量已用超。
2. 根据账本算出下一个必须做的步骤。
3. 判断此刻是否允许转人工。

流程的抽象：
- 事实（fact）：有若干取值的状态。由模型从顾客回复或工具结果中提取后写入账本。
- 路径（path）：一类问题的排查流程，有一个“症状”事实和一串按顺序检查的项目。
  症状正常就跳过整条路径；症状异常就逐项检查，找到第一个未查或有故障的项目。
- 问题（problem）：按顺序走过的若干路径。例如彩信问题要先确认有信号、能上网，再查彩信本身。

两条规则保证不漏步：
- 修好一项（某个事实从故障值变为正常值）后，所有症状清空，必须重新检查。
- 只有在流程明确允许时才能转人工：某项故障无法修复，或者一条路径的项目都正常而症状依旧。
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class FactSpec(BaseModel):
    description: str
    values: list[str]
    source: str = Field(description="谁能提供这个事实：user（顾客在手机上查看）或 agent（Agent 调用工具）")


class ItemSpec(BaseModel):
    id: str
    fact: str
    ok: list[str] = Field(description="视为正常的取值")
    check: str | None = Field(default=None, description="如何得到这个事实。为空表示这是一个必须执行的动作，未执行即视为需要修复")
    fix: str | None = Field(default=None, description="取值不正常时如何修复。为空表示只能转人工")
    escalate_if: list[str] = Field(default_factory=list, description="出现这些取值时直接允许转人工")
    when: dict[str, list[str]] = Field(default_factory=dict, description="只有这些事实取这些值时，该项才适用")


class PathSpec(BaseModel):
    title: str
    symptom: str
    symptom_ok: list[str]
    check_symptom: str
    items: list[ItemSpec]


class FlowSpec(BaseModel):
    name: str
    facts: dict[str, FactSpec]
    intake: list[ItemSpec] = Field(description="走任何路径之前必须完成的项目，例如核实身份、确认问题类型")
    problem_fact: str = Field(description="哪个事实决定问题类型")
    problems: dict[str, list[str]] = Field(description="问题类型 → 依次要走的路径")
    paths: dict[str, PathSpec]
    escalate: str = Field(default="Transfer the customer to a human agent now, following the policy.", description="允许转人工时给模型的指令")
    fix_tools: list[str] = Field(
        default_factory=list,
        description="Agent 自己调用的修复工具（例如开通漫游）。调用成功后清空所有症状，必须重新检查",
    )

    @model_validator(mode="after")
    def _check_references(self) -> "FlowSpec":
        def check_item(item: ItemSpec, where: str) -> None:
            if item.fact not in self.facts:
                raise ValueError(f"{where}/{item.id}: 未定义的事实 {item.fact}")
            allowed = set(self.facts[item.fact].values)
            unknown = (set(item.ok) | set(item.escalate_if)) - allowed
            if unknown:
                raise ValueError(f"{where}/{item.id}: {item.fact} 没有取值 {sorted(unknown)}")
            for fact, values in item.when.items():
                if fact not in self.facts or set(values) - set(self.facts[fact].values):
                    raise ValueError(f"{where}/{item.id}: when 条件引用了未定义的事实或取值 {fact}={values}")

        for item in self.intake:
            check_item(item, "intake")
        if self.problem_fact not in self.facts:
            raise ValueError(f"未定义的问题类型事实 {self.problem_fact}")
        if set(self.problems) != set(self.facts[self.problem_fact].values):
            raise ValueError("problems 的键必须和问题类型事实的取值一一对应")
        for problem, path_ids in self.problems.items():
            missing = [p for p in path_ids if p not in self.paths]
            if missing:
                raise ValueError(f"问题 {problem} 引用了未定义的路径 {missing}")
        for path_id, path in self.paths.items():
            if path.symptom not in self.facts:
                raise ValueError(f"路径 {path_id}: 未定义的症状事实 {path.symptom}")
            for item in path.items:
                check_item(item, path_id)
        return self

    @property
    def symptoms(self) -> set[str]:
        return {path.symptom for path in self.paths.values()}


def load_flow(path: str | Path) -> FlowSpec:
    return FlowSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class StepKind(str, Enum):
    CHECK = "check"
    FIX = "fix"
    CHECK_SYMPTOM = "check_symptom"
    ESCALATE = "escalate"
    RESOLVED = "resolved"
    # 以下用于请求清单（零售），排障流程不会产生
    COLLECT = "collect"
    CONFIRM = "confirm"
    EXECUTE = "execute"
    DENY = "deny"


class Step(BaseModel):
    kind: StepKind
    instruction: str
    path: str | None = None
    item: str | None = None
    fact: str | None = None


class Ledger(BaseModel):
    """排查账本。只通过 FlowEngine.update 修改，以便统一处理“修好后重新检查症状”。"""

    facts: dict[str, str] = Field(default_factory=dict)
    fixes: list[str] = Field(default_factory=list, description="已修好的项目，按发生顺序")


class FlowEngine:
    def __init__(self, spec: FlowSpec) -> None:
        self.spec = spec
        self._ok_values: dict[str, set[str]] = {}
        for item in self._all_items():
            # 同一个事实在不同项目里的“正常”标准可能不同（例如网络制式）。
            # 只要某个取值在任一项目里是故障值，从它变为别的值就算一次修复。
            self._ok_values.setdefault(item.fact, set(self.spec.facts[item.fact].values))
            self._ok_values[item.fact] &= set(item.ok)

    def _all_items(self) -> list[ItemSpec]:
        return [*self.spec.intake, *(item for path in self.spec.paths.values() for item in path.items)]

    def update(self, ledger: Ledger, new_facts: dict[str, str]) -> Ledger:
        """写入新事实。忽略未定义的事实和取值，返回新的账本。"""
        facts = dict(ledger.facts)
        fixes = list(ledger.fixes)
        fixed = False
        for name, value in new_facts.items():
            spec = self.spec.facts.get(name)
            if spec is None or value not in spec.values:
                continue
            before = facts.get(name)
            facts[name] = value
            if before is not None and before != value and name in self._ok_values:
                if before not in self._ok_values[name]:
                    fixed = True
                    fixes.append(f"{name}: {before} -> {value}")
        if fixed:
            # 修好一项后必须重新确认症状，除非这次更新里同时给出了新的症状结果
            for symptom in self.spec.symptoms:
                if symptom not in new_facts:
                    facts.pop(symptom, None)
        return Ledger(facts=facts, fixes=fixes)

    def apply_tool_fix(self, ledger: Ledger, tool_name: str) -> Ledger:
        """Agent 自己用工具完成了一次修复：清空所有症状，要求重新检查。

        顾客那边的修复靠“事实从故障值变为正常值”来识别；Agent 的修复不一定会被提取成这样的变化
        （例如提取时从没记下“漫游未开通”），所以按工具调用本身来判断，更可靠。
        """
        if tool_name not in self.spec.fix_tools:
            return ledger
        facts = {name: value for name, value in ledger.facts.items() if name not in self.spec.symptoms}
        return Ledger(facts=facts, fixes=[*ledger.fixes, f"tool {tool_name}"])

    def next_step(self, ledger: Ledger) -> Step:
        facts = ledger.facts
        for item in self.spec.intake:
            step = self._item_step(item, facts, path=None)
            if step is not None:
                return step
        problem = facts.get(self.spec.problem_fact)
        for path_id in self.spec.problems[problem]:
            path = self.spec.paths[path_id]
            symptom = facts.get(path.symptom)
            if symptom is None:
                # 已知一个只能转人工的故障（例如 SIM 卡被锁）时，不必再检查症状。
                # 回归：Agent 想转人工，引擎坚持先看状态栏，守卫拦了两次才放行
                for item in path.items:
                    if self._applicable(item, facts) and facts.get(item.fact) in item.escalate_if:
                        return self._item_step(item, facts, path=path_id)
                return Step(kind=StepKind.CHECK_SYMPTOM, instruction=path.check_symptom, path=path_id, fact=path.symptom)
            if symptom in path.symptom_ok:
                continue
            for item in path.items:
                step = self._item_step(item, facts, path=path_id)
                if step is not None:
                    return step
            return Step(
                kind=StepKind.ESCALATE,
                instruction=f"All steps for '{path.title}' are done and the issue persists. {self.spec.escalate}",
                path=path_id,
                fact=path.symptom,
            )
        # 这条指令会给模型看，所以用英文
        return Step(kind=StepKind.RESOLVED, instruction="All relevant issues are resolved. Confirm the customer has nothing else they need, then end the conversation.")

    @staticmethod
    def _applicable(item: ItemSpec, facts: dict[str, str]) -> bool:
        return all(facts.get(fact) in values for fact, values in item.when.items())

    def _item_step(self, item: ItemSpec, facts: dict[str, str], path: str | None) -> Step | None:
        if not self._applicable(item, facts):
            return None
        value = facts.get(item.fact)
        if value is None and item.check is not None:
            return Step(kind=StepKind.CHECK, instruction=item.check, path=path, item=item.id, fact=item.fact)
        if value in item.ok:
            return None
        if value in item.escalate_if or item.fix is None:
            return Step(
                kind=StepKind.ESCALATE,
                instruction=f"{item.fact} is {value}, which cannot be resolved here. {self.spec.escalate}",
                path=path,
                item=item.id,
                fact=item.fact,
            )
        return Step(kind=StepKind.FIX, instruction=item.fix, path=path, item=item.id, fact=item.fact)

    def transfer_allowed(self, ledger: Ledger) -> bool:
        return self.next_step(ledger).kind == StepKind.ESCALATE

    def summary(self, ledger: Ledger) -> str:
        """给模型看的账本摘要：已知事实和已完成的修复。"""
        lines = [f"- {name} = {value}" for name, value in sorted(ledger.facts.items())]
        if ledger.fixes:
            lines.append("已修复：" + "；".join(ledger.fixes))
        return "\n".join(lines) or "（尚无已确认的事实）"
