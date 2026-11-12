"""运行 τ²-bench 的命令行，但把自然语言断言的评分模型换成 DeepSeek。

为什么需要这个脚本：零售题的得分 = 数据库是否改对 × 自然语言断言是否成立。
断言由 LLM 判断，τ²-bench 把评分模型写死为 OpenAI 的模型，没有命令行参数可以改。
我们只有 DeepSeek 密钥，所以在运行前替换这个配置。不修改 τ²-bench 源码，
评测逻辑保持官方原样，只换了评分用的模型。

因为评分模型和模拟顾客都换成了 DeepSeek，分数不能和官方排行榜直接比较，
只能比较本项目不同版本之间的变化。

用法和 tau2 run 相同，例如：
    python scripts/run_tau2.py run --domain retail --task-ids 0 1 2 ...
"""

from __future__ import annotations

import os
import sys

JUDGE_LLM = os.environ.get("TAU2_JUDGE_LLM", "deepseek/deepseek-flash")
# 官方默认参数是 reasoning_effort=low，DeepSeek 不一定支持，所以换成固定温度
JUDGE_ARGS = {"temperature": 0.0}

from tau2 import config  # noqa: E402
from tau2.evaluator import evaluator_nl_assertions  # noqa: E402

# 评分模块在导入时已经拷贝了这两个常量，所以两处都要替换。
# τ²-bench 用线程并发，替换对所有任务都生效。
for module in (config, evaluator_nl_assertions):
    module.DEFAULT_LLM_NL_ASSERTIONS = JUDGE_LLM
    module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = JUDGE_ARGS

# τ²-bench 2.0 的 --domain telecom-workflow 跑不起来：题目里写的领域是 telecom，
# 而 2.0 新增的语言校验要求两者一致（multilingual/runtime.py 的
# "task scenario domain disagrees with language binding"）。
# 两种设置的唯一区别是交给 Agent 的资料：店规 + 文字手册，或店规 + 分步流程图。
# 所以仍用 --domain telecom，另外注册一个拿流程图资料的官方简单 Agent。
# 资料文本由 τ²-bench 自己的 get_environment(policy_type="workflow") 生成，
# 和 telecom-workflow 设置完全相同。
from tau2.agent.llm_agent import create_llm_agent  # noqa: E402
from tau2.domains.telecom.environment import get_environment  # noqa: E402
from tau2.registry import registry  # noqa: E402

_WORKFLOW_POLICY: str | None = None


def telecom_workflow_policy() -> str:
    global _WORKFLOW_POLICY
    if _WORKFLOW_POLICY is None:
        _WORKFLOW_POLICY = get_environment(policy_type="workflow").get_policy()
    return _WORKFLOW_POLICY


def create_llm_agent_workflow(tools, domain_policy, **kwargs):
    """官方 llm_agent，只把资料换成流程图版本。忽略传入的文字手册。"""
    return create_llm_agent(tools, telecom_workflow_policy(), **kwargs)


registry.register_agent_factory(create_llm_agent_workflow, "llm_agent_workflow")

# 本项目的 Agent。电信领域使用和基线相同的流程图资料，保证比较公平。
# 注册三个版本做对照实验：
#   sop_skeleton_workflow：只有 LangGraph 框架，行为同基线
#   sop_noguard_workflow：加上流程引擎（提取事实、账本、下一步提示），不加守卫
#   sop_agent_workflow：完整版，流程引擎 + 守卫
from sop_agent.agent import create_sop_agent  # noqa: E402


def _sop_factory(**options):
    def factory(tools, domain_policy, **kwargs):
        return create_sop_agent(tools, telecom_workflow_policy(), **options, **kwargs)

    return factory


registry.register_agent_factory(_sop_factory(), "sop_skeleton_workflow")
registry.register_agent_factory(_sop_factory(flow="telecom", use_guard=False), "sop_noguard_workflow")
registry.register_agent_factory(_sop_factory(flow="telecom", use_guard=True), "sop_agent_workflow")


# 零售领域：没有流程图版本，基线（τ²-bench 自带的 llm_agent）和本项目都使用 τ²-bench 给的零售店规。
def create_sop_agent_retail(tools, domain_policy, **kwargs):
    return create_sop_agent(tools, domain_policy, flow="retail", use_guard=True, **kwargs)


registry.register_agent_factory(create_sop_agent_retail, "sop_agent_retail")

from tau2.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
