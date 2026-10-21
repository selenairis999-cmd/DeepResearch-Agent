"""Planner：把研究话题拆成子问题；接收 Critic 缺口反馈做二次拆题"""
import config
from state import ResearchState
from tools.llm import chat_json

PROMPT = """你是研究规划师。把用户的研究话题拆成 {n} 个以内具体、可检索的子问题。
要求：
- 具体性优先：每个子问题应能直接拿到搜索引擎搜到结果，避免空泛概念（例如"某技术的具体架构组件""某领域2023-2025年的量化数据""某框架的真实落地案例"优于"某技术的概述""某领域的发展趋势"）；
- 覆盖面：覆盖话题的不同侧面，避免重叠；
- 按重要性排序。
只输出 JSON：{{"sub_questions": ["问题1", "问题2", ...]}}"""

GAP_PROMPT = """上一轮检索的证据存在以下缺口：
{gaps}
请针对缺口补充新的子问题（不要重复已有问题），合并后重新输出完整清单。"""


def planner(state: ResearchState) -> dict:
    prompt = PROMPT.format(n=config.MAX_SUB_QUESTIONS)
    critique = state.get("critique")
    if critique and critique.get("gaps"):
        prompt += "\n\n" + GAP_PROMPT.format(
            gaps="\n".join(f"- {g}" for g in critique["gaps"])
        )

    result = chat_json([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"研究话题：{state['topic']}"},
    ])
    sub_questions = result.get("sub_questions", [])[: config.MAX_SUB_QUESTIONS]
    return {
        "sub_questions": sub_questions,
        "trace": [{"node": "planner", "sub_questions": sub_questions}],
    }
