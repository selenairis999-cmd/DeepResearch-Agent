"""LangGraph 共享状态（黑板）：所有 Agent 不直接通信，围绕它读写"""
import operator
from typing import Annotated, TypedDict


class Evidence(TypedDict):
    question: str   # 对应子问题
    title: str
    url: str
    summary: str    # 证据摘要


class Critique(TypedDict, total=False):
    score: int      # 0-10
    passed: bool
    gaps: list[str]  # 缺口描述，打回时给 Planner 参考
    comment: str


class ResearchState(TypedDict, total=False):
    topic: str
    sub_questions: list[str]
    current_question: str                                      # Send API 派发给单个 Researcher 的子问题
    evidence: Annotated[list[Evidence], operator.add]          # 并行 Researcher 结果经 reducer 聚合
    final_evidence: list[Evidence]                             # 历史最高分轮的证据快照（最佳检查点回滚用）
    best_score: int                                            # Critic 历史最高分
    best_loop: int                                             # 最高分所在轮次
    critique: Critique
    draft: str
    report: str
    loop_count: int
    trace: Annotated[list[dict], operator.add]                 # 每个节点的关键输入输出，落盘做可观测
