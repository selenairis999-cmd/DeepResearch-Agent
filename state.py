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
    forced: bool        # 到达硬性上限后的强制放行（人工确认或 MAX_LOOPS+1 兜底）
    rollback: bool      # 放行时回滚到历史最高分轮证据
    needs_human: bool   # 打回次数用尽，转人工确认（human_gate 节点）
    human_verdict: str  # 人工最终选择：force_write / retry_once


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


def unique_evidence(evidence: list[dict] | None) -> list[dict]:
    """按 URL 去重（保留首次出现），Writer 引用编号与 Editor 参考来源共用同一列表。

    多轮打回重检会让同一 URL 反复进入累积证据（历史 trace 里 45 条参考中同一 URL 最多重复 3 次），
    统一在成稿入口收敛，两处编号天然一致。
    """
    seen: set[str] = set()
    out: list[dict] = []
    for e in evidence or []:
        url = e.get("url") or ""
        if url:
            if url in seen:
                continue
            seen.add(url)
        out.append(e)
    return out
