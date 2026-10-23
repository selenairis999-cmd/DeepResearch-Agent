"""Human Gate：MAX_LOOPS 轮仍不达标时的人工确认节点（interrupt 挂起 / Command(resume) 恢复）

为什么拆成独立节点（而不是在 Critic 里直接 interrupt）：
interrupt 恢复后整个节点会【从头重跑】——若打分和问人混在同一个节点，
恢复时 Critic 的 LLM 打分会再烧一遍、分数还可能变。
这是 langgraph-lab 第 6 课踩过的重入坑：有副作用的逻辑要么放 interrupt 前一个节点（已落检查点），
要么放后一个节点。Critic 节点在 interrupt 前完成并落检查点，恢复时直接从本节点续跑。

挂起语义：graph.invoke() 返回 {"__interrupt__": (Interrupt(value=...),)}，value 即下方 payload；
恢复语义：graph.invoke(Command(resume="force_write"/"retry_once"), 同 thread_id 配置)。
"""
from langgraph.types import interrupt

import config
from state import ResearchState


def _force_pass(critique: dict, state: ResearchState) -> dict:
    """放行出稿：与历史行为一致，历史最高分轮更好则回滚到该轮证据"""
    critique["passed"] = True
    critique["forced"] = True
    best_score = state.get("best_score", -1)
    if best_score > critique.get("score", 0):
        critique["rollback"] = True  # final_evidence 中已是最高分轮快照，Writer/Editor 优先使用
        critique["comment"] = f"人工确认放行，回滚至第 {state.get('best_loop')} 轮（历史最高 {best_score} 分）"
    else:
        critique["comment"] = "人工确认放行"
    return critique


def human_gate(state: ResearchState) -> dict:
    critique = dict(state["critique"])
    loop = state.get("loop_count", 0)
    verdict = interrupt({
        "question": (
            f"「{state.get('topic', '')}」已检索 {loop} 轮，Critic 最新评分 "
            f"{critique.get('score')}/10 仍低于及格线 {config.CRITIC_PASS_SCORE}"
            f"（历史最高 {state.get('best_score', 0)} 分在第 {state.get('best_loop')} 轮）。"
        ),
        "options": ["force_write", "retry_once"],
        "hint": "force_write＝用历史最高分轮证据直接出稿；retry_once＝再打回一轮（下轮仍不达标将自动放行）",
    })
    # 非法应答一律按保守分支处理：放行出稿
    if verdict not in ("force_write", "retry_once"):
        verdict = "force_write"

    if verdict == "force_write":
        critique = _force_pass(critique, state)
    else:
        critique["comment"] = "人工确认：再打回一轮补充检索"
    critique["human_verdict"] = verdict
    critique.pop("needs_human", None)  # 决策已做出，路由只看 passed

    return {
        "critique": critique,
        "trace": [{
            "node": "human_gate", "loop": loop, "verdict": verdict,
            "score": critique.get("score"),
        }],
    }
