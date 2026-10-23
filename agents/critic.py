"""Critic：评估证据覆盖度与可信度，决定放行或打回（质量反馈回路的核心）"""
import json
import re

import config
from state import ResearchState
from tools.llm import chat

PROMPT = """你是严格的研究评审。评估以下证据是否足以支撑一份关于「{topic}」的研究报告。
评估维度：子问题覆盖度、来源可信度、信息是否互相印证。
【评估标准自适应】先判断话题类型：
- 事实调研类（技术、行业、案例等有现成资料的）：按证据充分性严格评分；
- 前瞻/观点/预测类（如"某技术未来影响""某趋势展望"）：现实中本就没有量化数据或真实案例，观点多样性、专家论证、趋势分析均算有效覆盖，不要因"缺乏量化模型/真实案例"打低分。
【严格格式要求】只输出一个纯 JSON 对象，不要任何 markdown、不要解释、不要代码围栏、不要自然语言段落。
格式示例（请严格照抄字段名和类型，只替换值）：
{{"score": 7, "passed": true, "gaps": ["缺口1"], "comment": "一句话评语"}}
score 是 0~10 整数；score >= {pass_score} 才算 passed=true；不通过时 gaps 必须列出具体缺口。"""


def _truncate_evidence(evidence: list[dict], max_per_question: int = 5, max_summary_chars: int = 500) -> str:
    """按子问题分组截断，8B 模型长上下文会退化（零分、comment 缺失）"""
    from collections import OrderedDict
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for e in evidence:
        grouped.setdefault(e["question"], []).append(e)
    kept, lines = [], []
    for q, items in grouped.items():
        for e in items[:max_per_question]:
            e["summary"] = (e.get("summary") or "")[:max_summary_chars]
            kept.append(e)
    for i, e in enumerate(kept):
        lines.append(f"[{i + 1}] ({e['question']}) {e['title']}\n{e['summary']}")
    return "\n\n".join(lines) or "（无证据）", len(evidence), len(kept)


def _parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"JSON 未找到: {raw[:100]}")


def critic(state: ResearchState) -> dict:
    evidence_text, total_cnt, kept_cnt = _truncate_evidence(state.get("evidence", []))
    user_msg = f"子问题：{state['sub_questions']}\n\n证据：\n{evidence_text}"

    raw = None
    last_err = None
    for attempt in range(3):
        raw = chat([
            {"role": "system", "content": PROMPT.format(topic=state["topic"], pass_score=config.CRITIC_PASS_SCORE)},
            {"role": "user", "content": user_msg},
        ] + ([{"role": "user", "content": (
            "你的上一次输出不是合法 JSON。"
            "请只输出一个 JSON 对象，形如 {\"score\": 7, \"passed\": true, \"gaps\": [], \"comment\": \"...\"}，"
            "不要任何 markdown、解释、代码围栏或自然语言段落。"
        )}] if attempt > 0 else []))
        try:
            result = _parse_json(raw)
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
    else:
        result = {"score": 0, "passed": False, "gaps": [], "comment": f"JSON 解析失败: {last_err}"}

    result.setdefault("score", 0)
    result.setdefault("gaps", [])
    # passed 由分数硬决定，不采信模型自相矛盾的布尔输出（出现过 score=6 但 passed=false）
    result["passed"] = result["score"] >= config.CRITIC_PASS_SCORE

    loop_count = state.get("loop_count", 0) + 1
    # 最佳检查点：刷新历史最高分时，同步快照当前累积证据，供强制放行时回滚
    update: dict = {}
    if result["score"] > state.get("best_score", -1):
        update = {
            "best_score": result["score"],
            "best_loop": loop_count,
            "final_evidence": list(state.get("evidence", [])),
        }

    forced = False
    rollback = False
    if not result["passed"] and loop_count >= config.MAX_LOOPS:
        # 顶到打回上限：转人工确认（human_gate 节点 interrupt 问人，防重入已拆独立节点）；
        # 硬顶 MAX_LOOPS+1：人工确认后打回的一轮仍不达标 → 无条件放行，防死循环
        if loop_count >= config.MAX_LOOPS + 1 or not config.HUMAN_IN_LOOP:
            result["passed"] = True
            forced = True
            best_score = state.get("best_score", -1)
            if best_score > result["score"]:
                rollback = True  # final_evidence 中已是最高分轮快照，Writer/Editor 优先使用它
                result["comment"] = f"达到硬性上限（{config.MAX_LOOPS + 1} 轮），回滚至第 {state.get('best_loop')} 轮（历史最高 {best_score} 分）"
            else:
                result["comment"] = f"达到硬性上限（{config.MAX_LOOPS + 1} 轮），强制放行"
            result["forced"] = True
            result["rollback"] = rollback
        else:
            result["needs_human"] = True
            result["comment"] = (
                f"达到最大循环 {config.MAX_LOOPS} 轮仍 {result['score']} 分"
                f"（及格线 {config.CRITIC_PASS_SCORE}），转人工确认"
            )

    return {
        "critique": result,
        "loop_count": loop_count,
        **update,
        "trace": [{
            "node": "critic", "loop": loop_count, "score": result["score"],
            "passed": result["passed"], "forced": forced, "rollback": rollback,
            "needs_human": result.get("needs_human", False),
            "best_score": update.get("best_score", state.get("best_score")),
            "evidence_total": total_cnt, "evidence_kept": kept_cnt,
            "raw": raw, "error": str(last_err) if last_err and loop_count < 3 else None,
        }],
    }
