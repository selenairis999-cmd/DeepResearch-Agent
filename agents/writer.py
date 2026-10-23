"""Writer：综合证据写结构化初稿，强制引用标注"""
import config
from state import ResearchState, unique_evidence
from tools.llm import chat

PROMPT = """你是研究报告撰稿人。根据证据卡片撰写关于「{topic}」的研究报告初稿（Markdown）。
重要：直接输出报告正文本身，不要输出审稿意见、优化建议或元评论。
要求：
1. 结构：摘要 → 各子问题分析 → 结论；
2. 每个关键论点后用 [编号] 标注引用，编号对应证据列表；
3. 证据不足的子问题要如实说明，不要编造。"""


def writer(state: ResearchState) -> dict:
    # 优先用最佳检查点快照（Critic 回滚时为历史最高分轮证据），并按 URL 去重
    raw = state.get("final_evidence") or state.get("evidence", [])
    evidence = unique_evidence(raw)
    # 硬校验：无证据时不走 LLM，直接生成说明（8B 模型在空证据下会编造内容）
    if not evidence:
        notice = (
            f"# {state['topic']}\n\n"
            "> 未检索到有效证据，无法生成可靠报告。请检查搜索 API 配置后重试。"
        )
        return {"draft": notice, "trace": [{"node": "writer", "skipped": "no_evidence"}]}

    evidence_text = "\n\n".join(
        f"[{i + 1}] {e['title']} ({e['url']})\n{e['summary']}"
        for i, e in enumerate(evidence)
    )
    draft = chat([
        {"role": "system", "content": PROMPT.format(topic=state["topic"])},
        {"role": "user", "content": f"证据列表：\n{evidence_text}"},
    ], temperature=0.5, model=config.MODEL_WRITER)
    return {"draft": draft, "trace": [{
        "node": "writer", "draft_len": len(draft),
        "evidence_used": len(evidence), "dupes_removed": len(raw) - len(evidence),
    }]}
