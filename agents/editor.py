"""Editor：事实一致性与格式审校，输出终稿"""
import re

import config
from state import ResearchState, unique_evidence
from tools.llm import chat

PROMPT = """你是报告编辑。按以下清单审校初稿并直接输出终稿（Markdown）：
0. 重要：你必须输出修改后的完整报告正文，禁止输出审稿意见、优化建议或修改清单；
1. 每个结论是否有对应引用支撑，删除无依据的断言；
2. 引用编号是否都能在证据列表中找到，对不上的一并修正；
3. 结构、语言、格式是否清晰规范。

证据列表：
{evidence}"""


def _strip_fence(text: str) -> str:
    """剥掉模型输出可能带的 ```markdown 代码围栏"""
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def editor(state: ResearchState) -> dict:
    # 与 Writer 一致：优先最佳检查点快照；同一个 unique_evidence 列表保证引用编号与参考来源对得上
    raw = state.get("final_evidence") or state.get("evidence", [])
    evidence = unique_evidence(raw)
    # 无证据时 Writer 已生成说明稿，直接作为终稿，不再过 LLM
    if not evidence:
        return {"report": state["draft"], "trace": [{"node": "editor", "skipped": "no_evidence"}]}

    evidence_text = "\n".join(
        f"[{i + 1}] {e['title']} {e['url']}"
        for i, e in enumerate(evidence)
    )
    report = _strip_fence(chat([
        {"role": "system", "content": PROMPT.format(evidence=evidence_text)},
        {"role": "user", "content": f"初稿：\n{state['draft']}"},
    ], temperature=0.3, model=config.MODEL_EDITOR))
    # 参考来源由代码从证据直接拼接，杜绝 LLM 编造/重复/缺 URL
    refs = "\n\n---\n\n## 参考来源\n" + "\n".join(
        f"{i + 1}. {e['title']} — {e['url']}"
        for i, e in enumerate(evidence)
    )
    report = report.rstrip() + refs
    return {"report": report, "trace": [{
        "node": "editor", "report_len": len(report),
        "evidence_used": len(evidence), "dupes_removed": len(raw) - len(evidence),
    }]}
