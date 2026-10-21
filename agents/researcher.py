"""Researcher：处理单个子问题，搜索 → 抓取 → 提炼证据卡片（可溯源）"""
from concurrent.futures import ThreadPoolExecutor

import config
from state import ResearchState
from tools.llm import chat
from tools.scrape import scrape
from tools.search import search

SUMMARY_PROMPT = """你是研究助理。根据以下网页内容，提炼与子问题相关的关键事实（3-5 条，保留具体数据和细节）。
子问题：{question}

网页内容：
{content}"""


def _process(item: dict, question: str) -> dict | None:
    content = item["content"] or scrape(item["url"])
    if not content:
        return None
    # 摘要提炼是高频简单任务，走快模型
    summary = chat(
        [{"role": "system", "content": SUMMARY_PROMPT.format(question=question, content=content)}],
        model=config.MODEL_FAST,
    )
    return {"question": question, "title": item["title"], "url": item["url"], "summary": summary}


def researcher(state: ResearchState) -> dict:
    """由 Send API 派发，每次只处理一个子问题；结果经 reducer 聚合到主状态"""
    question = state["current_question"]
    # 线程池让网页抓取（网络 IO）与 LLM 摘要（本地推理）相互重叠
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda it: _process(it, question), search(question)))
    evidence = [r for r in results if r]
    return {
        "evidence": evidence,
        "trace": [{"node": "researcher", "question": question, "hits": len(evidence)}],
    }
