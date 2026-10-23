"""Tavily 搜索：专为 LLM 场景设计，返回干净的摘要结果；带网络重试兜底"""
import time

import httpx

import config

TAVILY_URL = "https://api.tavily.com/search"


def search(query: str, max_results: int | None = None) -> list[dict]:
    """返回 [{title, url, content}]；未配置 key 或重试穷尽时返回空列表（不抛异常炸整跑）"""
    if not config.TAVILY_API_KEY:
        print("[search] 未配置 TAVILY_API_KEY，跳过检索")
        return []
    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query,
        "max_results": max_results or config.SEARCH_RESULTS_PER_QUERY,
        "search_depth": "basic",
    }
    last_err: Exception | None = None
    for attempt in range(max(1, config.SEARCH_RETRIES)):
        try:
            resp = httpx.post(TAVILY_URL, json=payload, timeout=30)
            resp.raise_for_status()
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
                for r in resp.json().get("results", [])
            ]
        except httpx.HTTPError as e:
            last_err = e
            if attempt < config.SEARCH_RETRIES - 1:
                time.sleep(2 ** attempt)  # 1s, 2s 退避；搜索在关键路径上，不能太久
    print(f"[search] 重试 {config.SEARCH_RETRIES} 次仍失败，本子问题返回空结果：{last_err}")
    return []
