"""Tavily 搜索：专为 LLM 场景设计，返回干净的摘要结果"""
import httpx

import config

TAVILY_URL = "https://api.tavily.com/search"


def search(query: str, max_results: int | None = None) -> list[dict]:
    """返回 [{title, url, content}]；未配置 key 时返回空列表并提示"""
    if not config.TAVILY_API_KEY:
        print("[search] 未配置 TAVILY_API_KEY，跳过检索")
        return []
    resp = httpx.post(
        TAVILY_URL,
        json={
            "api_key": config.TAVILY_API_KEY,
            "query": query,
            "max_results": max_results or config.SEARCH_RESULTS_PER_QUERY,
            "search_depth": "basic",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in resp.json().get("results", [])
    ]
