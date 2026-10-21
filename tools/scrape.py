"""Jina Reader：把网页转成 LLM 友好的 Markdown，免费免 key"""
import httpx

import config


def scrape(url: str) -> str:
    """抓取网页正文并截断；失败返回空串"""
    try:
        resp = httpx.get(f"https://r.jina.ai/{url}", timeout=30)
        resp.raise_for_status()
        return resp.text[: config.SCRAPE_MAX_CHARS]
    except httpx.HTTPError:
        return ""
