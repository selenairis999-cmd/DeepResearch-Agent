"""LLM 统一接入层：普通对话 + JSON 结构化输出（带重试）；后端可切 Ollama / 千问 API"""
import json
import re
import time

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

import config

client = OpenAI(base_url=config.BACKENDS[config.LLM_BACKEND]["base_url"],
                api_key=config.BACKENDS[config.LLM_BACKEND]["api_key"])
# 快模型走后端可与强模型不同（如 Researcher 摘要走本地 Ollama 规避 API 限流）
client_fast = OpenAI(base_url=config.BACKENDS[config.FAST_BACKEND]["base_url"],
                     api_key=config.BACKENDS[config.FAST_BACKEND]["api_key"])

# 限流/网络类错误可重试；400 类参数错误不重试直接抛
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


def _strip_think(text: str) -> str:
    """qwen3 默认带 <think> 思考段，统一剥掉"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def chat(messages: list[dict], temperature: float = 0.3, model: str | None = None) -> str:
    """带指数退避重试：并行 Researcher 容易撞 API 限流（429）"""
    model = model or config.MODEL_STRONG
    cli = client_fast if model == config.MODEL_FAST else client
    last_err = None
    for attempt in range(5):
        try:
            resp = cli.chat.completions.create(model=model, messages=messages, temperature=temperature)
            return _strip_think(resp.choices[0].message.content or "")
        except RETRYABLE_ERRORS as e:
            last_err = e
            time.sleep(min(2 * (2 ** attempt), 30))  # 2s, 4s, 8s, 16s, 30s
    raise RuntimeError(f"LLM 调用重试 5 次仍失败: {last_err}")


def chat_json(messages: list[dict], retries: int = 2, temperature: float = 0.2, model: str | None = None) -> dict:
    """要求模型返回 JSON；8B 模型可能夹带多余文本，做正则提取 + 失败重试"""
    last_err = None
    for _ in range(retries + 1):
        raw = chat(messages, temperature=temperature, model=model)
        try:
            match = re.search(r"\{.*\}", raw, flags=re.S)
            if match:
                return json.loads(match.group(0))
            last_err = ValueError(f"输出中未找到 JSON: {raw[:100]}")
        except json.JSONDecodeError as e:
            last_err = e
        messages = messages + [
            {"role": "user", "content": "上次输出不是合法 JSON，请只输出一个 JSON 对象，不要任何其他文字。"}
        ]
    raise ValueError(f"JSON 解析失败: {last_err}")
