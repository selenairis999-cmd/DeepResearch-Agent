"""全局配置：从 .env 读取，集中管理可调参数"""
import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# 双后端混合调度：强模型走质量（默认本地），快模型走吞吐；可分别指定 ollama / dashscope
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")          # 强模型后端（Planner/Critic/Writer/Editor）
FAST_BACKEND = os.getenv("FAST_BACKEND", LLM_BACKEND)     # 快模型后端（Researcher 高频并发摘要），默认同强模型
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BACKENDS = {
    "ollama": {"base_url": OLLAMA_BASE_URL, "api_key": "ollama"},
    "dashscope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": DASHSCOPE_API_KEY},
}
MODEL_STRONG = os.getenv("MODEL_STRONG", "qwen-plus" if LLM_BACKEND == "dashscope" else "qwen3:8b")
MODEL_FAST = os.getenv("MODEL_FAST", "qwen-turbo" if FAST_BACKEND == "dashscope" else "qwen3.5:4b")
# 按角色细分：长文生成环节可单独换更快的模型（推理型模型成稿太慢），默认跟随强模型
MODEL_WRITER = os.getenv("MODEL_WRITER", MODEL_STRONG)   # Writer 初稿
MODEL_EDITOR = os.getenv("MODEL_EDITOR", MODEL_STRONG)   # Editor 审校
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

MAX_LOOPS = int(os.getenv("MAX_LOOPS", "3"))            # Critic 打回的最大循环次数，防死循环
CRITIC_PASS_SCORE = int(os.getenv("CRITIC_PASS_SCORE", "6"))  # 证据质量及格线（0-10）；调低避免第一轮刚好卡在及格线下
MAX_SUB_QUESTIONS = 5                                    # Planner 拆题上限
SEARCH_RESULTS_PER_QUERY = 3                             # 每个子问题的搜索结果数（演示速度优先）
SCRAPE_MAX_CHARS = 3000                                  # 每篇网页正文截断长度
