# 根目录 conftest：让 tests/ 能 import 项目模块；共享假 LLM 桩
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

import agents.critic, agents.editor, agents.planner, agents.researcher, agents.writer
import config
import tools.scrape


CRITIC_LOW = '{"score": 3, "passed": false, "gaps": ["缺量化数据"], "comment": "证据不足"}'
CRITIC_HIGH = '{"score": 8, "passed": true, "gaps": [], "comment": "充分"}'


@pytest.fixture
def fake_llm(monkeypatch):
    """全部 LLM / 检索 / 抓取换成桩；Critic 恒打低分 → 必触发 human_gate interrupt"""
    calls = {"critic": 0, "writer": 0, "editor": 0, "planner": 0, "researcher": 0}
    monkeypatch.setattr(agents.planner, "chat_json",
                        lambda m, **kw: (calls.__setitem__("planner", calls["planner"] + 1)
                                         or {"sub_questions": ["子问题A", "子问题B"]}))
    monkeypatch.setattr(agents.critic, "chat",
                        lambda m, **kw: (calls.__setitem__("critic", calls["critic"] + 1) or CRITIC_LOW))
    monkeypatch.setattr(agents.researcher, "search",
                        lambda q, **kw: [{"title": f"文档{q}", "url": f"http://e/{q}", "content": "正文"}])
    monkeypatch.setattr(tools.scrape, "scrape", lambda url: "抓取正文")
    monkeypatch.setattr(agents.researcher, "chat",
                        lambda m, **kw: (calls.__setitem__("researcher", calls["researcher"] + 1) or "摘要"))
    monkeypatch.setattr(agents.writer, "chat",
                        lambda m, **kw: (calls.__setitem__("writer", calls["writer"] + 1) or "# 初稿\n\n论点 [1]"))
    monkeypatch.setattr(agents.editor, "chat",
                        lambda m, **kw: (calls.__setitem__("editor", calls["editor"] + 1) or "# 终稿\n\n论点 [1]"))
    return calls


@pytest.fixture
def fake_llm_pass(monkeypatch, fake_llm):
    """Critic 打高分 → 一轮放行，不打断"""
    import agents.critic
    monkeypatch.setattr(agents.critic, "chat", lambda m, **kw: CRITIC_HIGH)
    return fake_llm
