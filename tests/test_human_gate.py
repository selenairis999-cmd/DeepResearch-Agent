"""human_gate 人在回路：interrupt 挂起 / 两分支 / 硬顶防死循环 / 防重入"""
import pytest
from langgraph.types import Command

from graph import build_graph


def _cfg(tid):
    return {"configurable": {"thread_id": tid}}


def test_interrupt_hangs_at_human_gate(fake_llm):
    g = build_graph(":memory:")
    result = g.invoke({"topic": "话题", "loop_count": 0}, _cfg("t-hitl"))
    assert "__interrupt__" in result
    intr = result["__interrupt__"][0].value
    assert intr["options"] == ["force_write", "retry_once"]
    assert "已检索 3 轮" in intr["question"]
    # 3 轮全打回：planner 3 次、critic 3 次，尚未成稿
    assert fake_llm["critic"] == 3
    assert fake_llm["writer"] == 0


def test_resume_force_write_no_rescore(fake_llm):
    """人工放行 → 用最佳快照出稿；Critic 不重跑（拆独立节点防重入）"""
    g = build_graph(":memory:")
    cfg = _cfg("t-force")
    g.invoke({"topic": "话题", "loop_count": 0}, cfg)
    final = g.invoke(Command(resume="force_write"), cfg)
    assert fake_llm["critic"] == 3, "resume 后 Critic LLM 打分不得重跑"
    assert final["report"]
    assert final["critique"]["forced"] is True
    assert final["critique"]["passed"] is True
    assert final["critique"].get("needs_human") is None
    # 轨迹里能看到 human_gate 的决策
    nodes = [t["node"] for t in final["trace"]]
    assert "human_gate" in nodes


def test_resume_retry_then_hard_cap(fake_llm):
    """人工选再打回一轮 → 第 4 轮仍不达标 → 硬性上限自动放行，不再问人"""
    g = build_graph(":memory:")
    cfg = _cfg("t-retry")
    g.invoke({"topic": "话题", "loop_count": 0}, cfg)
    result = g.invoke(Command(resume="retry_once"), cfg)
    assert "__interrupt__" not in result, "MAX_LOOPS+1 后不再挂起"
    assert fake_llm["critic"] == 4
    assert result["critique"]["forced"] is True
    assert "硬性上限" in result["critique"]["comment"]
    assert result["report"]
    assert fake_llm["writer"] == 1


def test_pass_first_round_no_interrupt(fake_llm_pass):
    """一轮达标路径不经过 human_gate（回归：原行为不变）"""
    g = build_graph(":memory:")
    result = g.invoke({"topic": "话题", "loop_count": 0}, _cfg("t-pass"))
    assert "__interrupt__" not in result
    assert result["report"]
    nodes = [t["node"] for t in result["trace"]]
    assert "human_gate" not in nodes


def test_human_gate_off_falls_back_to_autoforce(monkeypatch, fake_llm):
    """HUMAN_IN_LOOP=0 → 到上限直接自动放行（旧行为），不 interrupt"""
    import config
    monkeypatch.setattr(config, "HUMAN_IN_LOOP", False)
    g = build_graph(":memory:")
    result = g.invoke({"topic": "话题", "loop_count": 0}, _cfg("t-off"))
    assert "__interrupt__" not in result
    assert result["critique"]["forced"] is True
    assert result["report"]


@pytest.mark.parametrize("bad", ["yes", "", "FORCE", "随便什么"])
def test_invalid_verdict_defaults_to_force(fake_llm, bad):
    """非法定向应答按保守分支处理：放行出稿"""
    g = build_graph(":memory:")
    cfg = _cfg("t-bad")
    g.invoke({"topic": "话题", "loop_count": 0}, cfg)
    final = g.invoke(Command(resume=bad), cfg)
    assert final["report"]
    assert final["critique"]["human_verdict"] == "force_write"
