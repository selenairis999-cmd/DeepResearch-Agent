"""持久化：进程重启续跑 / 检查点历史 / 分叉（真 SqliteSaver 落盘）"""
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from graph import build_graph


def _saver(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def test_resume_after_process_restart(fake_llm, tmp_path):
    """挂起后模拟进程重启：新 saver、新 graph 实例，仍能从检查点续跑"""
    cfg = {"configurable": {"thread_id": "t-restart"}}
    g1 = build_graph(_saver(tmp_path / "ck.sqlite"))
    r1 = g1.invoke({"topic": "话题", "loop_count": 0}, cfg)
    assert "__interrupt__" in r1

    g2 = build_graph(_saver(tmp_path / "ck.sqlite"))  # 全新实例 = 模拟重启
    state = g2.get_state(cfg)
    assert state.values.get("topic") == "话题"
    assert "human_gate" in state.next  # 挂在等人工确认

    final = g2.invoke(Command(resume="force_write"), cfg)
    assert final["report"]
    assert fake_llm["critic"] == 3  # 重启后也没有重跑打分


def test_state_history_and_fork(fake_llm, tmp_path):
    """--history 列检查点；--fork 从历史检查点分叉续跑（回到过去改写走向）"""
    cfg = {"configurable": {"thread_id": "t-fork"}}
    g = build_graph(_saver(tmp_path / "ck2.sqlite"))
    g.invoke({"topic": "话题", "loop_count": 0}, cfg)
    g.invoke(Command(resume="force_write"), cfg)

    snaps = list(g.get_state_history(cfg))
    assert len(snaps) >= 5  # 每个超步一个检查点
    latest = snaps[0]
    assert not latest.next  # 已到终态
    assert latest.values.get("report")

    # 从较早的检查点（interrupt 挂起之前）分叉续跑：重放会再次撞上人工确认
    # ——跟 CLI --fork 的 REPL 语义一致：撞上就答一次
    old = snaps[min(4, len(snaps) - 1)]
    cid = old.config["configurable"]["checkpoint_id"]
    cfg_fork = {"configurable": {"thread_id": "t-fork", "checkpoint_id": cid}}
    snap = g.get_state(cfg_fork)
    assert snap.values  # 能定位到历史状态
    final = g.invoke(None, cfg_fork)  # payload=None：从该检查点继续
    if "__interrupt__" in final:  # 分叉重放必然回到挂起点
        # resume 必须摘掉 checkpoint_id 钉子，否则从旧检查点反复重放、永远撞同一 interrupt
        final = g.invoke(Command(resume="force_write"),
                         {"configurable": {"thread_id": "t-fork"}})
    assert final["report"]
    # 分叉后主线的最新状态被新分支覆盖，但仍是完整结果
    assert g.get_state(cfg).values.get("report")


def test_send_fanout_checkpointed(fake_llm, tmp_path):
    """Send 并行任务也在检查点覆盖内（重启后不丢并行进度）"""
    cfg = {"configurable": {"thread_id": "t-send"}}
    g1 = build_graph(_saver(tmp_path / "ck3.sqlite"))
    g1.invoke({"topic": "话题", "loop_count": 0}, cfg)
    g2 = build_graph(_saver(tmp_path / "ck3.sqlite"))
    state = g2.get_state(cfg)
    # 两路 Send 的证据都已聚合进 State（evidence reducer 跨轮累积：2 子问题 × 3 轮）
    assert len(state.values.get("evidence", [])) == 6
