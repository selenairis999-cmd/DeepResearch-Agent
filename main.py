"""CLI 入口：交互式研究 + 断点续跑 / 检查点历史 / 分叉

用法：
  python main.py                                   # 新任务（遇人工确认节点会停下来问）
  python main.py --list                            # 列出历史研究任务（thread_id + 话题）
  python main.py --resume <thread_id>              # 从最新断点继续（含挂起中的人工确认）
  python main.py --history <thread_id>             # 查看该任务的检查点序列（新→旧）
  python main.py --fork <thread_id> <checkpoint_id># 从任意历史检查点分叉续跑（跑出新分支）

每个任务一个 thread_id；节点每完成一步由 SqliteSaver 落检查点，
进程崩溃 / Ctrl+C 后 --resume 可接着跑，--fork 可回到过去任何一步改写走向。
"""
import argparse
import time

from langgraph.types import Command

import persistence
from graph import build_graph

def _new_thread_id(t0: float) -> str:
    return time.strftime(f"research-%Y%m%d-%H%M%S-") + str(int(t0 * 1000) % 1000)


def _repl(graph, payload, cfg) -> dict:
    """跑图；遇到 human_gate 的 interrupt 进命令行问答，Command(resume) 续跑直到出结果

    注意：--fork 传入的 cfg 里钉了 checkpoint_id（定位历史检查点用）；
    首次 invoke 之后必须摘掉，否则 Command(resume) 会带着旧检查点反复重放、
    永远撞同一个 interrupt（测试 test_state_history_and_fork 抓出来的）。
    """
    cfg = {"configurable": dict(cfg["configurable"])}  # 不改调用方的 dict
    result = graph.invoke(payload, cfg)
    cfg["configurable"].pop("checkpoint_id", None)  # resume 回到线程最新状态
    while result.get("__interrupt__"):
        intr = result["__interrupt__"][0].value
        print("\n" + "=" * 60)
        print("[人工确认] " + intr["question"])
        print("  1) force_write —— 用历史最高分轮证据直接出稿")
        print("  2) retry_once  —— 再打回一轮补充检索（下轮仍不达标将自动放行）")
        choice = input("选择 (1/2，回车=1)：").strip()
        verdict = "retry_once" if choice in ("2", "r", "retry", "retry_once") else "force_write"
        print(f"已选择 {verdict}，继续执行…")
        result = graph.invoke(Command(resume=verdict), cfg)
    return result


def _save_and_print(final: dict, thread_id: str, t0: float) -> None:
    report = final.get("report", "")
    print("\n" + "=" * 60 + "\n最终报告\n" + "=" * 60 + f"\n{report}")
    trace_file = persistence.save_artifacts(int(t0), report, final.get("trace", []))
    print(f"\n耗时 {time.time() - t0:.0f}s · thread_id={thread_id}\ntrace 已保存到 {trace_file}")


def cmd_run() -> None:
    topic = input("研究话题：").strip()
    if not topic:
        return
    thread_id = _new_thread_id(time.time())
    persistence.record_run(thread_id, topic)
    graph = build_graph()
    cfg = {"configurable": {"thread_id": thread_id}}
    t0 = time.time()
    final = _repl(graph, {"topic": topic, "loop_count": 0}, cfg)
    _save_and_print(final, thread_id, t0)


def cmd_resume(thread_id: str) -> None:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": thread_id}}
    snap = graph.get_state(cfg)
    if not snap or not snap.values:
        print(f"找不到任务 {thread_id} 的状态（先 --list 看有哪些）")
        return
    values = snap.values
    print(f"任务话题：{values.get('topic', '?')} · 已检索 {values.get('loop_count', 0)} 轮 · 历史最高 {values.get('best_score', 0)} 分")
    if snap.next:
        print(f"挂起中，下一步：{','.join(snap.next)}")
    t0 = time.time()
    final = _repl(graph, None, cfg)  # payload=None：从现有状态继续
    _save_and_print(final, thread_id, t0)


def cmd_history(thread_id: str) -> None:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": thread_id}}
    snaps = list(graph.get_state_history(cfg))
    if not snaps:
        print(f"任务 {thread_id} 无检查点记录")
        return
    print(f"共 {len(snaps)} 个检查点（新→旧）。--fork 续跑某一步时用第 2 列的 checkpoint_id：\n")
    for i, s in enumerate(snaps):
        v = s.values or {}
        cid = s.config["configurable"].get("checkpoint_id", "?")
        nxt = ",".join(s.next) or "(结束)"
        print(f"  #{i:<3} {cid[:16]:<18} step={s.metadata.get('step', '?'):<4} "
              f"loop={v.get('loop_count', 0)} best={v.get('best_score', 0)}  next={nxt}")
    print(f"\n示例：python main.py --fork {thread_id} {snaps[1].config['configurable']['checkpoint_id']}")


def cmd_fork(thread_id: str, checkpoint_id: str) -> None:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    snap = graph.get_state(cfg)
    if not snap or not snap.values:
        print(f"找不到检查点 {checkpoint_id}（先 --history {thread_id} 看列表）")
        return
    print(f"从检查点分叉：话题 {snap.values.get('topic', '?')}，loop={snap.values.get('loop_count', 0)}")
    t0 = time.time()
    final = _repl(graph, None, cfg)
    _save_and_print(final, thread_id, t0)


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepResearch Agent CLI")
    ap.add_argument("--list", action="store_true", help="列出历史研究任务")
    ap.add_argument("--resume", metavar="THREAD_ID", help="从最新断点继续指定任务")
    ap.add_argument("--history", metavar="THREAD_ID", help="查看任务的检查点历史")
    ap.add_argument("--fork", nargs=2, metavar=("THREAD_ID", "CHECKPOINT_ID"), help="从指定检查点分叉续跑")
    args = ap.parse_args()

    if args.list:
        runs = persistence.list_runs()
        if not runs:
            print("暂无历史任务记录")
            return
        for r in runs:
            print(f"  {r['thread_id']}  {r['topic']}  ({r['started']})")
    elif args.resume:
        cmd_resume(args.resume)
    elif args.history:
        cmd_history(args.history)
    elif args.fork:
        cmd_fork(args.fork[0], args.fork[1])
    else:
        cmd_run()


if __name__ == "__main__":
    main()
