"""状态持久化：SqliteSaver 检查点 + 任务索引（runs.json）

每个研究任务一个 thread_id；graph.compile(checkpointer=saver) 后，
每个超步（节点完成/Send 任务完成）自动写检查点，进程崩溃或中断后可按 thread_id 续跑。
另维护一份轻量 runs.json 索引（thread_id → 话题/时间），供 CLI --list 直接读，
不依赖 checkpointer 数据库的内部表结构。
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import config

BASE_DIR = Path(__file__).parent
RUNS_INDEX = BASE_DIR / "runs.json"


def db_path() -> Path:
    p = Path(config.CHECKPOINT_DB)
    return p if p.is_absolute() else BASE_DIR / p


def make_saver(db: str | None = None):
    """创建检查点存储。db=":memory:" 时返回内存版（测试用，不落盘）。"""
    if db == ":memory:":
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver
    conn = sqlite3.connect(db_path(), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # 幂等：CREATE TABLE IF NOT EXISTS
    return saver


def record_run(thread_id: str, topic: str) -> None:
    """CLI / Web 启动任务时登记一条（只追加，供 --list 快速回看）"""
    runs = _load_runs()
    runs.append({
        "thread_id": thread_id,
        "topic": topic,
        "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    RUNS_INDEX.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")


def save_artifacts(trace_id: int, report: str, trace: list) -> Path:
    """成稿落盘：报告 MD + 全链路 trace JSON（CLI 与 Web 共用同一份实现）"""
    trace_dir = BASE_DIR / "traces"
    trace_dir.mkdir(exist_ok=True)
    (trace_dir / f"report_{trace_id}.md").write_text(report, encoding="utf-8")
    trace_file = trace_dir / f"trace_{trace_id}.json"
    trace_file.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return trace_file


def list_runs() -> list[dict]:
    """runs.json 索引；再对照检查点库里实际存在的 thread_id 去掉已失效条目"""
    db = db_path()
    live: set[str] = set()
    if db.exists():
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            live = {r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
            conn.close()
        except sqlite3.Error:
            live = set()  # 表结构变化等异常时降级为只读索引
    runs = _load_runs()
    return [r for r in runs if not live or r["thread_id"] in live]


def _load_runs() -> list[dict]:
    if not RUNS_INDEX.exists():
        return []
    try:
        return json.loads(RUNS_INDEX.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
