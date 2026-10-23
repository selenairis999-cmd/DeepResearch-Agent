"""Web 演示入口：FastAPI + SSE 实时推送各 Agent 执行状态到前端

持久化与人在回路：
- 每次研究任务生成 thread_id（SqliteSaver 自动落检查点），SSE 首帧下发；
- critic 到 MAX_LOOPS 轮仍不达标 → human_gate 节点 interrupt 挂起 → SSE 推送
  {"type": "interrupt", ...}，本流结束（进程崩溃也不丢进度，thread_id 在）；
- 前端弹窗让人选 force_write / retry_once，再开一条 SSE 到
  /api/research/resume?thread_id=..&verdict=..，服务端 Command(resume) 续跑。
"""
import json
import queue
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langgraph.types import Command

import persistence
from graph import build_graph

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="DeepResearch Agent")
graph = build_graph()
_run_lock = threading.Lock()  # SqliteSaver 单连接 + 本地推理资源有限，同一时间只允许一个研究任务


def _node_event(node: str, update: dict) -> dict:
    """把 LangGraph 的节点更新提炼成前端友好的事件"""
    evt = {"type": "node", "node": node, "ts": round(time.time(), 1)}
    if node == "planner":
        evt["sub_questions"] = update.get("sub_questions", [])
        evt["loop"] = update.get("loop_count", 0)
    elif node == "researcher":
        tr = (update.get("trace") or [{}])[-1]
        evt["question"] = tr.get("question", "")
        evt["hits"] = tr.get("hits", 0)
    elif node == "critic":
        c = update.get("critique", {})
        evt.update(
            score=c.get("score"), passed=c.get("passed"),
            comment=c.get("comment", ""), loop=update.get("loop_count", 0),
        )
    elif node == "writer":
        evt["draft_len"] = len(update.get("draft", ""))
    elif node == "editor":
        evt["report_len"] = len(update.get("report", ""))
    return evt


def _new_thread_id(t0: float) -> str:
    return time.strftime(f"research-%Y%m%d-%H%M%S-") + str(int(t0 * 1000) % 1000)


def _pump(payload, cfg: dict, q: queue.Queue) -> str:
    """共用泵：把 graph.stream 的更新推给 SSE 队列；返回 "done" | "waiting" """
    t0 = time.time()
    thread_id = cfg["configurable"]["thread_id"]
    interrupted = False
    for chunk in graph.stream(payload, cfg, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                intr = update[0].value
                q.put({"type": "interrupt", "thread_id": thread_id, **intr})
                interrupted = True
                continue
            q.put(_node_event(node, update))
    if interrupted:
        # 进度都在检查点里，不落盘报告；等 /resume 续跑
        q.put({"type": "waiting", "thread_id": thread_id, "elapsed": round(time.time() - t0)})
        return "waiting"
    # 完整结果从检查点终态取（resume 续跑的场景流里只有后半段，终态才是全集）
    values = graph.get_state(cfg).values or {}
    report = values.get("report", "")
    trace = values.get("trace", [])
    ts = int(t0)
    trace_file = persistence.save_artifacts(ts, report, trace)
    q.put({
        "type": "done", "thread_id": thread_id, "report": report,
        "elapsed": round(time.time() - t0), "trace_file": trace_file.name,
    })
    return "done"


def _sse(q: queue.Queue) -> StreamingResponse:
    def gen():
        yield "retry: 3000\n\n"
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/research")
def research(topic: str = Query(..., min_length=1)):
    """SSE 流：运行研究流水线，逐节点推送状态；遇人工确认挂起则推 interrupt 帧后结束"""
    if not _run_lock.acquire(blocking=False):
        return JSONResponse({"detail": "已有研究任务在运行中，请等待完成"}, status_code=409)

    t0 = time.time()
    thread_id = _new_thread_id(t0)
    persistence.record_run(thread_id, topic)
    cfg = {"configurable": {"thread_id": thread_id}}
    q: queue.Queue = queue.Queue()
    q.put({"type": "start", "thread_id": thread_id, "topic": topic})

    def worker():
        try:
            _pump({"topic": topic, "loop_count": 0}, cfg, q)
        except Exception as e:  # noqa: BLE001 - 边界处统一兜底，错误推给前端展示
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)
            _run_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return _sse(q)


@app.get("/api/research/resume")
def resume(thread_id: str = Query(...), verdict: str = Query("force_write")):
    """人工确认后续跑：Command(resume) 从检查点恢复（进程重启后也有效）"""
    if verdict not in ("force_write", "retry_once"):
        return JSONResponse({"detail": f"非法 verdict: {verdict}"}, status_code=422)
    if not _run_lock.acquire(blocking=False):
        return JSONResponse({"detail": "已有研究任务在运行中，请等待完成"}, status_code=409)

    cfg = {"configurable": {"thread_id": thread_id}}
    snap = graph.get_state(cfg)
    if not snap or not snap.values:
        _run_lock.release()
        return JSONResponse({"detail": f"找不到任务 {thread_id} 的检查点"}, status_code=404)
    if "__interrupt__" not in snap.next and "human_gate" not in (snap.next or []):
        _run_lock.release()
        return JSONResponse({"detail": f"任务 {thread_id} 不在等待人工确认状态（当前：{','.join(snap.next) or '已结束'}）"}, status_code=409)

    q: queue.Queue = queue.Queue()

    def worker():
        try:
            _pump(Command(resume=verdict), cfg, q)
        except Exception as e:  # noqa: BLE001
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)
            _run_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return _sse(q)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
