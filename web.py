"""Web 演示入口：FastAPI + SSE 实时推送各 Agent 执行状态到前端"""
import json
import queue
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from graph import build_graph

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="DeepResearch Agent")
graph = build_graph()
_run_lock = threading.Lock()  # 本地推理资源有限，同一时间只允许一个研究任务


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


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/research")
def research(topic: str = Query(..., min_length=1)):
    """SSE 流：运行研究流水线，逐节点推送状态"""
    if not _run_lock.acquire(blocking=False):
        return JSONResponse({"detail": "已有研究任务在运行中，请等待完成"}, status_code=409)

    q: queue.Queue = queue.Queue()

    def worker():
        t0 = time.time()
        report, trace = "", []
        try:
            for chunk in graph.stream({"topic": topic, "loop_count": 0}, stream_mode="updates"):
                for node, update in chunk.items():
                    q.put(_node_event(node, update))
                    if node == "editor":
                        report = update.get("report", report)
                    trace.extend(update.get("trace", []))
            # 与 CLI 一致：报告 + trace 落盘
            trace_dir = BASE_DIR / "traces"
            trace_dir.mkdir(exist_ok=True)
            ts = int(t0)
            (trace_dir / f"report_{ts}.md").write_text(report, encoding="utf-8")
            (trace_dir / f"trace_{ts}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            q.put({
                "type": "done", "report": report,
                "elapsed": round(time.time() - t0),
                "trace_file": f"trace_{ts}.json",
            })
        except Exception as e:  # noqa: BLE001 - 边界处统一兜底，错误推给前端展示
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)
            _run_lock.release()

    threading.Thread(target=worker, daemon=True).start()

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
