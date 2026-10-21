"""CLI 入口：先跑通命令行版，Web 前端后续再接"""
import json
import time
from pathlib import Path

from graph import build_graph


def main():
    topic = input("研究话题：").strip()
    if not topic:
        return
    graph = build_graph()
    t0 = time.time()
    final = graph.invoke({"topic": topic, "loop_count": 0})

    report = final.get("report", "")
    print("\n" + "=" * 60 + "\n最终报告\n" + "=" * 60 + f"\n{report}")

    trace_dir = Path("traces")
    trace_dir.mkdir(exist_ok=True)
    (trace_dir / f"report_{int(t0)}.md").write_text(report, encoding="utf-8")
    trace_file = trace_dir / f"trace_{int(t0)}.json"
    trace_file.write_text(json.dumps(final.get("trace", []), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n耗时 {time.time() - t0:.0f}s，trace 已保存到 {trace_file}")


if __name__ == "__main__":
    main()
