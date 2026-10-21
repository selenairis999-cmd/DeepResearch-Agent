"""双模型对比 benchmark：在链路 3 类真实任务上测 qwen3:8b vs qwen3.5:4b"""
import json
import re
import time

from tools.llm import chat

MODELS = ["qwen3:8b", "qwen3.5:4b"]
TOPIC = "多智能体系统在大模型应用中的发展现状"

TASK_PLANNER = [
    {"role": "system", "content": '你是研究规划师。把研究话题拆成 5 个以内可检索的子问题。只输出 JSON：{"sub_questions": ["问题1", "问题2"]}'},
    {"role": "user", "content": f"研究话题：{TOPIC}"},
]
EVIDENCE = (
    "[1] LangGraph 官方文档：通过状态图编排多智能体，支持并行派发与循环边。\n"
    "[2] 某行业调研：2025 年多智能体框架在企业 PoC 中采用率约 35%，主要瓶颈是评估与可观测性。"
)
TASK_CRITIC = [
    {"role": "system", "content": f'你是研究评审。评估证据是否足以支撑「{TOPIC}」的报告。只输出 JSON：{{"score": 0到10的整数, "passed": true或false, "gaps": ["缺口"], "comment": "评语"}}'},
    {"role": "user", "content": f"证据：\n{EVIDENCE}"},
]
TASK_WRITER = [
    {"role": "system", "content": f"你是撰稿人。根据证据写 300 字左右关于「{TOPIC}」的分析段落，论点后用 [编号] 标注引用。"},
    {"role": "user", "content": f"证据：\n{EVIDENCE}"},
]


def valid_json(raw: str, keys: list[str]) -> bool:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return False
    try:
        return all(k in json.loads(m.group(0)) for k in keys)
    except json.JSONDecodeError:
        return False


def run(model: str) -> list[dict]:
    results = []
    for name, messages, keys in [
        ("planner_json", TASK_PLANNER, ["sub_questions"]),
        ("critic_json", TASK_CRITIC, ["score", "passed", "gaps"]),
        ("writer_text", TASK_WRITER, []),
    ]:
        t0 = time.time()
        out = chat(messages, model=model)
        dt = time.time() - t0
        ok = valid_json(out, keys) if keys else len(out) > 100
        results.append({"task": name, "secs": round(dt, 1), "ok": ok, "preview": out[:150]})
        print(f"[{model}] {name}: {dt:.1f}s ok={ok}", flush=True)
    return results


def main():
    all_results = {m: run(m) for m in MODELS}
    print("\n===== 汇总 =====")
    for m, rs in all_results.items():
        total = sum(r["secs"] for r in rs)
        oks = sum(1 for r in rs if r["ok"])
        print(f"{m}: 总耗时 {total:.1f}s, 合格 {oks}/3")
        for r in rs:
            print(f"  - {r['task']}: {r['secs']}s ok={r['ok']} | {r['preview']!r}")


if __name__ == "__main__":
    main()
