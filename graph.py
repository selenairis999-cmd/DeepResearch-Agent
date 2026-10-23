"""LangGraph 编排：状态机路由 + Send API 并行 fan-out + Critic 质量反馈回路 + 人在回路

路由：Critic 之后三分支——达标→Writer；未达标且未到上限→打回 Planner；
到 MAX_LOOPS 轮仍不达标→human_gate 节点 interrupt 问人（人工放行/再打回一轮）。
持久化：build_graph() 默认挂 SqliteSaver，每次 invoke 传 {"configurable": {"thread_id": ...}}，
节点每完成一步落检查点，崩溃/中断后同 thread_id 续跑；传 saver=":memory:" 可用内存版（测试）。
"""
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

import persistence
from agents import critic, editor, human_gate, planner, researcher, writer
from state import ResearchState


def fan_out(state: ResearchState) -> list[Send]:
    """Planner 之后：每个子问题派一个 Researcher 并行执行（map-reduce 模式）"""
    return [
        Send("researcher", {"topic": state["topic"], "current_question": q})
        for q in state["sub_questions"]
    ]


def quality_gate(state: ResearchState) -> str:
    """Critic / HumanGate 之后：达标进 Writer，不达标回 Planner，等人工确认进 human_gate"""
    c = state.get("critique") or {}
    if c.get("passed"):
        return "writer"
    if c.get("needs_human"):
        return "human_gate"
    return "planner"


def build_graph(saver=None):
    """saver=None → 默认 SqliteSaver（config.CHECKPOINT_DB）；传 ":memory:" → 内存版"""
    if saver is None:
        saver = persistence.make_saver()
    elif saver == ":memory:":
        saver = persistence.make_saver(":memory:")

    g = StateGraph(ResearchState)
    g.add_node("planner", planner.planner)
    g.add_node("researcher", researcher.researcher)
    g.add_node("critic", critic.critic)
    g.add_node("human_gate", human_gate.human_gate)
    g.add_node("writer", writer.writer)
    g.add_node("editor", editor.editor)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["researcher"])
    g.add_edge("researcher", "critic")
    g.add_conditional_edges(
        "critic", quality_gate, {"planner": "planner", "writer": "writer", "human_gate": "human_gate"},
    )
    g.add_conditional_edges("human_gate", quality_gate, {"planner": "planner", "writer": "writer"})
    g.add_edge("writer", "editor")
    g.add_edge("editor", END)
    return g.compile(checkpointer=saver)
