"""LangGraph 编排：状态机路由 + Send API 并行 fan-out + Critic 质量反馈回路"""
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents import critic, editor, planner, researcher, writer
from state import ResearchState


def fan_out(state: ResearchState) -> list[Send]:
    """Planner 之后：每个子问题派一个 Researcher 并行执行（map-reduce 模式）"""
    return [
        Send("researcher", {"topic": state["topic"], "current_question": q})
        for q in state["sub_questions"]
    ]


def quality_gate(state: ResearchState) -> str:
    """Critic 之后：不达标回 Planner 补充检索，达标进 Writer"""
    return "writer" if state["critique"]["passed"] else "planner"


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner.planner)
    g.add_node("researcher", researcher.researcher)
    g.add_node("critic", critic.critic)
    g.add_node("writer", writer.writer)
    g.add_node("editor", editor.editor)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["researcher"])
    g.add_edge("researcher", "critic")
    g.add_conditional_edges("critic", quality_gate, {"planner": "planner", "writer": "writer"})
    g.add_edge("writer", "editor")
    g.add_edge("editor", END)
    return g.compile()
