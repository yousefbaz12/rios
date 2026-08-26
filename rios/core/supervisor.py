"""Research Supervisor (LangGraph orchestrator)."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rios import engines
from rios.core.state import ResearchState
from rios.tools.arxiv_tool import ArxivTool
from rios.tools.llm_tool import LLMTool
from rios.tools.python_sandbox import PythonSandbox

_FORWARD = {
    "init": "knowledge",
    "knowledge": "planning",
    "planning": "execution",
    "execution": "validation",
    "learning": "done",
}


def _next_phase(state: ResearchState) -> tuple[str, str]:
    phase = state["phase"]
    if phase == "validation":
        v = state.get("validation", {})
        if v.get("passed") or state["iteration"] >= state["max_iterations"]:
            return "learning", "validation closed -> consolidate lessons"
        return "planning", f"validation failed {v.get('issues')} -> re-plan"
    note = (f"start -> {_FORWARD[phase]}" if phase == "init"
            else f"{phase} complete -> {_FORWARD[phase]}")
    return _FORWARD[phase], note


def supervisor(state: ResearchState) -> dict:
    nxt, note = _next_phase(state)
    update: dict = {
        "phase": nxt,
        "supervisor_note": note,
        "trace": [{"actor": "supervisor", "event": "route", "detail": note}],
    }
    if state["phase"] == "validation" and nxt == "planning":
        update["iteration"] = state["iteration"] + 1
    return update


def build_graph(checkpointer=None, arxiv_tool: ArxivTool | None = None,
                llm_tool: LLMTool | None = None, sandbox: PythonSandbox | None = None,
                top_k: int = 5, thinking: bool = False):
    nodes = {
        "knowledge": engines.make_knowledge_engine(arxiv_tool, max_results=top_k),
        "planning": engines.make_planning_engine(llm_tool, thinking=thinking),
        "execution": engines.make_execution_engine(llm_tool, sandbox),
        "validation": engines.validation_engine,
        "learning": engines.learning_engine,
    }
    g = StateGraph(ResearchState)
    g.add_node("supervisor", supervisor)
    for name, fn in nodes.items():
        g.add_node(name, fn)
        g.add_edge(name, "supervisor")
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["phase"],
                            {**{n: n for n in nodes}, "done": END})
    return g.compile(checkpointer=checkpointer)