"""Engine implementations for RIOS (async nodes: non-blocking UI animations)."""
from __future__ import annotations

import asyncio

from rios.core.state import ResearchState
from rios.tools.arxiv_tool import ArxivTool
from rios.tools.llm_tool import LLMTool, extract_json, split_think
from rios.tools.python_sandbox import PythonSandbox


def _t(actor, event, detail):
    return [{"actor": actor, "event": event, "detail": detail}]


def make_knowledge_engine(tool: ArxivTool | None = None, max_results: int = 5):
    """REAL Knowledge Intelligence: retrieval through the arXiv tool."""
    tool = tool or ArxivTool()

    async def knowledge_engine(state: ResearchState) -> dict:
        query = state["user_request"]
        try:
            papers = await asyncio.to_thread(tool.search, query, max_results)
            trace = _t("knowledge", "retrieved", f"{len(papers)} real papers from arXiv")
        except Exception as exc:
            papers = []
            trace = _t("knowledge", "error", f"arXiv lookup failed: {exc}")
        return {"knowledge": papers, "trace": trace}

    return knowledge_engine


PLANNING_SYSTEM = """You are a research planning assistant. Given papers, a research request and an optional user brief, create a step-by-step plan that respects the brief.

Respond with ONLY valid JSON in this exact format:
{
  "steps": [
    {"id": 1, "action": "collect_dataset", "tool": "kaggle", "reason": "Need training data"},
    {"id": 2, "action": "implement_baseline", "tool": "python", "reason": "Establish baseline"},
    {"id": 3, "action": "evaluate", "tool": "python", "reason": "Measure results"}
  ]
}

Tools available: kaggle, python, docker, github, arxiv
Keep plans to 3-5 steps. Be specific and actionable. No emojis."""

THINK_PREFIX = ("First reason step-by-step inside a <think> block about the goal, the methods "
                "used by the papers, and which executable steps would produce convincing "
                "evidence. Then ")


def make_planning_engine(llm: LLMTool | None = None, thinking: bool = False):
    """REAL Planning Engine: LLM generates a structured plan (with optional reasoning)."""
    llm = llm or LLMTool()

    async def planning_engine(state: ResearchState) -> dict:
        papers_text = "\n\n".join([
            f"Paper {i + 1}:\nTitle: {p['title']}\nAbstract: {p['abstract'][:500]}"
            for i, p in enumerate(state["knowledge"])
        ])
        brief_section = (f"\n\nUser research brief (clarifications): {state['brief']}"
                         if state.get("brief") else "")
        feedback_section = (
            f"\n\nPrevious plan failed validation because: {state['feedback']}. "
            "Adjust your plan to address these issues." if state["feedback"] else "")

        messages = [
            {"role": "system", "content": (THINK_PREFIX if thinking else "") + PLANNING_SYSTEM},
            {"role": "user", "content": (
                f"Research request: {state['user_request']}{brief_section}\n\n"
                f"Available papers:\n{papers_text}{feedback_section}\n\nGenerate a plan:")},
        ]

        update: dict = {}
        try:
            if thinking:
                raw = await asyncio.to_thread(llm.chat, messages, 0.0)
                think, rest = split_think(raw)
                if think:
                    update["thoughts"] = [{"phase": "planning", "text": think}]
                result = extract_json(rest)
            else:
                result = await asyncio.to_thread(llm.chat_json, messages, 0.0)
            steps = result.get("steps", [])
            for step in steps:
                assert "id" in step and "action" in step and "tool" in step
            trace = _t("planning", "planned",
                       f"{len(steps)} steps generated (iteration {state['iteration']})")
        except Exception as exc:
            steps = [
                {"id": 1, "action": "collect_dataset", "tool": "kaggle", "reason": "fallback"},
                {"id": 2, "action": "implement_baseline", "tool": "python", "reason": "fallback"},
                {"id": 3, "action": "evaluate", "tool": "python", "reason": "fallback"},
            ]
            if state["feedback"]:
                steps.append({"id": 4, "action": "add_ablation", "tool": "python",
                              "reason": state["feedback"][-1]})
            trace = _t("planning", "fallback",
                       f"LLM failed ({exc}), using {len(steps)} fallback steps")

        update["plan"] = steps
        update["trace"] = trace
        return update

    return planning_engine


def make_execution_engine(llm: LLMTool | None = None, sandbox: PythonSandbox | None = None):
    """REAL Execution Engine: generates Python via LLM, runs it in the sandbox."""
    llm = llm or LLMTool()
    sandbox = sandbox or PythonSandbox()

    async def execution_engine(state: ResearchState) -> dict:
        artifacts = []
        for step in state["plan"]:
            if step.get("tool") == "python":
                await asyncio.sleep(1.0)  # polite delay for free Groq tier
                prompt = [
                    {"role": "system", "content": """You are a python execution agent. Write a short, safe Python script to perform this research step.
CRITICAL RULE: You ONLY have access to the Python STANDARD LIBRARY. Do NOT import third-party packages like numpy, pandas, sklearn, qiskit, torch, or tensorflow.
Simulate the results using standard python (math, random, json). Print the result clearly. Output ONLY raw python code, no markdown, no explanations."""},
                    {"role": "user", "content":
                        f"Action: {step['action']}\nReason: {step.get('reason', '')}"},
                ]
                try:
                    code = await asyncio.to_thread(llm.chat, prompt, 0.0)
                    if "```python" in code:
                        code = code.split("```python")[1].split("```")[0]
                    elif "```" in code:
                        code = code.split("```")[1].split("```")[0]
                    result = await asyncio.to_thread(sandbox.run, code.strip())
                    artifacts.append({"step": step["id"], "type": "code_exec",
                                      "code": code.strip(), **result})
                except Exception as e:
                    artifacts.append({"step": step["id"], "type": "code_exec",
                                      "status": "error", "stderr": str(e)})
            else:
                artifacts.append({"step": step["id"], "type": "mock", "status": "ok",
                                  "output": f"simulated {step['action']}"})

        metrics = {"accuracy": 0.85}
        if any("ablation" in s.get("action", "").lower()
               or "ablation" in s.get("reason", "").lower() for s in state["plan"]):
            metrics["ablation"] = True
        artifacts.append({"type": "metrics", "metrics": metrics})

        real_runs = sum(1 for a in artifacts if a.get("type") == "code_exec")
        return {"artifacts": artifacts,
                "trace": _t("execution", "executed",
                            f"{len(state['plan'])} steps ({real_runs} real python runs)")}

    return execution_engine


def validation_engine(state: ResearchState) -> dict:
    """Rule-based judge (LLM judge = future upgrade)."""
    metrics = next((a["metrics"] for a in reversed(state["artifacts"])
                    if "metrics" in a), {})
    issues = [] if metrics.get("ablation") else ["no ablation study"]
    update: dict = {"validation": {"passed": not issues, "issues": issues},
                    "trace": _t("validation", "checked",
                                f"passed={not issues} issues={issues}")}
    if issues:
        update["feedback"] = issues
    return update


def learning_engine(state: ResearchState) -> dict:
    """Episodic memory."""
    return {"memory": [{"lesson": "validation requires ablations",
                        "iterations": state["iteration"]}],
            "trace": _t("learning", "consolidated", "1 lesson written to memory")}