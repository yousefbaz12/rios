"""RIOS CLI."""
from __future__ import annotations

import argparse
import asyncio
import os

from langgraph.checkpoint.memory import MemorySaver

from rios.core.state import ResearchState
from rios.core.supervisor import build_graph
from rios.tools.arxiv_tool import ArxivTool


def initial_state(request: str, max_iterations: int = 2) -> ResearchState:
    return {
        "user_request": request,
        "brief": "",
        "phase": "init",
        "iteration": 0,
        "max_iterations": max_iterations,
        "supervisor_note": "",
        "knowledge": [],
        "plan": [],
        "artifacts": [],
        "feedback": [],
        "validation": {},
        "memory": [],
        "thoughts": [],
        "trace": [],
    }


def _print_papers(papers: list[dict]) -> None:
    for i, p in enumerate(papers, 1):
        print(f"{i}. {p['title']} ({p['published'][:4]})")
        print(f"   {p['url']}")


def _print_plan(plan: list[dict]) -> None:
    for step in plan:
        print(f"{step['id']}. {step['action']} [{step['tool']}] — {step.get('reason', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rios")
    sub = parser.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("knowledge", help="REAL arXiv retrieval")
    k.add_argument("query")
    k.add_argument("--top-k", type=int, default=5)
    k.add_argument("--no-cache", action="store_true")

    p = sub.add_parser("plan", help="REAL LLM planning")
    p.add_argument("request")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--iterations", type=int, default=2)

    r = sub.add_parser("run", help="full supervised pipeline")
    r.add_argument("request")
    r.add_argument("--thread", default="demo-1")
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--iterations", type=int, default=2, help="max validation loops")

    c = sub.add_parser("chat", help="Launch the ChatGPT-like Research Workspace UI")
    c.add_argument("--port", type=int, default=8000)
    c.add_argument("--host", default="127.0.0.1")

    args = parser.parse_args()

    arxiv_tool = ArxivTool()

    if args.cmd == "knowledge":
        papers = arxiv_tool.search(args.query, max_results=args.top_k,
                                   use_cache=not args.no_cache)
        print(f"── {len(papers)} real papers from arXiv ──")
        _print_papers(papers)
        return

    from rios.tools.llm_tool import LLMTool
    llm_tool = LLMTool()

    if args.cmd == "plan":
        papers = arxiv_tool.search(args.request, max_results=args.top_k,
                                   use_cache=not args.no_cache)
        state = initial_state(args.request, max_iterations=args.iterations)
        state["knowledge"] = papers
        from rios.engines import make_planning_engine
        result = asyncio.run(make_planning_engine(llm_tool)(state))
        print(f"── plan generated ({len(result['plan'])} steps) ──")
        _print_plan(result["plan"])
        for ev in result["trace"]:
            print(f"[{ev['actor']}] {ev['event']}: {ev['detail']}")
        return

    if args.cmd == "run":
        from rios.tools.python_sandbox import PythonSandbox
        app = build_graph(checkpointer=MemorySaver(), arxiv_tool=arxiv_tool,
                          llm_tool=llm_tool, sandbox=PythonSandbox(), top_k=args.top_k)
        config = {"configurable": {"thread_id": args.thread}, "recursion_limit": 50}
        init = initial_state(args.request, max_iterations=args.iterations)

        for step in app.stream(init, config, stream_mode="updates"):
            for _, update in step.items():
                for ev in update.get("trace", []):
                    print(f"[{ev['actor']:<10}] {ev['event']:<11} {ev['detail']}")

        final = app.get_state(config).values
        print("\n── final ──")
        print(f"phase={final['phase']} iterations={final['iteration']} "
              f"validation={final['validation']}")
        print("── real knowledge used ──")
        _print_papers(final["knowledge"])
        print("── final plan ──")
        _print_plan(final["plan"])

        real_artifacts = [a for a in final.get("artifacts", [])
                          if a.get("type") == "code_exec"]
        if real_artifacts:
            print("── execution artifacts (Python Sandbox) ──")
            for art in real_artifacts[-5:]:
                print(f"  Step {art.get('step', '?')} status={art.get('status')}")
                if art.get("stdout"):
                    print(f"    stdout: {art['stdout'][:150]}")
                if art.get("stderr") and art.get("status") == "error":
                    print(f"    stderr: {art['stderr'][:150]}")
        return

    if args.cmd == "chat":
        import sys
        import chainlit.cli
        chat_file = os.path.join(os.path.dirname(__file__), "chat.py")
        sys.argv = ["chainlit", "run", chat_file,
                    "--port", str(args.port), "--host", args.host]
        chainlit.cli.cli()
        return


if __name__ == "__main__":
    main()