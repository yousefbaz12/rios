
"""Research Workspace — reasoning-mode, animated, systematic UI for RIOS."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import chainlit as cl
import chainlit.input_widget as iw
from langgraph.checkpoint.memory import MemorySaver

from rios.core.state import ResearchState
from rios.core.supervisor import build_graph
from rios.tools.arxiv_tool import ArxivTool
from rios.tools.llm_tool import LLMTool, extract_json, split_think
from rios.tools.python_sandbox import PythonSandbox

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
UPDATE_TIMEOUT = 2.0
SPINNER_STOP_TIMEOUT = 3.0
MAX_RESEARCH_SECONDS = 600

PHASES = {
    "knowledge": ("Knowledge Intelligence", "retrieving papers from arXiv"),
    "planning": ("Planning Engine", "drafting the research plan"),
    "execution": ("Execution Engine", "generating and executing Python"),
    "validation": ("Validation Engine", "reviewing artifacts"),
    "learning": ("Learning Engine", "storing lessons to memory"),
}

STARTERS = [
    "Deepfake Detection using Quantum",
    "Retrieval-augmented generation for medical QA",
    "Quantum machine learning for image classification",
]

CLARIFY_SYSTEM = (
    "You are RIOS, a research assistant. Given a research request, produce up to 3 concise "
    "clarifying questions that would materially change the research direction (scope, dataset, "
    "evaluation metric, deliverable). For each question give 2-4 short options. Also provide an "
    "'assumptions' string: the defaults you will proceed with if the user skips. "
    'Respond ONLY with JSON: {"questions": [{"question": str, "options": [str]}], '
    '"assumptions": str}. No emojis.'
)

CLARIFY_THINK_PREFIX = (
    "First reason briefly inside a <think> block about the research goal, "
    "what is ambiguous, and what must be known before starting. Then "
)


def initial_state(request: str, max_iterations: int, brief: str = "") -> ResearchState:
    return {
        "user_request": request,
        "brief": brief,
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


def build_report(final: dict, timings: list[dict]) -> str:
    lines = ["# Research Report", "", f"**Task:** {final['user_request']}", ""]

    if final.get("brief"):
        lines += ["**Brief:** " + final["brief"], ""]

    if final.get("thoughts"):
        lines += ["", "## Reasoning Traces", ""]
        lines += [f"> {t['text']}\n" for t in final["thoughts"]]

    lines += ["## Sources", ""]
    lines += [
        f"{i}. [{p['title']}]({p['url']}) ({p['published'][:4]})"
        for i, p in enumerate(final.get("knowledge", []), 1)
    ]

    lines += ["", "## Final Plan", ""]
    lines += [
        f"{s['id']}. `{s['action']}` via **{s['tool']}** — {s.get('reason', '')}"
        for s in final.get("plan", [])
    ]

    lines += ["", "## Execution Outputs", ""]
    for artifact in final.get("artifacts", []):
        if artifact.get("type") == "code_exec":
            lines.append(f"### Step {artifact.get('step')} — {artifact.get('status')}")

            if artifact.get("code"):
                lines += ["```python", artifact["code"], "```"]

            if artifact.get("stdout"):
                lines += ["```", artifact["stdout"], "```"]

    lines += ["", "## Validation", ""]
    validation = final.get("validation", {})
    lines.append(
        f"Passed: {validation.get('passed')} | "
        f"Issues: {', '.join(validation.get('issues', [])) or 'none'}"
    )

    lines += ["", "## Lessons", ""]
    lines += [f"- {memory.get('lesson')}" for memory in final.get("memory", [])]

    lines += ["", "## Timings", "", "| Phase | Iteration | Seconds |", "|---|---|---|"]
    lines += [
        f"| {timing['phase']} | {timing['iteration']} | {timing['duration']:.1f} |"
        for timing in timings
    ]

    return "\n".join(lines)


async def spin_while(msg: cl.Message, label: str, coro):
    """Animate a braille spinner + elapsed seconds on msg while coro runs."""
    stop = asyncio.Event()
    start = time.perf_counter()

    async def safe_update() -> bool:
        try:
            await asyncio.wait_for(msg.update(), timeout=UPDATE_TIMEOUT)
            return True
        except Exception:
            return False

    async def tick():
        i = 0

        while not stop.is_set():
            elapsed = time.perf_counter() - start
            msg.content = f"{SPINNER[i % len(SPINNER)]}  {label}  ({elapsed:.0f}s)"
            await safe_update()

            i += 1

            try:
                await asyncio.wait_for(stop.wait(), timeout=0.12)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(tick())

    try:
        return await coro
    finally:
        stop.set()

        try:
            await asyncio.wait_for(task, timeout=SPINNER_STOP_TIMEOUT)
        except Exception:
            task.cancel()

            try:
                await task
            except BaseException:
                pass

        msg.content = ""
        await safe_update()


@cl.on_chat_start
async def on_start():
    settings = await cl.ChatSettings(
        [
            iw.Switch(
                id="thinking",
                label="Reasoning mode (show thinking traces)",
                initial=True,
            ),
            iw.Switch(
                id="clarify",
                label="Ask clarifying questions first",
                initial=True,
            ),
            iw.Select(
                id="provider",
                label="LLM provider",
                values=["groq", "openai", "ollama"],
                initial_value="groq",
            ),
            iw.Slider(
                id="top_k",
                label="Papers to retrieve",
                initial=5,
                min=1,
                max=10,
                step=1,
            ),
            iw.Slider(
                id="max_iterations",
                label="Self-correction budget",
                initial=2,
                min=0,
                max=4,
                step=1,
            ),
        ]
    ).send()

    cl.user_session.set("settings", settings)
    cl.user_session.set("research_running", False)

    await cl.Message(
        author="RIOS",
        content=(
            "Welcome to RIOS — your Research Intelligence Operating System.\n\n"
            "With Reasoning mode on, I think visibly before acting. I clarify your goal, "
            "retrieve real papers, draft a plan, execute Python in a sandbox, and self-correct "
            "until the research validates. Every stage animates independently.\n\n"
            "Pick a starter or type your own research question:"
        ),
        actions=[
            cl.Action(name="starter", label=starter, payload={"prompt": starter})
            for starter in STARTERS
        ],
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    cl.user_session.set("settings", settings)


async def understand(prompt: str, llm: LLMTool, thinking: bool) -> str:
    msg = cl.Message(author="RIOS", content="")
    await msg.send()

    system = (CLARIFY_THINK_PREFIX if thinking else "") + CLARIFY_SYSTEM

    raw = await spin_while(
        msg,
        "analyzing your request...",
        asyncio.to_thread(
            llm.chat,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        ),
    )

    think, rest = split_think(raw)

    if thinking and think:
        thinking_msg = cl.Message(author="RIOS", content="")
        await thinking_msg.send()
        await thinking_msg.stream_token("**Thinking**\n\n" + think)

    try:
        data = extract_json(rest)
    except Exception:
        data = {}

    questions = data.get("questions", [])
    assumptions = data.get("assumptions", "")

    if not questions:
        await msg.stream_token("**Understanding** — proceeding with the request as stated.")
        return assumptions

    questions_md = "\n".join(
        f"{i}. {question['question']}"
        + (
            f"  *({' / '.join(question['options'])})*"
            if question.get("options")
            else ""
        )
        for i, question in enumerate(questions, 1)
    )

    await msg.stream_token(
        "**Understanding** — to target this research properly, please clarify:\n\n"
        + questions_md
        + f"\n\n*Reply with your answers, or type **skip** to proceed autonomously using: "
        f"{assumptions}*"
    )

    response = await cl.AskUserMessage(
        content="Your answers (or 'skip'):",
        timeout=300,
    ).send()

    answer = ""

    if response:
        answer = (
            response.get("content", "")
            if isinstance(response, dict)
            else getattr(response, "content", "")
        )

    if answer.strip().lower() in ("", "skip", "-"):
        await msg.stream_token("\n\nProceeding autonomously with the stated assumptions.")
        return assumptions

    await msg.stream_token("\n\nClarifications received — incorporated into the research brief.")
    return answer.strip()


async def run_research(prompt: str) -> None:
    if cl.user_session.get("research_running"):
        await cl.Message(
            author="RIOS",
            content="A research run is already in progress. Please wait for it to finish.",
        ).send()
        return

    cl.user_session.set("research_running", True)

    settings = cl.user_session.get("settings") or {}
    top_k = int(settings.get("top_k", 5))
    max_iterations = int(settings.get("max_iterations", 2))
    provider = settings.get("provider", "groq")
    thinking = bool(settings.get("thinking", True))

    try:
        llm = LLMTool(provider=provider)
    except Exception as exc:
        error_msg = cl.Message(author="RIOS", content="")
        await error_msg.send()
        await error_msg.stream_token(f"Configuration error: `{exc}`")
        cl.user_session.set("research_running", False)
        return

    brief = ""

    try:
        if settings.get("clarify", True):
            brief = await understand(prompt, llm, thinking)

        graph = build_graph(
            checkpointer=MemorySaver(),
            arxiv_tool=ArxivTool(),
            llm_tool=llm,
            sandbox=PythonSandbox(),
            top_k=top_k,
            thinking=thinking,
        )

        counter = cl.user_session.get("counter", 0) + 1
        cl.user_session.set("counter", counter)

        config = {
            "configurable": {"thread_id": f"{cl.context.session.id}-{counter}"},
            "recursion_limit": 50,
        }

        start_time = time.perf_counter()
        phase_start: dict[str, float] = {}
        timings: list[dict] = []
        iteration = 0
        current: cl.Message | None = None
        pending_title = ""

        spinner: dict = {"task": None, "stop": None, "msg": None}

        async def safe_message_update(msg: cl.Message) -> bool:
            try:
                await asyncio.wait_for(msg.update(), timeout=UPDATE_TIMEOUT)
                return True
            except Exception:
                return False

        async def stop_spinner():
            if spinner["task"] is not None:
                spinner["stop"].set()

                try:
                    await asyncio.wait_for(
                        spinner["task"],
                        timeout=SPINNER_STOP_TIMEOUT,
                    )
                except Exception:
                    spinner["task"].cancel()

                    try:
                        await spinner["task"]
                    except BaseException:
                        pass

                spinner["task"] = None
                spinner["stop"] = None

            if spinner["msg"] is not None and spinner["msg"].content:
                spinner["msg"].content = ""
                await safe_message_update(spinner["msg"])
                spinner["msg"] = None

        async def start_spinner(msg: cl.Message, label: str):
            await stop_spinner()

            stop_event = asyncio.Event()
            spinner_start = time.perf_counter()

            async def tick():
                i = 0

                while not stop_event.is_set():
                    elapsed = time.perf_counter() - spinner_start
                    msg.content = f"{SPINNER[i % len(SPINNER)]}  {label}  ({elapsed:.0f}s)"
                    await safe_message_update(msg)

                    i += 1

                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=0.12)
                    except asyncio.TimeoutError:
                        pass

            spinner["stop"] = stop_event
            spinner["msg"] = msg
            spinner["task"] = asyncio.create_task(tick())

        try:
            async with asyncio.timeout(MAX_RESEARCH_SECONDS):
                async for event in graph.astream(
                    initial_state(prompt, max_iterations, brief),
                    config,
                    stream_mode="updates",
                ):
                    await stop_spinner()

                    for node, update in event.items():
                        if node == "supervisor":
                            phase = update.get("phase")
                            iteration = update.get("iteration", iteration)

                            if phase in PHASES:
                                name, doing = PHASES[phase]
                                phase_start[phase] = time.perf_counter()

                                current = cl.Message(author="RIOS", content="")
                                await current.send()

                                pending_title = f"**{name}** — {doing}..."
                                await start_spinner(current, f"{name} — {doing}...")

                            continue

                        if node in phase_start:
                            timings.append(
                                {
                                    "phase": node,
                                    "iteration": iteration,
                                    "duration": time.perf_counter()
                                    - phase_start.pop(node),
                                }
                            )

                        if current is None:
                            continue

                        if pending_title:
                            await current.stream_token(pending_title + "\n")
                            pending_title = ""

                        if node == "knowledge":
                            for i, paper in enumerate(update.get("knowledge", []), 1):
                                await current.stream_token(
                                    f"\n{i}. [{paper['title']}]({paper['url']}) "
                                    f"— {paper['published'][:4]}"
                                )

                        if node == "planning":
                            for thought in update.get("thoughts", []):
                                thinking_msg = cl.Message(author="RIOS", content="")
                                await thinking_msg.send()
                                await thinking_msg.stream_token(
                                    "**Thinking**\n\n" + thought["text"]
                                )

                            for step in update.get("plan", []):
                                await current.stream_token(
                                    f"\n{step['id']}. `{step['action']}` "
                                    f"via **{step['tool']}** — {step.get('reason', '')}"
                                )

                        if node == "execution":
                            code_artifacts = [
                                artifact
                                for artifact in update.get("artifacts", [])
                                if artifact.get("type") == "code_exec"
                            ]

                            for artifact in code_artifacts:
                                await current.stream_token(
                                    f"\n\n*Step {artifact.get('step')} — generated code:*"
                                )

                                if artifact.get("code"):
                                    await current.stream_token(
                                        f"\n```python\n{artifact['code']}\n```"
                                    )

                                if artifact.get("stdout"):
                                    await current.stream_token(
                                        f"\n*Sandbox output:*\n```\n{artifact['stdout']}\n```"
                                    )

                                if artifact.get("status") == "error" and artifact.get("stderr"):
                                    await current.stream_token(
                                        f"\n*Error:* `{artifact['stderr'][:200]}`"
                                    )

                        if node == "validation":
                            validation = update.get("validation", {})

                            if validation.get("passed"):
                                await current.stream_token("\n\nReview passed.")
                            else:
                                await current.stream_token(
                                    f"\n\nIssues found: "
                                    f"{', '.join(validation.get('issues', []))}. "
                                    "Requesting a revised plan."
                                )

                        if node == "learning":
                            for memory in update.get("memory", []):
                                await current.stream_token(
                                    f"\n\nLesson stored: *{memory.get('lesson', '')}*"
                                )

            await stop_spinner()

            final = graph.get_state(config).values
            total = time.perf_counter() - start_time

            summary = cl.Message(author="RIOS", content="")
            await summary.send()

            rows = [
                "**Research Summary**",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Self-correction loops | {final['iteration']} |",
                f"| Validation | {'passed' if final['validation'].get('passed') else 'failed'} |",
                f"| Papers used | {len(final.get('knowledge', []))} |",
                f"| Total runtime | {total:.1f}s |",
                "",
                "| Phase | Iteration | Seconds |",
                "|---|---|---|",
            ]

            rows += [
                f"| {timing['phase']} | {timing['iteration']} | {timing['duration']:.1f} |"
                for timing in timings
            ]

            await summary.stream_token("\n".join(rows))

            cl.user_session.set("final", final)
            cl.user_session.set("report", build_report(final, timings))

        except asyncio.TimeoutError:
            await stop_spinner()

            target = current or cl.Message(author="RIOS", content="")

            if current is None:
                await target.send()

            await target.stream_token(
                f"\n\nResearch stopped after {MAX_RESEARCH_SECONDS} seconds "
                "to prevent an infinite running task."
            )

        except Exception as exc:
            await stop_spinner()

            target = current or cl.Message(author="RIOS", content="")

            if current is None:
                await target.send()

            await target.stream_token(f"\n\nError during research: `{exc}`")

    finally:
        cl.user_session.set("research_running", False)

    await cl.Message(
        author="RIOS",
        content="Report ready. Choose an action:",
        actions=[
            cl.Action(name="view_report", label="View full report", payload={}),
            cl.Action(name="download_report", label="Download .md", payload={}),
            cl.Action(name="explain", label="Explain findings simply", payload={}),
            cl.Action(
                name="next_experiments",
                label="Suggest next experiments",
                payload={},
            ),
        ],
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    await run_research(message.content)


@cl.action_callback("starter")
async def on_starter(action: cl.Action):
    await run_research(action.payload["prompt"])


@cl.action_callback("view_report")
async def on_view_report(action: cl.Action):
    await cl.Message(
        author="RIOS",
        content="Opening the full report in the side panel.",
        elements=[
            cl.Text(
                name="Research Report",
                content=cl.user_session.get("report", ""),
                mime="text/markdown",
                display="side",
            )
        ],
    ).send()


@cl.action_callback("download_report")
async def on_download_report(action: cl.Action):
    out = Path(".rios_reports")
    out.mkdir(exist_ok=True)

    path = out / f"report_{int(time.time())}.md"
    path.write_text(cl.user_session.get("report", ""), encoding="utf-8")

    await cl.Message(
        author="RIOS",
        content="Report file attached.",
        elements=[
            cl.File(
                name=path.name,
                path=str(path),
                mime="text/markdown",
            )
        ],
    ).send()


@cl.action_callback("explain")
async def on_explain(action: cl.Action):
    final = cl.user_session.get("final")

    if not final:
        await cl.Message(
            author="RIOS",
            content="No completed report is available yet.",
        ).send()
        return

    msg = cl.Message(author="RIOS", content="")
    await msg.send()

    text = await spin_while(
        msg,
        "explaining findings...",
        asyncio.to_thread(
            LLMTool().chat,
            [
                {
                    "role": "system",
                    "content": (
                        "Explain these research results in plain, simple language "
                        "for a non-expert. No emojis. Short paragraphs."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {final['user_request']}\n"
                        f"Brief: {final.get('brief', '')}\n"
                        f"Outputs: {final['artifacts']}\n"
                        f"Validation: {final['validation']}"
                    ),
                },
            ],
        ),
    )

    await msg.stream_token(text)


@cl.action_callback("next_experiments")
async def on_next_experiments(action: cl.Action):
    final = cl.user_session.get("final")

    if not final:
        await cl.Message(
            author="RIOS",
            content="No completed report is available yet.",
        ).send()
        return

    msg = cl.Message(author="RIOS", content="")
    await msg.send()

    text = await spin_while(
        msg,
        "designing follow-up experiments...",
        asyncio.to_thread(
            LLMTool().chat,
            [
                {
                    "role": "system",
                    "content": (
                        "Given this completed research, propose 3 concrete follow-up "
                        "experiments. Markdown numbered list. No emojis."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {final['user_request']}\n"
                        f"Final plan: {final['plan']}\n"
                        f"Outputs: {final['artifacts']}"
                    ),
                },
            ],
        ),
    )

    await msg.stream_token(text)