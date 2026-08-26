# RIOS — Research Intelligence Operating System

**An experimental AI research workspace for evidence-oriented, supervised, and self-correcting research workflows.**

RIOS transforms a natural-language research question into a structured research process. It retrieves academic literature, generates an executable research plan, runs supported Python experiments, validates produced artifacts, generates actionable feedback, and revises the workflow when validation requirements are not satisfied.

The system is designed around a shared research state and a supervised **LangGraph orchestration architecture**.

RIOS is intended for exploring:

* AI-assisted scientific research
* Stateful agent architectures
* Automated research workflows
* Evidence-oriented planning
* Tool-using LLM systems
* Feedback-driven agent correction
* Research automation
* Reproducible experimental workflows

# **High Level Architecure**
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/50fc3012-c4ee-424a-8b08-99a1f6a57935" />


---

## Table of Contents

* [Why RIOS](#why-rios)
* [Core Features](#core-features)
  * [Feature Walkthrough](#feature-walkthrough)
  * [Additional Capabilities](#additional-capabilities)
* [System Architecture](#system-architecture)
* [How RIOS Works](#how-rios-works)
* [Research Pipeline](#research-pipeline)
* [Self-Correction Workflow](#self-correction-workflow)
* [Research State](#research-state)
* [Knowledge Retrieval](#knowledge-retrieval)
* [Research Planning](#research-planning)
* [Experiment Execution](#experiment-execution)
* [Validation and Feedback](#validation-and-feedback)
* [Learning and Memory](#learning-and-memory)
* [Interfaces](#interfaces)
* [LLM Provider Support](#llm-provider-support)
* [Research Reports](#research-reports)
* [Reliability and Safety](#reliability-and-safety)
* [Observability](#observability)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Technology Stack](#technology-stack)
* [Current Limitations](#current-limitations)
* [Testing](#testing)
* [Contributing](#contributing)
* [License](#license)

---

# Why RIOS

Many AI research assistants focus primarily on literature summarization or conversational question answering.

RIOS explores a different architecture.

Instead of treating research as a single prompt-response interaction, RIOS models a research task as a **stateful computational workflow**.

A research request moves through several specialized stages:

```text
Research Question
        │
        ▼
Knowledge Retrieval
        │
        ▼
Research Planning
        │
        ▼
Experiment Execution
        │
        ▼
Validation
        │
        ├──── Validation Failed
        │           │
        │           ▼
        │       Feedback
        │           │
        │           ▼
        │       Replanning
        │           │
        │           └────────► Re-execution
        │
        ▼
Learning
        │
        ▼
Research Artifacts
```

The important distinction is the **validation-feedback loop**.

RIOS does not simply generate a plan and stop. It can inspect generated artifacts, identify missing requirements, produce structured feedback, revise the plan, and execute the updated steps again.

The correction cycle remains bounded by a configurable maximum iteration count.

---

# Core Features

RIOS provides a set of integrated features designed to support the complete research workflow. These features cover user interaction, research clarification, academic knowledge retrieval, structured planning, automated experiment execution, validation, and self-correction. Together, they allow a research question to progress through a supervised and evidence-oriented pipeline rather than a single AI response.

---

## Feature Walkthrough

### Interactive Research Workspace

RIOS provides an interactive research workspace where users can begin with a natural-language research question instead of manually configuring a complex pipeline. The interface offers predefined research starters while also supporting fully custom research topics.

The workspace provides live visibility into the research process. Users can configure important execution settings, select the preferred LLM provider, control the number of retrieved papers, define the maximum correction iterations, and optionally enable reasoning visibility.

Rather than treating the interaction as a simple chatbot conversation, RIOS uses the submitted question as the starting point for a complete structured research workflow.

**Key capabilities:**

* Natural-language research input
* Ready-to-use research starters
* Interactive Chainlit interface
* Configurable research settings
* Multi-provider LLM support
* Live phase and execution updates
* Downloadable research reports

---

### Intelligent Research Clarification

Before executing a research workflow, RIOS can identify missing or ambiguous requirements and request targeted clarification from the user.

Instead of making arbitrary assumptions, the system analyzes the initial research question and determines which details could significantly influence the research strategy. These may include the target dataset, methodological direction, evaluation criteria, model family, or experimental objective.

The clarification stage improves the quality of downstream retrieval and planning by converting a broad research question into a more precise **research brief**. Users can answer the generated questions or allow RIOS to continue using reasonable defaults.

This creates a balance between **human supervision** and **autonomous research execution**.

**Key capabilities:**

* Automatic ambiguity detection
* Context-aware clarification questions
* Research objective refinement
* Dataset and evaluation selection
* Methodology preference collection
* Optional autonomous continuation
* Structured clarified research brief

---

### Knowledge Intelligence and Research Planning

RIOS automatically transforms the clarified research problem into an evidence-informed research workflow.

The **Knowledge Intelligence Engine** first searches academic literature and retrieves relevant papers from arXiv. Retrieved results are represented using structured metadata so they can be used by subsequent research stages rather than simply displayed as search results.

The **Planning Engine** then combines the research question, clarified requirements, and retrieved evidence to construct a structured multi-step research plan. Each step describes a concrete objective and can be associated with an appropriate execution tool.

For example, a research workflow may include dataset preparation, preprocessing, model implementation, experimental evaluation, and research artifact generation.

This separation between **evidence retrieval** and **execution planning** allows RIOS to build a research strategy before attempting experiments.

**Knowledge Intelligence features:**

* Automated academic paper retrieval
* arXiv integration
* Configurable top-k retrieval
* Structured paper metadata
* Cached academic queries
* Retry and backoff mechanisms

**Planning features:**

* LLM-generated research plans
* Structured multi-step workflows
* Tool assignment for individual steps
* Explicit experimental objectives
* Plan status tracking
* Support for later feedback-driven replanning

---

### Automated Python Experiment Execution

RIOS can move beyond research planning by generating and executing Python experiments for supported research steps.

The **Execution Engine** translates executable plan steps into Python code and runs them through a bounded subprocess environment. The generated program, execution status, standard output, standard error, and resulting artifacts are captured as part of the shared research state.

This makes computational results directly available to later stages of the pipeline. The Validation Engine can inspect these artifacts and determine whether the experiment satisfies the requirements defined by the research plan.

If important requirements are missing, RIOS can use the validation feedback to revise the plan and execute an updated experiment.

The current research prototype supports lightweight Python experimentation. Generated results should therefore be verified before being treated as scientific evidence.

**Key capabilities:**

* Automatic Python code generation
* Programmatic experiment execution
* Sandboxed-style bounded subprocess runner
* Execution timeout protection
* Standard output and error capture
* Experimental result tracking
* Artifact storage
* Integration with validation and self-correction

Together, these features allow the workflow to progress from:

```text
Research Question → Clarification → Evidence Retrieval → Planning → Experiment Execution → Validation → Correction
```

---

## Additional Capabilities

The following capabilities extend and support the workflow described above, covering orchestration internals, reliability mechanisms, and operational details.

## 1. End-to-End Research Workflow

RIOS provides an integrated pipeline covering multiple stages of a research task.

The system can:

* Accept natural-language research questions
* Retrieve related academic papers
* Generate structured research plans
* Select tools for individual plan steps
* Generate Python experiments
* Execute supported experiments
* Capture execution outputs
* Validate produced artifacts
* Generate feedback from validation failures
* Replan unsuccessful workflows
* Store lessons from completed runs
* Generate structured research reports

This creates a workflow closer to a small research operating environment than a traditional chatbot.

---

## 2. Stateful Research Orchestration

Every research run is represented through a shared typed state called:

```python
ResearchState
```

The state moves between the different research engines.

This allows separate components to operate on the same evolving research context.

Typical state fields include:

```text
request
clarified_brief
papers
plan
artifacts
feedback
validation
memories
reasoning_traces
events
timings
configuration
```

The architecture allows later stages to inspect outputs produced by earlier stages without rebuilding the entire context.

---

## 3. LangGraph-Based Supervisor

RIOS uses **LangGraph** to coordinate the research workflow.

The supervisor determines which research engine should execute next.

Typical routing follows:

```text
Knowledge
   ↓
Planning
   ↓
Execution
   ↓
Validation
   ↓
Learning
```

Validation can modify the normal flow.

If requirements are not satisfied:

```text
Validation
   ↓
Feedback
   ↓
Replanning
   ↓
Execution
   ↓
Validation
```

This makes the workflow explicitly stateful rather than relying entirely on prompt chaining.

---

## 4. Academic Knowledge Retrieval

The Knowledge Engine retrieves academic literature using the arXiv API.

Current retrieval capabilities include:

* Keyword-based academic search
* Configurable top-k results
* Structured paper metadata
* Title extraction
* Author information
* Abstract retrieval
* Publication metadata
* Request retry handling
* Backoff behavior
* Request timeouts
* 24-hour local caching

Example:

```bash
rios knowledge "quantum machine learning for image classification" --top-k 5
```

The retrieved literature becomes part of the shared research state and can subsequently inform planning.

---

## 5. Structured Research Planning

The Planning Engine transforms the research problem and retrieved evidence into an executable plan.

A typical plan contains approximately three to five steps.

Each step can describe:

* Research objective
* Required action
* Assigned tool
* Expected output
* Execution requirements

Example conceptual plan:

```text
1. Review retrieved literature.
2. Define experimental assumptions.
3. Generate a baseline experiment.
4. Evaluate experimental results.
5. Summarize findings.
```

The generated plan is represented using structured data rather than unrestricted text whenever possible.

---

## 6. Automated Python Experiment Generation

RIOS can generate Python programs for supported experimental steps.

The Execution Engine can:

* Generate Python source code
* Execute generated programs
* Record execution status
* Capture standard output
* Capture standard error
* Detect execution failures
* Store produced artifacts
* Associate results with plan steps

This enables the research workflow to move beyond text generation into limited computational experimentation.

---

## 7. Bounded Python Execution

Generated Python is executed through a bounded subprocess environment.

Current safeguards include:

* Per-process timeout
* Captured standard output
* Captured standard error
* Output size limits
* Failure detection
* Controlled subprocess lifecycle

The default execution timeout is intentionally small to prevent generated experiments from hanging indefinitely.

> The current executor is **not an OS-level security sandbox**.

RIOS should therefore be executed inside an isolated and trusted development environment.

---

## 8. Rule-Based Research Validation

The Validation Engine examines accumulated research artifacts using explicit validation criteria.

Validation can identify problems such as:

* Missing experiment outputs
* Missing required artifacts
* Incomplete execution results
* Missing plan requirements
* Unsupported steps
* Failed generated programs

Instead of simply returning a Boolean result, validation can produce actionable issues.

Example:

```text
Validation failed:

- Experimental output was not produced.
- The required comparison step was not executed.
- The generated artifact does not satisfy the requested format.
```

These issues become input to the correction workflow.

---

## 9. Self-Correcting Research Loop

One of the central features of RIOS is bounded workflow correction.

When validation fails, RIOS can generate feedback and revise the current research plan.

```text
Validation Failure
        │
        ▼
Feedback Generation
        │
        ▼
Plan Revision
        │
        ▼
Re-execution
        │
        ▼
Validation
```

Correction continues while:

```text
validation_passed == False
```

and:

```text
iteration < I_max
```

where `I_max` represents the maximum configured correction budget.

The loop therefore cannot continue indefinitely.

---

## 10. Feedback-Driven Replanning

Validation issues are converted into structured feedback.

The planner receives:

* Previous plan
* Validation findings
* Failed artifact information
* Missing requirements
* Remaining research objectives

It then creates an updated plan.

This approach allows the workflow to react to execution outcomes instead of relying entirely on the original plan.

---

## 11. Episodic Learning

After completing the validation cycle, RIOS enters a Learning phase.

The Learning Engine generates a compact lesson summarizing useful information from the research run.

Examples may include:

* Which experiment failed
* Which assumption was incorrect
* Which correction improved the result
* Which validation requirement was initially missed

These lessons remain available inside the current research state.

Current memory is primarily **run-level episodic memory**.

It does not yet represent a persistent cross-session knowledge system.

---

## 12. Multi-Provider LLM Architecture

RIOS includes a provider abstraction layer for multiple language model backends.

Supported providers include:

* Groq
* OpenAI
* Anthropic
* Ollama

Groq is currently the default provider.

The adapter isolates provider-specific behavior from the research engines.

This allows the orchestration architecture to remain mostly independent from the selected model provider.

---

## 13. Interactive Chainlit Workspace

RIOS includes an interactive web interface built using Chainlit.

Launch it with:

```bash
rios chat --host 127.0.0.1 --port 8000
```

The interface provides:

* Natural-language research input
* Live research phase updates
* Provider configuration
* Paper retrieval configuration
* Maximum correction iterations
* Optional clarification behavior
* Optional reasoning-trace display
* Research progress visibility
* Downloadable Markdown reports

The interface allows users to observe how a research task progresses across the internal engines.

---

## 14. Command-Line Interface

RIOS also provides a CLI for direct experimentation and automation.

Available workflows include:

```bash
rios knowledge
```

Retrieve academic papers.

```bash
rios plan
```

Generate a research plan.

```bash
rios run
```

Execute the complete research pipeline.

```bash
rios chat
```

Start the interactive research workspace.

This allows RIOS to operate both as an interactive application and as a command-line research tool.

---

## 15. Configurable Research Runs

Important research settings can be changed for individual runs.

Examples include:

* Number of papers retrieved
* Maximum correction iterations
* Selected LLM provider
* Reasoning trace visibility
* Clarification behavior
* Execution timeout behavior

This makes the system suitable for experimenting with different agent configurations.

---

## 16. Research Artifact Tracking

RIOS records artifacts generated throughout a research run.

Artifacts may contain:

* Generated Python
* Execution output
* Error output
* Intermediate results
* Validation results
* Research plans
* Retrieved evidence
* Learned lessons

Artifacts remain connected to the research state so downstream engines can inspect them.

---

## 17. Research Report Generation

Interactive sessions can produce downloadable Markdown reports.

A generated report can include:

* Original research request
* Clarified research brief
* Retrieved academic papers
* Final research plan
* Generated Python code
* Execution results
* Validation findings
* Feedback
* Consolidated lessons
* Reasoning traces
* Per-phase execution timings

The report provides a traceable record of how the research workflow evolved.

---

## 18. Execution Observability

RIOS records events and timing information throughout the workflow.

Observability information can include:

* Current research phase
* Phase transitions
* Execution status
* Validation status
* Correction iteration number
* Engine runtime
* Errors
* Reasoning events
* Supervisor routing events

This makes agent behavior easier to inspect and debug.

---

## 19. Retry and Cache Infrastructure

External academic retrieval includes reliability mechanisms.

These include:

* Bounded retries
* Request timeouts
* Exponential backoff
* 24-hour caching

Caching reduces unnecessary calls to external academic services and improves repeated-query performance.

---

## 20. Concurrent Run Protection

Within an interactive Chainlit session, RIOS blocks overlapping research runs.

This avoids multiple pipelines simultaneously modifying the same interactive session state.

---

# System Architecture

RIOS follows a layered architecture.

```text
┌──────────────────────────────────────────────────────────────┐
│                        USER ACCESS                           │
│                                                              │
│             Chainlit UI              CLI                    │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                LANGGRAPH SUPERVISOR                          │
│                                                              │
│ Knowledge → Planning → Execution → Validation → Learning     │
│                         │                                    │
│                         ▼                                    │
│                Feedback + Replanning                         │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                  SHARED RESEARCH STATE                       │
│                                                              │
│ request │ papers │ plan │ artifacts │ feedback │ validation │
│ memories │ traces │ events │ timings │ configuration         │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    RESEARCH ENGINES                          │
│                                                              │
│ Knowledge │ Planning │ Execution │ Validation │ Learning     │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                     TOOLS & SERVICES                         │
│                                                              │
│ arXiv │ LLM Adapter │ Python Runner │ Cache │ Reporting      │
└──────────────────────────────────────────────────────────────┘
```

The architecture separates orchestration logic from individual research capabilities.

This makes individual engines easier to modify, replace, or extend.

---

# How RIOS Works

A typical research run begins with a question such as:

```text
Deepfake detection using quantum machine learning methods
```

RIOS then performs the following workflow.

### Stage 1 — Knowledge

Relevant academic papers are retrieved.

```text
Question
   ↓
arXiv Search
   ↓
Structured Paper Metadata
```

### Stage 2 — Planning

The research question and retrieved papers are sent to the planning engine.

```text
Question + Evidence
        ↓
       LLM
        ↓
Structured Research Plan
```

### Stage 3 — Execution

Supported steps are converted into executable Python.

```text
Plan Step
   ↓
Python Generation
   ↓
Bounded Execution
   ↓
Artifacts
```

### Stage 4 — Validation

Generated artifacts are checked against explicit requirements.

```text
Artifacts
    ↓
Validation Rules
    ↓
Pass / Issues
```

### Stage 5 — Correction

When validation fails:

```text
Issues
  ↓
Feedback
  ↓
Replanning
  ↓
Re-execution
```

### Stage 6 — Learning

A compact lesson from the completed cycle is stored in the current research state.

---

# Research Pipeline

| Phase          | Responsibility                                                |
| -------------- | ------------------------------------------------------------- |
| **Knowledge**  | Retrieve relevant academic literature and structured metadata |
| **Planning**   | Generate a structured executable research plan                |
| **Execution**  | Generate and execute supported Python experiments             |
| **Validation** | Evaluate accumulated artifacts using explicit rules           |
| **Feedback**   | Convert validation failures into actionable corrections       |
| **Replanning** | Revise the research plan using validation feedback            |
| **Learning**   | Store compact lessons from the completed workflow             |

---

# Self-Correction Workflow

RIOS uses validation as a routing decision.

Normal execution:

```text
Knowledge
    ↓
Planning
    ↓
Execution
    ↓
Validation
    ↓
Learning
```

Failure execution:

```text
Knowledge
    ↓
Planning
    ↓
Execution
    ↓
Validation
    │
    └── Failed
          ↓
       Feedback
          ↓
       Replanning
          ↓
       Execution
          ↓
       Validation
```

The correction process remains bounded by:

```text
I_max
```

Once the maximum iteration count is reached, the workflow exits the correction cycle and proceeds to Learning.

---

# Research State

A central `ResearchState` object carries information between engines.

Conceptually:

```python
ResearchState(
    request=...,
    clarified_brief=...,
    papers=...,
    plan=...,
    artifacts=...,
    feedback=...,
    validation=...,
    memories=...,
    reasoning_traces=...,
    events=...,
    timings=...,
    config=...
)
```

This shared-state architecture is important because research tasks are inherently multi-stage.

Each engine operates on the accumulated state rather than receiving an isolated prompt.

---

# Knowledge Retrieval

The current Knowledge Engine uses the **arXiv Atom API**.

Example:

```bash
rios knowledge \
  "retrieval augmented generation for medical question answering" \
  --top-k 5
```

Current retrieval features include:

* Query-based paper search
* Structured metadata
* Configurable result count
* Caching
* Retry handling
* Timeout handling
* Backoff behavior

---

# Research Planning

Generate a plan without executing the complete workflow:

```bash
rios plan \
  "quantum machine learning for image classification" \
  --top-k 5
```

The planner uses:

```text
Research Question
       +
Retrieved Literature
       +
Research Configuration
       ↓
       LLM
       ↓
Executable Research Plan
```

Planning outputs are designed to remain compact enough for downstream execution.

---

# Experiment Execution

Execute a complete workflow:

```bash
rios run \
  "deepfake detection using quantum methods" \
  --iterations 2 \
  --top-k 5
```

Supported Python-oriented steps can result in artifacts containing:

```text
Generated source code
Execution status
Standard output
Standard error
Execution timing
```

Unsupported non-Python actions are currently represented using placeholder artifacts.

---

# Validation and Feedback

Validation provides an explicit checkpoint between execution and learning.

Instead of assuming that generated output is correct, RIOS evaluates whether required artifacts have actually been produced.

When problems are detected, feedback is attached to the research state.

The planner can then revise its strategy using that feedback.

This creates the core closed-loop behavior:

```text
Plan
 ↓
Execute
 ↓
Validate
 ↓
Feedback
 ↓
Replan
```

---

# Learning and Memory

The Learning Engine creates compact lessons after the validation cycle.

Example conceptual lesson:

```text
The initial experiment omitted the required baseline comparison.
Validation detected the missing artifact.
The revised plan added the baseline before final evaluation.
```

Current memory is limited to the active research run.

Persistent semantic and episodic memory across independent sessions is not currently implemented.

---

# Interfaces

## Chainlit Web Interface

Start the UI:

```bash
rios chat --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The interface supports configuration for:

* LLM provider
* Retrieved paper count
* Maximum correction iterations
* Clarification behavior
* Reasoning-trace visibility

---

## CLI Interface

Main commands:

```bash
rios knowledge "research topic"
```

```bash
rios plan "research topic"
```

```bash
rios run "research topic"
```

```bash
rios chat
```

---

# LLM Provider Support

Groq is the default provider.

Create an environment configuration file:

```bash
cp .env.example .env
```

Configure Groq:

```env
GROQ_API_KEY=your_api_key_here
```

Use OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
```

Use Anthropic:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key_here
```

Use Ollama:

```env
LLM_PROVIDER=ollama
```

Ollama does not require a remote API key when running locally.

> Never commit `.env` files or API credentials.

---

# Research Reports

The Chainlit application can export Markdown research reports.

Reports may contain:

```text
Research Request
Clarified Brief
Retrieved Sources
Research Plan
Generated Code
Execution Results
Validation Results
Feedback
Lessons Learned
Reasoning Traces
Phase Timings
```

This provides a portable representation of a completed research workflow.

---

# Reliability and Safety

RIOS includes several mechanisms intended to prevent uncontrolled agent behavior.

### Bounded Correction

Research correction is limited by a configurable maximum number of iterations.

### Execution Timeout

Generated Python processes have a fixed execution timeout.

### Bounded Output

Subprocess output capture is restricted.

### Retrieval Timeouts

External literature retrieval cannot wait indefinitely.

### Retry Limits

External requests use bounded retries rather than unlimited retry loops.

### Session Run Locking

Multiple research pipelines cannot execute concurrently inside the same interactive session.

### Defensive LLM Parsing

Structured LLM responses are parsed defensively.

Malformed structured responses can trigger one retry.

### Secret Management

API credentials are loaded using environment variables.

`.env` files should remain excluded from version control.

---

# Observability

RIOS records information about how research runs progress.

Tracked information can include:

* Engine transitions
* Current phase
* Execution results
* Validation decisions
* Correction iterations
* Timing measurements
* Supervisor events
* Reasoning traces

This information can support debugging, evaluation, and future analysis of agent behavior.

---

# Installation

## Requirements

```text
Python 3.11+
```

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rios.git
cd rios
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install RIOS:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

---

# Configuration

Create a local environment configuration:

```bash
cp .env.example .env
```

Example:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key_here
```

Provider-specific API keys should never be committed to Git.

---

# Usage

## Retrieve Academic Papers

```bash
rios knowledge \
  "retrieval augmented generation for medical question answering" \
  --top-k 5
```

---

## Generate a Research Plan

```bash
rios plan \
  "quantum machine learning for image classification" \
  --top-k 5
```

---

## Run the Complete Research Workflow

```bash
rios run \
  "deepfake detection using quantum methods" \
  --iterations 2 \
  --top-k 5
```

---

## Start the Interactive Workspace

```bash
rios chat --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

---

# Project Structure

```text
RIOS/
├── rios/
│   ├── core/
│   │   ├── state.py
│   │   └── supervisor.py
│   │
│   ├── tools/
│   │   ├── arxiv_tool.py
│   │   ├── llm_tool.py
│   │   └── python_sandbox.py
│   │
│   ├── engines.py
│   ├── cli.py
│   └── chat.py
│
├── tests/
│   └── test_supervisor.py
│
├── pyproject.toml
└── README.md
```

### Core

`state.py`

Defines the shared typed `ResearchState`.

`supervisor.py`

Defines LangGraph routing, phase transitions, and correction logic.

### Tools

`arxiv_tool.py`

Handles academic paper retrieval, retries, and caching.

`llm_tool.py`

Provides the abstraction over supported LLM providers.

`python_sandbox.py`

Executes generated Python through bounded subprocesses.

### Engines

`engines.py`

Contains the specialized research engines:

```text
Knowledge Engine
Planning Engine
Execution Engine
Validation Engine
Learning Engine
```

### Interfaces

`cli.py`

Implements command-line workflows.

`chat.py`

Implements the Chainlit research workspace.

---

# Technology Stack

| Component             | Technology                |
| --------------------- | ------------------------- |
| Core Language         | Python 3.11+              |
| Agent Orchestration   | LangGraph                 |
| Interactive UI        | Chainlit                  |
| Academic Retrieval    | arXiv Atom API            |
| Default LLM           | Groq                      |
| Additional LLMs       | OpenAI, Anthropic, Ollama |
| Concurrent Operations | AsyncIO                   |
| Python Execution      | Subprocess                |
| Testing               | Pytest                    |
| Configuration         | Environment variables     |
| Reporting             | Markdown                  |

---

# Current Limitations

RIOS is intentionally presented as an experimental system.

## No OS-Level Python Sandbox

The Python execution component uses bounded subprocess execution.

It does **not** currently provide strong process isolation.

Generated code should therefore only be executed inside a trusted isolated environment.

---

## Limited Experiment Environment

Generated experiments currently focus primarily on standard-library Python.

Complex machine-learning experiments requiring external datasets or large frameworks are not fully automated.

---

## Placeholder Non-Python Actions

Research plan steps assigned to unsupported tools may currently generate placeholder artifacts instead of real actions.

---

## Rule-Based Validation

The current validator checks explicit programmatic requirements.

It should not be interpreted as genuine scientific peer review.

---

## Run-Level Memory

Lessons are stored during the active workflow.

Cross-session persistent research memory is not currently available.

---

## Basic Literature Retrieval

Current academic retrieval primarily uses arXiv keyword search.

The system does not currently provide:

* Full-text retrieval
* Semantic reranking
* Citation graph analysis
* Citation verification
* Paper deduplication
* Systematic-review-grade evidence synthesis

---

## Generated Results Require Human Verification

Generated plans, experiments, outputs, and conclusions may contain errors.

RIOS should support human researchers rather than replace scientific judgment.

---

# Testing

Run the test suite using:

```bash
pytest -q
```

Tests use a fake arXiv implementation where appropriate so core retrieval contracts can be evaluated without requiring live network access.

Tests currently focus on components such as:

* Supervisor routing
* Research-state transitions
* Retrieval contracts
* Correction behavior
* Failure handling

---

# Contributing

RIOS is an experimental project exploring automated research systems and stateful AI agents.

Contributions involving:

* Research engines
* Validation strategies
* Retrieval systems
* Tool integrations
* Agent evaluation
* Execution safety
* Testing
* Observability

are welcome.

For substantial architectural changes, opening an issue before submitting a large pull request is recommended.

---

# License

A project license has not yet been selected.

Before external reuse or distribution, consider adopting a standard open-source license such as:

* MIT License
* Apache License 2.0

---

# Author

Developed as an experimental platform for exploring:

**Stateful AI agents, automated research workflows, evidence-oriented research, tool-using language models, and feedback-driven LLM orchestration.**
