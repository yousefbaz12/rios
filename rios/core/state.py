"""Shared ResearchState: the only communication channel in RIOS."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

Phase = Literal["init", "knowledge", "planning", "execution",
                "validation", "learning", "done"]


class ResearchState(TypedDict):
    # input
    user_request: str
    brief: str
    # orchestration
    phase: Phase
    iteration: int
    max_iterations: int
    supervisor_note: str
    # engine outputs
    knowledge: Annotated[list[dict], operator.add]
    plan: list[dict]                                   # overwritten on each (re)plan
    artifacts: Annotated[list[dict], operator.add]
    feedback: Annotated[list[str], operator.add]
    validation: dict
    memory: Annotated[list[dict], operator.add]
    thoughts: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]