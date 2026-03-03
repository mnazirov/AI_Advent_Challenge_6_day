from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    PLANNING = "PLANNING"
    EXECUTION = "EXECUTION"
    VALIDATION = "VALIDATION"
    DONE = "DONE"


@dataclass
class ShortTermMessage:
    session_id: str
    role: str
    content: str
    timestamp: str


@dataclass
class TaskContext:
    session_id: str
    task_id: str
    goal: str
    state: TaskState = TaskState.PLANNING
    plan: list[str] = field(default_factory=list)
    current_step: str = ""
    done_steps: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    vars: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "goal": self.goal,
            "state": self.state.value,
            "plan": list(self.plan),
            "current_step": self.current_step,
            "done_steps": list(self.done_steps),
            "open_questions": list(self.open_questions),
            "artifacts": list(self.artifacts),
            "vars": dict(self.vars),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskContext":
        return cls(
            session_id=str(payload.get("session_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            goal=str(payload.get("goal") or ""),
            state=TaskState(str(payload.get("state") or TaskState.PLANNING.value)),
            plan=[str(x) for x in (payload.get("plan") or [])],
            current_step=str(payload.get("current_step") or ""),
            done_steps=[str(x) for x in (payload.get("done_steps") or [])],
            open_questions=[str(x) for x in (payload.get("open_questions") or [])],
            artifacts=[str(x) for x in (payload.get("artifacts") or [])],
            vars=dict(payload.get("vars") or {}),
            updated_at=str(payload.get("updated_at") or datetime.utcnow().isoformat()),
        )


@dataclass
class MemoryWriteEvent:
    layer: str
    keys: list[str]
    reason: str = ""
