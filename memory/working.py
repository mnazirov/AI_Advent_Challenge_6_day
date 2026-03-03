from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import storage
from memory.models import TaskContext, TaskState


class WorkingMemory:
    def start_task(self, session_id: str, goal: str) -> TaskContext:
        ctx = TaskContext(
            session_id=session_id,
            task_id=f"task-{uuid4().hex[:8]}",
            goal=str(goal or "").strip(),
            state=TaskState.PLANNING,
            updated_at=datetime.utcnow().isoformat(),
        )
        self.save(ctx)
        return ctx

    def load(self, session_id: str) -> TaskContext | None:
        payload = storage.memory_load_working_task(session_id=session_id)
        if not payload:
            return None
        return TaskContext.from_dict(payload)

    def save(self, ctx: TaskContext) -> None:
        if ctx.state == TaskState.DONE:
            # Completed task should not stay in working memory.
            self.clear_session(ctx.session_id)
            return
        payload = ctx.to_dict()
        storage.memory_save_working_task(
            session_id=payload["session_id"],
            task_id=payload["task_id"],
            goal=payload["goal"],
            state=payload["state"],
            plan=payload["plan"],
            current_step=payload["current_step"],
            done_steps=payload["done_steps"],
            open_questions=payload["open_questions"],
            artifacts=payload["artifacts"],
            vars_data=payload["vars"],
            updated_at=payload["updated_at"],
        )

    def clear_session(self, session_id: str) -> None:
        storage.memory_clear_working_task(session_id=session_id)

    def update(self, session_id: str, **patch: object) -> TaskContext:
        ctx = self.load(session_id)
        if ctx is None:
            goal = str(patch.get("goal") or "")
            ctx = self.start_task(session_id=session_id, goal=goal or "Текущая задача")

        if "goal" in patch and patch["goal"]:
            ctx.goal = str(patch["goal"])
        if "plan" in patch and isinstance(patch["plan"], list):
            ctx.plan = [str(x) for x in patch["plan"]]
        if "current_step" in patch and patch["current_step"] is not None:
            ctx.current_step = str(patch["current_step"])
        if "done_steps" in patch and isinstance(patch["done_steps"], list):
            ctx.done_steps = [str(x) for x in patch["done_steps"]]
        if "open_questions" in patch and isinstance(patch["open_questions"], list):
            ctx.open_questions = [str(x) for x in patch["open_questions"]]
        if "artifacts" in patch and isinstance(patch["artifacts"], list):
            ctx.artifacts = [str(x) for x in patch["artifacts"]]
        if "vars" in patch and isinstance(patch["vars"], dict):
            merged = dict(ctx.vars)
            merged.update(patch["vars"])
            ctx.vars = merged

        if "state" in patch and patch["state"]:
            self.transition_state(ctx, TaskState(str(patch["state"])))

        ctx.updated_at = datetime.utcnow().isoformat()
        self.save(ctx)
        return ctx

    def transition_state(self, ctx: TaskContext, new_state: TaskState) -> TaskContext:
        old = ctx.state
        if old == new_state:
            return ctx

        if old == TaskState.PLANNING and new_state == TaskState.EXECUTION:
            if not ctx.plan or not ctx.current_step:
                raise ValueError("Cannot move to EXECUTION without plan and current_step")
        elif old == TaskState.EXECUTION and new_state == TaskState.VALIDATION:
            if ctx.current_step and ctx.current_step not in ctx.done_steps:
                raise ValueError("Cannot move to VALIDATION before current_step is done")
        elif old == TaskState.VALIDATION and new_state == TaskState.DONE:
            if ctx.open_questions:
                raise ValueError("Cannot move to DONE while open_questions exist")
        elif (old, new_state) not in {
            (TaskState.PLANNING, TaskState.EXECUTION),
            (TaskState.EXECUTION, TaskState.VALIDATION),
            (TaskState.VALIDATION, TaskState.DONE),
            (TaskState.EXECUTION, TaskState.PLANNING),
            (TaskState.VALIDATION, TaskState.EXECUTION),
        }:
            raise ValueError(f"Unsupported transition: {old.value} -> {new_state.value}")

        ctx.state = new_state
        return ctx

    def ensure_task(self, session_id: str, goal: str = "") -> TaskContext:
        current = self.load(session_id)
        if current is not None:
            return current
        return self.start_task(session_id=session_id, goal=goal or "Текущая задача")
