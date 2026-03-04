from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
import logging
import re
from typing import TYPE_CHECKING

from memory.long_term import LongTermMemory
from memory.models import MemoryWriteEvent, ProfileSource, TaskContext, TaskState
from memory.prompt_builder import PromptBuilder
from memory.router import MemoryRouter
from memory.short_term import ShortTermMemory
from memory.working import WorkingMemory

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = logging.getLogger("memory")

PREVIEW_LENGTH = 120
MAX_DEBUG_TEXT_CHARS = 4000
MAX_WRITE_EVENTS = 50


class MemoryManager:
    def __init__(
        self,
        short_term_limit: int = 30,
        *,
        llm_client: "LLMClient | None" = None,
        step_parser_model: str = "gpt-5-nano",
    ):
        self.short_term = ShortTermMemory(limit_n=short_term_limit)
        self.working = WorkingMemory()
        self.long_term = LongTermMemory()
        self.router = MemoryRouter(llm_client=llm_client, step_parser_model=step_parser_model)
        self.prompt_builder = PromptBuilder()
        self._write_events: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=MAX_WRITE_EVENTS))

    def route_user_message(self, *, session_id: str, user_id: str, user_message: str):
        events = self.router.route_user_message(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            working=self.working,
            long_term=self.long_term,
        )
        self._record_write_events(session_id=session_id, events=events, source="router")
        return events

    def build_messages(
        self,
        *,
        session_id: str,
        user_id: str,
        system_instructions: str,
        data_context: str,
        user_query: str,
    ) -> tuple[list[dict[str, str]], dict[str, str], dict]:
        short_term_context = self.short_term.get_context(session_id)
        working_ctx = self.working.load(session_id)
        long_term_ctx = self.long_term.retrieve(user_id=user_id, query=user_query, top_k=3)
        messages, preview = self.prompt_builder.build(
            system_instructions=system_instructions,
            data_context=data_context,
            long_term=long_term_ctx,
            working=working_ctx,
            short_term_messages=short_term_context,
            user_query=user_query,
        )
        read_meta = dict(long_term_ctx.get("read_meta") or {})
        logger.info(
            "[MEMORY_READ] layer=long_term hits=%s ids=%s reason=%s",
            int(read_meta.get("decision_hits", 0)) + int(read_meta.get("note_hits", 0)),
            (read_meta.get("decision_ids") or []) + (read_meta.get("note_ids") or []),
            "match/score",
        )
        logger.info("[MEMORY_READ] layer=working present=%s", bool(working_ctx))
        logger.info("[MEMORY_READ] layer=short_term turns=%s", len(short_term_context))
        return messages, preview, read_meta

    def append_turn(self, *, session_id: str, user_message: str, assistant_message: str) -> None:
        self.short_term.append(session_id=session_id, role="user", content=user_message)
        self.short_term.append(session_id=session_id, role="assistant", content=assistant_message)

    def clear_session(self, session_id: str) -> None:
        self.short_term.clear_session(session_id)
        self.working.clear_session(session_id)
        self._write_events.pop(session_id, None)

    def clear_working_layer(self, *, session_id: str) -> bool:
        had_task = self.working.load(session_id) is not None
        self.working.clear_session(session_id)
        if had_task:
            self._record_write_event(
                session_id=session_id,
                layer="working",
                keys=["task_context"],
                operation="clear",
                source="debug_ui",
            )
        return had_task

    def delete_long_term_entry(self, *, session_id: str, user_id: str, entry_type: str, entry_id: int) -> bool:
        normalized = str(entry_type or "").strip().lower()
        deleted = False
        if normalized == "decision":
            deleted = self.long_term.delete_decision(user_id=user_id, decision_id=int(entry_id))
            layer = "long_term.decision"
        elif normalized == "note":
            deleted = self.long_term.delete_note(user_id=user_id, note_id=int(entry_id))
            layer = "long_term.note"
        else:
            raise ValueError("entry_type must be 'decision' or 'note'")

        if deleted:
            self._record_write_event(
                session_id=session_id,
                layer=layer,
                keys=["id"],
                operation="delete",
                source="debug_ui",
                entry_id=int(entry_id),
            )
        return deleted

    def get_profile_snapshot(self, *, user_id: str) -> dict:
        return self.long_term.get_profile(user_id=user_id)

    def debug_update_profile_field(
        self,
        *,
        session_id: str,
        user_id: str,
        field: str,
        value: object,
    ) -> dict:
        if str(field) in self.long_term.CANONICAL_FIELDS:
            self.long_term.update_profile_field(
                user_id=user_id,
                field=field,
                value=value,
                source=ProfileSource.DEBUG_MENU,
                verified=True,
            )
        else:
            self.long_term.add_profile_extra_field(
                user_id=user_id,
                field=field,
                value=value,
                source=ProfileSource.DEBUG_MENU,
            )
        self._record_write_event(
            session_id=session_id,
            layer="long_term.profile",
            keys=[str(field)],
            operation="save",
            source="debug_menu",
        )
        return self.get_profile_snapshot(user_id=user_id)

    def debug_delete_profile_field(self, *, session_id: str, user_id: str, field: str) -> dict:
        self.long_term.delete_profile_field(user_id=user_id, field=field)
        self._record_write_event(
            session_id=session_id,
            layer="long_term.profile",
            keys=[str(field)],
            operation="delete",
            source="debug_menu",
        )
        return self.get_profile_snapshot(user_id=user_id)

    def debug_add_profile_extra_field(
        self,
        *,
        session_id: str,
        user_id: str,
        field: str,
        value: object,
    ) -> dict:
        self.long_term.add_profile_extra_field(
            user_id=user_id,
            field=field,
            value=value,
            source=ProfileSource.DEBUG_MENU,
        )
        self._record_write_event(
            session_id=session_id,
            layer="long_term.profile",
            keys=[str(field)],
            operation="save",
            source="debug_menu",
        )
        return self.get_profile_snapshot(user_id=user_id)

    def debug_confirm_profile_field(self, *, session_id: str, user_id: str, field: str) -> dict:
        self.long_term.confirm_profile_field(user_id=user_id, field=field)
        self._record_write_event(
            session_id=session_id,
            layer="long_term.profile",
            keys=[str(field)],
            operation="save",
            source="debug_menu",
        )
        return self.get_profile_snapshot(user_id=user_id)

    def debug_resolve_profile_conflict(
        self,
        *,
        session_id: str,
        user_id: str,
        field: str,
        chosen_value: object | None = None,
        keep_existing: bool = False,
    ) -> dict:
        self.long_term.resolve_profile_conflict(
            user_id=user_id,
            field=field,
            chosen_value=chosen_value,
            keep_existing=keep_existing,
        )
        self._record_write_event(
            session_id=session_id,
            layer="long_term.profile",
            keys=[str(field), "conflicts"],
            operation="save",
            source="debug_menu",
        )
        return self.get_profile_snapshot(user_id=user_id)

    def hydrate_short_term(self, session_id: str, messages: list[dict[str, str]]) -> None:
        self.short_term.hydrate(session_id=session_id, messages=messages)

    def enforce_planning_gate(self, *, session_id: str, user_message: str) -> str | None:
        ctx: TaskContext | None = self.working.load(session_id)
        if not ctx:
            return None

        msg = (user_message or "").strip().lower()
        status = self.working.get_step_status(ctx)
        step_index = status.get("step_index")
        total_steps = status.get("total_steps")

        if ctx.state == TaskState.PLANNING:
            plan_empty = self._plan_is_empty(ctx)
            if self._is_skip_request(msg) and plan_empty:
                return "Сначала сформируйте план задачи, затем можно перейти к выполнению."
            if bool((ctx.vars or {}).get("plan_guidance_required")):
                return "Опишите шаги плана или нажмите 'Сформировать план автоматически'."
            if plan_empty and self._is_start_execution_request(msg):
                return "Сначала сформируйте план задачи, затем можно перейти к выполнению."
            if plan_empty or self._is_plan_formation_message(msg) or self._is_goal_clarification_message(msg):
                return None
            wants_start = self._is_start_execution_request(msg)
            if wants_start and ctx.plan and ctx.current_step == ctx.plan[0]:
                try:
                    self.working.transition_state(ctx, TaskState.EXECUTION)
                    ctx.updated_at = datetime.utcnow().isoformat()
                    self.working.save(ctx)
                except ValueError as exc:
                    return str(exc)
                return None
            return "Ожидается план и старт задачи."

        if ctx.state == TaskState.EXECUTION:
            if ctx.current_step is None and ctx.done == ctx.plan and ctx.plan:
                if self._is_validation_request(msg):
                    try:
                        self.working.request_validation(session_id)
                        return "Задача переведена в VALIDATION."
                    except ValueError as exc:
                        return str(exc)
                return "All steps completed. Call request_validation() to proceed to VALIDATION."

            if self._is_validation_request(msg):
                try:
                    self.working.request_validation(session_id)
                    return "Задача переведена в VALIDATION."
                except ValueError as exc:
                    return str(exc)

            if self._is_execution_allowed_message(msg, ctx.current_step or ""):
                return None
            return (
                f"Currently executing step {step_index}/{total_steps}: '{ctx.current_step}'. "
                "Complete it before switching context."
            )

        if ctx.state == TaskState.VALIDATION:
            if self._is_done_confirmation(msg):
                try:
                    self.working.transition_state(ctx, TaskState.DONE)
                    ctx.updated_at = datetime.utcnow().isoformat()
                    self.working.save(ctx)
                    return "Задача переведена в DONE."
                except ValueError as exc:
                    return str(exc)
            if self._is_reexecution_request(msg):
                try:
                    self.working.transition_state(ctx, TaskState.EXECUTION)
                    ctx.updated_at = datetime.utcnow().isoformat()
                    self.working.save(ctx)
                except ValueError as exc:
                    return str(exc)
                return None
            return "Состояние VALIDATION: подтвердите завершение или вернитесь к исполнению."

        if ctx.state == TaskState.DONE:
            return "Working memory is frozen in DONE state"
        return None

    def get_working_view(self, *, session_id: str) -> dict:
        ctx = self.working.load(session_id)
        if not ctx:
            return {
                "state": None,
                "current_step": None,
                "step_index": None,
                "total_steps": 0,
                "done": [],
                "plan": [],
            }
        status = self.working.get_step_status(ctx)
        return {
            "state": status["state"],
            "current_step": status["current_step"],
            "step_index": status["step_index"],
            "total_steps": status["total_steps"],
            "done": list(ctx.done),
            "plan": list(ctx.plan),
            "awaiting_validation": bool(
                ctx.state == TaskState.EXECUTION
                and ctx.current_step is None
                and bool(ctx.plan)
                and ctx.done == ctx.plan
            ),
        }

    def get_working_actions(self, *, session_id: str) -> list[dict]:
        ctx = self.working.load(session_id)
        state = ctx.state if ctx else TaskState.PLANNING

        if state == TaskState.PLANNING:
            return [
                {
                    "id": "auto_plan",
                    "label": "Сформировать план автоматически",
                    "kind": "message",
                    "message": "Сформируй план задачи автоматически на основе цели",
                },
                {
                    "id": "clarify_goal",
                    "label": "Уточнить цель задачи",
                    "kind": "message",
                    "message": "Давай уточним цель задачи перед планированием",
                },
            ]

        if state == TaskState.EXECUTION:
            return [
                {
                    "id": "complete_step",
                    "label": "Шаг выполнен",
                    "kind": "message",
                    "message": "Текущий шаг выполнен, переходим к следующему",
                },
                {
                    "id": "show_status",
                    "label": "Статус задачи",
                    "kind": "message",
                    "message": "Покажи текущий статус и прогресс задачи",
                },
            ]

        if state == TaskState.VALIDATION:
            return [
                {
                    "id": "confirm_done",
                    "label": "Подтвердить завершение",
                    "kind": "message",
                    "message": "Подтверждаю выполнение задачи, переходим к итогам",
                },
                {
                    "id": "back_to_execution",
                    "label": "Вернуться к выполнению",
                    "kind": "message",
                    "message": "Нужно доработать, возвращаемся к выполнению",
                },
            ]

        if state == TaskState.DONE:
            return [
                {
                    "id": "new_task",
                    "label": "Начать новую задачу",
                    "kind": "client_action",
                    "client_action": "reset_chat",
                },
                {
                    "id": "save_results",
                    "label": "Сохранить итоги в память",
                    "kind": "message",
                    "message": "Сохрани ключевые решения и итоги задачи в долгосрочную память",
                },
            ]
        return []

    def stats(self, *, session_id: str, user_id: str) -> dict:
        working = self.working.load(session_id)
        longterm = self.long_term.retrieve(user_id=user_id, query="", top_k=3)
        read_meta = dict(longterm.get("read_meta") or {})
        profile = longterm.get("profile") or {}
        has_profile = bool(
            (profile.get("stack_tools") or {}).get("value")
            or (profile.get("response_style") or {}).get("value")
            or (profile.get("hard_constraints") or {}).get("value")
            or (profile.get("user_role_level") or {}).get("value")
            or (profile.get("project_context") or {}).get("value")
            or (profile.get("extra_fields") or {})
            or (profile.get("conflicts") or [])
        )
        return {
            "short_term_messages": len(self.short_term.get_context(session_id)),
            "working_state": working.state.value if working else None,
            "working_task_id": working.task_id if working else None,
            "longterm_profile": has_profile,
            "longterm_decisions": len(longterm.get("decisions") or []),
            "longterm_notes": len(longterm.get("notes") or []),
            "memory_read": read_meta,
            "recent_writes": len(self.get_recent_write_events(session_id=session_id, limit=10)),
        }

    def _record_write_event(
        self,
        *,
        session_id: str,
        layer: str,
        keys: list[str],
        operation: str = "save",
        source: str = "router",
        entry_id: int | None = None,
    ) -> None:
        event = {
            "layer": layer,
            "keys": list(keys or []),
            "operation": operation,
            "source": source,
            "entry_id": entry_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._write_events[session_id].append(event)

    def _record_write_events(self, *, session_id: str, events: list[MemoryWriteEvent], source: str = "router") -> None:
        for event in events:
            self._record_write_event(
                session_id=session_id,
                layer=event.layer,
                keys=event.keys,
                operation="save",
                source=source,
            )

    def get_recent_write_events(self, *, session_id: str, limit: int = 10) -> list[dict]:
        events = list(self._write_events.get(session_id) or [])
        lim = max(1, int(limit))
        return events[-lim:]

    def _format_entry_for_debug(self, entry: dict) -> dict:
        text = str(entry.get("text") or "")
        preview = text[:PREVIEW_LENGTH] + ("…" if len(text) > PREVIEW_LENGTH else "")
        full = text[:MAX_DEBUG_TEXT_CHARS]
        full_truncated = len(text) > MAX_DEBUG_TEXT_CHARS
        if full_truncated:
            full = full + "…"
        return {
            "id": entry.get("id"),
            "tags": list(entry.get("tags") or []),
            "created_at": entry.get("created_at", ""),
            "preview": preview,
            "full": full,
            "full_truncated": full_truncated,
            "entry_type": str(entry.get("type") or ""),
            "source": str(entry.get("source") or ""),
        }

    def debug_snapshot(
        self,
        *,
        session_id: str,
        user_id: str,
        query: str = "",
        top_k: int = 3,
    ) -> dict:
        """Снимок трёх слоёв памяти для Debug UI. top_k в диапазоне 1..10."""
        top_k = max(1, min(10, int(top_k)))
        short_snapshot = self.short_term.snapshot(session_id)
        working_ctx = self.working.load(session_id)
        working_present = working_ctx is not None
        working_task = None
        if working_ctx:
            d = working_ctx.to_dict()
            working_task = {
                "task_id": d.get("task_id"),
                "task": d.get("task") or d.get("goal"),
                "goal": d.get("goal"),
                "state": d.get("state"),
                "plan": d.get("plan") or [],
                "current_step": d.get("current_step") or "",
                "done": d.get("done") or d.get("done_steps") or [],
                "done_steps": d.get("done_steps") or [],
                "open_questions": d.get("open_questions") or [],
                "artifacts": d.get("artifacts") or [],
                "vars": d.get("vars") or {},
                "updated_at": d.get("updated_at") or "",
            }
        long_ctx = self.long_term.retrieve(user_id=user_id, query=query or "", top_k=top_k)
        profile_compact = long_ctx.get("profile") or {}
        decisions = long_ctx.get("decisions") or []
        notes = long_ctx.get("notes") or []
        decisions_top_k = [self._format_entry_for_debug(e) for e in decisions]
        notes_top_k = [self._format_entry_for_debug(e) for e in notes]
        read_meta = dict(long_ctx.get("read_meta") or {})
        return {
            "short_term": short_snapshot,
            "working": {
                "present": working_present,
                "task": working_task,
            },
            "long_term": {
                "profile": profile_compact,
                "decisions_top_k": decisions_top_k,
                "notes_top_k": notes_top_k,
                "read_meta": read_meta,
            },
            "memory_writes": self.get_recent_write_events(session_id=session_id, limit=10),
        }

    def _is_start_execution_request(self, msg: str) -> bool:
        triggers = [
            "напиши код",
            "сразу код",
            "реализуй",
            "implementation",
            "код",
            "старт",
            "начнем исполнение",
            "перейди к выполнению",
            "start execution",
            "run plan",
            "сразу план",
            "финальный план",
            "план бюджета",
            "сразу рекомендации",
        ]
        return any(trigger in msg for trigger in triggers)

    def _is_plan_formation_message(self, msg: str) -> bool:
        if not msg:
            return False
        patterns = [
            r"\bсформируй план\b",
            r"\bсостав(?:ь|ьте)? план\b",
            r"\bразбей на шаги\b",
            r"\bкакие шаги\b",
            r"\bплан задачи\b",
            r"\bс чего начать\b",
            r"\bавтоматически\b",
            r"\bform plan\b",
            r"\bcreate plan\b",
        ]
        return any(re.search(pattern, msg, flags=re.IGNORECASE) for pattern in patterns)

    def _is_skip_request(self, msg: str) -> bool:
        if not msg:
            return False
        patterns = [
            r"\bдай финальн",
            r"\bсразу результат\b",
            r"\bпропусти планирован",
            r"\bskip\b",
            r"\bфинальный ответ сейчас\b",
            r"\bсразу к делу\b",
        ]
        return any(re.search(pattern, msg, flags=re.IGNORECASE) for pattern in patterns)

    def _is_goal_clarification_message(self, msg: str) -> bool:
        if not msg:
            return False
        patterns = [
            r"\bуточним цель\b",
            r"\bуточнить цель\b",
            r"\bцель задачи\b",
            r"\bуточни\b.*\bцель\b",
            r"\bclarify goal\b",
        ]
        return any(re.search(pattern, msg, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _plan_is_empty(ctx: TaskContext) -> bool:
        return len(list(ctx.plan or [])) == 0

    def _is_validation_request(self, msg: str) -> bool:
        triggers = [
            "валидац",
            "провер",
            "подтверди шаги",
            "request_validation",
            "перейди в validation",
        ]
        return any(trigger in msg for trigger in triggers)

    def _is_done_confirmation(self, msg: str) -> bool:
        triggers = ["подтверждаю завершение", "завершаем", "переведи в done", "confirm done", "done"]
        return any(trigger in msg for trigger in triggers)

    def _is_reexecution_request(self, msg: str) -> bool:
        triggers = ["верни в execution", "вернуться к шагам", "переисполн", "redo", "rollback"]
        return any(trigger in msg for trigger in triggers)

    def _is_execution_allowed_message(self, msg: str, current_step: str) -> bool:
        if not msg:
            return False
        step_lower = current_step.lower()
        allowed_markers = [
            "текущий шаг",
            "шаг выполнен",
            "выполнил",
            "выполнено",
            "артефакт",
            "статус",
            "progress",
            "done",
            "completed",
            "код",
            "реализуй",
        ]
        if step_lower and step_lower in msg:
            return True
        return any(marker in msg for marker in allowed_markers)
