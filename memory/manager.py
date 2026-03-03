from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
import logging

from memory.long_term import LongTermMemory
from memory.models import MemoryWriteEvent, TaskContext, TaskState
from memory.prompt_builder import PromptBuilder
from memory.router import MemoryRouter
from memory.short_term import ShortTermMemory
from memory.working import WorkingMemory

logger = logging.getLogger("memory")

PREVIEW_LENGTH = 120
MAX_DEBUG_TEXT_CHARS = 4000
MAX_WRITE_EVENTS = 50


class MemoryManager:
    def __init__(self, short_term_limit: int = 30):
        self.short_term = ShortTermMemory(limit_n=short_term_limit)
        self.working = WorkingMemory()
        self.long_term = LongTermMemory()
        self.router = MemoryRouter()
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

    def hydrate_short_term(self, session_id: str, messages: list[dict[str, str]]) -> None:
        self.short_term.hydrate(session_id=session_id, messages=messages)

    def enforce_planning_gate(self, *, session_id: str, user_message: str) -> str | None:
        ctx: TaskContext | None = self.working.load(session_id)
        if not ctx:
            return None

        wants_direct_solution = any(
            trigger in user_message.lower()
            for trigger in ["напиши код", "сразу код", "реализуй", "implementation", "код"]
        )
        wants_direct_solution = wants_direct_solution or any(
            trigger in user_message.lower()
            for trigger in ["сразу план", "финальный план", "план бюджета", "без шагов", "сразу рекомендации"]
        )
        if ctx.state == TaskState.PLANNING and wants_direct_solution:
            if ctx.plan and ctx.current_step:
                self.working.update(session_id, state=TaskState.EXECUTION.value)
                return None
            return (
                "Сейчас задача находится в состоянии PLANNING, поэтому я не перехожу к финальному решению сразу. "
                "Сначала зафиксируем план шагов и текущий шаг, затем переведём задачу в EXECUTION."
            )
        return None

    def stats(self, *, session_id: str, user_id: str) -> dict:
        working = self.working.load(session_id)
        longterm = self.long_term.retrieve(user_id=user_id, query="", top_k=3)
        read_meta = dict(longterm.get("read_meta") or {})
        return {
            "short_term_messages": len(self.short_term.get_context(session_id)),
            "working_state": working.state.value if working else None,
            "working_task_id": working.task_id if working else None,
            "longterm_profile": bool(longterm.get("profile")),
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
                "goal": d.get("goal"),
                "state": d.get("state"),
                "plan": d.get("plan") or [],
                "current_step": d.get("current_step") or "",
                "done_steps": d.get("done_steps") or [],
                "open_questions": d.get("open_questions") or [],
                "artifacts": d.get("artifacts") or [],
                "vars": d.get("vars") or {},
                "updated_at": d.get("updated_at") or "",
            }
        long_ctx = self.long_term.retrieve(user_id=user_id, query=query or "", top_k=top_k)
        profile = long_ctx.get("profile") or {}
        profile_compact = {
            "style": profile.get("style") or "",
            "constraints": list(profile.get("constraints") or []),
            "context": list(profile.get("context") or []),
            "tags": list(profile.get("tags") or []),
        }
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
