from __future__ import annotations

import logging
import re

from memory.long_term import LongTermMemory
from memory.models import MemoryWriteEvent
from memory.working import WorkingMemory

logger = logging.getLogger("memory")


class MemoryRouter:
    PROFILE_PATTERNS = [
        r"\bвсегда\b",
        r"с этого момента",
        r"отвечай кратко",
        r"пиши на",
        r"не используй",
    ]
    DECISION_PATTERNS = [r"\bвыбираем\b", r"\bрешили\b", r"\bиспользуем\b", r"\bстандарт\b", r"\bдоговорились\b"]
    WORKING_PATTERNS = [r"\bтребован", r"\bэндпоинт", r"добавь шаг", r"сделаем так", r"текущий шаг", r"план"]
    NOTE_PATTERNS = [r"\bзапомни\b", r"\bважно\b", r"\bучти\b", r"\bфакт\b"]

    def route_user_message(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        working: WorkingMemory,
        long_term: LongTermMemory,
    ) -> list[MemoryWriteEvent]:
        text = (user_message or "").strip()
        lower = text.lower()
        events: list[MemoryWriteEvent] = []

        confirm_match = re.search(r"подтверждаю память\s*#\s*(\d+)", lower, flags=re.IGNORECASE)
        if confirm_match:
            pending_id = int(confirm_match.group(1))
            approved = long_term.approve_pending_entry(user_id=user_id, pending_id=pending_id)
            if approved:
                events.append(MemoryWriteEvent(layer="long_term.approval", keys=["pending_id", "type"]))
            else:
                logger.info("[MEMORY_WRITE] layer=long_term.approval status=not_found pending_id=%s", pending_id)
            return events

        if self._matches_any(lower, self.PROFILE_PATTERNS):
            style = ""
            constraints: list[str] = []
            context: list[str] = []
            tags = ["profile"]

            if "кратко" in lower:
                style = "concise"
                tags.append("concise")
            if "подроб" in lower:
                style = "detailed"
                tags.append("detailed")
            if "на русском" in lower:
                constraints.append("language:ru")
            if "на англий" in lower:
                constraints.append("language:en")

            not_use_match = re.search(r"не используй([^.!?]+)", lower)
            if not_use_match:
                constraints.append("avoid:" + not_use_match.group(1).strip())
            context.append(text)

            long_term.upsert_profile(
                user_id=user_id,
                style=style or None,
                constraints=constraints,
                context=context,
                tags=tags,
                source="user",
            )
            events.append(MemoryWriteEvent(layer="long_term.profile", keys=["style", "constraints", "context"]))

        if self._matches_any(lower, self.DECISION_PATTERNS):
            tags = ["decision"]
            if "стандарт" in lower:
                tags.append("standard")
            result = long_term.add_decision(user_id=user_id, text=text, tags=tags, source="user")
            if result.get("status") == "pending":
                events.append(MemoryWriteEvent(layer="long_term.pending", keys=["pending_id"]))
            else:
                events.append(MemoryWriteEvent(layer="long_term.decision", keys=["text", "tags"]))

        if self._matches_any(lower, self.NOTE_PATTERNS):
            tags = ["note", "stability"]
            result = long_term.add_note(
                user_id=user_id,
                text=text,
                tags=tags,
                source="user",
                ttl_days=90,
            )
            if result.get("status") == "pending":
                events.append(MemoryWriteEvent(layer="long_term.pending", keys=["pending_id"]))
            else:
                events.append(MemoryWriteEvent(layer="long_term.note", keys=["text", "tags", "ttl_days"]))

        if self._matches_any(lower, self.WORKING_PATTERNS):
            ctx = working.ensure_task(session_id=session_id, goal="Текущая задача")
            vars_patch = dict(ctx.vars)
            requirements = list(vars_patch.get("requirements") or [])
            artifacts = list(ctx.artifacts)
            plan = list(ctx.plan)
            current_step = ctx.current_step

            if "требован" in lower or "эндпоинт" in lower:
                requirements.append(text)
                vars_patch["requirements"] = requirements

            step_match = re.search(r"добавь шаг[:\s]+(.+)$", text, flags=re.IGNORECASE)
            if step_match:
                step = step_match.group(1).strip()
                if step and step not in plan:
                    plan.append(step)

            current_step_match = re.search(r"текущий шаг[:\s]+(.+)$", text, flags=re.IGNORECASE)
            if current_step_match:
                current_step = current_step_match.group(1).strip()

            if "артефакт" in lower:
                artifacts.append(text)

            working.update(
                session_id,
                goal=ctx.goal or "Текущая задача",
                vars=vars_patch,
                plan=plan,
                current_step=current_step,
                artifacts=artifacts,
            )
            events.append(MemoryWriteEvent(layer="working", keys=["vars", "plan", "current_step", "artifacts"]))

        if not events:
            logger.info("[MEMORY_WRITE] layer=none reason=no_policy_match")
        else:
            for event in events:
                logger.info("[MEMORY_WRITE] layer=%s keys=%s", event.layer, ",".join(event.keys))

        return events

    def _matches_any(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
