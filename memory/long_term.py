from __future__ import annotations

import re

import storage


class LongTermMemory:
    def upsert_profile(
        self,
        user_id: str,
        *,
        style: str | None = None,
        constraints: list[str] | None = None,
        context: list[str] | None = None,
        tags: list[str] | None = None,
        source: str = "user",
    ) -> dict:
        if source not in {"user", "assistant_confirmed"}:
            raise ValueError("Long-term profile updates require source=user or assistant_confirmed")
        existing = storage.memory_load_longterm_profile(user_id=user_id) or {}
        merged_constraints = list(existing.get("constraints") or [])
        merged_context = list(existing.get("context") or [])
        merged_tags = list(existing.get("tags") or [])

        if constraints:
            for item in constraints:
                if item and item not in merged_constraints:
                    merged_constraints.append(item)
        if context:
            for item in context:
                if item and item not in merged_context:
                    merged_context.append(item)
        if tags:
            for item in tags:
                if item and item not in merged_tags:
                    merged_tags.append(item)

        payload = {
            "style": style or existing.get("style") or "",
            "constraints": merged_constraints,
            "context": merged_context,
            "tags": merged_tags,
            "source": source,
        }
        storage.memory_upsert_longterm_profile(user_id=user_id, **payload)
        return payload

    def add_decision(
        self,
        user_id: str,
        text: str,
        tags: list[str] | None = None,
        source: str = "user",
        ttl_days: int | None = None,
    ) -> dict:
        if source == "assistant":
            pending_id = storage.memory_add_longterm_pending(
                user_id=user_id,
                entry_type="decision",
                text=text,
                tags=tags or [],
                source=source,
                ttl_days=ttl_days,
            )
            return {"status": "pending", "pending_id": pending_id}
        if source not in {"user", "assistant_confirmed"}:
            raise ValueError("Unsupported source for long-term decision")
        storage.memory_add_longterm_decision(
            user_id=user_id,
            text=text,
            tags=tags or [],
            source=source,
            ttl_days=ttl_days,
            entry_type="decision",
        )
        return {"status": "committed"}

    def add_note(
        self,
        user_id: str,
        text: str,
        tags: list[str] | None = None,
        source: str = "user",
        ttl_days: int | None = 90,
    ) -> dict:
        if source == "assistant":
            pending_id = storage.memory_add_longterm_pending(
                user_id=user_id,
                entry_type="note",
                text=text,
                tags=tags or [],
                source=source,
                ttl_days=ttl_days,
            )
            return {"status": "pending", "pending_id": pending_id}
        if source not in {"user", "assistant_confirmed"}:
            raise ValueError("Unsupported source for long-term note")
        storage.memory_add_longterm_note(
            user_id=user_id,
            text=text,
            tags=tags or [],
            source=source,
            ttl_days=ttl_days,
            entry_type="note",
        )
        return {"status": "committed"}

    def retrieve(self, user_id: str, query: str, top_k: int = 3) -> dict:
        profile = storage.memory_load_longterm_profile(user_id=user_id) or {}
        decisions = storage.memory_list_longterm_decisions(user_id=user_id, limit=100)
        notes = storage.memory_list_longterm_notes(user_id=user_id, limit=100)

        q_tokens = self._tokenize(query)
        top = max(1, int(top_k))
        ranked_decisions = sorted(decisions, key=lambda d: self._score_entry(d, q_tokens), reverse=True)[:top]
        ranked_notes = sorted(notes, key=lambda n: self._score_entry(n, q_tokens), reverse=True)[:top]

        decision_ids = [int(d["id"]) for d in ranked_decisions]
        note_ids = [int(n["id"]) for n in ranked_notes]

        return {
            "profile": profile,
            "decisions": ranked_decisions,
            "notes": ranked_notes,
            "read_meta": {
                "top_k": top,
                "decision_hits": len(ranked_decisions),
                "note_hits": len(ranked_notes),
                "decision_ids": decision_ids,
                "note_ids": note_ids,
                "decision_reason": "match/score",
                "note_reason": "match/score",
            },
        }

    def delete_decision(self, *, user_id: str, decision_id: int) -> bool:
        """Удаляет решение long-term памяти пользователя."""
        return storage.memory_delete_longterm_decision(user_id=user_id, decision_id=int(decision_id)) > 0

    def delete_note(self, *, user_id: str, note_id: int) -> bool:
        """Удаляет заметку long-term памяти пользователя."""
        return storage.memory_delete_longterm_note(user_id=user_id, note_id=int(note_id)) > 0

    def propose_assistant_entry(
        self,
        *,
        user_id: str,
        entry_type: str,
        text: str,
        tags: list[str] | None = None,
        ttl_days: int | None = None,
    ) -> int:
        return storage.memory_add_longterm_pending(
            user_id=user_id,
            entry_type=entry_type,
            text=text,
            tags=tags or [],
            source="assistant",
            ttl_days=ttl_days,
        )

    def approve_pending_entry(self, *, user_id: str, pending_id: int) -> dict | None:
        pending = storage.memory_get_pending_by_id(user_id=user_id, pending_id=pending_id)
        if not pending or pending.get("status") != "pending":
            return None
        entry_type = str(pending.get("type") or "note")
        text = str(pending.get("text") or "")
        tags = list(pending.get("tags") or [])
        ttl_days = pending.get("ttl_days")

        if entry_type == "decision":
            self.add_decision(
                user_id=user_id,
                text=text,
                tags=tags,
                source="assistant_confirmed",
                ttl_days=ttl_days,
            )
        elif entry_type == "profile":
            self.upsert_profile(
                user_id=user_id,
                context=[text],
                tags=tags,
                source="assistant_confirmed",
            )
        else:
            self.add_note(
                user_id=user_id,
                text=text,
                tags=tags,
                source="assistant_confirmed",
                ttl_days=ttl_days if ttl_days is not None else 90,
            )

        storage.memory_mark_pending_approved(user_id=user_id, pending_id=pending_id)
        return {"status": "approved", "pending_id": pending_id, "type": entry_type}

    def _score_entry(self, entry: dict, q_tokens: set[str]) -> tuple[int, int]:
        text = str(entry.get("text") or "")
        tags = entry.get("tags") or []
        tokens = self._tokenize(text + " " + " ".join(str(t) for t in tags))
        overlap = len(q_tokens.intersection(tokens))
        tag_overlap = len(q_tokens.intersection({str(t).lower() for t in tags}))
        like_bonus = 0
        lowered = text.lower()
        if q_tokens:
            like_bonus = sum(1 for token in q_tokens if token in lowered)
        created_at = str(entry.get("created_at") or "")
        recency = int("".join(ch for ch in created_at if ch.isdigit())[:8] or 0)
        return overlap + tag_overlap + like_bonus, recency

    def _tokenize(self, text: str) -> set[str]:
        raw = re.split(r"[^\wа-яА-ЯёЁ]+", (text or "").lower())
        return {t for t in raw if len(t) > 2}
