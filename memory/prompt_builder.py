from __future__ import annotations

from memory.models import TaskContext


class PromptBuilder:
    def build(
        self,
        *,
        system_instructions: str,
        data_context: str,
        long_term: dict,
        working: TaskContext | None,
        short_term_messages: list[dict[str, str]],
        user_query: str,
    ) -> tuple[list[dict[str, str]], dict[str, str]]:
        sections: list[str] = [system_instructions.strip()]

        if data_context.strip():
            sections.append("[DATA_CONTEXT]\n" + data_context.strip())

        long_blocks: list[str] = []
        profile = long_term.get("profile") or {}
        constraints = list(profile.get("constraints") or [])[:3]
        context_items = list(profile.get("context") or [])[:3]
        profile_style = profile.get("style") or ""
        long_blocks.append(
            "[LONG_TERM_PROFILE]\n"
            f"style={profile_style}\n"
            f"constraints={constraints}\n"
            f"context={context_items}"
        )
        profile_preview = (
            f"[PROFILE] style={profile_style} "
            f"constraints={constraints} context={context_items}"
        )[:220]

        decisions = long_term.get("decisions") or []
        decision_preview_lines: list[str] = []
        if decisions:
            lines = []
            for d in decisions:
                text = str(d.get("text") or "").replace("\n", " ").strip()
                if not text:
                    continue
                text_short = text[:160]
                lines.append(f"- [{d.get('id')}] {text_short}")
                decision_preview_lines.append(
                    f"- {text[:80]} | tags={d.get('tags') or []}"
                )
            if lines:
                long_blocks.append("[LONG_TERM_DECISIONS]\n" + "\n".join(lines))

        notes = long_term.get("notes") or []
        note_preview_lines: list[str] = []
        if notes:
            lines = []
            for n in notes:
                text = str(n.get("text") or "").replace("\n", " ").strip()
                if not text:
                    continue
                text_short = text[:160]
                lines.append(f"- [{n.get('id')}] {text_short}")
                note_preview_lines.append(
                    f"- {text[:80]} | tags={n.get('tags') or []}"
                )
            if lines:
                long_blocks.append("[LONG_TERM_NOTES]\n" + "\n".join(lines))

        sections.append("\n\n".join(long_blocks))

        working_preview = "[WORKING] none"
        if working:
            sections.append(
                "[WORKING_TASK]\n"
                f"task_id={working.task_id}\n"
                f"goal={working.goal}\n"
                f"state={working.state.value}\n"
                f"plan={working.plan}\n"
                f"current_step={working.current_step}\n"
                f"done_steps={working.done_steps}\n"
                f"open_questions={working.open_questions}\n"
                f"artifacts={working.artifacts}\n"
                f"vars={working.vars}"
            )
            working_preview = (
                f"[WORKING] state={working.state.value} plan={working.plan} "
                f"current_step={working.current_step} done={working.done_steps} "
                f"open={working.open_questions}"
            )[:220]

        system_content = "\n\n".join(s for s in sections if s)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        if short_term_messages:
            messages.extend(short_term_messages)
        short_preview = "[SHORT_TERM] " + " | ".join(
            f"{m.get('role')}:{str(m.get('content') or '')[:80]}" for m in short_term_messages[-5:]
        )
        short_preview = short_preview[:220]

        messages.append({"role": "user", "content": user_query})

        preview = {
            "system": system_instructions[:220],
            "long_term": "yes",
            "working": working.state.value if working else "none",
            "short_term_count": str(len(short_term_messages)),
            "user": user_query[:220],
            "section_profile": profile_preview or "[PROFILE] style= constraints=[] context=[]",
            "section_decisions": ("[DECISIONS]\n" + "\n".join(decision_preview_lines[:3]))[:220]
            if decision_preview_lines
            else "[DECISIONS] none",
            "section_notes": ("[NOTES]\n" + "\n".join(note_preview_lines[:3]))[:220]
            if note_preview_lines
            else "[NOTES] none",
            "section_working": working_preview,
            "section_short_term": short_preview if short_term_messages else "[SHORT_TERM] none",
        }
        return messages, preview
