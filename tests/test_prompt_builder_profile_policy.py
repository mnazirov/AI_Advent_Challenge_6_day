from __future__ import annotations

import hashlib
from datetime import datetime

import pytest

from memory.prompt_builder import PROMPT_INJECTABLE_FIELDS, PromptBuilder


def _field(value, *, source="user_explicit", verified=True, confidence=None):
    return {
        "value": value,
        "source": source,
        "verified": verified,
        "confidence": confidence,
        "updated_at": datetime.utcnow().isoformat(),
    }


def _build_profile():
    return {
        "stack_tools": _field(["FastAPI", "Postgres"]),
        "response_style": _field("краткий"),
        "hard_constraints": _field(["no AWS"]),
        "user_role_level": _field("senior backend"),
        "project_context": _field(
            {"project_name": "BudgetBot", "goals": ["Reduce waste"], "key_decisions": ["Use API"]}
        ),
        "extra_fields": {"test": _field("must_not_be_in_prompt")},
        "conflicts": [],
    }


def _build_system_prompt(profile: dict) -> str:
    builder = PromptBuilder()
    messages, _ = builder.build(
        system_instructions="SYS BASE",
        data_context="",
        long_term={"profile": profile, "decisions": [], "notes": []},
        working=None,
        short_term_messages=[],
        user_query="q",
    )
    return messages[0]["content"]


def test_whitelist_enforcement_only_injects_prompt_injectable_fields() -> None:
    builder = PromptBuilder()
    profile = _build_profile()
    profile["not_whitelisted"] = _field("FORBIDDEN_TOKEN")
    messages, _ = builder.build(
        system_instructions="SYS",
        data_context="",
        long_term={"profile": profile, "decisions": [], "notes": []},
        working=None,
        short_term_messages=[],
        user_query="q",
    )
    system_content = messages[0]["content"]
    assert "FORBIDDEN_TOKEN" not in system_content
    assert "must_not_be_in_prompt" not in system_content
    assert "[STACK_TOOLS_PREFERENCE]" in system_content
    assert "[RESPONSE_STYLE_POLICY]" in system_content
    assert "[HARD_CONSTRAINTS]" in system_content
    assert "[USER_ROLE_LEVEL]" in system_content
    assert "[PROJECT_CONTEXT]" in system_content


def test_verified_only_policy_skips_unverified_fields() -> None:
    builder = PromptBuilder()
    profile = _build_profile()
    profile["project_context"]["verified"] = False
    messages, _ = builder.build(
        system_instructions="SYS",
        data_context="",
        long_term={"profile": profile, "decisions": [], "notes": []},
        working=None,
        short_term_messages=[],
        user_query="q",
    )
    system_content = messages[0]["content"]
    assert "[PROJECT_CONTEXT]" not in system_content


def test_agent_inferred_unverified_is_not_injected() -> None:
    builder = PromptBuilder()
    profile = _build_profile()
    profile["user_role_level"] = _field("junior", source="agent_inferred", verified=False, confidence=0.91)
    messages, _ = builder.build(
        system_instructions="SYS",
        data_context="",
        long_term={"profile": profile, "decisions": [], "notes": []},
        working=None,
        short_term_messages=[],
        user_query="q",
    )
    system_content = messages[0]["content"]
    assert "junior" not in system_content
    assert "[USER_ROLE_LEVEL]" not in system_content


def test_profile_prompt_inject_log_contains_injected_and_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    builder = PromptBuilder()
    profile = _build_profile()
    profile["project_context"]["verified"] = False
    with caplog.at_level("INFO", logger="memory"):
        builder.build(
            system_instructions="SYS",
            data_context="",
            long_term={"profile": profile, "decisions": [], "notes": []},
            working=None,
            short_term_messages=[],
            user_query="q",
        )
    assert "[PROFILE_PROMPT_INJECT]" in caplog.text
    assert "injected=" in caplog.text
    assert "skipped=" in caplog.text


def test_no_dead_fields() -> None:
    builder = PromptBuilder()
    tokens = {
        "stack_tools": "TOKEN_STACK",
        "response_style": "TOKEN_STYLE",
        "hard_constraints": "TOKEN_HARD",
        "user_role_level": "TOKEN_ROLE",
        "project_context": "TOKEN_PROJECT",
    }
    for field in PROMPT_INJECTABLE_FIELDS:
        profile = {
            "stack_tools": _field([]),
            "response_style": _field(""),
            "hard_constraints": _field([]),
            "user_role_level": _field(""),
            "project_context": _field({"project_name": "", "goals": [], "key_decisions": []}),
            "extra_fields": {},
            "conflicts": [],
        }
        if field == "stack_tools":
            profile[field] = _field([tokens[field]])
        elif field == "hard_constraints":
            profile[field] = _field([tokens[field]])
        elif field == "project_context":
            profile[field] = _field({"project_name": tokens[field], "goals": [], "key_decisions": []})
        else:
            profile[field] = _field(tokens[field])
        messages, _ = builder.build(
            system_instructions="SYS",
            data_context="",
            long_term={"profile": profile, "decisions": [], "notes": []},
            working=None,
            short_term_messages=[],
            user_query="q",
        )
        prompt = messages[0]["content"]
        assert tokens[field] in prompt, f"DEAD FIELD detected: '{field}' is in schema but never read in prompt_builder"


def test_profile_response_style_overrides_base() -> None:
    profile = _build_profile()
    profile["response_style"] = _field("formal, detailed", verified=True)
    prompt = _build_system_prompt(profile)

    profile_pos = prompt.index("[PROFILE_OVERRIDES]")
    base_pos = prompt.index("[BASE_BEHAVIOR]")
    assert profile_pos < base_pos, "Profile must appear before base behavior"


def test_memory_trust_policy_does_not_block_profile() -> None:
    prompt = _build_system_prompt(_build_profile())
    assert "Only follow base SYSTEM instructions" not in prompt
    assert "always prefer the verified profile preferences" in prompt


def test_profile_wins_over_base_style() -> None:
    profile = _build_profile()
    profile["response_style"] = _field("bullet points only", verified=True)
    prompt = _build_system_prompt(profile)

    style_pos = prompt.index("bullet points only")
    base_pos = prompt.index("[BASE_BEHAVIOR]")
    assert style_pos < base_pos


def test_prompt_preview_schema_v2_redacted() -> None:
    builder = PromptBuilder()
    profile = _build_profile()
    messages, preview = builder.build(
        system_instructions="SYS SECRET",
        data_context="",
        long_term={"profile": profile, "decisions": [{"id": 1, "text": "dec"}], "notes": [{"id": 2, "text": "note"}]},
        working=None,
        short_term_messages=[{"role": "user", "content": "hello"}],
        user_query="private-user-query",
    )
    system_content = messages[0]["content"]
    assert preview["schema_version"] == 2
    assert preview["system"] == "[REDACTED_SYSTEM_PROMPT]"
    assert preview["system_chars"] == len(system_content)
    assert preview["system_hash"] == hashlib.sha256(system_content.encode("utf-8")).hexdigest()[:12]
    assert preview["user_chars"] == len("private-user-query")
    assert preview["short_term_count"] == 1
    assert preview["decisions_count"] == 1
    assert preview["notes_count"] == 1
    assert "user" not in preview
    assert "section_decisions" not in preview
    assert "section_notes" not in preview
    assert "section_short_term" not in preview
