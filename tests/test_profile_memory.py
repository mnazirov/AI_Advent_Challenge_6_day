from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import storage
from memory.long_term import LongTermMemory
from memory.models import ProfileSource


def _setup_temp_db() -> None:
    storage.DB_PATH = Path(tempfile.mktemp(suffix=".db"))
    storage.UPLOADS_DIR = Path(tempfile.mkdtemp())
    storage.init_db()


def test_profile_schema_shape_has_canonical_fields() -> None:
    _setup_temp_db()
    lt = LongTermMemory()
    profile = lt.get_profile(user_id="u")
    assert set(profile.keys()) >= {
        "stack_tools",
        "response_style",
        "hard_constraints",
        "user_role_level",
        "project_context",
        "extra_fields",
        "conflicts",
    }


def test_explicit_write_persists_after_reload_with_metadata() -> None:
    _setup_temp_db()
    lt = LongTermMemory()
    uid = "u_persist"
    lt.update_profile_field(uid, "response_style", "краткий", ProfileSource.USER_EXPLICIT)
    reloaded = LongTermMemory().get_profile(user_id=uid)
    assert reloaded["response_style"]["value"] == "краткий"
    assert reloaded["response_style"]["source"] == "user_explicit"
    assert reloaded["response_style"]["verified"] is True


def test_inferred_low_confidence_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    _setup_temp_db()
    lt = LongTermMemory()
    uid = "u_low_conf"
    before = lt.get_profile(user_id=uid)
    with caplog.at_level("INFO", logger="memory"):
        status = lt.update_profile_field(
            uid,
            "user_role_level",
            "senior backend",
            ProfileSource.AGENT_INFERRED,
            confidence=0.75,
        )
    after = lt.get_profile(user_id=uid)
    assert status == "skipped_low_confidence"
    assert before == after
    assert "[PROFILE_SKIP_LOW_CONFIDENCE]" in caplog.text


def test_inferred_defaults_to_unverified_and_confirm_sets_verified() -> None:
    _setup_temp_db()
    lt = LongTermMemory()
    uid = "u_inferred"
    lt.update_profile_field(
        uid,
        "user_role_level",
        "senior backend",
        ProfileSource.AGENT_INFERRED,
        confidence=0.9,
    )
    profile = lt.get_profile(user_id=uid)
    assert profile["user_role_level"]["verified"] is False
    assert profile["user_role_level"]["value"] == "senior backend"

    lt.confirm_profile_field(uid, "user_role_level")
    confirmed = lt.get_profile(user_id=uid)
    assert confirmed["user_role_level"]["verified"] is True
    assert confirmed["user_role_level"]["value"] == "senior backend"


def test_conflict_recorded_and_verified_value_not_overwritten() -> None:
    _setup_temp_db()
    lt = LongTermMemory()
    uid = "u_conflict"
    lt.update_profile_field(uid, "response_style", "краткий", ProfileSource.USER_EXPLICIT)
    status = lt.update_profile_field(
        uid,
        "response_style",
        "подробный",
        ProfileSource.AGENT_INFERRED,
        confidence=0.95,
    )
    profile = lt.get_profile(user_id=uid)
    assert status == "conflict_recorded"
    assert profile["response_style"]["value"] == "краткий"
    assert len(profile["conflicts"]) == 1


def test_resolve_conflict_with_chosen_value_and_keep_existing() -> None:
    _setup_temp_db()
    lt = LongTermMemory()
    uid = "u_resolve_conflict"

    lt.update_profile_field(uid, "response_style", "краткий", ProfileSource.USER_EXPLICIT)
    lt.update_profile_field(uid, "response_style", "подробный", ProfileSource.AGENT_INFERRED, confidence=0.95)
    lt.resolve_profile_conflict(uid, "response_style", chosen_value="подробный")
    profile = lt.get_profile(user_id=uid)
    assert profile["response_style"]["value"] == "подробный"
    assert profile["response_style"]["verified"] is True
    assert len(profile["conflicts"]) == 0

    lt.update_profile_field(uid, "response_style", "еще вариант", ProfileSource.AGENT_INFERRED, confidence=0.95)
    lt.resolve_profile_conflict(uid, "response_style", keep_existing=True)
    profile_keep = lt.get_profile(user_id=uid)
    assert profile_keep["response_style"]["value"] == "подробный"
    assert len(profile_keep["conflicts"]) == 0


def test_overwrite_semantics_response_style_replaces_value_without_growth() -> None:
    _setup_temp_db()
    lt = LongTermMemory()
    uid = "u_overwrite"

    sizes = []
    for idx in range(10):
        value = f"style_{idx:02d}"
        lt.update_profile_field(uid, "response_style", value, ProfileSource.USER_EXPLICIT)
        payload = storage.memory_load_longterm_profile(uid) or {}
        sizes.append(len(str(payload.get("response_style", {}).get("value", ""))))

    profile = lt.get_profile(user_id=uid)
    assert profile["response_style"]["value"] == "style_09"
    assert sizes[-1] == sizes[0]


def test_budget_overflow_raises_exact_message() -> None:
    _setup_temp_db()
    lt = LongTermMemory()
    uid = "u_budget"
    with pytest.raises(ValueError, match=r"(?s)Profile budget exceeded: .*Cannot add field 'extra_"):
        for idx in range(100):
            lt.add_profile_extra_field(
                uid,
                f"extra_{idx}",
                "x" * 300,
                ProfileSource.DEBUG_MENU,
            )


def test_limits_and_truncation_are_enforced() -> None:
    _setup_temp_db()
    lt = LongTermMemory()
    uid = "u_limits"
    with pytest.raises(ValueError, match="stack_tools exceeds limit of 12"):
        lt.update_profile_field(
            uid,
            "stack_tools",
            [f"tool_{i}" for i in range(13)],
            ProfileSource.USER_EXPLICIT,
        )

    with pytest.raises(ValueError, match="hard_constraints exceeds limit of 20"):
        lt.update_profile_field(
            uid,
            "hard_constraints",
            [f"c_{i}" for i in range(21)],
            ProfileSource.USER_EXPLICIT,
        )

    long_text = "x" * 500
    lt.update_profile_field(uid, "response_style", long_text, ProfileSource.USER_EXPLICIT)
    profile = lt.get_profile(user_id=uid)
    assert len(profile["response_style"]["value"]) == 300
