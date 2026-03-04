from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import storage
from memory.models import ArtifactType, TaskState
from memory.working import WorkingMemory


def _setup_temp_db() -> None:
    storage.DB_PATH = Path(tempfile.mktemp(suffix=".db"))
    storage.UPLOADS_DIR = Path(tempfile.mkdtemp())
    storage.init_db()


def _build_execution_context() -> tuple[WorkingMemory, str]:
    _setup_temp_db()
    wm = WorkingMemory()
    sid = storage.create_session()
    wm.start_task(session_id=sid, goal="Цель")
    wm.update(session_id=sid, plan=["A", "B"], current_step="A")
    ctx = wm.load(sid)
    assert ctx is not None
    wm.transition_state(ctx, TaskState.EXECUTION)
    ctx.updated_at = "2026-03-04T00:00:00"
    wm.save(ctx)
    return wm, sid


def test_forbidden_transition_message_is_exact() -> None:
    _setup_temp_db()
    wm = WorkingMemory()
    sid = storage.create_session()
    ctx = wm.start_task(session_id=sid, goal="Запрет")
    with pytest.raises(ValueError, match="Transition PLANNING -> DONE is forbidden"):
        wm.transition_state(ctx, TaskState.DONE)


def test_planning_to_execution_requires_plan_and_first_step() -> None:
    _setup_temp_db()
    wm = WorkingMemory()
    sid = storage.create_session()
    ctx = wm.start_task(session_id=sid, goal="Переход")

    with pytest.raises(ValueError, match="plan is empty"):
        wm.transition_state(ctx, TaskState.EXECUTION)

    wm.update(session_id=sid, plan=["S1"], current_step="S1")
    ctx = wm.load(sid)
    assert ctx is not None
    ctx.current_step = "S2"
    with pytest.raises(ValueError, match="current_step must equal plan\\[0\\]"):
        wm.transition_state(ctx, TaskState.EXECUTION)

    wm.update(session_id=sid, current_step="S1")
    ctx = wm.load(sid)
    assert ctx is not None
    wm.transition_state(ctx, TaskState.EXECUTION)
    assert ctx.state == TaskState.EXECUTION


def test_execution_to_validation_requires_done_equal_plan_in_order() -> None:
    wm, sid = _build_execution_context()
    ctx = wm.load(sid)
    assert ctx is not None
    with pytest.raises(ValueError, match="done != plan: not all steps completed in order"):
        wm.transition_state(ctx, TaskState.VALIDATION)

    wm.complete_current_step(sid)
    wm.complete_current_step(sid)
    ctx = wm.request_validation(sid)
    assert ctx.state == TaskState.VALIDATION


def test_request_validation_fails_with_short_message_when_done_not_full() -> None:
    wm, sid = _build_execution_context()
    wm.complete_current_step(sid)
    with pytest.raises(ValueError, match="done != plan"):
        wm.request_validation(sid)


def test_rollbacks_preserve_artifacts_and_open_questions() -> None:
    wm, sid = _build_execution_context()
    wm.append_artifact_for_current_step(
        sid,
        artifact={"step": "A", "type": ArtifactType.FILE.value, "ref": "/tmp/report.csv"},
    )
    wm.update(session_id=sid, open_questions=["Q1"])

    ctx = wm.load(sid)
    assert ctx is not None
    wm.transition_state(ctx, TaskState.PLANNING)
    wm.save(ctx)
    assert ctx.current_step is None
    assert ctx.done == []
    assert len(ctx.artifacts) == 1
    assert ctx.open_questions == ["Q1"]

    wm.update(session_id=sid, plan=["A", "B"], current_step="A")
    ctx = wm.load(sid)
    assert ctx is not None
    wm.transition_state(ctx, TaskState.EXECUTION)
    wm.save(ctx)
    wm.complete_current_step(sid)
    wm.complete_current_step(sid)
    wm.request_validation(sid)

    ctx = wm.load(sid)
    assert ctx is not None
    wm.transition_state(ctx, TaskState.EXECUTION)
    assert ctx.current_step == "B"
    assert len(ctx.artifacts) == 1
    assert ctx.open_questions == ["Q1"]


def test_done_is_terminal_and_frozen_after_reload() -> None:
    wm, sid = _build_execution_context()
    wm.complete_current_step(sid)
    wm.complete_current_step(sid)
    wm.request_validation(sid)
    ctx = wm.load(sid)
    assert ctx is not None
    wm.transition_state(ctx, TaskState.DONE)
    wm.save(ctx)

    reloaded = wm.load(sid)
    assert reloaded is not None
    assert reloaded.state == TaskState.DONE
    assert reloaded.current_step is None
    with pytest.raises(ValueError, match="Working memory is frozen in DONE state"):
        wm.update(session_id=sid, open_questions=["new"])


def test_complete_current_step_updates_one_by_one() -> None:
    wm, sid = _build_execution_context()
    first = wm.complete_current_step(sid)
    assert first.done == ["A"]
    assert first.current_step == "B"
    second = wm.complete_current_step(sid)
    assert second.done == ["A", "B"]
    assert second.current_step is None

    with pytest.raises(ValueError, match="current_step '' not found in plan"):
        wm.complete_current_step(sid)


def test_integrity_checks_duplicates_and_current_step_membership() -> None:
    _setup_temp_db()
    wm = WorkingMemory()
    sid = storage.create_session()
    wm.start_task(session_id=sid, goal="Проверка")

    with pytest.raises(ValueError, match="duplicate step in plan: 'A'"):
        wm.update(session_id=sid, plan=["A", "A"])

    wm.update(session_id=sid, plan=["A", "B"])
    with pytest.raises(ValueError, match="current_step 'Z' not found in plan"):
        wm.update(session_id=sid, current_step="Z")

    with pytest.raises(ValueError, match="duplicate entry in done: 'A'"):
        wm.update(session_id=sid, done=["A", "A"])


def test_cannot_bulk_complete_via_update_in_execution() -> None:
    wm, sid = _build_execution_context()
    with pytest.raises(ValueError, match="done can only be mutated via complete_current_step\\(\\)"):
        wm.update(session_id=sid, done=["A", "B"])


def test_persistence_roundtrip_preserves_state_current_step_and_done_order() -> None:
    wm, sid = _build_execution_context()
    wm.complete_current_step(sid)
    ctx = wm.load(sid)
    assert ctx is not None
    assert ctx.state == TaskState.EXECUTION
    assert ctx.current_step == "B"
    assert ctx.done == ["A"]

    again = wm.load(sid)
    assert again is not None
    assert again.state == TaskState.EXECUTION
    assert again.current_step == "B"
    assert again.done == ["A"]


def test_legacy_artifacts_are_migrated_to_object_schema() -> None:
    _setup_temp_db()
    sid = storage.create_session()
    storage.memory_save_working_task(
        session_id=sid,
        task_id="task-legacy-artifacts",
        goal="Legacy",
        state="PLANNING",
        plan=["A"],
        current_step="A",
        done_steps=[],
        open_questions=[],
        artifacts=["raw://artifact-1", "raw://artifact-2"],
        vars_data={},
    )
    wm = WorkingMemory()
    ctx = wm.load(sid)
    assert ctx is not None
    assert len(ctx.artifacts) == 2
    assert ctx.artifacts[0].type == ArtifactType.LEGACY
    assert ctx.artifacts[0].ref == "raw://artifact-1"
