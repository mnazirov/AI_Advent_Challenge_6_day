"""
Интеграционные тесты для GET /debug/memory-layers.
Требуют установленных зависимостей проекта (pandas, flask, …).
Запуск с venv: pytest tests/test_debug_memory_layers.py -v
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

import storage
from memory.models import ProfileSource

# Временная БД до импорта app
storage.DB_PATH = Path(tempfile.mktemp(suffix=".db"))
storage.UPLOADS_DIR = Path(tempfile.mkdtemp())
storage.init_db()

try:
    from app import app  # noqa: E402
    _app_available = True
    _app_import_error = None
except Exception as e:
    app = None
    _app_available = False
    _app_import_error = e


def _requires_app():
    if not _app_available:
        pytest.skip(f"app not available (install deps): {_app_import_error}")


def _cookie_headers(session_id: str, user_id: str) -> dict[str, str]:
    return {"Cookie": f"session_id={session_id}; user_id={user_id}"}


def test_debug_memory_layers_endpoint_200() -> None:
    """Эндпоинт возвращает 200 и структуру всех трёх слоёв."""
    _requires_app()
    client = app.test_client()
    rv = client.get("/debug/memory-layers")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data is not None
    assert "short_term" in data
    assert "working" in data
    assert "long_term" in data
    st = data["short_term"]
    assert "limit_n" in st
    assert "turns_count" in st
    assert "turns" in st
    assert isinstance(st["turns"], list)
    wk = data["working"]
    assert "present" in wk
    assert "task" in wk
    lt = data["long_term"]
    assert "profile" in lt
    assert "decisions_top_k" in lt
    assert "notes_top_k" in lt
    assert "read_meta" in lt
    assert "memory_writes" in data
    assert isinstance(data["memory_writes"], list)


def test_debug_memory_layers_endpoint_top_k_clamp() -> None:
    """top_k ограничивается диапазоном 1..10."""
    _requires_app()
    client = app.test_client()
    rv = client.get("/debug/memory-layers?top_k=0")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data is not None
    meta = data.get("long_term", {}).get("read_meta", {})
    assert meta.get("top_k") == 1

    rv2 = client.get("/debug/memory-layers?top_k=999")
    assert rv2.status_code == 200
    data2 = rv2.get_json()
    assert data2 is not None
    meta2 = data2.get("long_term", {}).get("read_meta", {})
    assert meta2.get("top_k") == 10


def test_debug_clear_working_memory_only() -> None:
    """Очистка working через debug-эндпоинт не трогает short-term и long-term."""
    _requires_app()
    client = app.test_client()
    sid = storage.create_session()
    uid = "debug_user_clear_working"
    storage.memory_append_short_term_message(
        sid, "user", "short-turn", datetime.utcnow().isoformat()
    )
    storage.memory_save_working_task(
        session_id=sid,
        task_id="task-debug-1",
        goal="Проверка очистки",
        state="PLANNING",
        plan=["Шаг 1"],
        current_step="Шаг 1",
        done_steps=[],
        open_questions=[],
        artifacts=[],
        vars_data={"k": "v"},
        updated_at=datetime.utcnow().isoformat(),
    )
    storage.memory_add_longterm_decision(user_id=uid, text="Сохраняем long-term", tags=["debug"])

    rv = client.post("/debug/memory/working/clear", headers=_cookie_headers(sid, uid), json={})
    assert rv.status_code == 200
    data = rv.get_json()
    assert data is not None
    assert data["success"] is True
    assert data["snapshot"]["working"]["present"] is False
    assert data["snapshot"]["short_term"]["turns_count"] >= 1
    assert len(data["snapshot"]["long_term"]["decisions_top_k"]) >= 1


def test_debug_delete_long_term_entry() -> None:
    """Удаление long-term записи удаляет только целевой entry и сразу обновляет snapshot."""
    _requires_app()
    client = app.test_client()
    sid = storage.create_session()
    uid = "debug_user_delete_lt"
    storage.memory_append_short_term_message(
        sid, "user", "short-turn", datetime.utcnow().isoformat()
    )
    storage.memory_save_working_task(
        session_id=sid,
        task_id="task-debug-2",
        goal="Проверка удаления",
        state="PLANNING",
        plan=["Шаг 1"],
        current_step="Шаг 1",
        done_steps=[],
        open_questions=[],
        artifacts=[],
        vars_data={},
        updated_at=datetime.utcnow().isoformat(),
    )
    storage.memory_add_longterm_note(user_id=uid, text="Удаляемая заметка", tags=["debug-note"])
    notes = storage.memory_list_longterm_notes(uid, limit=10)
    assert notes
    note_id = int(notes[0]["id"])

    rv = client.post(
        "/debug/memory/long-term/delete",
        headers=_cookie_headers(sid, uid),
        json={"entry_type": "note", "id": note_id},
    )
    assert rv.status_code == 200
    data = rv.get_json()
    assert data is not None
    assert data["success"] is True
    assert data["deleted"] is True
    snapshot = data["snapshot"]
    assert all(int(n["id"]) != note_id for n in snapshot["long_term"]["notes_top_k"])
    assert snapshot["working"]["present"] is True
    assert snapshot["short_term"]["turns_count"] >= 1


def test_session_restore_includes_working_view() -> None:
    _requires_app()
    client = app.test_client()
    sid = storage.create_session()
    storage.save_message(sid, "user", "hi")
    storage.memory_save_working_task(
        session_id=sid,
        task_id="task-restore-working-view",
        goal="Тест working_view",
        state="PLANNING",
        plan=["Шаг 1"],
        current_step="Шаг 1",
        done_steps=[],
        open_questions=[],
        artifacts=[],
        vars_data={},
        updated_at=datetime.utcnow().isoformat(),
    )
    rv = client.get("/session/restore", headers=_cookie_headers(sid, "debug_user_restore"))
    assert rv.status_code == 200
    payload = rv.get_json()
    assert payload is not None
    assert payload.get("found") is True
    assert "working_view" in payload
    assert payload["working_view"]["state"] == "PLANNING"
    assert payload["working_view"]["plan"] == ["Шаг 1"]


def test_chat_response_includes_working_view() -> None:
    _requires_app()
    import app as app_module

    client = app.test_client()
    sid = storage.create_session()
    uid = "debug_user_chat_working_view"

    original_chat = app_module.agent.chat
    original_token_stats = app_module.agent.last_token_stats
    original_memory_stats = app_module.agent.last_memory_stats
    original_prompt_preview = app_module.agent.last_prompt_preview
    try:
        def _fake_chat(message: str, session_id: str | None = None, user_id: str | None = None) -> str:
            del message, user_id
            if session_id:
                app_module.agent.memory.working.start_task(session_id=session_id, goal="Тест чата")
            app_module.agent.last_token_stats = {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "cost_usd": 0.0,
                "latency_ms": 1,
            }
            app_module.agent.last_memory_stats = {}
            app_module.agent.last_prompt_preview = {}
            return "ok"

        app_module.agent.chat = _fake_chat
        rv = client.post(
            "/chat",
            headers=_cookie_headers(sid, uid),
            json={"message": "test"},
        )
        assert rv.status_code == 200
        payload = rv.get_json()
        assert payload is not None
        assert "working_view" in payload
        assert payload["working_view"]["state"] in {"PLANNING", "EXECUTION", "VALIDATION", "DONE", None}
    finally:
        app_module.agent.chat = original_chat
        app_module.agent.last_token_stats = original_token_stats
        app_module.agent.last_memory_stats = original_memory_stats
        app_module.agent.last_prompt_preview = original_prompt_preview


def test_index_contains_main_task_state_panel() -> None:
    _requires_app()
    client = app.test_client()
    rv = client.get("/")
    assert rv.status_code == 200
    html = rv.get_data(as_text=True)
    assert 'id="taskStateBadge"' in html
    assert 'id="taskStepCounter"' in html
    assert "Шаг 0 из 0" in html


def test_debug_profile_endpoints_crud_and_confirm() -> None:
    _requires_app()
    import app as app_module

    client = app.test_client()
    sid = storage.create_session()
    uid = "debug_profile_user"

    rv_get = client.get("/debug/memory/profile", headers=_cookie_headers(sid, uid))
    assert rv_get.status_code == 200
    profile = (rv_get.get_json() or {}).get("profile") or {}
    assert "response_style" in profile

    rv_patch = client.patch(
        "/debug/memory/profile/field",
        headers=_cookie_headers(sid, uid),
        json={"field": "response_style", "value": "краткий"},
    )
    assert rv_patch.status_code == 200
    profile_after_patch = (rv_patch.get_json() or {}).get("profile") or {}
    assert profile_after_patch["response_style"]["value"] == "краткий"
    assert profile_after_patch["response_style"]["source"] == "debug_menu"
    assert profile_after_patch["response_style"]["verified"] is True

    rv_add = client.post(
        "/debug/memory/profile/field",
        headers=_cookie_headers(sid, uid),
        json={"field": "custom_x", "value": "abc"},
    )
    assert rv_add.status_code == 200
    profile_after_add = (rv_add.get_json() or {}).get("profile") or {}
    assert profile_after_add["extra_fields"]["custom_x"]["value"] == "abc"

    app_module.agent.memory.long_term.update_profile_field(
        uid,
        "user_role_level",
        "senior backend",
        ProfileSource.AGENT_INFERRED,
        confidence=0.9,
    )
    rv_confirm = client.post(
        "/debug/memory/profile/confirm",
        headers=_cookie_headers(sid, uid),
        json={"field": "user_role_level"},
    )
    assert rv_confirm.status_code == 200
    profile_after_confirm = (rv_confirm.get_json() or {}).get("profile") or {}
    assert profile_after_confirm["user_role_level"]["verified"] is True

    rv_delete = client.delete(
        "/debug/memory/profile/field",
        headers=_cookie_headers(sid, uid),
        json={"field": "custom_x"},
    )
    assert rv_delete.status_code == 200
    profile_after_delete = (rv_delete.get_json() or {}).get("profile") or {}
    assert "custom_x" not in (profile_after_delete.get("extra_fields") or {})


def test_debug_profile_conflict_resolution_endpoint() -> None:
    _requires_app()
    import app as app_module

    client = app.test_client()
    sid = storage.create_session()
    uid = "debug_profile_conflict_user"

    app_module.agent.memory.long_term.update_profile_field(
        uid,
        "response_style",
        "краткий",
        ProfileSource.USER_EXPLICIT,
    )
    app_module.agent.memory.long_term.update_profile_field(
        uid,
        "response_style",
        "подробный",
        ProfileSource.AGENT_INFERRED,
        confidence=0.95,
    )

    rv_keep = client.post(
        "/debug/memory/profile/conflict/resolve",
        headers=_cookie_headers(sid, uid),
        json={"field": "response_style", "keep_existing": True},
    )
    assert rv_keep.status_code == 200
    profile_after_keep = (rv_keep.get_json() or {}).get("profile") or {}
    assert profile_after_keep["response_style"]["value"] == "краткий"
    assert len(profile_after_keep.get("conflicts") or []) == 0


def test_debug_profile_edit_reflects_in_next_chat_without_stale_cache() -> None:
    _requires_app()
    import app as app_module

    client = app.test_client()
    sid = storage.create_session()
    uid = "debug_profile_stale_cache_user"

    rv_patch = client.patch(
        "/debug/memory/profile/field",
        headers=_cookie_headers(sid, uid),
        json={"field": "response_style", "value": "краткий"},
    )
    assert rv_patch.status_code == 200

    original_chat = app_module.agent.chat
    try:
        def _fake_chat(message: str, session_id: str | None = None, user_id: str | None = None) -> str:
            del message, session_id
            profile = app_module.agent.memory.long_term.get_profile(user_id=user_id or uid)
            style = profile.get("response_style", {}).get("value")
            app_module.agent.last_token_stats = {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "cost_usd": 0.0,
                "latency_ms": 1,
            }
            app_module.agent.last_memory_stats = {}
            app_module.agent.last_prompt_preview = {}
            return f"style={style}"

        app_module.agent.chat = _fake_chat
        rv_chat = client.post(
            "/chat",
            headers=_cookie_headers(sid, uid),
            json={"message": "test"},
        )
        assert rv_chat.status_code == 200
        payload = rv_chat.get_json() or {}
        assert payload.get("reply") == "style=краткий"
    finally:
        app_module.agent.chat = original_chat


if __name__ == "__main__":
    test_debug_memory_layers_endpoint_200()
    test_debug_memory_layers_endpoint_top_k_clamp()
    test_debug_clear_working_memory_only()
    test_debug_delete_long_term_entry()
    test_session_restore_includes_working_view()
    test_chat_response_includes_working_view()
    test_index_contains_main_task_state_panel()
    test_debug_profile_endpoints_crud_and_confirm()
    test_debug_profile_conflict_resolution_endpoint()
    test_debug_profile_edit_reflects_in_next_chat_without_stale_cache()
    print("\n🎉 Debug memory-layers endpoint tests passed")
