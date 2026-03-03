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


if __name__ == "__main__":
    test_debug_memory_layers_endpoint_200()
    test_debug_memory_layers_endpoint_top_k_clamp()
    test_debug_clear_working_memory_only()
    test_debug_delete_long_term_entry()
    print("\n🎉 Debug memory-layers endpoint tests passed")
