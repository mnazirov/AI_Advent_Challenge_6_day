"""
Интеграционные тесты для GET /debug/memory-layers.
Требуют установленных зависимостей проекта (pandas, flask, …).
Запуск с venv: pytest tests/test_debug_memory_layers.py -v
"""

from __future__ import annotations

import tempfile
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


if __name__ == "__main__":
    test_debug_memory_layers_endpoint_200()
    test_debug_memory_layers_endpoint_top_k_clamp()
    print("\n🎉 Debug memory-layers endpoint tests passed")
