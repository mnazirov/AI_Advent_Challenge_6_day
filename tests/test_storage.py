"""
Запуск: python tests/test_storage.py
Проверяет что сессии корректно сохраняются и восстанавливаются.
"""

import tempfile
from pathlib import Path

import storage

# Используем временную БД для теста
storage.DB_PATH = Path(tempfile.mktemp(suffix=".db"))
storage.UPLOADS_DIR = Path(tempfile.mkdtemp())

from storage import (
    clear_session_messages,
    create_session,
    init_db,
    load_session,
    save_csv_meta,
    save_ctx_state,
    save_message,
    session_exists,
)

init_db()

# Тест 1: создание и существование сессии
sid = create_session()
assert session_exists(sid), "сессия должна существовать после создания"
assert not session_exists("fake-uuid"), "несуществующая сессия должна вернуть False"
print("✅ Тест 1: create_session + session_exists")

# Тест 2: сохранение CSV-метаданных
save_csv_meta(sid, "test.csv", "## Сводка", {"date": "Дата"}, "/tmp/test.csv")
data = load_session(sid)
assert data is not None
assert data["filename"] == "test.csv"
assert data["csv_summary"] == "## Сводка"
assert data["schema_map"] == {"date": "Дата"}
print("✅ Тест 2: save_csv_meta + load_session")

# Тест 3: сохранение сообщений
save_message(sid, "user", "Привет")
save_message(sid, "assistant", "Здравствуйте!")
data = load_session(sid)
assert data is not None
assert len(data["messages"]) == 2
assert data["messages"][0]["role"] == "user"
assert data["messages"][1]["content"] == "Здравствуйте!"
print("✅ Тест 3: save_message + load_session messages")

# Тест 3.1: сохранение ctx_state
ctx_state = {"active": "sticky_facts", "sticky_facts": {"facts": {"goal": "накопить 100к"}}}
save_ctx_state(sid, ctx_state)
data = load_session(sid)
assert data is not None
assert data["ctx_state"]["active"] == "sticky_facts"
print("✅ Тест 3.1: save_ctx_state + load_session ctx_state")

# Тест 4: очистка истории
clear_session_messages(sid)
data = load_session(sid)
assert data is not None
assert len(data["messages"]) == 0
assert data["filename"] == "test.csv"  # метаданные сохранились
assert data["ctx_state"] == {}
print("✅ Тест 4: clear_session_messages (сообщения удалены, метаданные сохранены)")

# Тест 5: несуществующая сессия
data = load_session("non-existent-id")
assert data is None
print("✅ Тест 5: load_session несуществующей сессии → None")

# Cleanup
storage.DB_PATH.unlink(missing_ok=True)
print("\n🎉 Все тесты пройдены")
