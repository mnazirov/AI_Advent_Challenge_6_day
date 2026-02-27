"""
storage.py — SQLite-хранилище сессий для финансового AI-агента.
Обеспечивает персистентность диалога и метаданных CSV между перезагрузками страницы.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("storage")

DB_PATH = Path("data/agent.db")
UPLOADS_DIR = Path("uploads")
MAX_RESTORE_MESSAGES = 50


def init_db() -> None:
    """Инициализирует базу данных и создаёт таблицы если их нет."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    with _get_conn() as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                filename    TEXT,
                csv_summary         TEXT,
                ctx_summary         TEXT    DEFAULT '',
                ctx_summarized_upto INTEGER DEFAULT 0,
                schema_map          TEXT DEFAULT '{}',
                csv_path    TEXT,
                total_tokens_in  INTEGER DEFAULT 0,
                total_tokens_out INTEGER DEFAULT 0,
                total_cost_usd   REAL    DEFAULT 0.0,
                cost_history     TEXT    DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content     TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_planning BOOLEAN DEFAULT 0,
                tokens_used INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);
            """
        )
        _ensure_sessions_columns(conn)

    cleaned = cleanup_old_sessions(30)
    logger.info("[STORAGE] БД инициализирована: %s | очищено_старых_сессий=%s", DB_PATH.resolve(), cleaned)


@contextmanager
def _get_conn():
    """Контекстный менеджер соединения с авто-commit и авто-close."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_session() -> str:
    """Создаёт новую сессию и возвращает её ID."""
    sid = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute("INSERT INTO sessions (id) VALUES (?)", (sid,))
    logger.info("[STORAGE] Создана новая сессия: %s…", sid[:8])
    return sid


def session_exists(session_id: str) -> bool:
    """Проверяет существование сессии в БД."""
    with _get_conn() as conn:
        row = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
    return row is not None


def get_latest_session_id() -> str | None:
    """
    Возвращает ID последней активной сессии.
    Берётся сессия с максимальным updated_at.
    """
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    return row["id"] if row else None


def save_csv_meta(
    session_id: str,
    filename: str,
    csv_summary: str,
    schema_map: dict,
    csv_path: str,
) -> None:
    """Сохраняет метаданные загруженного CSV в сессию."""
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET filename=?, csv_summary=?, schema_map=?, csv_path=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                filename,
                csv_summary,
                json.dumps(schema_map, ensure_ascii=False),
                csv_path,
                session_id,
            ),
        )
    logger.info("[STORAGE] CSV-метаданные сохранены: сессия=%s… файл=%s", session_id[:8], filename)


def save_context_summary(session_id: str, summary: str, summarized_up_to: int) -> None:
    """
    Сохраняет текущий summary контекста.
    Вызывается в app.py после каждого chat-запроса если summary изменился.
    """
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET ctx_summary = ?, ctx_summarized_upto = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (summary, int(summarized_up_to), session_id),
        )
    logger.info(
        "[STORAGE] ctx_summary сохранён: сессия=%s… len=%s summarized_up_to=%s",
        session_id[:8],
        len(summary),
        summarized_up_to,
    )


def save_message(
    session_id: str,
    role: str,
    content: str,
    is_planning: bool = False,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    tokens_used: int | None = None,
) -> None:
    """Сохраняет одно сообщение диалога."""
    message_tokens = int(tokens_used) if tokens_used is not None else int(tokens_in) + int(tokens_out)
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, is_planning, tokens_used)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, int(bool(is_planning)), message_tokens),
        )
        if role == "assistant":
            current_cost_raw = conn.execute(
                "SELECT total_cost_usd, cost_history FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
            if current_cost_raw:
                try:
                    cost_history = json.loads(current_cost_raw["cost_history"] or "[]")
                except json.JSONDecodeError:
                    cost_history = []
                if not isinstance(cost_history, list):
                    cost_history = []
                cost_history.append(float(cost_usd))
                conn.execute(
                    """
                    UPDATE sessions
                    SET total_tokens_in = total_tokens_in + ?,
                        total_tokens_out = total_tokens_out + ?,
                        total_cost_usd = total_cost_usd + ?,
                        cost_history = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (
                        int(tokens_in),
                        int(tokens_out),
                        float(cost_usd),
                        json.dumps(cost_history, ensure_ascii=False),
                        session_id,
                    ),
                )
            else:
                conn.execute("UPDATE sessions SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (session_id,))
        else:
            conn.execute("UPDATE sessions SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (session_id,))


def add_usage(
    session_id: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float = 0.0,
) -> None:
    """
    Добавляет usage-токены/стоимость к агрегатам сессии без сохранения сообщения.
    Используется для LLM-вызовов вне чата (например, при загрузке CSV).
    """
    with _get_conn() as conn:
        current = conn.execute(
            "SELECT cost_history FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        cost_history = []
        if current:
            try:
                cost_history = json.loads(current["cost_history"] or "[]")
            except json.JSONDecodeError:
                cost_history = []
        if not isinstance(cost_history, list):
            cost_history = []
        if cost_usd > 0:
            cost_history.append(float(cost_usd))
        conn.execute(
            """
            UPDATE sessions
            SET total_tokens_in = total_tokens_in + ?,
                total_tokens_out = total_tokens_out + ?,
                total_cost_usd = total_cost_usd + ?,
                cost_history = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                int(tokens_in),
                int(tokens_out),
                float(cost_usd),
                json.dumps(cost_history, ensure_ascii=False),
                session_id,
            ),
        )


def load_session(session_id: str) -> dict | None:
    """Загружает полное состояние сессии для восстановления агента."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            return None

        msgs = conn.execute(
            """
            SELECT role, content, is_planning, tokens_used, created_at
            FROM messages
            WHERE session_id=?
            ORDER BY id DESC
            """,
            (session_id,),
        ).fetchall()

    ordered_msgs = list(reversed(msgs))
    return {
        "session_id": row["id"],
        "filename": row["filename"],
        "csv_summary": row["csv_summary"],
        "ctx_summary": row["ctx_summary"] or "",
        "ctx_summarized_upto": int(row["ctx_summarized_upto"] or 0),
        "schema_map": json.loads(row["schema_map"] or "{}"),
        "csv_path": row["csv_path"],
        "total_tokens_in": int(row["total_tokens_in"] or 0),
        "total_tokens_out": int(row["total_tokens_out"] or 0),
        "total_cost_usd": float(row["total_cost_usd"] or 0.0),
        "cost_history": json.loads(row["cost_history"] or "[]"),
        "messages": [
            {
                "role": m["role"],
                "content": m["content"],
                "is_planning": bool(m["is_planning"]),
            }
            for m in ordered_msgs
        ],
    }


def clear_session_messages(session_id: str) -> None:
    """Очищает историю сообщений (метаданные CSV сохраняются)."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute(
            """
            UPDATE sessions
            SET total_tokens_in=0,
                total_tokens_out=0,
                total_cost_usd=0.0,
                cost_history='[]',
                ctx_summary='',
                ctx_summarized_upto=0,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (session_id,),
        )
    logger.info("[STORAGE] История очищена: сессия=%s…", session_id[:8])


def clear_session_csv(session_id: str, delete_file: bool = True) -> None:
    """
    Очищает привязанный CSV у сессии (метаданные и путь).
    При delete_file=True удаляет файл из uploads, если он существует.
    """
    csv_path = None
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT csv_path FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        if row:
            csv_path = row["csv_path"]

        conn.execute(
            """
            UPDATE sessions
            SET filename=NULL, csv_summary=NULL, schema_map='{}', csv_path=NULL, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (session_id,),
        )

    if delete_file and csv_path:
        file_path = Path(csv_path)
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info("[STORAGE] CSV-файл удалён: %s", file_path)
        except Exception as exc:
            logger.warning("[STORAGE] Не удалось удалить CSV-файл %s: %s", file_path, exc)

    logger.info("[STORAGE] CSV-данные очищены: сессия=%s…", session_id[:8])


def save_csv_file(file_content: bytes, filename: str) -> str:
    """Сохраняет CSV-файл на диск и возвращает путь к нему."""
    safe_name = Path(filename).stem[:40] + "_" + str(uuid.uuid4())[:8] + ".csv"
    filepath = UPLOADS_DIR / safe_name
    filepath.write_bytes(file_content)
    logger.info("[STORAGE] CSV сохранён: %s", filepath)
    return str(filepath)


def load_csv_file(csv_path: str) -> bytes | None:
    """Читает CSV-файл с диска для восстановления self.df агента."""
    path = Path(csv_path)
    if not path.exists():
        logger.warning("[STORAGE] CSV не найден на диске: %s", csv_path)
        return None
    return path.read_bytes()


def cleanup_old_sessions(days: int = 30) -> int:
    """Удаляет сессии старше N дней и возвращает их количество."""
    with _get_conn() as conn:
        result = conn.execute(
            """
            DELETE FROM sessions
            WHERE updated_at < datetime('now', ? || ' days')
            """,
            (f"-{int(days)}",),
        )
    deleted = result.rowcount or 0
    if deleted:
        logger.info("[STORAGE] Очищено старых сессий: %s", deleted)
    return int(deleted)


def _ensure_sessions_columns(conn: sqlite3.Connection) -> None:
    """Добавляет недостающие столбцы в sessions для старых БД."""
    rows = conn.execute("PRAGMA table_info(sessions)").fetchall()
    existing = {row["name"] for row in rows}
    migration = [
        ("total_tokens_in", "INTEGER DEFAULT 0"),
        ("total_tokens_out", "INTEGER DEFAULT 0"),
        ("total_cost_usd", "REAL DEFAULT 0.0"),
        ("cost_history", "TEXT DEFAULT '[]'"),
        ("ctx_summary", "TEXT DEFAULT ''"),
        ("ctx_summarized_upto", "INTEGER DEFAULT 0"),
    ]
    for column_name, column_def in migration:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {column_name} {column_def}")
