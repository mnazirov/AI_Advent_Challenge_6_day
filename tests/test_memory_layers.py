from __future__ import annotations

import tempfile
from pathlib import Path

import storage
from memory.manager import MemoryManager
from memory.models import TaskState
from memory.prompt_builder import PromptBuilder
from memory.working import WorkingMemory
from scripts.demo_memory_layers import _format_short_term_snapshot


def _setup_temp_db() -> None:
    storage.DB_PATH = Path(tempfile.mktemp(suffix=".db"))
    storage.UPLOADS_DIR = Path(tempfile.mkdtemp())
    storage.init_db()


def test_short_term_limit_n() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=3)
    sid = storage.create_session()
    mem.short_term.append(sid, "user", "u1")
    mem.short_term.append(sid, "assistant", "a1")
    mem.short_term.append(sid, "user", "u2")
    mem.short_term.append(sid, "assistant", "a2")

    context = mem.short_term.get_context(sid)
    assert len(context) == 3
    assert context[0]["content"] == "a1"


def test_short_term_append_prunes_to_n_per_session() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()

    for i in range(1, 13):
        mem.short_term.append(sid, "user", f"m{i}")

    rows = storage.memory_load_short_term_messages(sid)
    assert len(rows) == 5
    assert [r["content"] for r in rows] == ["m8", "m9", "m10", "m11", "m12"]


def test_short_term_prune_isolated_by_session() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=3)
    sid1 = storage.create_session()
    sid2 = storage.create_session()

    for i in range(1, 8):
        mem.short_term.append(sid1, "user", f"s1-{i}")
    for i in range(1, 3):
        mem.short_term.append(sid2, "user", f"s2-{i}")

    rows1 = storage.memory_load_short_term_messages(sid1)
    rows2 = storage.memory_load_short_term_messages(sid2)

    assert [r["content"] for r in rows1] == ["s1-5", "s1-6", "s1-7"]
    assert [r["content"] for r in rows2] == ["s2-1", "s2-2"]


def test_short_term_read_returns_last_n_in_order() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=4)
    sid = storage.create_session()

    for i in range(1, 7):
        mem.short_term.append(sid, "user", f"turn-{i}")

    context = mem.short_term.get_context(sid)
    assert len(context) == 4
    assert [m["content"] for m in context] == ["turn-3", "turn-4", "turn-5", "turn-6"]


def test_short_term_is_runtime_scoped() -> None:
    _setup_temp_db()
    sid = storage.create_session()
    old_runtime = storage.SHORT_TERM_RUNTIME_ID
    try:
        storage.SHORT_TERM_RUNTIME_ID = "run_a"
        storage.memory_append_short_term_message(sid, "user", "from-run-a", "2026-03-03T10:00:00")
        assert len(storage.memory_load_short_term_messages(sid)) == 1

        storage.SHORT_TERM_RUNTIME_ID = "run_b"
        # Сообщения другого runtime не должны попадать в current short-term context.
        assert len(storage.memory_load_short_term_messages(sid)) == 0

        storage.memory_append_short_term_message(sid, "user", "from-run-b", "2026-03-03T10:01:00")
        rows = storage.memory_load_short_term_messages(sid)
        assert len(rows) == 1
        assert rows[0]["content"] == "from-run-b"
    finally:
        storage.SHORT_TERM_RUNTIME_ID = old_runtime


def test_demo_short_term_snapshot_format() -> None:
    messages = []
    for i in range(1, 41):
        messages.append(
            {
                "role": "user" if i % 2 else "assistant",
                "content": f"msg-{i}-" + ("x" * 120),
            }
        )

    lines = _format_short_term_snapshot(messages, limit_n=30)
    assert lines[0] == "[SHORT_TERM_SNAPSHOT] last_turns=30"
    assert len(lines) == 31
    assert lines[1].startswith("  - user: msg-11-")  # kept only last 30 turns
    for line in lines[1:]:
        assert line.startswith("  - ")
        payload = line.split(": ", 1)[1]
        assert len(payload) <= 80


def test_router_writes_to_correct_layers() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=10)
    sid = storage.create_session()
    uid = "u"

    events_profile = mem.route_user_message(
        session_id=sid,
        user_id=uid,
        user_message="С этого момента всегда отвечай кратко и не используй англицизмы",
    )
    assert any(e.layer == "long_term.profile" for e in events_profile)

    events_decision = mem.route_user_message(
        session_id=sid,
        user_id=uid,
        user_message="Решили использовать единый стандарт логирования",
    )
    assert any(e.layer == "long_term.decision" for e in events_decision)

    events_working = mem.route_user_message(
        session_id=sid,
        user_id=uid,
        user_message="Требование: добавь шаг интеграции API",
    )
    assert any(e.layer == "working" for e in events_working)


def test_prompt_builder_sections() -> None:
    builder = PromptBuilder()
    messages, _ = builder.build(
        system_instructions="SYS",
        data_context="DATA",
        long_term={
            "profile": {"style": "concise", "constraints": ["ru"], "context": ["ctx"]},
            "decisions": [{"text": "Use Flask"}],
            "notes": [{"text": "Note A"}],
        },
        working=None,
        short_term_messages=[{"role": "user", "content": "Prev Q"}],
        user_query="Now Q",
    )
    assert messages[0]["role"] == "system"
    assert "[LONG_TERM_PROFILE]" in messages[0]["content"]
    assert "[LONG_TERM_DECISIONS]" in messages[0]["content"]
    assert messages[-1]["content"] == "Now Q"


def test_working_and_longterm_save_load() -> None:
    _setup_temp_db()
    wm = WorkingMemory()
    sid = storage.create_session()
    ctx = wm.start_task(session_id=sid, goal="Сделать видео")
    wm.update(sid, plan=["Шаг 1", "Шаг 2"], current_step="Шаг 1")
    loaded = wm.load(sid)
    assert loaded is not None
    assert loaded.goal == "Сделать видео"
    assert loaded.plan == ["Шаг 1", "Шаг 2"]

    mem = MemoryManager(short_term_limit=5)
    mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Решили использовать SQLite для памяти",
    )
    lt = mem.long_term.retrieve(user_id="u", query="sqlite")
    assert len(lt["decisions"]) >= 1


def test_working_clears_on_done_state() -> None:
    _setup_temp_db()
    wm = WorkingMemory()
    sid = storage.create_session()
    ctx = wm.start_task(session_id=sid, goal="Закрыть задачу")
    ctx.state = TaskState.DONE
    wm.save(ctx)
    assert wm.load(sid) is None


def test_strict_planning_gate() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    mem.working.start_task(session_id=sid, goal="Новая фича")
    blocked = mem.enforce_planning_gate(session_id=sid, user_message="Сразу напиши код")
    assert blocked is not None

    mem.working.update(sid, plan=["step1"], current_step="step1")
    ctx = mem.working.load(sid)
    assert ctx is not None
    mem.working.transition_state(ctx, ctx.state.__class__("EXECUTION"))
    mem.working.save(ctx)
    blocked_after = mem.enforce_planning_gate(session_id=sid, user_message="Сразу напиши код")
    assert blocked_after is None


def test_strict_planning_gate_financial_trigger() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    mem.working.start_task(session_id=sid, goal="Оптимизация бюджета")
    blocked = mem.enforce_planning_gate(session_id=sid, user_message="Сразу дай финальный план бюджета")
    assert blocked is not None


def test_planning_gate_financial_trigger_allows_after_plan() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    mem.working.start_task(session_id=sid, goal="Оптимизация бюджета")
    mem.working.update(sid, plan=["Собрать метрики"], current_step="Собрать метрики")

    blocked = mem.enforce_planning_gate(session_id=sid, user_message="Сразу дай финальный план бюджета")
    assert blocked is None
    ctx = mem.working.load(sid)
    assert ctx is not None
    assert ctx.state.value == "EXECUTION"


def test_planning_gate_auto_transitions_when_ready() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    mem.working.start_task(session_id=sid, goal="Новая фича")

    mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Требование: добавь шаг реализовать endpoint",
    )
    mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Текущий шаг: реализовать endpoint",
    )

    blocked = mem.enforce_planning_gate(session_id=sid, user_message="Напиши код endpoint")
    assert blocked is None
    ctx = mem.working.load(sid)
    assert ctx is not None
    assert ctx.state.value == "EXECUTION"


def test_memory_debug_snapshot_short_term_shape() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=4)
    sid = storage.create_session()
    for i in range(1, 8):
        mem.short_term.append(sid, "user", f"msg-{i}")
        mem.short_term.append(sid, "assistant", f"reply-{i}")

    snap = mem.debug_snapshot(session_id=sid, user_id="u", query="", top_k=3)
    st = snap["short_term"]
    assert st["limit_n"] == 4
    assert st["turns_count"] <= 4
    assert len(st["turns"]) == st["turns_count"]
    for t in st["turns"]:
        assert "id" in t
        assert "role" in t
        assert "timestamp" in t
        assert "preview" in t
        assert "full" in t
        assert "full_truncated" in t
    roles = [t["role"] for t in st["turns"]]
    assert "user" in roles or "assistant" in roles
    first_preview = st["turns"][0]["preview"]
    assert len(first_preview) <= 121


def test_memory_debug_snapshot_working_present_and_empty() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    uid = "user1"

    snap_empty = mem.debug_snapshot(session_id=sid, user_id=uid, query="", top_k=3)
    assert snap_empty["working"]["present"] is False
    assert snap_empty["working"]["task"] is None

    mem.working.start_task(session_id=sid, goal="Цель задачи")
    snap_present = mem.debug_snapshot(session_id=sid, user_id=uid, query="", top_k=3)
    assert snap_present["working"]["present"] is True
    assert snap_present["working"]["task"] is not None
    assert snap_present["working"]["task"]["goal"] == "Цель задачи"
    assert "state" in snap_present["working"]["task"]
    assert "task_id" in snap_present["working"]["task"]


def test_memory_debug_snapshot_long_term_top_k() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    uid = "u_lt"
    mem.long_term.add_decision(uid, "Решение 1", tags=["a"])
    mem.long_term.add_decision(uid, "Решение 2", tags=["b"])
    mem.long_term.add_decision(uid, "Решение 3", tags=["c"])
    mem.long_term.add_note(uid, "Заметка 1", tags=["x"])
    mem.long_term.add_note(uid, "Заметка 2", tags=["y"])

    snap = mem.debug_snapshot(session_id=sid, user_id=uid, query="", top_k=2)
    lt = snap["long_term"]
    assert len(lt["decisions_top_k"]) <= 2
    assert len(lt["notes_top_k"]) <= 2
    for d in lt["decisions_top_k"]:
        assert "id" in d
        assert "preview" in d
        assert "full" in d
        assert "full_truncated" in d

    snap10 = mem.debug_snapshot(session_id=sid, user_id=uid, query="", top_k=10)
    assert len(snap10["long_term"]["decisions_top_k"]) <= 10
    assert len(snap10["long_term"]["notes_top_k"]) <= 10


def test_memory_debug_snapshot_full_truncation_flag() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=3)
    sid = storage.create_session()
    long_text = "x" * 5000
    mem.short_term.append(sid, "user", long_text)

    snap = mem.debug_snapshot(session_id=sid, user_id="u", query="", top_k=3)
    turns = snap["short_term"]["turns"]
    assert len(turns) == 1
    assert turns[0]["full_truncated"] is True
    assert len(turns[0]["full"]) <= 4001
    assert len(turns[0]["preview"]) <= 121


if __name__ == "__main__":
    test_short_term_limit_n()
    test_short_term_append_prunes_to_n_per_session()
    test_short_term_prune_isolated_by_session()
    test_short_term_read_returns_last_n_in_order()
    test_short_term_is_runtime_scoped()
    test_demo_short_term_snapshot_format()
    test_router_writes_to_correct_layers()
    test_prompt_builder_sections()
    test_working_and_longterm_save_load()
    test_working_clears_on_done_state()
    test_strict_planning_gate()
    test_strict_planning_gate_financial_trigger()
    test_planning_gate_financial_trigger_allows_after_plan()
    test_planning_gate_auto_transitions_when_ready()
    test_memory_debug_snapshot_short_term_shape()
    test_memory_debug_snapshot_working_present_and_empty()
    test_memory_debug_snapshot_long_term_top_k()
    test_memory_debug_snapshot_full_truncation_flag()
    print("\n🎉 Memory layers tests passed")
