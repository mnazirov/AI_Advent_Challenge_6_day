from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import storage
from llm.mock_client import MockLLMClient
from memory.manager import MemoryManager
from memory.models import TaskState
from memory.prompt_builder import PromptBuilder
from memory.working import WorkingMemory
from scripts.demo_memory_layers import _format_short_term_snapshot


def _setup_temp_db() -> None:
    storage.DB_PATH = Path(tempfile.mktemp(suffix=".db"))
    storage.UPLOADS_DIR = Path(tempfile.mkdtemp())
    storage.init_db()


def _build_agent_with_mock_llm(mock_llm: MockLLMClient) -> Any:
    try:
        from agent import FinancialAgent
    except ModuleNotFoundError as exc:
        pytest.skip(f"agent dependencies are not installed: {exc}")

    class _CtxStub:
        def stats(self, history: list[dict]) -> dict:
            del history
            return {}

    agent = FinancialAgent.__new__(FinancialAgent)
    agent.llm_client = mock_llm
    agent.model = "gpt-5-mini"
    agent.conversation_history = []
    agent.ctx = _CtxStub()
    agent.memory = MemoryManager(short_term_limit=30, llm_client=mock_llm, step_parser_model="gpt-5-nano")
    agent.csv_summary = None
    agent.summary_sections = {}
    agent.expense_cache = {}
    agent.df = None
    agent.last_token_stats = None
    agent.last_schema_token_stats = None
    agent.last_memory_stats = None
    agent.last_prompt_preview = None
    agent._last_encoding = None
    return agent


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


def test_short_term_filters_system_role_from_context() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    mem.short_term.append(sid, "user", "u1")
    mem.short_term.append(sid, "system", "internal system text")
    mem.short_term.append(sid, "assistant", "a1")

    context = mem.short_term.get_context(sid)
    assert [m["role"] for m in context] == ["user", "assistant"]
    assert all(m["role"] != "system" for m in context)


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
    assert not any(e.layer == "long_term.profile" for e in events_profile)

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
            "profile": {
                "stack_tools": {
                    "value": ["FastAPI", "Postgres"],
                    "source": "user_explicit",
                    "verified": True,
                    "confidence": None,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                "response_style": {
                    "value": "краткий",
                    "source": "user_explicit",
                    "verified": True,
                    "confidence": None,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                "hard_constraints": {
                    "value": ["no AWS"],
                    "source": "user_explicit",
                    "verified": True,
                    "confidence": None,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                "user_role_level": {
                    "value": "senior backend",
                    "source": "user_explicit",
                    "verified": True,
                    "confidence": None,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                "project_context": {
                    "value": {"project_name": "BudgetBot", "goals": ["Reduce churn"], "key_decisions": ["Use API"]},
                    "source": "user_explicit",
                    "verified": True,
                    "confidence": None,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                "extra_fields": {"x": {"value": "y", "source": "debug_menu", "verified": True, "confidence": None, "updated_at": datetime.utcnow().isoformat()}},
                "conflicts": [],
            },
            "decisions": [{"text": "Use Flask"}],
            "notes": [{"text": "Note A"}],
        },
        working=None,
        short_term_messages=[{"role": "user", "content": "Prev Q"}],
        user_query="Now Q",
    )
    assert messages[0]["role"] == "system"
    assert "[RESPONSE_STYLE_POLICY]" in messages[0]["content"]
    assert "[HARD_CONSTRAINTS]" in messages[0]["content"]
    assert "[STACK_TOOLS_PREFERENCE]" in messages[0]["content"]
    assert "[USER_ROLE_LEVEL]" in messages[0]["content"]
    assert "[PROJECT_CONTEXT]" in messages[0]["content"]
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


def test_working_done_state_is_persisted_and_frozen() -> None:
    _setup_temp_db()
    wm = WorkingMemory()
    sid = storage.create_session()
    ctx = wm.start_task(session_id=sid, goal="Закрыть задачу")
    ctx.plan = ["Шаг 1"]
    ctx.current_step = "Шаг 1"
    wm.transition_state(ctx, TaskState.EXECUTION)
    ctx.done = ["Шаг 1"]
    wm.transition_state(ctx, TaskState.VALIDATION)
    wm.transition_state(ctx, TaskState.DONE)
    wm.save(ctx)
    loaded = wm.load(sid)
    assert loaded is not None
    assert loaded.state == TaskState.DONE
    with pytest.raises(ValueError, match="Working memory is frozen in DONE state"):
        wm.update(session_id=sid, open_questions=["x"])


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


def test_router_autofills_current_step_from_first_added_step() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()

    mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Требование: добавь шаг реализовать endpoint",
    )
    ctx = mem.working.load(sid)
    assert ctx is not None
    assert ctx.plan == ["реализовать endpoint"]
    assert ctx.current_step == "реализовать endpoint"


def test_router_autofill_current_step_does_not_override_existing() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()

    mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Требование: добавь шаг шаг 1",
    )
    mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Текущий шаг: ручной текущий шаг",
    )
    mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Требование: добавь шаг шаг 2",
    )
    ctx = mem.working.load(sid)
    assert ctx is not None
    assert ctx.plan == ["шаг 1", "ручной текущий шаг", "шаг 2"]
    assert ctx.current_step == "ручной текущий шаг"


def test_router_llm_patch_updates_working_without_regex_markers() -> None:
    _setup_temp_db()

    def responder(messages, kwargs):
        del messages
        if kwargs.get("model") == "gpt-5-nano":
            return (
                '{"is_working_update": true, "plan_steps_to_add": ["Собрать метрики подписок"], '
                '"current_step": "Собрать метрики подписок", "done_steps_to_add": [], '
                '"requirements_to_add": [], "artifacts_to_add": [], "confidence": 0.92}'
            )
        return "ok"

    mem = MemoryManager(short_term_limit=5, llm_client=MockLLMClient(responder=responder))
    sid = storage.create_session()

    events = mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Начнем с анализа подписок за месяц",
    )
    ctx = mem.working.load(sid)
    assert ctx is not None
    assert "Собрать метрики подписок" in ctx.plan
    assert ctx.current_step == "Собрать метрики подписок"
    assert any(e.layer == "working" for e in events)


def test_router_plan_formation_llm_transitions_to_execution() -> None:
    _setup_temp_db()

    def responder(messages, kwargs):
        del messages
        if kwargs.get("model") == "gpt-5-nano":
            return (
                '{"is_working_update": true, "task": "План оптимизации бюджета", '
                '"plan": ["Собрать метрики расходов", "Определить лимиты по категориям"], '
                '"plan_steps_to_add": [], "current_step": "Собрать метрики расходов", '
                '"done_steps_to_add": [], "requirements_to_add": [], "artifacts_to_add": [], "confidence": 0.9}'
            )
        return "ok"

    mem = MemoryManager(short_term_limit=5, llm_client=MockLLMClient(responder=responder))
    sid = storage.create_session()

    events = mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Сформируй план задачи",
    )
    ctx = mem.working.load(sid)
    assert ctx is not None
    assert ctx.plan == ["Собрать метрики расходов", "Определить лимиты по категориям"]
    assert ctx.current_step == "Собрать метрики расходов"
    assert ctx.state == TaskState.EXECUTION
    assert any(e.layer == "working" and "plan" in e.keys for e in events)
    assert any(e.layer == "working" and "state" in e.keys for e in events)


def test_router_llm_invalid_json_falls_back_to_regex() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5, llm_client=MockLLMClient(responder=lambda m, k: "not-json"))
    sid = storage.create_session()

    mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Требование: добавь шаг подготовить отчет",
    )
    ctx = mem.working.load(sid)
    assert ctx is not None
    assert ctx.plan == ["подготовить отчет"]
    assert ctx.current_step == "подготовить отчет"


def test_router_llm_low_confidence_patch_is_ignored() -> None:
    _setup_temp_db()

    def responder(messages, kwargs):
        del messages
        if kwargs.get("model") == "gpt-5-nano":
            return (
                '{"is_working_update": true, "plan_steps_to_add": ["Шаг из low confidence"], '
                '"current_step": "Шаг из low confidence", "done_steps_to_add": [], '
                '"requirements_to_add": [], "artifacts_to_add": [], "confidence": 0.21}'
            )
        return "ok"

    mem = MemoryManager(short_term_limit=5, llm_client=MockLLMClient(responder=responder))
    sid = storage.create_session()

    events = mem.route_user_message(
        session_id=sid,
        user_id="u",
        user_message="Просто болтаем без постановки задачи",
    )
    assert not events
    assert mem.working.load(sid) is None


def test_router_task_intent_autostarts_when_llm_confidence_zero(caplog: pytest.LogCaptureFixture) -> None:
    _setup_temp_db()

    def responder(messages, kwargs):
        del messages
        if kwargs.get("model") == "gpt-5-nano":
            return (
                '{"is_working_update": true, "task": "Составить детальный план на месяц", '
                '"plan_steps_to_add": [], "current_step": "", "done_steps_to_add": [], '
                '"requirements_to_add": [], "artifacts_to_add": [], "confidence": 0.0}'
            )
        return "ok"

    mem = MemoryManager(short_term_limit=5, llm_client=MockLLMClient(responder=responder))
    sid = storage.create_session()
    msg = "Составь мне детальный план на месяц"

    with caplog.at_level("INFO", logger="memory"):
        events = mem.route_user_message(
            session_id=sid,
            user_id="u",
            user_message=msg,
        )

    ctx = mem.working.load(sid)
    assert ctx is not None
    assert ctx.task == msg[:100]
    assert ctx.state == TaskState.PLANNING
    assert any(e.layer == "working" for e in events)
    assert "[TASK_AUTO_START]" in caplog.text
    assert "[WORKING_EXTRACT_FALLBACK]" in caplog.text
    assert "applied=True confidence=0.85" in caplog.text


def test_router_zero_confidence_plan_formation_sets_guidance_flag() -> None:
    _setup_temp_db()

    def responder(messages, kwargs):
        del messages
        if kwargs.get("model") == "gpt-5-nano":
            return (
                '{"is_working_update": true, "task": "Сформировать план", '
                '"plan": [], "plan_steps_to_add": [], "current_step": "", '
                '"done_steps_to_add": [], "requirements_to_add": [], "artifacts_to_add": [], "confidence": 0.0}'
            )
        return "ok"

    mem = MemoryManager(short_term_limit=5, llm_client=MockLLMClient(responder=responder))
    sid = storage.create_session()
    mem.route_user_message(session_id=sid, user_id="u", user_message="Сформируй план задачи")
    ctx = mem.working.load(sid)
    assert ctx is not None
    assert ctx.state == TaskState.PLANNING
    assert bool((ctx.vars or {}).get("plan_guidance_required")) is True


def test_planning_gate_allows_plan_formation_when_plan_empty() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    mem.working.start_task(session_id=sid, goal="Оптимизация бюджета")

    blocked = mem.enforce_planning_gate(session_id=sid, user_message="Сформируй план задачи")
    assert blocked is None


def test_working_actions_financial_only_no_code_shortcuts() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    mem.working.start_task(session_id=sid, goal="Оптимизация бюджета")

    planning_actions = mem.get_working_actions(session_id=sid)
    planning_labels = {a.get("label") for a in planning_actions}
    planning_messages = " ".join(str(a.get("message") or "") for a in planning_actions).lower()
    assert planning_labels == {"Сформировать план автоматически", "Уточнить цель задачи"}
    assert "код" not in planning_messages
    assert "deploy" not in planning_messages

    mem.working.update(sid, plan=["A", "B"], current_step="A")
    ctx = mem.working.load(sid)
    assert ctx is not None
    mem.working.transition_state(ctx, TaskState.EXECUTION)
    mem.working.save(ctx)
    exec_actions = mem.get_working_actions(session_id=sid)
    exec_labels = {a.get("label") for a in exec_actions}
    assert exec_labels == {"Шаг выполнен", "Статус задачи"}


def test_working_load_repairs_legacy_current_step_and_persists() -> None:
    _setup_temp_db()
    sid = storage.create_session()
    storage.memory_save_working_task(
        session_id=sid,
        task_id="task-legacy",
        goal="Legacy task",
        state="PLANNING",
        plan=["A", "B"],
        current_step="",
        done_steps=["A"],
        open_questions=[],
        artifacts=[],
        vars_data={},
        updated_at=datetime.utcnow().isoformat(),
    )

    wm = WorkingMemory()
    ctx = wm.load(sid)
    assert ctx is not None
    assert ctx.current_step == "B"
    persisted = storage.memory_load_working_task(sid)
    assert persisted is not None
    assert persisted["current_step"] == "B"


def test_working_load_repairs_legacy_current_step_when_all_done() -> None:
    _setup_temp_db()
    sid = storage.create_session()
    storage.memory_save_working_task(
        session_id=sid,
        task_id="task-legacy-all-done",
        goal="Legacy task",
        state="PLANNING",
        plan=["A"],
        current_step="",
        done_steps=["A"],
        open_questions=[],
        artifacts=[],
        vars_data={},
        updated_at=datetime.utcnow().isoformat(),
    )

    wm = WorkingMemory()
    ctx = wm.load(sid)
    assert ctx is not None
    assert ctx.current_step == "A"
    persisted = storage.memory_load_working_task(sid)
    assert persisted is not None
    assert persisted["current_step"] == "A"


def test_agent_chat_populates_working_current_step_via_mock_llm() -> None:
    _setup_temp_db()

    def responder(messages, kwargs):
        del messages
        if kwargs.get("model") == "gpt-5-nano":
            return (
                '{"is_working_update": true, "plan_steps_to_add": ["Анализ подписок"], '
                '"current_step": "Анализ подписок", "done_steps_to_add": [], '
                '"requirements_to_add": [], "artifacts_to_add": [], "confidence": 0.9}'
            )
        return "Принято. Двигаемся по шагам."

    mock = MockLLMClient(responder=responder)
    agent = _build_agent_with_mock_llm(mock)
    sid = storage.create_session()
    uid = "integration_user"

    reply = agent.chat("Начнем с анализа подписок и зафиксируем стартовый шаг", session_id=sid, user_id=uid)
    assert reply

    snapshot = agent.memory.debug_snapshot(session_id=sid, user_id=uid, query="", top_k=3)
    assert snapshot["working"]["present"] is True
    task = snapshot["working"]["task"] or {}
    assert task.get("current_step") == "Анализ подписок"


def test_agent_chat_plan_formation_in_planning_calls_llm_and_transitions() -> None:
    _setup_temp_db()

    def responder(messages, kwargs):
        del messages
        if kwargs.get("model") == "gpt-5-nano":
            return (
                '{"is_working_update": true, "task": "Оптимизировать бюджет", '
                '"plan": ["Собрать метрики расходов", "Поставить лимиты по категориям"], '
                '"plan_steps_to_add": [], "current_step": "Собрать метрики расходов", '
                '"done_steps_to_add": [], "requirements_to_add": [], "artifacts_to_add": [], "confidence": 0.9}'
            )
        return "План сформирован. Начинаем выполнение."

    mock = MockLLMClient(responder=responder)
    agent = _build_agent_with_mock_llm(mock)
    sid = storage.create_session()
    uid = "planning_flow_user"
    agent.memory.working.start_task(session_id=sid, goal="Оптимизация бюджета")

    reply = agent.chat("сформируй план задачи", session_id=sid, user_id=uid)
    assert "План" in reply or "план" in reply
    stats = agent.last_token_stats or {}
    assert int(stats.get("prompt_tokens", 0) or 0) > 0
    assert str(stats.get("finish_reason") or "") != "state_blocked_planning"

    working_view = agent.memory.get_working_view(session_id=sid)
    assert working_view.get("state") == "EXECUTION"
    assert working_view.get("step_index") == 1
    labels = {a.get("label") for a in (agent.memory.get_working_actions(session_id=sid) or [])}
    assert "Сразу код" not in labels


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


def test_memory_write_feed_records_router_layers() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    uid = "writer_user"

    mem.route_user_message(
        session_id=sid,
        user_id=uid,
        user_message="С этого момента всегда отвечай кратко. Решили использовать SQLite.",
    )
    writes = mem.get_recent_write_events(session_id=sid, limit=10)
    layers = {w.get("layer") for w in writes}
    assert "long_term.profile" not in layers
    assert "long_term.decision" in layers
    # Пассивное сохранение short-term turn не должно попадать в индикатор save policy.
    assert "short_term" not in layers


def test_passive_chat_does_not_update_profile_after_20_messages() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    uid = "no_profile_passive"
    before = mem.long_term.get_profile(user_id=uid)
    for i in range(20):
        mem.route_user_message(
            session_id=sid,
            user_id=uid,
            user_message=f"С этого момента всегда правило #{i}",
        )
    after = mem.long_term.get_profile(user_id=uid)
    assert before == after


def test_task_artifacts_do_not_leak_into_profile_fields() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    uid = "artifact_profile_guard"

    mem.working.start_task(session_id=sid, goal="Task")
    mem.working.update(session_id=sid, plan=["A"], current_step="A")
    ctx = mem.working.load(sid)
    assert ctx is not None
    mem.working.transition_state(ctx, TaskState.EXECUTION)
    mem.working.save(ctx)
    mem.working.complete_current_step(
        sid,
        artifact={"step": "A", "type": "response", "ref": "artifact://result"},
    )

    profile = mem.long_term.get_profile(user_id=uid)
    assert profile["stack_tools"]["value"] == []
    assert profile["hard_constraints"]["value"] == []
    assert profile["project_context"]["value"]["key_decisions"] == []


def test_clear_working_layer_does_not_touch_other_layers() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    uid = "clear_working_user"
    mem.short_term.append(sid, "user", "short-term message")
    mem.working.start_task(session_id=sid, goal="Сделать анализ")
    mem.long_term.add_decision(uid, "Используем консервативный бюджет", tags=["budget"])

    cleared = mem.clear_working_layer(session_id=sid)
    assert cleared is True
    assert mem.working.load(sid) is None
    assert len(mem.short_term.get_context(sid)) == 1
    assert len(mem.long_term.retrieve(user_id=uid, query="budget", top_k=3)["decisions"]) >= 1


def test_delete_long_term_entry_does_not_touch_short_or_working() -> None:
    _setup_temp_db()
    mem = MemoryManager(short_term_limit=5)
    sid = storage.create_session()
    uid = "delete_lt_user"
    mem.short_term.append(sid, "user", "keep me")
    mem.working.start_task(session_id=sid, goal="Задача")
    mem.working.update(sid, plan=["Шаг 1"], current_step="Шаг 1")
    mem.long_term.add_note(uid, "Проверять категорию подписок", tags=["subscriptions"])

    lt_before = mem.long_term.retrieve(user_id=uid, query="подписок", top_k=3)
    assert lt_before["notes"], "expected at least one long-term note before delete"
    note_id = int(lt_before["notes"][0]["id"])

    deleted = mem.delete_long_term_entry(
        session_id=sid,
        user_id=uid,
        entry_type="note",
        entry_id=note_id,
    )
    assert deleted is True
    assert len(mem.short_term.get_context(sid)) == 1
    assert mem.working.load(sid) is not None
    lt_after = mem.long_term.retrieve(user_id=uid, query="подписок", top_k=3)
    assert all(int(n["id"]) != note_id for n in lt_after["notes"])


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
    test_working_done_state_is_persisted_and_frozen()
    test_strict_planning_gate()
    test_strict_planning_gate_financial_trigger()
    test_planning_gate_financial_trigger_allows_after_plan()
    test_planning_gate_auto_transitions_when_ready()
    test_router_autofills_current_step_from_first_added_step()
    test_router_autofill_current_step_does_not_override_existing()
    test_router_llm_patch_updates_working_without_regex_markers()
    test_router_llm_invalid_json_falls_back_to_regex()
    test_router_llm_low_confidence_patch_is_ignored()
    test_working_load_repairs_legacy_current_step_and_persists()
    test_working_load_repairs_legacy_current_step_when_all_done()
    test_agent_chat_populates_working_current_step_via_mock_llm()
    test_memory_debug_snapshot_short_term_shape()
    test_memory_debug_snapshot_working_present_and_empty()
    test_memory_debug_snapshot_long_term_top_k()
    test_memory_debug_snapshot_full_truncation_flag()
    test_memory_write_feed_records_router_layers()
    test_passive_chat_does_not_update_profile_after_20_messages()
    test_task_artifacts_do_not_leak_into_profile_fields()
    test_clear_working_layer_does_not_touch_other_layers()
    test_delete_long_term_entry_does_not_touch_short_or_working()
    print("\n🎉 Memory layers tests passed")
