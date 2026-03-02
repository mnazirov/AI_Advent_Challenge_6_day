from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from agent import FinancialAgent


def _make_agent() -> FinancialAgent:
    agent = FinancialAgent.__new__(FinancialAgent)
    agent.model = "gpt-5-mini"
    agent.summary_sections = {}
    agent.csv_summary = ""
    agent.expense_cache = {}
    agent.df = None
    agent.conversation_history = []
    return agent


def _fake_chat_response(content: str):
    return SimpleNamespace(
        id="resp_test",
        model="gpt-5-mini",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
    )


def test_route_expense_scope_mapping() -> None:
    agent = _make_agent()
    agent.df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-11-01", "2024-11-02"]),
            "amount": [1000, 2500],
            "category": ["Продукты", "Транспорт"],
            "description": ["Пятерочка", "Метро"],
            "op_type": ["расход", "расход"],
        }
    )
    agent._create_chat_completion = lambda **kwargs: _fake_chat_response(
        '{"needs_data": true, "reason":"period analysis", "expense_scope":"time_trend", '
        '"context_profile":"deep", "queries":[{"type":"by_period","month":"2024-11"}]}'
    )

    decision = agent._route("Почему в ноябре расходы выше?")
    assert decision["needs_data"] is True
    assert decision["expense_scope"] == "time_trend"
    assert decision["context_profile"] == "deep"
    assert len(decision["queries"]) == 1


def test_route_scope_fallback_when_missing_fields() -> None:
    agent = _make_agent()
    agent.df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-12-01"]),
            "amount": [1200],
            "category": ["Подписки"],
            "description": ["Spotify"],
            "op_type": ["расход"],
        }
    )
    agent._create_chat_completion = lambda **kwargs: _fake_chat_response(
        '{"needs_data": true, "reason":"details", "queries":[{"type":"by_description","keyword":"подпис"}]}'
    )

    decision = agent._route("Покажи детали по подпискам")
    assert decision["expense_scope"] == "category_breakdown"
    assert decision["context_profile"] == "medium"


def test_compose_system_context_section_selection() -> None:
    agent = _make_agent()
    agent.summary_sections = {
        "overview": "OVERVIEW",
        "expense_categories": "CATEGORIES",
        "monthly_dynamics": "MONTHLY",
        "top_expenses": "TOP",
        "anomalies": "ANOMALIES",
    }

    overview_ctx = agent._compose_system_context({"expense_scope": "overview"})
    assert "OVERVIEW" in overview_ctx and "CATEGORIES" in overview_ctx and "TOP" in overview_ctx
    assert "MONTHLY" not in overview_ctx

    trend_ctx = agent._compose_system_context({"expense_scope": "time_trend"})
    assert "OVERVIEW" in trend_ctx and "MONTHLY" in trend_ctx and "ANOMALIES" in trend_ctx
    assert "TOP" not in trend_ctx

    detail_ctx = agent._compose_system_context({"expense_scope": "merchant_detail"})
    assert "OVERVIEW" in detail_ctx and "CATEGORIES" in detail_ctx
    assert "TOP" not in detail_ctx and "MONTHLY" not in detail_ctx


def test_detail_pack_token_budget() -> None:
    agent = _make_agent()
    rows = 80
    agent.df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "amount": [1000 + i * 17 for i in range(rows)],
            "category": ["Продукты"] * rows,
            "description": [("Очень длинное описание транзакции " * 12) + str(i) for i in range(rows)],
            "op_type": ["расход"] * rows,
        }
    )
    agent.expense_cache = agent._build_expense_cache(agent.df)

    detail = agent._fetch_detail(
        [{"type": "top_expenses", "top_n": 50, "sort_by": "amount_desc"}],
        {"context_profile": "deep", "expense_scope": "merchant_detail"},
    )
    assert detail
    assert "Summary metrics" in detail
    assert len(detail) <= agent.MAX_DETAIL_CHARS


def test_expense_only_filter_in_detail() -> None:
    agent = _make_agent()
    agent.df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-11-01", "2024-11-02", "2024-11-03"]),
            "amount": [1900, 4500, 120000],
            "category": ["Продукты", "Продукты", "Продукты"],
            "description": ["Пятерочка", "Перекресток", "SALARYMARK"],
            "op_type": ["расход", "расход", "доход"],
        }
    )
    agent.expense_cache = agent._build_expense_cache(agent.df)

    detail = agent._fetch_detail(
        [{"type": "by_category", "category": "продукт", "sort_by": "amount_desc"}],
        {"context_profile": "deep", "expense_scope": "category_breakdown"},
    )
    assert "SALARYMARK" not in detail
    assert "Пятерочка" in detail or "Перекресток" in detail


def test_light_profile_skips_detail_pack() -> None:
    agent = _make_agent()
    agent.df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-11-01", "2024-11-02"]),
            "amount": [1900, 4500],
            "category": ["Продукты", "Транспорт"],
            "description": ["Пятерочка", "Метро"],
            "op_type": ["расход", "расход"],
        }
    )
    detail = agent._fetch_detail(
        [{"type": "top_expenses", "top_n": 5, "sort_by": "amount_desc"}],
        {"context_profile": "light", "expense_scope": "overview"},
    )
    assert detail == ""


def test_medium_profile_has_no_sample_transactions() -> None:
    agent = _make_agent()
    agent.df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-11-01", "2024-11-02", "2024-11-03"]),
            "amount": [1900, 4500, 2100],
            "category": ["Продукты", "Транспорт", "Продукты"],
            "description": ["Пятерочка", "Метро", "Перекресток"],
            "op_type": ["расход", "расход", "расход"],
        }
    )
    detail = agent._fetch_detail(
        [{"type": "top_expenses", "top_n": 5, "sort_by": "amount_desc"}],
        {"context_profile": "medium", "expense_scope": "category_breakdown"},
    )
    assert "Summary metrics" in detail
    assert "Aggregates: categories" in detail
    assert "#### Sample transactions" not in detail
