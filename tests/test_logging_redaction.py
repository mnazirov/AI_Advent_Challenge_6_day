from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")
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


def test_redact_messages_for_log_shape() -> None:
    agent = _make_agent()
    redacted = agent._redact_messages_for_log(
        [
            {"role": "system", "content": "SYSTEM SECRET"},
            {"role": "user", "content": "USER SECRET"},
        ]
    )
    assert redacted[0]["role"] == "system"
    assert redacted[0]["chars"] == len("SYSTEM SECRET")
    assert len(redacted[0]["hash"]) == 12
    assert "content" not in redacted[0]
    assert redacted[1]["role"] == "user"
    assert redacted[1]["chars"] == len("USER SECRET")


def test_agent_source_has_messages_redacted_in_all_three_points() -> None:
    source = Path("agent.py").read_text(encoding="utf-8")
    assert source.count("messages_redacted") >= 3
    assert '"messages": messages' not in source
    assert '"messages": route_messages' not in source
    assert '"messages": [{"role": "user", "content": prompt}]' not in source


def test_router_request_log_does_not_dump_raw_message_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = _make_agent()
    agent._create_chat_completion = lambda **kwargs: _fake_chat_response(
        '{"needs_data": false, "reason":"ok", "expense_scope":"overview", "context_profile":"light", "queries":[]}'
    )
    secret_marker = "VERY_SECRET_USER_MARKER_123"

    with caplog.at_level("INFO", logger="agent"):
        decision = agent._route(secret_marker)

    assert decision["needs_data"] is False
    assert "messages_redacted" in caplog.text
    assert secret_marker not in caplog.text
