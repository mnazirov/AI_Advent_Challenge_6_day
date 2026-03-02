"""
Запуск: python tests/test_strategies.py
Проверяет стратегии управления контекстом и менеджер стратегий.
"""

from unittest.mock import MagicMock

from context_strategies import (
    BranchingStrategy,
    ContextStrategyManager,
    HistoryCompressionStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)


def make_history(n: int) -> list[dict]:
    history = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": f"Сообщение {i + 1}"})
    return history


def test_sliding_window() -> None:
    strategy = SlidingWindowStrategy()
    history = make_history(30)
    context = strategy.build_context(history)
    assert len(context) == strategy.WINDOW_SIZE
    assert context[0]["content"] == f"Сообщение {30 - strategy.WINDOW_SIZE + 1}"

    stats = strategy.stats(history)
    assert stats["strategy"] == "sliding_window"
    assert stats["dropped"] == 20

    strategy.reset()
    strategy.restore({"unused": True})
    assert strategy.dump() == {}


def test_sticky_facts() -> None:
    mock_client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = (
        '{"goal":"накопить 200000","constraints":"нерегулярный доход",'
        '"preferences":"не трогать путешествия","decisions":["лимит 8000"],'
        '"agreements":["чек по воскресеньям"],"profile":"москва"}'
    )
    response.usage.total_tokens = 111
    mock_client.chat.completions.create.return_value = response

    strategy = StickyFactsStrategy(client=mock_client)
    history = make_history(12)
    strategy.update_facts("Хочу экономить", history)

    assert strategy.facts["goal"] == "накопить 200000"
    assert "лимит 8000" in strategy.facts["decisions"]

    context = strategy.build_context(history)
    assert context[0]["role"] == "system"
    assert "USER MEMORY" in context[0]["content"]
    assert len([m for m in context if m["role"] != "system"]) == min(len(history), strategy.RECENT_N)

    dumped = strategy.dump()
    strategy.reset()
    assert strategy.facts["goal"] == ""
    strategy.restore(dumped)
    assert strategy.facts["goal"] == "накопить 200000"


def test_branching() -> None:
    strategy = BranchingStrategy()
    strategy.add_message("user", "Привет")
    strategy.add_message("assistant", "Здравствуйте")

    cp = strategy.create_checkpoint("cp_1")
    assert cp["msg_index"] == 2

    new_branch = strategy.fork("cp_1")
    strategy.switch_branch(new_branch)
    strategy.add_message("user", "Новая ветка")

    context = strategy.build_context([])
    assert context[-1]["content"] == "Новая ветка"
    assert strategy.active_branch == new_branch

    stats = strategy.stats()
    assert stats["strategy"] == "branching"
    assert len(stats["branches"]) == 2


def test_history_compression() -> None:
    mock_client = MagicMock()
    strategy = HistoryCompressionStrategy(client=mock_client)

    # Подменяем LLM-вызов, чтобы тест не зависел от API.
    strategy._compress_chunk = lambda chunk: f"Резюме из {len(chunk)} сообщений"

    short_history = make_history(8)
    short_context = strategy.build_context(short_history)
    assert len(short_context) == 8
    assert strategy.summary == ""

    long_history = make_history(21)
    long_context = strategy.build_context(long_history)
    assert strategy.summarized_up_to == strategy.CHUNK_SIZE
    assert strategy.summary.startswith("Резюме")
    assert long_context[0]["role"] == "system"

    dumped = strategy.dump()
    strategy.reset()
    assert strategy.summary == ""
    strategy.restore(dumped)
    assert strategy.summarized_up_to == strategy.CHUNK_SIZE


def test_context_strategy_manager() -> None:
    mock_client = MagicMock()
    manager = ContextStrategyManager(client=mock_client)
    history = make_history(25)

    assert manager.active == "sticky_facts"
    assert len(manager.build_context(history)) == 25

    manager.set_strategy("branching")
    manager.strategy.add_message("user", "ветка")
    assert manager.stats(history)["strategy"] == "branching"

    dumped = manager.dump()
    restored = ContextStrategyManager(client=mock_client)
    restored.restore(dumped)
    assert restored.active == "branching"

    try:
        manager.set_strategy("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Ожидали ValueError для неизвестной стратегии")


if __name__ == "__main__":
    test_sliding_window()
    print("✅ Sliding Window")

    test_sticky_facts()
    print("✅ Sticky Facts")

    test_branching()
    print("✅ Branching")

    test_history_compression()
    print("✅ History Compression")

    test_context_strategy_manager()
    print("✅ ContextStrategyManager")

    print("\n🎉 Все тесты стратегий пройдены")
