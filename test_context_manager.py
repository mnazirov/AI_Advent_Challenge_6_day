"""
Запуск: python test_context_manager.py
Проверяет что ContextManager корректно сжимает историю.
"""
from unittest.mock import MagicMock

from context_manager import CHUNK_SIZE, RECENT_N, ContextManager

# ── Мок OpenAI клиента ──
mock_client = MagicMock()
mock_response = MagicMock()
mock_response.choices[0].message.content = "Тестовое резюме диалога."
mock_response.usage.total_tokens = 150
mock_response.usage.completion_tokens = 50
mock_client.chat.completions.create.return_value = mock_response

ctx = ContextManager(client=mock_client)


def make_history(n):
    """Генерирует историю из n сообщений (user/assistant чередуются)."""
    history = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": f"Сообщение {i + 1}"})
    return history


# Тест 1: короткая история — summary не создаётся
history = make_history(8)
result = ctx.build_context(history)
assert ctx.summary == "", "При < RECENT_N сообщений summary не должен создаваться"
assert len(result) == 8, "Все сообщения должны быть в recent"
print("✅ Тест 1: короткая история — summary не создаётся")

# Тест 2: достаточно для первого chunk
ctx.reset()
history = make_history(CHUNK_SIZE + RECENT_N + 1)  # 21 сообщение
result = ctx.build_context(history)
assert ctx.summary != "", "Summary должен быть создан"
assert ctx.summarized_up_to == CHUNK_SIZE, f"Должно быть свёрнуто {CHUNK_SIZE} сообщений"
# В result должен быть summary-блок + последние RECENT_N
summary_msgs = [m for m in result if '[ИСТОРИЯ ДИАЛОГА' in m.get('content', '')]
assert len(summary_msgs) == 1, "Должен быть ровно один summary-блок"
recent_msgs = [m for m in result if '[ИСТОРИЯ ДИАЛОГА' not in m.get('content', '')]
assert len(recent_msgs) == RECENT_N, f"Должно быть {RECENT_N} recent-сообщений"
print(f"✅ Тест 2: первый chunk свёрнут, summarized_up_to={ctx.summarized_up_to}")

# Тест 3: restore восстанавливает состояние
ctx2 = ContextManager(client=mock_client)
ctx2.restore(summary="Восстановленное резюме", summarized_up_to=20)
assert ctx2.summary == "Восстановленное резюме"
assert ctx2.summarized_up_to == 20
print("✅ Тест 3: restore работает корректно")

# Тест 4: reset очищает всё
ctx2.reset()
assert ctx2.summary == ""
assert ctx2.summarized_up_to == 0
print("✅ Тест 4: reset очищает состояние")

# Тест 5: при ошибке LLM summary не ломается
ctx3 = ContextManager(client=mock_client)
mock_client.chat.completions.create.side_effect = Exception("API error")
history = make_history(CHUNK_SIZE + RECENT_N + 1)
result = ctx3.build_context(history)  # не должен бросить исключение
assert ctx3.summary == "", "При ошибке LLM summary должен остаться пустым"
print("✅ Тест 5: ошибка LLM обрабатывается без краша")

print("\n🎉 Все тесты пройдены")
