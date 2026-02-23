"""
Запуск: python test_planner.py
Проверяет полный planning-цикл на тестовых данных.
"""
import pandas as pd
from openai import OpenAI

from planner import FinancialPlanner

client = OpenAI()
planner = FinancialPlanner(client=client)

# Тестовый DataFrame (минимальный)
df = pd.DataFrame(
    {
        "date": ["2024-11-15", "2024-12-03", "2024-10-22", "2025-01-10", "2025-01-15"],
        "amount": [85000, 3499, 1250, 85000, 12000],
        "category": ["Зарплата", "Подписки", "Транспорт", "Зарплата", "Рестораны"],
        "description": ["ООО Ромашка", "Spotify", "Метро", "ООО Ромашка", "Кафе Пушкин"],
        "op_type": ["доход", "расход", "расход", "доход", "расход"],
    }
)

csv_summary = "Тестовые данные: 5 транзакций, 2 месяца."

# Тест 1: planning-запрос → должен вернуть план
result = planner.run("Составь план как мне сэкономить", df, csv_summary)
assert result is not None, "planning-запрос должен вернуть план"
assert "PLANNING" in result, "ответ должен содержать PLANNING"
print("✅ Тест 1 пройден: planning-запрос обработан")

# Тест 2: не-planning-запрос → должен вернуть None
result = planner.run("Сколько я потратил в декабре?", df, csv_summary)
assert result is None, "аналитический запрос должен вернуть None"
print("✅ Тест 2 пройден: аналитический запрос корректно отклонён")

print("🎉 Все тесты пройдены")
