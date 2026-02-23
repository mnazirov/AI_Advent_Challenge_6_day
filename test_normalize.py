"""
Запуск: python test_normalize.py
Проверяет нормализацию на 4 реальных форматах банковских выгрузок.
"""
from agent import FinancialAgent

agent = FinancialAgent()

# Формат 1: Дзен-банк (разделитель ;, суммы "1 234,56")
CSV_ZEN = """Дата операции;Сумма операции;Категория;Описание
22.10.2024;-1 250,00;Проезд;Метро
15.11.2024;85 000,00;Зарплата;ООО Ромашка
03.12.2024;-3 499,00;Подписки;Spotify
"""

# Формат 2: Сбер (две колонки Приход / Расход)
CSV_SBER = """Дата,Описание,Приход,Расход,Категория
15.11.2024,Зарплата,85000,,Зарплата
03.12.2024,Spotify,,3499,Подписки
22.10.2024,Метро,,1250,Транспорт
"""

# Формат 3: Тинькофф (amount со знаком, разделитель ,)
CSV_TINKOFF = """Date,Amount,Category,Description
2024-11-15,85000.00,Зарплата,ООО Ромашка
2024-12-03,-3499.00,Подписки,Spotify
2024-10-22,-1250.00,Транспорт,Метро
"""

# Формат 4: Альфа (разделитель ;, суммы в кавычках "1 234,56")
CSV_ALFA = """дата;сумма;тип;категория
15.11.2024;"85 000,00";Зачисление;Зарплата
03.12.2024;"-3 499,00";Списание;Подписки
22.10.2024;"-1 250,00";Списание;Транспорт
"""


def test(name, csv_str, encoding="utf-8"):
    result = agent.load_csv(csv_str.encode(encoding), f"test_{name}.csv")
    assert result["success"], f"[{name}] ОШИБКА: {result.get('error')}"
    a = result["analysis"]
    income = a.get("total_income", 0) or 0
    expenses = a.get("total_expenses", 0) or 0
    print(
        f"✅ {name:10} | schema={a['schema_source']:7} | "
        f"доходы={income:>10,.0f} ₽ | расходы={expenses:>10,.0f} ₽ | "
        f"строк={a['rows']}"
    )
    assert income > 1000, f"[{name}] total_income подозрительно маленький: {income}"
    assert expenses > 100, f"[{name}] total_expenses подозрительно маленький: {expenses}"


if __name__ == "__main__":
    print("─" * 75)
    test("zen", CSV_ZEN)
    test("sber", CSV_SBER)
    test("tinkoff", CSV_TINKOFF)
    test("alfa", CSV_ALFA)
    print("─" * 75)
    print("🎉 Все форматы прошли проверку")
