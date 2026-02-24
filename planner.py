from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from typing import Any, Callable

import pandas as pd
from openai import OpenAI

logger = logging.getLogger("planner")


def _df_to_md(df: pd.DataFrame) -> str:
    """Конвертирует DataFrame в markdown-таблицу без внешних зависимостей."""
    if df.empty:
        return "_Нет данных._\n"

    work_df = df.copy()
    for col in work_df.select_dtypes(include="number").columns:
        work_df[col] = work_df[col].apply(
            lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "—"
        )

    headers = list(work_df.columns)
    sep = ["---"] * len(headers)
    rows = []
    for _, row in work_df.iterrows():
        cells = [str(v) if pd.notna(v) else "—" for v in row]
        rows.append(cells)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ] + ["| " + " | ".join(r) + " |" for r in rows]

    return "\n".join(lines) + "\n"


class FinancialPlanner:
    """Planning Agent: определяет цель, декомпозирует анализ и синтезирует план."""

    AVAILABLE_TOOLS = {
        "top_expenses_by_category": "Топ категорий расходов по сумме",
        "monthly_dynamics": "Доходы и расходы по месяцам",
        "top_transactions": "Топ-N крупнейших транзакций",
        "category_detail": "Детальные транзакции по конкретной категории",
        "savings_rate": "Расчёт нормы сбережений по месяцам",
        "income_stability": "Анализ стабильности доходов",
        "expense_volatility": "Анализ волатильности расходов по категориям",
        "anomaly_months": "Выявление аномальных месяцев",
        "recurring_expenses": "Регулярные расходы (аренда, подписки)",
    }

    COST_PER_1M = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.00, "output": 15.00},
    }

    def __init__(self, client: OpenAI, model: str = "gpt-5.2"):
        self.client = client
        self.model = model
        self.on_step: Callable[[dict[str, str]], None] | None = None
        self._run_calls = 0
        self._run_prompt_tokens = 0
        self._run_completion_tokens = 0
        self._run_total_cost = 0.0

    def run(self, user_message: str, df: pd.DataFrame, csv_summary: str) -> str | None:
        """Запускает planning-цикл. Возвращает None, если запрос не планировочный."""
        self._run_calls = 0
        self._run_prompt_tokens = 0
        self._run_completion_tokens = 0
        self._run_total_cost = 0.0
        started_total = perf_counter()

        goal = self._detect_goal(user_message, csv_summary)
        if goal is None:
            return None

        logger.info(
            '[PLANNER] Старт | goal="%s" focus=%s horizon=%s',
            goal.get("goal_text", ""),
            goal.get("focus_area", "не указан"),
            goal.get("horizon", "не указан"),
        )

        tasks = self._decompose(goal, csv_summary)
        results = self._execute_all(tasks, df)

        replan = self._replan(goal, tasks, results)
        if replan.get("need_more") and replan.get("extra_tasks"):
            extra_tasks = replan.get("extra_tasks", [])[:2]
            extra_results = self._execute_all(extra_tasks, df)
            results.extend(extra_results)

        final_text = self._synthesize(goal, results, csv_summary)
        analysis_facts = self._build_analysis_facts(df)
        final_text = self._verify_and_fix_plan(
            draft_text=final_text,
            goal=goal,
            analysis_facts=analysis_facts,
            csv_summary=csv_summary,
        )
        final_text = self._sanitize_reply_text(final_text)

        latency_total_ms = int((perf_counter() - started_total) * 1000)
        total_tokens = self._run_prompt_tokens + self._run_completion_tokens
        logger.info(
            "[PLANNER] Завершён | всего_вызовов=%s всего_токенов=%s общая_стоимость=$%.6f latency_ms=%s",
            self._run_calls,
            total_tokens,
            self._run_total_cost,
            latency_total_ms,
        )
        return final_text

    def _detect_goal(self, user_message: str, csv_summary: str) -> dict[str, Any] | None:
        """Определяет, требуется ли planning-цикл для текущего запроса."""
        self._emit("goal", "running", "Определяю цель запроса")

        csv_summary_short = (csv_summary or "")[:500]
        prompt = f"""Ты — классификатор запросов финансового агента.

Запрос пользователя: \"{user_message}\"

Доступный контекст данных (краткая сводка):
{csv_summary_short}

Определи: требует ли этот запрос составления финансового плана?

Planning-запрос (needs_planning=true):
- Пользователь просит составить план, стратегию, roadmap
- Пользователь спрашивает \"что мне делать\" / \"с чего начать\"
- Пользователь хочет оптимизировать / улучшить / исправить ситуацию
- Пользователь просит совет с конкретными шагами

НЕ planning-запрос (needs_planning=false):
- Фактический вопрос о данных (\"сколько потратил\", \"покажи категории\")
- Аналитический вопрос (\"в каком месяце больше трат\")
- Уточняющий вопрос (\"а что с подарками?\")
- Приветствие или общий вопрос

Если needs_planning=true, извлеки:
- goal_text: цель в одном предложении
- horizon: горизонт планирования (\"неделя\" / \"месяц\" / \"квартал\" / \"год\" / \"не указан\")
- focus_area: основная область (\"расходы\" / \"доходы\" / \"сбережения\" / \"долги\" / \"общее\")
- constraint: ограничение если есть (\"сократить на 20%\" / \"откладывать 10000 в месяц\" / null)

Ответь ТОЛЬКО валидным JSON:
{{
  \"needs_planning\": true/false,
  \"goal_text\": \"...\",
  \"horizon\": \"...\",
  \"focus_area\": \"...\",
  \"constraint\": \"... или null\"
}}"""

        try:
            raw, meta = self._call_llm(
                scope="planner_goal",
                prompt=prompt,
                max_completion_tokens=150,
                temperature=0,
            )
            data = self._parse_json(raw, fallback={})
            needs_planning = bool(data.get("needs_planning")) if isinstance(data, dict) else False

            logger.info(
                "[PLANNER][GOAL] needs_planning=%s | latency_ms=%s",
                needs_planning,
                meta.get("latency_ms", 0),
            )

            if not needs_planning:
                self._emit("goal", "skip", "Planning-цикл не требуется")
                return None

            goal = {
                "needs_planning": True,
                "goal_text": data.get("goal_text") or user_message,
                "horizon": data.get("horizon") or "не указан",
                "focus_area": data.get("focus_area") or "общее",
                "constraint": data.get("constraint"),
            }
            self._emit("goal", "done", "Цель определена", goal.get("goal_text", ""))
            return goal
        except Exception as exc:
            logger.warning("[PLANNER][GOAL] Ошибка: %s", exc)
            self._emit("goal", "error", "Не удалось определить цель", str(exc))
            return None

    def _decompose(self, goal: dict[str, Any], csv_summary: str) -> list[dict[str, Any]]:
        """Разбивает цель на 2-5 аналитических подзадач."""
        self._emit("decompose", "running", "Декомпозиция задач")

        tools_list = "\n".join([f"- {k}: {v}" for k, v in self.AVAILABLE_TOOLS.items()])
        prompt = f"""Ты — аналитический планировщик финансового агента.

ЦЕЛЬ ПОЛЬЗОВАТЕЛЯ: {goal.get("goal_text", "")}
ГОРИЗОНТ: {goal.get("horizon", "не указан")}
ФОКУС: {goal.get("focus_area", "общее")}
ОГРАНИЧЕНИЕ: {goal.get("constraint")}

СВОДКА ДАННЫХ:
{csv_summary}

ДОСТУПНЫЕ АНАЛИТИЧЕСКИЕ ИНСТРУМЕНТЫ:
{tools_list}

Разбей цель на 2-5 подзадач. Каждая подзадача — это конкретный анализ данных
который нужно провести чтобы составить обоснованный план.

Правила:
- Начинай с самых важных задач (высокий приоритет = выполняется первым)
- Не дублируй задачи с похожим смыслом
- Каждая задача должна добавлять новую информацию к плану
- Максимум 5 задач

Ответь ТОЛЬКО валидным JSON — массивом задач:
[
  {{
    "id": 1,
    "tool": "название_инструмента",
    "description": "что именно нужно узнать и зачем",
    "params": {{
      "category": "название категории или null",
      "top_n": 10,
      "sort_by": "amount_desc"
    }},
    "priority": "high|medium|low"
  }}
]"""

        try:
            raw, meta = self._call_llm(
                scope="planner_decompose",
                prompt=prompt,
                max_completion_tokens=400,
                temperature=0,
            )
            parsed = self._parse_json(raw, fallback=[])
            tasks = parsed if isinstance(parsed, list) else []

            prepared: list[dict[str, Any]] = []
            used_tools: set[str] = set()
            for idx, task in enumerate(tasks, start=1):
                if not isinstance(task, dict):
                    continue
                tool = str(task.get("tool", "")).strip()
                if tool not in self.AVAILABLE_TOOLS:
                    continue
                if tool in used_tools:
                    continue
                used_tools.add(tool)
                prepared.append(
                    {
                        "id": int(task.get("id") or idx),
                        "tool": tool,
                        "description": str(task.get("description") or self.AVAILABLE_TOOLS[tool]),
                        "params": task.get("params") if isinstance(task.get("params"), dict) else {},
                        "priority": str(task.get("priority") or "medium"),
                    }
                )

            if not prepared:
                prepared = self._default_tasks()

            priority_rank = {"high": 0, "medium": 1, "low": 2}
            prepared = sorted(
                prepared[:5],
                key=lambda t: priority_rank.get(str(t.get("priority", "medium")).lower(), 1),
            )

            logger.info("[PLANNER][DECOMP] tasks=%s | latency_ms=%s", len(prepared), meta.get("latency_ms", 0))
            self._emit("decompose", "done", f"Сформировано {len(prepared)} задач", f"{len(prepared)} шагов анализа")
            return prepared
        except Exception as exc:
            logger.warning("[PLANNER][DECOMP] Ошибка: %s — использую fallback", exc)
            self._emit("decompose", "error", "Ошибка декомпозиции", str(exc))
            return self._default_tasks()

    def _execute_all(self, tasks: list[dict[str, Any]], df: pd.DataFrame) -> list[dict[str, Any]]:
        """Последовательно выполняет все задачи и возвращает markdown-результаты."""
        results: list[dict[str, Any]] = []
        total = len(tasks)
        if total == 0:
            return results

        for i, task in enumerate(tasks, start=1):
            tool = str(task.get("tool", ""))
            self._emit(f"execute_{i}", "running", f"Анализ данных ({i}/{total})", tool)
            try:
                result = self._execute_task(task, df)
                row_count = self._count_md_rows(result)
                logger.info(
                    "[PLANNER][EXEC] task=%s/%s tool=%s rows=%s | ok",
                    i,
                    total,
                    tool,
                    row_count,
                )
                self._emit(
                    f"execute_{i}",
                    "done",
                    f"Готово ({i}/{total})",
                    str(task.get("description") or tool),
                )
                results.append({"task": task, "result": result})
            except Exception as exc:
                logger.warning("[PLANNER][EXEC] task=%s/%s tool=%s | error=%s", i, total, tool, exc)
                self._emit(f"execute_{i}", "error", f"Ошибка ({i}/{total})", tool)
                results.append({"task": task, "result": f"_Не удалось выполнить: {tool}_"})

        return results

    def _execute_task(self, task: dict[str, Any], df: pd.DataFrame) -> str:
        """Выполняет одну задачу через pandas и возвращает markdown-результат."""
        tool = str(task.get("tool", ""))
        params = task.get("params", {}) if isinstance(task.get("params"), dict) else {}
        top_n = int(params.get("top_n", 10) or 10)
        top_n = max(1, min(top_n, 50))

        work_df = df.copy()

        try:
            if tool == "top_expenses_by_category":
                if "amount" not in work_df.columns or "category" not in work_df.columns:
                    return "### Топ категорий расходов\n_Недостаточно данных (нужны amount и category)._\n"

                expenses = (
                    work_df[work_df["op_type"] == "расход"].copy()
                    if "op_type" in work_df.columns
                    else work_df.copy()
                )
                grouped = (
                    expenses.groupby("category", dropna=False)["amount"]
                    .agg(["sum", "count", "mean"])
                    .sort_values("sum", ascending=False)
                    .head(top_n)
                    .reset_index()
                )
                if grouped.empty:
                    return "### Топ категорий расходов\n_Нет данных._\n"

                grouped.columns = ["Категория", "Сумма ₽", "Транзакций", "Среднее ₽"]
                total_sum = float(pd.to_numeric(expenses["amount"], errors="coerce").fillna(0).sum())
                grouped["% от расходов"] = (
                    (grouped["Сумма ₽"] / total_sum * 100).fillna(0).round(1).astype(str) + "%"
                    if total_sum
                    else "0.0%"
                )
                return f"### Топ-{top_n} категорий расходов\n" + _df_to_md(grouped)

            if tool == "monthly_dynamics":
                if "date" not in work_df.columns or "amount" not in work_df.columns:
                    return "### Динамика по месяцам\n_Недостаточно данных (нужны date и amount)._\n"

                work_df["_month"] = pd.to_datetime(work_df["date"], errors="coerce").dt.to_period("M")
                data = work_df.dropna(subset=["_month"])
                if data.empty:
                    return "### Динамика по месяцам\n_Нет валидных дат для анализа._\n"

                if "op_type" in data.columns:
                    inc = data[data["op_type"] == "доход"].groupby("_month")["amount"].sum()
                    exp = data[data["op_type"] == "расход"].groupby("_month")["amount"].sum()
                else:
                    inc = data[data["amount"] >= 0].groupby("_month")["amount"].sum()
                    exp = data[data["amount"] < 0].groupby("_month")["amount"].sum().abs()

                dyn = pd.DataFrame({"Доход": inc, "Расход": exp}).fillna(0)
                dyn["Баланс"] = dyn["Доход"] - dyn["Расход"]
                dyn.index = dyn.index.astype(str)
                dyn = dyn.reset_index().rename(columns={"_month": "Месяц"})
                return "### Динамика по месяцам\n" + _df_to_md(dyn)

            if tool == "top_transactions":
                if "amount" not in work_df.columns:
                    return "### Топ крупнейших трат\n_Нет колонки amount._\n"

                expenses = (
                    work_df[work_df["op_type"] == "расход"].copy()
                    if "op_type" in work_df.columns
                    else work_df.copy()
                )
                top = expenses.nlargest(top_n, "amount")
                cols = [c for c in ["date", "amount", "category", "description"] if c in top.columns]
                top = top[cols]
                top.columns = ["Дата", "Сумма ₽", "Категория", "Описание"][: len(cols)]
                return f"### Топ-{top_n} крупнейших трат\n" + _df_to_md(top)

            if tool == "category_detail":
                if "category" not in work_df.columns:
                    return "### Детализация категории\n_Нет колонки category._\n"

                cat = str(params.get("category", "")).strip()
                if not cat:
                    return "### Детализация категории\n_Не передана категория._\n"

                mask = work_df["category"].astype(str).str.lower().str.contains(cat.lower(), na=False)
                filtered_all = work_df[mask].copy()
                if filtered_all.empty:
                    return f"### Транзакции «{cat}»\n_Нет данных по категории._\n"

                total_period = float(pd.to_numeric(filtered_all["amount"], errors="coerce").fillna(0).sum())
                months_with_spending = 0
                if "date" in filtered_all.columns:
                    date_series = pd.to_datetime(filtered_all["date"], errors="coerce")
                    months_with_spending = int(date_series.dt.to_period("M").dropna().nunique())
                average_per_month = total_period / months_with_spending if months_with_spending > 0 else total_period

                filtered = filtered_all.sort_values("amount", ascending=False).head(top_n)
                cols = [c for c in ["date", "amount", "description"] if c in filtered.columns]
                result = filtered[cols]
                result.columns = ["Дата", "Сумма ₽", "Описание"][: len(cols)]

                summary_df = pd.DataFrame(
                    {
                        "Показатель": ["Сумма за период", "Месяцев с тратами", "Среднее в месяц"],
                        "Значение": [
                            f"{total_period:,.0f} ₽".replace(",", " "),
                            str(months_with_spending) if months_with_spending else "—",
                            f"{average_per_month:,.0f} ₽".replace(",", " "),
                        ],
                    }
                )

                return (
                    f"### Транзакции «{cat}»\n"
                    f"{_df_to_md(summary_df)}\n"
                    f"### Топ-{top_n} транзакций категории\n"
                    f"{_df_to_md(result)}"
                )

            if tool == "savings_rate":
                if "date" not in work_df.columns or "amount" not in work_df.columns:
                    return "### Норма сбережений\n_Недостаточно данных (нужны date и amount)._\n"

                work_df["_month"] = pd.to_datetime(work_df["date"], errors="coerce").dt.to_period("M")
                data = work_df.dropna(subset=["_month"])
                if data.empty:
                    return "### Норма сбережений\n_Нет валидных дат для анализа._\n"

                if "op_type" in data.columns:
                    inc = data[data["op_type"] == "доход"].groupby("_month")["amount"].sum()
                    exp = data[data["op_type"] == "расход"].groupby("_month")["amount"].sum()
                else:
                    inc = data[data["amount"] > 0].groupby("_month")["amount"].sum()
                    exp = data[data["amount"] < 0].groupby("_month")["amount"].sum().abs()

                merged = pd.DataFrame({"income": inc, "expense": exp}).fillna(0)
                merged = merged[merged["income"] > 0]
                if merged.empty:
                    return "### Норма сбережений по месяцам\n_Нет месяцев с доходами._\n"

                rate = ((merged["income"] - merged["expense"]) / merged["income"] * 100).round(1).reset_index()
                rate.columns = ["Месяц", "Норма сбережений %"]
                rate["Месяц"] = rate["Месяц"].astype(str)
                return "### Норма сбережений по месяцам\n" + _df_to_md(rate)

            if tool == "income_stability":
                if "date" not in work_df.columns or "amount" not in work_df.columns:
                    return "### Стабильность доходов\n_Недостаточно данных (нужны date и amount)._\n"

                work_df["_month"] = pd.to_datetime(work_df["date"], errors="coerce").dt.to_period("M")
                data = work_df.dropna(subset=["_month"])

                if "op_type" in data.columns:
                    monthly_inc = data[data["op_type"] == "доход"].groupby("_month")["amount"].sum()
                else:
                    monthly_inc = data[data["amount"] > 0].groupby("_month")["amount"].sum()

                if monthly_inc.empty:
                    return "### Стабильность доходов\n_Нет данных о доходах._\n"

                mean_val = float(monthly_inc.mean())
                std_val = float(monthly_inc.std() or 0)
                stats = {
                    "Среднемесячный доход": f"{mean_val:,.0f} ₽".replace(",", " "),
                    "Минимальный месяц": f"{monthly_inc.min():,.0f} ₽".replace(",", " "),
                    "Максимальный месяц": f"{monthly_inc.max():,.0f} ₽".replace(",", " "),
                    "Стандартное отклонение": f"{std_val:,.0f} ₽".replace(",", " "),
                    "Коэффициент вариации": f"{(std_val / mean_val * 100):.1f}%" if mean_val else "0.0%",
                }
                rows = "\n".join([f"| {k} | {v} |" for k, v in stats.items()])
                return f"### Стабильность доходов\n| Показатель | Значение |\n|---|---|\n{rows}\n"

            if tool == "expense_volatility":
                if "date" not in work_df.columns or "amount" not in work_df.columns or "category" not in work_df.columns:
                    return "### Волатильность расходов\n_Недостаточно данных (нужны date, amount и category)._\n"

                expenses = (
                    work_df[work_df["op_type"] == "расход"].copy()
                    if "op_type" in work_df.columns
                    else work_df.copy()
                )
                expenses["_month"] = pd.to_datetime(expenses["date"], errors="coerce").dt.to_period("M")
                expenses = expenses.dropna(subset=["_month"])
                if expenses.empty:
                    return "### Волатильность расходов\n_Нет валидных дат для анализа._\n"

                vol = (
                    expenses.groupby(["category", "_month"])["amount"]
                    .sum()
                    .groupby("category")
                    .std()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .reset_index()
                )
                vol.columns = ["Категория", "Волатильность (std) ₽"]
                return "### Самые непредсказуемые категории расходов\n" + _df_to_md(vol)

            if tool == "anomaly_months":
                if "date" not in work_df.columns or "amount" not in work_df.columns:
                    return "### Аномальные месяцы\n_Недостаточно данных (нужны date и amount)._\n"

                work_df["_month"] = pd.to_datetime(work_df["date"], errors="coerce").dt.to_period("M")
                data = work_df.dropna(subset=["_month"])
                if "op_type" in data.columns:
                    monthly_exp = data[data["op_type"] == "расход"].groupby("_month")["amount"].sum()
                else:
                    monthly_exp = data[data["amount"] < 0].groupby("_month")["amount"].sum().abs()

                if monthly_exp.empty:
                    return "### Аномальные месяцы\n_Нет данных о расходах._\n"

                mean_exp = monthly_exp.mean()
                std_exp = monthly_exp.std() or 0
                anomalies = monthly_exp[abs(monthly_exp - mean_exp) > std_exp].reset_index()
                anomalies.columns = ["Месяц", "Расход ₽"]
                anomalies["Отклонение"] = ((anomalies["Расход ₽"] - mean_exp) / mean_exp * 100).round(1)
                anomalies["Месяц"] = anomalies["Месяц"].astype(str)
                anomalies["Отклонение"] = anomalies["Отклонение"].astype(str) + "%"
                return "### Аномальные месяцы (отклонение > 1σ)\n" + _df_to_md(anomalies)

            if tool == "recurring_expenses":
                if "description" not in work_df.columns or "date" not in work_df.columns:
                    return "### Регулярные расходы\n_Нет колонки description/date для анализа._\n"

                expenses = (
                    work_df[work_df["op_type"] == "расход"].copy()
                    if "op_type" in work_df.columns
                    else work_df.copy()
                )
                expenses["_month"] = pd.to_datetime(expenses["date"], errors="coerce").dt.to_period("M")
                expenses = expenses.dropna(subset=["_month"])
                if expenses.empty:
                    return "### Регулярные расходы\n_Нет валидных дат для анализа._\n"

                freq = expenses.groupby("description")["_month"].nunique().reset_index()
                freq.columns = ["Описание", "Месяцев"]
                total_months = max(int(expenses["_month"].nunique()), 1)
                recurring = freq[freq["Месяцев"] >= total_months * 0.5].sort_values(
                    "Месяцев", ascending=False
                ).head(top_n)
                return "### Регулярные расходы\n" + _df_to_md(recurring)

            return f"_Инструмент {tool!r} не реализован._"

        except Exception:
            return f"_Не удалось выполнить: {tool}_"

    def _replan(
        self,
        goal: dict[str, Any],
        tasks: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Проверяет полноту данных и при необходимости добавляет до 2 задач."""
        self._emit("replan", "running", "Проверяю полноту данных")

        tools_list = "\n".join([f"- {k}: {v}" for k, v in self.AVAILABLE_TOOLS.items()])
        chunks = []
        for item in results:
            task = item.get("task", {})
            desc = str(task.get("description", ""))
            res = str(item.get("result", ""))[:200]
            chunks.append(f"- {desc}: {res}")
        tasks_and_results = "\n".join(chunks)

        prompt = f"""Ты — аналитик финансового агента. Проверяешь полноту данных перед составлением плана.

ЦЕЛЬ ПОЛЬЗОВАТЕЛЯ: {goal.get("goal_text", "")}

ВЫПОЛНЕННЫЕ ЗАДАЧИ И ИХ РЕЗУЛЬТАТЫ:
{tasks_and_results}

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools_list}

Вопрос: достаточно ли этих данных чтобы составить конкретный, обоснованный план?
Или есть критически важные пробелы которые нужно закрыть?

Критерии полноты:
- Есть данные о доходах и расходах
- Известны основные проблемные категории
- Есть понимание динамики (улучшается / ухудшается)
- Достаточно для 3-5 конкретных рекомендаций

Ответь ТОЛЬКО валидным JSON:
{{
  "need_more": true/false,
  "reason": "одно предложение почему",
  "extra_tasks": [
    {{
      "id": 6,
      "tool": "название_инструмента",
      "description": "что нужно узнать",
      "params": {{"top_n": 10}},
      "priority": "high"
    }}
  ]
}}

Если need_more=false, extra_tasks должен быть [].
Максимум 2 задачи в extra_tasks.
"""

        try:
            raw, meta = self._call_llm(
                scope="planner_replan",
                prompt=prompt,
                max_completion_tokens=300,
                temperature=0,
            )
            parsed = self._parse_json(raw, fallback={})
            if not isinstance(parsed, dict):
                parsed = {}

            need_more = bool(parsed.get("need_more"))
            extra_tasks = parsed.get("extra_tasks") if isinstance(parsed.get("extra_tasks"), list) else []

            validated_extra: list[dict[str, Any]] = []
            for idx, task in enumerate(extra_tasks[:2], start=1):
                if not isinstance(task, dict):
                    continue
                tool = str(task.get("tool", "")).strip()
                if tool not in self.AVAILABLE_TOOLS:
                    continue
                validated_extra.append(
                    {
                        "id": int(task.get("id") or (len(tasks) + idx)),
                        "tool": tool,
                        "description": str(task.get("description") or self.AVAILABLE_TOOLS[tool]),
                        "params": task.get("params") if isinstance(task.get("params"), dict) else {},
                        "priority": str(task.get("priority") or "high"),
                    }
                )

            result = {
                "need_more": need_more and len(validated_extra) > 0,
                "reason": str(parsed.get("reason") or ""),
                "extra_tasks": validated_extra,
            }

            logger.info(
                '[PLANNER][REPLAN] need_more=%s reason="%s" | latency_ms=%s',
                result["need_more"],
                result["reason"],
                meta.get("latency_ms", 0),
            )

            if result["need_more"]:
                self._emit("replan", "done", f"Добавляю {len(validated_extra)} доп. задачи")
            else:
                self._emit("replan", "done", "Данных достаточно")
            return result
        except Exception as exc:
            logger.warning("[PLANNER][REPLAN] Ошибка: %s", exc)
            self._emit("replan", "error", "Не удалось проверить полноту", str(exc))
            return {"need_more": False, "extra_tasks": []}

    def _synthesize(
        self,
        goal: dict[str, Any],
        all_results: list[dict[str, Any]],
        csv_summary: str,
    ) -> str:
        """Синтезирует финальный план в формате PLANNING."""
        self._emit("synthesize", "running", "Синтез финального плана")

        blocks = []
        for item in all_results:
            task = item.get("task", {})
            blocks.append(
                f"### Задача: {task.get('description', '')}\n"
                f"Инструмент: {task.get('tool', '')}\n"
                f"{item.get('result', '')}\n"
            )
        all_results_formatted = "\n".join(blocks)

        prompt = f"""Ты — AI Financial Coach. Составь персональный финансовый план на основе реальных данных.

ЦЕЛЬ ПОЛЬЗОВАТЕЛЯ: {goal.get("goal_text", "")}
ГОРИЗОНТ: {goal.get("horizon", "не указан")}
ОГРАНИЧЕНИЕ: {goal.get("constraint")}

АНАЛИТИЧЕСКИЕ ДАННЫЕ (результаты анализа):
{all_results_formatted}

ОБЩИЙ КОНТЕКСТ (сводка):
{csv_summary}

Составь план строго в формате ниже.
Каждая рекомендация должна опираться на конкретные цифры из данных выше.
Не давай общих советов — только то что следует из этих конкретных данных.

PLANNING
* Определение цели
[1-2 предложения: что пользователь хочет достичь и почему это реалистично
 исходя из данных — укажи конкретные числа]

* Декомпозиция задачи
[3-5 конкретных подзадач с дедлайнами и ответственными действиями.
 Каждая привязана к конкретной цифре из данных]

* Стратегия выполнения
[Пронумерованный план: 3-7 действий.
 Для каждого: ЧТО + КОГДА + СКОЛЬКО ВРЕМЕНИ + ПОЧЕМУ + ожидаемый эффект в рублях.
 Включи 1-2 автоматизации и 1 поведенческую тактику (if-then / трение / предобязательство)]

* Репланирование
[2-3 правила: как скорректировать план если что-то пойдёт не так.
 Конкретные триггеры: "Если в следующем месяце расходы на X превысят Y ₽, то..."]

* Оценка
[2-4 измеримые метрики с конкретными целевыми значениями и сроками.
 Пример: "Расходы на подарки: с 9 015 ₽/мес до 5 000 ₽/мес к апрелю"]
"""

        try:
            text, meta = self._call_llm(
                scope="planner_synthesize",
                prompt=prompt,
                max_completion_tokens=1500,
                temperature=0.4,
            )
            if not text.strip():
                raise ValueError("Пустой ответ synthesis")

            logger.info(
                "[PLANNER][SYNTH] tokens_in=%s tokens_out=%s cost=$%.6f | latency_ms=%s",
                meta.get("prompt_tokens", 0),
                meta.get("completion_tokens", 0),
                meta.get("cost_usd", 0.0),
                meta.get("latency_ms", 0),
            )
            self._emit("synthesize", "done", "План готов")
            return text
        except Exception as exc:
            logger.warning("[PLANNER][SYNTH] Ошибка: %s", exc)
            self._emit("synthesize", "error", "Не удалось синтезировать план", str(exc))
            fallback_lines = [
                "PLANNING",
                "* Определение цели",
                str(goal.get("goal_text") or "Цель не определена."),
                "",
                "* Декомпозиция задачи",
                "Собраны аналитические блоки ниже.",
                "",
                "* Стратегия выполнения",
                "Используйте результаты анализа для пошаговых действий.",
                "",
                "* Репланирование",
                "Пересматривайте план раз в неделю по фактическим расходам.",
                "",
                "* Оценка",
                "Сверяйте прогресс по 2-4 ключевым метрикам.",
                "",
                "### Сырые результаты",
            ]
            for item in all_results:
                fallback_lines.append(str(item.get("result", "")))
            return "\n".join(fallback_lines)

    def _verify_and_fix_plan(
        self,
        draft_text: str,
        goal: dict[str, Any],
        analysis_facts: dict[str, Any],
        csv_summary: str,
    ) -> str:
        """
        Проверяет итоговый план на числовые несоответствия и при необходимости исправляет его.
        При сбое возвращает исходный текст.
        """
        if not draft_text.strip():
            return draft_text

        facts_text = self._pretty_json(analysis_facts)
        summary_short = (csv_summary or "")[:1200]
        prompt = f"""Ты — верификатор финансового плана.

Проверь согласованность чисел и формулировок в плане с фактами из данных.

ЦЕЛЬ:
{goal.get("goal_text", "")}

ПЛАН ДЛЯ ПРОВЕРКИ:
{draft_text}

ФАКТЫ ИЗ ДАННЫХ (источник истины):
{facts_text}

КРАТКИЙ КОНТЕКСТ:
{summary_short}

Что проверить:
1) Нет ли перепутывания "за период" и "в месяц".
2) Нет ли чисел, противоречащих фактам.
3) Не изменена ли суть рекомендаций без причины.

Верни ТОЛЬКО валидный JSON:
{{
  "is_consistent": true/false,
  "issues": ["..."],
  "fixed_text": "исправленный текст или null"
}}
"""

        try:
            raw, _ = self._call_llm(
                scope="planner_verify",
                prompt=prompt,
                max_completion_tokens=1200,
                temperature=0,
            )
            parsed = self._parse_json(raw, fallback={})
            if not isinstance(parsed, dict):
                parsed = {}

            is_consistent = bool(parsed.get("is_consistent"))
            issues_raw = parsed.get("issues")
            issues = issues_raw if isinstance(issues_raw, list) else []
            issues = [str(item) for item in issues]
            fixed_text = parsed.get("fixed_text")
            fixed_text = fixed_text.strip() if isinstance(fixed_text, str) and fixed_text.strip() else None

            logger.info("[PLANNER][VERIFY] is_consistent=%s issues=%s", is_consistent, issues)
            if not is_consistent and fixed_text:
                return fixed_text
            return draft_text
        except Exception as exc:
            logger.warning("[PLANNER][VERIFY] Ошибка: %s", exc)
            return draft_text

    def _build_analysis_facts(self, df: pd.DataFrame) -> dict[str, Any]:
        """Строит компактные факт-агрегаты для верификации числовых утверждений."""
        facts: dict[str, Any] = {
            "rows": int(len(df)),
            "total_income": None,
            "total_expenses": None,
            "months_total": 0,
            "categories": [],
        }

        work_df = df.copy()
        if "date" in work_df.columns:
            parsed_dates = pd.to_datetime(work_df["date"], errors="coerce")
            facts["months_total"] = int(parsed_dates.dt.to_period("M").dropna().nunique())
        else:
            parsed_dates = pd.Series([pd.NaT] * len(work_df))

        if "amount" in work_df.columns and "op_type" in work_df.columns:
            income = pd.to_numeric(work_df.loc[work_df["op_type"] == "доход", "amount"], errors="coerce").dropna()
            expenses = pd.to_numeric(work_df.loc[work_df["op_type"] == "расход", "amount"], errors="coerce").dropna()
            facts["total_income"] = float(income.sum()) if not income.empty else 0.0
            facts["total_expenses"] = float(expenses.sum()) if not expenses.empty else 0.0

        if {"category", "amount", "op_type"}.issubset(set(work_df.columns)):
            expenses_df = work_df[work_df["op_type"] == "расход"].copy()
            if not expenses_df.empty:
                expenses_df["amount"] = pd.to_numeric(expenses_df["amount"], errors="coerce").fillna(0)
                if "date" in expenses_df.columns:
                    expenses_df["_date_parsed"] = pd.to_datetime(expenses_df["date"], errors="coerce")
                    expenses_df["_month"] = expenses_df["_date_parsed"].dt.to_period("M")
                else:
                    expenses_df["_month"] = pd.NaT

                grouped = (
                    expenses_df.groupby("category", dropna=False)["amount"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(20)
                )

                categories: list[dict[str, Any]] = []
                for category_name, total_sum in grouped.items():
                    cat_mask = expenses_df["category"] == category_name
                    months_with_spending = int(
                        expenses_df.loc[cat_mask, "_month"].dropna().nunique()
                    )
                    avg_month = float(total_sum) / months_with_spending if months_with_spending else float(total_sum)
                    categories.append(
                        {
                            "category": str(category_name) if pd.notna(category_name) else "Без категории",
                            "sum_period": float(total_sum),
                            "months_with_spending": months_with_spending,
                            "avg_per_month": avg_month,
                        }
                    )

                facts["categories"] = categories

        return facts

    def _sanitize_reply_text(self, text: str) -> str:
        """Удаляет markdown-heading маркеры и нормализует лишние пустые строки."""
        if not text:
            return ""
        sanitized = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", str(text))
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()

    def _call_llm(
        self,
        scope: str,
        prompt: str,
        max_completion_tokens: int,
        temperature: float,
    ) -> tuple[str, dict[str, Any]]:
        """Выполняет вызов LLM и возвращает текст + метрики."""
        messages = [{"role": "user", "content": prompt}]
        self._log_openai_request(
            scope=scope,
            model=self.model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
        )

        started = perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            self._log_openai_error(
                scope=scope,
                model=self.model,
                messages_count=len(messages),
                error=exc,
                extra={
                    "max_completion_tokens": max_completion_tokens,
                    "temperature": temperature,
                },
            )
            raise
        latency_ms = int((perf_counter() - started) * 1000)

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        assistant_text = (response.choices[0].message.content or "").strip()

        self._run_calls += 1
        self._run_prompt_tokens += prompt_tokens
        self._run_completion_tokens += completion_tokens
        self._run_total_cost += cost_usd

        self._log_openai_response(
            scope=scope,
            response=response,
            latency_ms=latency_ms,
            assistant_text=assistant_text,
            cost_usd=cost_usd,
            extra={
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "finish_reason": finish_reason,
            },
        )

        return (
            assistant_text,
            {
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "finish_reason": finish_reason,
            },
        )

    def _parse_json(self, raw: str, fallback: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
        """Пытается извлечь JSON из ответа LLM, иначе возвращает fallback."""
        if not raw:
            return fallback

        try:
            data = json.loads(raw)
            if isinstance(fallback, dict) and isinstance(data, dict):
                return data
            if isinstance(fallback, list) and isinstance(data, list):
                return data
        except Exception:
            pass

        pattern = r"\{.*\}" if isinstance(fallback, dict) else r"\[.*\]"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(fallback, dict) and isinstance(data, dict):
                    return data
                if isinstance(fallback, list) and isinstance(data, list):
                    return data
            except Exception:
                return fallback

        return fallback

    def _emit(self, step_id: str, status: str, label: str, detail: str = "") -> None:
        """Вызывает callback шага, если он установлен."""
        if self.on_step is None:
            return
        try:
            self.on_step(
                {
                    "id": step_id,
                    "status": status,
                    "label": label,
                    "detail": detail,
                }
            )
        except Exception:
            pass

    def _default_tasks(self) -> list[dict[str, Any]]:
        """Возвращает минимальный fallback-набор задач."""
        return [
            {
                "id": 1,
                "tool": "top_expenses_by_category",
                "description": "Понять крупнейшие категории расходов",
                "params": {"top_n": 10},
                "priority": "high",
            },
            {
                "id": 2,
                "tool": "monthly_dynamics",
                "description": "Оценить динамику доходов/расходов по месяцам",
                "params": {"top_n": 12},
                "priority": "high",
            },
        ]

    def _count_md_rows(self, md_text: str) -> int:
        """Оценивает число строк данных в markdown-таблице для логов."""
        lines = [line for line in md_text.splitlines() if line.strip().startswith("|")]
        return max(0, len(lines) - 2)

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Оценивает стоимость запроса в USD по текущей модели."""
        price = self.COST_PER_1M.get(self.model, self.COST_PER_1M["gpt-4o-mini"])
        in_cost = (prompt_tokens / 1_000_000) * price["input"]
        out_cost = (completion_tokens / 1_000_000) * price["output"]
        return round(in_cost + out_cost, 6)

    def _pretty_json(self, payload: Any) -> str:
        """Возвращает payload в читаемом JSON-формате."""
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(payload)

    def _log_openai_request(
        self,
        scope: str,
        model: str,
        messages: list[dict[str, str]],
        max_completion_tokens: int,
        temperature: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Логирует payload запроса к OpenAI."""
        payload: dict[str, Any] = {
            "scope": scope,
            "model": model,
            "messages_count": len(messages),
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
        }
        if extra:
            payload.update(extra)
        logger.info("[API][OpenAI][Запрос]\n%s", self._pretty_json(payload))

    def _log_openai_response(
        self,
        scope: str,
        response: Any,
        latency_ms: int,
        assistant_text: str,
        cost_usd: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Логирует payload ответа от OpenAI."""
        usage = getattr(response, "usage", None)
        payload: dict[str, Any] = {
            "scope": scope,
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", self.model),
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
            "usage": {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "assistant_message": assistant_text,
        }
        if extra:
            payload.update(extra)
        logger.info("[API][OpenAI][Ответ]\n%s", self._pretty_json(payload))

    def _log_openai_error(
        self,
        scope: str,
        model: str,
        messages_count: int,
        error: Exception,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Логирует ошибку вызова OpenAI."""
        payload: dict[str, Any] = {
            "scope": scope,
            "model": model,
            "messages_count": messages_count,
            "error": str(error),
        }
        if extra:
            payload.update(extra)
        logger.error("[API][OpenAI][Ошибка]\n%s", self._pretty_json(payload))
