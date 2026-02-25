from __future__ import annotations

import io
import json
import logging
import os
import re
from time import perf_counter
from typing import Any

import pandas as pd
from openai import OpenAI
from planner import FinancialPlanner

SYSTEM_PROMPT = """You are an AI Financial Organizer & Coach.
Your primary mission is to help the user bring personal finances into order: clarity, stability, and sustainable habits.
You are not a broker and do not sell financial products. You prioritize safety, simplicity, and behaviorally realistic plans.

---

## USER DATA CONTEXT
The user's financial data is provided below as analytical tables (TOON format).
Use exact numbers from the data. Do not invent figures.

---

## CORE PRINCIPLES
1. Order first, optimization later:
   Cashflow clarity → Budgeting system → Emergency fund → Debt plan → Risk basics → Only then investing.
2. Behavior > spreadsheets:
   Design defaults, automation, friction, and small steps. Avoid shame and moralizing.
3. Evidence-based coaching:
   Use implementation intentions ("If X then Y"), habit stacking, reducing decision fatigue, pre-commitment.
4. Transparency:
   State assumptions, uncertainty, trade-offs. Ask for missing data only when truly necessary.
5. Respect user constraints:
   Time, energy, irregular income, family context. Plans must be executable in bad weeks.

---

## SAFETY & SCOPE
- Do NOT provide legal/tax advice. Give high-level info and recommend a licensed professional.
- Do NOT promise returns or guarantee outcomes.
- Do NOT recommend specific securities. Keep investing general and only after base is stable.
- If acute financial crisis (no food, imminent eviction, severe distress): prioritize immediate safety steps first.

---

## INTENT DETECTION — CLASSIFY EVERY MESSAGE FIRST

Before responding, silently classify the user's message into ONE intent type.
The intent determines the response format. Do not announce the classification.

### INTENT TYPES AND THEIR FORMATS:

---

### [ANALYTICS] — Factual question about data
Triggers: "на что трачу", "сколько ушло", "какая категория", "покажи", "где больше всего",
          "сравни", "в каком месяце", "топ трат", "что за транзакции"

Response format:
- Answer the question directly with exact numbers from the data (2–5 sentences max)
- Add 1 insight or observation if it adds value
- End with 1 short follow-up question OR nothing if the answer is self-contained
- NO plan, NO recommendations, NO metrics section
- Length: 3–8 sentences

Example:
"Больше всего вы тратите на подарки — 1 199 036 ₽ за 17 месяцев (30% расходов).
Это нетипично высокая доля: обычно подарки занимают 3–7% бюджета.
Хотите разобраться, что именно входит в эту категорию?"

---

### [DIAGNOSIS] — Request for assessment or evaluation
Triggers: "оцени", "как у меня дела", "что не так", "где проблемы", "анализ",
          "финансовые привычки", "что думаешь о моих расходах"

Response format:
### Где вы сейчас
- [3–5 фактических наблюдений с цифрами]

### Сильные стороны
- [1–2 пункта]

### Зоны риска
- [1–3 пункта с конкретными цифрами]

### Один приоритет
[Одно чёткое действие на следующую неделю]

- NO full planning sections
- Length: 150–300 words

---

### [PLANNING] — Request for a plan, advice, "what to do"
Triggers: "составь план", "что мне делать", "как улучшить", "помоги сэкономить",
          "с чего начать", "как оптимизировать", "план на месяц"

Response format — ONLY for this intent:
PLANNING
* Определение цели
[1 sentence: what the user wants to achieve]

* Декомпозиция задачи
[2–4 concrete sub-tasks with deadlines]

* Стратегия выполнения
[3–5 numbered actions: what + when + how long + why]
Include: 1–2 automations, 1 behavior tactic (friction / if-then / pre-commitment)

* Репланирование
[2–3 rules or automations to sustain the plan]

* Оценка
[2–3 metrics to track progress]

- This is the ONLY intent that uses the PLANNING structure
- Length: 200–400 words

---

### [ADVISORY] — Request for recommendation or tips
Triggers: "где сэкономить", "совет", "как лучше", "что посоветуешь",
          "стоит ли", "имеет ли смысл", "лучший способ"

Response format:
- Start with 1 sentence diagnosis based on data
- Give 3–5 specific, actionable recommendations (numbered)
- Each recommendation: what to do + expected effect
- End with 1 behavior tactic
- NO full PLANNING structure
- Length: 100–250 words

---

### [CLARIFICATION] — Vague or ambiguous message
Triggers: short messages without clear intent, first message in session,
          "привет", "начнём", "помоги"

Response format:
- Acknowledge briefly (1 sentence)
- Ask at most 2 concrete questions to identify the goal
- Offer 2–3 example directions the user can choose from
- Length: 3–6 sentences

---

## BUDGETING METHODS (select based on user situation)
- Chaotic spending → "containers" (needs/wants/savings) + spending caps + friction
- Stable but no progress → optimize savings rate via automation + category cuts
- Debt heavy → highest interest first, ensure minimums, buffer to avoid new debt
- Irregular income → baseline budget + income smoothing + larger buffer

---

## BEHAVIOR TOOLKIT (use at least one per PLANNING or ADVISORY response)
- Implementation intentions: "If I feel like buying X, then I wait 24h and check Y."
- Defaults & automation: pay yourself first, separate accounts
- Friction: remove cards from apps, shopping lists only
- Pre-commitment: "fun money" envelope, rules for big purchases
- Reduce cognitive load: weekly 15-min money review, simple categories

---

## TONE
- Calm, pragmatic, non-judgmental
- Encourage progress, not perfection
- Use plain language
- Respond in the same language the user writes in (Russian → Russian, English → English)

---

## ANTI-PATTERNS — NEVER DO THESE
- Never repeat the same recommendation twice in a conversation
- Never give a full PLANNING response to a simple factual question
- Never output all 5 PLANNING sections when intent is ANALYTICS or ADVISORY
- Never ask more than 2 questions in one response
- Never pad a short answer with unnecessary sections to appear thorough
"""

SCHEMA_MAP = {
    "date": [
        "дата", "date", "дата операции", "дата транзакции", "дата платежа",
        "дата проведения", "период", "timestamp", "datetime",
        "transaction date", "operation date", "день",
    ],
    "amount": [
        "сумма", "amount", "sum", "итого", "деньги", "руб", "rub",
        "сумма операции", "сумма платежа", "стоимость", "цена",
        "total", "value", "money", "price", "сумма (руб)", "сумма руб",
        "сумма в валюте счёта", "сумма операции в валюте счёта",
    ],
    "category": [
        "категория", "category", "категория расхода", "категория операции",
        "тип операции", "вид операции", "тип расхода", "mcc",
        "merchant category", "раздел",
    ],
    "description": [
        "описание", "description", "наименование", "название", "комментарий",
        "получатель", "отправитель", "контрагент", "назначение",
        "memo", "note", "merchant", "магазин", "место", "платёж",
    ],
    "op_type": [
        "тип", "type", "вид", "направление", "приход/расход",
        "операция", "operation", "flow", "зачисление/списание",
        "доход/расход", "cr/dr", "credit/debit",
    ],
}

SCHEMA_MAP_INCOME_COL = ["приход", "зачисление", "credit", "доход", "пополнение", "income"]
SCHEMA_MAP_EXPENSE_COL = ["расход", "списание", "debit", "трата", "расходы", "expense", "outcome"]

INCOME_WORDS = {"доход", "income", "приход", "зачисление", "credit", "пополнение", "+"}
EXPENSE_WORDS = {"расход", "expense", "списание", "debit", "трата", "покупка", "-"}

logger = logging.getLogger("agent")

class FinancialAgent:
    """Финансовый агент: загрузка CSV, нормализация схемы и чат с LLM."""

    MAX_HISTORY_MESSAGES = 200
    ENABLE_PLANNING_ROUTER = False
    PLANNING_CONFIDENCE_THRESHOLD = 0.75
    COST_PER_1M = {
        "gpt-5.2": {"input": 1.75, "output": 14.00},
        "gpt-5.1": {"input": 1.25, "output": 10.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "o3": {"input": 2.00, "output": 8.00},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self):
        """Инициализирует клиент OpenAI и состояние сессии."""
        self._load_env_if_needed()
        self.client = OpenAI()
        self.model = "gpt-3.5-turbo"
        self.planner = FinancialPlanner(client=self.client, model=self.model)
        self.conversation_history: list[dict[str, str]] = []
        self.csv_summary: str | None = None
        self.df: pd.DataFrame | None = None
        self.last_token_stats: dict[str, Any] | None = None
        self.last_schema_token_stats: dict[str, Any] | None = None
        self._last_encoding: str | None = None
        logger.info("[INIT] FinancialAgent инициализирован model=%s", self.model)

    def load_csv(self, file_content: bytes, filename: str, restore_mode: bool = False) -> dict:
        """
        Выполняет полный пайплайн обработки CSV (шаги 1-6).

        При restore_mode=True загружает DataFrame без сброса истории диалога.
        """
        try:
            self.last_schema_token_stats = None
            text = self._decode(file_content)
            delimiter = self._detect_delimiter(text)
            logger.info("[CSV] Кодировка=%s разделитель='%s'", self._last_encoding, delimiter)

            df = pd.read_csv(io.StringIO(text), sep=delimiter)
            columns_original = [str(c) for c in df.columns]

            schema = self._llm_detect_schema(df)
            schema_source = "llm"
            if not schema:
                schema = self._keyword_detect_schema(df)
                schema_source = "keyword"

            schema = self._enrich_split_schema(schema, df)
            logger.info("[SCHEMA] Источник=%s", schema_source)

            normalized_df = self._apply_schema(df, schema)
            columns_normalized = [str(c) for c in normalized_df.columns]
            logger.info("[NORMALIZE] Итоговые колонки: %s", columns_normalized)

            if "amount" not in normalized_df.columns or normalized_df["amount"].notna().sum() == 0:
                return {
                    "success": False,
                    "error": "Не найдена колонка с суммами. Убедитесь что в файле есть числовые данные.",
                }

            self._log_normalize_stats(normalized_df)

            summary = self._build_toon_summary(normalized_df, filename)
            self.df = normalized_df
            if restore_mode:
                self.csv_summary = summary
            else:
                self.csv_summary = summary
                self.conversation_history = []

            total_income, total_expenses = self._compute_totals(normalized_df)

            date_from = None
            date_to = None
            if "date" in normalized_df.columns and normalized_df["date"].notna().any():
                date_from = normalized_df["date"].min().strftime("%d.%m.%Y")
                date_to = normalized_df["date"].max().strftime("%d.%m.%Y")

            categories: list[str] = []
            if "category" in normalized_df.columns:
                categories = [
                    str(v)
                    for v in normalized_df["category"].dropna().astype(str).unique().tolist()[:20]
                    if str(v).strip()
                ]

            analysis = {
                "rows": int(len(normalized_df)),
                "columns_original": columns_original,
                "columns_normalized": columns_normalized,
                "schema_detected": schema,
                "schema_source": schema_source,
                "total_income": float(total_income) if total_income is not None else None,
                "total_expenses": float(total_expenses) if total_expenses is not None else None,
                "date_from": date_from,
                "date_to": date_to,
                "categories": categories,
                "preview_toon": self._build_preview_toon(normalized_df, n_rows=5),
            }

            return {"success": True, "analysis": analysis}
        except Exception as exc:
            logger.exception("[CSV] Ошибка обработки файла: %s", exc)
            return {"success": False, "error": str(exc)}

    def chat(self, user_message: str) -> str:
        """
        Основной метод диалога с Router Agent для динамического обогащения контекста.

        Пайплайн:
        1. Router анализирует вопрос — нужны ли детали из DataFrame
        2. Если нужны — _fetch_detail достаёт нужные строки через pandas
        3. Детали добавляются к сообщению пользователя
        4. Основной LLM-вызов с обогащённым контекстом
        """
        # Делегируем planning-запросы только если включён routing.
        if self.ENABLE_PLANNING_ROUTER and self.df is not None:
            planning_decision = self._route_planning(user_message)
            use_planner = (
                bool(planning_decision.get("needs_planning"))
                and float(planning_decision.get("confidence", 0.0) or 0.0) >= self.PLANNING_CONFIDENCE_THRESHOLD
            )
            logger.info(
                "[CHAT][ROUTE] planner=%s confidence=%.2f threshold=%.2f reason=%s",
                use_planner,
                float(planning_decision.get("confidence", 0.0) or 0.0),
                self.PLANNING_CONFIDENCE_THRESHOLD,
                planning_decision.get("reason", ""),
            )
            if use_planner:
                planning_result = self.planner.run(
                    user_message=user_message,
                    df=self.df,
                    csv_summary=self.csv_summary or "",
                )
                if planning_result is not None:
                    planning_result = self._sanitize_reply_text(planning_result)
                    planner_stats = getattr(self.planner, "last_run_token_stats", None) or {}
                    self.last_token_stats = {
                        "prompt_tokens": int(planner_stats.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(planner_stats.get("completion_tokens", 0) or 0),
                        "total_tokens": int(planner_stats.get("total_tokens", 0) or 0),
                        "cost_usd": float(planner_stats.get("cost_usd", 0.0) or 0.0),
                        "latency_ms": int(planner_stats.get("latency_ms", 0) or 0),
                        "scope": "planner",
                    }
                    self.conversation_history.append({"role": "user", "content": user_message})
                    self.conversation_history.append({"role": "assistant", "content": planning_result})
                    if len(self.conversation_history) > self.MAX_HISTORY_MESSAGES:
                        self.conversation_history = self.conversation_history[-self.MAX_HISTORY_MESSAGES :]
                    logger.info("[CHAT] Ответ сформирован через Planning Agent")
                    return planning_result
        elif not self.ENABLE_PLANNING_ROUTER:
            logger.info("[CHAT][ROUTE] planning_router disabled")

        detail_block = ""
        route_decision = {"needs_data": False, "queries": []}

        if self.df is not None:
            route_decision = self._route(user_message)
            if route_decision.get("needs_data") and route_decision.get("queries"):
                detail_block = self._fetch_detail(route_decision["queries"])
                if detail_block:
                    logger.info("[CHAT] Контекст обогащён детальными данными (%s символов)", len(detail_block))

        if detail_block:
            enriched_message = (
                f"{user_message}\n\n"
                "[ДЕТАЛЬНЫЕ ДАННЫЕ ДЛЯ ОТВЕТА]\n"
                "Ниже — конкретные транзакции из базы данных пользователя, относящиеся к этому вопросу:\n\n"
                f"{detail_block}"
            )
        else:
            enriched_message = user_message

        # В историю сохраняем чистое пользовательское сообщение (без детализации).
        self.conversation_history.append({"role": "user", "content": user_message})

        system_content = SYSTEM_PROMPT + (self.csv_summary or "")
        history_without_last = self.conversation_history[:-1]
        messages = (
            [{"role": "system", "content": system_content}]
            + history_without_last
            + [{"role": "user", "content": enriched_message}]
        )

        enriched_flag = bool(detail_block)
        request_payload = {
            "scope": "chat",
            "model": self.model,
            "messages_count": len(messages),
            "temperature": 0.7,
            "enriched": enriched_flag,
            "messages": messages,
        }
        logger.info("[API][OpenAI][Запрос]\n%s", self._pretty_json(request_payload))
        logger.info(
            "[CHAT][HISTORY] stored_messages=%s max=%s",
            len(self.conversation_history),
            self.MAX_HISTORY_MESSAGES,
        )
        logger.info("[CHAT][REQ] model=%s messages_count=%s enriched=%s", self.model, len(messages), enriched_flag)

        started_at = perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
        except Exception as exc:
            error_payload = {
                "scope": "chat",
                "model": self.model,
                "messages_count": len(messages),
                "temperature": 0.7,
                "enriched": enriched_flag,
                "error": str(exc),
            }
            logger.error("[API][OpenAI][Ошибка]\n%s", self._pretty_json(error_payload))
            raise

        latency_ms = int((perf_counter() - started_at) * 1000)

        assistant_message = self._sanitize_reply_text(response.choices[0].message.content or "")
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        if len(self.conversation_history) > self.MAX_HISTORY_MESSAGES:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY_MESSAGES :]

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)
        finish_reason = getattr(response.choices[0], "finish_reason", None)

        response_payload = {
            "scope": "chat",
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", self.model),
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "enriched": enriched_flag,
            "assistant_message": assistant_message,
        }
        logger.info("[API][OpenAI][Ответ]\n%s", self._pretty_json(response_payload))
        self.last_token_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": float(round(cost_usd, 6)),
            "latency_ms": latency_ms,
            "scope": "chat",
        }

        logger.info(
            "[CHAT] вход=%s выход=%s всего=%s стоимость=$%.6f model=%s messages=%s enriched=%s",
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
            self.model,
            len(messages),
            enriched_flag,
        )
        logger.info("[CHAT][RESP] finish_reason=%s latency_ms=%s", finish_reason, latency_ms)

        return assistant_message

    def reset(self):
        """Сбрасывает диалог и загруженные данные."""
        self.conversation_history = []
        self.csv_summary = None
        self.df = None
        self.last_token_stats = None
        self.last_schema_token_stats = None
        self.planner.on_step = None

    def _route_planning(self, user_message: str) -> dict:
        """
        Отдельный LLM-роутер: определяет, нужен ли planning-режим для запроса.
        Возвращает needs_planning, confidence и reason.
        """
        if self.df is None:
            return {"needs_planning": False, "confidence": 0.0, "reason": "df_not_loaded"}

        summary_short = (self.csv_summary or "")[:600]
        prompt = f"""Ты — Router финансового ассистента.
Твоя задача: определить, требуется ли для ответа ПОЛНЫЙ planning-режим (многошаговый план), или достаточно обычного краткого ответа.

Запрос пользователя:
"{user_message}"

Краткий контекст данных:
{summary_short}

Правило:
- needs_planning=true только когда пользователь явно или по смыслу просит составить план действий/стратегию/пошаговый roadmap.
- needs_planning=false для аналитики, уточнений, коротких реакций, вопросов по фактам и комментариев.

Ответь ТОЛЬКО валидным JSON:
{{
  "needs_planning": true/false,
  "confidence": 0.0,
  "reason": "краткая причина"
}}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            request_payload = {
                "scope": "planning_router",
                "model": self.model,
                "messages_count": len(messages),
                "temperature": 0,
                "messages": messages,
            }
            logger.info("[API][OpenAI][Запрос]\n%s", self._pretty_json(request_payload))

            started_at = perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            latency_ms = int((perf_counter() - started_at) * 1000)

            raw = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
            cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)
            finish_reason = getattr(response.choices[0], "finish_reason", None)

            response_payload = {
                "scope": "planning_router",
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", self.model),
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "assistant_message": raw,
            }
            logger.info("[API][OpenAI][Ответ]\n%s", self._pretty_json(response_payload))

            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("JSON не найден в ответе planning router")

            parsed = json.loads(json_match.group())
            if not isinstance(parsed, dict):
                raise ValueError("planning router вернул не dict")

            confidence = parsed.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            decision = {
                "needs_planning": bool(parsed.get("needs_planning")),
                "confidence": confidence,
                "reason": str(parsed.get("reason") or ""),
            }
            return decision
        except Exception as exc:
            error_payload = {
                "scope": "planning_router",
                "model": self.model,
                "messages_count": 1,
                "temperature": 0,
                "error": str(exc),
            }
            logger.error("[API][OpenAI][Ошибка]\n%s", self._pretty_json(error_payload))
            logger.warning("[CHAT][ROUTE] planning router error: %s", exc)
            return {"needs_planning": False, "confidence": 0.0, "reason": "router_error"}

    def _route(self, user_message: str) -> dict:
        """
        Дешёвый LLM-вызов для определения нужны ли детальные данные.
        Вызывается только если self.df не None.
        При ошибке возвращает {"needs_data": False} — основной запрос идёт без деталей.
        """
        if self.df is None:
            return {"needs_data": False}

        categories = []
        if "category" in self.df.columns:
            categories = self.df["category"].dropna().astype(str).unique().tolist()[:30]

        available_months = []
        if "date" in self.df.columns:
            try:
                available_months = (
                    pd.to_datetime(self.df["date"], errors="coerce")
                    .dt.to_period("M")
                    .dropna()
                    .unique()
                    .astype(str)
                    .tolist()[-6:]
                )
            except Exception:
                pass

        prompt = f"""Ты — Router для финансового AI-агента.
Твоя задача: определить нужны ли детальные транзакции из базы данных чтобы ответить на вопрос пользователя.

ДОСТУПНЫЕ ДАННЫЕ В БАЗЕ:
- Категории транзакций: {categories}
- Доступные месяцы (последние 6): {available_months}
- Полный список транзакций с полями: date, amount, category, description, op_type

ВОПРОС ПОЛЬЗОВАТЕЛЯ: "{user_message}"

В system prompt агента уже есть агрегированная сводка (итоги по категориям, динамика по месяцам).
Детальные строки транзакций в system prompt НЕ СОДЕРЖАТСЯ.

Определи:
1. needs_data: нужны ли детальные строки транзакций для полного ответа?
   - true: если вопрос про конкретные транзакции, список покупок, что именно куплено,
     детали по конкретной категории/периоду/магазину
   - false: если вопрос про общую аналитику, советы, сравнение, план действий

2. Если needs_data=true — опиши какие именно данные нужны (1-2 запроса максимум).

Ответь ТОЛЬКО валидным JSON без пояснений:
{{
  "needs_data": true/false,
  "reason": "одно предложение почему",
  "queries": [
    {{
      "type": "by_category|by_period|by_description|top_expenses|top_income|anomaly_detail",
      "category": "название категории или null",
      "month": "YYYY-MM или null",
      "keyword": "ключевое слово или null",
      "top_n": 20,
      "sort_by": "amount_desc|date_desc"
    }}
  ]
}}

Если needs_data=false, queries должен быть пустым массивом [].
"""

        try:
            route_messages = [{"role": "user", "content": prompt}]
            request_payload = {
                "scope": "router",
                "model": self.model,
                "messages_count": len(route_messages),
                "temperature": 0,
                "messages": route_messages,
            }
            logger.info("[API][OpenAI][Запрос]\n%s", self._pretty_json(request_payload))

            started_at = perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=route_messages,
                temperature=0,
            )
            latency_ms = int((perf_counter() - started_at) * 1000)

            raw = (response.choices[0].message.content or "").strip()

            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
            cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)
            finish_reason = getattr(response.choices[0], "finish_reason", None)

            response_payload = {
                "scope": "router",
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", self.model),
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "assistant_message": raw,
            }
            logger.info("[API][OpenAI][Ответ]\n%s", self._pretty_json(response_payload))
            logger.info("[ROUTER] Ответ: %s", raw)
            logger.info(
                "[ROUTER] latency_ms=%s вход=%s выход=%s всего=%s стоимость=$%.6f finish_reason=%s",
                latency_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                finish_reason,
            )

            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("JSON не найден в ответе Router")

            decision = json.loads(json_match.group())
            if not isinstance(decision, dict):
                raise ValueError("Router вернул не dict")

            needs_data = bool(decision.get("needs_data"))
            queries = decision.get("queries", [])
            if not isinstance(queries, list):
                queries = []

            normalized = {
                "needs_data": needs_data,
                "reason": decision.get("reason", ""),
                "queries": queries[:2],
            }
            logger.info("[ROUTER] needs_data=%s reason=%s", normalized.get("needs_data"), normalized.get("reason"))
            return normalized

        except Exception as exc:
            error_payload = {
                "scope": "router",
                "model": self.model,
                "messages_count": 1,
                "temperature": 0,
                "error": str(exc),
            }
            logger.error("[API][OpenAI][Ошибка]\n%s", self._pretty_json(error_payload))
            logger.warning("[ROUTER] Ошибка: %s — пропускаю обогащение", exc)
            return {"needs_data": False}

    def _fetch_detail(self, queries: list) -> str:
        """
        Выполняет pandas-запросы к self.df по инструкциям Router.
        Возвращает markdown-таблицы с детальными транзакциями.
        Максимум 2 запроса и 30 строк суммарно.
        """
        if not queries or self.df is None:
            return ""

        results = []
        total_rows = 0
        max_total_rows = 30

        for query in queries[:2]:
            if total_rows >= max_total_rows:
                break

            q_type = query.get("type")
            requested_top_n = int(query.get("top_n", 20) or 20)
            top_n = max(1, min(requested_top_n, max_total_rows - total_rows))
            sort_by = query.get("sort_by", "amount_desc")
            df = self.df.copy()
            total_filtered_rows = 0

            try:
                if q_type == "by_category":
                    cat = query.get("category")
                    if cat and "category" in df.columns:
                        mask = df["category"].astype(str).str.lower().str.contains(str(cat).lower(), na=False)
                        df = df[mask]
                        title = f"Транзакции по категории «{cat}»"
                    else:
                        continue

                elif q_type == "by_period":
                    month = query.get("month")
                    if month and "date" in df.columns:
                        df["_period"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M")
                        df = df[df["_period"].astype(str) == str(month)]
                        df = df.drop(columns=["_period"])
                        title = f"Транзакции за {month}"
                    else:
                        continue

                elif q_type == "by_description":
                    keyword = query.get("keyword", "")
                    if keyword and "description" in df.columns:
                        mask = df["description"].astype(str).str.lower().str.contains(str(keyword).lower(), na=False)
                        df = df[mask]
                        title = f"Транзакции по ключевому слову «{keyword}»"
                    else:
                        continue

                elif q_type == "top_expenses":
                    if "op_type" in df.columns:
                        df = df[df["op_type"] == "расход"]
                    title = f"Топ-{top_n} крупнейших расходов"

                elif q_type == "top_income":
                    if "op_type" in df.columns:
                        df = df[df["op_type"] == "доход"]
                    title = f"Топ-{top_n} крупнейших доходов"

                elif q_type == "anomaly_detail":
                    month = query.get("month")
                    if month and "date" in df.columns:
                        df["_period"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M")
                        df = df[df["_period"].astype(str) == str(month)]
                        df = df.drop(columns=["_period"])
                        title = f"Транзакции аномального месяца {month}"
                    else:
                        continue
                else:
                    continue

                if sort_by == "amount_desc" and "amount" in df.columns:
                    df = df.sort_values("amount", ascending=False)
                elif sort_by == "date_desc" and "date" in df.columns:
                    df = df.sort_values("date", ascending=False)

                total_filtered_rows = len(df)
                df = df.head(top_n)

                if df.empty:
                    results.append(f"### {title}\n_Транзакций не найдено._\n")
                    continue

                display_cols = [c for c in ["date", "amount", "category", "description"] if c in df.columns]
                df_display = df[display_cols].copy()

                if "amount" in df_display.columns:
                    df_display["amount"] = df_display["amount"].apply(
                        lambda x: f"{x:,.0f} ₽".replace(",", " ") if pd.notna(x) else "—"
                    )

                rename_display = {
                    "date": "Дата",
                    "amount": "Сумма",
                    "category": "Категория",
                    "description": "Описание",
                }
                df_display = df_display.rename(columns=rename_display)

                table_md = self._df_to_markdown(df_display)
                results.append(
                    f"### {title} (показано {len(df)} из {total_filtered_rows} транзакций)\n{table_md}\n"
                )
                total_rows += len(df)
                logger.info("[FETCH_DETAIL] type=%s rows=%s", q_type, len(df))

            except Exception as exc:
                logger.warning("[FETCH_DETAIL] Ошибка запроса %s: %s", q_type, exc)
                continue

        return "\n".join(results)

    def _df_to_markdown(self, df: pd.DataFrame) -> str:
        """Конвертирует DataFrame в markdown-таблицу без внешних зависимостей."""
        headers = list(df.columns)
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = []
        for _, row in df.iterrows():
            cells = [str(v) if pd.notna(v) else "—" for v in row]
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([header_row, sep_row] + rows)

    def _decode(self, file_content: bytes) -> str:
        """Определяет кодировку CSV (utf-8 → cp1251 → latin-1)."""
        for encoding in ("utf-8", "cp1251", "latin-1"):
            try:
                text = file_content.decode(encoding)
                self._last_encoding = encoding
                return text
            except UnicodeDecodeError:
                continue
        raise ValueError("Не удалось определить кодировку файла")

    def _detect_delimiter(self, text: str) -> str:
        """Определяет разделитель по первой непустой строке (, ; \t)."""
        first_line = ""
        for line in text.splitlines():
            if line.strip():
                first_line = line
                break

        if not first_line:
            return ","

        candidates = [",", ";", "\t"]
        counts = {sep: first_line.count(sep) for sep in candidates}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","

    def _llm_detect_schema(self, df: pd.DataFrame) -> dict:
        """
        Делает один вызов LLM для определения маппинга колонок CSV.
        Отправляет только схему (названия + примеры значений) — не сами данные.
        При любой ошибке возвращает {} и управление переходит к keyword-fallback.
        """
        schema_lines = []
        for col in df.columns:
            samples = df[col].dropna().head(3).astype(str).tolist()
            dtype = str(df[col].dtype)
            schema_lines.append(f'- "{col}" (тип: {dtype}): примеры → {samples}')

        schema_text = "\n".join(schema_lines)

        prompt = f"""У меня есть CSV-файл с финансовыми транзакциями.
Вот все колонки файла и примеры их значений:

{schema_text}

Определи какая колонка соответствует каждому стандартному полю.
Используй ТОЛЬКО имена колонок из списка выше, не придумывай новые.

Стандартные поля:
- date        — дата операции
- amount      — сумма транзакции (одна числовая колонка)
- category    — категория / тип операции
- description — описание / назначение / получатель платежа
- op_type     — тип операции (доход/расход), если есть отдельная колонка
- income_col  — колонка ТОЛЬКО с доходами (если доходы и расходы в разных колонках)
- expense_col — колонка ТОЛЬКО с расходами (если доходы и расходы в разных колонках)

Также определи:
- amount_format: как записана сумма:
    "standard"    — обычное число (1234.56 или -1234)
    "space_comma" — пробел как тысячный, запятая как дробный ("1 234,56")
    "space_dot"   — пробел как тысячный, точка как дробный ("1 234.56")
- amount_sign: как определить знак операции:
    "signed"     — знак в самой колонке amount (плюс/минус)
    "split_cols" — доходы и расходы в разных колонках (income_col / expense_col)
    "op_type_col"— есть отдельная колонка с типом операции (op_type)

Ответь ТОЛЬКО валидным JSON, без пояснений, без markdown-блоков:
{{
  "date":          "имя колонки или null",
  "amount":        "имя колонки или null",
  "category":      "имя колонки или null",
  "description":   "имя колонки или null",
  "op_type":       "имя колонки или null",
  "income_col":    "имя колонки или null",
  "expense_col":   "имя колонки или null",
  "amount_format": "standard|space_comma|space_dot",
  "amount_sign":   "signed|split_cols|op_type_col"
}}"""

        try:
            request_payload = {
                "scope": "schema",
                "model": self.model,
                "messages_count": 1,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
            logger.info("[API][OpenAI][Запрос]\n%s", self._pretty_json(request_payload))

            started_at = perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=8,
            )
            latency_ms = int((perf_counter() - started_at) * 1000)

            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
            cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)
            finish_reason = getattr(response.choices[0], "finish_reason", None)

            raw = (response.choices[0].message.content or "").strip()

            response_payload = {
                "scope": "schema",
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", self.model),
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "assistant_message": raw,
            }
            logger.info("[API][OpenAI][Ответ]\n%s", self._pretty_json(response_payload))
            self.last_schema_token_stats = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": float(round(cost_usd, 6)),
                "latency_ms": latency_ms,
                "scope": "schema",
            }

            logger.info(
                "[API][SCHEMA][REQ] model=%s messages_count=1 prompt_tokens=%s completion_tokens=%s total_tokens=%s cost_usd=%.6f",
                self.model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
            )
            logger.info("[API][SCHEMA][RESP] finish_reason=%s latency_ms=%s", finish_reason, latency_ms)

            logger.info("[SCHEMA][LLM] Ответ: %s", raw)

            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("JSON не найден в ответе LLM")

            schema = json.loads(json_match.group())

            defaults = {
                "date": None,
                "amount": None,
                "category": None,
                "description": None,
                "op_type": None,
                "income_col": None,
                "expense_col": None,
                "amount_format": "standard",
                "amount_sign": "signed",
            }
            defaults.update(schema)
            schema = defaults

            valid_cols = set(df.columns)
            for field, col in list(schema.items()):
                if field in ("amount_format", "amount_sign"):
                    continue
                if col and col not in valid_cols:
                    logger.warning(
                        "[SCHEMA][LLM] Колонка '%s' для поля '%s' не найдена — обнуляю",
                        col,
                        field,
                    )
                    schema[field] = None

            if schema["amount_format"] not in {"standard", "space_comma", "space_dot"}:
                schema["amount_format"] = "standard"
            if schema["amount_sign"] not in {"signed", "split_cols", "op_type_col"}:
                schema["amount_sign"] = "signed"

            logger.info("[SCHEMA][LLM] Маппинг: %s", schema)
            return schema

        except Exception as exc:
            error_payload = {
                "scope": "schema",
                "model": self.model,
                "messages_count": 1,
                "temperature": 0,
                "error": str(exc),
            }
            self.last_schema_token_stats = None
            logger.error("[API][OpenAI][Ошибка]\n%s", self._pretty_json(error_payload))
            logger.warning("[SCHEMA][LLM] Ошибка: %s — переключаюсь на keyword-fallback", exc)
            return {}

    def _keyword_detect_schema(self, df: pd.DataFrame) -> dict:
        """
        Fallback-нормализация через список синонимов.
        Используется только если _llm_detect_schema вернул ошибку.
        """
        cols_norm = {str(col).strip().lower(): col for col in df.columns}

        schema = {
            "date": None,
            "amount": None,
            "category": None,
            "description": None,
            "op_type": None,
            "income_col": None,
            "expense_col": None,
            "amount_format": "standard",
            "amount_sign": "signed",
        }

        for std_name, keywords in SCHEMA_MAP.items():
            for kw in keywords:
                kw_l = kw.strip().lower()
                exact = cols_norm.get(kw_l)
                if exact is not None:
                    schema[std_name] = exact
                    break

                contains = [orig for norm, orig in cols_norm.items() if kw_l in norm]
                if contains:
                    schema[std_name] = contains[0]
                    break

        for kw in SCHEMA_MAP_INCOME_COL:
            kw_l = kw.strip().lower()
            exact = cols_norm.get(kw_l)
            if exact is not None:
                schema["income_col"] = exact
                schema["amount_sign"] = "split_cols"
                break
            contains = [orig for norm, orig in cols_norm.items() if kw_l in norm]
            if contains:
                schema["income_col"] = contains[0]
                schema["amount_sign"] = "split_cols"
                break

        for kw in SCHEMA_MAP_EXPENSE_COL:
            kw_l = kw.strip().lower()
            exact = cols_norm.get(kw_l)
            if exact is not None:
                schema["expense_col"] = exact
                schema["amount_sign"] = "split_cols"
                break
            contains = [orig for norm, orig in cols_norm.items() if kw_l in norm]
            if contains:
                schema["expense_col"] = contains[0]
                schema["amount_sign"] = "split_cols"
                break

        if schema["op_type"] and schema["amount_sign"] != "split_cols":
            schema["amount_sign"] = "op_type_col"

        if not schema["amount"]:
            for col in df.columns:
                numeric_candidate = pd.to_numeric(df[col], errors="coerce")
                if numeric_candidate.notna().sum() == 0:
                    continue
                if not self._is_id_column(df[col]):
                    schema["amount"] = col
                    break

        probe_col = schema.get("amount") or schema.get("income_col") or schema.get("expense_col")
        if probe_col and probe_col in df.columns:
            schema["amount_format"] = self._detect_amount_format(df[probe_col])

        logger.info("[SCHEMA][KEYWORD] Маппинг: %s", schema)
        return schema

    def _enrich_split_schema(self, schema: dict, df: pd.DataFrame) -> dict:
        """
        Достраивает split-схему (income/expense колонки), если она распознана неполно.
        Это нужно для CSV, где доходы и расходы хранятся в разных столбцах.
        """
        if not isinstance(schema, dict):
            return schema

        result = dict(schema)
        if result.get("amount_sign") == "split_cols":
            return result

        cols_norm = {str(col).strip().lower(): col for col in df.columns}
        income_col = result.get("income_col")
        expense_col = result.get("expense_col")

        def find_col(keywords: list[str]) -> str | None:
            for kw in keywords:
                kw_l = kw.strip().lower()
                if kw_l in cols_norm:
                    return cols_norm[kw_l]
                for norm_name, orig_name in cols_norm.items():
                    if kw_l in norm_name:
                        return orig_name
            return None

        if not income_col:
            income_col = find_col(SCHEMA_MAP_INCOME_COL)
        if not expense_col:
            expense_col = find_col(SCHEMA_MAP_EXPENSE_COL)

        if income_col and expense_col and income_col != expense_col:
            result["income_col"] = income_col
            result["expense_col"] = expense_col
            result["amount_sign"] = "split_cols"
            logger.info(
                "[SCHEMA] Авто-достроена split-схема: income_col=%s expense_col=%s",
                income_col,
                expense_col,
            )

        return result

    def _is_id_column(self, series: pd.Series) -> bool:
        """Проверяет что колонка является ID или порядковым номером, а не суммой."""
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) == 0:
            return True

        is_sequential = (numeric.diff().dropna() == 1).mean() > 0.9 if len(numeric) > 1 else False
        if is_sequential:
            return True

        if numeric.max() <= len(series):
            return True

        return False

    def _apply_schema(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Применяет маппинг из schema к DataFrame:
        — переименовывает колонки в стандартные имена
        — чистит и преобразует колонку amount
        — выводит op_type если его нет
        — обрабатывает split_cols (отдельные колонки дохода/расхода)
        """
        work_df = df.copy()

        rename = {}
        for std_name in ["date", "amount", "category", "description", "op_type"]:
            col = schema.get(std_name)
            if col and col in work_df.columns and col != std_name:
                rename[col] = std_name
        work_df = work_df.rename(columns=rename)
        logger.info("[APPLY_SCHEMA] Переименовано: %s", rename)

        if schema.get("amount_sign") == "split_cols":
            raw_inc_col = schema.get("income_col")
            raw_exp_col = schema.get("expense_col")
            inc_col = rename.get(raw_inc_col, raw_inc_col)
            exp_col = rename.get(raw_exp_col, raw_exp_col)

            has_inc = bool(inc_col and inc_col in work_df.columns)
            has_exp = bool(exp_col and exp_col in work_df.columns)

            if has_inc or has_exp:
                inc = self._clean_amount_column(work_df[inc_col], schema).fillna(0) if has_inc else pd.Series(0, index=work_df.index)
                exp = self._clean_amount_column(work_df[exp_col], schema).fillna(0) if has_exp else pd.Series(0, index=work_df.index)
                work_df["amount"] = inc - exp

                drop_cols = [
                    c
                    for c in [inc_col, exp_col]
                    if c in work_df.columns and c not in {"amount", "date", "category", "description", "op_type"}
                ]
                if drop_cols:
                    work_df = work_df.drop(columns=drop_cols)
                logger.info(
                    "[APPLY_SCHEMA] Объединены колонки income=%s expense=%s → amount",
                    inc_col if has_inc else None,
                    exp_col if has_exp else None,
                )

        if "amount" in work_df.columns:
            work_df["amount"] = self._clean_amount_column(work_df["amount"], schema)

        if "date" in work_df.columns:
            work_df["date"] = self._parse_date_column(work_df["date"])

        if "op_type" not in work_df.columns and "amount" in work_df.columns:
            work_df["op_type"] = work_df["amount"].apply(
                lambda x: "доход" if pd.notna(x) and x >= 0 else "расход"
            )
            logger.info("[APPLY_SCHEMA] op_type выведен автоматически по знаку amount")
        elif "op_type" in work_df.columns:
            def normalize_op(val: Any) -> str:
                v = str(val).strip().lower()
                if any(word in v for word in INCOME_WORDS):
                    return "доход"
                if any(word in v for word in EXPENSE_WORDS):
                    return "расход"
                return v

            work_df["op_type"] = work_df["op_type"].apply(normalize_op)

            if "amount" in work_df.columns:
                missing_mask = ~work_df["op_type"].isin(["доход", "расход"])
                work_df.loc[missing_mask & (work_df["amount"] >= 0), "op_type"] = "доход"
                work_df.loc[missing_mask & (work_df["amount"] < 0), "op_type"] = "расход"

        if "amount" in work_df.columns:
            work_df["amount"] = work_df["amount"].abs()

        return work_df

    def _clean_amount_column(self, series: pd.Series, schema: dict) -> pd.Series:
        """
        Приводит строковые суммы к float.
        Учитывает форматы: "1 234,56" / "1 234.56" / "-1234" / "+1 234"
        """
        fmt = schema.get("amount_format", "standard")
        s = series.astype(str).str.strip()

        if fmt == "space_comma":
            s = s.str.replace(r"\s", "", regex=True).str.replace(",", ".", regex=False)
        elif fmt == "space_dot":
            s = s.str.replace(r"\s", "", regex=True)
        else:
            s = s.str.replace(r"\s", "", regex=True).str.replace(",", ".", regex=False)

        s = s.str.replace(r"[^\d.\-+]", "", regex=True)

        def normalize_token(token: Any) -> str:
            if pd.isna(token):
                return ""

            token = str(token)
            sign = ""
            if token.startswith("+"):
                sign, token = "+", token[1:]
            elif token.startswith("-"):
                sign, token = "-", token[1:]

            if token.count(".") > 1:
                parts = token.split(".")
                token = "".join(parts[:-1]) + "." + parts[-1]

            return sign + token

        s = s.apply(normalize_token)
        return pd.to_numeric(s, errors="coerce")

    def _build_toon_summary(self, df: pd.DataFrame, filename: str) -> str:
        """
        Строит агрегированную сводку в TOON-формате (markdown-таблицы).
        Результат: ~2-4к токенов — весь финансовый контекст для LLM.
        """
        total_income, total_expenses = self._compute_totals(df)
        balance = (total_income or 0.0) - (total_expenses or 0.0)

        has_date = "date" in df.columns and df["date"].notna().any()
        has_category = "category" in df.columns

        date_from = None
        date_to = None
        months_count = 0

        if has_date:
            date_from = df["date"].min()
            date_to = df["date"].max()
            months_count = int(df["date"].dt.to_period("M").nunique())

        avg_income = (total_income / months_count) if (months_count and total_income is not None) else None
        avg_expenses = (total_expenses / months_count) if (months_count and total_expenses is not None) else None

        parts: list[str] = []
        parts.append(f"## 📁 Файл: {filename}")
        if date_from is not None and date_to is not None:
            parts.append(f"- Период: {date_from.strftime('%d.%m.%Y')} – {date_to.strftime('%d.%m.%Y')}")
        else:
            parts.append("- Период: —")
        parts.append(f"- Всего транзакций: {self._fmt_num(len(df))}")
        parts.append(f"- Месяцев в данных: {self._fmt_num(months_count) if months_count else '—'}")
        parts.append("")

        parts.append("## 💰 Общие показатели")
        parts.append(
            self._md_table(
                ["Показатель", "Значение"],
                [
                    ["Доходы (итого)", f"{self._fmt_num(total_income)} ₽"],
                    ["Расходы (итого)", f"{self._fmt_num(total_expenses)} ₽"],
                    ["Баланс", f"{self._fmt_signed(balance)} ₽"],
                    ["Средний доход/мес", f"{self._fmt_num(avg_income)} ₽" if avg_income is not None else "—"],
                    ["Средний расход/мес", f"{self._fmt_num(avg_expenses)} ₽" if avg_expenses is not None else "—"],
                ],
            )
        )

        if has_category:
            expense_df = df[df["op_type"] == "расход"].copy() if "op_type" in df.columns else pd.DataFrame(columns=df.columns)
            if not expense_df.empty:
                grouped = (
                    expense_df.groupby("category", dropna=False)["amount"]
                    .agg(["sum", "count", "mean"])
                    .sort_values("sum", ascending=False)
                )
                grouped = grouped.head(10)
                total_exp = float(total_expenses or 0.0)

                rows = []
                for cat, values in grouped.iterrows():
                    cat_name = str(cat) if pd.notna(cat) and str(cat).strip() else "Без категории"
                    cat_sum = float(values["sum"])
                    cat_pct = (cat_sum / total_exp * 100.0) if total_exp > 0 else 0.0
                    rows.append([
                        cat_name,
                        self._fmt_num(cat_sum),
                        f"{cat_pct:.1f}%",
                        self._fmt_num(values["count"]),
                        self._fmt_num(values["mean"]),
                    ])
            else:
                rows = [["—", "0", "0.0%", "0", "0"]]

            parts.append("")
            parts.append("## 📊 Расходы по категориям")
            parts.append(
                self._md_table(
                    ["Категория", "Сумма ₽", "% от расходов", "Транзакций", "Среднее ₽"],
                    rows,
                )
            )

        monthly_expense = pd.Series(dtype="float64")
        if has_date:
            month_df = df.copy()
            month_df["month"] = month_df["date"].dt.to_period("M").astype(str)

            income_by_month = month_df[month_df["op_type"] == "доход"].groupby("month")["amount"].sum()
            expense_by_month = month_df[month_df["op_type"] == "расход"].groupby("month")["amount"].sum()
            monthly_expense = expense_by_month

            all_months = sorted(set(income_by_month.index).union(set(expense_by_month.index)))
            month_rows = []
            for month in all_months[-24:]:
                inc = float(income_by_month.get(month, 0.0))
                exp = float(expense_by_month.get(month, 0.0))
                month_rows.append([
                    month,
                    self._fmt_num(inc),
                    self._fmt_num(exp),
                    self._fmt_signed(inc - exp),
                ])

            parts.append("")
            parts.append("## 📅 Динамика по месяцам")
            parts.append(
                self._md_table(
                    ["Месяц", "Доход ₽", "Расход ₽", "Баланс ₽"],
                    month_rows if month_rows else [["—", "0", "0", "0"]],
                )
            )

        parts.append("")
        parts.append("## 🔝 Топ-10 крупнейших трат")
        expense_top = df[df["op_type"] == "расход"].copy() if "op_type" in df.columns else pd.DataFrame(columns=df.columns)
        if not expense_top.empty:
            expense_top = expense_top.sort_values("amount", ascending=False).head(10)
            top_rows = []
            for _, row in expense_top.iterrows():
                top_rows.append([
                    self._fmt_date(row["date"]) if "date" in expense_top.columns else "—",
                    self._fmt_num(row.get("amount")),
                    self._safe(row.get("category")) if "category" in expense_top.columns else "—",
                    self._safe(row.get("description")) if "description" in expense_top.columns else "—",
                ])
        else:
            top_rows = [["—", "0", "—", "Нет расходов"]]

        parts.append(self._md_table(["Дата", "Сумма ₽", "Категория", "Описание"], top_rows))

        if has_date:
            anomaly_rows = []
            if not monthly_expense.empty:
                mean_exp = float(monthly_expense.mean())
                if mean_exp > 0:
                    for month, value in monthly_expense.items():
                        val = float(value)
                        deviation = (val - mean_exp) / mean_exp
                        if abs(deviation) > 0.30:
                            direction = "выше" if deviation > 0 else "ниже"
                            anomaly_rows.append([
                                month,
                                f"Расходы на {abs(deviation) * 100:.0f}% {direction} среднего ({self._fmt_num(val)} ₽)",
                            ])

            if not anomaly_rows:
                anomaly_rows = [["—", "Существенных отклонений по расходам не найдено"]]

            parts.append("")
            parts.append("## ⚠️ Аномалии и наблюдения")
            parts.append(self._md_table(["Месяц", "Наблюдение"], anomaly_rows[:12]))

        summary = "\n".join(parts)
        if len(summary) > 12000:
            summary = summary[:12000] + "\n\n*Сводка сокращена для экономии токенов.*"

        return summary

    def _load_env_if_needed(self) -> None:
        """Подгружает OPENAI_API_KEY из .env, если переменная окружения не установлена."""
        if os.getenv("OPENAI_API_KEY"):
            return

        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if not os.path.exists(env_path):
            return

        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key.strip() == "OPENAI_API_KEY":
                        os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
                        logger.info("[ENV] OPENAI_API_KEY загружен из .env")
                        return
        except Exception as exc:
            logger.warning("[ENV] Ошибка чтения .env: %s", exc)

    def _detect_amount_format(self, series: pd.Series) -> str:
        """Оценивает формат чисел в колонке суммы."""
        samples = series.dropna().astype(str).head(10).tolist()
        if not samples:
            return "standard"

        joined = " | ".join(samples)
        if re.search(r"\d\s+\d+,\d+", joined):
            return "space_comma"
        if re.search(r"\d\s+\d+\.\d+", joined):
            return "space_dot"
        return "standard"

    def _parse_date_column(self, series: pd.Series) -> pd.Series:
        """
        Надёжно парсит даты из смешанных форматов без шумных предупреждений.
        Сначала пробует ISO (%Y-%m-%d), затем dayfirst-парсинг.
        """
        raw = series.astype(str).str.strip()
        parsed = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
        mask = parsed.isna()
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(raw.loc[mask], errors="coerce", dayfirst=True)
        return parsed

    def _normalize_op_type(self, value: Any) -> str:
        """Нормализует значение op_type к 'доход' или 'расход' когда возможно."""
        text = str(value).strip().lower()
        if any(word in text for word in INCOME_WORDS):
            return "доход"
        if any(word in text for word in EXPENSE_WORDS):
            return "расход"
        return text

    def _compute_totals(self, df: pd.DataFrame) -> tuple[float | None, float | None]:
        """Считает итоговые доходы и расходы по нормализованным данным."""
        if "amount" not in df.columns or "op_type" not in df.columns:
            return None, None

        income = pd.to_numeric(df.loc[df["op_type"] == "доход", "amount"], errors="coerce").dropna()
        expenses = pd.to_numeric(df.loc[df["op_type"] == "расход", "amount"], errors="coerce").dropna()

        total_income = float(income.sum()) if not income.empty else None
        total_expenses = float(expenses.sum()) if not expenses.empty else None
        return total_income, total_expenses

    def _build_preview_toon(self, df: pd.DataFrame, n_rows: int = 5) -> str:
        """Формирует markdown-предпросмотр первых строк нормализованного DataFrame."""
        if df.empty:
            return "| Данные |\n|---|\n| Нет строк |"

        preferred = ["date", "amount", "op_type", "category", "description"]
        cols = [c for c in preferred if c in df.columns]
        if not cols:
            cols = [str(c) for c in df.columns[:5]]

        preview = df[cols].head(n_rows).copy()
        if "date" in preview.columns:
            preview["date"] = preview["date"].apply(self._fmt_date)
        if "amount" in preview.columns:
            preview["amount"] = preview["amount"].apply(self._fmt_num)

        rows: list[list[str]] = []
        for _, row in preview.iterrows():
            rows.append([self._safe(row.get(col)) for col in cols])

        return self._md_table(cols, rows)

    def _log_normalize_stats(self, df: pd.DataFrame) -> None:
        """Логирует статистику после нормализации amount и op_type."""
        if "amount" in df.columns:
            amount = pd.to_numeric(df["amount"], errors="coerce").dropna()
            if not amount.empty:
                logger.info(
                    "[NORMALIZE] amount: min=%s max=%s sum=%s",
                    self._fmt_num(amount.min()),
                    self._fmt_num(amount.max()),
                    self._fmt_num(amount.sum()),
                )
            else:
                logger.info("[NORMALIZE] amount: нет валидных числовых значений")

        if "op_type" in df.columns:
            distribution = df["op_type"].fillna("<null>").astype(str).value_counts().to_dict()
            logger.info("[NORMALIZE] op_type распределение: %s", distribution)

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Оценивает стоимость запроса к модели в USD."""
        pricing = self.COST_PER_1M.get(self.model, self.COST_PER_1M["gpt-4o-mini"])
        return (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]

    def _fmt_num(self, value: Any) -> str:
        """Форматирует число в вид 1 234 567."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        try:
            return f"{float(value):,.0f}".replace(",", " ")
        except Exception:
            return "—"

    def _fmt_signed(self, value: Any) -> str:
        """Форматирует число со знаком."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        num = float(value)
        sign = "+" if num > 0 else ""
        return f"{sign}{self._fmt_num(num)}"

    def _fmt_date(self, value: Any) -> str:
        """Форматирует дату в DD.MM.YYYY."""
        if value is None:
            return "—"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%d.%m.%Y") if not pd.isna(value) else "—"
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
        return parsed.strftime("%d.%m.%Y") if not pd.isna(parsed) else "—"

    def _safe(self, value: Any) -> str:
        """Экранирует значение для markdown-таблицы."""
        if value is None:
            return "—"
        if isinstance(value, float) and pd.isna(value):
            return "—"

        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return "—"
        return text.replace("|", "\\|").replace("\n", " ")

    def _md_table(self, headers: list[str], rows: list[list[Any]]) -> str:
        """Собирает markdown-таблицу."""
        head = "| " + " | ".join(self._safe(h) for h in headers) + " |"
        sep = "|" + "|".join(["---"] * len(headers)) + "|"
        body = ["| " + " | ".join(self._safe(cell) for cell in row) + " |" for row in rows]
        return "\n".join([head, sep] + body)

    def _pretty_json(self, payload: Any) -> str:
        """Сериализует структуру в читаемый JSON для логов."""
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(payload)

    def _sanitize_reply_text(self, text: str) -> str:
        """Удаляет markdown-heading маркеры и нормализует формат ответа."""
        if not text:
            return ""
        sanitized = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", str(text))
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()
