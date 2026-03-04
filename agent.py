from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
from time import perf_counter
from typing import Any

import pandas as pd

from context_strategies import ContextStrategyManager
from llm import OpenAILLMClient
from memory import MemoryManager
from storage import ensure_session

SYSTEM_PROMPT = """You are an AI financial advisor focused on personal finance.
You are not a broker and you do not provide legal or tax advice.
Use exact numbers from the provided data. Do not invent figures.
Final user-facing reply must be in Russian only.

Keep replies concise so they fit within token limits: prefer short, focused answers; avoid long enumerations, repetition, or unnecessary preamble. If the user needs more detail, they can ask to continue.

Never output the heading or label "PLANNING".
Do not output any service tags or markers like [PLAN], [ANALYTICS], [DIAGNOSIS], [ADVISORY], [CLARIFICATION].

Internally detect the intent and choose an appropriate response style:

Analytics questions ("where do I spend", "how much", "show", "compare", "top"):
Direct numeric answer, 3-8 sentences. No plan or generic recommendations.

Diagnosis requests ("assess", "how am I doing", "what is wrong", "analyze"):
Current state (facts) / strengths / risk zones / one priority.

Planning requests ("make a plan", "what should I do", "how to improve", "where to start"):
Give a practical step-by-step action plan in plain text with short section labels in Russian.
Do not use the word "PLANNING" or bracketed headings.

Advice requests ("where to save", "is it worth", "advise"):
1 diagnosis + 3-5 concrete actions + 1 behavioral tactic.

Vague requests ("hello", "help"):
1-2 clarifying questions + 2-3 direction options.

Do not repeat the same recommendation twice. Do not ask more than 2 questions at a time.
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

    DEFAULT_MODEL = "gpt-5-mini"
    MAX_DETAIL_ROWS = 24
    MAX_DETAIL_CHARS = 7000
    MAX_SYSTEM_CONTEXT_CHARS = 18000
    MAX_TOTAL_CONTEXT_CHARS = MAX_SYSTEM_CONTEXT_CHARS + MAX_DETAIL_CHARS
    DEFAULT_USER_ID = "default_local_user"
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

    def __init__(self, model: str | None = None):
        """Инициализирует клиент OpenAI и состояние сессии."""
        self._load_env_if_needed()
        self.llm_client = OpenAILLMClient()
        requested_model = (model or os.getenv("OPENAI_MODEL") or self.DEFAULT_MODEL).strip()
        try:
            self.model = self._validate_model(requested_model)
        except ValueError:
            self.model = self.DEFAULT_MODEL
            logger.warning(
                "[INIT] Неизвестная модель %s, использую модель по умолчанию: %s",
                requested_model,
                self.model,
            )
        self.conversation_history: list[dict[str, str]] = []
        self.ctx = ContextStrategyManager(client=self.llm_client, model=self.model)
        self.memory = MemoryManager(
            short_term_limit=30,
            llm_client=self.llm_client,
            step_parser_model="gpt-5-nano",
        )
        self.csv_summary: str | None = None
        self.summary_sections: dict[str, str] = {}
        self.expense_cache: dict[str, Any] = {}
        self.df: pd.DataFrame | None = None
        self.last_token_stats: dict[str, Any] | None = None
        self.last_schema_token_stats: dict[str, Any] | None = None
        self.last_memory_stats: dict[str, Any] | None = None
        self.last_prompt_preview: dict[str, str] | None = None
        self.last_chat_response_meta: dict[str, Any] | None = None
        self._last_encoding: str | None = None
        logger.info("[INIT] FinancialAgent инициализирован model=%s", self.model)

    @classmethod
    def available_models(cls) -> list[str]:
        """Возвращает список поддерживаемых моделей."""
        return list(cls.COST_PER_1M.keys())

    def set_model(self, model: str) -> str:
        """Переключает модель для всех LLM-вызовов агента."""
        validated = self._validate_model(model)
        if validated == self.model:
            return self.model

        self.model = validated
        self.ctx.set_model(validated)
        logger.info("[MODEL] Модель переключена на %s", validated)
        return validated

    @classmethod
    def _validate_model(cls, model: str) -> str:
        """Проверяет, что модель поддерживается приложением."""
        normalized = (model or "").strip()
        if not normalized:
            raise ValueError("Не указана модель")
        if normalized not in cls.COST_PER_1M:
            supported = ", ".join(cls.available_models())
            raise ValueError(f"Неподдерживаемая модель: {normalized}. Доступны: {supported}")
        return normalized

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
            sections = self._extract_summary_sections(summary)
            expense_cache = self._build_expense_cache(normalized_df)
            self.df = normalized_df
            if restore_mode:
                self.csv_summary = summary
                self.summary_sections = sections
                self.expense_cache = expense_cache
            else:
                self.csv_summary = summary
                self.summary_sections = sections
                self.expense_cache = expense_cache
                self.conversation_history = []
                self.ctx.reset_all()

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

    def chat(self, user_message: str, session_id: str | None = None, user_id: str | None = None) -> str:
        """
        Основной метод диалога с Router Agent для динамического обогащения контекста.

        Пайплайн:
        1. Router анализирует вопрос — нужны ли детали из DataFrame
        2. Если нужны — _fetch_detail достаёт нужные строки через pandas
        3. Детали добавляются к сообщению пользователя
        4. Основной LLM-вызов с обогащённым контекстом
        """
        t_start = perf_counter()

        current_session_id = str(session_id or "default_session")
        current_user_id = str(user_id or self.DEFAULT_USER_ID)
        ensure_session(current_session_id)
        detail_block = ""
        route_decision = {
            "needs_data": False,
            "reason": "df_not_loaded",
            "queries": [],
            "expense_scope": "overview",
            "context_profile": "light",
        }

        if self.df is not None:
            route_decision = self._route(user_message)
            if route_decision.get("needs_data") and route_decision.get("queries"):
                detail_block = self._fetch_detail(route_decision["queries"], route_decision)
                if detail_block:
                    logger.info("[CHAT] Контекст обогащён детальными данными (%s символов)", len(detail_block))

        summary_context = self._compose_system_context(route_decision)
        system_content = SYSTEM_PROMPT if not summary_context else f"{SYSTEM_PROMPT}\n\n{summary_context}"
        system_content, detail_block = self._fit_context_budget(system_content, detail_block, route_decision)

        if detail_block:
            enriched_message_for_model = (
                f"{user_message}\n\n"
                "[DETAILED DATA FOR RESPONSE]\n"
                "Below are concrete transactions from the user's dataset related to this question:\n\n"
                f"{detail_block}"
            )
        else:
            enriched_message_for_model = user_message

        guard_ctx_before = self.memory.working.load(current_session_id)
        gate_message = self.memory.enforce_planning_gate(
            session_id=current_session_id,
            user_message=user_message,
        )
        if gate_message:
            is_planning_block = bool(guard_ctx_before and guard_ctx_before.state.value == "PLANNING")
            finish_reason = "state_blocked_planning" if is_planning_block else "state_blocked"
            if is_planning_block:
                goal = str(guard_ctx_before.task or "Текущая задача").strip() or "Текущая задача"
                assistant_message = self._sanitize_reply_text(
                    f"Задача создана: '{goal}'.\n"
                    "Чтобы начать выполнение — сформируйте план.\n"
                    "Напишите шаги или нажмите 'Сформировать план автоматически'."
                )
            else:
                assistant_message = self._sanitize_reply_text(gate_message)
            self.memory.append_turn(
                session_id=current_session_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            latency = round((perf_counter() - t_start) * 1000)
            self.last_memory_stats = self.memory.stats(session_id=current_session_id, user_id=current_user_id)
            self.last_token_stats = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": int(latency),
                "scope": "chat",
                "strategy": "memory_layers",
                "ctx_stats": self.ctx.stats(self.conversation_history),
                "memory_stats": self.last_memory_stats,
                "finish_reason": finish_reason,
            }
            self.last_prompt_preview = {
                "schema_version": 2,
                "system": "[REDACTED_SYSTEM_PROMPT]",
                "system_chars": 0,
                "system_hash": "",
                "user_chars": len(user_message or ""),
                "short_term_count": 0,
                "working_state": self.memory.get_working_view(session_id=current_session_id).get("state"),
                "profile_injected": [],
                "profile_skipped": [],
                "decisions_count": 0,
                "notes_count": 0,
            }
            self.last_chat_response_meta = {
                "finish_reason": finish_reason,
                "working_view": self.memory.get_working_view(session_id=current_session_id),
                "working_actions": self.memory.get_working_actions(session_id=current_session_id),
            }
            if is_planning_block:
                logger.info("[CHAT][STATE_GUARD] blocked_in_planning session=%s", current_session_id)
            else:
                logger.info("[CHAT][STATE_GUARD] blocked session=%s", current_session_id)
            return assistant_message

        self.memory.route_user_message(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
        )
        guard_ctx_after_route = self.memory.working.load(current_session_id)
        if (
            guard_ctx_after_route
            and guard_ctx_after_route.state.value == "PLANNING"
            and bool((guard_ctx_after_route.vars or {}).get("plan_guidance_required"))
        ):
            goal = str(guard_ctx_after_route.task or "Текущая задача").strip() or "Текущая задача"
            assistant_message = self._sanitize_reply_text(
                f"Задача создана: '{goal}'.\n"
                "Чтобы начать выполнение — сформируйте план.\n"
                "Опишите шаги плана или нажмите 'Сформировать план автоматически'."
            )
            vars_patch = dict(guard_ctx_after_route.vars)
            vars_patch.pop("plan_guidance_required", None)
            self.memory.working.update(current_session_id, vars=vars_patch)
            self.memory.append_turn(
                session_id=current_session_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            latency = round((perf_counter() - t_start) * 1000)
            self.last_memory_stats = self.memory.stats(session_id=current_session_id, user_id=current_user_id)
            self.last_token_stats = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": int(latency),
                "scope": "chat",
                "strategy": "memory_layers",
                "ctx_stats": self.ctx.stats(self.conversation_history),
                "memory_stats": self.last_memory_stats,
                "finish_reason": "state_blocked_planning",
            }
            self.last_chat_response_meta = {
                "finish_reason": "state_blocked_planning",
                "working_view": self.memory.get_working_view(session_id=current_session_id),
                "working_actions": self.memory.get_working_actions(session_id=current_session_id),
            }
            logger.info("[CHAT][STATE_GUARD] blocked_in_planning session=%s reason=plan_extract_fallback", current_session_id)
            return assistant_message

        messages, prompt_preview, read_meta = self.memory.build_messages(
            session_id=current_session_id,
            user_id=current_user_id,
            system_instructions=system_content,
            data_context="",
            user_query=enriched_message_for_model,
        )
        self.last_prompt_preview = prompt_preview

        enriched_flag = bool(detail_block)
        current_memory_stats = self.memory.stats(session_id=current_session_id, user_id=current_user_id)
        request_payload = {
            "scope": "chat",
            "model": self.model,
            "messages_count": len(messages),
            "temperature": 0.7,
            "enriched": enriched_flag,
            "strategy": "memory_layers",
            "expense_scope": route_decision.get("expense_scope"),
            "context_profile": route_decision.get("context_profile"),
            "system_chars": len(system_content),
            "detail_chars": len(detail_block),
            "session_id": current_session_id,
            "memory_stats": current_memory_stats,
            "memory_read": read_meta,
            "messages_redacted": self._redact_messages_for_log(messages),
        }
        logger.info("[API][OpenAI][Запрос]\n%s", self._pretty_json(request_payload))
        logger.info(
            "[CHAT][HISTORY] short_term_messages=%s",
            len(self.memory.short_term.get_context(current_session_id)),
        )
        logger.info(
            "[CHAT][REQ] model=%s messages_count=%s enriched=%s strategy=%s",
            self.model,
            len(messages),
            enriched_flag,
            "memory_layers",
        )

        try:
            response = self._create_chat_completion(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=0.7,
            )
        except Exception as exc:
            error_payload = {
                "scope": "chat",
                "model": self.model,
                "messages_count": len(messages),
                "temperature": 0.7,
                "enriched": enriched_flag,
                "strategy": "memory_layers",
                "error": str(exc),
            }
            logger.error("[API][OpenAI][Ошибка]\n%s", self._pretty_json(error_payload))
            raise

        finish_reason = getattr(response.choices[0], "finish_reason", None)
        raw_content = response.choices[0].message.content or ""
        assistant_message = self._sanitize_reply_text(raw_content)
        if not assistant_message:
            if finish_reason == "length":
                assistant_message = (
                    "Ответ обрезан из-за лимита длины. Попросите «продолжи» или задайте вопрос короче."
                )
            else:
                assistant_message = (
                    "Не удалось сформировать ответ в текущем лимите генерации. "
                    "Уточните запрос или попросите ответ короче."
                )
        elif finish_reason == "length":
            assistant_message = assistant_message.rstrip() + "\n\n_Ответ обрезан по длине. Можете попросить продолжить._"
        self.memory.append_turn(
            session_id=current_session_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)
        latency = round((perf_counter() - t_start) * 1000)

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
            "latency_ms": latency,
            "cost_usd": cost_usd,
            "enriched": enriched_flag,
            "strategy": "memory_layers",
            "assistant_message": assistant_message,
        }
        logger.info("[API][OpenAI][Ответ]\n%s", self._pretty_json(response_payload))
        self.last_memory_stats = self.memory.stats(session_id=current_session_id, user_id=current_user_id)
        self.last_token_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": float(round(cost_usd, 6)),
            "latency_ms": int(latency),
            "scope": "chat",
            "strategy": "memory_layers",
            "ctx_stats": self.ctx.stats(self.conversation_history),
            "memory_stats": self.last_memory_stats,
            "memory_read": read_meta,
            "prompt_preview": self.last_prompt_preview,
            "finish_reason": finish_reason,
        }
        self.last_chat_response_meta = {
            "finish_reason": finish_reason,
            "working_view": self.memory.get_working_view(session_id=current_session_id),
            "working_actions": self.memory.get_working_actions(session_id=current_session_id),
        }

        logger.info(
            "[CHAT] вход=%s выход=%s всего=%s стоимость=$%.6f model=%s messages=%s enriched=%s strategy=%s",
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
            self.model,
            len(messages),
            enriched_flag,
            "memory_layers",
        )
        logger.info("[CHAT][RESP] finish_reason=%s latency_ms=%s", finish_reason, latency)

        return assistant_message

    def reset(self):
        """Сбрасывает диалог и загруженные данные."""
        self.conversation_history = []
        self.csv_summary = None
        self.summary_sections = {}
        self.expense_cache = {}
        self.df = None
        self.ctx.reset_all()
        self.last_token_stats = None
        self.last_schema_token_stats = None
        self.last_memory_stats = None
        self.last_prompt_preview = None

    def clear_session_memory(self, session_id: str) -> None:
        """Очищает short-term и working память конкретной сессии."""
        self.memory.clear_session(session_id=session_id)

    def restore_memory_session(self, session_id: str, messages: list[dict[str, str]] | None = None) -> None:
        """
        Восстанавливает memory-слои сессии без ре-гидрации short-term из persisted history.
        Short-term контекст живёт только в рамках текущего runtime.
        """
        _ = messages  # backward-compatible сигнатура
        self.memory.short_term.get_context(session_id)

    def _route(self, user_message: str) -> dict:
        """
        Дешёвый LLM-вызов для определения нужны ли детальные данные.
        Вызывается только если self.df не None.
        При ошибке возвращает {"needs_data": False} — основной запрос идёт без деталей.
        """
        if self.df is None:
            return {
                "needs_data": False,
                "reason": "df_not_loaded",
                "queries": [],
                "expense_scope": "overview",
                "context_profile": "light",
            }

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

        prompt = f"""You are a router for a financial AI agent.
Your task is to decide whether detailed transaction rows are needed to answer the user's question well.

AVAILABLE DATABASE CONTEXT:
- Transaction categories: {categories}
- Available months (last 6): {available_months}
- Full transaction schema: date, amount, category, description, op_type

USER QUESTION: "{user_message}"

The main system prompt already includes aggregated summary metrics (category totals and monthly dynamics).
Detailed transaction rows are NOT included there.

Decide:
1. needs_data: do we need detailed transaction rows for a complete answer?
   - true: requests about specific transactions, purchase lists, exact items, or details by category/period/merchant
   - false: general analytics, advice, comparisons, planning

2. Select expense_scope:
   - overview: high-level expense answer
   - category_breakdown: expenses by category
   - time_trend: monthly/period trend
   - merchant_detail: merchants/descriptions/transaction details
   - anomaly: unusual spikes or outliers

3. Select context_profile:
   - light: only summary sections
   - medium: summary + aggregates
   - deep: summary + aggregates + sample transactions

4. If needs_data=true, specify what data is needed (max 1-2 queries).

Return ONLY valid JSON with no explanations:
{{
  "needs_data": true/false,
  "reason": "one short sentence",
  "expense_scope": "overview|category_breakdown|time_trend|merchant_detail|anomaly",
  "context_profile": "light|medium|deep",
  "queries": [
    {{
      "type": "by_category|by_period|by_description|top_expenses|top_income|anomaly_detail",
      "category": "category name or null",
      "month": "YYYY-MM or null",
      "keyword": "keyword or null",
      "top_n": 20,
      "sort_by": "amount_desc|date_desc"
    }}
  ]
}}

If needs_data=false, queries must be [].
"""

        try:
            route_messages = [{"role": "user", "content": prompt}]
            request_payload = {
                "scope": "router",
                "model": self.model,
                "messages_count": len(route_messages),
                "temperature": 0,
                "messages_redacted": self._redact_messages_for_log(route_messages),
            }
            logger.info("[API][OpenAI][Запрос]\n%s", self._pretty_json(request_payload))

            started_at = perf_counter()
            response = self._create_chat_completion(
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
            queries = self._normalize_router_queries(queries)
            if not needs_data:
                queries = []

            scope = str(decision.get("expense_scope") or "").strip().lower()
            if scope not in {"overview", "category_breakdown", "time_trend", "merchant_detail", "anomaly"}:
                scope = self._infer_expense_scope(user_message)

            profile = str(decision.get("context_profile") or "").strip().lower()
            if profile not in {"light", "medium", "deep"}:
                profile = self._default_context_profile(scope, needs_data)

            normalized = {
                "needs_data": needs_data,
                "reason": decision.get("reason", ""),
                "queries": queries,
                "expense_scope": scope,
                "context_profile": profile,
            }
            logger.info(
                "[ROUTER] needs_data=%s scope=%s profile=%s reason=%s",
                normalized.get("needs_data"),
                normalized.get("expense_scope"),
                normalized.get("context_profile"),
                normalized.get("reason"),
            )
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
            fallback_scope = self._infer_expense_scope(user_message)
            return {
                "needs_data": False,
                "reason": "router_error",
                "queries": [],
                "expense_scope": fallback_scope,
                "context_profile": "light",
            }

    def _fetch_detail(self, queries: list, route_decision: dict | None = None) -> str:
        """
        Формирует компактный TOON detail-pack:
        1) summary-метрики
        2) агрегаты (категории/мерчанты)
        3) sample-транзакции (в зависимости от context_profile).
        """
        if not queries or self.df is None:
            return ""

        context_profile = str((route_decision or {}).get("context_profile") or "medium").lower()
        if context_profile not in {"light", "medium", "deep"}:
            context_profile = "medium"

        if context_profile == "light":
            return ""

        packs: list[str] = []
        for query in queries[:2]:
            try:
                filtered_df, title = self._apply_detail_query(query)
                pack = self._build_expense_detail_pack(
                    title=title,
                    df=filtered_df,
                    query=query,
                    context_profile=context_profile,
                )
                if pack:
                    packs.append(pack)
                if sum(len(p) for p in packs) >= self.MAX_DETAIL_CHARS:
                    break
            except Exception as exc:
                logger.warning("[FETCH_DETAIL] Ошибка запроса %s: %s", query.get("type"), exc)
                continue

        detail_block = "\n\n".join(packs)
        if len(detail_block) > self.MAX_DETAIL_CHARS:
            detail_block = self._degrade_detail_block(detail_block, self.MAX_DETAIL_CHARS)

        return detail_block

    def _apply_detail_query(self, query: dict) -> tuple[pd.DataFrame, str]:
        """Применяет query к DataFrame и возвращает выборку + заголовок блока."""
        if self.df is None:
            return pd.DataFrame(), "Detail"

        q_type = str(query.get("type") or "").strip()
        sort_by = str(query.get("sort_by") or "amount_desc")
        try:
            top_n = int(query.get("top_n", 20))
        except Exception:
            top_n = 20
        top_n = max(1, min(50, top_n))
        df = self.df.copy()
        title = "Detail"

        if q_type == "by_category":
            cat = str(query.get("category") or "").strip()
            title = f"Category detail: {cat or 'unknown'}"
            if cat and "category" in df.columns:
                mask = df["category"].astype(str).str.lower().str.contains(cat.lower(), na=False)
                df = df[mask]
            else:
                return pd.DataFrame(), title
        elif q_type == "by_period":
            month = str(query.get("month") or "").strip()
            title = f"Period detail: {month or 'unknown'}"
            if month and "date" in df.columns:
                periods = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
                df = df[periods == month]
            else:
                return pd.DataFrame(), title
        elif q_type == "by_description":
            keyword = str(query.get("keyword") or "").strip()
            title = f"Merchant/detail keyword: {keyword or 'unknown'}"
            if keyword and "description" in df.columns:
                mask = df["description"].astype(str).str.lower().str.contains(keyword.lower(), na=False)
                df = df[mask]
            else:
                return pd.DataFrame(), title
        elif q_type == "top_expenses":
            title = "Top expenses detail"
        elif q_type == "top_income":
            title = "Top income detail"
            if "op_type" in df.columns:
                df = df[df["op_type"] == "доход"]
        elif q_type == "anomaly_detail":
            month = str(query.get("month") or "").strip()
            title = f"Anomaly month detail: {month or 'unknown'}"
            if month and "date" in df.columns:
                periods = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
                df = df[periods == month]
            else:
                return pd.DataFrame(), title
        else:
            return pd.DataFrame(), title

        if q_type in {"by_category", "by_period", "by_description", "top_expenses", "anomaly_detail"}:
            df = self._keep_expense_rows(df)

        if sort_by == "date_desc" and "date" in df.columns:
            df = df.sort_values("date", ascending=False)
        elif sort_by == "amount_desc" and "amount" in df.columns:
            df = df.sort_values("amount", ascending=False)

        if q_type in {"top_expenses", "top_income"}:
            df = df.head(top_n)

        return df, title

    def _keep_expense_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Для expense-query всегда оставляет только расходные строки."""
        if "op_type" not in df.columns:
            return df
        return df[df["op_type"] == "расход"]

    def _build_expense_detail_pack(
        self,
        title: str,
        df: pd.DataFrame,
        query: dict,
        context_profile: str,
    ) -> str:
        """Собирает TOON-блок с метриками, агрегатами и sample-транзакциями."""
        if df.empty:
            return f"### {title}\n_No transactions found._"

        amount = pd.to_numeric(df.get("amount"), errors="coerce").dropna() if "amount" in df.columns else pd.Series(dtype="float64")
        total = float(amount.sum()) if not amount.empty else 0.0
        count = int(len(amount))
        avg = float(amount.mean()) if not amount.empty else 0.0
        median = float(amount.median()) if not amount.empty else 0.0
        p90 = float(amount.quantile(0.9)) if not amount.empty else 0.0

        parts: list[str] = [f"### {title}"]
        parts.append("#### Summary metrics")
        parts.append(
            self._md_table(
                ["sum ₽", "count", "avg ₽", "median ₽", "p90 ₽"],
                [[self._fmt_num(total), count, self._fmt_num(avg), self._fmt_num(median), self._fmt_num(p90)]],
            )
        )

        cat_rows = self._build_category_aggregate_rows(df, query)
        if cat_rows:
            parts.append("#### Aggregates: categories")
            parts.append(self._md_table(["Category", "Amount ₽", "Transactions"], cat_rows))

        merchant_rows = self._build_merchant_aggregate_rows(df, query)
        if merchant_rows:
            parts.append("#### Aggregates: merchants/descriptions")
            parts.append(self._md_table(["Merchant/Description", "Amount ₽", "Transactions"], merchant_rows))

        sample_n = 0
        if context_profile == "deep":
            sample_n = self.MAX_DETAIL_ROWS

        if sample_n > 0:
            sample_df = df.head(sample_n).copy()
            if "description" in sample_df.columns:
                sample_df["description"] = sample_df["description"].apply(lambda x: self._truncate_text(str(x), 72))
            if "amount" in sample_df.columns:
                sample_df["amount"] = sample_df["amount"].apply(self._fmt_num)
            if "date" in sample_df.columns:
                sample_df["date"] = sample_df["date"].apply(self._fmt_date)

            cols = [c for c in ["date", "amount", "category", "description"] if c in sample_df.columns]
            if cols:
                parts.append("#### Sample transactions")
                parts.append(self._df_to_markdown(sample_df[cols].rename(columns={
                    "date": "Date",
                    "amount": "Amount ₽",
                    "category": "Category",
                    "description": "Description",
                })))

        return "\n".join(parts)

    def _build_category_aggregate_rows(self, df: pd.DataFrame, query: dict) -> list[list[Any]]:
        """Возвращает top-категории по сумме (использует кэш, когда возможно)."""
        q_type = str(query.get("type") or "")
        if q_type == "top_expenses" and isinstance(self.expense_cache.get("expense_by_category"), pd.DataFrame):
            source = self.expense_cache["expense_by_category"].head(5)
            return [
                [self._safe(row.get("category")), self._fmt_num(row.get("sum")), int(row.get("count", 0))]
                for _, row in source.iterrows()
            ]

        if "category" not in df.columns or "amount" not in df.columns:
            return []
        grouped = (
            df.groupby("category", dropna=False)["amount"]
            .agg(["sum", "count"])
            .sort_values("sum", ascending=False)
            .head(5)
            .reset_index()
        )
        return [
            [self._safe(row.get("category")), self._fmt_num(row.get("sum")), int(row.get("count", 0))]
            for _, row in grouped.iterrows()
        ]

    def _build_merchant_aggregate_rows(self, df: pd.DataFrame, query: dict) -> list[list[Any]]:
        """Возвращает top-мерчанты/описания (использует кэш recurring, когда возможно)."""
        q_type = str(query.get("type") or "")
        if q_type == "top_expenses" and isinstance(self.expense_cache.get("recurring_expenses"), pd.DataFrame):
            source = self.expense_cache["recurring_expenses"].head(5)
            return [
                [self._safe(row.get("description")), self._fmt_num(row.get("total_amount")), int(row.get("count", 0))]
                for _, row in source.iterrows()
            ]

        if "description" not in df.columns or "amount" not in df.columns:
            return []
        grouped = (
            df.groupby("description", dropna=False)["amount"]
            .agg(["sum", "count"])
            .sort_values("sum", ascending=False)
            .head(5)
            .reset_index()
        )
        return [
            [self._truncate_text(self._safe(row.get("description")), 60), self._fmt_num(row.get("sum")), int(row.get("count", 0))]
            for _, row in grouped.iterrows()
        ]

    def _degrade_detail_block(self, detail_block: str, limit: int) -> str:
        """Понижает детализацию detail-блока в несколько шагов до целевого лимита."""
        if len(detail_block) <= limit:
            return detail_block

        # 1) убрать sample-транзакции (самый тяжёлый слой).
        compact = re.sub(r"\n#### Sample transactions[\s\S]*?(?=\n### |\Z)", "", detail_block, flags=re.MULTILINE)
        if len(compact) <= limit:
            return compact

        # 2) подрезать длинные строки.
        trimmed_lines = []
        for line in compact.splitlines():
            if len(line) > 180:
                trimmed_lines.append(line[:177] + "...")
            else:
                trimmed_lines.append(line)
        compact = "\n".join(trimmed_lines)
        if len(compact) <= limit:
            return compact

        # 3) жёсткое ограничение.
        return compact[:limit] + "\n\n_Detail pack was truncated to fit context budget._"

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

        prompt = f"""I have a CSV file with financial transactions.
Here are all columns and sample values:

{schema_text}

Map source columns to the standard schema fields.
Use ONLY column names from the list above. Do not invent new names.

Standard fields:
- date        - transaction date
- amount      - transaction amount (single numeric column)
- category    - category / operation type
- description - description / payment purpose / counterparty
- op_type     - operation type (income/expense), if present
- income_col  - income-only column (if income and expenses are split)
- expense_col - expense-only column (if income and expenses are split)

Also infer:
- amount_format:
    "standard"    - regular number (1234.56 or -1234)
    "space_comma" - thousand separator as space and decimal as comma ("1 234,56")
    "space_dot"   - thousand separator as space and decimal as dot ("1 234.56")
- amount_sign:
    "signed"      - sign is inside amount column (+/-)
    "split_cols"  - income and expenses are in separate columns (income_col/expense_col)
    "op_type_col" - separate operation-type column exists (op_type)

Return ONLY valid JSON, no explanations, no markdown:
{{
  "date":          "column name or null",
  "amount":        "column name or null",
  "category":      "column name or null",
  "description":   "column name or null",
  "op_type":       "column name or null",
  "income_col":    "column name or null",
  "expense_col":   "column name or null",
  "amount_format": "standard|space_comma|space_dot",
  "amount_sign":   "signed|split_cols|op_type_col"
}}"""

        try:
            request_payload = {
                "scope": "schema",
                "model": self.model,
                "messages_count": 1,
                "temperature": 0,
                "messages_redacted": self._redact_messages_for_log([{"role": "user", "content": prompt}]),
            }
            logger.info("[API][OpenAI][Запрос]\n%s", self._pretty_json(request_payload))

            started_at = perf_counter()
            response = self._create_chat_completion(
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
        parts.append(f"## 📁 File: {filename}")
        if date_from is not None and date_to is not None:
            parts.append(f"- Period: {date_from.strftime('%d.%m.%Y')} – {date_to.strftime('%d.%m.%Y')}")
        else:
            parts.append("- Period: —")
        parts.append(f"- Total transactions: {self._fmt_num(len(df))}")
        parts.append(f"- Months in data: {self._fmt_num(months_count) if months_count else '—'}")
        parts.append("")

        parts.append("## 💰 Overall Metrics")
        parts.append(
            self._md_table(
                ["Metric", "Value"],
                [
                    ["Total income", f"{self._fmt_num(total_income)} ₽"],
                    ["Total expenses", f"{self._fmt_num(total_expenses)} ₽"],
                    ["Balance", f"{self._fmt_signed(balance)} ₽"],
                    ["Average income/month", f"{self._fmt_num(avg_income)} ₽" if avg_income is not None else "—"],
                    ["Average expense/month", f"{self._fmt_num(avg_expenses)} ₽" if avg_expenses is not None else "—"],
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
                    cat_name = str(cat) if pd.notna(cat) and str(cat).strip() else "Uncategorized"
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
            parts.append("## 📊 Expenses by Category")
            parts.append(
                self._md_table(
                    ["Category", "Amount ₽", "% of expenses", "Transactions", "Average ₽"],
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
            parts.append("## 📅 Monthly Dynamics")
            parts.append(
                self._md_table(
                    ["Month", "Income ₽", "Expense ₽", "Balance ₽"],
                    month_rows if month_rows else [["—", "0", "0", "0"]],
                )
            )

        parts.append("")
        parts.append("## 🔝 Top-10 Largest Expenses")
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
            top_rows = [["—", "0", "—", "No expenses"]]

        parts.append(self._md_table(["Date", "Amount ₽", "Category", "Description"], top_rows))

        if has_date:
            anomaly_rows = []
            if not monthly_expense.empty:
                mean_exp = float(monthly_expense.mean())
                if mean_exp > 0:
                    for month, value in monthly_expense.items():
                        val = float(value)
                        deviation = (val - mean_exp) / mean_exp
                        if abs(deviation) > 0.30:
                            direction = "higher" if deviation > 0 else "lower"
                            anomaly_rows.append([
                                month,
                                f"Spending is {abs(deviation) * 100:.0f}% {direction} than average ({self._fmt_num(val)} ₽)",
                            ])

            if not anomaly_rows:
                anomaly_rows = [["—", "No significant spending deviations found"]]

            parts.append("")
            parts.append("## ⚠️ Anomalies and Observations")
            parts.append(self._md_table(["Month", "Observation"], anomaly_rows[:12]))

        summary = "\n".join(parts)
        if len(summary) > 12000:
            summary = summary[:12000] + "\n\n*Summary was truncated to reduce token usage.*"

        return summary

    def _extract_summary_sections(self, summary: str) -> dict[str, str]:
        """Разбивает полный TOON-summary на логические секции для селективной передачи в LLM."""
        if not summary:
            return {}

        blocks: list[tuple[str, list[str]]] = []
        current_title = ""
        current_lines: list[str] = []
        for line in summary.splitlines():
            if line.startswith("## "):
                if current_title or current_lines:
                    blocks.append((current_title, current_lines))
                current_title = line
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_title or current_lines:
            blocks.append((current_title, current_lines))

        indexed = {title: "\n".join(lines).strip() for title, lines in blocks}
        sections: dict[str, str] = {}

        overview_blocks: list[str] = []
        for title in indexed:
            if "📁 File" in title or "💰 Overall Metrics" in title:
                overview_blocks.append(indexed[title])
        sections["overview"] = "\n\n".join(overview_blocks).strip() if overview_blocks else summary

        sections["expense_categories"] = next(
            (block for title, block in indexed.items() if "📊 Expenses by Category" in title),
            "",
        )
        sections["monthly_dynamics"] = next(
            (block for title, block in indexed.items() if "📅 Monthly Dynamics" in title),
            "",
        )
        sections["top_expenses"] = next(
            (block for title, block in indexed.items() if "🔝 Top-10 Largest Expenses" in title),
            "",
        )
        sections["anomalies"] = next(
            (block for title, block in indexed.items() if "⚠️ Anomalies and Observations" in title),
            "",
        )

        return sections

    def _compose_system_context(self, route_decision: dict) -> str:
        """Собирает селективный контекст из summary-секций по типу пользовательского запроса."""
        if not self.summary_sections:
            return self.csv_summary or ""

        scope = str(route_decision.get("expense_scope") or "overview")
        selected_keys: list[str]
        if scope in {"time_trend", "anomaly"}:
            selected_keys = ["overview", "monthly_dynamics", "anomalies"]
        elif scope in {"category_breakdown", "merchant_detail"}:
            selected_keys = ["overview", "expense_categories"]
        else:
            selected_keys = ["overview", "expense_categories", "top_expenses"]

        chunks = [self.summary_sections.get(k, "").strip() for k in selected_keys]
        chunks = [chunk for chunk in chunks if chunk]
        if not chunks:
            return self.csv_summary or ""
        return "\n\n".join(chunks)

    def _fit_context_budget(self, system_content: str, detail_block: str, route_decision: dict) -> tuple[str, str]:
        """Ограничивает размер system+detail контекста в несколько шагов деградации."""
        system_text = system_content
        detail_text = detail_block

        if len(system_text) > self.MAX_SYSTEM_CONTEXT_CHARS:
            system_text = system_text[: self.MAX_SYSTEM_CONTEXT_CHARS] + "\n\n[System summary truncated]"
        if len(detail_text) > self.MAX_DETAIL_CHARS:
            detail_text = self._degrade_detail_block(detail_text, self.MAX_DETAIL_CHARS)

        total_len = len(system_text) + len(detail_text)
        if total_len <= self.MAX_TOTAL_CONTEXT_CHARS:
            return system_text, detail_text

        if detail_text:
            detail_text = re.sub(r"\n#### Sample transactions[\s\S]*?(?=\n### |\Z)", "", detail_text, flags=re.MULTILINE)
            total_len = len(system_text) + len(detail_text)
            if total_len <= self.MAX_TOTAL_CONTEXT_CHARS:
                return system_text, detail_text

        minimal_sections = {
            "needs_data": False,
            "expense_scope": "overview",
            "context_profile": "light",
            "queries": [],
        }
        minimal_summary = self._compose_system_context(minimal_sections)
        minimal_context = SYSTEM_PROMPT if not minimal_summary else f"{SYSTEM_PROMPT}\n\n{minimal_summary}"
        if len(minimal_context) > self.MAX_SYSTEM_CONTEXT_CHARS:
            minimal_context = minimal_context[: self.MAX_SYSTEM_CONTEXT_CHARS] + "\n\n[System summary truncated]"
        system_text = minimal_context

        if len(system_text) + len(detail_text) > self.MAX_TOTAL_CONTEXT_CHARS:
            allowed_detail = max(0, self.MAX_TOTAL_CONTEXT_CHARS - len(system_text))
            if allowed_detail == 0:
                detail_text = ""
            else:
                detail_text = self._degrade_detail_block(detail_text, allowed_detail)

        return system_text, detail_text

    def _infer_expense_scope(self, user_message: str) -> str:
        """Эвристика fallback для expense_scope, если роутер не вернул валидного значения."""
        text = (user_message or "").lower()
        if any(word in text for word in ["аномал", "скач", "выброс", "нетип", "откл"]):
            return "anomaly"
        if any(word in text for word in ["месяц", "месяцам", "динамик", "тренд", "период", "ноябр", "декабр", "январ"]):
            return "time_trend"
        if any(word in text for word in ["подписк", "категор", "где трачу", "на что трачу"]):
            return "category_breakdown"
        if any(word in text for word in ["транзакц", "чек", "мерчант", "магазин", "детали", "подроб"]):
            return "merchant_detail"
        return "overview"

    def _default_context_profile(self, scope: str, needs_data: bool) -> str:
        """Возвращает глубину контекста по умолчанию."""
        if not needs_data:
            return "light"
        if scope in {"merchant_detail", "anomaly"}:
            return "deep"
        if scope in {"category_breakdown", "time_trend"}:
            return "medium"
        return "medium"

    def _normalize_router_queries(self, queries: list[dict]) -> list[dict]:
        """Нормализует и ограничивает router-queries для безопасного исполнения."""
        allowed_types = {
            "by_category",
            "by_period",
            "by_description",
            "top_expenses",
            "top_income",
            "anomaly_detail",
        }
        allowed_sort = {"amount_desc", "date_desc"}
        normalized: list[dict] = []

        for raw in queries[:2]:
            if not isinstance(raw, dict):
                continue

            q_type = str(raw.get("type") or "").strip()
            if q_type not in allowed_types:
                continue

            category = self._truncate_text(str(raw.get("category") or "").strip(), 64) or None
            month = str(raw.get("month") or "").strip()
            if month and not re.fullmatch(r"\d{4}-\d{2}", month):
                month = None
            keyword = self._truncate_text(str(raw.get("keyword") or "").strip(), 64) or None

            try:
                top_n = int(raw.get("top_n", 20))
            except Exception:
                top_n = 20
            top_n = max(1, min(50, top_n))

            sort_by = str(raw.get("sort_by") or "amount_desc").strip()
            if sort_by not in allowed_sort:
                sort_by = "amount_desc"

            normalized.append(
                {
                    "type": q_type,
                    "category": category,
                    "month": month,
                    "keyword": keyword,
                    "top_n": top_n,
                    "sort_by": sort_by,
                }
            )

        return normalized

    def _build_expense_cache(self, df: pd.DataFrame) -> dict[str, Any]:
        """Готовит лёгкий кэш агрегатов по расходам для ускорения detail-pack."""
        cache: dict[str, Any] = {
            "expense_by_category": pd.DataFrame(),
            "expense_by_month": pd.DataFrame(),
            "recurring_expenses": pd.DataFrame(),
        }
        if df.empty:
            return cache

        expense_df = df.copy()
        if "op_type" in expense_df.columns:
            expense_df = expense_df[expense_df["op_type"] == "расход"]
        if expense_df.empty:
            return cache

        if "category" in expense_df.columns and "amount" in expense_df.columns:
            cache["expense_by_category"] = (
                expense_df.groupby("category", dropna=False)["amount"]
                .agg(["sum", "count", "mean"])
                .sort_values("sum", ascending=False)
                .reset_index()
            )

        if "date" in expense_df.columns and "amount" in expense_df.columns:
            work = expense_df.copy()
            work["month"] = pd.to_datetime(work["date"], errors="coerce").dt.to_period("M").astype(str)
            cache["expense_by_month"] = (
                work.groupby("month", dropna=False)["amount"]
                .agg(["sum", "count", "mean"])
                .sort_values("sum", ascending=False)
                .reset_index()
            )

        if "description" in expense_df.columns and "amount" in expense_df.columns:
            work = expense_df.copy()
            work["description"] = work["description"].astype(str).str.strip()
            recurring = (
                work.groupby("description", dropna=False)["amount"]
                .agg(total_amount="sum", count="count", avg_amount="mean")
                .sort_values(["count", "total_amount"], ascending=[False, False])
                .reset_index()
            )
            cache["recurring_expenses"] = recurring[recurring["count"] >= 2]

        return cache

    def _truncate_text(self, text: str, limit: int) -> str:
        """Обрезает длинный текст для экономии контекста."""
        clean = str(text or "").replace("\n", " ").strip()
        if len(clean) <= limit:
            return clean
        return clean[: max(0, limit - 1)] + "…"

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

    def _redact_messages_for_log(self, messages: list[dict]) -> list[dict[str, Any]]:
        redacted: list[dict[str, Any]] = []
        for msg in messages or []:
            content = str(msg.get("content") or "")
            redacted.append(
                {
                    "role": str(msg.get("role") or ""),
                    "chars": len(content),
                    "hash": hashlib.sha256(content.encode("utf-8")).hexdigest()[:12],
                }
            )
        return redacted

    def _create_chat_completion(self, **kwargs):
        """Единый вызов через абстракцию LLMClient."""
        return self.llm_client.chat_completion(**kwargs)

    def _sanitize_reply_text(self, text: str) -> str:
        """Удаляет markdown-heading маркеры и нормализует формат ответа."""
        if not text:
            return ""
        sanitized = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", str(text))
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()
