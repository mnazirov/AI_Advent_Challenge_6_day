"""
context_manager.py — Управление контекстом диалога с rolling summary.

Принцип работы:
- Полная история хранится в БД и памяти (conversation_history)
- В LLM отправляется: system + summary прошлых chunk'ов + последние RECENT_N сообщений
- Summary обновляется когда накопилось CHUNK_SIZE новых сообщений с момента последнего summary
"""

import logging

from openai import OpenAI

logger = logging.getLogger("context_manager")

CHUNK_SIZE = 10   # сколько сообщений сжимать в один summary
RECENT_N = 10     # сколько последних сообщений передавать "как есть"


class ContextManager:

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

        # summary — строка с резюме всей истории до recent-хвоста
        # Хранится в памяти и персистируется в SQLite
        self.summary: str = ""

        # Индекс до которого история уже свёрнута в summary
        # summary покрывает conversation_history[:summarized_up_to]
        self.summarized_up_to: int = 0

    # ── ПУБЛИЧНЫЙ ИНТЕРФЕЙС ────────────────────────────────────────────────

    def build_context(self, history: list[dict]) -> list[dict]:
        """
        Основной метод. Принимает полную историю диалога.
        Возвращает оптимизированный список сообщений для отправки в LLM.

        Формат возврата:
        [
            {"role": "system",    "content": "...summary_block..."},  ← если summary есть
            {"role": "user",      "content": "..."},                  ← последние RECENT_N
            {"role": "assistant", "content": "..."},
            ...
        ]

        Примечание: system prompt с csv_summary добавляется в agent.py отдельно,
        этот метод возвращает только историю.
        """
        # Обновляем summary если накопилось достаточно новых сообщений
        self._maybe_update_summary(history)

        # Берём последние RECENT_N сообщений "как есть"
        recent = history[-RECENT_N:] if len(history) > RECENT_N else history[:]

        # Строим итоговый список
        result = []

        if self.summary:
            result.append(
                {
                    "role": "system",
                    "content": self._format_summary_block(self.summary),
                }
            )

        result.extend(recent)
        return result

    def needs_update(self, history: list[dict]) -> bool:
        """
        Проверяет нужно ли обновить summary.
        Используется для предварительной проверки перед build_context.
        """
        unsummarized = len(history) - self.summarized_up_to
        # Есть достаточно сообщений И они не войдут в recent хвост
        return unsummarized >= CHUNK_SIZE and len(history) > RECENT_N + CHUNK_SIZE

    def restore(self, summary: str, summarized_up_to: int) -> None:
        """
        Восстанавливает состояние из БД при перезагрузке страницы.
        Вызывается в /session/restore.
        """
        self.summary = summary or ""
        self.summarized_up_to = summarized_up_to or 0
        logger.info(
            "[CTX] Восстановлен: summary=%s символов, summarized_up_to=%s",
            len(self.summary),
            self.summarized_up_to,
        )

    def reset(self) -> None:
        """Сброс при начале новой сессии."""
        self.summary = ""
        self.summarized_up_to = 0

    # ── ПРИВАТНЫЕ МЕТОДЫ ───────────────────────────────────────────────────

    def _maybe_update_summary(self, history: list[dict]) -> None:
        """
        Проверяет нужно ли обновить summary и делает это если нужно.

        Условие: накопилось CHUNK_SIZE новых сообщений за пределами recent-хвоста.
        recent-хвост не сжимаем — он передаётся как есть.

        Пример при CHUNK_SIZE=10, RECENT_N=10, len(history)=35:
        - summarized_up_to = 20  (уже свёрнуто 20 сообщений)
        - unsummarized = 35 - 20 = 15  (15 новых)
        - safe_to_summarize = 35 - 10 = 25  (до recent-хвоста)
        - chunk = history[20:25]  (5 сообщений, но ждём ещё 5)

        Пример при len(history)=30:
        - unsummarized = 30 - 20 = 10  ← ровно CHUNK_SIZE
        - safe_to_summarize = 30 - 10 = 20  (не трогаем recent)
        - chunk = history[20:20] = []  ← ничего не делаем (всё в recent)

        Пример при len(history)=40:
        - unsummarized = 40 - 20 = 20  >= CHUNK_SIZE
        - safe_to_summarize = 40 - 10 = 30
        - chunk = history[20:30]  ← 10 сообщений → сжимаем
        """
        if len(history) <= RECENT_N:
            return  # история короче recent — сжимать нечего

        safe_to_summarize = len(history) - RECENT_N
        new_chunk_end = (self.summarized_up_to // CHUNK_SIZE + 1) * CHUNK_SIZE

        if new_chunk_end > safe_to_summarize:
            return  # новый chunk ещё не накопился или уходит в recent

        chunk = history[self.summarized_up_to : new_chunk_end]
        if not chunk:
            return

        logger.info(
            "[CTX] Сжимаем сообщения [%s:%s] (%s сообщений)",
            self.summarized_up_to,
            new_chunk_end,
            len(chunk),
        )

        new_summary = self._summarize_chunk(chunk, existing_summary=self.summary)

        if new_summary:
            self.summary = new_summary
            self.summarized_up_to = new_chunk_end
            logger.info(
                "[CTX] Summary обновлён: %s символов, покрыто до сообщения #%s",
                len(self.summary),
                self.summarized_up_to,
            )

    def _summarize_chunk(self, chunk: list[dict], existing_summary: str) -> str:
        """
        LLM-вызов для сжатия chunk'а в текст.

        Если уже есть summary — обновляем его новыми фактами (не конкатенируем).
        Температура 0 — детерминированный результат.
        Стоимость: ~$0.00002 за вызов (150-200 токенов вход, 200-300 выход).
        """
        chunk_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:500]}"  # обрезаем длинные сообщения
            for m in chunk
        )

        if existing_summary:
            prompt = f"""Ты — ассистент для сжатия контекста диалога.

ТЕКУЩЕЕ РЕЗЮМЕ (уже накопленное):
{existing_summary}

НОВЫЕ СООБЩЕНИЯ ДИАЛОГА:
{chunk_text}

Обнови резюме — добавь важные факты из новых сообщений.
Не теряй важную информацию из текущего резюме.
Убирай детали которые уже не актуальны или противоречат новым данным.

Что обязательно сохранять в резюме:
- Финансовые цели пользователя (явные и неявные)
- Конкретные цифры которые обсуждались (суммы, категории, проценты)
- Принятые решения и договорённости
- Персональный контекст (упомянутые жизненные ситуации)
- На каком вопросе/задаче остановились

Формат — сжатый связный текст на русском, 5-10 предложений максимум.
Только текст, без заголовков и списков."""

        else:
            prompt = f"""Ты — ассистент для сжатия контекста диалога.

СООБЩЕНИЯ ДИАЛОГА:
{chunk_text}

Составь краткое резюме этого фрагмента диалога.

Что обязательно включить:
- Финансовые цели пользователя (явные и неявные)
- Конкретные цифры которые обсуждались (суммы, категории, проценты)
- Принятые решения и договорённости
- Персональный контекст (упомянутые жизненные ситуации)
- На каком вопросе/задаче остановились

Формат — сжатый связный текст на русском, 5-10 предложений максимум.
Только текст, без заголовков и списков."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            summary = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens
            cost = tokens * 0.00000015 + response.usage.completion_tokens * 0.00000045
            logger.info("[CTX][SUMMARY] tokens=%s cost=$%.6f", tokens, cost)
            return summary

        except Exception as e:
            logger.error("[CTX][SUMMARY] Ошибка LLM: %s — summary не обновлён", e)
            return ""  # безопасный fallback — оставляем старый summary

    def _format_summary_block(self, summary: str) -> str:
        """
        Оборачивает summary в блок для system prompt.
        Явная разметка помогает модели понять что это — сжатая история, а не инструкция.
        """
        return (
            "[ИСТОРИЯ ДИАЛОГА — СВОДКА]\n"
            "Ниже — краткое резюме предыдущей части нашего разговора. "
            "Используй эту информацию как контекст, но не ссылайся на неё явно.\n\n"
            f"{summary}\n"
            "[КОНЕЦ СВОДКИ]"
        )
