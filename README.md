# 💰 ФинСоветник

Веб-приложение на `Python + Flask`, где AI-агент анализирует CSV с транзакциями и отвечает на вопросы по личным финансам.

## Что умеет

- Загружать CSV (drag&drop или через выбор файла)
- Автоматически нормализовать колонки и считать сводку
- Вести чат с memory layers: short-term / working / long-term
- Показывать токены/стоимость и заполненность контекстного окна
- Переключать модель прямо в UI
- Переключать стратегию управления контекстом через debug-панель
- Селективно передавать расходный контекст (summary sections + compact detail pack)

## Текущие дефолты

- Модель по умолчанию: `gpt-5-mini`
- Runtime по умолчанию: `memory_layers`
- `Short-term` лимит: последние `30` сообщений
- `Long-term retrieve`: `top_k=3` релевантных decisions/notes
- `Long-term` не очищается при `/reset` и `/session/new`

## Структура проекта

```text
AI_Advent_Challenge_6_day/
├── agent.py                # основной агент: CSV + чат + роутинг детализации
├── app.py                  # Flask API и веб-роуты
├── context_strategies.py   # стратегии контекста (sliding, sticky, branching, compression)
├── llm/
│   ├── client.py           # интерфейс LLMClient и унифицированные типы ответа
│   ├── openai_client.py    # реализация OpenAI + model-compat
│   └── mock_client.py      # mock-клиент для тестов
├── memory/
│   ├── manager.py          # фасад memory layers
│   ├── short_term.py       # short-term память по session_id
│   ├── working.py          # рабочая память задачи + state machine
│   ├── long_term.py        # долгосрочная память profile/decisions/notes
│   ├── router.py           # write policy (memory routing)
│   ├── prompt_builder.py   # сборка prompt по слоям
│   └── models.py           # модели памяти
├── storage.py              # SQLite-хранилище сессий/сообщений
├── scripts/
│   └── demo_memory_layers.py
├── templates/
│   └── index.html          # интерфейс (HTML/CSS/JS)
├── tests/
│   ├── test_agent_expense_context.py
│   ├── test_llm_client.py
│   ├── test_memory_layers.py
│   ├── test_normalize.py
│   ├── test_storage.py
│   └── test_strategies.py
├── requirements.txt
└── README.md
```

## Стек

- Backend: `Python`, `Flask`
- LLM: `OpenAI API`
- Data: `pandas`
- Storage: `SQLite`
- Frontend: `HTML/CSS/Vanilla JS`

## Быстрый старт

### 1. Создать и активировать окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Для Windows:

```bat
.venv\Scripts\activate
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Задать API-ключ OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

Или через `.env`:

```text
OPENAI_API_KEY=sk-...
```

### 4. Запустить сервер

```bash
python3 app.py
```

Откройте: [http://localhost:5000](http://localhost:5000)

## Формат CSV

Агент пытается автоматически определить схему. Лучше всего работают файлы с полями вида:

- `date` / `дата`
- `amount` / `сумма`
- `category` / `категория`
- `description` / `описание`
- `op_type` / `тип`

Поддерживается несколько форматов:

- единая колонка суммы с `+/-`
- раздельные колонки доход/расход
- разные разделители и кодировки (в т.ч. UTF-8/CP1251)

## Выбор модели

Модель можно менять:

- через селектор в шапке интерфейса
- через API `POST /model`

Текущий список поддерживаемых моделей берётся из `FinancialAgent.COST_PER_1M`.

## API

### Основные

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/` | UI |
| `GET` | `/models` | Текущая модель + список доступных |
| `POST` | `/model` | Сменить модель |
| `POST` | `/upload` | Загрузить и проанализировать CSV |
| `POST` | `/chat` | Сообщение в чат |
| `POST` | `/reset` | Сброс истории и CSV |
| `GET` | `/session/restore` | Восстановить сессию |
| `POST` | `/session/new` | Создать новую сессию |

### Debug/context

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/debug/memory-layers` | Снимок трёх слоёв памяти для DebugMenu |
| `POST` | `/debug/memory/working/clear` | Очистить только рабочую память текущей сессии |
| `POST` | `/debug/memory/long-term/delete` | Удалить конкретную запись `decision/note` из long-term |
| `POST` | `/debug/ctx-strategy` | Сменить контекстную стратегию |
| `POST` | `/ctx/checkpoint` | Создать checkpoint (branching) |
| `POST` | `/ctx/fork` | Создать ветку от checkpoint (branching) |
| `POST` | `/ctx/switch-branch` | Переключить ветку (branching) |

### Пример `/chat`

```json
POST /chat
Content-Type: application/json

{
  "message": "На что я трачу больше всего?",
  "model": "gpt-5-mini"
}
```

## Контекстные стратегии

- `sliding_window` — последние N сообщений
- `sticky_facts` — факты + последние N сообщений (legacy/debug)
- `branching` — ветки диалога
- `history_compression` — summary + последние N

## Memory Layers

### Слои памяти

1. `Short-term memory`  
   Хранит только последние сообщения текущей сессии (`role`, `content`, `timestamp`) с лимитом `N`.  
   Политика retention: при каждом `append` выполняется pruning по `session_id` (в таблице short-term остаются только последние `N` записей).
2. `Working memory`  
   Хранит `TaskContext` текущей задачи: `task_id`, `goal`, `state`, `plan`, `current_step`, `done_steps`, `open_questions`, `artifacts`, `vars`.
3. `Long-term memory`  
   Хранит устойчивые данные между сессиями:
   - `Profile` (style/constraints/context)
   - `Decisions`
   - `Notes`

### Memory Router (write policy)

- Long-term записывается только из `source=user` (источник истины).
- Для `source=assistant` long-term запись идёт только в `pending`; применение только после явного подтверждения пользователя: `ПОДТВЕРЖДАЮ ПАМЯТЬ #<id>`.
- В `long_term.profile` записываются только устойчивые инструкции (например, «всегда…», «с этого момента…», «не используй…»).
- В `long_term.decision` записываются только явные решения («решили…», «используем…», «стандарт…»).
- В `long_term.note` записываются только устойчивые заметки из user-ввода; `ttl_days` по умолчанию `90`.
- В `working` записываются требования/изменения текущей задачи («требование…», «эндпоинт…», «добавь шаг…»).
- Не записываются в long-term разовые детали, предположения модели и неподтверждённые факты.
- На каждом ходе пишется лог: `[MEMORY_WRITE] layer=... keys=...`.

### Memory Reader (read policy)

- `Long-term`: на каждом ходе выполняется retrieve по запросу, в prompt добавляется компактный `[PROFILE]` + top-k (`3`) релевантных `[DECISIONS]`/`[NOTES]`.
- `Working`: если есть активная задача, всегда добавляется `TaskContext`.
- `Short-term`: всегда читаются последние `N` сообщений из short-term таблицы (already pruned).
- Логи чтения:
  - `[MEMORY_READ] layer=long_term hits=<n> ids=[...] reason=<match/score>`
  - `[MEMORY_READ] layer=working present=<true/false>`
  - `[MEMORY_READ] layer=short_term turns=<n>`

### DebugMenu по слоям памяти

- В DebugMenu блок `Слои памяти` показывает три отдельные вкладки: `Краткосрочная`, `Рабочая`, `Долговременная`.
- Есть ручные действия:
  - `Очистить рабочую` — очищает только `working` слой (short-term/long-term не трогаются).
  - `Удалить` на карточках `Решение/Заметка` — удаляет только выбранную long-term запись.
- Есть индикатор последних записей (`layer`, `operation`, `keys`, `timestamp`) для проверки write-policy в реальном времени.

### Prompt Builder

Порядок сборки prompt:
1. `SYSTEM INSTRUCTIONS`
2. `LONG-TERM`
3. `WORKING`
4. `SHORT-TERM`
5. `USER QUERY`

При `state=PLANNING` агент блокирует запросы вида «сразу код» до перехода в `EXECUTION`.

## Передача данных о расходах

- В `agent.py` summary разбивается на секции: `overview`, `expense_categories`, `monthly_dynamics`, `top_expenses`, `anomalies`
- Для каждого вопроса роутер выбирает `expense_scope` и `context_profile` (`light|medium|deep`)
- Детализация отправляется как compact detail pack:
  - `Summary metrics` (sum/count/avg/median/p90)
  - агрегаты по категориям и мерчантам
  - короткий sample транзакций (только для `deep`)
- Для расходных detail-запросов применяется фильтр `op_type == "расход"`
- Ограничения контекста:
  - `MAX_DETAIL_ROWS = 24`
  - `MAX_DETAIL_CHARS = 7000`
  - `MAX_SYSTEM_CONTEXT_CHARS = 18000`

## Хранилище памяти (SQLite)

Новые таблицы:
- `memory_short_term_messages`
- `memory_working_tasks`
- `memory_longterm_profile`
- `memory_longterm_decisions`
- `memory_longterm_notes`
- `memory_longterm_pending` (для assistant-proposed long-term записей, требующих подтверждения)

Разделение:
- `short-term` и `working` — по `session_id`
- `long-term` — по `user_id` (между сессиями)
- Полная debug-история диалога хранится отдельно в `messages` и не участвует в short-term retention.

## Demo для видео

Команда (real OpenAI):

```bash
PYTHONPATH=. python3 -m scripts.demo_memory_layers --real-llm
```

Требуется переменная окружения:

```bash
export OPENAI_API_KEY="sk-..."
```

### Video checklist

1. Запустить demo-скрипт и показать `SCENE A`:
   - short-term окно переполняется
   - ранние реплики исчезают из контекста
2. Показать `SCENE B`:
   - `PLANNING` блокирует «сразу финальный план бюджета»
   - после перехода в `EXECUTION` пошаговый финансовый план разрешается
3. Показать `SCENE C`:
   - записываются long-term правила/решения
   - после «рестарта» агент отвечает с учётом сохранённой долгосрочной памяти
4. Показать блок `[DEMO CONCLUSIONS]` в конце демо.
5. В кадре показать:
   - `Short-term snapshot`: последние N turn’ов (`role + content[:80]`)
   - `Working snapshot`: `state`, `plan`, `current_step`, `done_steps`, `open_questions`
   - `Long-term snapshot`: profile + top-k decisions/notes (`title + tags`)
   - `Prompt preview`: `[PROFILE]`, `[DECISIONS]`, `[WORKING]`, `[SHORT_TERM]`
   - логи `[MEMORY_WRITE]`

## Тесты

Запуск через `pytest`:

```bash
PYTHONPATH=. python3 -m pytest tests -q
```

Или напрямую:

```bash
PYTHONPATH=. python3 tests/test_normalize.py
PYTHONPATH=. python3 tests/test_storage.py
PYTHONPATH=. python3 tests/test_strategies.py
PYTHONPATH=. python3 tests/test_memory_layers.py
PYTHONPATH=. python3 tests/test_llm_client.py
```

## Хранилище данных

- SQLite база: `data/agent.db`
- Загруженные CSV: `uploads/`

## Лицензия

MIT
