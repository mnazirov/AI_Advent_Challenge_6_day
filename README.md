# 💰 ФинСоветник

Веб-приложение на `Python + Flask`, где AI-агент анализирует CSV с транзакциями и отвечает на вопросы по личным финансам.

## Что умеет

- Загружать CSV (drag&drop или через выбор файла)
- Автоматически нормализовать колонки и считать сводку
- Вести чат с контекстной памятью
- Показывать токены/стоимость и заполненность контекстного окна
- Переключать модель прямо в UI
- Переключать стратегию управления контекстом через debug-панель
- Селективно передавать расходный контекст (summary sections + compact detail pack)

## Текущие дефолты

- Модель по умолчанию: `gpt-5-mini`
- Стратегия контекста по умолчанию: `sticky_facts`
- Для `sticky_facts` в контексте сохраняются последние `30` сообщений

## Структура проекта

```text
AI_Advent_Challenge_6_day/
├── agent.py                # основной агент: CSV + чат + роутинг детализации
├── app.py                  # Flask API и веб-роуты
├── context_strategies.py   # стратегии контекста (sliding, sticky, branching, compression)
├── storage.py              # SQLite-хранилище сессий/сообщений
├── templates/
│   └── index.html          # интерфейс (HTML/CSS/JS)
├── tests/
│   ├── test_agent_expense_context.py
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
- `sticky_facts` — факты + последние N сообщений (дефолт, N=30)
- `branching` — ветки диалога
- `history_compression` — summary + последние N

## Передача данных о расходах

- В `agent.py` summary разбивается на секции: `overview`, `expense_categories`, `monthly_dynamics`, `top_expenses`, `anomalies`
- Для каждого вопроса роутер выбирает `expense_scope` и `context_profile` (`light|medium|deep`)
- Детализация отправляется как compact detail pack:
  - `Summary metrics` (sum/count/avg/median/p90)
  - агрегаты по категориям и мерчантам
  - короткий sample транзакций (только для `deep`)
- Для расходных detail-запросов применяется фильтр `op_type == "расход"`
- Ограничения контекста:
  - `MAX_DETAIL_ROWS = 12`
  - `MAX_DETAIL_CHARS = 3500`
  - `MAX_SYSTEM_CONTEXT_CHARS = 9000`

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
```

## Хранилище данных

- SQLite база: `data/agent.db`
- Загруженные CSV: `uploads/`

## Лицензия

MIT
