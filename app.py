import json
import logging
import os
import queue
import threading

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from agent import FinancialAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

app = Flask(__name__)

# Один экземпляр агента на сессию (для учебного задания этого достаточно)
agent = FinancialAgent()
logger.info("[INIT] FinancialAgent инициализирован")

def _pretty_json(payload: dict) -> str:
    """Форматирует JSON для читаемых логов."""
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(payload)


def _client_ip() -> str:
    """Возвращает IP клиента с учётом прокси-заголовка."""
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _log_http_request(route: str, payload: dict) -> None:
    """Логирует входящий HTTP-запрос."""
    log_payload = {
        "route": route,
        "method": request.method,
        "client_ip": _client_ip(),
        "payload": payload,
    }
    logger.info("[HTTP][Запрос]\n%s", _pretty_json(log_payload))


def _log_http_response(route: str, status_code: int, payload: dict) -> None:
    """Логирует исходящий HTTP-ответ."""
    log_payload = {
        "route": route,
        "status_code": status_code,
        "response": payload,
    }
    logger.info("[HTTP][Ответ]\n%s", _pretty_json(log_payload))


@app.route("/")
def index():
    logger.info("[HTTP] Запрос GET /")
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_csv():
    """Принимает CSV-файл и передаёт агенту для анализа."""
    logger.info("[UPLOAD] Получен запрос на загрузку CSV")

    if "file" not in request.files:
        _log_http_request(
            route="/upload",
            payload={"has_file": False, "filename": None, "size_bytes": 0},
        )
        logger.warning("[UPLOAD] В запросе отсутствует поле 'file'")
        response_payload = {"success": False, "error": "Файл не найден"}
        _log_http_response("/upload", 400, response_payload)
        return jsonify(response_payload), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        logger.warning("[UPLOAD] Неверное расширение файла: %s", file.filename)
        response_payload = {"success": False, "error": "Нужен файл формата .csv"}
        _log_http_response("/upload", 400, response_payload)
        return jsonify(response_payload), 400

    content = file.read()
    _log_http_request(
        route="/upload",
        payload={"has_file": True, "filename": file.filename, "size_bytes": len(content)},
    )
    logger.info("[UPLOAD] Обрабатываю файл=%s размер_байт=%s", file.filename, len(content))

    result = agent.load_csv(content, file.filename)
    if result.get("success"):
        analysis = result.get("analysis", {})
        columns_normalized = analysis.get("columns_normalized", []) or []
        schema_detected = analysis.get("schema_detected", {}) or {}
        logger.info(
            "[UPLOAD] CSV успешно проанализирован строк=%s колонок=%s колонка_суммы=%s источник_схемы=%s",
            analysis.get("rows"),
            len(columns_normalized),
            schema_detected.get("amount"),
            analysis.get("schema_source"),
        )
    else:
        logger.error("[UPLOAD] Ошибка анализа CSV: %s", result.get("error"))

    try:
        json.dumps(result, allow_nan=False)
        logger.info("[UPLOAD] Проверка JSON-ответа пройдена")
    except (TypeError, ValueError) as payload_error:
        logger.error("[UPLOAD] Проверка JSON-ответа не пройдена: %s", payload_error)

    _log_http_response(
        "/upload",
        200,
        {
            "success": result.get("success"),
            "error": result.get("error"),
            "analysis_rows": (result.get("analysis") or {}).get("rows"),
            "analysis_columns": (result.get("analysis") or {}).get("columns_normalized"),
        },
    )
    return jsonify(result)


@app.route("/chat", methods=["POST"])
def chat():
    """Принимает сообщение пользователя и возвращает ответ агента."""
    logger.info("[CHAT] Получен запрос сообщения")

    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        _log_http_request("/chat", {"message": message})

        if not message:
            logger.warning("[CHAT] Пустое сообщение")
            response_payload = {"error": "Пустое сообщение"}
            _log_http_response("/chat", 400, response_payload)
            return jsonify(response_payload), 400

        reply = agent.chat(message)
        logger.info("[CHAT] Ответ успешно сформирован")
        response_payload = {"reply": reply}
        _log_http_response("/chat", 200, {"reply_len": len(reply), "reply": reply})
        return jsonify(response_payload)
    except Exception as e:
        logger.exception("[CHAT] Ошибка обработки запроса чата: %s", e)
        response_payload = {"error": str(e)}
        _log_http_response("/chat", 500, response_payload)
        return jsonify(response_payload), 500


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """SSE-эндпоинт: стримит шаги planning-цикла и финальный ответ."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    _log_http_request("/chat/stream", {"message": message})
    if not message:
        response_payload = {"error": "Пустое сообщение"}
        _log_http_response("/chat/stream", 400, response_payload)
        return jsonify(response_payload), 400

    def generate():
        events: queue.Queue[tuple[str, dict]] = queue.Queue()
        stream_status = {"done": False, "error": None}

        def on_step(event: dict):
            logger.info(
                "[CHAT][SSE][STEP] id=%s status=%s label=%s detail=%s",
                event.get("id"),
                event.get("status"),
                event.get("label"),
                event.get("detail"),
            )
            events.put(("step", event))

        def run_worker():
            agent.planner.on_step = on_step
            try:
                result = agent.chat(message)
                stream_status["done"] = True
                logger.info("[CHAT][SSE][DONE] reply_len=%s", len(result))
                events.put(("done", {"text": result}))
            except Exception as exc:
                logger.exception("[CHAT][SSE] Ошибка обработки: %s", exc)
                stream_status["error"] = str(exc)
                logger.error("[CHAT][SSE][ERROR] %s", exc)
                events.put(("error", {"error": str(exc)}))
            finally:
                agent.planner.on_step = None
                events.put(("end", {}))

        worker = threading.Thread(target=run_worker, daemon=True)
        worker.start()

        while True:
            event_type, payload = events.get()
            if event_type == "end":
                break
            yield f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        _log_http_response(
            "/chat/stream",
            200,
            {
                "done": stream_status["done"],
                "error": stream_status["error"],
            },
        )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Сброс диалога и данных."""
    logger.info("[RESET] Получен запрос на сброс")
    agent.reset()
    logger.info("[RESET] Состояние агента очищено")
    return jsonify({"success": True})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info("[START] Сервер запущен → http://localhost:%s", port)
    app.run(debug=True, port=port)
