import json
import logging
import os
import queue
import threading

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from agent import FinancialAgent
from storage import (
    clear_session_csv,
    clear_session_messages,
    create_session,
    init_db,
    load_csv_file,
    load_session,
    save_csv_file,
    save_csv_meta,
    save_message,
    session_exists,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

app = Flask(__name__)
init_db()

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


def _get_or_create_session() -> str:
    """Читает session_id из cookie или создаёт новую сессию."""
    sid = request.cookies.get("session_id")
    if sid and session_exists(sid):
        return sid
    return create_session()


def _set_session_cookie(resp: Response, session_id: str) -> Response:
    """Устанавливает cookie сессии на 30 дней."""
    resp.set_cookie("session_id", session_id, max_age=60 * 60 * 24 * 30, httponly=True)
    return resp


@app.route("/")
def index():
    logger.info("[HTTP] Запрос GET /")
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_csv():
    """Принимает CSV-файл, анализирует и сохраняет метаданные в сессию."""
    logger.info("[UPLOAD] Получен запрос на загрузку CSV")

    if "file" not in request.files:
        _log_http_request(route="/upload", payload={"has_file": False, "filename": None, "size_bytes": 0})
        response_payload = {"success": False, "error": "Файл не найден"}
        _log_http_response("/upload", 400, response_payload)
        return jsonify(response_payload), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        response_payload = {"success": False, "error": "Нужен файл формата .csv"}
        _log_http_response("/upload", 400, response_payload)
        return jsonify(response_payload), 400

    session_id = _get_or_create_session()
    content = file.read()
    _log_http_request(
        route="/upload",
        payload={"has_file": True, "filename": file.filename, "size_bytes": len(content), "session_id": session_id},
    )

    csv_path = save_csv_file(content, file.filename)
    result = agent.load_csv(content, file.filename)

    if result.get("success"):
        save_csv_meta(
            session_id=session_id,
            filename=file.filename,
            csv_summary=agent.csv_summary or "",
            schema_map=(result.get("analysis") or {}).get("schema_detected", {}),
            csv_path=csv_path,
        )

    response = jsonify(result)
    _log_http_response(
        "/upload",
        200,
        {
            "session_id": session_id,
            "success": result.get("success"),
            "error": result.get("error"),
            "analysis_rows": (result.get("analysis") or {}).get("rows"),
        },
    )
    return _set_session_cookie(response, session_id)


@app.route("/chat", methods=["POST"])
def chat():
    """Принимает сообщение пользователя и возвращает ответ агента."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    session_id = _get_or_create_session()

    _log_http_request("/chat", {"message": message, "session_id": session_id})
    if not message:
        response_payload = {"error": "Пустое сообщение"}
        _log_http_response("/chat", 400, response_payload)
        return jsonify(response_payload), 400

    try:
        reply = agent.chat(message)
        save_message(session_id, "user", message)
        save_message(session_id, "assistant", reply)

        response = jsonify({"reply": reply})
        _log_http_response("/chat", 200, {"session_id": session_id, "reply_len": len(reply)})
        return _set_session_cookie(response, session_id)
    except Exception as exc:
        logger.exception("[CHAT] Ошибка обработки запроса чата: %s", exc)
        response_payload = {"error": str(exc)}
        _log_http_response("/chat", 500, response_payload)
        return jsonify(response_payload), 500


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """SSE-эндпоинт: стримит шаги planning-цикла и финальный ответ."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    session_id = _get_or_create_session()
    _log_http_request("/chat/stream", {"message": message, "session_id": session_id})

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
                if agent.df is not None:
                    result = agent.planner.run(
                        user_message=message,
                        df=agent.df,
                        csv_summary=agent.csv_summary or "",
                    )
                else:
                    result = None

                if result is None:
                    result = agent.chat(message)
                    is_planning = False
                else:
                    is_planning = True

                save_message(session_id, "user", message, is_planning=False)
                save_message(session_id, "assistant", result, is_planning=is_planning)

                stream_status["done"] = True
                logger.info("[CHAT][SSE][DONE] reply_len=%s planning=%s", len(result), is_planning)
                events.put(("done", {"text": result}))
            except Exception as exc:
                logger.exception("[CHAT][SSE] Ошибка обработки: %s", exc)
                stream_status["error"] = str(exc)
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
                "session_id": session_id,
                "done": stream_status["done"],
                "error": stream_status["error"],
            },
        )

    response = Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    return _set_session_cookie(response, session_id)


@app.route("/session/restore", methods=["GET"])
def restore_session():
    """Восстанавливает состояние агента из БД по session_id cookie."""
    session_id = request.cookies.get("session_id")
    _log_http_request("/session/restore", {"session_id": session_id})

    if not session_id or not session_exists(session_id):
        payload = {"found": False}
        _log_http_response("/session/restore", 200, payload)
        return jsonify(payload)

    data = load_session(session_id)
    if not data:
        payload = {"found": False}
        _log_http_response("/session/restore", 200, payload)
        return jsonify(payload)

    has_messages = bool(data.get("messages"))
    has_csv_meta = bool(data.get("csv_summary") or data.get("filename") or data.get("csv_path"))
    if not has_messages and not has_csv_meta:
        payload = {"found": False}
        _log_http_response("/session/restore", 200, payload)
        return jsonify(payload)

    agent.conversation_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in data.get("messages", [])
    ]

    if data.get("csv_summary"):
        agent.csv_summary = data.get("csv_summary")

    df_available = False
    csv_path = data.get("csv_path")
    filename = data.get("filename")
    if csv_path and filename:
        csv_bytes = load_csv_file(csv_path)
        if csv_bytes is None:
            logger.warning("[RESTORE] CSV на диске не найден, работаем только со сводкой")
            df_available = False
            agent.df = None
        else:
            try:
                restored = agent.load_csv(csv_bytes, filename, restore_mode=True)
                df_available = bool(restored.get("success"))
                if not df_available:
                    logger.warning("[RESTORE] load_csv вернул ошибку: %s", restored.get("error"))
            except Exception as exc:
                logger.warning("[RESTORE] Не удалось восстановить DataFrame: %s", exc)
                df_available = False
                agent.df = None

    logger.info(
        "[RESTORE] Сессия восстановлена: %s… сообщений=%s df_available=%s",
        session_id[:8],
        len(data.get("messages", [])),
        df_available,
    )

    payload = {
        "found": True,
        "session_id": session_id,
        "filename": data.get("filename"),
        "has_csv": bool(data.get("csv_summary")),
        "df_available": df_available,
        "messages": data.get("messages", []),
    }
    _log_http_response(
        "/session/restore",
        200,
        {
            "found": payload["found"],
            "session_id": payload["session_id"],
            "messages_count": len(payload["messages"]),
            "has_csv": payload["has_csv"],
            "df_available": payload["df_available"],
        },
    )
    return jsonify(payload)


@app.route("/session/new", methods=["POST"])
def new_session():
    """Создаёт новую сессию и сбрасывает состояние агента."""
    sid = create_session()
    agent.reset()
    payload = {"session_id": sid, "success": True}
    response = jsonify(payload)
    _log_http_response("/session/new", 200, {"success": True, "session_id": sid})
    return _set_session_cookie(response, sid)


@app.route("/reset", methods=["POST"])
def reset():
    """Очищает историю и CSV текущей сессии, затем сбрасывает состояние агента."""
    session_id = request.cookies.get("session_id")
    _log_http_request("/reset", {"session_id": session_id})

    if session_id and session_exists(session_id):
        clear_session_messages(session_id)
        clear_session_csv(session_id, delete_file=True)

    agent.reset()
    payload = {"success": True, "cleared": ["history", "csv"]}
    _log_http_response("/reset", 200, payload)
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info("[START] Сервер запущен → http://localhost:%s", port)
    app.run(debug=True, port=port)
