import json
import logging
import os
import re
from uuid import uuid4

from flask import Flask, Response, jsonify, render_template, request

from agent import FinancialAgent
from storage import (
    add_usage,
    clear_session_csv,
    clear_session_messages,
    create_session,
    get_latest_session_id,
    init_db,
    load_csv_file,
    load_session,
    save_csv_file,
    save_csv_meta,
    save_ctx_state,
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


def _extract_context_overflow(error_text: str) -> dict | None:
    """Пытается извлечь фактический размер контекста из ошибки превышения лимита."""
    if not error_text:
        return None
    max_match = re.search(r"maximum context length is (\d+) tokens", error_text)
    used_match = re.search(r"messages resulted in (\d+) tokens", error_text)
    if not max_match and not used_match:
        return None
    return {
        "context_limit": int(max_match.group(1)) if max_match else 0,
        "prompt_tokens": int(used_match.group(1)) if used_match else 0,
    }


def _build_ctx_state() -> dict:
    """Формирует состояние активной context-стратегии для фронтенда."""
    stats = agent.ctx.stats(agent.conversation_history)
    return {
        "strategy": agent.ctx.active,
        "stats": stats,
        "memory": agent.last_memory_stats or {},
    }


def _extract_last_turn_for_storage(default_user: str, default_assistant: str) -> tuple[str, str]:
    """Возвращает последние user/assistant реплики для сохранения в БД."""
    if agent.ctx.active == "branching":
        branch_msgs = agent.ctx.strategy.branches.get(agent.ctx.strategy.active_branch, [])
        assistant_content = default_assistant
        user_content = default_user
        if branch_msgs:
            if branch_msgs[-1].get("role") == "assistant":
                assistant_content = branch_msgs[-1].get("content", default_assistant)
            for msg in reversed(branch_msgs):
                if msg.get("role") == "user":
                    user_content = msg.get("content", default_user)
                    break
        return user_content, assistant_content

    user_content = default_user
    if len(agent.conversation_history) >= 2:
        user_content = agent.conversation_history[-2].get("content", default_user)
    assistant_content = default_assistant
    if agent.conversation_history:
        assistant_content = agent.conversation_history[-1].get("content", default_assistant)
    return user_content, assistant_content


def _get_or_create_session() -> str:
    """Читает session_id из cookie или создаёт новую сессию."""
    sid = request.cookies.get("session_id")
    if sid and session_exists(sid):
        return sid
    latest_sid = get_latest_session_id()
    if latest_sid and session_exists(latest_sid):
        logger.info("[SESSION] Cookie не найдена, использую последнюю сессию: %s…", latest_sid[:8])
        return latest_sid
    return create_session()


def _get_or_create_user_id() -> str:
    """Читает user_id из cookie или создаёт новый."""
    existing = (request.cookies.get("user_id") or "").strip()
    if existing:
        return existing
    return f"user_{uuid4().hex}"


def _set_session_cookie(resp: Response, session_id: str, user_id: str | None = None) -> Response:
    """Устанавливает cookie сессии и пользователя на 30 дней."""
    resp.set_cookie("session_id", session_id, max_age=60 * 60 * 24 * 30, httponly=True)
    if user_id:
        resp.set_cookie("user_id", user_id, max_age=60 * 60 * 24 * 30, httponly=True)
    return resp


@app.route("/")
def index():
    logger.info("[HTTP] Запрос GET /")
    return render_template("index.html")


@app.route("/models", methods=["GET"])
def get_models():
    """Возвращает текущую модель и список доступных моделей."""
    payload = {
        "current_model": agent.model,
        "available_models": agent.available_models(),
    }
    _log_http_response("/models", 200, payload)
    return jsonify(payload)


@app.route("/model", methods=["POST"])
def set_model():
    """Переключает модель для последующих запросов."""
    data = request.get_json(silent=True) or {}
    model = str(data.get("model", "")).strip()
    _log_http_request("/model", {"model": model})
    try:
        current_model = agent.set_model(model)
        payload = {
            "success": True,
            "current_model": current_model,
            "available_models": agent.available_models(),
        }
        _log_http_response("/model", 200, payload)
        return jsonify(payload)
    except ValueError as exc:
        payload = {
            "success": False,
            "error": str(exc),
            "current_model": agent.model,
            "available_models": agent.available_models(),
        }
        _log_http_response("/model", 400, payload)
        return jsonify(payload), 400


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
        schema_stats = agent.last_schema_token_stats or {}
        if schema_stats:
            add_usage(
                session_id,
                tokens_in=int(schema_stats.get("prompt_tokens", 0) or 0),
                tokens_out=int(schema_stats.get("completion_tokens", 0) or 0),
                cost_usd=float(schema_stats.get("cost_usd", 0.0) or 0.0),
            )
        save_csv_meta(
            session_id=session_id,
            filename=file.filename,
            csv_summary=agent.csv_summary or "",
            schema_map=(result.get("analysis") or {}).get("schema_detected", {}),
            csv_path=csv_path,
        )
        save_ctx_state(session_id, agent.ctx.dump())

    if result.get("success"):
        result["token_stats"] = agent.last_schema_token_stats or {}
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
    requested_model = str(data.get("model", "")).strip()
    session_id = _get_or_create_session()
    user_id = _get_or_create_user_id()

    if requested_model:
        try:
            agent.set_model(requested_model)
        except ValueError as exc:
            response_payload = {"error": str(exc), "current_model": agent.model}
            _log_http_response("/chat", 400, response_payload)
            return jsonify(response_payload), 400

    _log_http_request(
        "/chat",
        {
            "message": message,
            "session_id": session_id,
            "user_id": user_id,
            "requested_model": requested_model or None,
            "current_model": agent.model,
        },
    )
    if not message:
        response_payload = {"error": "Пустое сообщение"}
        _log_http_response("/chat", 400, response_payload)
        return jsonify(response_payload), 400

    try:
        reply = agent.chat(message, session_id=session_id, user_id=user_id)
        user_content, assistant_content = _extract_last_turn_for_storage(message, reply)
        save_message(session_id, "user", user_content)
        token_stats = agent.last_token_stats or {}
        save_message(
            session_id,
            "assistant",
            assistant_content,
            tokens_in=int(token_stats.get("prompt_tokens", 0) or 0),
            tokens_out=int(token_stats.get("completion_tokens", 0) or 0),
            cost_usd=float(token_stats.get("cost_usd", 0.0) or 0.0),
        )

        save_ctx_state(session_id, agent.ctx.dump())

        ctx_state = _build_ctx_state()
        response_payload = {
            "reply": reply,
            "model": agent.model,
            "token_stats": token_stats,
            "memory_stats": agent.last_memory_stats or {},
            "prompt_preview": agent.last_prompt_preview or {},
            "ctx_state": ctx_state,
            "ctx_stats": ctx_state["stats"],
            "ctx_strategy": ctx_state["strategy"],
        }
        response = jsonify(response_payload)
        _log_http_response(
            "/chat",
            200,
            {
                "session_id": session_id,
                "reply_len": len(reply),
                "token_stats": token_stats,
                "memory_stats": agent.last_memory_stats or {},
            },
        )
        return _set_session_cookie(response, session_id, user_id=user_id)
    except Exception as exc:
        logger.exception("[CHAT] Ошибка обработки запроса чата: %s", exc)
        err_text = str(exc)
        overflow = _extract_context_overflow(err_text)
        response_payload = {"error": err_text}
        status = 500
        if overflow:
            response_payload["token_stats"] = {
                "prompt_tokens": int(overflow.get("prompt_tokens") or 0),
                "completion_tokens": 0,
                "total_tokens": int(overflow.get("prompt_tokens") or 0),
                "cost_usd": 0.0,
                "latency_ms": 0,
                "scope": "chat",
                "error": True,
                "context_limit": int(overflow.get("context_limit") or 0),
            }
            status = 400
        _log_http_response("/chat", status, response_payload)
        return jsonify(response_payload), status


@app.route("/debug/memory-layers", methods=["GET"])
def debug_memory_layers():
    """Возвращает снимок трёх слоёв памяти для Debug UI."""
    session_id = _get_or_create_session()
    user_id = _get_or_create_user_id()
    query = (request.args.get("q") or "").strip()
    try:
        top_k = int(request.args.get("top_k", 3))
    except (TypeError, ValueError):
        top_k = 3
    top_k = max(1, min(10, top_k))
    _log_http_request(
        "/debug/memory-layers",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…", "q": query[:50] or "(empty)", "top_k": top_k},
    )
    try:
        snapshot = agent.memory.debug_snapshot(
            session_id=session_id,
            user_id=user_id,
            query=query,
            top_k=top_k,
        )
        summary = {
            "short_term_turns": snapshot.get("short_term", {}).get("turns_count", 0),
            "working_present": snapshot.get("working", {}).get("present", False),
            "long_term_decisions": len(snapshot.get("long_term", {}).get("decisions_top_k") or []),
            "long_term_notes": len(snapshot.get("long_term", {}).get("notes_top_k") or []),
            "memory_writes": len(snapshot.get("memory_writes") or []),
        }
        _log_http_response("/debug/memory-layers", 200, summary)
        return jsonify(snapshot)
    except Exception as exc:
        logger.exception("[DEBUG] memory-layers failed: %s", exc)
        _log_http_response("/debug/memory-layers", 500, {"error": str(exc)})
        return jsonify({"error": str(exc)}), 500


@app.route("/debug/memory/working/clear", methods=["POST"])
def debug_clear_working_memory():
    """Очищает только рабочую память текущей сессии."""
    session_id = _get_or_create_session()
    user_id = _get_or_create_user_id()
    data = request.get_json(silent=True) or {}
    query = str(data.get("q") or "").strip()
    try:
        top_k = int(data.get("top_k", 3))
    except (TypeError, ValueError):
        top_k = 3
    top_k = max(1, min(10, top_k))
    _log_http_request(
        "/debug/memory/working/clear",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…", "top_k": top_k},
    )
    try:
        cleared = agent.memory.clear_working_layer(session_id=session_id)
        snapshot = agent.memory.debug_snapshot(session_id=session_id, user_id=user_id, query=query, top_k=top_k)
        payload = {"success": True, "cleared": bool(cleared), "snapshot": snapshot}
        _log_http_response(
            "/debug/memory/working/clear",
            200,
            {
                "success": True,
                "cleared": bool(cleared),
                "working_present": snapshot.get("working", {}).get("present", False),
                "memory_writes": len(snapshot.get("memory_writes") or []),
            },
        )
        return jsonify(payload)
    except Exception as exc:
        logger.exception("[DEBUG] memory working clear failed: %s", exc)
        _log_http_response("/debug/memory/working/clear", 500, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/debug/memory/long-term/delete", methods=["POST"])
def debug_delete_longterm_entry():
    """Удаляет конкретную запись из long-term памяти (decision/note)."""
    session_id = _get_or_create_session()
    user_id = _get_or_create_user_id()
    data = request.get_json(silent=True) or {}
    entry_type = str(data.get("entry_type") or "").strip().lower()
    entry_id_raw = data.get("id")
    query = str(data.get("q") or "").strip()
    try:
        top_k = int(data.get("top_k", 3))
    except (TypeError, ValueError):
        top_k = 3
    top_k = max(1, min(10, top_k))
    _log_http_request(
        "/debug/memory/long-term/delete",
        {
            "session_id": session_id[:8] + "…",
            "user_id": user_id[:12] + "…",
            "entry_type": entry_type,
            "id": entry_id_raw,
            "top_k": top_k,
        },
    )
    if entry_type not in {"decision", "note"}:
        payload = {"success": False, "error": "entry_type должен быть decision или note"}
        _log_http_response("/debug/memory/long-term/delete", 400, payload)
        return jsonify(payload), 400
    try:
        entry_id = int(entry_id_raw)
    except (TypeError, ValueError):
        payload = {"success": False, "error": "id должен быть целым числом"}
        _log_http_response("/debug/memory/long-term/delete", 400, payload)
        return jsonify(payload), 400

    try:
        deleted = agent.memory.delete_long_term_entry(
            session_id=session_id,
            user_id=user_id,
            entry_type=entry_type,
            entry_id=entry_id,
        )
        snapshot = agent.memory.debug_snapshot(session_id=session_id, user_id=user_id, query=query, top_k=top_k)
        payload = {"success": True, "deleted": bool(deleted), "snapshot": snapshot}
        _log_http_response(
            "/debug/memory/long-term/delete",
            200,
            {
                "success": True,
                "deleted": bool(deleted),
                "entry_type": entry_type,
                "id": entry_id,
                "long_term_decisions": len(snapshot.get("long_term", {}).get("decisions_top_k") or []),
                "long_term_notes": len(snapshot.get("long_term", {}).get("notes_top_k") or []),
                "memory_writes": len(snapshot.get("memory_writes") or []),
            },
        )
        return jsonify(payload)
    except Exception as exc:
        logger.exception("[DEBUG] memory long-term delete failed: %s", exc)
        _log_http_response("/debug/memory/long-term/delete", 500, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/debug/ctx-strategy", methods=["POST"])
def debug_ctx_strategy():
    """Переключает стратегию управления контекстом."""
    data = request.get_json(silent=True) or {}
    strategy = data.get("strategy", "sticky_facts")
    try:
        agent.ctx.set_strategy(strategy)
        save_ctx_state(_get_or_create_session(), agent.ctx.dump())
        return jsonify(
            {
                "success": True,
                "strategy": agent.ctx.active,
                "stats": agent.ctx.stats(agent.conversation_history),
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/ctx/checkpoint", methods=["POST"])
def ctx_checkpoint():
    """Создать checkpoint в текущей точке диалога."""
    data = request.get_json(silent=True) or {}
    if agent.ctx.active != "branching":
        return jsonify({"error": "Branching стратегия не активна"}), 400
    name = data.get("name", f"cp_{len(agent.ctx.strategy.checkpoints) + 1}")
    result = agent.ctx.strategy.create_checkpoint(name)
    save_ctx_state(_get_or_create_session(), agent.ctx.dump())
    return jsonify({"success": True, "checkpoint": result, "stats": agent.ctx.stats(agent.conversation_history)})


@app.route("/ctx/fork", methods=["POST"])
def ctx_fork():
    """Создать ветку от checkpoint."""
    data = request.get_json(silent=True) or {}
    if agent.ctx.active != "branching":
        return jsonify({"error": "Branching стратегия не активна"}), 400
    checkpoint_name = data.get("checkpoint")
    branch_name = data.get("branch_name")
    try:
        new_branch = agent.ctx.strategy.fork(checkpoint_name, branch_name)
        save_ctx_state(_get_or_create_session(), agent.ctx.dump())
        return jsonify(
            {
                "success": True,
                "new_branch": new_branch,
                "branches": agent.ctx.strategy.list_branches(),
                "stats": agent.ctx.stats(agent.conversation_history),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/ctx/switch-branch", methods=["POST"])
def ctx_switch_branch():
    """Переключиться на ветку."""
    data = request.get_json(silent=True) or {}
    if agent.ctx.active != "branching":
        return jsonify({"error": "Branching стратегия не активна"}), 400
    branch = data.get("branch")
    try:
        agent.ctx.strategy.switch_branch(branch)
        save_ctx_state(_get_or_create_session(), agent.ctx.dump())
        return jsonify(
            {
                "success": True,
                "active_branch": agent.ctx.strategy.active_branch,
                "stats": agent.ctx.stats(agent.conversation_history),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/session/restore", methods=["GET"])
def restore_session():
    """Восстанавливает состояние агента из БД по session_id cookie."""
    session_id = request.cookies.get("session_id")
    user_id = _get_or_create_user_id()
    if not session_id or not session_exists(session_id):
        session_id = get_latest_session_id()
    _log_http_request("/session/restore", {"session_id": session_id})

    if not session_id:
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

    full_messages = data.get("messages", [])
    agent.conversation_history = [{"role": msg["role"], "content": msg["content"]} for msg in full_messages]
    agent.restore_memory_session(session_id=session_id, messages=agent.conversation_history)
    agent.last_memory_stats = agent.memory.stats(session_id=session_id, user_id=user_id)

    if data.get("csv_summary"):
        agent.csv_summary = data.get("csv_summary")

    if data.get("ctx_state"):
        agent.ctx.restore(data["ctx_state"])

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
        len(full_messages),
        df_available,
    )

    ui_messages = full_messages

    ctx_state = _build_ctx_state()
    payload = {
        "found": True,
        "session_id": session_id,
        "current_model": agent.model,
        "available_models": agent.available_models(),
        "filename": data.get("filename"),
        "has_csv": bool(data.get("csv_summary")),
        "df_available": df_available,
        "messages": ui_messages,
        "token_stats_session": {
            "total_tokens_in": int(data.get("total_tokens_in", 0) or 0),
            "total_tokens_out": int(data.get("total_tokens_out", 0) or 0),
            "total_cost_usd": float(data.get("total_cost_usd", 0.0) or 0.0),
            "cost_history": data.get("cost_history", []),
        },
        "ctx_state": ctx_state,
        "ctx_stats": ctx_state["stats"],
        "ctx_strategy": ctx_state["strategy"],
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
    response = jsonify(payload)
    return _set_session_cookie(response, session_id, user_id=user_id)


@app.route("/session/new", methods=["POST"])
def new_session():
    """Создаёт новую сессию и сбрасывает состояние агента."""
    sid = create_session()
    user_id = _get_or_create_user_id()
    agent.reset()
    agent.clear_session_memory(sid)
    payload = {"session_id": sid, "success": True}
    response = jsonify(payload)
    _log_http_response("/session/new", 200, {"success": True, "session_id": sid})
    return _set_session_cookie(response, sid, user_id=user_id)


@app.route("/reset", methods=["POST"])
def reset():
    """Очищает историю и CSV текущей сессии, затем сбрасывает состояние агента."""
    session_id = request.cookies.get("session_id")
    _log_http_request("/reset", {"session_id": session_id})

    if session_id and session_exists(session_id):
        clear_session_messages(session_id)
        clear_session_csv(session_id, delete_file=True)
        agent.clear_session_memory(session_id)

    agent.reset()
    payload = {"success": True, "cleared": ["history", "csv"]}
    _log_http_response("/reset", 200, payload)
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info("[START] Сервер запущен → http://localhost:%s", port)
    app.run(debug=True, port=port)
