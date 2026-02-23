from flask import Flask, request, jsonify, render_template
from agent import FinancialAgent
import os
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

app = Flask(__name__)

# Один экземпляр агента на сессию (для учебного задания этого достаточно)
agent = FinancialAgent()
logger.info("[INIT] FinancialAgent инициализирован")


@app.route("/")
def index():
    logger.info("[HTTP] Запрос GET /")
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_csv():
    """Принимает CSV-файл и передаёт агенту для анализа."""
    logger.info("[UPLOAD] Получен запрос на загрузку CSV")

    if "file" not in request.files:
        logger.warning("[UPLOAD] В запросе отсутствует поле 'file'")
        return jsonify({"success": False, "error": "Файл не найден"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        logger.warning("[UPLOAD] Неверное расширение файла: %s", file.filename)
        return jsonify({"success": False, "error": "Нужен файл формата .csv"}), 400

    content = file.read()
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

    return jsonify(result)


@app.route("/chat", methods=["POST"])
def chat():
    """Принимает сообщение пользователя и возвращает ответ агента."""
    logger.info("[CHAT] Получен запрос сообщения")

    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()

        if not message:
            logger.warning("[CHAT] Пустое сообщение")
            return jsonify({"error": "Пустое сообщение"}), 400

        reply = agent.chat(message)
        logger.info("[CHAT] Ответ успешно сформирован")
        return jsonify({"reply": reply})
    except Exception as e:
        logger.exception("[CHAT] Ошибка обработки запроса чата: %s", e)
        return jsonify({"error": str(e)}), 500


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
