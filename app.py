import datetime
import json
import os
import sqlite3
import time
from functools import wraps
from flask import (
    Flask,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from google import genai
from google.genai import types

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-prod")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "study_planner.db")
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
VALID_SUBJECTS = ("Math", "English", "Science", "GK")
STUDY_TRACKS = (
    "RIMC Entrance",
    "Sainik School",
    "Navodaya (JNV)",
    "Grade 6",
    "Grade 7",
    "Grade 8",
    "Grade 9",
    "Grade 10",
    "Olympiad / General",
)


def get_db():
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            grade_level TEXT NOT NULL DEFAULT 'RIMC Entrance',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS study_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            planned_minutes INTEGER NOT NULL DEFAULT 30,
            completed_minutes INTEGER NOT NULL DEFAULT 0,
            is_done INTEGER NOT NULL DEFAULT 0,
            study_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS mock_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            test_date TEXT NOT NULL,
            subject TEXT NOT NULL,
            score REAL NOT NULL,
            max_score REAL NOT NULL,
            weak_topics TEXT NOT NULL DEFAULT '',
            mistake_notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS daily_activity (
            user_id INTEGER NOT NULL,
            activity_date TEXT NOT NULL,
            PRIMARY KEY (user_id, activity_date),
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "grade_level" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN grade_level TEXT NOT NULL DEFAULT 'RIMC Entrance';")
    conn.commit()
    conn.close()


init_db()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


def get_current_grade() -> str:
    return (session.get("grade_level") or "RIMC Entrance").strip()


def get_active_gemini_key() -> str | None:
    user_key = (session.get("gemini_api_key") or "").strip()
    if user_key:
        return user_key
    env_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    return env_key or None


def get_gemini_client() -> genai.Client:
    api_key = get_active_gemini_key()
    if not api_key:
        raise ValueError(
            "No Gemini API key configured. Enter one on the Sign-In page or set GEMINI_API_KEY."
        )
    return genai.Client(api_key=api_key)


def generate_with_retry(
    client: genai.Client,
    prompt: str,
    config: types.GenerateContentConfig | None = None,
    attempts: int = 3,
):
    last_exc = None
    for idx in range(attempts):
        try:
            if config is not None:
                return client.models.generate_content(
                    model=DEFAULT_GEMINI_MODEL,
                    contents=prompt,
                    config=config,
                )
            return client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=prompt,
            )
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            is_transient = "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg.lower()
            if not is_transient or idx == attempts - 1:
                raise
            time.sleep(1.5 * (idx + 1))
    raise last_exc


def record_activity(user_id: int, target_date: str | None = None):
    day_str = target_date or datetime.date.today().isoformat()
    db = get_db()
    db.execute(
        "INSERT OR IGNORE INTO daily_activity (user_id, activity_date) VALUES (?, ?)",
        (user_id, day_str),
    )
    db.commit()


def calculate_streak(user_id: int) -> int:
    db = get_db()
    rows = db.execute(
        "SELECT activity_date FROM daily_activity WHERE user_id = ? ORDER BY activity_date DESC",
        (user_id,),
    ).fetchall()
    if not rows:
        return 0

    active_days = {datetime.date.fromisoformat(r["activity_date"]) for r in rows}
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    if today in active_days:
        cursor = today
    elif yesterday in active_days:
        cursor = yesterday
    else:
        return 0

    streak = 0
    while cursor in active_days:
        streak += 1
        cursor -= datetime.timedelta(days=1)
    return streak


def get_today_stats(user_id: int) -> dict:
    today_str = datetime.date.today().isoformat()
    db = get_db()
    today_rows = db.execute(
        """
        SELECT planned_minutes, completed_minutes, is_done
        FROM study_blocks
        WHERE user_id = ? AND study_date = ?
        """,
        (user_id, today_str),
    ).fetchall()

    planned_minutes = sum(r["planned_minutes"] for r in today_rows)
    completed_minutes = sum(
        r["completed_minutes"]
        if r["completed_minutes"] > 0
        else (r["planned_minutes"] if r["is_done"] else 0)
        for r in today_rows
    )

    return {
        "planned_minutes": planned_minutes,
        "completed_minutes": completed_minutes,
        "streak_days": calculate_streak(user_id),
        "grade_level": get_current_grade(),
        "gemini_connected": bool(get_active_gemini_key()),
    }


def render_app_page(template_name: str, active_page: str, **extra):
    user_id = session["user_id"]
    stats = get_today_stats(user_id)
    return render_template(
        template_name,
        username=session.get("username"),
        subjects=VALID_SUBJECTS,
        tracks=STUDY_TRACKS,
        active_page=active_page,
        stats=stats,
        **extra,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session and request.method == "GET":
        return redirect(url_for("blocks_page"))

    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""
        grade_level = (request.form.get("grade_level") or "RIMC Entrance").strip()
        gemini_key = (request.form.get("gemini_api_key") or "").strip()

        if not username or not password:
            error = "Please enter both a username and password."
        else:
            db = get_db()
            user = db.execute(
                "SELECT id, username, password_hash, grade_level FROM users WHERE username = ?",
                (username,),
            ).fetchone()

            if user is None:
                pw_hash = generate_password_hash(password)
                now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                cur = db.execute(
                    """
                    INSERT INTO users (username, password_hash, grade_level, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (username, pw_hash, grade_level, now_iso),
                )
                db.commit()
                user_id = cur.lastrowid
                saved_grade = grade_level
            else:
                if not check_password_hash(user["password_hash"], password):
                    error = "Invalid username or password."
                    user_id = None
                    saved_grade = grade_level
                else:
                    user_id = user["id"]
                    if request.form.get("grade_level"):
                        db.execute(
                            "UPDATE users SET grade_level = ? WHERE id = ?",
                            (grade_level, user_id),
                        )
                        db.commit()
                        saved_grade = grade_level
                    else:
                        saved_grade = user["grade_level"] or "RIMC Entrance"

            if user_id is not None:
                session.clear()
                session["user_id"] = user_id
                session["username"] = username
                session["grade_level"] = saved_grade
                if gemini_key:
                    session["gemini_api_key"] = gemini_key
                return redirect(url_for("blocks_page"))

    return render_template(
        "login.html",
        error=error,
        tracks=STUDY_TRACKS,
        has_env_key=bool((os.environ.get("GEMINI_API_KEY") or "").strip()),
    )


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/profile/grade", methods=["PATCH"])
@login_required
def update_grade():
    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    grade_level = (data.get("grade_level") or "").strip()
    if not grade_level:
        return jsonify({"error": "Grade or exam level cannot be empty."}), 400

    db = get_db()
    db.execute("UPDATE users SET grade_level = ? WHERE id = ?", (grade_level, user_id))
    db.commit()
    session["grade_level"] = grade_level
    return jsonify({"grade_level": grade_level})


@app.route("/")
@login_required
def blocks_page():
    return render_app_page("blocks.html", "blocks")


@app.route("/ai")
@login_required
def ai_page():
    return render_app_page("ai_studio.html", "ai")


@app.route("/focus")
@login_required
def focus_page():
    return render_app_page("focus.html", "focus")


@app.route("/mocks")
@login_required
def mocks_page():
    return render_app_page("mocks.html", "mocks")


@app.route("/api/blocks", methods=["GET", "POST"])
@login_required
def blocks_collection():
    user_id = session["user_id"]
    db = get_db()

    if request.method == "GET":
        subject_filter = (request.args.get("subject") or "").strip()
        status_filter = (request.args.get("status") or "all").strip()

        query = "SELECT * FROM study_blocks WHERE user_id = ?"
        params: list = [user_id]

        if subject_filter in VALID_SUBJECTS:
            query += " AND subject = ?"
            params.append(subject_filter)

        if status_filter == "pending":
            query += " AND is_done = 0"
        elif status_filter == "done":
            query += " AND is_done = 1"

        query += " ORDER BY is_done ASC, id DESC"
        blocks = [dict(r) for r in db.execute(query, params).fetchall()]
        return jsonify({"blocks": blocks, "stats": get_today_stats(user_id)})

    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "Math").strip()
    if subject not in VALID_SUBJECTS:
        subject = "Math"

    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Topic name is required."}), 400

    try:
        planned_minutes = max(5, min(240, int(data.get("planned_minutes", 30))))
    except (TypeError, ValueError):
        planned_minutes = 30

    today_str = datetime.date.today().isoformat()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    cur = db.execute(
        """
        INSERT INTO study_blocks
            (user_id, subject, topic, planned_minutes, completed_minutes, is_done, study_date, created_at)
        VALUES (?, ?, ?, ?, 0, 0, ?, ?)
        """,
        (user_id, subject, topic, planned_minutes, today_str, now_iso),
    )
    db.commit()

    row = db.execute(
        "SELECT * FROM study_blocks WHERE id = ? AND user_id = ?",
        (cur.lastrowid, user_id),
    ).fetchone()
    return jsonify({"block": dict(row), "stats": get_today_stats(user_id)}), 201


@app.route("/api/blocks/<int:block_id>", methods=["PATCH", "DELETE"])
@login_required
def mutate_block(block_id: int):
    user_id = session["user_id"]
    db = get_db()
    existing = db.execute(
        "SELECT * FROM study_blocks WHERE id = ? AND user_id = ?",
        (block_id, user_id),
    ).fetchone()
    if existing is None:
        return jsonify({"error": "Study block not found."}), 404

    if request.method == "DELETE":
        db.execute("DELETE FROM study_blocks WHERE id = ? AND user_id = ?", (block_id, user_id))
        db.commit()
        return jsonify({"deleted": True, "id": block_id, "stats": get_today_stats(user_id)})

    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or existing["subject"]).strip()
    if subject not in VALID_SUBJECTS:
        subject = existing["subject"]

    topic = (data.get("topic") if "topic" in data else existing["topic"]).strip()
    if not topic:
        return jsonify({"error": "Topic name cannot be empty."}), 400

    planned_minutes = existing["planned_minutes"]
    if "planned_minutes" in data:
        try:
            planned_minutes = max(5, min(240, int(data["planned_minutes"])))
        except (TypeError, ValueError):
            pass

    completed_minutes = existing["completed_minutes"]
    if "add_completed_minutes" in data:
        try:
            added = max(0, int(data["add_completed_minutes"]))
            completed_minutes += added
            if added > 0:
                record_activity(user_id)
        except (TypeError, ValueError):
            pass

    is_done = existing["is_done"]
    if "is_done" in data:
        is_done = 1 if bool(data["is_done"]) else 0
        if is_done == 1:
            if completed_minutes == 0:
                completed_minutes = planned_minutes
            record_activity(user_id)

    db.execute(
        """
        UPDATE study_blocks
        SET subject = ?, topic = ?, planned_minutes = ?, completed_minutes = ?, is_done = ?
        WHERE id = ? AND user_id = ?
        """,
        (subject, topic, planned_minutes, completed_minutes, is_done, block_id, user_id),
    )
    db.commit()
    updated = db.execute(
        "SELECT * FROM study_blocks WHERE id = ? AND user_id = ?",
        (block_id, user_id),
    ).fetchone()
    return jsonify({"block": dict(updated), "stats": get_today_stats(user_id)})


@app.route("/api/mocks", methods=["GET", "POST"])
@login_required
def mocks_collection():
    user_id = session["user_id"]
    db = get_db()

    if request.method == "GET":
        mocks = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM mock_tests WHERE user_id = ? ORDER BY test_date DESC, id DESC",
                (user_id,),
            ).fetchall()
        ]
        return jsonify({"mocks": mocks, "stats": get_today_stats(user_id)})

    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "Math").strip()
    if subject not in VALID_SUBJECTS:
        subject = "Math"

    test_date = (data.get("test_date") or datetime.date.today().isoformat()).strip()
    weak_topics = (data.get("weak_topics") or "").strip()
    mistake_notes = (data.get("mistake_notes") or "").strip()

    try:
        score = float(data.get("score", 0))
        max_score = max(1.0, float(data.get("max_score", 100)))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric score values."}), 400

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cur = db.execute(
        """
        INSERT INTO mock_tests
            (user_id, test_date, subject, score, max_score, weak_topics, mistake_notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, test_date, subject, score, max_score, weak_topics, mistake_notes, now_iso),
    )
    record_activity(user_id, test_date)
    db.commit()

    row = db.execute(
        "SELECT * FROM mock_tests WHERE id = ? AND user_id = ?",
        (cur.lastrowid, user_id),
    ).fetchone()
    return jsonify({"mock": dict(row), "stats": get_today_stats(user_id)}), 201


@app.route("/api/ai/split-topic", methods=["POST"])
@login_required
def ai_split_topic():
    data = request.get_json(silent=True) or {}
    chapter = (data.get("chapter") or "").strip()
    subject = (data.get("subject") or "Math").strip()
    grade_level = get_current_grade()
    if subject not in VALID_SUBJECTS:
        subject = "Math"

    if not chapter:
        return jsonify({"error": "Please enter a chapter or broad topic to split."}), 400

    try:
        client = get_gemini_client()
        prompt = (
            f"Student target / grade: {grade_level}. "
            f"Break the {subject} chapter '{chapter}' into 4 to 6 focused, bite-sized study blocks "
            f"suited for {grade_level}. Return ONLY a JSON array of objects, each with keys "
            f"'topic' (string, concise subtopic title) and 'planned_minutes' (integer between 15 and 60)."
        )
        response = generate_with_retry(
            client,
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        parsed = json.loads((response.text or "[]").strip())
        blocks = []
        if isinstance(parsed, list):
            for item in parsed:
                t = str(item.get("topic", "")).strip()
                mins = max(10, min(90, int(item.get("planned_minutes", 25))))
                if t:
                    blocks.append({"subject": subject, "topic": t, "planned_minutes": mins})

        if not blocks:
            return jsonify({"error": "Gemini returned no valid subtopics."}), 502

        return jsonify({"blocks": blocks})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/ai/summary", methods=["POST"])
@login_required
def ai_summary():
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "Science").strip()
    topic = (data.get("topic") or "").strip()
    grade_level = get_current_grade()

    if not topic:
        return jsonify({"error": "Topic is required for a summary."}), 400

    try:
        client = get_gemini_client()
        prompt = (
            f"Student target / grade: {grade_level}. "
            f"Provide a clear, plain-language study overview for a student preparing at the "
            f"'{grade_level}' level who is about to study '{topic}' under {subject}. Include:\n"
            f"1. Core Intuition (2-3 short sentences)\n"
            f"2. Key Formulas / Rules / Definitions (bullet points)\n"
            f"3. One Common Exam Trap to Watch For"
        )
        response = generate_with_retry(
            client,
            prompt,
            config=types.GenerateContentConfig(),
        )
        return jsonify({"summary": (response.text or "").strip()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/ai/quiz", methods=["POST"])
@login_required
def ai_quiz():
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "Math").strip()
    topic = (data.get("topic") or "").strip()
    grade_level = get_current_grade()

    if not topic:
        return jsonify({"error": "Topic is required to generate a practice quiz."}), 400

    try:
        client = get_gemini_client()
        prompt = (
            f"Student target / grade: {grade_level}. "
            f"Create 4 concise multiple-choice practice questions on '{topic}' ({subject}) "
            f"calibrated for '{grade_level}'. Return ONLY a JSON array of objects, each with:\n"
            f"- 'question': string\n"
            f"- 'options': array of 4 short strings\n"
            f"- 'answer_index': integer from 0 to 3\n"
            f"- 'explanation': 1-2 sentence explanation"
        )
        response = generate_with_retry(
            client,
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        questions = json.loads((response.text or "[]").strip())
        if not isinstance(questions, list) or not questions:
            return jsonify({"error": "Could not generate quiz questions."}), 502
        return jsonify({"questions": questions})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/ai/weak-topics", methods=["POST"])
@login_required
def ai_weak_topics():
    user_id = session["user_id"]
    grade_level = get_current_grade()
    data = request.get_json(silent=True) or {}
    manual_weak = (data.get("weak_topics") or "").strip()

    db = get_db()
    recent_mocks = db.execute(
        """
        SELECT test_date, subject, score, max_score, weak_topics, mistake_notes
        FROM mock_tests
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 8
        """,
        (user_id,),
    ).fetchall()

    mock_context = [dict(r) for r in recent_mocks]
    if not manual_weak and not mock_context:
        return jsonify(
            {"error": "Log a mock test with weak topics or enter weak topics to build a plan."}
        ), 400

    try:
        client = get_gemini_client()
        prompt = (
            f"You are an academic coach helping a student studying for '{grade_level}'. "
            f"Based on the missed/weak topics and recent mock tests below, propose a focused "
            f"revision order from highest priority to lowest, with suggested study minutes and "
            f"a concrete action per topic.\n\n"
            f"Additional weak topics entered: {manual_weak or 'None'}\n"
            f"Recent mock tests: {json.dumps(mock_context)}"
        )
        response = generate_with_retry(
            client,
            prompt,
            config=types.GenerateContentConfig(),
        )
        return jsonify({"plan": (response.text or "").strip()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
