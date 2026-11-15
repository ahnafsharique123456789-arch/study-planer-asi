import os
import sqlite3
from functools import wraps

import psycopg
from psycopg.rows import dict_row
from flask import Flask, flash, redirect, render_template, request, session, url_for
from google import genai

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-prod")

DB_URL = os.environ.get("DATABASE_URL", "").strip()
USE_POSTGRES = bool(DB_URL)


def get_connection():
    if USE_POSTGRES:
        normalized_url = DB_URL
        if normalized_url.startswith("postgres://"):
            normalized_url = normalized_url.replace("postgres://", "postgresql://", 1)
        return psycopg.connect(normalized_url, row_factory=dict_row)

    conn = sqlite3.connect("study_planner.db")
    conn.row_factory = sqlite3.Row
    return conn


def adapt_sql(sql: str) -> str:
    if USE_POSTGRES:
        return sql.replace("?", "%s")
    return sql


def execute_query(sql: str, params: tuple = (), fetch: str = "none"):
    query = adapt_sql(sql)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            result = None
            if fetch == "one":
                row = cur.fetchone()
                result = dict(row) if row is not None else None
            elif fetch == "all":
                rows = cur.fetchall()
                result = [dict(r) for r in rows]
        conn.commit()
    return result


def init_db():
    id_col = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    text_type = "TEXT"
    int_type = "INTEGER"
    real_type = "REAL"

    execute_query(
        f"""
        CREATE TABLE IF NOT EXISTS users (
            id {id_col},
            username {text_type} UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    execute_query(
        f"""
        CREATE TABLE IF NOT EXISTS study_blocks (
            id {id_col},
            user_id {int_type} NOT NULL,
            subject {text_type} NOT NULL,
            topic {text_type} NOT NULL,
            duration_mins {int_type} NOT NULL DEFAULT 30,
            completed {int_type} NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    execute_query(
        f"""
        CREATE TABLE IF NOT EXISTS focus_sessions (
            id {id_col},
            user_id {int_type} NOT NULL,
            label {text_type} NOT NULL,
            minutes {int_type} NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    execute_query(
        f"""
        CREATE TABLE IF NOT EXISTS mock_scores (
            id {id_col},
            user_id {int_type} NOT NULL,
            test_name {text_type} NOT NULL,
            score {real_type} NOT NULL,
            max_score {real_type} NOT NULL DEFAULT 100,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


init_db()


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return execute_query("SELECT id, username FROM users WHERE id = ?", (user_id,), fetch="one")


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    user = get_current_user()
    if not user:
        session.clear()
        return redirect(url_for("login"))

    if request.method == "POST":
        subject = (request.form.get("subject") or "").strip()
        topic = (request.form.get("topic") or "").strip()
        try:
            duration_mins = max(1, int(request.form.get("duration_mins") or 30))
        except ValueError:
            duration_mins = 30

        if subject and topic:
            execute_query(
                "INSERT INTO study_blocks (user_id, subject, topic, duration_mins) VALUES (?, ?, ?, ?)",
                (user["id"], subject, topic, duration_mins),
            )
            flash("Study block saved.", "success")
        else:
            flash("Please enter both a subject and a topic.", "error")
        return redirect(url_for("index"))

    blocks = execute_query(
        "SELECT id, subject, topic, duration_mins, completed, created_at FROM study_blocks WHERE user_id = ? ORDER BY id DESC",
        (user["id"],),
        fetch="all",
    )
    mocks = execute_query(
        "SELECT id, test_name, score, max_score, created_at FROM mock_scores WHERE user_id = ? ORDER BY id DESC",
        (user["id"],),
        fetch="all",
    )
    return render_template("index.html", user=user, blocks=blocks, mocks=mocks)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        api_key = (request.form.get("gemini_api_key") or request.form.get("api_key") or "").strip()

        if not username:
            flash("Please enter your name to continue.", "error")
            return render_template("login.html")

        existing = execute_query(
            "SELECT id, username FROM users WHERE LOWER(username) = LOWER(?)",
            (username,),
            fetch="one",
        )
        if existing:
            user = existing
        else:
            execute_query("INSERT INTO users (username) VALUES (?)", (username,))
            user = execute_query(
                "SELECT id, username FROM users WHERE LOWER(username) = LOWER(?)",
                (username,),
                fetch="one",
            )

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        if api_key:
            session["gemini_api_key"] = api_key

        return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/mocks", methods=["GET", "POST"])
@login_required
def mocks():
    user = get_current_user()
    if request.method == "POST":
        test_name = (request.form.get("test_name") or request.form.get("title") or "Mock Test").strip()
        raw_score = request.form.get("score")
        raw_max = request.form.get("max_score") or "100"

        try:
            score = float(raw_score)
            max_score = float(raw_max)
            if max_score <= 0:
                max_score = 100.0
            execute_query(
                "INSERT INTO mock_scores (user_id, test_name, score, max_score) VALUES (?, ?, ?, ?)",
                (user["id"], test_name, score, max_score),
            )
            flash("Mock score saved.", "success")
        except (TypeError, ValueError):
            flash("Enter a valid numeric score.", "error")

        return redirect(url_for("mocks"))

    scores = execute_query(
        "SELECT id, test_name, score, max_score, created_at FROM mock_scores WHERE user_id = ? ORDER BY id DESC",
        (user["id"],),
        fetch="all",
    )
    return render_template("mocks.html", user=user, scores=scores)


@app.route("/focus", methods=["GET", "POST"])
@login_required
def focus():
    user = get_current_user()
    if request.method == "POST":
        label = (request.form.get("label") or "Focus Session").strip()
        try:
            minutes = max(1, int(request.form.get("minutes") or 25))
            execute_query(
                "INSERT INTO focus_sessions (user_id, label, minutes) VALUES (?, ?, ?)",
                (user["id"], label, minutes),
            )
            flash("Focus session logged.", "success")
        except ValueError:
            flash("Enter a valid number of minutes.", "error")
        return redirect(url_for("focus"))

    sessions = execute_query(
        "SELECT id, label, minutes, created_at FROM focus_sessions WHERE user_id = ? ORDER BY id DESC",
        (user["id"],),
        fetch="all",
    )
    return render_template("focus.html", user=user, sessions=sessions)


@app.route("/ai", methods=["GET", "POST"])
@login_required
def ai():
    user = get_current_user()
    reply = None
    prompt = ""

    if request.method == "POST":
        prompt = (request.form.get("prompt") or "").strip()
        api_key = (
            request.form.get("gemini_api_key")
            or session.get("gemini_api_key")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        ).strip()

        if not prompt:
            flash("Please enter a study question or prompt.", "error")
        elif not api_key:
            flash("Provide a Gemini API key on login or set GEMINI_API_KEY in environment variables.", "error")
        else:
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                reply = response.text
            except Exception as exc:
                flash(f"AI request failed: {exc}", "error")

    return render_template("ai.html", user=user, prompt=prompt, reply=reply)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
