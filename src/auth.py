"""
src/auth.py
===========
SQLite-backed authentication: user registration/login, daily chat limits.

Schema:
  users       — email, bcrypt password hash, admin flag
  daily_usage — per-user per-day chat count, enforced at DAILY_LIMIT

Admin account (ADMIN_EMAIL) is auto-seeded from ADMIN_PASSWORD env var on
every init_db() call so it's always present after first launch.

Usage:
  Call init_db() once at app startup (module-level in app.py).
  All other functions are called per-request.
"""

import os
import sqlite3
import bcrypt
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DAILY_LIMIT = 15
ADMIN_EMAIL = "waseekirtefa@gmail.com"
DB_PATH = Path("data/auth.db")


def _get_conn() -> sqlite3.Connection:
    """Open (or create) the SQLite DB, ensuring the data/ directory exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Return dicts instead of tuples for readability
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create tables and seed the admin account.

    Safe to call on every app startup — CREATE IF NOT EXISTS + INSERT OR IGNORE
    make it idempotent. Admin password comes from ADMIN_PASSWORD env var; if not
    set, the admin account is skipped (user can register manually later).
    """
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin      INTEGER DEFAULT 0,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_usage (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                date       TEXT NOT NULL,
                chat_count INTEGER DEFAULT 0,
                UNIQUE(user_id, date),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()

        # Seed admin account if ADMIN_PASSWORD is configured
        admin_password = os.getenv("ADMIN_PASSWORD", "").strip()
        if admin_password:
            existing = conn.execute(
                "SELECT id FROM users WHERE email = ?", (ADMIN_EMAIL,)
            ).fetchone()

            if existing:
                # Always keep admin flag set, even if someone changed it via SQL
                conn.execute(
                    "UPDATE users SET is_admin = 1 WHERE email = ?", (ADMIN_EMAIL,)
                )
            else:
                pw_hash = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt()).decode()
                conn.execute(
                    "INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, 1)",
                    (ADMIN_EMAIL, pw_hash),
                )
            conn.commit()


def register_user(email: str, password: str) -> tuple:
    """
    Register a new user account.

    Returns (True, "") on success.
    Returns (False, error_message) on validation or duplicate failure.
    Admin email auto-gets is_admin=1 even when registered through the normal form.
    """
    email = email.strip().lower()

    if not email or "@" not in email or "." not in email.split("@")[-1]:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    is_admin = 1 if email == ADMIN_EMAIL else 0

    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, ?)",
                (email, pw_hash, is_admin),
            )
            conn.commit()
        return True, ""
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def login_user(email: str, password: str) -> tuple:
    """
    Verify credentials and return user info.

    Returns (True, user_dict) on success where user_dict = {id, email, is_admin}.
    Returns (False, {"error": message}) on failure.
    """
    email = email.strip().lower()

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, email, password_hash, is_admin FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if not row:
        return False, {"error": "No account found with that email."}

    if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return False, {"error": "Incorrect password."}

    return True, {
        "id": row["id"],
        "email": row["email"],
        "is_admin": bool(row["is_admin"]),
    }


def get_chat_count_today(user_id: int) -> int:
    """Return how many chats this user has sent today (resets at midnight)."""
    today = date.today().isoformat()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT chat_count FROM daily_usage WHERE user_id = ? AND date = ?",
            (user_id, today),
        ).fetchone()
    return row["chat_count"] if row else 0


def increment_chat_count(user_id: int) -> None:
    """Increment today's chat count for a user (upsert — safe to call repeatedly)."""
    today = date.today().isoformat()
    with _get_conn() as conn:
        # INSERT OR IGNORE creates the row on first chat; UPDATE increments it after
        conn.execute(
            "INSERT OR IGNORE INTO daily_usage (user_id, date, chat_count) VALUES (?, ?, 0)",
            (user_id, today),
        )
        conn.execute(
            "UPDATE daily_usage SET chat_count = chat_count + 1 WHERE user_id = ? AND date = ?",
            (user_id, today),
        )
        conn.commit()


def is_daily_limit_reached(user_id: int, is_admin: bool) -> bool:
    """Return True if the user has hit their daily chat limit. Admins are never limited."""
    if is_admin:
        return False
    return get_chat_count_today(user_id) >= DAILY_LIMIT


def login_or_create_google_user(email: str, name: str) -> dict:
    """
    Upsert a Google OAuth user. Returns the same dict shape as login_user():
    {id, email, is_admin}. OAuth accounts have an empty password_hash since
    they authenticate via Google — they can't use the email/password form.

    Admin auto-detection: if the email matches ADMIN_EMAIL, is_admin is set to 1.
    """
    email = email.strip().lower()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, email, is_admin FROM users WHERE email = ?", (email,)
        ).fetchone()
        if row:
            return {"id": row["id"], "email": row["email"], "is_admin": bool(row["is_admin"])}

        # New Google user — no password needed
        is_admin = 1 if email == ADMIN_EMAIL else 0
        conn.execute(
            "INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, ?)",
            (email, "", is_admin),
        )
        conn.commit()
        new_id = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()["id"]
        return {"id": new_id, "email": email, "is_admin": bool(is_admin)}
