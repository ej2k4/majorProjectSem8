"""
data/database.py
----------------
PostgreSQL connection pool + all query helpers used by the API.
Uses psycopg2 with a simple connection pool.

Set DATABASE_URL env var:
  postgresql://user:password@localhost:5432/adaptive_learning
"""

from __future__ import annotations
import os
import json
import uuid
import random
from typing import Optional
from datetime import datetime
from contextlib import contextmanager

#import psycopg2
#from psycopg2 import pool
#from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:root_pw@localhost:5432/adaptive_learning"
)

# ── Connection pool (created once at import time) ─────────────────────────────
_pool: Optional[pool.SimpleConnectionPool] = None

def get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(1, 10, DATABASE_URL)
    return _pool

@contextmanager
def get_conn():
    conn = get_pool().getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        get_pool().putconn(conn)

@contextmanager
def get_cursor():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur


# ── Question queries ──────────────────────────────────────────────────────────

def get_random_question(module: str, difficulty: int,
                        exclude_ids: list[str] | None = None) -> Optional[dict]:
    """
    Fetch a random question for a given module + difficulty.
    Excludes recently seen question IDs to avoid immediate repeats.
    """
    exclude_ids = exclude_ids or []
    with get_cursor() as cur:
        if exclude_ids:
            cur.execute("""
                SELECT question_id, module, topic, difficulty,
                       question_text, correct_answer, answer_options, visual_asset_url
                FROM questions
                WHERE module = %s AND difficulty = %s
                  AND question_id != ALL(%s)
                ORDER BY RANDOM()
                LIMIT 1
            """, (module, difficulty, exclude_ids))
        else:
            cur.execute("""
                SELECT question_id, module, topic, difficulty,
                       question_text, correct_answer, answer_options, visual_asset_url
                FROM questions
                WHERE module = %s AND difficulty = %s
                ORDER BY RANDOM()
                LIMIT 1
            """, (module, difficulty))
        row = cur.fetchone()
        return dict(row) if row else None


def get_question_by_id(question_id: str) -> Optional[dict]:
    with get_cursor() as cur:
        cur.execute("""
            SELECT question_id, module, topic, difficulty,
                   question_text, correct_answer, answer_options, visual_asset_url
            FROM questions WHERE question_id = %s
        """, (question_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# ── Student queries ───────────────────────────────────────────────────────────

def get_or_create_student(student_id: str, name: str = "", age: int = 5) -> dict:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM students WHERE student_id = %s", (student_id,))
        row = cur.fetchone()
        if row:
            return dict(row)
        cur.execute("""
            INSERT INTO students (student_id, name, age)
            VALUES (%s, %s, %s)
            RETURNING *
        """, (student_id, name, age))
        return dict(cur.fetchone())


def get_student_stats(student_id: str, module: str, topic: str) -> dict:
    """
    Returns aggregated accuracy stats needed by the feature engineer.
    These replace the dummy values that were hardcoded in the API before.
    """
    with get_cursor() as cur:
        # Overall event count
        cur.execute("""
            SELECT COUNT(*) as total_events
            FROM question_events
            WHERE student_id = %s
        """, (student_id,))
        total_events = cur.fetchone()["total_events"]

        # Accuracy on this specific topic
        cur.execute("""
            SELECT
                COUNT(*) as attempts,
                SUM(CASE WHEN is_correct THEN 1 ELSE 0 END)::float /
                    NULLIF(COUNT(*), 0) as accuracy
            FROM question_events
            WHERE student_id = %s AND topic = %s
        """, (student_id, topic))
        topic_row = cur.fetchone()

        # Accuracy across the whole module
        cur.execute("""
            SELECT
                SUM(CASE WHEN is_correct THEN 1 ELSE 0 END)::float /
                    NULLIF(COUNT(*), 0) as accuracy
            FROM question_events
            WHERE student_id = %s AND module = %s
        """, (student_id, module))
        module_row = cur.fetchone()

        # Session count
        cur.execute("""
            SELECT COUNT(DISTINCT session_id) as session_count
            FROM question_events WHERE student_id = %s
        """, (student_id,))
        session_count = cur.fetchone()["session_count"]

        # Time since last event
        cur.execute("""
            SELECT EXTRACT(EPOCH FROM (NOW() - MAX(created_at))) as seconds_ago
            FROM question_events WHERE student_id = %s
        """, (student_id,))
        time_row = cur.fetchone()

        return {
            "total_events":          int(total_events),
            "past_accuracy_topic":   float(topic_row["accuracy"] or 0.5),
            "attempts_on_topic":     int(topic_row["attempts"] or 0),
            "past_accuracy_module":  float(module_row["accuracy"] or 0.5),
            "session_number":        int(session_count or 1),
            "time_since_last_sec":   float(time_row["seconds_ago"] or 60.0),
        }


def get_student_cluster(student_id: str) -> Optional[int]:
    with get_cursor() as cur:
        cur.execute(
            "SELECT cluster_label FROM students WHERE student_id = %s",
            (student_id,)
        )
        row = cur.fetchone()
        return row["cluster_label"] if row else None


# ── Session queries ───────────────────────────────────────────────────────────

def create_session(session_id: str, student_id: str, module: str) -> None:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO sessions (session_id, student_id, module)
            VALUES (%s, %s, %s)
        """, (session_id, student_id, module))


def close_session(session_id: str, total_q: int, total_correct: int,
                  avg_rt: float) -> None:
    accuracy = total_correct / total_q if total_q > 0 else 0.0
    with get_cursor() as cur:
        cur.execute("""
            UPDATE sessions
            SET ended_at = NOW(),
                total_questions = %s,
                total_correct   = %s,
                session_accuracy = %s,
                avg_response_sec = %s
            WHERE session_id = %s
        """, (total_q, total_correct, accuracy, avg_rt, session_id))


# ── Event logging ─────────────────────────────────────────────────────────────

def log_event(session_id: str, student_id: str, question_id: str,
              module: str, topic: str, difficulty: int,
              response_time_sec: float, answer_given: str, is_correct: bool,
              attempt_number: int, stats: dict,
              predicted_correct: float, recommended_difficulty: int,
              cluster_at_time: Optional[int]) -> None:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO question_events (
                event_id, session_id, student_id, question_id,
                module, topic, difficulty,
                response_time_sec, answer_given, is_correct, attempt_number,
                past_accuracy_topic, past_accuracy_module, attempts_on_topic,
                session_number, time_since_last_sec,
                predicted_correct, recommended_difficulty, cluster_at_time
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s
            )
        """, (
            str(uuid.uuid4()), session_id, student_id, question_id,
            module, topic, difficulty,
            response_time_sec, answer_given, is_correct, attempt_number,
            stats["past_accuracy_topic"], stats["past_accuracy_module"],
            stats["attempts_on_topic"], stats["session_number"],
            stats["time_since_last_sec"],
            predicted_correct, recommended_difficulty, cluster_at_time
        ))


# ── Reward logging ────────────────────────────────────────────────────────────

def log_reward(student_id: str, session_id: str,
               reward_type: str, reward_value: dict) -> None:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO rewards (student_id, session_id, reward_type, reward_value)
            VALUES (%s, %s, %s, %s)
        """, (student_id, session_id, reward_type, json.dumps(reward_value)))


# ── Student profile for frontend ─────────────────────────────────────────────

def get_student_profile(student_id: str) -> dict:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM students WHERE student_id = %s", (student_id,))
        student = cur.fetchone()
        if not student:
            return {}

        cur.execute("""
            SELECT module,
                   COUNT(*) as total_attempts,
                   SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
                   ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END)::numeric, 3) as accuracy,
                   MAX(difficulty) as max_difficulty_reached
            FROM question_events
            WHERE student_id = %s
            GROUP BY module
        """, (student_id,))
        module_stats = {row["module"]: dict(row) for row in cur.fetchall()}

        cur.execute("""
            SELECT SUM(CASE WHEN reward_type = 'xp' THEN (reward_value->>'xp')::int ELSE 0 END) as total_xp
            FROM rewards WHERE student_id = %s
        """, (student_id,))
        xp_row = cur.fetchone()

        return {
            "student_id":    student_id,
            "name":          student["name"],
            "age":           student["age"],
            "cluster_label": student["cluster_label"],
            "warmup_complete": student["warmup_complete"],
            "total_xp":      int(xp_row["total_xp"] or 0),
            "module_stats":  module_stats,
        }