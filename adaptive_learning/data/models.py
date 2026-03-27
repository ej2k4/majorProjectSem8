"""
data/models.py
--------------
Pydantic models for events, features, and ML I/O.
Designed for ages 4–7: difficulty is capped at 3 levels,
response-time windows are wider (young children are slower).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import uuid
from datetime import datetime


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

class Module(str, Enum):
    MATH    = "math"
    SCIENCE = "science"
    SOCIAL  = "social"

class Difficulty(int, Enum):
    EASY   = 1
    MEDIUM = 2
    HARD   = 3

class ClusterLabel(int, Enum):
    FAST_LEARNER   = 0   # high accuracy, fast response
    CONSISTENT     = 1   # steady accuracy, normal pace
    DISTRACTED     = 2   # high variance response time, moderate accuracy
    MIXED          = 3   # inconsistent across sessions

class AccuracyBucket(str, Enum):
    LOW  = "low"    # < 0.40
    MID  = "mid"    # 0.40 – 0.75
    HIGH = "high"   # > 0.75


# ──────────────────────────────────────────────
# Raw event (what the game sends us)
# ──────────────────────────────────────────────

@dataclass
class RawEvent:
    student_id:        str
    session_id:        str
    question_id:       str
    module:            Module
    topic:             str
    difficulty:        Difficulty
    response_time_sec: float        # tap-to-answer latency
    answer_given:      str
    is_correct:        bool
    attempt_number:    int = 1
    created_at:        datetime = field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Feature vector (fed into supervised model + RL)
# ──────────────────────────────────────────────

@dataclass
class StudentFeatures:
    """
    All features used by the supervised model and RL state encoder.
    Derived from RawEvent + historical aggregates fetched from DB.
    """
    # ── Categorical (encoded before model.predict) ──
    module:                 Module
    difficulty:             Difficulty

    # ── Historical aggregates ──
    past_accuracy_topic:    float       # 0–1, accuracy on this specific topic
    past_accuracy_module:   float       # 0–1, accuracy across the whole module
    attempts_on_topic:      int         # total attempts on this topic ever
    session_number:         int         # how many sessions the student has had

    # ── Timing ──
    response_time_sec:      float
    time_since_last_sec:    float       # recency signal

    # ── Meta ──
    cluster_label:          Optional[int] = None   # None during cold-start warmup

    def to_vector(self) -> list[float]:
        """
        Returns ordered numeric list for sklearn models.
        Categoricals are ordinally encoded (fine for tree models;
        use OneHotEncoder for logistic regression — see pipeline.py).
        """
        return [
            self.module.value == "math",        # one-hot stub
            self.module.value == "science",
            self.module.value == "social",
            float(self.difficulty.value),
            self.past_accuracy_topic,
            self.past_accuracy_module,
            float(self.attempts_on_topic),
            float(self.session_number),
            self.response_time_sec,
            self.time_since_last_sec,
        ]

    def accuracy_bucket(self) -> AccuracyBucket:
        a = self.past_accuracy_module
        if a < 0.40:  return AccuracyBucket.LOW
        if a < 0.75:  return AccuracyBucket.MID
        return AccuracyBucket.HIGH


# ──────────────────────────────────────────────
# Model outputs
# ──────────────────────────────────────────────

@dataclass
class PredictionOutput:
    probability_correct:    float           # P(student answers correctly)
    recommended_difficulty: Difficulty      # what to show next
    confidence:             float           # model confidence in recommendation
    is_warmup:              bool = False    # True if cold-start rules applied


# ──────────────────────────────────────────────
# RL state/action
# ──────────────────────────────────────────────

@dataclass
class RLState:
    accuracy_bucket: AccuracyBucket
    current_difficulty: Difficulty
    cluster_label: int

    def key(self) -> str:
        return f"{self.accuracy_bucket.value}_{self.current_difficulty.value}_{self.cluster_label}"

@dataclass
class RLAction:
    next_difficulty: Difficulty

    @staticmethod
    def all_actions() -> list["RLAction"]:
        return [RLAction(d) for d in Difficulty]