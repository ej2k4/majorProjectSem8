"""
data/feature_engineering.py
-----------------------------
Converts raw game events into StudentFeatures vectors.
Handles cold-start (< 5 events) by returning warmup defaults.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from data.models import (
    RawEvent, StudentFeatures, Module, Difficulty
)


# Minimum events before we trust historical aggregates
WARMUP_THRESHOLD = 5

# Expected response-time range for ages 4–7 (seconds)
# Under 2s → likely random tap; over 45s → likely distracted
RESPONSE_TIME_MIN = 2.0
RESPONSE_TIME_MAX = 45.0


class FeatureEngineer:
    """
    Stateless transformer: takes a RawEvent + DB aggregates
    and returns a clean StudentFeatures object.

    In production, DB aggregates are fetched by the API layer
    and passed in. This keeps the class easily testable.
    """

    def build(
        self,
        event:                  RawEvent,
        past_accuracy_topic:    float,
        past_accuracy_module:   float,
        attempts_on_topic:      int,
        session_number:         int,
        time_since_last_sec:    float,
        cluster_label:          Optional[int],
        total_events:           int,            # used for cold-start detection
    ) -> StudentFeatures:

        is_warmup = total_events < WARMUP_THRESHOLD

        # Clamp response time — outliers hurt the model badly for young learners
        rt = np.clip(event.response_time_sec, RESPONSE_TIME_MIN, RESPONSE_TIME_MAX)
        rt_normalised = (rt - RESPONSE_TIME_MIN) / (RESPONSE_TIME_MAX - RESPONSE_TIME_MIN)

        # During warmup, substitute neutral priors
        if is_warmup:
            past_accuracy_topic  = 0.5
            past_accuracy_module = 0.5
            attempts_on_topic    = 0
            cluster_label        = None

        return StudentFeatures(
            module=event.module,
            difficulty=event.difficulty,
            past_accuracy_topic=past_accuracy_topic,
            past_accuracy_module=past_accuracy_module,
            attempts_on_topic=min(attempts_on_topic, 50),   # cap to avoid scale issues
            session_number=min(session_number, 100),
            response_time_sec=rt_normalised,
            time_since_last_sec=min(time_since_last_sec, 3600) / 3600,  # normalise to [0,1]
            cluster_label=cluster_label,
        )

    def clamp_response_time(self, rt: float) -> float:
        return float(np.clip(rt, RESPONSE_TIME_MIN, RESPONSE_TIME_MAX))

    def is_warmup(self, total_events: int) -> bool:
        return total_events < WARMUP_THRESHOLD