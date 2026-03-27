"""
engine/reward_engine.py
------------------------
Gamification layer: XP, badges, streaks.
Designed for ages 4–7 — rewards are frequent, encouraging,
and tied to effort as much as correctness.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RewardType(str, Enum):
    XP     = "xp"
    BADGE  = "badge"
    STREAK = "streak"


@dataclass
class RewardEvent:
    reward_type:  RewardType
    value:        int | str       # XP amount or badge/streak name
    message:      str             # child-friendly message to display


@dataclass
class SessionRewardSummary:
    total_xp:    int
    badges:      list[str]
    streak:      int
    events:      list[RewardEvent]


# ── XP table ──────────────────────────────────────────────────
XP_CORRECT_EASY   = 5
XP_CORRECT_MEDIUM = 10
XP_CORRECT_HARD   = 20
XP_TRY_BONUS      = 2    # just for attempting (effort reward — critical for this age)
XP_STREAK_BONUS   = 5    # bonus every 3-in-a-row correct

# ── Badge definitions ──────────────────────────────────────────
BADGES = {
    "first_star":       {"condition": "first_correct",     "label": "⭐ First Star!"},
    "math_explorer":    {"condition": "5_correct_math",    "label": "🔢 Math Explorer"},
    "science_friend":   {"condition": "5_correct_science", "label": "🔭 Science Friend"},
    "kind_kid":         {"condition": "5_correct_social",  "label": "💛 Kind Kid"},
    "super_streak":     {"condition": "streak_5",          "label": "🔥 Super Streak!"},
    "never_give_up":    {"condition": "3_retries",         "label": "💪 Never Give Up"},
    "session_complete": {"condition": "finish_session",    "label": "🏆 Session Champ!"},
}


class RewardEngine:

    def __init__(self):
        self._session_events: list[RewardEvent] = []
        self._session_xp    = 0
        self._streak        = 0
        self._badges_earned: list[str] = []

    def reset_session(self) -> None:
        self._session_events = []
        self._session_xp     = 0
        self._streak         = 0
        self._badges_earned  = []

    # ── Per-question reward ───────────────────────────────────

    def evaluate_answer(
        self,
        is_correct:    bool,
        difficulty:    int,         # 1, 2, or 3
        attempt_num:   int,         # 1 = first try
        total_correct: int,         # across this session so far
        module:        str,
    ) -> list[RewardEvent]:
        events: list[RewardEvent] = []

        # Always give XP for trying
        events.append(RewardEvent(
            reward_type = RewardType.XP,
            value       = XP_TRY_BONUS,
            message     = "Good try! 🌟",
        ))
        self._session_xp += XP_TRY_BONUS

        if is_correct:
            xp = {1: XP_CORRECT_EASY, 2: XP_CORRECT_MEDIUM, 3: XP_CORRECT_HARD}[difficulty]
            self._streak += 1
            self._session_xp += xp

            events.append(RewardEvent(
                reward_type = RewardType.XP,
                value       = xp,
                message     = self._correct_message(difficulty),
            ))

            # Streak bonus every 3 correct
            if self._streak > 0 and self._streak % 3 == 0:
                self._session_xp += XP_STREAK_BONUS
                events.append(RewardEvent(
                    reward_type = RewardType.STREAK,
                    value       = self._streak,
                    message     = f"🔥 {self._streak} in a row! Amazing!",
                ))

            # Badge triggers
            if total_correct == 1:
                events += self._award_badge("first_star")
            if total_correct == 5 and module == "math":
                events += self._award_badge("math_explorer")
            if total_correct == 5 and module == "science":
                events += self._award_badge("science_friend")
            if total_correct == 5 and module == "social":
                events += self._award_badge("kind_kid")
            if self._streak >= 5:
                events += self._award_badge("super_streak")

        else:
            self._streak = 0
            if attempt_num >= 3:
                events += self._award_badge("never_give_up")

        self._session_events.extend(events)
        return events

    def end_session(self) -> SessionRewardSummary:
        self._session_events += self._award_badge("session_complete")
        return SessionRewardSummary(
            total_xp = self._session_xp,
            badges   = self._badges_earned[:],
            streak   = self._streak,
            events   = self._session_events[:],
        )

    # ── Helpers ───────────────────────────────────────────────

    def _award_badge(self, badge_key: str) -> list[RewardEvent]:
        if badge_key in self._badges_earned:
            return []
        self._badges_earned.append(badge_key)
        return [RewardEvent(
            reward_type = RewardType.BADGE,
            value       = badge_key,
            message     = BADGES[badge_key]["label"],
        )]

    @staticmethod
    def _correct_message(difficulty: int) -> str:
        messages = {
            1: "✅ Great job!",
            2: "🌟 Brilliant!",
            3: "🚀 Superstar!",
        }
        return messages.get(difficulty, "✅ Correct!")