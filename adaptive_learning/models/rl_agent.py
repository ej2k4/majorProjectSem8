"""
models/rl_agent.py
------------------
Q-Learning agent for adaptive teaching strategy.

State space:  accuracy_bucket(3) × difficulty(3) × cluster(4) = 36 states
Action space: {easy, medium, hard} = 3 actions

Ages 4–7 specifics:
- Reward is shaped to STRONGLY penalise difficulty spikes
  (going from easy to hard in one step upsets young children)
- Exploration rate (epsilon) decays slowly — we prefer safe known
  strategies over risky exploration with this age group
- Discount factor (gamma) is lower — children benefit from
  immediate reinforcement more than delayed rewards
"""

from __future__ import annotations
import json
import os
import random
from typing import Optional

from adaptive_learning.data.models import (
    RLState, RLAction, Difficulty, AccuracyBucket, StudentFeatures
)


# ── Hyper-parameters ─────────────────────────────────────────

ALPHA        = 0.1    # learning rate
GAMMA        = 0.6    # discount factor (lower than typical — prioritise immediate reward)
EPSILON_START = 0.3   # initial exploration rate
EPSILON_MIN  = 0.05   # minimum exploration (never fully greedy)
EPSILON_DECAY = 0.995 # per-episode decay

# Reward shaping constants
REWARD_CORRECT     = +1.0
REWARD_INCORRECT   = -0.5
REWARD_ENGAGEMENT  = +0.3    # bonus if response_time is in the "thinking" zone
PENALTY_SPIKE      = -0.8    # penalty for jumping difficulty by >1 level


class QLearningAgent:

    def __init__(self, q_table: Optional[dict[str, list[float]]] = None):
        # q_table maps state_key → [q_easy, q_mid, q_hard]
        self.q_table: dict[str, list[float]] = q_table or {}
        self.epsilon = EPSILON_START
        self.episode_count = 0

    # ── Q-table access ───────────────────────────────────────

    def _get_q(self, state: RLState) -> list[float]:
        key = state.key()
        if key not in self.q_table:
            self.q_table[key] = [0.0, 0.0, 0.0]
        return self.q_table[key]

    def _set_q(self, state: RLState, action_idx: int, value: float) -> None:
        q = self._get_q(state)
        q[action_idx] = value

    # ── Action selection ─────────────────────────────────────

    def choose_action(self, state: RLState) -> RLAction:
        """
        Epsilon-greedy action selection with safety constraint:
        never jump difficulty by more than 1 level.
        """
        q_values = self._get_q(state)
        safe_actions = self._safe_actions(state.current_difficulty)

        if random.random() < self.epsilon:
            # Explore — but only among safe actions
            return random.choice(safe_actions)

        # Exploit — pick best safe action
        best = max(safe_actions, key=lambda a: q_values[a.next_difficulty.value - 1])
        return best

    def _safe_actions(self, current: Difficulty) -> list[RLAction]:
        """
        For ages 4–7: only allow ±1 difficulty change per step.
        Never jump from easy directly to hard.
        """
        actions = []
        for d in Difficulty:
            if abs(d.value - current.value) <= 1:
                actions.append(RLAction(d))
        return actions

    # ── Learning update ──────────────────────────────────────

    def update(
        self,
        state:       RLState,
        action:      RLAction,
        reward:      float,
        next_state:  RLState,
    ) -> None:
        """
        Standard Q-update:
        Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        """
        a_idx    = action.next_difficulty.value - 1
        q_sa     = self._get_q(state)[a_idx]
        q_next   = max(self._get_q(next_state))

        new_q = q_sa + ALPHA * (reward + GAMMA * q_next - q_sa)
        self._set_q(state, a_idx, new_q)

    # ── Reward calculation ───────────────────────────────────

    @staticmethod
    def compute_reward(
        is_correct:         bool,
        response_time_norm: float,    # normalised [0,1]
        prev_difficulty:    Difficulty,
        new_difficulty:     Difficulty,
    ) -> float:
        """
        Reward signal for ages 4–7:
        - Correct answer → +1.0
        - Wrong answer   → -0.5
        - "Thinking time" response (0.2–0.7 norm) → +0.3 engagement bonus
        - Difficulty spike (>1 level jump) → -0.8 penalty
        """
        reward = REWARD_CORRECT if is_correct else REWARD_INCORRECT

        # Engagement bonus: response in the "thinking" window (not too fast, not too slow)
        if 0.2 <= response_time_norm <= 0.7:
            reward += REWARD_ENGAGEMENT

        # Penalise unsafe jumps (safety net on top of _safe_actions)
        if abs(new_difficulty.value - prev_difficulty.value) > 1:
            reward += PENALTY_SPIKE

        return reward

    # ── Epsilon decay ────────────────────────────────────────

    def end_episode(self) -> None:
        self.episode_count += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "q_table":       self.q_table,
            "epsilon":       self.epsilon,
            "episode_count": self.episode_count,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path) as f:
            data = json.load(f)
        agent = cls(q_table=data["q_table"])
        agent.epsilon       = data.get("epsilon", EPSILON_START)
        agent.episode_count = data.get("episode_count", 0)
        return agent