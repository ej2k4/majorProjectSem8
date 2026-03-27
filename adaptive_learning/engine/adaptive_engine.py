"""
engine/adaptive_engine.py
--------------------------
Orchestrates all ML models into a single decision point.

Decision priority:
  1. Cold-start warmup  → rule-based (first 5 events)
  2. RL agent           → if trained (Phase 4)
  3. Supervised model   → if trained (Phase 2–3)
  4. Rule-based fallback → always available (Phase 1)

Each layer can be toggled on/off via feature flags,
letting you phase in ML gradually without breaking the game.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from data.models import (
    StudentFeatures, PredictionOutput, Difficulty,
    RLState, RLAction, AccuracyBucket
)
from models.performance_model import PerformanceModel, ModelRegistry
from models.clustering_model  import BehaviourClusterer, CLUSTER_STRATEGY
from models.rl_agent          import QLearningAgent


@dataclass
class EngineDecision:
    recommended_difficulty: Difficulty
    probability_correct:    float
    cluster_label:          Optional[int]
    cluster_strategy:       dict
    decision_source:        str           # "warmup" | "rule" | "supervised" | "rl"
    next_question_hints:    dict          # passed to content selector


class AdaptiveEngine:
    """
    Single entry point for the game to ask:
    "Given this student's state, what difficulty should I set next?"
    """

    def __init__(
        self,
        registry:   ModelRegistry,
        clusterer:  BehaviourClusterer,
        rl_agent:   QLearningAgent,
        flags: Optional[dict] = None,
    ):
        self.registry  = registry
        self.clusterer = clusterer
        self.rl_agent  = rl_agent

        # Feature flags — toggle layers on/off per environment
        self.flags = flags or {
            "use_supervised": True,
            "use_rl":         False,   # flip to True in Phase 4
            "use_clustering": True,
        }

    # ── Main entry point ─────────────────────────────────────

    def decide(
        self,
        features:         StudentFeatures,
        current_difficulty: Difficulty,
        total_events:     int,
    ) -> EngineDecision:

        # ── Step 1: Cold-start warmup ─────────────────────
        if total_events < 5:
            return self._warmup_decision(features)

        # ── Step 2: Get cluster strategy ──────────────────
        cluster_id, cluster_strategy = self._get_cluster(features)

        # ── Step 3: Get supervised prediction ─────────────
        prediction = self._get_supervised_prediction(features)

        # ── Step 4: RL override (if enabled) ──────────────
        if self.flags["use_rl"]:
            decision = self._rl_decision(features, current_difficulty, cluster_id)
            rec_diff = decision
            source   = "rl"
        else:
            rec_diff = prediction.recommended_difficulty
            source   = "supervised" if self.flags["use_supervised"] else "rule"

        # ── Step 5: Safety clamp (never jump >1 level) ────
        rec_diff = self._clamp_difficulty(current_difficulty, rec_diff)

        return EngineDecision(
            recommended_difficulty = rec_diff,
            probability_correct    = prediction.probability_correct,
            cluster_label          = cluster_id,
            cluster_strategy       = cluster_strategy,
            decision_source        = source,
            next_question_hints    = self._build_hints(cluster_strategy, rec_diff),
        )

    # ── Layer implementations ─────────────────────────────────

    def _warmup_decision(self, features: StudentFeatures) -> EngineDecision:
        return EngineDecision(
            recommended_difficulty = Difficulty.EASY,
            probability_correct    = 0.5,
            cluster_label          = None,
            cluster_strategy       = CLUSTER_STRATEGY[3],
            decision_source        = "warmup",
            next_question_hints    = {"visual_weight": "high", "pacing": "slow"},
        )

    def _get_cluster(self, features: StudentFeatures) -> tuple[int, dict]:
        if not self.flags["use_clustering"] or not self.clusterer.is_fitted:
            return 3, CLUSTER_STRATEGY[3]
        if features.cluster_label is not None:
            return features.cluster_label, CLUSTER_STRATEGY.get(features.cluster_label, CLUSTER_STRATEGY[3])
        return 3, CLUSTER_STRATEGY[3]

    def _get_supervised_prediction(self, features: StudentFeatures) -> PredictionOutput:
        if not self.flags["use_supervised"]:
            return self._rule_based_prediction(features)
        model = self.registry.get(features.module.value)
        return model.predict(features)

    def _rule_based_prediction(self, features: StudentFeatures) -> PredictionOutput:
        """
        Baseline rule: accuracy > 0.78 → promote, < 0.42 → demote.
        This is the Phase 1 fallback.
        """
        acc = features.past_accuracy_module
        d   = features.difficulty

        if acc > 0.78 and d.value < 3:
            rec = Difficulty(d.value + 1)
        elif acc < 0.42 and d.value > 1:
            rec = Difficulty(d.value - 1)
        else:
            rec = d

        return PredictionOutput(
            probability_correct    = acc,
            recommended_difficulty = rec,
            confidence             = 0.0,
            is_warmup              = False,
        )

    def _rl_decision(
        self,
        features:           StudentFeatures,
        current_difficulty: Difficulty,
        cluster_id:         int,
    ) -> Difficulty:
        state  = RLState(
            accuracy_bucket    = features.accuracy_bucket(),
            current_difficulty = current_difficulty,
            cluster_label      = cluster_id,
        )
        action = self.rl_agent.choose_action(state)
        return action.next_difficulty

    def _clamp_difficulty(self, current: Difficulty, recommended: Difficulty) -> Difficulty:
        """Hard safety: max ±1 level per decision."""
        delta = recommended.value - current.value
        if delta > 1:
            return Difficulty(current.value + 1)
        if delta < -1:
            return Difficulty(current.value - 1)
        return recommended

    def _build_hints(self, strategy: dict, difficulty: Difficulty) -> dict:
        return {
            "visual_weight":    strategy.get("visual_weight", "medium"),
            "pacing":           strategy.get("question_pacing", "standard"),
            "repetition_style": strategy.get("repetition_style", "moderate"),
            "difficulty":       difficulty.value,
        }

    # ── RL feedback loop ─────────────────────────────────────

    def record_outcome(
        self,
        prev_features:  StudentFeatures,
        new_features:   StudentFeatures,
        action_taken:   Difficulty,
        is_correct:     bool,
    ) -> None:
        """
        Called after a question is answered.
        Updates the Q-table and ends the episode if RL is active.
        """
        if not self.flags["use_rl"]:
            return

        cluster_id = prev_features.cluster_label or 3
        prev_state = RLState(
            accuracy_bucket    = prev_features.accuracy_bucket(),
            current_difficulty = prev_features.difficulty,
            cluster_label      = cluster_id,
        )
        next_state = RLState(
            accuracy_bucket    = new_features.accuracy_bucket(),
            current_difficulty = new_features.difficulty,
            cluster_label      = cluster_id,
        )
        reward = QLearningAgent.compute_reward(
            is_correct         = is_correct,
            response_time_norm = new_features.response_time_sec,
            prev_difficulty    = prev_features.difficulty,
            new_difficulty     = action_taken,
        )
        self.rl_agent.update(
            state      = prev_state,
            action     = RLAction(action_taken),
            reward     = reward,
            next_state = next_state,
        )