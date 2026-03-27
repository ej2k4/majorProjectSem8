"""
models/performance_model.py
----------------------------
Supervised model: predicts P(correct) and recommended difficulty.

Phase 1: Logistic Regression (interpretable, trains on small data)
Phase 2: XGBoost (upgrade once you have 500+ labelled sessions)

Ages 4–7 note: difficulty range is 1–3 only.
Model is trained per-module to capture domain differences.
"""

from __future__ import annotations
import os
import pickle
import numpy as np
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Uncomment when upgrading to phase 2:
# from xgboost import XGBClassifier

from data.models import (
    StudentFeatures, PredictionOutput, Difficulty
)


FEATURE_NAMES = [
    "is_math", "is_science", "is_social",
    "difficulty",
    "past_accuracy_topic",
    "past_accuracy_module",
    "attempts_on_topic",
    "session_number",
    "response_time_norm",
    "time_since_last_norm",
]

# Difficulty thresholds for recommendation logic
PROMOTE_THRESHOLD  = 0.78   # if P(correct) > this AND current difficulty < 3 → go harder
DEMOTE_THRESHOLD   = 0.42   # if P(correct) < this AND current difficulty > 1 → go easier
# (Thresholds are gentler than typical because 4–7 yr olds need more reinforcement
#  before being challenged — standard 0.8/0.4 split is too aggressive at this age)


class PerformanceModel:
    """
    Wraps an sklearn Pipeline for P(correct) prediction.
    One instance per module (math / science / social).
    """

    def __init__(self, module: str, model_type: str = "logistic"):
        self.module     = module
        self.model_type = model_type
        self.pipeline:  Optional[Pipeline] = None
        self.is_trained = False

    # ── Build ────────────────────────────────────────────────

    def _build_pipeline(self) -> Pipeline:
        if self.model_type == "logistic":
            clf = LogisticRegression(
                C=1.0,
                max_iter=500,
                class_weight="balanced",   # handles class imbalance in early data
                random_state=42,
            )
        elif self.model_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,               # shallow trees prevent overfit on small data
                class_weight="balanced",
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    clf),
        ])

    # ── Training ─────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,      # shape (n_samples, 10)
        y: np.ndarray,      # shape (n_samples,)  binary: 1=correct, 0=incorrect
        eval: bool = True,
    ) -> dict:
        """
        Train and optionally evaluate on a hold-out split.
        Returns a metrics dict.
        """
        self.pipeline = self._build_pipeline()

        if eval and len(X) >= 50:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            self.pipeline.fit(X_train, y_train)
            probs = self.pipeline.predict_proba(X_val)[:, 1]
            metrics = {
                "module":   self.module,
                "n_train":  len(X_train),
                "n_val":    len(X_val),
                "auc":      round(roc_auc_score(y_val, probs), 4),
                "accuracy": round(accuracy_score(y_val, probs >= 0.5), 4),
            }
        else:
            # Not enough data for split — train on everything
            self.pipeline.fit(X, y)
            metrics = {"module": self.module, "n_train": len(X), "note": "no eval split"}

        self.is_trained = True
        return metrics

    # ── Prediction ───────────────────────────────────────────

    def predict(self, features: StudentFeatures) -> PredictionOutput:
        """
        Returns P(correct) and recommended next difficulty.
        Falls back to rule-based warmup if model not trained yet.
        """
        if not self.is_trained or self.pipeline is None:
            return self._warmup_prediction(features)

        vec = np.array(features.to_vector()).reshape(1, -1)
        prob_correct = float(self.pipeline.predict_proba(vec)[0, 1])
        rec_difficulty = self._recommend_difficulty(prob_correct, features.difficulty)

        return PredictionOutput(
            probability_correct=prob_correct,
            recommended_difficulty=rec_difficulty,
            confidence=abs(prob_correct - 0.5) * 2,    # 0 = uncertain, 1 = certain
        )

    def _recommend_difficulty(
        self,
        prob_correct: float,
        current: Difficulty,
    ) -> Difficulty:
        """
        Ages 4–7: be conservative. Only promote after sustained high performance.
        """
        if prob_correct > PROMOTE_THRESHOLD and current.value < 3:
            return Difficulty(current.value + 1)
        if prob_correct < DEMOTE_THRESHOLD and current.value > 1:
            return Difficulty(current.value - 1)
        return current

    def _warmup_prediction(self, features: StudentFeatures) -> PredictionOutput:
        """
        Cold-start rule: always start at difficulty 1 (EASY).
        P(correct) assumed 0.5 until data accumulates.
        """
        return PredictionOutput(
            probability_correct=0.5,
            recommended_difficulty=Difficulty.EASY,
            confidence=0.0,
            is_warmup=True,
        )

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "module": self.module}, f)

    @classmethod
    def load(cls, path: str, module: str) -> "PerformanceModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = cls(module)
        m.pipeline  = data["pipeline"]
        m.is_trained = True
        return m


# ── Model registry (one model per module) ────────────────────

class ModelRegistry:
    """
    Manages one PerformanceModel per module.
    In production, models are loaded from disk / MLflow.
    """

    def __init__(self, model_dir: str = "artifacts/models"):
        self.model_dir = model_dir
        self._models: dict[str, PerformanceModel] = {}

    def get(self, module: str) -> PerformanceModel:
        if module not in self._models:
            path = os.path.join(self.model_dir, f"{module}_model.pkl")
            if os.path.exists(path):
                self._models[module] = PerformanceModel.load(path, module)
            else:
                self._models[module] = PerformanceModel(module)   # untrained → warmup mode
        return self._models[module]

    def train_all(self, datasets: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
        results = []
        for module, (X, y) in datasets.items():
            model = PerformanceModel(module)
            metrics = model.train(X, y)
            model.save(os.path.join(self.model_dir, f"{module}_model.pkl"))
            self._models[module] = model
            results.append(metrics)
        return results