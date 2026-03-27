"""
models/clustering_model.py
---------------------------
Unsupervised: clusters children into learning behaviour types.
Uses K-Means (k=4) on session-level aggregate features.

Cluster interpretation (post-hoc labelling after fitting):
  0 → Fast learner      (high accuracy, fast response)
  1 → Consistent        (steady accuracy, normal pace)
  2 → Distracted        (high RT variance, inconsistent)
  3 → Mixed             (module-dependent performance)

Ages 4–7 note: clusters are used to tune CONTENT STYLE,
not to penalise children. Cluster 2 (distracted) gets
shorter sessions and more visual content, not harder work.
"""

from __future__ import annotations
import os
import pickle
import numpy as np
from typing import Optional

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


K = 4   # Number of clusters. Revisit with silhouette analysis after real data.

# Minimum sessions before clustering is meaningful
MIN_SESSIONS_FOR_CLUSTER = 3


# ── Cluster feature names ─────────────────────────────────────
CLUSTER_FEATURES = [
    "mean_accuracy",            # rolling accuracy across all sessions
    "accuracy_std",             # consistency — low std = consistent learner
    "mean_response_time_norm",  # normalised avg RT
    "rt_std_norm",              # RT variance — high = distracted signal
    "sessions_completed",       # engagement proxy
    "mean_attempts_per_q",      # how often they retry
]

# Human-readable labels assigned after inspecting cluster centroids
CLUSTER_LABELS = {
    0: "fast_learner",
    1: "consistent",
    2: "distracted",
    3: "mixed",
}

# Content strategy per cluster
CLUSTER_STRATEGY = {
    0: {
        "description":      "Fast learner",
        "question_pacing":  "fast",        # fewer seconds between questions
        "repetition_style": "minimal",     # move on quickly after 1 correct
        "visual_weight":    "low",         # text-forward content is fine
        "session_length":   "standard",
    },
    1: {
        "description":      "Consistent learner",
        "question_pacing":  "standard",
        "repetition_style": "moderate",    # 2 correct before advancing
        "visual_weight":    "medium",
        "session_length":   "standard",
    },
    2: {
        "description":      "Distracted learner",
        "question_pacing":  "slow",        # more pause/animation between questions
        "repetition_style": "high",        # 3 correct before advancing
        "visual_weight":    "high",        # heavy visuals, short text
        "session_length":   "short",       # cap at 5 minutes
    },
    3: {
        "description":      "Mixed learner",
        "question_pacing":  "standard",
        "repetition_style": "adaptive",    # adjust per-module
        "visual_weight":    "high",
        "session_length":   "standard",
    },
}


class BehaviourClusterer:

    def __init__(self, k: int = K):
        self.k = k
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        self.centroid_labels: dict[int, int] = {}   # raw cluster → semantic label

    # ── Build ────────────────────────────────────────────────

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler",  StandardScaler()),
            ("kmeans",  KMeans(n_clusters=self.k, n_init=15, random_state=42)),
        ])

    # ── Fit ──────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> dict:
        """
        Fit on session-level feature matrix.
        X shape: (n_students, 6) — one row per student using their aggregate stats.
        Returns inertia and auto-assigned label mapping.
        """
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X)
        self.is_fitted = True

        kmeans = self.pipeline.named_steps["kmeans"]
        scaler = self.pipeline.named_steps["scaler"]

        # Auto-label centroids by sorting on mean_accuracy + mean_rt
        centroids_scaled = kmeans.cluster_centers_
        centroids_raw    = scaler.inverse_transform(centroids_scaled)

        # Sort clusters: accuracy desc, rt asc to identify fast/slow learners
        order = np.argsort(
            -centroids_raw[:, 0] + centroids_raw[:, 2]
        )
        # Assign semantic labels in order
        semantic = [0, 1, 3, 2]   # fast, consistent, mixed, distracted
        self.centroid_labels = {int(order[i]): semantic[i] for i in range(self.k)}

        return {
            "inertia":  round(kmeans.inertia_, 2),
            "k":        self.k,
            "labels":   self.centroid_labels,
        }

    # ── Predict ──────────────────────────────────────────────

    def predict_cluster(self, x: np.ndarray) -> tuple[int, dict]:
        """
        Predict semantic cluster label for a single student feature vector.
        x shape: (1, 6)
        Returns (semantic_cluster_id, strategy_dict)
        """
        if not self.is_fitted:
            return 3, CLUSTER_STRATEGY[3]   # default to mixed during warmup

        raw_cluster = int(self.pipeline.predict(x)[0])
        semantic    = self.centroid_labels.get(raw_cluster, 3)
        return semantic, CLUSTER_STRATEGY[semantic]

    def has_enough_data(self, sessions_completed: int) -> bool:
        return sessions_completed >= MIN_SESSIONS_FOR_CLUSTER

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline":        self.pipeline,
                "centroid_labels": self.centroid_labels,
                "k":               self.k,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BehaviourClusterer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(k=data["k"])
        obj.pipeline        = data["pipeline"]
        obj.centroid_labels = data["centroid_labels"]
        obj.is_fitted       = True
        return obj


# ── Helper: build cluster feature vector from DB aggregates ──

def build_cluster_features(
    mean_accuracy:         float,
    accuracy_std:          float,
    mean_response_time:    float,   # already normalised [0,1]
    rt_std:                float,
    sessions_completed:    int,
    mean_attempts_per_q:   float,
) -> np.ndarray:
    return np.array([[
        mean_accuracy,
        accuracy_std,
        mean_response_time,
        rt_std,
        float(min(sessions_completed, 100)) / 100,
        float(min(mean_attempts_per_q, 5)) / 5,
    ]])