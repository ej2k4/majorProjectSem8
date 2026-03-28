"""
models/training_pipeline.py
----------------------------
Full training pipeline:
  1. Loads (or generates synthetic) training data
  2. Trains PerformanceModel per module
  3. Trains BehaviourClusterer
  4. Saves all artifacts

Run this script directly to bootstrap the system before
real student data is available:
    python -m models.training_pipeline --synthetic
"""

from __future__ import annotations
import argparse
import os
import random
import numpy as np
from datetime import datetime

from adaptive_learning.models.performance_model import PerformanceModel, ModelRegistry
from adaptive_learning.models.clustering_model  import BehaviourClusterer, build_cluster_features
from adaptive_learning.models.rl_agent          import QLearningAgent
from adaptive_learning.data.models              import Module, Difficulty


ARTIFACT_DIR = "artifacts"
RANDOM_SEED  = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# (Replace with real DB queries once you have student data)
# ─────────────────────────────────────────────────────────────

def generate_supervised_data(
    n_samples: int = 800,
    module: str = "math",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic (X, y) pairs for supervised model training.
    Encodes plausible correlations:
      - Higher accuracy_topic → more likely correct
      - Harder difficulty    → less likely correct
      - Very fast response   → likely random tap (penalised)
    """
    X, y = [], []

    for _ in range(n_samples):
        is_math    = int(module == "math")
        is_science = int(module == "science")
        is_social  = int(module == "social")

        difficulty          = random.randint(1, 3)
        past_acc_topic      = random.uniform(0.2, 0.95)
        past_acc_module     = past_acc_topic * random.uniform(0.85, 1.05)
        past_acc_module     = min(max(past_acc_module, 0.0), 1.0)
        attempts_on_topic   = random.randint(0, 30)
        session_number      = random.randint(1, 20)
        response_time_norm  = random.uniform(0.0, 1.0)
        time_since_last_norm= random.uniform(0.0, 1.0)

        # Synthetic probability of correct answer
        prob = (
            0.55 * past_acc_topic
            + 0.20 * past_acc_module
            - 0.15 * (difficulty - 1) / 2       # harder = harder
            + 0.08 * min(attempts_on_topic, 10) / 10
            - 0.08 * max(0.15 - response_time_norm, 0)  # too-fast penalty
        )
        prob = float(np.clip(prob, 0.05, 0.95))
        label = int(random.random() < prob)

        X.append([
            is_math, is_science, is_social,
            float(difficulty),
            past_acc_topic,
            past_acc_module,
            float(min(attempts_on_topic, 50)),
            float(min(session_number, 100)),
            response_time_norm,
            time_since_last_norm,
        ])
        y.append(label)

    return np.array(X), np.array(y)


def generate_cluster_data(n_students: int = 200) -> np.ndarray:
    """
    Generates synthetic student-level aggregate features
    representing 4 rough learner archetypes.
    """
    rows = []
    archetypes = [
        # (mean_acc, acc_std, mean_rt, rt_std, sessions, mean_attempts)
        (0.85, 0.05, 0.25, 0.08, 0.8, 0.2),   # fast learner
        (0.70, 0.08, 0.45, 0.10, 0.7, 0.3),   # consistent
        (0.55, 0.20, 0.60, 0.30, 0.4, 0.6),   # distracted
        (0.65, 0.18, 0.50, 0.20, 0.5, 0.5),   # mixed
    ]
    for i in range(n_students):
        arch = archetypes[i % 4]
        noise = lambda sd: np.random.normal(0, sd)
        row = [
            np.clip(arch[0] + noise(0.07), 0.1, 1.0),
            np.clip(arch[1] + noise(0.03), 0.01, 0.5),
            np.clip(arch[2] + noise(0.08), 0.0, 1.0),
            np.clip(arch[3] + noise(0.05), 0.0, 1.0),
            np.clip(arch[4] + noise(0.1),  0.0, 1.0),
            np.clip(arch[5] + noise(0.1),  0.0, 1.0),
        ]
        rows.append(row)
    return np.array(rows)


# ─────────────────────────────────────────────────────────────
# 2. TRAIN ALL MODELS
# ─────────────────────────────────────────────────────────────

def train_supervised(model_dir: str) -> list[dict]:
    print("\n── Training supervised models ──────────────────────────")
    registry = ModelRegistry(model_dir=model_dir)
    datasets = {}
    for module in ["math", "science", "social"]:
        X, y = generate_supervised_data(n_samples=800, module=module)
        datasets[module] = (X, y)
    results = registry.train_all(datasets)
    for r in results:
        print(f"  {r['module']:8s}  AUC={r.get('auc','N/A')}  acc={r.get('accuracy','N/A')}")
    return results


def train_clusterer(model_dir: str) -> dict:
    print("\n── Training behaviour clusterer ────────────────────────")
    X = generate_cluster_data(n_students=200)
    clusterer = BehaviourClusterer(k=4)
    result = clusterer.fit(X)
    path = os.path.join(model_dir, "clusterer.pkl")
    clusterer.save(path)
    print(f"  Inertia={result['inertia']}  labels={result['labels']}")
    return result


def init_rl_agent(model_dir: str) -> None:
    print("\n── Initialising RL agent (empty Q-table) ───────────────")
    agent = QLearningAgent()
    path  = os.path.join(model_dir, "rl_agent.json")
    agent.save(path)
    print(f"  Saved to {path}")


# ─────────────────────────────────────────────────────────────
# 3. ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no DB needed)")
    parser.add_argument("--model-dir", default=os.path.join(ARTIFACT_DIR, "models"))
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Training pipeline starting — {datetime.utcnow().isoformat()}")
    print(f"Artifact dir: {args.model_dir}")

    train_supervised(args.model_dir)
    train_clusterer(args.model_dir)
    init_rl_agent(args.model_dir)

    print(f"\n✓ All models saved to {args.model_dir}/")


if __name__ == "__main__":
    main()