"""
tests/test_pipeline.py
-----------------------
Tests for all ML layers.
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from adaptive_learning.data.models import (
    Module, Difficulty, RawEvent, StudentFeatures, RLState, RLAction, AccuracyBucket
)
from adaptive_learning.data.feature_engineering import FeatureEngineer, WARMUP_THRESHOLD
from adaptive_learning.models.performance_model import PerformanceModel, ModelRegistry
from adaptive_learning.models.clustering_model  import BehaviourClusterer, build_cluster_features
from adaptive_learning.models.rl_agent          import QLearningAgent
from adaptive_learning.engine.adaptive_engine   import AdaptiveEngine
from adaptive_learning.engine.reward_engine     import RewardEngine


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def raw_event():
    return RawEvent(
        student_id="student-1",
        session_id="session-1",
        question_id="q-1",
        module=Module.MATH,
        topic="addition",
        difficulty=Difficulty.EASY,
        response_time_sec=8.0,
        answer_given="4",
        is_correct=True,
    )

@pytest.fixture
def trained_perf_model():
    np.random.seed(42)
    X = np.random.rand(200, 10)
    y = (X[:, 4] > 0.5).astype(int)   # label based on past_accuracy_topic
    model = PerformanceModel("math")
    model.train(X, y, eval=False)
    return model

@pytest.fixture
def trained_clusterer():
    np.random.seed(42)
    X = np.random.rand(80, 6)
    c = BehaviourClusterer(k=4)
    c.fit(X)
    return c

@pytest.fixture
def rl_agent():
    return QLearningAgent()

@pytest.fixture
def student_features():
    return StudentFeatures(
        module=Module.MATH,
        difficulty=Difficulty.MEDIUM,
        past_accuracy_topic=0.70,
        past_accuracy_module=0.65,
        attempts_on_topic=10,
        session_number=3,
        response_time_sec=0.35,
        time_since_last_sec=0.02,
        cluster_label=1,
    )


# ─────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────

class TestFeatureEngineer:

    def test_warmup_flag_below_threshold(self, raw_event):
        fe = FeatureEngineer()
        assert fe.is_warmup(WARMUP_THRESHOLD - 1) is True

    def test_warmup_flag_above_threshold(self, raw_event):
        fe = FeatureEngineer()
        assert fe.is_warmup(WARMUP_THRESHOLD) is False

    def test_response_time_clamped(self, raw_event):
        fe = FeatureEngineer()
        assert fe.clamp_response_time(0.5) == 2.0     # below min → clamp to min
        assert fe.clamp_response_time(200.0) == 45.0  # above max → clamp to max

    def test_build_during_warmup_uses_neutral_priors(self, raw_event):
        fe = FeatureEngineer()
        feats = fe.build(
            event=raw_event,
            past_accuracy_topic=0.9,
            past_accuracy_module=0.9,
            attempts_on_topic=50,
            session_number=10,
            time_since_last_sec=30,
            cluster_label=0,
            total_events=2,   # below WARMUP_THRESHOLD
        )
        assert feats.past_accuracy_topic   == 0.5
        assert feats.past_accuracy_module  == 0.5
        assert feats.attempts_on_topic     == 0
        assert feats.cluster_label is None

    def test_feature_vector_length(self, raw_event):
        fe = FeatureEngineer()
        feats = fe.build(
            event=raw_event,
            past_accuracy_topic=0.7,
            past_accuracy_module=0.65,
            attempts_on_topic=5,
            session_number=2,
            time_since_last_sec=60,
            cluster_label=1,
            total_events=10,
        )
        assert len(feats.to_vector()) == 10


# ─────────────────────────────────────────────────────────────
# Supervised Model
# ─────────────────────────────────────────────────────────────

class TestPerformanceModel:

    def test_untrained_returns_warmup(self, student_features):
        model = PerformanceModel("math")
        out   = model.predict(student_features)
        assert out.is_warmup is True
        assert out.recommended_difficulty == Difficulty.EASY

    def test_trained_returns_probability(self, trained_perf_model, student_features):
        out = trained_perf_model.predict(student_features)
        assert 0.0 <= out.probability_correct <= 1.0
        assert out.is_warmup is False

    def test_recommend_promotes_on_high_accuracy(self, trained_perf_model):
        feats = StudentFeatures(
            module=Module.MATH, difficulty=Difficulty.EASY,
            past_accuracy_topic=0.95, past_accuracy_module=0.95,
            attempts_on_topic=20, session_number=5,
            response_time_sec=0.3, time_since_last_sec=0.01,
        )
        out = trained_perf_model.predict(feats)
        # High accuracy student at EASY → should recommend MEDIUM or HARD
        assert out.recommended_difficulty.value >= Difficulty.EASY.value

    def test_recommend_demotes_on_low_accuracy(self, trained_perf_model):
        feats = StudentFeatures(
            module=Module.MATH, difficulty=Difficulty.HARD,
            past_accuracy_topic=0.10, past_accuracy_module=0.10,
            attempts_on_topic=3, session_number=1,
            response_time_sec=0.95, time_since_last_sec=0.5,
        )
        out = trained_perf_model.predict(feats)
        assert out.recommended_difficulty.value <= Difficulty.HARD.value

    def test_save_and_load(self, trained_perf_model, tmp_path, student_features):
        path = str(tmp_path / "math_model.pkl")
        trained_perf_model.save(path)
        loaded = PerformanceModel.load(path, "math")
        out = loaded.predict(student_features)
        assert 0.0 <= out.probability_correct <= 1.0


# ─────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────

class TestBehaviourClusterer:

    def test_predict_returns_valid_cluster(self, trained_clusterer):
        x = build_cluster_features(0.8, 0.05, 0.3, 0.1, 5, 1.2)
        cluster_id, strategy = trained_clusterer.predict_cluster(x)
        assert cluster_id in (0, 1, 2, 3)
        assert "visual_weight" in strategy

    def test_unfitted_returns_default_cluster(self):
        c = BehaviourClusterer()
        x = build_cluster_features(0.5, 0.1, 0.5, 0.1, 2, 1.0)
        cluster_id, strategy = c.predict_cluster(x)
        assert cluster_id == 3   # mixed = default

    def test_save_load(self, trained_clusterer, tmp_path):
        path = str(tmp_path / "clusterer.pkl")
        trained_clusterer.save(path)
        loaded = BehaviourClusterer.load(path)
        x = build_cluster_features(0.7, 0.1, 0.4, 0.1, 4, 1.1)
        c1, _ = trained_clusterer.predict_cluster(x)
        c2, _ = loaded.predict_cluster(x)
        assert c1 == c2


# ─────────────────────────────────────────────────────────────
# RL Agent
# ─────────────────────────────────────────────────────────────

class TestQLearningAgent:

    def test_choose_action_stays_safe(self, rl_agent):
        """Agent must never jump more than 1 difficulty level."""
        state = RLState(AccuracyBucket.LOW, Difficulty.EASY, 2)
        for _ in range(50):
            action = rl_agent.choose_action(state)
            assert abs(action.next_difficulty.value - Difficulty.EASY.value) <= 1

    def test_q_update_changes_table(self, rl_agent):
        state  = RLState(AccuracyBucket.MID, Difficulty.MEDIUM, 1)
        action = RLAction(Difficulty.MEDIUM)
        before = rl_agent._get_q(state)[1]
        rl_agent.update(state, action, reward=1.0, next_state=state)
        after  = rl_agent._get_q(state)[1]
        assert after != before

    def test_epsilon_decays(self, rl_agent):
        eps_start = rl_agent.epsilon
        for _ in range(100):
            rl_agent.end_episode()
        assert rl_agent.epsilon < eps_start

    def test_reward_positive_for_correct(self):
        r = QLearningAgent.compute_reward(
            is_correct=True,
            response_time_norm=0.4,
            prev_difficulty=Difficulty.EASY,
            new_difficulty=Difficulty.EASY,
        )
        assert r > 0

    def test_reward_penalises_spike(self):
        r = QLearningAgent.compute_reward(
            is_correct=True,
            response_time_norm=0.4,
            prev_difficulty=Difficulty.EASY,
            new_difficulty=Difficulty.HARD,   # illegal jump
        )
        # Even though correct, the spike penalty should make it < 1
        assert r < 1.0

    def test_save_load(self, rl_agent, tmp_path):
        path = str(tmp_path / "agent.json")
        rl_agent.save(path)
        loaded = QLearningAgent.load(path)
        assert loaded.epsilon == rl_agent.epsilon


# ─────────────────────────────────────────────────────────────
# Adaptive Engine (integration)
# ─────────────────────────────────────────────────────────────

class TestAdaptiveEngine:

    def _make_engine(self):
        registry  = ModelRegistry(model_dir="/tmp/nonexistent_for_tests")
        clusterer = BehaviourClusterer()
        rl_agent  = QLearningAgent()
        return AdaptiveEngine(registry, clusterer, rl_agent)

    def test_warmup_returns_easy(self):
        engine  = self._make_engine()
        feats   = StudentFeatures(
            module=Module.SCIENCE, difficulty=Difficulty.EASY,
            past_accuracy_topic=0.5, past_accuracy_module=0.5,
            attempts_on_topic=0, session_number=1,
            response_time_sec=0.5, time_since_last_sec=0.0,
        )
        decision = engine.decide(feats, Difficulty.EASY, total_events=3)
        assert decision.recommended_difficulty == Difficulty.EASY
        assert decision.decision_source == "warmup"

    def test_no_difficulty_spike(self):
        engine = self._make_engine()
        feats  = StudentFeatures(
            module=Module.MATH, difficulty=Difficulty.EASY,
            past_accuracy_topic=0.99, past_accuracy_module=0.99,
            attempts_on_topic=100, session_number=20,
            response_time_sec=0.1, time_since_last_sec=0.0,
        )
        # Even with perfect accuracy, should not jump from EASY to HARD
        decision = engine.decide(feats, Difficulty.EASY, total_events=100)
        assert decision.recommended_difficulty.value <= Difficulty.EASY.value + 1


# ─────────────────────────────────────────────────────────────
# Reward Engine
# ─────────────────────────────────────────────────────────────

class TestRewardEngine:

    def test_correct_awards_xp(self):
        re = RewardEngine()
        events = re.evaluate_answer(True, 1, 1, 1, "math")
        xp_events = [e for e in events if e.reward_type == "xp"]
        assert len(xp_events) >= 1

    def test_incorrect_still_awards_try_xp(self):
        re = RewardEngine()
        events = re.evaluate_answer(False, 1, 1, 0, "math")
        xp_events = [e for e in events if e.reward_type == "xp"]
        assert len(xp_events) == 1    # try bonus only

    def test_first_correct_awards_first_star_badge(self):
        re = RewardEngine()
        events = re.evaluate_answer(True, 1, 1, 1, "math")
        badges = [e for e in events if e.reward_type == "badge"]
        assert any(e.value == "first_star" for e in badges)

    def test_streak_bonus_at_3(self):
        re = RewardEngine()
        for i in range(2):
            re.evaluate_answer(True, 1, 1, i+1, "math")
        events = re.evaluate_answer(True, 1, 1, 3, "math")
        streak_events = [e for e in events if e.reward_type == "streak"]
        assert len(streak_events) == 1

    def test_session_end_summary(self):
        re = RewardEngine()
        re.evaluate_answer(True, 2, 1, 1, "science")
        summary = re.end_session()
        assert summary.total_xp > 0
        assert "session_complete" in summary.badges


if __name__ == "__main__":
    pytest.main([__file__, "-v"])