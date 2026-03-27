"""
api/main.py  (Phase 2 complete — DB wired)
"""
from __future__ import annotations
import os, uuid, json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.models              import Module, Difficulty, RawEvent
from data.feature_engineering import FeatureEngineer
from data.database            import (
    get_or_create_student, get_student_stats, get_student_cluster,
    get_random_question, create_session, close_session,
    log_event, log_reward, get_student_profile as db_get_profile,
)
from engine.adaptive_engine   import AdaptiveEngine
from engine.reward_engine     import RewardEngine
from models.performance_model import ModelRegistry
from models.clustering_model  import BehaviourClusterer
from models.rl_agent          import QLearningAgent

app = FastAPI(title="Adaptive Learning API", version="2.0.0")
@app.get("/")
def root():
    return {"message": "API is running 🚀"}
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts/models")

registry  = ModelRegistry(model_dir=ARTIFACT_DIR)
_clu_path = os.path.join(ARTIFACT_DIR, "clusterer.pkl")
clusterer = BehaviourClusterer.load(_clu_path) if os.path.exists(_clu_path) else BehaviourClusterer()
_rl_path  = os.path.join(ARTIFACT_DIR, "rl_agent.json")
rl_agent  = QLearningAgent.load(_rl_path) if os.path.exists(_rl_path) else QLearningAgent()
engine    = AdaptiveEngine(registry, clusterer, rl_agent)
fe        = FeatureEngineer()

_rewards:        dict[str, RewardEngine] = {}
_seen_questions: dict[str, list[str]]   = {}


class StartSessionRequest(BaseModel):
    student_id: str
    module:     str
    name:       str = ""
    age:        int = 5

class QuestionOut(BaseModel):
    question_id:    str
    module:         str
    topic:          str
    difficulty:     int
    question_text:  str
    correct_answer: str
    answer_options: dict
    visual_hint:    str

class StartSessionResponse(BaseModel):
    session_id: str
    difficulty: int
    question:   QuestionOut

class AnswerRequest(BaseModel):
    session_id:        str
    student_id:        str
    question_id:       str
    module:            str
    topic:             str
    difficulty:        int
    answer_given:      str
    is_correct:        bool
    response_time_sec: float
    attempt_number:    int = 1

class AnswerResponse(BaseModel):
    next_difficulty:     int
    probability_correct: float
    decision_source:     str
    cluster_label:       Optional[int]
    rewards:             list[dict]
    next_question:       Optional[QuestionOut]

class EndSessionRequest(BaseModel):
    session_id:       str
    student_id:       str
    total_correct:    int
    total_questions:  int
    avg_response_sec: float = 0.0


def _fmt(row: dict) -> QuestionOut:
    opts = row.get("answer_options", {})
    if isinstance(opts, str):
        opts = json.loads(opts)
    return QuestionOut(
        question_id   = str(row["question_id"]),
        module        = row["module"],
        topic         = row["topic"],
        difficulty    = int(row["difficulty"]),
        question_text = row["question_text"],
        correct_answer= row["correct_answer"],
        answer_options= opts,
        visual_hint   = row.get("visual_asset_url", ""),
    )


@app.post("/session/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    get_or_create_student(req.student_id, req.name, req.age)
    session_id = str(uuid.uuid4())
    create_session(session_id, req.student_id, req.module)
    _rewards[session_id]        = RewardEngine()
    _seen_questions[session_id] = []
    row = get_random_question(req.module, difficulty=1)
    if not row:
        raise HTTPException(404, f"No questions for module '{req.module}'")
    _seen_questions[session_id].append(str(row["question_id"]))
    return StartSessionResponse(session_id=session_id, difficulty=1, question=_fmt(row))


@app.post("/session/answer", response_model=AnswerResponse)
def submit_answer(req: AnswerRequest):
    stats         = get_student_stats(req.student_id, req.module, req.topic)
    cluster_label = get_student_cluster(req.student_id)
    raw = RawEvent(
        student_id=req.student_id, session_id=req.session_id,
        question_id=req.question_id, module=Module(req.module),
        topic=req.topic, difficulty=Difficulty(req.difficulty),
        response_time_sec=req.response_time_sec, answer_given=req.answer_given,
        is_correct=req.is_correct, attempt_number=req.attempt_number,
    )
    features = fe.build(
        event=raw,
        past_accuracy_topic  = stats["past_accuracy_topic"],
        past_accuracy_module = stats["past_accuracy_module"],
        attempts_on_topic    = stats["attempts_on_topic"],
        session_number       = stats["session_number"],
        time_since_last_sec  = stats["time_since_last_sec"],
        cluster_label        = cluster_label,
        total_events         = stats["total_events"],
    )
    decision = engine.decide(features=features,
                              current_difficulty=Difficulty(req.difficulty),
                              total_events=stats["total_events"])
    log_event(
        session_id=req.session_id, student_id=req.student_id,
        question_id=req.question_id, module=req.module, topic=req.topic,
        difficulty=req.difficulty, response_time_sec=req.response_time_sec,
        answer_given=req.answer_given, is_correct=req.is_correct,
        attempt_number=req.attempt_number, stats=stats,
        predicted_correct=decision.probability_correct,
        recommended_difficulty=decision.recommended_difficulty.value,
        cluster_at_time=cluster_label,
    )
    if req.session_id not in _rewards:
        _rewards[req.session_id] = RewardEngine()
    reward_events = _rewards[req.session_id].evaluate_answer(
        is_correct=req.is_correct, difficulty=req.difficulty,
        attempt_num=req.attempt_number,
        total_correct=stats["total_events"], module=req.module,
    )
    for r in reward_events:
        if r.reward_type == "xp":
            log_reward(req.student_id, req.session_id, r.reward_type, {"xp": r.value})

    seen   = _seen_questions.get(req.session_id, [])
    nd     = decision.recommended_difficulty.value
    next_q = get_random_question(req.module, nd, exclude_ids=seen) or \
             get_random_question(req.module, nd)
    if next_q:
        seen.append(str(next_q["question_id"]))
        _seen_questions[req.session_id] = seen[-20:]

    return AnswerResponse(
        next_difficulty     = nd,
        probability_correct = round(decision.probability_correct, 4),
        decision_source     = decision.decision_source,
        cluster_label       = decision.cluster_label,
        rewards=[{"type": str(r.reward_type), "value": r.value, "message": r.message}
                 for r in reward_events],
        next_question = _fmt(next_q) if next_q else None,
    )


@app.get("/question/next", response_model=QuestionOut)
def get_next_question(module: str, difficulty: int):
    row = get_random_question(module, difficulty)
    if not row:
        raise HTTPException(404, f"No questions for module='{module}' difficulty={difficulty}")
    return _fmt(row)


@app.post("/session/end")
def end_session(req: EndSessionRequest):
    close_session(req.session_id, req.total_questions, req.total_correct, req.avg_response_sec)
    summary = None
    if req.session_id in _rewards:
        summary = _rewards.pop(req.session_id).end_session()
        _seen_questions.pop(req.session_id, None)
    return {
        "session_id": req.session_id,
        "total_xp":   summary.total_xp if summary else 0,
        "badges":     summary.badges   if summary else [],
        "message":    "Great job! See you next time!",
    }


@app.get("/student/{student_id}/profile")
def get_student_profile_endpoint(student_id: str):
    profile = db_get_profile(student_id)
    if not profile:
        raise HTTPException(404, "Student not found")
    return profile


@app.post("/admin/train")
def retrain_models():
    from models.training_pipeline import train_supervised, train_clusterer
    return {"supervised": train_supervised(ARTIFACT_DIR),
            "clustering": train_clusterer(ARTIFACT_DIR)}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}