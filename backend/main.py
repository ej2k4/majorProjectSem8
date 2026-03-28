from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import logging
import pickle
import os
import json
import re
from datetime import datetime
import uuid

# Adaptive Learning imports
from adaptive_learning.data.models import Module, Difficulty, RawEvent
from adaptive_learning.data.feature_engineering import FeatureEngineer
from adaptive_learning.data.database import (
    get_or_create_student, get_student_stats, get_student_cluster,
    get_random_question, create_session, close_session
)
from adaptive_learning.engine.adaptive_engine import AdaptiveEngine
from adaptive_learning.engine.reward_engine import RewardEngine
from adaptive_learning.models.performance_model import ModelRegistry
from adaptive_learning.models.clustering_model import BehaviourClusterer
from adaptive_learning.models.rl_agent import QLearningAgent

# ---- Models ----
from sentence_prediction.model import Encoder, Decoder, Seq2Seq
from cartoonImage_model.generator import Generator
from text_model.model import TinyLSTM

# =====================================================
# Setup
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Comfort Story + Cartoon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# Cartoon Model
# =====================================================

NUM_CLASSES = 24
NOISE_DIM = 100
NUM_EMOTIONS = 4

SCENARIO_TO_LABEL = {
    "dentist": 0,
    "doctor_visit": 1,
    "haircut": 2
}

EMOTION_TO_LABEL = {
    "excited": 0,
    "nervous": 1,
    "sad": 2,
    "angry": 3
}

G = None
try:
    G = Generator(NOISE_DIM, NUM_CLASSES, NUM_EMOTIONS).to(DEVICE)
    GEN_PATH = os.path.join(BASE_DIR, "cartoonImage_model", "generator.pth")
    G.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
    G.eval()
    logger.info("Cartoon model loaded")
except Exception as e:
    logger.warning(f"Cartoon model not loaded: {e}")
    G = None

# =====================================================
# Sentence Model
# =====================================================

VOCAB_PATH = os.path.join(BASE_DIR, "sentence_prediction", "vocab.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "sentence_prediction", "asd_model.pt")
LOG_PATH = os.path.join(BASE_DIR, "sentence_prediction", "interaction_log.json")

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

inv_vocab = {v: k for k, v in vocab.items()}

encoder = Encoder(len(vocab), 128, 256)
decoder = Decoder(len(vocab), 128, 256)

sentence_model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
sentence_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
sentence_model.eval()

logger.info("Sentence model loaded")

# =====================================================
# Adaptive Learning
# =====================================================

ARTIFACT_DIR = os.path.join(BASE_DIR, "adaptive_learning", "artifacts", "models")

registry = ModelRegistry(model_dir=ARTIFACT_DIR)

clu_path = os.path.join(ARTIFACT_DIR, "clusterer.pkl")
clusterer = BehaviourClusterer.load(clu_path) if os.path.exists(clu_path) else BehaviourClusterer()

rl_path = os.path.join(ARTIFACT_DIR, "rl_agent.json")
rl_agent = QLearningAgent.load(rl_path) if os.path.exists(rl_path) else QLearningAgent()

adaptive_engine = AdaptiveEngine(registry, clusterer, rl_agent)
feature_engineer = FeatureEngineer()

session_rewards = {}
session_seen_questions = {}

# =====================================================
# Story Model
# =====================================================

STORY_VOCAB_PATH = os.path.join(BASE_DIR, "text_model", "vocab.json")
STORY_MODEL_PATH = os.path.join(BASE_DIR, "text_model", "tiny_lstm.pth")

with open(STORY_VOCAB_PATH, "r") as f:
    story_word2idx = json.load(f)

story_idx2word = {int(v): k for k, v in story_word2idx.items()}

checkpoint = torch.load(STORY_MODEL_PATH, map_location=DEVICE)

story_model = TinyLSTM(checkpoint["vocab_size"]).to(DEVICE)
story_model.load_state_dict(checkpoint["model_state_dict"])
story_model.eval()

logger.info("Story model loaded")

# =====================================================
# Schemas
# =====================================================

class FullGenerateRequest(BaseModel):
    scenario: str
    name: str
    emotion: str

class SentenceRequest(BaseModel):
    text: str

# =====================================================
# Helpers
# =====================================================

def tensor_to_base64(image_tensor):
    image_tensor = image_tensor.squeeze(0)
    image = transforms.ToPILImage()(torch.clamp(image_tensor, 0, 1).cpu())
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def predict_sentence_model(text):
    tokens = text.lower().split()
    indices = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    indices += [vocab["<pad>"]] * (15 - len(indices))

    src = torch.tensor(indices[:15]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hidden = sentence_model.encoder(src)

    token = torch.tensor([vocab["<sos>"]]).to(DEVICE)
    output_words = []

    for _ in range(15):
        output, hidden = sentence_model.decoder(token, hidden)
        top = output.argmax(1).item()
        if top in [vocab["<eos>"], vocab["<pad>"]]:
            break
        output_words.append(inv_vocab.get(top, ""))
        token = torch.tensor([top]).to(DEVICE)

    return " ".join(output_words)

def generate_story(seed_text):
    words = seed_text.lower().split()
    state = None

    for _ in range(80):
        input_ids = torch.tensor([[story_word2idx.get(w, 0) for w in words[-10:]]]).to(DEVICE)
        output, state = story_model(input_ids, state)

        probs = torch.softmax(output[0, -1] / 0.8, dim=0)
        top_probs, top_idx = torch.topk(probs, 10)
        next_word = story_idx2word[top_idx[torch.multinomial(top_probs, 1)].item()]

        if next_word == "<end>":
            break

        words.append(next_word)

    return " ".join(words)

# =====================================================
# Routes
# =====================================================

@app.get("/")
def home():
    return {"message": "API running successfully"}

@app.post("/generate-full")
def generate_full(request: FullGenerateRequest):
    img_base64 = None

    if G is not None and request.scenario in SCENARIO_TO_LABEL:
        scenario_label = torch.tensor([SCENARIO_TO_LABEL[request.scenario]]).to(DEVICE)
        emotion_label = torch.tensor([EMOTION_TO_LABEL.get(request.emotion, 0)]).to(DEVICE)
        noise = torch.randn(1, NOISE_DIM).to(DEVICE)

        with torch.no_grad():
            fake = G(noise, scenario_label, emotion_label)

        img_base64 = tensor_to_base64(fake)

    seed = f"<scenario_{request.scenario}> <emotion_{request.emotion}> <name> </start>"
    story = generate_story(seed)

    return {"story": story, "image": img_base64}

@app.post("/predict-sentence")
def predict_sentence(request: SentenceRequest):
    return {"prediction": predict_sentence_model(request.text)}

# =====================================================
# Adaptive Routes
# =====================================================

@app.post("/adaptive/start")
def adaptive_start(student_id: str, module: str):
    get_or_create_student(student_id, "", 5)

    session_id = str(uuid.uuid4())
    create_session(session_id, student_id, module)

    session_rewards[session_id] = RewardEngine()

    q = get_random_question(module, difficulty=1)

    if not q:
        raise HTTPException(404, "No questions")

    return {"session_id": session_id, "question": q}

@app.post("/adaptive/answer")
def adaptive_answer(data: dict):
    stats = get_student_stats(data["student_id"], data["module"], data["topic"])
    cluster = get_student_cluster(data["student_id"])

    raw = RawEvent(
        student_id=data["student_id"],
        session_id=data["session_id"],
        question_id=data["question_id"],
        module=Module(data["module"]),
        topic=data["topic"],
        difficulty=Difficulty(data["difficulty"]),
        response_time_sec=data["response_time_sec"],
        answer_given=data["answer_given"],
        is_correct=data["is_correct"]
    )

    features = feature_engineer.build(
        event=raw,
        past_accuracy_topic=stats["past_accuracy_topic"],
        past_accuracy_module=stats["past_accuracy_module"],
        attempts_on_topic=stats["attempts_on_topic"],
        session_number=stats["session_number"],
        time_since_last_sec=stats["time_since_last_sec"],
        cluster_label=cluster,
        total_events=stats["total_events"],
    )

    decision = adaptive_engine.decide(features, Difficulty(data["difficulty"]), stats["total_events"])

    next_q = get_random_question(data["module"], decision.recommended_difficulty.value)

    return {
        "next_difficulty": decision.recommended_difficulty.value,
        "next_question": next_q
    }

@app.post("/adaptive/end")
def adaptive_end(session_id: str):
    close_session(session_id, 0, 0, 0)
    return {"message": "Session ended"}