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

# ---- Models ----
from sentence_prediction.model import Encoder, Decoder, Seq2Seq
from cartoonImage_model.generator import Generator
from text_model.model import TinyLSTM   # adjust folder if needed

# =====================================================
# -------------------- Setup ---------------------------
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Comfort Story + Cartoon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173", 
        "http://localhost:5003",
        "http://127.0.0.1:5003",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# -------------------- Cartoon Model -------------------
# =====================================================

NUM_CLASSES = 24
NOISE_DIM = 100

SCENARIO_TO_LABEL = {
    "dentist": 0,
    "doctor_visit": 1,
    "haircut": 2
}

G = Generator(NOISE_DIM, NUM_CLASSES).to(DEVICE)

try:
    GEN_PATH = os.path.join(BASE_DIR, "cartoonImage_model", "generator.pth")
    G.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
    G.eval()
    logger.info("Cartoon model loaded")
except Exception as e:
    logger.warning(f"Cartoon model not loaded: {e}")

# =====================================================
# ---------------- Sentence Model ----------------------
# =====================================================

VOCAB_PATH = os.path.join(BASE_DIR, "sentence_prediction", "vocab.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "sentence_prediction", "asd_model.pt")
LOG_PATH = os.path.join(BASE_DIR, "sentence_prediction", "interaction_log.json")

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

inv_vocab = {v: k for k, v in vocab.items()}

VOCAB_SIZE = len(vocab)
EMBED_SIZE = 128
HIDDEN_SIZE = 256
MAX_LEN = 15

encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)

sentence_model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
sentence_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
sentence_model.eval()

logger.info("Sentence model loaded")

# =====================================================
# ---------------- Story TinyLSTM Model ----------------
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
# ---------------- Request Models ----------------------
# =====================================================

class FullGenerateRequest(BaseModel):
    scenario: str
    name: str
    emotion: str

class SentenceRequest(BaseModel):
    text: str

# =====================================================
# ---------------- Helper Functions --------------------
# =====================================================

def tensor_to_base64(image_tensor):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    image_tensor = torch.clamp(image_tensor, 0, 1)
    image = transforms.ToPILImage()(image_tensor.cpu())

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ---------- Sentence Prediction ----------

def predict_sentence_model(text):
    tokens = text.lower().split()
    indices = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    indices = indices[:MAX_LEN]
    indices += [vocab["<pad>"]] * (MAX_LEN - len(indices))

    src = torch.tensor(indices).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hidden = sentence_model.encoder(src)

    input_token = torch.tensor([vocab["<sos>"]]).to(DEVICE)
    output_words = []

    for _ in range(MAX_LEN):
        with torch.no_grad():
            output, hidden = sentence_model.decoder(input_token, hidden)

        top1 = output.argmax(1).item()

        if top1 in [vocab["<eos>"], vocab["<pad>"]]:
            break

        output_words.append(inv_vocab.get(top1, ""))
        input_token = torch.tensor([top1]).to(DEVICE)

    return " ".join(output_words)

def append_to_log(inp, pred):
    entry = {
        "input": inp,
        "prediction": pred,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []
    else:
        data = []

    data.append(entry)

    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=4)

# ---------- Story Generation ----------

def generate_story(seed_text, max_words=80):
    words = seed_text.lower().split()
    state = None

    for _ in range(max_words):
        input_ids = torch.tensor(
            [[story_word2idx.get(w, story_word2idx.get("<unk>", 0)) for w in words[-10:]]]
        ).to(DEVICE)

        with torch.no_grad():
            output, state = story_model(input_ids, state)

        logits = output[0, -1]

        temperature = 0.8
        probs = torch.softmax(logits / temperature, dim=0)

        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs / torch.sum(top_probs)

        predicted_idx = top_indices[torch.multinomial(top_probs, 1)].item()
        next_word = story_idx2word[predicted_idx]

        if next_word == "<end>":
            break

        words.append(next_word)

    return " ".join(words)

def enrich_emotion(story, name, emotion):
    emotional_lines = {
        "excited": [
            f"{name.capitalize()} feels butterflies of excitement inside.",
            f"{name.capitalize()} cannot wait to see what happens next."
        ],
        "nervous": [
            f"{name.capitalize()}'s heart beats a little faster.",
            "It is okay to feel nervous sometimes."
        ],
        "sad": [
            f"{name.capitalize()}'s eyes feel a little heavy.",
            "It is okay to feel sad."
        ],
        "angry": [
            f"{name.capitalize()}'s hands feel tight for a moment.",
            "Taking slow deep breaths can help."
        ],
        "scared": [
            f"{name.capitalize()} feels a small shiver inside.",
            "It is safe, and grown-ups are there to help."
        ]
    }

    if emotion in emotional_lines:
        story = " ".join(emotional_lines[emotion]) + " " + story

    return story

# =====================================================
# -------------------- Routes --------------------------
# =====================================================

@app.get("/")
def home():
    return {"message": "API running successfully"}

@app.post("/generate-full")
def generate_full(request: FullGenerateRequest):
    try:
        # ---- Cartoon ----
        img_base64 = None
        if request.scenario in SCENARIO_TO_LABEL:
            label = torch.tensor([SCENARIO_TO_LABEL[request.scenario]]).to(DEVICE)
            noise = torch.randn(1, NOISE_DIM).to(DEVICE)

            with torch.no_grad():
                fake = G(noise, label)

            img_base64 = tensor_to_base64(fake)

        # ---- Story ----
        seed = f"<scenario_{request.scenario}> <emotion_{request.emotion}> <name> </start>"
        raw_story = generate_story(seed)

        raw_story = raw_story.replace("<name>", request.name.lower())
        raw_story = re.sub(r"<.*?>", "", raw_story)
        raw_story = re.sub(r"\s+", " ", raw_story).strip()

        sentences = [s.strip().capitalize() for s in raw_story.split(".") if s.strip()]
        story = ". ".join(sentences) + "."

        story = enrich_emotion(story, request.name, request.emotion)

        return {
            "story": story,
            "image": img_base64,
            "scenario": request.scenario.replace("_", " ")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-sentence")
def predict_sentence(request: SentenceRequest):
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty input")

        prediction = predict_sentence_model(text)
        append_to_log(text, prediction)

        return {
            "input": text,
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))