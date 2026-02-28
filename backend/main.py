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

from sentence_prediction.model import Encoder, Decoder, Seq2Seq
from cartoonImage_model.generator import Generator

# -------------------- Setup --------------------

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

# -------------------- Cartoon Model --------------------

NUM_CLASSES = 3
NOISE_DIM = 100

LABELS_MAP = {
    0: "dentist",
    1: "doctor",
    2: "haircut"
}

SCENARIO_TO_LABEL = {
    "dentist": 0,
    "doctor_visit": 1,
    "haircut": 2
}

G = Generator(NOISE_DIM, NUM_CLASSES).to(DEVICE)

try:
    G.load_state_dict(torch.load("generator.pth", map_location=DEVICE))
    G.eval()
    logger.info("Cartoon generator loaded successfully")
except Exception as e:
    logger.warning(f"Generator not loaded: {e}")

# -------------------- Sentence Model --------------------

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOCAB_PATH = os.path.join(BASE_DIR, "sentence_prediction", "vocab.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "sentence_prediction", "asd_model.pt")

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

try:
    sentence_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    sentence_model.eval()
    logger.info("Sentence model loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentence model: {e}")

# -------------------- Request Models --------------------

class FullGenerateRequest(BaseModel):
    scenario: str
    name: str
    emotion: str

class SentenceRequest(BaseModel):
    text: str

# -------------------- Helpers --------------------

def tensor_to_base64(image_tensor):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    image_tensor = torch.clamp(image_tensor, 0, 1)
    transform = transforms.ToPILImage()
    image = transform(image_tensor.cpu())

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def predict_sentence_model(text):
    sentence_model.eval()

    tokens = text.lower().split()

    # Convert input to indices and pad to MAX_LEN
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    indices = indices[:MAX_LEN]
    indices += [vocab["<pad>"]] * (MAX_LEN - len(indices))

    src_tensor = torch.tensor(indices).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hidden = sentence_model.encoder(src_tensor)

    input_token = torch.tensor([vocab["<sos>"]]).to(DEVICE)

    generated_tokens = []

    for _ in range(MAX_LEN):
        with torch.no_grad():
            output, hidden = sentence_model.decoder(input_token, hidden)

        top1 = output.argmax(1)
        token_id = top1.item()

        if token_id == vocab["<eos>"] or token_id == vocab["<pad>"]:
            break

        generated_tokens.append(inv_vocab.get(token_id, ""))

        input_token = top1

    return " ".join(generated_tokens)

# -------------------- Routes --------------------

@app.get("/")
def home():
    return {"message": "API running successfully"}

@app.post("/generate-full")
def generate_full(request: FullGenerateRequest):
    try:
        if request.scenario not in SCENARIO_TO_LABEL:
            raise HTTPException(status_code=400, detail="Invalid scenario")

        label_value = SCENARIO_TO_LABEL[request.scenario]

        # Generate cartoon
        noise = torch.randn(1, NOISE_DIM, device=DEVICE)
        label = torch.tensor([label_value], device=DEVICE)

        with torch.no_grad():
            fake_image = G(noise, label)

        img_base64 = tensor_to_base64(fake_image)

        scenario_text = request.scenario.replace("_", " ")

        story = f"""
        Once upon a time, {request.name} was feeling {request.emotion}.
        Today was the day of the {scenario_text}.
        At first, it felt a little scary.
        But the people there were kind and friendly.
        {request.name} stayed brave and everything went well!
        """

        return {
            "story": story.strip(),
            "image": img_base64,
            "scenario": scenario_text
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-sentence")
def predict_sentence(request: SentenceRequest):
    try:
        fragmented = request.text.strip()
        prediction = predict_sentence_model(fragmented)

        return {
            "input": fragmented,
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))