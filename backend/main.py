from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import logging

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

# -------------------- Load Model --------------------

G = Generator(NOISE_DIM, NUM_CLASSES).to(DEVICE)

try:
    G.load_state_dict(torch.load("generator.pth", map_location=DEVICE))
    G.eval()
    logger.info("Generator model loaded successfully")
except Exception as e:
    logger.warning(f"Generator not loaded: {e}")

# -------------------- Request Model --------------------

class FullGenerateRequest(BaseModel):
    scenario: str
    name: str
    emotion : str

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

        # Generate story
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