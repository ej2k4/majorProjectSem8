from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import torch
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import logging

from majorProjectSem8.cartoonImage_model.generator import Generator  # from existing repo file

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cartoon GAN API")

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- GLOBALS ---------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
NOISE_DIM = 100
LABELS_MAP = {0: "dentist", 1: "doctor", 2: "haircut"}

# Load Generator 
G = Generator(NOISE_DIM, NUM_CLASSES).to(DEVICE)

# Try loading trained weights
try:
    G.load_state_dict(torch.load("generator.pth", map_location=DEVICE))
    G.eval()
    logger.info("Generator model loaded successfully")
except FileNotFoundError:
    logger.warning("No trained generator.pth found. Using untrained model.")
except Exception as e:
    logger.error(f"Error loading model: {e}")

# --------- REQUEST MODEL ---------
class GenerateRequest(BaseModel):
    label: int = Field(..., ge=0, le=2, description="Label must be 0, 1, or 2")
    
    @validator('label')
    def validate_label(cls, v):
        if v not in LABELS_MAP:
            raise ValueError(f'Label must be one of {list(LABELS_MAP.keys())}')
        return v

# --------- HELPER FUNCTIONS ---------
def tensor_to_base64(image_tensor):
    """Convert tensor to base64 encoded image"""
    # Ensure tensor is in correct format (C, H, W) and values in [0,1]
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # Clamp values to [0,1] in case they're outside this range
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    transform = transforms.ToPILImage()
    image = transform(image_tensor.cpu())
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --------- ENDPOINTS ---------

@app.get("/")
def home():
    return {
        "message": "Cartoon GAN API is running",
        "device": str(DEVICE),
        "model_loaded": hasattr(G, 'state_dict')
    }

@app.get("/labels")
def get_labels():
    return {"labels": LABELS_MAP}

@app.post("/generate")
def generate_image(request: GenerateRequest):
    try:
        # Generate random noise
        noise = torch.randn(1, NOISE_DIM, device=DEVICE)
        label = torch.tensor([request.label], device=DEVICE)

        with torch.no_grad():
            fake_image = G(noise, label)

        # Convert to base64
        img_base64 = tensor_to_base64(fake_image)

        return {
            "image": img_base64,
            "label": request.label,
            "label_name": LABELS_MAP[request.label]
        }
    
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": hasattr(G, 'state_dict')
    }