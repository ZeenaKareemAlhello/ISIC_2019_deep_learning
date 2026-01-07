import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from pathlib import Path
import sys
from src.modules.model import DenseNetClassifier

ROOT = Path.cwd()
sys.path.append(str(ROOT))


# --------------------
# CONFIG
# --------------------
MODEL_PATH = "src/scripts/densenet_isic.pth"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Melanoma",
    "Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
]

# --------------------
# APP
# --------------------
app = FastAPI(title="ISIC Skin Lesion Classifier")

# --------------------
# LOAD MODEL (ONCE)
# --------------------
model = DenseNetClassifier(num_classes=len(CLASS_NAMES), freeze_backbone=False)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

# --------------------
# TRANSFORM
# --------------------
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# --------------------
# HEALTH CHECK
# --------------------
@app.get("/")
def health_check():
    return {"status": "ok", "device": DEVICE}


# --------------------
# PREDICTION ENDPOINT
# --------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    return {
        "filename": file.filename,
        "prediction": CLASS_NAMES[pred.item()],
        "confidence": round(confidence.item(), 4),
    }
