import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.modules.model import DenseNetClassifier


# =====================
# CONFIG
# =====================
MODEL_PATH = "notebooks/densenet_isic.pth"
IMAGE_PATH = "data/slice/ISIC_SUBSET/images"
VAL_CSV = "data/slice/ISIC_SUBSET/val.csv"

IMG_SIZE = 224

CLASS_NAMES = [
    "Melanoma",
    "Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# LOAD MODEL
# =====================
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = DenseNetClassifier(
        num_classes=checkpoint["num_classes"], freeze_backbone=False
    ).to(DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    return model


# =====================
# IMAGE TRANSFORM
# =====================
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# =====================
# PREDICT SINGLE IMAGE
# =====================
def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return CLASS_NAMES[pred.item()], conf.item()


# =====================
# RUN PREDICTION
# =====================
def main():
    model = load_model(MODEL_PATH)
    df = pd.read_csv(VAL_CSV)

    results = []
    for idx, row in df.iterrows():
        img_name = row["image"].strip() + ".jpg"
        img_path = Path(IMAGE_PATH) / img_name

        if not img_path.exists():
            print(f"WARNING: {img_path} not found, skipping.")
            continue

        pred_class, confidence = predict_image(model, img_path)
        results.append(
            {"image": img_name, "prediction": pred_class, "confidence": confidence}
        )
        print(f"{img_name}: {pred_class} ({confidence:.2f})")

    # save predictions
    results_df = pd.DataFrame(results)
    results_df.to_csv("predictions_val.csv", index=False)
    print("Predictions saved to predictions_val.csv")


if __name__ == "__main__":
    main()
