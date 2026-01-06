# validate_isic_onehot.py
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.modules.model import DenseNetClassifier

# =====================
# CONFIG
# =====================
MODEL_PATH = "src/scripts/densenet_isic.pth"
IMAGE_DIR = "data/slice/ISIC_SUBSET/images"
VAL_CSV = "data/slice/ISIC_SUBSET/val.csv"
IMG_SIZE = 224

# Only include the classes used for training
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# LOAD MODEL
# =====================
def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = DenseNetClassifier(num_classes=len(CLASS_NAMES), freeze_backbone=False).to(
        DEVICE
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# =====================
# PREDICT SINGLE IMAGE
# =====================
def predict(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        pred_idx = outputs.argmax(1).item()
        confidence = torch.softmax(outputs, dim=1)[0, pred_idx].item()
    return pred_idx, CLASS_NAMES[pred_idx], confidence


# =====================
# MAIN VALIDATION
# =====================
def main():
    model = load_model(MODEL_PATH)
    df = pd.read_csv(VAL_CSV)

    total = 0
    correct = 0
    results = []

    for _, row in df.iterrows():
        img_name = row["image"].strip() + ".jpg"
        img_path = Path(IMAGE_DIR) / img_name
        if not img_path.exists():
            print(f"WARNING: {img_path} not found, skipping.")
            continue

        pred_idx, pred_class, conf = predict(model, img_path)

        # --- ground truth from one-hot columns ---
        gt_idx = row[1 : 1 + len(CLASS_NAMES)].values.argmax()
        gt_class = CLASS_NAMES[gt_idx]

        total += 1
        correct += int(pred_idx == gt_idx)

        results.append(
            {
                "image": img_name,
                "gt_label": gt_class,
                "prediction": pred_class,
                "confidence": round(conf, 3),
            }
        )

        print(f"{img_name}: GT={gt_class} | PRED={pred_class} ({conf:.2f})")

    # --- final accuracy ---
    val_acc = correct / total if total > 0 else 0
    print(f"\nValidation Accuracy: {val_acc:.4f}")

    # --- save predictions ---
    pd.DataFrame(results).to_csv("src/scripts/predictions_val.csv", index=False)
    print("Predictions saved to predictions_val.csv")


if __name__ == "__main__":
    main()
