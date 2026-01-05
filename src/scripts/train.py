import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.modules.trainer import Trainer
from src.modules.dataset import ISICDataset
from src.modules.model import DenseNetClassifier

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ISICDataset(
        csv_file="../data/slice/ISIC_SUBSET_5000/train.csv",
        image_dir="../data/slice/ISIC_SUBSET_5000/images",
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = DenseNetClassifier(num_classes=7).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model, device, optimizer, criterion)

    epochs = 1
    best_loss = float("inf")
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_one_epoch(train_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f}"
        )

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                {
                    "model_name": "densenet121",
                    "num_classes": 7,
                    "freeze_backbone": True,
                    "state_dict": model.state_dict(),
                },
                "densenet_isic.pth",
            )


main()