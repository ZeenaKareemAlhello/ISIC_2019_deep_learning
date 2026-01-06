# model.py
import torch.nn as nn
from torchvision import models


class DenseNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes=7,
        freeze_backbone=True,
        dropout=0.5,
    ):
        super().__init__()

        self.model = models.densenet121(weights="IMAGENET1K_V1")

        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

        in_features = self.model.classifier.in_features

        # Custom classifier head
        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)
