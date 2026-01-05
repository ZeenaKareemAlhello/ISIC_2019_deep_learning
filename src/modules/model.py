# model.py
import torch.nn as nn
from torchvision import models


class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes=7, freeze_backbone=True):
        super().__init__()

        self.model = models.densenet121(weights="IMAGENET1K_V1")

        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)
    