import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, in_features=28 * 28, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_features, n_classes))

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_features=28 * 28, hidden=256, n_classes=10, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=10, p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
