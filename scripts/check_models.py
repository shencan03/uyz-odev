import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import get_device  # noqa: E402
from src.models import MLP, LogisticRegression, SmallCNN, n_params  # noqa: E402

device = get_device()
print("Cihaz:", device)
for name, model, shape in [
    ("LogisticRegression (Fashion-MNIST)", LogisticRegression(), (8, 1, 28, 28)),
    ("MLP (Fashion-MNIST)", MLP(), (8, 1, 28, 28)),
    ("SmallCNN (CIFAR-10)", SmallCNN(), (8, 3, 32, 32)),
]:
    model = model.to(device)
    out = model(torch.randn(*shape, device=device))
    assert out.shape == (shape[0], 10)
    print(f"{name:36s} parametre: {n_params(model):>9,}  çıktı: {tuple(out.shape)}")
