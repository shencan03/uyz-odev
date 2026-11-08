"""Veri yükleme: Fashion-MNIST ve CIFAR-10 resmi kaynaklarından indirilir.

Veriler torchvision üzerinden orijinal kaynaklardan (Zalando Research ve
Toronto Üniversitesi) indirilir ve NumPy dizileri olarak döndürülür.
"""
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

FMNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_fashion_mnist():
    """(x_train, y_train, x_test, y_test) döndürür; x: uint8 (N, 28, 28)."""
    tr = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    te = datasets.FashionMNIST(DATA_DIR, train=False, download=True)
    return (tr.data.numpy(), tr.targets.numpy(),
            te.data.numpy(), te.targets.numpy())


def load_cifar10():
    """(x_train, y_train, x_test, y_test) döndürür; x: uint8 (N, 32, 32, 3)."""
    tr = datasets.CIFAR10(DATA_DIR, train=True, download=True)
    te = datasets.CIFAR10(DATA_DIR, train=False, download=True)
    return (tr.data, np.array(tr.targets),
            te.data, np.array(te.targets))


if __name__ == "__main__":
    for name, loader in [("Fashion-MNIST", load_fashion_mnist),
                         ("CIFAR-10", load_cifar10)]:
        xtr, ytr, xte, yte = loader()
        print(f"{name}: train {xtr.shape} {xtr.dtype}, test {xte.shape}, "
              f"etiketler {np.unique(ytr).size} sınıf")
    print("Cihaz:", get_device())
