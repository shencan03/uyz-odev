"""EDA: örnek görüntüler, sınıf dağılımları ve eğitim/test dağılımı.

Çıktılar figures/ klasörüne kaydedilir.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import (CIFAR10_CLASSES, FMNIST_CLASSES, ROOT,  # noqa: E402
                      load_cifar10, load_fashion_mnist)

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def sample_grid(x, y, classes, title, path, per_class=8, cmap=None):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(classes), per_class,
                             figsize=(per_class * 1.1, len(classes) * 1.15))
    for c, name in enumerate(classes):
        idx = rng.choice(np.flatnonzero(y == c), per_class, replace=False)
        for j, i in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(x[i], cmap=cmap)
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center",
                              fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def class_distribution(ytr, yte, classes, title, path):
    tr = np.bincount(ytr, minlength=len(classes))
    te = np.bincount(yte, minlength=len(classes))
    pos = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(pos - 0.2, tr, 0.4, label=f"Eğitim (n={tr.sum():,})")
    ax.bar(pos + 0.2, te, 0.4, label=f"Test (n={te.sum():,})")
    ax.set_xticks(pos, classes, rotation=30, ha="right")
    ax.set_ylabel("Görüntü sayısı")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return tr, te


def main():
    xtr, ytr, xte, yte = load_fashion_mnist()
    sample_grid(xtr, ytr, FMNIST_CLASSES, "Fashion-MNIST örnek görüntüler",
                FIG / "fmnist_samples.png", cmap="gray")
    tr, te = class_distribution(ytr, yte, FMNIST_CLASSES,
                                "Fashion-MNIST sınıf dağılımı",
                                FIG / "fmnist_class_dist.png")
    print("Fashion-MNIST sınıf başına eğitim/test:", tr.tolist(), te.tolist())

    xtr, ytr, xte, yte = load_cifar10()
    sample_grid(xtr, ytr, CIFAR10_CLASSES, "CIFAR-10 örnek görüntüler",
                FIG / "cifar10_samples.png")
    tr, te = class_distribution(ytr, yte, CIFAR10_CLASSES,
                                "CIFAR-10 sınıf dağılımı",
                                FIG / "cifar10_class_dist.png")
    print("CIFAR-10 sınıf başına eğitim/test:", tr.tolist(), te.tolist())


if __name__ == "__main__":
    main()
