"""EDA: piksel yoğunluk dağılımı, RGB kanal analizi ve sınıf ortalama görüntüleri.

Kanal ortalama/std değerleri yalnızca EĞİTİM setinden hesaplanır ve
results/pixel_stats.json dosyasına yazılır (normalizasyonda kullanılır).
"""
import json
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


def class_means(x, y, classes, title, path, cmap=None):
    fig, axes = plt.subplots(1, len(classes), figsize=(len(classes) * 1.3, 1.8))
    for c, ax in enumerate(axes):
        m = x[y == c].mean(0)
        ax.imshow(m.astype(np.uint8) if m.ndim == 3 else m, cmap=cmap)
        ax.set_title(classes[c], fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    stats = {}

    # Fashion-MNIST: tek kanal (grayscale)
    xtr, ytr, _, _ = load_fashion_mnist()
    f = xtr.astype(np.float32) / 255.0
    stats["fashion_mnist"] = {"mean": [float(f.mean())], "std": [float(f.std())],
                              "zero_pixel_ratio": float((xtr == 0).mean())}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(xtr.ravel(), bins=64, color="gray", log=True)
    ax.set_xlabel("Piksel değeri (0-255)")
    ax.set_ylabel("Frekans (log)")
    ax.set_title("Fashion-MNIST piksel yoğunluk dağılımı (grayscale)")
    fig.tight_layout()
    fig.savefig(FIG / "fmnist_pixel_hist.png", dpi=150)
    plt.close(fig)
    class_means(xtr, ytr, FMNIST_CLASSES, "Fashion-MNIST sınıf ortalama görüntüleri",
                FIG / "fmnist_class_means.png", cmap="gray")

    # CIFAR-10: üç kanal (RGB)
    xtr, ytr, _, _ = load_cifar10()
    c = xtr.astype(np.float32) / 255.0
    stats["cifar10"] = {"mean": c.mean((0, 1, 2)).tolist(),
                        "std": c.std((0, 1, 2)).tolist()}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for ch, col in enumerate(["red", "green", "blue"]):
        ax.hist(xtr[..., ch].ravel(), bins=64, histtype="step", color=col,
                label=f"{col[0].upper()} (ort={stats['cifar10']['mean'][ch]:.3f})")
    ax.set_xlabel("Piksel değeri (0-255)")
    ax.set_ylabel("Frekans")
    ax.set_title("CIFAR-10 RGB kanal dağılımları")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cifar10_rgb_hist.png", dpi=150)
    plt.close(fig)
    class_means(xtr, ytr, CIFAR10_CLASSES, "CIFAR-10 sınıf ortalama görüntüleri",
                FIG / "cifar10_class_means.png")

    print(json.dumps(stats, indent=2))
    (ROOT / "results" / "pixel_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
