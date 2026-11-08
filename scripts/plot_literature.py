"""Literatürde raporlanan test doğrulukları (figures/literature_results.png).

Değerler docs/literature.md tablosundaki makale sonuçlarıdır.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import ROOT  # noqa: E402

FMNIST = [("Logistic Reg. [1]", 84.2), ("MLP (100) [1]", 87.1), ("SVC RBF [1]", 89.7),
          ("WRN-28-10 [5]", 95.99), ("WRN-28-10 + RE [5]", 96.35)]
CIFAR = [("ResNet-110 [2]", 93.57), ("WRN-28-10 [3]", 96.00),
         ("DenseNet-BC k=40 [4]", 96.54), ("WRN-28-10 + RE [5]", 96.92)]


def panel(ax, rows, title, color, lo):
    names, vals = zip(*rows)
    bars = ax.barh(names, vals, color=color)
    ax.bar_label(bars, labels=[f"%{v:.1f}" for v in vals], padding=3, fontsize=9)
    ax.set_xlim(lo, 100)
    ax.set_title(title)
    ax.set_xlabel("Test doğruluğu (%)")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    panel(axes[0], FMNIST, "Fashion-MNIST", "#5b8db8", 80)
    panel(axes[1], CIFAR, "CIFAR-10", "#d07a4a", 90)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "literature_results.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
