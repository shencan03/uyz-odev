"""Seçilen üç mimarinin blok diyagramı (figures/architectures.png)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import ROOT  # noqa: E402
from src.models import MLP, LogisticRegression, SmallCNN, n_params  # noqa: E402

COLORS = {"in": "#d9d9d9", "lin": "#7fb3d5", "act": "#f7dc6f",
          "conv": "#82c99a", "pool": "#f0b27a", "out": "#c39bd3"}

ROWS = [
    (f"Logistic Regression — Fashion-MNIST ({n_params(LogisticRegression()):,} param.)",
     [("Girdi\n1×28×28", "in"), ("Flatten\n784", "in"), ("Linear\n784→10", "lin"),
      ("Softmax\n10 sınıf", "out")]),
    (f"MLP — Fashion-MNIST ({n_params(MLP()):,} param.)",
     [("Girdi\n1×28×28", "in"), ("Flatten\n784", "in"), ("Linear\n784→256", "lin"),
      ("ReLU +\nDropout 0.2", "act"), ("Linear\n256→10", "lin"), ("Softmax\n10 sınıf", "out")]),
    (f"SmallCNN — CIFAR-10 ({n_params(SmallCNN()):,} param.)",
     [("Girdi\n3×32×32", "in"), ("Conv 3×3, 32\nBN + ReLU", "conv"), ("MaxPool\n→32×16×16", "pool"),
      ("Conv 3×3, 64\nBN + ReLU", "conv"), ("MaxPool\n→64×8×8", "pool"),
      ("Linear\n4096→128", "lin"), ("ReLU +\nDropout 0.3", "act"), ("Linear\n128→10", "lin"),
      ("Softmax\n10 sınıf", "out")]),
]


def main():
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.85, 6.15)
    ax.axis("off")
    bw, bh, gap = 1.22, 0.85, 0.2
    for r, (title, blocks) in enumerate(ROWS):
        y = 5.0 - r * 2.0
        ax.text(0.1, y + bh + 0.2, title, fontsize=11, fontweight="bold")
        for i, (label, kind) in enumerate(blocks):
            x = 0.1 + i * (bw + gap)
            ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=COLORS[kind], ec="#555555", lw=0.8))
            ax.text(x + bw / 2, y + bh / 2, label, ha="center", va="center", fontsize=8.5)
            if i < len(blocks) - 1:
                ax.annotate("", xy=(x + bw + gap, y + bh / 2), xytext=(x + bw, y + bh / 2),
                            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "architectures.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
