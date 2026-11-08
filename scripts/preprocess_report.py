"""Ön işleme öncesi/sonrası karşılaştırma tablosu ve görselleri.

Çıktılar: results/preprocess_comparison.md, figures/*_preprocess_hist.png,
figures/split_distribution.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import ROOT, load_cifar10, load_fashion_mnist  # noqa: E402
from src.preprocess import prepare  # noqa: E402

FIG = ROOT / "figures"


def describe(name, raw, p):
    xtr, ytr, xte, _ = raw
    rows = [
        ("Veri sayısı",
         f"Eğitim {len(xtr):,} / Test {len(xte):,}",
         f"Eğitim {len(p['x_train']):,} / Doğrulama {len(p['x_val']):,} / "
         f"Test {len(p['x_test']):,}"),
        ("Veri boyutu", f"{tuple(xtr.shape[1:])}", f"{tuple(p['x_train'].shape[1:])} (C,H,W)"),
        ("Veri formatı", f"{xtr.dtype}, [0, 255]", f"{p['x_train'].dtype}, standardize"),
        ("Eksik veri", "Yok", "Yok"),
        ("Normalizasyon", "Yok",
         "Kanal bazında (x-μ)/σ; μ=" + ", ".join(f"{m:.3f}" for m in p["mean"])
         + "; σ=" + ", ".join(f"{s:.3f}" for s in p["std"])),
        ("Eğitim ort./std (sonra)", "-",
         f"{p['x_train'].mean():.3f} / {p['x_train'].std():.3f}"),
        ("Etiket durumu", "0-9 tam sayı, eksiksiz", "Değişmedi; her bölmede 10 sınıf dengeli"),
    ]
    md = [f"### {name}", "", "| Özellik | Ön İşleme Öncesi | Ön İşleme Sonrası |",
          "|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(md)


def hist(name, raw_x, p, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].hist(raw_x.ravel()[::7], bins=64, color="gray")
    axes[0].set_title(f"{name} — önce (uint8)")
    axes[1].hist(p["x_train"].ravel()[::7], bins=64, color="tab:blue")
    axes[1].set_title(f"{name} — sonra (standardize)")
    for ax in axes:
        ax.set_ylabel("Frekans")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    sections, splits = [], {}
    for name, key, loader in [("Fashion-MNIST", "fmnist", load_fashion_mnist),
                              ("CIFAR-10", "cifar10", load_cifar10)]:
        raw = loader()
        p = prepare(*raw)
        sections.append(describe(name, raw, p))
        hist(name, raw[0], p, FIG / f"{key}_preprocess_hist.png")
        splits[name] = (len(p["x_train"]), len(p["x_val"]), len(p["x_test"]))
        print(name, "val sınıf sayıları:", np.bincount(p["y_val"]).tolist())

    fig, ax = plt.subplots(figsize=(7, 3))
    names = list(splits)
    left = np.zeros(len(names))
    for i, (lbl, col) in enumerate([("Eğitim", "tab:blue"),
                                    ("Doğrulama", "tab:orange"),
                                    ("Test", "tab:green")]):
        vals = np.array([splits[n][i] for n in names])
        ax.barh(names, vals, left=left, color=col, label=lbl)
        for y, (l, v) in enumerate(zip(left, vals)):
            ax.text(l + v / 2, y, f"{v:,}", ha="center", va="center",
                    color="white", fontsize=9)
        left += vals
    ax.set_xlabel("Görüntü sayısı")
    ax.set_title("Eğitim / doğrulama / test dağılımı")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "split_distribution.png", dpi=150)
    plt.close(fig)

    text = "\n\n".join(sections)
    (ROOT / "results" / "preprocess_comparison.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
