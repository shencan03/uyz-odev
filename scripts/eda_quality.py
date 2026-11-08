"""EDA: veri kalitesi kontrolleri.

- Eksik/bozuk veri (NaN olamaz ama boyut, tip ve etiket aralığı kontrol edilir)
- Etiket durumu (tüm örneklerin geçerli bir etiketi var mı)
- Birebir yinelenen görüntüler (eğitim içi, test içi, eğitim-test arası)
  Bayt düzeyinde hash ile tespit edilir.

Sonuçlar results/quality.json dosyasına yazılır.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import ROOT, load_cifar10, load_fashion_mnist  # noqa: E402


def hashes(x):
    return [hashlib.md5(img.tobytes()).hexdigest() for img in x]


def duplicate_report(xtr, ytr, xte, yte):
    htr, hte = hashes(xtr), hashes(xte)
    groups = defaultdict(list)
    for i, h in enumerate(htr):
        groups[h].append(i)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    conflicting = sum(len(set(ytr[g])) > 1 for g in dup_groups)
    train_set = set(htr)
    cross = [i for i, h in enumerate(hte) if h in train_set]
    return {
        "train_duplicate_images": int(sum(len(g) - 1 for g in dup_groups)),
        "train_duplicate_groups_conflicting_labels": int(conflicting),
        "test_duplicate_images": int(len(hte) - len(set(hte))),
        "test_images_also_in_train": len(cross),
    }


def basic_report(xtr, ytr, xte, yte, n_classes):
    return {
        "train_shape": list(xtr.shape),
        "test_shape": list(xte.shape),
        "dtype": str(xtr.dtype),
        "pixel_min": int(min(xtr.min(), xte.min())),
        "pixel_max": int(max(xtr.max(), xte.max())),
        "labels_valid": bool(np.isin(ytr, range(n_classes)).all()
                             and np.isin(yte, range(n_classes)).all()),
        "missing_labels": int(len(xtr) - len(ytr) + len(xte) - len(yte)),
        "all_black_images": int((xtr.reshape(len(xtr), -1).max(1) == 0).sum()),
    }


def main():
    out = {}
    for name, loader in [("fashion_mnist", load_fashion_mnist),
                         ("cifar10", load_cifar10)]:
        xtr, ytr, xte, yte = loader()
        rep = basic_report(xtr, ytr, xte, yte, 10)
        rep.update(duplicate_report(xtr, ytr, xte, yte))
        out[name] = rep
        print(name, json.dumps(rep, indent=2))
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "quality.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
