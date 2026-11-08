"""Ön işleme: ölçekleme, doğrulama ayırma ve normalizasyon."""
import numpy as np


def to_float(x: np.ndarray) -> np.ndarray:
    """uint8 [0,255] pikselleri float32 [0,1] aralığına ölçekler.

    Fashion-MNIST (N,28,28) -> (N,1,28,28), CIFAR-10 (N,32,32,3) -> (N,3,32,32)
    biçimine (PyTorch'un kanal-önce düzeni) getirilir.
    """
    x = x.astype(np.float32) / 255.0
    if x.ndim == 3:
        return x[:, None, :, :]
    return x.transpose(0, 3, 1, 2).copy()


def stratified_split(y: np.ndarray, val_ratio: float = 0.1, seed: int = 42):
    """Her sınıftan aynı oranda örnek alarak eğitim/doğrulama indeksleri üretir.

    Sabit seed ile tekrarlanabilir. Test seti yalnızca son değerlendirmede
    kullanılacağından bu bölmeye dahil edilmez.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_val = int(round(len(idx) * val_ratio))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))
