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


def channel_stats(x: np.ndarray):
    """(N,C,H,W) dizisinden kanal başına ortalama ve standart sapma."""
    return x.mean(axis=(0, 2, 3)), x.std(axis=(0, 2, 3))


def normalize(x: np.ndarray, mean, std) -> np.ndarray:
    """Kanal bazında standardizasyon: (x - mean) / std."""
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
    return (x - mean) / std


def prepare(x_train, y_train, x_test, y_test, val_ratio=0.1, seed=42):
    """Tam ön işleme hattı.

    1) [0,1] ölçekleme  2) stratified doğrulama ayırma
    3) ortalama/std YALNIZCA eğitim alt kümesinden hesaplanır (veri sızıntısı yok)
    4) eğitim, doğrulama ve test aynı istatistiklerle normalize edilir.
    """
    xtr, xte = to_float(x_train), to_float(x_test)
    tr_idx, va_idx = stratified_split(y_train, val_ratio, seed)
    mean, std = channel_stats(xtr[tr_idx])
    return {
        "x_train": normalize(xtr[tr_idx], mean, std), "y_train": y_train[tr_idx],
        "x_val": normalize(xtr[va_idx], mean, std), "y_val": y_train[va_idx],
        "x_test": normalize(xte, mean, std), "y_test": y_test,
        "mean": mean.tolist(), "std": std.tolist(),
    }
