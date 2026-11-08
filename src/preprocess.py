import numpy as np


def to_float(x: np.ndarray) -> np.ndarray:
    """[0,1] ölçekleme ve PyTorch kanal-önce düzeni: (N,H,W[,C]) -> (N,C,H,W)."""
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
    return x.mean(axis=(0, 2, 3)), x.std(axis=(0, 2, 3))


def normalize(x: np.ndarray, mean, std) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
    return (x - mean) / std


def prepare(x_train, y_train, x_test, y_test, val_ratio=0.1, seed=42):
    """Ortalama/std yalnızca eğitim alt kümesinden hesaplanır (veri sızıntısını önler)."""
    xtr, xte = to_float(x_train), to_float(x_test)
    tr_idx, va_idx = stratified_split(y_train, val_ratio, seed)
    mean, std = channel_stats(xtr[tr_idx])
    return {
        "x_train": normalize(xtr[tr_idx], mean, std), "y_train": y_train[tr_idx],
        "x_val": normalize(xtr[va_idx], mean, std), "y_val": y_train[va_idx],
        "x_test": normalize(xte, mean, std), "y_test": y_test,
        "mean": mean.tolist(), "std": std.tolist(),
    }
