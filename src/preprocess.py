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
