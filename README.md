# Fashion-MNIST ve CIFAR-10 Üzerinde Görüntü Sınıflandırma

Uygulamalı Yapay Zeka dersi dönem projesi — Hüseyin Mehmet Şencan (bireysel).

Fashion-MNIST üzerinde Logistic Regression ve MLP, CIFAR-10 üzerinde küçük bir CNN
eğitilerek modellerin başarıları ve sık yaptıkları sınıflandırma hataları incelenecektir.

## Ortam

Proje Apple Silicon (M4) üzerinde geliştirilmektedir. PyTorch'un MPS (Metal) arka ucu
kullanılır; MPS yoksa CUDA, o da yoksa CPU seçilir.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```
