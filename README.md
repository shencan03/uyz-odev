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

## Çalıştırma

```bash
python src/data.py                  # veri setlerini indirir (data/ altına)
python scripts/eda_overview.py      # örnek görüntüler, sınıf dağılımları
python scripts/eda_quality.py       # eksik veri, etiket, yinelenen görüntü kontrolü
python scripts/eda_pixels.py        # piksel / RGB kanal analizi
python scripts/preprocess_report.py # ön işleme öncesi/sonrası karşılaştırma
python scripts/check_models.py      # model tanımlarının MPS üzerinde testi
```

Grafikler `figures/`, sayısal çıktılar `results/` klasörüne yazılır.

## Proje yapısı

```
src/data.py          veri yükleme, cihaz seçimi (MPS > CUDA > CPU)
src/preprocess.py    ölçekleme, stratified doğrulama ayırma, normalizasyon
src/models.py        LogisticRegression, MLP, SmallCNN
scripts/             EDA ve kontrol betikleri
docs/                proje önerisi, literatür taraması
figures/  results/   üretilen grafikler ve tablolar
```

## Faz 1 özeti

| Veri seti | Eğitim / Doğrulama / Test | Görüntü | Sınıf |
|---|---|---|---|
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | 54.000 / 6.000 / 10.000 | 28×28 grayscale | 10, dengeli |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | 45.000 / 5.000 / 10.000 | 32×32 RGB | 10, dengeli |

- Eksik veri, geçersiz etiket, birebir yinelenen görüntü veya eğitim-test sızıntısı bulunmadı.
- Ön işleme: [0,1] ölçekleme → %10 stratified doğrulama ayırma (seed=42) → yalnızca eğitim alt kümesinden hesaplanan kanal ortalama/std ile normalizasyon.
- Modeller: Logistic Regression (7.850 param.) ve MLP (203.530 param.) Fashion-MNIST'te; SmallCNN (545.290 param.) CIFAR-10'da.
- Metrikler: doğruluk, makro-F1, eğitim süresi; karışıklık matrisi ile hata analizi.
- Literatür özeti: [docs/literature.md](docs/literature.md)
