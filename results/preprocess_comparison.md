### Fashion-MNIST

| Özellik | Ön İşleme Öncesi | Ön İşleme Sonrası |
|---|---|---|
| Veri sayısı | Eğitim 60,000 / Test 10,000 | Eğitim 54,000 / Doğrulama 6,000 / Test 10,000 |
| Veri boyutu | (28, 28) | (1, 28, 28) (C,H,W) |
| Veri formatı | uint8, [0, 255] | float32, standardize |
| Eksik veri | Yok | Yok |
| Normalizasyon | Yok | Kanal bazında (x-μ)/σ; μ=0.286; σ=0.353 |
| Eğitim ort./std (sonra) | - | 0.000 / 1.000 |
| Etiket durumu | 0-9 tam sayı, eksiksiz | Değişmedi; her bölmede 10 sınıf dengeli |

### CIFAR-10

| Özellik | Ön İşleme Öncesi | Ön İşleme Sonrası |
|---|---|---|
| Veri sayısı | Eğitim 50,000 / Test 10,000 | Eğitim 45,000 / Doğrulama 5,000 / Test 10,000 |
| Veri boyutu | (32, 32, 3) | (3, 32, 32) (C,H,W) |
| Veri formatı | uint8, [0, 255] | float32, standardize |
| Eksik veri | Yok | Yok |
| Normalizasyon | Yok | Kanal bazında (x-μ)/σ; μ=0.492, 0.483, 0.447; σ=0.247, 0.244, 0.262 |
| Eğitim ort./std (sonra) | - | -0.000 / 1.000 |
| Etiket durumu | 0-9 tam sayı, eksiksiz | Değişmedi; her bölmede 10 sınıf dengeli |
