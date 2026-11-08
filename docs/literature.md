# Literatür Taraması

Değerler ilgili makalelerin tablolarından alınmıştır (doğruluk = 100 − test hatası).

| No | Çalışma | Yıl | Veri Seti | Yöntem | Temel Sonuç (test doğruluğu) |
|---|---|---|---|---|---|
| 1 | Xiao vd., *Fashion-MNIST: a Novel Image Dataset for Benchmarking ML Algorithms* [1] | 2017 | Fashion-MNIST | scikit-learn sınıflandırıcıları (LogisticRegression, MLPClassifier, SVC, RF, kNN vb.), ham pikseller | SVC (RBF, C=10) **%89,7**; MLP (100 gizli, ReLU) **%87,1**; Logistic Regression **%84,2** |
| 2 | He vd., *Deep Residual Learning for Image Recognition* [2] | 2016 | CIFAR-10 | ResNet (skip connection), flip + translation artırma | ResNet-110: %6,43 hata → **%93,57** |
| 3 | Zagoruyko ve Komodakis, *Wide Residual Networks* [3] | 2016 | CIFAR-10 | WRN (geniş, sığ residual ağ), flip + translation | WRN-28-10: %4,00 hata → **%96,00** (dropout ile %96,11) |
| 4 | Huang vd., *Densely Connected Convolutional Networks* [4] | 2017 | CIFAR-10 | DenseNet-BC (yoğun bağlantılar, özellik yeniden kullanımı) | DenseNet-BC (L=190, k=40): %3,46 hata → **%96,54** |
| 5 | Zhong vd., *Random Erasing Data Augmentation* [5] | 2020 | CIFAR-10, Fashion-MNIST | WRN-28-10, ResNet + Random Erasing | CIFAR-10: %96,20 → **%96,92**; Fashion-MNIST: %95,99 → **%96,35** |

## Yöntemlerin analizi

- **Klasik/sığ modeller (Fashion-MNIST):** Ham piksellerle eğitilen doğrusal modeller ~%84'te kalır; tek gizli katmanlı MLP ~%87'ye, çekirdekli SVM ~%90'a çıkar. Doğrusal olmayan karar sınırları belirgin kazanç sağlar.
- **Derin CNN'ler (CIFAR-10):** Residual (ResNet), geniş residual (WRN) ve yoğun bağlantılı (DenseNet) mimariler %93,5–%96,5 aralığındadır. Derinlik/genişlik arttıkça doğruluk artar ama parametre sayısı (1,7M → 36,5M) ve eğitim maliyeti de artar.
- **Veri artırma:** Tüm CIFAR-10 sonuçları en azından flip + translation artırmasıyla elde edilmiştir; Random Erasing bunun üzerine ~0,7 puan daha kazandırır. Artırma yapılmayan bir modelle bu sonuçların doğrudan karşılaştırılması adil değildir.
- **Metrikler:** Çalışmaların hepsi yalnızca top-1 doğruluk/hata raporlar. Veri setleri dengeli olduğundan doğruluk anlamlıdır; ancak sınıf bazlı hatalar (ör. Shirt / T-shirt / Pullover karışması) incelenmemiştir.

## Genel değerlendirme ve projenin katkısı

Literatür, CIFAR-10'da milyonlarca parametreli derin ağlara odaklanırken; küçük, hızlı eğitilen modellerin hangi sınıflarda neden hata yaptığına daha az yer verir. Bu proje, Fashion-MNIST'te doğrusal ve doğrusal olmayan modeli aynı koşullarda doğrudan karşılaştırarak ve ~0,5M parametreli küçük bir CNN'in CIFAR-10'daki sonucunu literatürdeki büyük mimarilerle maliyet/başarı açısından konumlandırarak; karışıklık matrisleri ve makro-F1 ile sınıf bazlı hata analizine odaklanır.

## Kaynakça

[1] H. Xiao, K. Rasul ve R. Vollgraf, "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms," arXiv:1708.07747, 2017. https://doi.org/10.48550/arXiv.1708.07747
[2] K. He, X. Zhang, S. Ren ve J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 770–778, 2016. https://doi.org/10.1109/CVPR.2016.90
[3] S. Zagoruyko ve N. Komodakis, "Wide Residual Networks," BMVC, 87.1–87.12, 2016. https://doi.org/10.5244/C.30.87
[4] G. Huang, Z. Liu, L. van der Maaten ve K. Q. Weinberger, "Densely Connected Convolutional Networks," CVPR, 2261–2269, 2017. https://doi.org/10.1109/CVPR.2017.243
[5] Z. Zhong, L. Zheng, G. Kang, S. Li ve Y. Yang, "Random Erasing Data Augmentation," AAAI, 34(7), 13001–13008, 2020. https://doi.org/10.1609/aaai.v34i07.7000

Veri setleri:
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
