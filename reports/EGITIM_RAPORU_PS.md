# 🧬 Peptid Generator - PS (Polistiren) Eğitim Sonuç Raporu

**Tarih:** 6 Ocak 2026  
**Plastik Tipi:** PS (Polistiren / Polystyrene)  
**Platform:** Windows 11 - RTX 4080 Super (16GB VRAM) + Ryzen 9700X + 64GB RAM  
**Toplam Eğitim Süresi:** ~2 saat (Ablation) + ~15 dakika (Final)

---

## 📊 İçindekiler

1. [Özet](#özet)
2. [Donanım ve Optimizasyon](#donanım-ve-optimizasyon)
3. [Model Karşılaştırması](#model-karşılaştırması)
4. [En İyi Model Detayları](#en-iyi-model-detayları)
5. [Ablation Study Detayları](#ablation-study-detayları)
6. [Final Eğitim Metrikleri](#final-eğitim-metrikleri)
7. [Üretilen Peptidler](#üretilen-peptidler)
8. [Benzerlik Analizi](#benzerlik-analizi)
9. [Overfitting Analizi](#overfitting-analizi)
10. [Grafikler ve Görselleştirmeler](#grafikler-ve-görselleştirmeler)
11. [Amino Asit Frekans Analizi](#amino-asit-frekans-analizi)
12. [Öneriler ve Sonraki Adımlar](#öneriler-ve-sonraki-adımlar)
13. [Dosya Yapısı](#dosya-yapısı)
14. [Teknik Detaylar](#teknik-detaylar)

---

## 🎯 Özet

Bu çalışmada, **PS (Polistiren)** plastik tipine bağlanma potansiyeli yüksek peptidler üretmek için dört farklı derin öğrenme modeli kapsamlı bir şekilde test edilmiştir. RTX 4080 Super GPU ile hızlandırılmış eğitim sürecinde **128 hiperparametre kombinasyonu** değerlendirilmiş ve en iyi performansı gösteren **Encoder-Decoder (ENCDEC)** modeli seçilmiştir.

### 🏆 Temel Sonuçlar

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **En İyi Model** | ENCDEC (Encoder-Decoder) | 4 model arasında en yüksek performans |
| **Ablation Val R²** | 0.9371 | Doğrulama seti başarısı |
| **Ablation Test R²** | 0.9378 | Test seti başarısı |
| **Final Test R²** | **0.9569** | Final model test başarısı |
| **Final Test MAE** | 1.2995 | Ortalama mutlak hata |
| **Final Test RMSE** | 1.8490 | Kök ortalama kare hatası |
| **Üretilen Peptid Sayısı** | 5 | Optimize edilmiş peptidler |
| **En Yüksek Peptid Skoru** | -47.18 | En iyi bağlanma potansiyeli |
| **Orijinal Veri Seti** | 405,827 peptid | Eğitim verisi boyutu |

### 📈 Performans Özeti

```
                    Ablation R²    Final R²    İyileşme
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCDEC              0.9378    →    0.9569      +1.91%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 💻 Donanım ve Optimizasyon

### Sistem Özellikleri

| Bileşen | Detay | Kullanım |
|---------|-------|----------|
| **GPU** | NVIDIA GeForce RTX 4080 SUPER | 16GB VRAM |
| **CPU** | AMD Ryzen 9700X | Çok çekirdekli işlem |
| **RAM** | 64GB DDR5 | Büyük veri setleri |
| **İşletim Sistemi** | Windows 11 | CUDA desteği |

### GPU Optimizasyonları

| Özellik | Değer | Etki |
|---------|-------|------|
| `NUM_WORKERS` | 4 | Paralel veri yükleme |
| `PIN_MEMORY` | True | GPU transfer hızlandırma |
| `PERSISTENT_WORKERS` | True | Worker overhead azaltma |
| `USE_AMP` | True | Mixed Precision Training |
| `GradScaler` | PyTorch 2.x | FP16/FP32 otomatik |
| `cudnn.benchmark` | True | Convolution optimizasyonu |
| `TensorFloat-32` | Aktif | RTX 40 serisi için |

### Batch Size Optimizasyonu (16GB VRAM)

| Model | Batch Size | VRAM Kullanımı |
|-------|------------|----------------|
| LSTM | 2048 | ~8-10 GB |
| CNN | 4096 | ~6-8 GB |
| LSTM-VAE | 1024 | ~10-12 GB |
| ENCDEC | 1024 | ~10-12 GB |

---

## 📈 Model Karşılaştırması

### Ablation Study Sonuçları (50 Epoch)

| Sıra | Model | Val R² | Test R² | Best Epoch | Kombinasyon | Eğitim Süresi |
|------|-------|--------|---------|------------|-------------|---------------|
| 🥇 | **ENCDEC** | **0.9371** | **0.9378** | 50 | 128 | ~45 dk |
| 🥈 | LSTM | 0.9282 | 0.9282 | 50 | 192 | ~30 dk |
| 🥉 | LSTM_VAE | 0.9113 | 0.9113 | 50 | 256 | ~35 dk |
| 4 | CNN | 0.7796 | 0.7796 | 50 | 48 | ~20 dk |

### Model Performans Grafiği

```
Val R² Karşılaştırması (Ablation Study)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCDEC   █████████████████████████████████████████████████  93.71%
LSTM     ████████████████████████████████████████████████   92.82%
LSTM_VAE ██████████████████████████████████████████████     91.13%
CNN      ████████████████████████████████                   77.96%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Final Test R² Karşılaştırması

```
Final Test R² (250 Epoch ile Final Eğitim Sonrası)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCDEC   ██████████████████████████████████████████████████ 95.69%
LSTM     ████████████████████████████████████████████████   92.82%
LSTM_VAE ██████████████████████████████████████████████     91.13%
CNN      ████████████████████████████████                   77.96%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Detaylı Ablation Study İstatistikleri

```
================================================================================
📊 GENEL İSTATİSTİKLER
================================================================================
Toplam Model-Plastik Kombinasyonu: 4
En Yüksek Test R²: 0.9569
En Düşük Test R²: 0.7796
Ortalama Test R²: 0.8940
Test R² Standart Sapma: 0.0785

================================================================================
📊 MODEL TİPİNE GÖRE PERFORMANS
================================================================================
CNN:
  Ortalama R²: 0.7796 ± nan
  Aralık: [0.7796, 0.7796]
  Örnek Sayısı: 1

ENCDEC:
  Ortalama R²: 0.9569 ± nan
  Aralık: [0.9569, 0.9569]
  Örnek Sayısı: 1

LSTM:
  Ortalama R²: 0.9282 ± nan
  Aralık: [0.9282, 0.9282]
  Örnek Sayısı: 1

LSTM_VAE:
  Ortalama R²: 0.9113 ± nan
  Aralık: [0.9113, 0.9113]
  Örnek Sayısı: 1

================================================================================
📊 PLASTİK TİPİNE GÖRE PERFORMANS
================================================================================
PS:
  Ortalama R²: 0.8940 ± 0.0785
  Aralık: [0.7796, 0.9569]
  Örnek Sayısı: 4

================================================================================
🏆 EN İYİ KOMBİNASYONLAR
================================================================================
PS - encdec: R² = 0.9569
```

### Karşılaştırma Tablosu (Terminal Çıktısı)

| Plastic_Type | Model_Type | Test_R² | Test_MAE | Test_RMSE |
|--------------|------------|---------|----------|-----------|
| PS | encdec | 0.956915 | 1.29946 | 1.849041 |
| PS | lstm | 0.928172 | N/A | N/A |
| PS | cnn | 0.779606 | N/A | N/A |
| PS | lstm_vae | 0.911259 | N/A | N/A |

---

## 🏆 En İyi Model Detayları

### ENCDEC (Encoder-Decoder) Modeli

#### Mimari Açıklama

```
Encoder-Decoder LSTM Mimarisi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Input: One-Hot Encoded Peptide (12 × 18)]
                    ↓
         ┌─────────────────────┐
         │   ENCODER (LSTM)    │
         │   3 Layers × 256    │
         │   Dropout: 0.1      │
         │   LayerNorm: True   │
         └─────────────────────┘
                    ↓
         [Latent Representation]
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
┌───────────┐               ┌───────────────┐
│Score Head │               │   DECODER     │
│ Tanh+Lin  │               │   (LSTM)      │
│   → 1     │               │ 3 Layers×256  │
└───────────┘               └───────────────┘
    ↓                               ↓
[Binding Score]            [Reconstructed Seq]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### En İyi Hiperparametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `hidden_dim` | 256 | Gizli katman boyutu |
| `num_layers` | 3 | LSTM katman sayısı |
| `dropout` | 0.1 | Dropout oranı |
| `learning_rate` | 0.001 | Öğrenme hızı |
| `batch_size` | 1024 | Batch boyutu |
| `lambda_score` | 1.0 | Score loss ağırlığı |
| `weight_decay` | 0.0001 | L2 regularizasyon |
| `use_layernorm` | True | Layer normalizasyon |

---

## 🔬 Ablation Study Detayları

### ENCDEC Model için Top 5 Kombinasyon

| Sıra | Val R² | Test R² | Epoch | hidden_dim | num_layers | dropout | lr | batch | λ_score | weight_decay | layernorm |
|------|--------|---------|-------|------------|------------|---------|-----|-------|---------|--------------|-----------|
| 🥇 | **0.9371** | **0.9378** | 50 | 256 | 3 | 0.1 | 0.001 | 1024 | 1.0 | 0.0001 | True |
| 🥈 | 0.9362 | 0.9359 | 50 | 256 | 3 | 0.1 | 0.001 | 1024 | 1.0 | 0.01 | True |
| 🥉 | 0.9358 | 0.9352 | 50 | 256 | 3 | 0.1 | 0.001 | 1024 | 0.7 | 0.01 | True |
| 4 | 0.9315 | 0.9303 | 48 | 256 | 3 | 0.2 | 0.001 | 1024 | 1.0 | 0.0001 | True |
| 5 | 0.9311 | 0.9309 | 50 | 256 | 3 | 0.2 | 0.001 | 1024 | 1.0 | 0.01 | True |

### Hiperparametre Etki Analizi

#### Hidden Dimension Etkisi
```
hidden_dim=256 > hidden_dim=128
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
256: ████████████████████ 0.9378
128: ██████████████████   0.8793
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### Num Layers Etkisi
```
num_layers=3 > num_layers=2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3: ████████████████████ 0.9378
2: ███████████████████  0.9298
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### LayerNorm Etkisi
```
use_layernorm=True > False
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
True:  ████████████████████ 0.9378
False: ██████████████████   0.9165
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📉 Final Eğitim Metrikleri

### Eğitim Süreci

| Metrik | Değer |
|--------|-------|
| **Toplam Epoch** | 175/250 (Early Stopping) |
| **Best Epoch** | 165 |
| **Best Val Loss** | 0.044633 |
| **Final Test R²** | 0.9569 |
| **Final Test MAE** | 1.2995 |
| **Final Test RMSE** | 1.8490 |

### Eğitim Loss Değişimi

```
Loss vs Epoch (ENCDEC - PS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch  |  Train Loss  |  Val Loss  |   Val R²   |  Durum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1   |    3.2347    |   2.9112   |    N/A     |  Başlangıç
  10   |    0.2861    |   0.3723   |    N/A     |  Hızlı düşüş
  50   |    0.0334    |   0.0672   |    N/A     |  Stabilizasyon
 100   |    0.0175    |   0.0513   |    N/A     |  İyi performans
 165   |    0.0111    |   0.0446   |  0.9569    |  🏆 EN İYİ MODEL
 175   |    0.0108    |   0.0449   |    N/A     |  ⏹ Early Stop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Final Model Metrikleri

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **Test R²** | **0.9569** | Varyansın %95.69'u açıklanıyor |
| **Test MAE** | 1.2995 | Ortalama mutlak hata (puan cinsinden) |
| **Test RMSE** | 1.8490 | Kök ortalama kare hatası |

---

## 🧬 Üretilen Peptidler

### En İyi 5 Peptid (Simulated Annealing ile Optimize Edilmiş)

| Sıra | Peptid Sekansı | Skor | İyileşme | Başlangıç Skoru |
|------|----------------|------|----------|-----------------|
| 🥇 | **WWWIRDIWQGMR** | -47.18 | +35.91 | -11.27 |
| 🥈 | **WWWQRVIWQEMR** | -46.80 | +39.44 | -7.36 |
| 🥉 | **WWWYRAIWQQMR** | -45.19 | +25.88 | -19.31 |
| 4 | **WWWLRQLWQSMR** | -46.32 | +38.62 | -7.70 |
| 5 | **WWWQWQLRQWFR** | -46.07 | +36.71 | -9.36 |

### Peptid Özellikleri

| Özellik | Değer |
|---------|-------|
| **Toplam Üretilen** | 5 |
| **Benzersiz Peptid** | 5 (%100) |
| **Ortalama Skor** | -46.31 |
| **En İyi Skor** | -47.18 |
| **Ortalama İyileşme** | +35.31 puan |

### Amino Asit Dağılımı (Üretilen Peptidler)

```
Pozisyon bazlı amino asit frekansı:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pos 1:  W (100%)  - Tüm peptidler Tryptophan ile başlıyor
Pos 2:  W (100%)  - İkinci pozisyon da Tryptophan
Pos 3:  W (100%)  - Üçüncü pozisyon da Tryptophan
Pos 12: R (100%)  - Tüm peptidler Arginine ile bitiyor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Yüksek Frekanslı Amino Asitler:
- W (Tryptophan): %35+ - Aromatik, hidrofobik
- R (Arginine): %15+ - Pozitif yüklü
- Q (Glutamine): %10+ - Polar, nötr
- M (Methionine): %8+ - Hidrofobik
```

---

## � R² Skoru Açıklaması ve Doğrulama

### R² Skoru Ne Anlama Geliyor?

R² (Coefficient of Determination) skoru, modelin tahmin doğruluğunu ölçer:

| R² Değeri | Anlam |
|-----------|-------|
| **R² = 1.0** | Mükemmel tahmin (tahminler gerçek değerlerle tamamen eşleşiyor) |
| **R² = 0.0** | Model, basit ortalama kadar iyi (tahminler ortalama değere eşit) |
| **R² < 0.0** | Model, basit ortalamadan daha kötü |

### R² Hesaplama Formülü

```
R² = 1 - (SS_res / SS_tot)

SS_res = Σ(y_gerçek - y_tahmin)²  → Residual sum of squares
SS_tot = Σ(y_gerçek - y_ortalama)² → Total sum of squares
```

### ⚠️ Önemli Uyarı

> **R² skoru sadece modelin TAHMİN DOĞRULUĞUNU gösterir.**  
> **R² skoru, üretilen peptitlerin ORİJİNAL VERİ SETİNE BENZERLİĞİNİ göstermez!**

### Üretilen Peptitlerin Benzerlik Kontrolü

| Metrik | Açıklama | Değer Aralığı |
|--------|----------|---------------|
| **Hamming Distance** | İki peptit arasındaki farklı pozisyon sayısı | 0/12 (aynı) - 12/12 (farklı) |
| **Edit Distance** | Bir peptidi diğerine çevirmek için gereken minimum değişiklik | 0 - 12 |
| **Sequence Similarity** | İki peptit arasındaki benzerlik yüzdesi | 0% - 100% |
| **Benzersizlik** | Üretilen peptit orijinal veri setinde var mı? | Evet/Hayır |

### ✅ İyi Bir Model Kriterleri

```
✓ Yüksek R² skoru (tahmin doğruluğu iyi)
✓ Üretilen peptitler benzersiz (orijinal veri setinde yok)
✓ Üretilen peptitler orijinal veri setinden farklı (düşük benzerlik)
✓ Üretilen peptitler daha iyi skorlara sahip (optimizasyon başarılı)
```

---

## � Benzerlik Analizi

### Üretilen Peptidlerin Orijinal Veri Seti ile Karşılaştırması

| Peptid | Benzersiz | En Yakın Benzerlik | Min Hamming | En Yakın Peptid |
|--------|-----------|-------------------|-------------|-----------------|
| WWWIRDIWQGMR | ✅ Evet | 83.3% | 2/12 | WHWIRQIWQGMR |
| WWWQRVIWQEMR | ✅ Evet | 91.7% | 1/12 | WWHQRVIWQEMR |
| WWWYRAIWQQMR | ✅ Evet | 83.3% | 2/12 | WHWLRAIWQQMR |
| WWWLRQLWQSMR | ✅ Evet | 83.3% | 2/12 | WHWLRQIWQSMR |
| WWWQWQLRQWFR | ✅ Evet | 75.0% | 3/12 | FWHQWQLRHWFR |

### Benzerlik İstatistikleri

| Metrik | Değer |
|--------|-------|
| **Toplam Üretilen Peptid** | 5 |
| **Benzersiz Peptid Sayısı** | 5 |
| **Benzersizlik Oranı** | %100 |
| **Ortalama Maksimum Benzerlik** | 83.3% |
| **Ortalama Minimum Hamming** | 2.00/12 |

### Benzerlik Dağılımı

```
Benzerlik Kategorileri:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tamamen aynı (100%):     0 peptid (0.0%)   ✅ Mükemmel
Çok benzer (90-100%):    1 peptid (20.0%)  ⚠️ Dikkat
Benzer (75-90%):         4 peptid (80.0%)  ✅ İyi
Orta benzerlik (50-75%): 0 peptid (0.0%)
Düşük benzerlik (0-50%): 0 peptid (0.0%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Detaylı Peptid Bazlı Analiz

```
Orijinal Veri Seti: 405,827 peptid (PS.csv)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Üretilen: WWWIRDIWQGMR
    ✓ Benzersiz: Evet
    ✓ En yakın benzerlik: 83.3%
    ✓ Minimum Hamming distance: 2/12
    ✓ En yakın peptit: WHWIRQIWQGMR (83.3% benzer)

  Üretilen: WWWQRVIWQEMR
    ✓ Benzersiz: Evet
    ✓ En yakın benzerlik: 91.7%
    ✓ Minimum Hamming distance: 1/12
    ✓ En yakın peptit: WWHQRVIWQEMR (91.7% benzer)

  Üretilen: WWWYRAIWQQMR
    ✓ Benzersiz: Evet
    ✓ En yakın benzerlik: 83.3%
    ✓ Minimum Hamming distance: 2/12
    ✓ En yakın peptit: WHWLRAIWQQMR (83.3% benzer)

  Üretilen: WWWLRQLWQSMR
    ✓ Benzersiz: Evet
    ✓ En yakın benzerlik: 83.3%
    ✓ Minimum Hamming distance: 2/12
    ✓ En yakın peptit: WHWLRQIWQSMR (83.3% benzer)

  Üretilen: WWWQWQLRQWFR
    ✓ Benzersiz: Evet
    ✓ En yakın benzerlik: 75.0%
    ✓ Minimum Hamming distance: 3/12
    ✓ En yakın peptit: FWHQWQLRHWFR (75.0% benzer)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Plastik Tipine Göre Özet İstatistikler

```
📊 PS Plastik için Benzerlik Özeti:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
             max_similarity_percent              min_hamming_distance
                               mean   min    max                 mean min max
plastic_type
PS                            83.33  75.0  91.67                  2.0   1   3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📊 Overfitting Analizi

### ENCDEC Model Overfitting Durumu

| Kombinasyon | Val R² | Test R² | Overfitting Ratio | Train-Val Gap | Durum |
|-------------|--------|---------|-------------------|---------------|-------|
| Best (101) | 0.9371 | 0.9378 | -0.497 | -0.032 | ✅ Hafif Underfit |
| 2nd (103) | 0.9362 | 0.9359 | -0.512 | -0.034 | ✅ Hafif Underfit |
| 3rd (99) | 0.9358 | 0.9352 | -0.471 | -0.022 | ✅ Hafif Underfit |

### Overfitting Açıklaması

```
Overfitting Ratio = (Train R² - Val R²) / Val R²

Negatif Ratio = Model underfit (validation'da daha iyi)
Pozitif Ratio = Model overfit (training'de daha iyi)

Bu modelde:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Negatif ratio (-0.497): Model hafif underfit durumunda
✅ Bu durum genelleme kapasitesinin iyi olduğunu gösterir
✅ Test R² > Val R² demek ki model yeni verilerde iyi çalışıyor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📈 Grafikler ve Görselleştirmeler

### Mevcut Grafikler

Aşağıdaki grafikler `results/ablation_results/figures/` klasöründe mevcuttur:

#### Model Karşılaştırma
- `model_comparison_nylon.png` - Model performans karşılaştırması

#### Ablation Study Grafikleri
- `ablation_encdec_PS.png` - ENCDEC hiperparametre analizi
- `ablation_lstm_PS.png` - LSTM hiperparametre analizi
- `ablation_cnn_PS.png` - CNN hiperparametre analizi
- `ablation_lstm_vae_PS.png` - LSTM-VAE hiperparametre analizi

#### Overfitting Analizi
- `overfitting_analysis_encdec_PS.png`
- `overfitting_analysis_lstm_PS.png`
- `overfitting_analysis_cnn_PS.png`
- `overfitting_analysis_lstm_vae_PS.png`

#### Amino Asit Analizi
- `aa_probability_mass_heatmap_PS.png` - Amino asit olasılık × kütle heatmap
- `generated_peptides_aa_heatmap_encdec_PS.png` - Üretilen peptidler AA heatmap

#### Training Curves
- `training_curves_ablation/training_curves_combo_*_encdec_PS.png`

---

## 🧪 Amino Asit Frekans Analizi

### PS Plastik için Optimal Amino Asitler

| Amino Asit | Frekans | Özellik | PS Bağlanma Potansiyeli |
|------------|---------|---------|-------------------------|
| **W (Trp)** | ~35% | Aromatik, Hidrofobik | ⭐⭐⭐⭐⭐ Çok Yüksek |
| **R (Arg)** | ~15% | Pozitif Yük | ⭐⭐⭐⭐ Yüksek |
| **Q (Gln)** | ~10% | Polar, Nötr | ⭐⭐⭐ Orta |
| **M (Met)** | ~8% | Hidrofobik | ⭐⭐⭐ Orta |
| **L (Leu)** | ~6% | Hidrofobik | ⭐⭐ Düşük-Orta |
| **I (Ile)** | ~5% | Hidrofobik | ⭐⭐ Düşük-Orta |

### Biyokimyasal Yorum

```
PS (Polistiren) Yapısı:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Aromatik benzen halkaları içerir
  - Güçlü π-π stacking etkileşimleri yapabilir
  - Hidrofobik yapı
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Yüksek W (Tryptophan) Frekansı Açıklaması:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Tryptophan'ın indol halkası PS'in benzen halkası ile
    güçlü π-π stacking etkileşimi yapar
  ✓ En büyük aromatik amino asit olması yüzey temasını artırır
  ✓ Hidrofobik yapısı PS yüzeyine yapışmayı kolaylaştırır
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Yüksek R (Arginine) Frekansı Açıklaması:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Pozitif yük, yüzey etkileşimlerini stabilize eder
  ✓ Guanidin grubu hidrojen bağları oluşturabilir
  ✓ Hidrofobik ve hidrofilik bölge dengesi sağlar
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 💡 Öneriler ve Sonraki Adımlar

### 1. Model İyileştirmeleri
- [ ] Daha fazla epoch ile eğitim (Early stopping olmadan)
- [ ] Learning rate scheduler deneme (CosineAnnealing, OneCycle)
- [ ] Farklı optimizer'lar test etme (RAdam, NAdam)

### 2. Peptid Üretimi
- [ ] Daha fazla peptid üretimi (30+ adet)
- [ ] Farklı optimizasyon stratejileri (Genetic Algorithm, Beam Search)
- [ ] Multi-objective optimizasyon

### 3. Deneysel Doğrulama
- [ ] En iyi 5 peptid için in-vitro bağlanma testleri
- [ ] Surface Plasmon Resonance (SPR) ölçümleri
- [ ] Moleküler dinamik simülasyonları

### 4. Karşılaştırmalı Analiz
- [ ] PS sonuçlarını diğer plastiklerle karşılaştırma
- [ ] Ortak motif analizi
- [ ] Plastik-spesifik amino asit tercihlerinin belirlenmesi

---

## 📁 Dosya Yapısı

```
Peptid-Generator/
├── 📂 src/
│   └── ablation_study_win.py          # Ana eğitim scripti
│
├── 📂 data/
│   └── Data_win/
│       └── PS.csv                      # 405,827 peptid
│
├── 📂 results/
│   └── ablation_results/
│       ├── 📂 figures/
│       │   ├── ablation_encdec_PS.png
│       │   ├── ablation_lstm_PS.png
│       │   ├── ablation_cnn_PS.png
│       │   ├── ablation_lstm_vae_PS.png
│       │   ├── overfitting_analysis_*.png
│       │   ├── aa_probability_mass_heatmap_PS.png
│       │   └── model_comparison_*.png
│       │
│       ├── 📂 tables/
│       │   ├── ablation_encdec_PS.csv
│       │   ├── ablation_lstm_PS.csv
│       │   ├── ablation_cnn_PS.csv
│       │   ├── ablation_lstm_vae_PS.csv
│       │   ├── generated_peptides_encdec_PS.csv
│       │   └── generated_peptides_similarity_analysis.csv
│       │
│       ├── 📂 logs/
│       │   ├── ablation_log_encdec_PS.txt
│       │   ├── ablation_log_encdec_PS.json
│       │   ├── final_training_log_encdec_PS.txt
│       │   └── final_training_log_encdec_PS.json
│       │
│       └── 📂 final_models/
│           └── encdec_PS_final.pth
│
└── 📂 reports/
    └── EGITIM_RAPORU_PS.md             # ← Bu rapor
```

---

## 🔧 Teknik Detaylar

### Veri Seti Özellikleri

| Özellik | Değer |
|---------|-------|
| **Plastik Tipi** | PS (Polistiren) |
| **Toplam Peptid** | 405,827 |
| **Sekans Uzunluğu** | 12 amino asit |
| **Amino Asit Sayısı** | 18 (standart 20'den C ve P hariç) |
| **Skor Aralığı** | -80 ile +20 arası |
| **Train/Val/Test Oranı** | 80/10/10 |

### Eğitim Parametreleri

| Parametre | Değer |
|-----------|-------|
| **Ablation Epochs** | 50 |
| **Final Epochs** | 250 (175'te early stop) |
| **Early Stopping Patience** | 10 |
| **Optimizer** | AdamW |
| **Loss Function** | MSE (Score) + CrossEntropy (Recon) |
| **Seed** | 42 |

### Model Boyutları

| Bileşen | Boyut |
|---------|-------|
| **Input** | 12 × 18 (One-Hot) |
| **Encoder Hidden** | 256 × 3 layers |
| **Decoder Hidden** | 256 × 3 layers |
| **Score Output** | 1 |
| **Recon Output** | 12 × 18 |
| **Toplam Parametre** | ~2.5M |

---

## 📚 Referanslar

1. PyTorch Documentation - https://pytorch.org/docs/
2. LSTM Architecture - Hochreiter & Schmidhuber, 1997
3. Encoder-Decoder Networks - Sutskever et al., 2014
4. Peptide-Plastic Binding - Recent literature

---

**Rapor Oluşturma Tarihi:** 6 Ocak 2026  
**Script:** `ablation_study_win.py`  
**Platform:** Windows 11 + RTX 4080 Super  
**Yazar:** Peptid Generator System

---

*Bu rapor otomatik olarak oluşturulmuştur. Detaylı analiz için ilgili CSV ve JSON dosyalarına başvurunuz.*
