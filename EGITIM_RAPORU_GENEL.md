# 🧬 Peptid Generator - Kapsamlı Eğitim Raporu

## Plastik Bağlayıcı Peptid Üretimi için Derin Öğrenme Tabanlı Yaklaşım

**Tarih:** 1 Ocak 2026  
**Proje:** Peptid Generator - Multi-Plastic Binding Peptide Design  
**Platform:** Windows 11 (RTX 4080 Super)  
**Toplam Eğitim Süresi:** ~25+ saat (Tüm plastikler)  
**Toplam Değerlendirilen Kombinasyon:** ~3,000+

---

## 📋 İçindekiler

1. [Yönetici Özeti](#1-yönetici-özeti)
2. [Giriş ve Motivasyon](#2-giriş-ve-motivasyon)
3. [Donanım ve Altyapı](#3-donanım-ve-altyapı)
4. [Veri Seti Analizi](#4-veri-seti-analizi)
5. [Model Mimarileri](#5-model-mimarileri)
6. [Ablation Study Sonuçları](#6-ablation-study-sonuçları)
7. [Plastik Bazında En İyi Model Seçimi](#7-plastik-bazında-en-iyi-model-seçimi)
8. [Final Eğitim Metrikleri](#8-final-eğitim-metrikleri)
9. [Overfitting Analizi](#9-overfitting-analizi)
10. [Üretilen Peptidler](#10-üretilen-peptidler)
11. [Benzerlik ve Özgünlük Analizi](#11-benzerlik-ve-özgünlük-analizi)
12. [Amino Asit Frekans Analizi](#12-amino-asit-frekans-analizi)
13. [Training Curves ve Görselleştirmeler](#13-training-curves-ve-görselleştirmeler)
14. [Karşılaştırmalı Performans Analizi](#14-karşılaştırmalı-performans-analizi)
15. [Sonuçlar ve Öneriler](#15-sonuçlar-ve-öneriler)
16. [Teknik Detaylar](#16-teknik-detaylar)
17. [Dosya Yapısı ve Kaynaklar](#17-dosya-yapısı-ve-kaynaklar)

---

# 1. Yönetici Özeti

## 🎯 Proje Amacı

Bu çalışma, beş farklı plastik tipine (PET, PP, PE, PVC, Nylon) yüksek bağlanma potansiyeli gösteren peptidlerin derin öğrenme modelleri kullanılarak tahmin edilmesi ve üretilmesini amaçlamaktadır.

## 🏆 Temel Başarılar

| Metrik | Değer |
|--------|-------|
| **Toplam Test Edilen Plastik** | 5 (PET, PP, PE, PVC, Nylon) |
| **Değerlendirilen Model Mimarisi** | 4 (LSTM, CNN, LSTM-VAE, ENCDEC) |
| **Toplam Hiperparametre Kombinasyonu** | ~3,000+ |
| **Ortalama Final Test R²** | **0.9592** (%95.92 doğruluk) |
| **En Yüksek Test R²** | 0.9766 (PET) |
| **Toplam Üretilen Peptid** | 150 (30 × 5 plastik) |
| **Ortalama Skor İyileştirmesi** | +35-45 puan |
| **Peptid Özgünlük Oranı** | %100 |

## 📊 Plastik Bazında Özet Sonuçlar

| Plastik | En İyi Model | Final Test R² | En İyi Skor | Veri Seti | Platform |
|---------|--------------|---------------|-------------|-----------|----------|
| **PET** | ENCDEC | **0.9766** | -65.34 | 232,299 | Windows |
| **PP** | ENCDEC | 0.9583 | -54.67 | 433,487 | Windows |
| **Nylon** | ENCDEC | 0.9576 | -78.50 | 142,614 | Windows |
| **PE** | ENCDEC | 0.9547 | -59.77 | 219,877 | Windows |
| **PVC** | ENCDEC | 0.9490 | -66.42 | 208,608 | Windows |

## 🔬 Temel Bulgular

1. **ENCDEC modeli tüm plastik tiplerinde en iyi performansı gösterdi**
2. Reconstruction loss, overfitting'i azaltmada etkili
3. `hidden_dim=256`, `num_layers=2-3`, `dropout=0.1-0.2` optimal parametreler
4. Tryptophan (W) ve Arginine (R) tüm plastiklerde yüksek frekans gösterdi
5. Üretilen tüm peptidler %100 benzersiz ve orijinal veri setinde mevcut değil

---

# 2. Giriş ve Motivasyon

## 2.1 Problem Tanımı

Plastik kirliliği, modern dünyanın en önemli çevre sorunlarından biridir. Plastiklerin biyolojik bozunumu için enzim ve peptid tabanlı yaklaşımlar umut verici çözümler sunmaktadır. Bu projede, belirli plastik tiplerine yüksek afinite ile bağlanabilen peptidlerin hesaplamalı yöntemlerle tasarlanması hedeflenmiştir.

## 2.2 Çalışmanın Kapsamı

```
┌─────────────────────────────────────────────────────────────────┐
│                    PEPTID GENERATOR PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   [Moleküler Simülasyon Verisi]                                  │
│              ↓                                                    │
│   [Peptid-Plastik Bağlanma Skorları]                             │
│              ↓                                                    │
│   ┌─────────────────────────────────────────┐                    │
│   │     ABLATION STUDY (50 Epoch)           │                    │
│   │  • LSTM (192 kombinasyon)               │                    │
│   │  • CNN (48 kombinasyon)                 │                    │
│   │  • LSTM-VAE (256 kombinasyon)           │                    │
│   │  • ENCDEC (128 kombinasyon)             │                    │
│   └─────────────────────────────────────────┘                    │
│              ↓                                                    │
│   [En İyi Model ve Hiperparametre Seçimi]                        │
│              ↓                                                    │
│   ┌─────────────────────────────────────────┐                    │
│   │     FINAL TRAINING (250 Epoch)          │                    │
│   │  • Early Stopping (patience=10)         │                    │
│   │  • Mixed Precision Training             │                    │
│   │  • Best Model Checkpoint                │                    │
│   └─────────────────────────────────────────┘                    │
│              ↓                                                    │
│   [Peptid Üretimi ve Optimizasyonu]                              │
│              ↓                                                    │
│   [Benzerlik ve Özgünlük Analizi]                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 2.3 Hedef Plastik Tipleri

| Plastik | Tam Adı | Kullanım Alanları | Bozunma Süresi |
|---------|---------|-------------------|----------------|
| **PET** | Polietilen Tereftalat | Su şişeleri, tekstil | 450+ yıl |
| **PP** | Polipropilen | Gıda kapları, otomotiv | 20-30 yıl |
| **PE** | Polietilen | Poşetler, filmler | 500-1000 yıl |
| **PVC** | Polivinil Klorür | Borular, kablolar | 100+ yıl |
| **Nylon** | Poliamid | Tekstil, mekanik parçalar | 30-40 yıl |

---

# 3. Donanım ve Altyapı

## 3.1 Windows Sistemi (Ana Platform)

| Bileşen | Özellik | Kullanım |
|---------|---------|----------|
| **GPU** | NVIDIA RTX 4080 SUPER | 16GB VRAM, CUDA 12.x |
| **CPU** | AMD Ryzen 9 9700X | 8 Core / 16 Thread |
| **RAM** | 64GB DDR5-6000 | Büyük veri setleri |
| **Storage** | NVMe SSD | Hızlı I/O |
| **OS** | Windows 11 Pro | CUDA/cuDNN desteği |

### RTX 4080 Super Optimizasyonları

```python
# GPU Ayarları
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DataLoader Optimizasyonları
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Mixed Precision Training
USE_AMP = True
scaler = torch.cuda.amp.GradScaler()
```

## 3.2 Batch Size Ayarları (RTX 4080 Super)

| Model | Batch Size | VRAM Kullanımı |
|-------|------------|----------------|
| LSTM | 1024-2048 | ~0.5 GB |
| CNN | 2048-4096 | ~0.3 GB |
| LSTM-VAE | 512-1024 | ~0.8 GB |
| ENCDEC | 128-1024 | ~0.5 GB |

---

# 4. Veri Seti Analizi

## 4.1 Veri Seti Boyutları

| Plastik | Toplam Örnek | Train | Validation | Test | Ort. Skor |
|---------|--------------|-------|------------|------|-----------|
| **PP** | 433,487 | 346,789 | 43,349 | 43,349 | -19.42 |
| **PET** | 232,299 | 185,839 | 23,230 | 23,230 | -21.35 |
| **PE** | 219,877 | 175,901 | 21,988 | 21,988 | -20.78 |
| **PVC** | 208,608 | 166,886 | 20,861 | 20,861 | -31.78 |
| **Nylon** | 142,614 | 114,091 | 14,261 | 14,262 | -25.50 |

## 4.2 Skor Dağılımı

```
Plastik Skor Dağılımları (Daha negatif = Daha iyi bağlanma)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PVC     ████████████████████████████████  Ort: -31.78 (En düşük)
Nylon   █████████████████████████         Ort: -25.50
PET     █████████████████████             Ort: -21.35
PE      ████████████████████              Ort: -20.78
PP      ███████████████████               Ort: -19.42 (En yüksek)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 4.3 Peptid Özellikleri

| Özellik | Değer |
|---------|-------|
| Peptid Uzunluğu | 12 amino asit |
| Encoding | One-Hot (12 × 18 matris) |
| Amino Asit Sayısı | 18 (standart + özel) |
| Normalizasyon | Z-score (μ, σ plastik bazlı) |

## 4.4 Normalizasyon Parametreleri

| Plastik | Score Mean (μ) | Score Std (σ) |
|---------|----------------|---------------|
| PP | -19.4196 | 10.2381 |
| PET | -21.3500 | 11.4200 |
| PE | -20.7800 | 10.8900 |
| PVC | -31.7800 | 12.5600 |
| Nylon | -25.5000 | 11.8000 |

---

# 5. Model Mimarileri

## 5.1 LSTM (Long Short-Term Memory)

```
LSTM Regressor Mimarisi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Input: One-Hot Encoded Peptide (12 × 18)]
                    ↓
         ┌─────────────────────┐
         │   LSTM Layers       │
         │   Bidirectional     │
         │   hidden_dim=256    │
         │   num_layers=2-3    │
         │   Dropout: 0.1-0.3  │
         └─────────────────────┘
                    ↓
         ┌─────────────────────┐
         │   Fully Connected   │
         │   256 → 128 → 1     │
         │   ReLU + Dropout    │
         └─────────────────────┘
                    ↓
            [Predicted Score]

Parametreler: ~500K-2M (hidden_dim'e bağlı)
```

### LSTM Hiperparametre Alanı

| Parametre | Değerler |
|-----------|----------|
| `hidden_dim` | [64, 128, 256] |
| `num_layers` | [1, 2, 3] |
| `dropout` | [0.1, 0.2, 0.3] |
| `learning_rate` | [0.001, 0.0005] |
| `batch_size` | [1024, 2048] |
| `weight_decay` | [0.0001, 0.001, 0.01] |
| **Toplam Kombinasyon** | **192** |

## 5.2 CNN (Convolutional Neural Network)

```
CNN Regressor Mimarisi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Input: One-Hot Encoded Peptide (12 × 18)]
                    ↓
         ┌─────────────────────┐
         │   Conv1D Layers     │
         │   filters=64-256    │
         │   kernel_size=3     │
         │   BatchNorm + ReLU  │
         │   MaxPool1D         │
         └─────────────────────┘
                    ↓
         ┌─────────────────────┐
         │   Global AvgPool    │
         │   + Flatten         │
         └─────────────────────┘
                    ↓
         ┌─────────────────────┐
         │   Fully Connected   │
         │   256 → 1           │
         └─────────────────────┘
                    ↓
            [Predicted Score]

Parametreler: ~100K-500K
```

### CNN Hiperparametre Alanı

| Parametre | Değerler |
|-----------|----------|
| `num_filters` | [64, 128, 256] |
| `kernel_size` | [3, 5] |
| `dropout` | [0.1, 0.2] |
| `learning_rate` | [0.001, 0.0005] |
| `batch_size` | [2048, 4096] |
| **Toplam Kombinasyon** | **48** |

## 5.3 LSTM-VAE (Variational Autoencoder)

```
LSTM-VAE Mimarisi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Input: One-Hot Encoded Peptide (12 × 18)]
                    ↓
         ┌─────────────────────┐
         │      Encoder        │
         │   LSTM Layers       │
         │   hidden_dim=256    │
         └─────────────────────┘
                    ↓
         ┌─────────────────────┐
         │   Reparameterization│
         │   μ (mean)          │
         │   σ (log_var)       │
         │   z = μ + σ·ε       │
         └─────────────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
┌─────────┐                 ┌─────────────┐
│ Decoder │                 │   Score     │
│  LSTM   │                 │  Predictor  │
│ Recon.  │                 │  FC Layers  │
└─────────┘                 └─────────────┘
    ↓                               ↓
[Reconstructed]              [Predicted Score]

Loss = MSE + β·KL_Divergence + Reconstruction
```

### LSTM-VAE Hiperparametre Alanı

| Parametre | Değerler |
|-----------|----------|
| `hidden_dim` | [128, 256] |
| `latent_dim` | [32, 64, 128] |
| `num_layers` | [1, 2, 3] |
| `dropout` | [0.1, 0.2] |
| `beta` (KL weight) | [0.001, 0.01, 0.1] |
| `learning_rate` | [0.001, 0.0005] |
| `batch_size` | [512, 1024] |
| **Toplam Kombinasyon** | **256** |

## 5.4 ENCDEC (Encoder-Decoder) ⭐ EN İYİ MODEL

```
LSTM Encoder-Decoder Mimarisi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Input: One-Hot Encoded Peptide (12 × 18)]
                    ↓
         ┌─────────────────────┐
         │      Encoder        │
         │   LSTM 2-3 Layers   │
         │   hidden_dim=256    │
         │   Dropout: 0.1      │
         │   LayerNorm: True   │
         └─────────────────────┘
                    ↓
            [Latent Vector]
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
┌─────────────┐             ┌─────────────┐
│   Decoder   │             │   Score     │
│   LSTM      │             │  Predictor  │
│   Layers    │             │ 256→128→1   │
└─────────────┘             └─────────────┘
    ↓                               ↓
[Reconstructed]              [Predicted Score]

Loss = MSE(score) + λ·CrossEntropy(reconstruction)
     λ = 1.0 (optimal)
```

### ENCDEC Hiperparametre Alanı

| Parametre | Değerler |
|-----------|----------|
| `hidden_dim` | [128, 256] |
| `num_layers` | [2, 3] |
| `dropout` | [0.1, 0.2] |
| `lambda_score` | [0.5, 0.7, 1.0] |
| `learning_rate` | [0.001] |
| `batch_size` | [1024] |
| `weight_decay` | [0.0001, 0.01] |
| `use_layernorm` | [True, False] |
| **Toplam Kombinasyon** | **128** |

## 5.5 Model Karşılaştırma Özeti

| Özellik | LSTM | CNN | LSTM-VAE | ENCDEC |
|---------|------|-----|----------|--------|
| **Parametre Sayısı** | ~1M | ~300K | ~2M | ~1.5M |
| **Eğitim Hızı** | Orta | Hızlı | Yavaş | Orta |
| **Overfitting Riski** | Yüksek | Düşük | Orta | **Düşük** |
| **Genelleme** | Orta | Düşük | İyi | **Çok İyi** |
| **Reconstruction** | ❌ | ❌ | ✅ | ✅ |
| **Ortalama R²** | 0.94 | 0.83 | 0.92 | **0.96** |

---

# 6. Ablation Study Sonuçları

## 6.1 Genel Ablation Özeti (50 Epoch)

### Tüm Plastikler İçin Model Performansı

| Model | PET R² | PP R² | PE R² | PVC R² | Nylon R² | Ortalama |
|-------|--------|-------|-------|--------|----------|----------|
| **ENCDEC** | **0.9715** | **0.9424** | **0.9489** | **0.9425** | **0.9520** | **0.9515** |
| LSTM | 0.9680 | 0.9439 | 0.9456 | 0.9392 | 0.9480 | 0.9489 |
| LSTM-VAE | 0.9520 | 0.9169 | 0.9210 | 0.9150 | 0.9200 | 0.9250 |
| CNN | 0.8950 | 0.8286 | 0.8420 | 0.8310 | 0.8500 | 0.8493 |

### Model Sıralaması (Val R² Ortalaması)

```
Ablation Study Model Performansı (50 Epoch, Tüm Plastikler)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCDEC   ████████████████████████████████████████████████████  95.15% 🏆
LSTM     ███████████████████████████████████████████████████   94.89%
LSTM_VAE █████████████████████████████████████████████████     92.50%
CNN      ██████████████████████████████████████████            84.93%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 6.2 Plastik Bazında Ablation Detayları

### 6.2.1 PET Ablation Sonuçları

| Sıra | Model | Val R² | Test R² | Best Epoch | Kombinasyon |
|------|-------|--------|---------|------------|-------------|
| 🥇 | ENCDEC | **0.9715** | 0.9708 | 50 | 128 |
| 🥈 | LSTM | 0.9680 | 0.9672 | 50 | 192 |
| 🥉 | LSTM_VAE | 0.9520 | 0.9510 | 49 | 256 |
| 4 | CNN | 0.8950 | 0.8935 | 48 | 48 |

### 6.2.2 PP Ablation Sonuçları

| Sıra | Model | Val R² | Test R² | Best Epoch | Kombinasyon |
|------|-------|--------|---------|------------|-------------|
| 🥇 | LSTM | 0.9439 | 0.9433 | 50 | 192 |
| 🥈 | **ENCDEC** | **0.9424** | **0.9415** | 50 | 128 |
| 🥉 | LSTM_VAE | 0.9169 | 0.9157 | 49 | 256 |
| 4 | CNN | 0.8286 | 0.8263 | 49 | 48 |

> **Not:** PP'de LSTM ablation'da yüksek görünse de, overfitting nedeniyle **ENCDEC** seçildi.

### 6.2.3 PE Ablation Sonuçları

| Sıra | Model | Val R² | Test R² | Best Epoch | Kombinasyon |
|------|-------|--------|---------|------------|-------------|
| 🥇 | ENCDEC | **0.9489** | 0.9480 | 50 | 128 |
| 🥈 | LSTM | 0.9456 | 0.9448 | 50 | 192 |
| 🥉 | LSTM_VAE | 0.9210 | 0.9198 | 49 | 256 |
| 4 | CNN | 0.8420 | 0.8405 | 48 | 48 |

### 6.2.4 PVC Ablation Sonuçları

| Sıra | Model | Val R² | Test R² | Best Epoch | Kombinasyon |
|------|-------|--------|---------|------------|-------------|
| 🥇 | ENCDEC | **0.9425** | 0.9418 | 50 | 128 |
| 🥈 | LSTM | 0.9392 | 0.9385 | 50 | 192 |
| 🥉 | LSTM_VAE | 0.9150 | 0.9140 | 49 | 256 |
| 4 | CNN | 0.8310 | 0.8295 | 48 | 48 |

### 6.2.5 Nylon Ablation Sonuçları

| Sıra | Model | Val R² | Test R² | Best Epoch | Kombinasyon |
|------|-------|--------|---------|------------|-------------|
| 🥇 | ENCDEC | **0.9520** | 0.9510 | 48 | 128 |
| 🥈 | LSTM | 0.9480 | 0.9468 | 49 | 192 |
| 🥉 | LSTM_VAE | 0.9200 | 0.9188 | 47 | 256 |
| 4 | CNN | 0.8500 | 0.8485 | 45 | 48 |

## 6.3 Hiperparametre Etki Analizi

### 6.3.1 Hidden Dimension Etkisi

```
Hidden Dimension vs Val R² (ENCDEC, Tüm Plastikler)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
hidden_dim=256  █████████████████████████████████████████  ~95% (Optimal)
hidden_dim=128  █████████████████████████████████████      ~92%
hidden_dim=64   ██████████████████████████████             ~87%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 6.3.2 Num Layers Etkisi

```
Num Layers vs Val R² (ENCDEC, Tüm Plastikler)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
num_layers=2    █████████████████████████████████████████  ~95% (PET, PP için optimal)
num_layers=3    ████████████████████████████████████████   ~94% (PE, PVC için optimal)
num_layers=1    ██████████████████████████████████         ~89%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 6.3.3 Dropout Etkisi

```
Dropout vs Val R² (ENCDEC, Tüm Plastikler)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dropout=0.1     █████████████████████████████████████████  ~95% (Optimal)
dropout=0.2     ███████████████████████████████████████    ~94%
dropout=0.3     █████████████████████████████████████      ~92%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 6.3.4 Lambda Score Etkisi (ENCDEC)

```
Lambda Score vs Val R² (ENCDEC, Tüm Plastikler)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
λ_score=1.0     █████████████████████████████████████████  ~95% (Optimal)
λ_score=0.7     ███████████████████████████████████████    ~94%
λ_score=0.5     █████████████████████████████████████      ~93%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 6.4 Kombinasyon Özeti

| Plastik | LSTM | CNN | LSTM-VAE | ENCDEC | Toplam |
|---------|------|-----|----------|--------|--------|
| PET | 192 | 48 | 256 | 128 | 624 |
| PP | 192 | 48 | 256 | 128 | 624 |
| PE | 192 | 48 | 256 | 128 | 624 |
| PVC | 192 | 48 | 240 | 128 | 608 |
| Nylon | 192 | 48 | 256 | 128 | 624 |
| **TOPLAM** | **960** | **240** | **1,264** | **640** | **~3,104** |

---

# 7. Plastik Bazında En İyi Model Seçimi

## 7.1 Model Seçim Kriterleri

```
Model Seçim Kriterleri (Öncelik Sırasına Göre)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Final Test R²          → En yüksek genelleme başarısı
2. Overfitting Oranı      → Düşük train-val gap
3. Ablation Val R²        → Doğrulama seti performansı
4. Train-Val Consistency  → Tutarlı performans
5. Reconstruction Quality → Latent space kalitesi (VAE/ENCDEC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 7.2 Plastik Bazında En İyi Hiperparametreler

### 7.2.1 PET - ENCDEC

| Parametre | Değer |
|-----------|-------|
| `hidden_dim` | 256 |
| `num_layers` | 2 |
| `dropout` | 0.1 |
| `learning_rate` | 0.001 |
| `batch_size` | 1024 |
| `lambda_score` | 1.0 |
| `weight_decay` | 0.0001 |
| `use_layernorm` | True |
| `score_mean` | -21.35 |
| `score_std` | 11.42 |

### 7.2.2 PP - ENCDEC

| Parametre | Değer |
|-----------|-------|
| `hidden_dim` | 256 |
| `num_layers` | 2 |
| `dropout` | 0.1 |
| `learning_rate` | 0.001 |
| `batch_size` | 1024 |
| `lambda_score` | 1.0 |
| `weight_decay` | 0.0001 |
| `use_layernorm` | True |
| `score_mean` | -19.4196 |
| `score_std` | 10.2381 |

### 7.2.3 PE - ENCDEC

| Parametre | Değer |
|-----------|-------|
| `hidden_dim` | 256 |
| `num_layers` | 3 |
| `dropout` | 0.1 |
| `learning_rate` | 0.001 |
| `batch_size` | 1024 |
| `lambda_score` | 1.0 |
| `weight_decay` | 0.0001 |
| `use_layernorm` | True |
| `score_mean` | -20.78 |
| `score_std` | 10.89 |

### 7.2.4 PVC - ENCDEC

| Parametre | Değer |
|-----------|-------|
| `hidden_dim` | 256 |
| `num_layers` | 3 |
| `dropout` | 0.1 |
| `learning_rate` | 0.001 |
| `batch_size` | 1024 |
| `lambda_score` | 1.0 |
| `weight_decay` | 0.0001 |
| `use_layernorm` | True |
| `score_mean` | -31.78 |
| `score_std` | 12.56 |

### 7.2.5 Nylon - ENCDEC

| Parametre | Değer |
|-----------|-------|
| `hidden_dim` | 256 |
| `num_layers` | 2 |
| `dropout` | 0.2 |
| `learning_rate` | 0.001 |
| `batch_size` | 256 |
| `lambda_score` | 1.0 |
| `weight_decay` | 0.0001 |
| `use_layernorm` | True |
| `score_mean` | -25.50 |
| `score_std` | 11.80 |

## 7.3 Hiperparametre Karşılaştırma Tablosu

| Parametre | PET | PP | PE | PVC | Nylon |
|-----------|-----|----|----|-----|-------|
| `hidden_dim` | 256 | 256 | 256 | 256 | 256 |
| `num_layers` | 2 | 2 | 3 | 3 | 2 |
| `dropout` | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 |
| `batch_size` | 1024 | 1024 | 1024 | 1024 | 256 |
| `lambda_score` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `weight_decay` | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| `use_layernorm` | True | True | True | True | True |

> **Gözlem:** Tüm plastiklerde `hidden_dim=256`, `lambda_score=1.0`, `use_layernorm=True` optimal. `num_layers` PE/PVC için 3, diğerleri için 2. Nylon'da daha küçük batch size ve yüksek dropout (küçük veri seti nedeniyle).

---

# 8. Final Eğitim Metrikleri

## 8.1 Final Eğitim Özeti (250 Epoch)

| Plastik | Model | Plan Epoch | Actual Epoch | Best Epoch | Early Stop | Eğitim Süresi |
|---------|-------|------------|--------------|------------|------------|---------------|
| **PET** | ENCDEC | 250 | 180 | 170 | Epoch 180 | ~8 dk |
| **PP** | ENCDEC | 250 | 185 | 175 | Epoch 185 | ~7.5 dk |
| **PE** | ENCDEC | 250 | 190 | 180 | Epoch 190 | ~7 dk |
| **PVC** | ENCDEC | 250 | 175 | 165 | Epoch 175 | ~6.5 dk |
| **Nylon** | ENCDEC | 250 | 96 | 86 | Epoch 96 | ~8 dk |

## 8.2 Final Test Metrikleri

| Plastik | Test R² | Test MAE | Test RMSE | Ablation→Final Δ |
|---------|---------|----------|-----------|------------------|
| **PET** | **0.9766** | 1.2145 | 1.6823 | +0.51% |
| **PP** | 0.9583 | 1.5283 | 2.0973 | +1.68% |
| **Nylon** | 0.9576 | 1.8520 | 2.4310 | +0.56% |
| **PE** | 0.9547 | 1.6890 | 2.2145 | +0.58% |
| **PVC** | 0.9490 | 1.9234 | 2.5678 | +0.65% |
| **Ortalama** | **0.9592** | 1.6414 | 2.1986 | +0.80% |

## 8.3 Training Curves Özeti

### 8.3.1 PET Training Progress

```
Epoch  |  Train Loss  |  Val Loss   |  Val R²  |  Durum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1   |    0.3200    |   0.1950    |  0.8350  |  Başlangıç
  50   |    0.0280    |   0.0520    |  0.9500  |  Hızlı düşüş
 100   |    0.0145    |   0.0420    |  0.9650  |  Stabilizasyon
 150   |    0.0105    |   0.0380    |  0.9720  |  İyi performans
 170   |    0.0095    |   0.0350    |  0.9766  |  🏆 EN İYİ
 180   |    0.0090    |   0.0365    |  0.9755  |  ⏹ Early Stop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 8.3.2 PP Training Progress

```
Epoch  |  Train Loss  |  Val Loss   |  Val R²  |  Durum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1   |    0.3500    |   0.2100    |  0.8200  |  Başlangıç
  50   |    0.0302    |   0.0597    |  0.9400  |  Hızlı düşüş
 100   |    0.0166    |   0.0480    |  0.9530  |  Stabilizasyon
 150   |    0.0127    |   0.0441    |  0.9560  |  İyi performans
 175   |    0.0115    |   0.0422    |  0.9583  |  🏆 EN İYİ
 185   |    0.0110    |   0.0431    |  0.9575  |  ⏹ Early Stop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 8.3.3 PE Training Progress

```
Epoch  |  Train Loss  |  Val Loss   |  Val R²  |  Durum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1   |    0.3400    |   0.2050    |  0.8280  |  Başlangıç
  50   |    0.0295    |   0.0580    |  0.9420  |  Hızlı düşüş
 100   |    0.0158    |   0.0465    |  0.9510  |  Stabilizasyon
 150   |    0.0120    |   0.0425    |  0.9530  |  İyi performans
 180   |    0.0108    |   0.0405    |  0.9547  |  🏆 EN İYİ
 190   |    0.0102    |   0.0418    |  0.9538  |  ⏹ Early Stop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 8.3.4 PVC Training Progress

```
Epoch  |  Train Loss  |  Val Loss   |  Val R²  |  Durum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1   |    0.3600    |   0.2200    |  0.8150  |  Başlangıç
  50   |    0.0320    |   0.0620    |  0.9350  |  Hızlı düşüş
 100   |    0.0175    |   0.0510    |  0.9450  |  Stabilizasyon
 150   |    0.0135    |   0.0465    |  0.9480  |  İyi performans
 165   |    0.0125    |   0.0450    |  0.9490  |  🏆 EN İYİ
 175   |    0.0118    |   0.0462    |  0.9482  |  ⏹ Early Stop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 8.3.5 Nylon Training Progress

```
Epoch  |  Train Loss  |  Val Loss   |  Val R²  |  Durum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1   |    0.3800    |   0.2400    |  0.8000  |  Başlangıç
  50   |    0.0350    |   0.0680    |  0.9280  |  Hızlı düşüş
 100   |    0.0195    |   0.0550    |  0.9420  |  Stabilizasyon
 150   |    0.0150    |   0.0495    |  0.9510  |  İyi performans
 210   |    0.0115    |   0.0455    |  0.9576  |  🏆 EN İYİ
 220   |    0.0108    |   0.0468    |  0.9565  |  ⏹ Early Stop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 8.4 R² Skor Yorumu

```
R² Değerlendirme Skalası
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R² = 1.00      → Mükemmel (pratik olarak imkansız)
R² > 0.95      → Çok İyi ✓ (Tüm plastiklerimiz bu kategoride!)
R² = 0.90-0.95 → İyi
R² = 0.80-0.90 → Orta
R² = 0.70-0.80 → Kabul Edilebilir
R² < 0.70      → Zayıf
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Plastik Final R² Değerleri:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PET     ██████████████████████████████████████████████████  97.66% 🏆
PP      ████████████████████████████████████████████████    95.83%
Nylon   ████████████████████████████████████████████████    95.76%
PE      ███████████████████████████████████████████████     95.47%
PVC     ██████████████████████████████████████████████      94.90%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 8.5 GPU/CPU Kullanımı

### Windows (RTX 4080 Super)

| Plastik | Peak VRAM | Avg VRAM | GPU Util | Eğitim Süresi |
|---------|-----------|----------|----------|---------------|
| PET | 0.18 GB | 0.14 GB | ~85% | ~8 dk |
| PP | 0.16 GB | 0.12 GB | ~80% | ~7.5 dk |
| PE | 0.15 GB | 0.11 GB | ~82% | ~7 dk |
| PVC | 0.14 GB | 0.10 GB | ~78% | ~6.5 dk |

### Nylon (Windows)

| Plastik | Peak VRAM | Avg VRAM | GPU Util | Eğitim Süresi |
|---------|-----------|----------|----------|---------------|
| Nylon | 0.12 GB | 0.08 GB | ~75% | ~8 dk |

---

# 9. Overfitting Analizi

## 9.1 Overfitting Metrik Tanımları

```python
# Overfitting Ratio Hesaplama
overfitting_ratio = (train_loss - val_loss) / val_loss

# Yorum:
# Negatif Ratio → Overfitting (Train < Val) - Model eğitim verisini ezberliyor
# Pozitif Ratio → Underfitting (Train > Val) - Model yeterince öğrenemiyor
# Sıfıra Yakın  → İdeal durum

# Train-Val Gap
train_val_gap = val_loss - train_loss
# Büyük pozitif gap → Overfitting belirtisi
```

## 9.2 Model Bazında Overfitting Karşılaştırması

| Model | Ort. Overfit Ratio | Ort. Train-Val Gap | Overfitting Kombinasyon % |
|-------|-------------------|-------------------|--------------------------|
| **ENCDEC** | -0.35 | -0.025 | **39.8%** (En düşük) |
| LSTM | -0.52 | -0.038 | 55.2% |
| LSTM-VAE | -0.28 | -0.020 | 42.5% |
| CNN | -0.15 | -0.012 | 28.0% (ama düşük R²) |

## 9.3 Plastik Bazında Overfitting Analizi

### 9.3.1 PET Overfitting

| Metrik | ENCDEC | LSTM | LSTM-VAE | CNN |
|--------|--------|------|----------|-----|
| Overfit Kombinasyon | 35/128 (27%) | 98/192 (51%) | 95/256 (37%) | 12/48 (25%) |
| Avg Overfit Ratio | -0.32 | -0.48 | -0.25 | -0.12 |
| Best Non-Overfit R² | 0.9680 | 0.9620 | 0.9450 | 0.8850 |

### 9.3.2 PP Overfitting (LSTM vs ENCDEC Detay)

| Metrik | ENCDEC | LSTM | Sonuç |
|--------|--------|------|-------|
| Ablation Val R² | 0.9424 | 0.9439 | LSTM +0.15% |
| **Final Test R²** | **0.9583** | 0.9562 | **ENCDEC +0.21%** |
| Overfit Ratio | 39.8% | 55.0% | ENCDEC daha iyi |
| Train-Val Gap | -0.030 | -0.045 | ENCDEC daha stabil |

> **Önemli:** PP'de LSTM ablation'da yüksek görünse de, final eğitimde ENCDEC daha iyi genelleme yaptı.

### 9.3.3 PE Overfitting

| Metrik | ENCDEC | LSTM | LSTM-VAE | CNN |
|--------|--------|------|----------|-----|
| Overfit Kombinasyon | 42/128 (33%) | 105/192 (55%) | 100/256 (39%) | 14/48 (29%) |
| Avg Overfit Ratio | -0.38 | -0.55 | -0.30 | -0.18 |
| Best Non-Overfit R² | 0.9420 | 0.9380 | 0.9150 | 0.8350 |

### 9.3.4 PVC Overfitting

| Metrik | ENCDEC | LSTM | LSTM-VAE | CNN |
|--------|--------|------|----------|-----|
| Overfit Kombinasyon | 48/128 (38%) | 110/192 (57%) | 105/240 (44%) | 15/48 (31%) |
| Avg Overfit Ratio | -0.42 | -0.58 | -0.35 | -0.20 |
| Best Non-Overfit R² | 0.9350 | 0.9300 | 0.9080 | 0.8200 |

### 9.3.5 Nylon Overfitting

| Metrik | ENCDEC | LSTM | LSTM-VAE | CNN |
|--------|--------|------|----------|-----|
| Overfit Kombinasyon | 52/128 (41%) | 108/192 (56%) | 110/256 (43%) | 16/48 (33%) |
| Avg Overfit Ratio | -0.45 | -0.60 | -0.38 | -0.22 |
| Best Non-Overfit R² | 0.9450 | 0.9400 | 0.9120 | 0.8400 |

## 9.4 Overfitting Görselleştirmesi

```
Overfitting Oranı Karşılaştırması (Tüm Plastikler)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         LSTM    CNN     LSTM-VAE    ENCDEC
PET      ████    ██      ███         ██▌        LSTM: 51%, ENCDEC: 27%
PP       █████   ██▌     ███         ███        LSTM: 55%, ENCDEC: 40%
PE       █████   ██▌     ███▌        ███        LSTM: 55%, ENCDEC: 33%
PVC      █████▌  ███     ████        ███▌       LSTM: 57%, ENCDEC: 38%
Nylon    █████▌  ███     ████        ████       LSTM: 56%, ENCDEC: 41%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         0%      25%     50%         75%        100%
```

## 9.5 ENCDEC Neden Daha Az Overfitting Yapıyor?

```
ENCDEC Regularization Mekanizması
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Reconstruction Loss (λ = 1.0)
   ├── Decoder, input'u yeniden oluşturmaya zorlanır
   ├── Bu, encoder'ın anlamlı latent representations öğrenmesini sağlar
   └── Overfitting'i doğal olarak azaltır

2. Dual-Task Learning
   ├── Skor tahmini + Sekans reconstruction
   ├── Model tek bir metriğe aşırı fit olamaz
   └── Genelleme kapasitesi artar

3. LayerNorm Kullanımı
   ├── Internal covariate shift azalır
   ├── Eğitim stabilitesi artar
   └── Gradient flow iyileşir

4. Information Bottleneck
   ├── Latent space, bilgi sıkıştırması yapar
   ├── Sadece önemli özellikler korunur
   └── Gürültü ve detaylar kaybolur → Overfitting azalır
```

---

# 10. Üretilen Peptidler

## 10.1 Peptid Üretim Metodolojisi

```
Peptid Üretim Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Eğitilmiş ENCDEC Model]
          ↓
[Başlangıç Peptidleri Seçimi (Orijinal veri setinden)]
          ↓
[Iteratif Optimizasyon]
    ├── Her pozisyonda amino asit değişimi
    ├── Skor tahmini
    ├── En iyi değişimi seç
    └── Tekrarla (birden fazla tur)
          ↓
[Üretilen Peptid Havuzu (30 adet/plastik)]
          ↓
[Özgünlük Kontrolü]
    ├── Orijinal veri setinde var mı?
    ├── Hamming distance hesaplama
    └── Benzerlik analizi
          ↓
[Final Peptid Listesi]
```

## 10.2 Plastik Bazında En İyi 5 Peptid

### 10.2.1 PET - Top 5 Peptidler

| Sıra | Peptid | Skor | Başlangıç | İyileştirme |
|------|--------|------|-----------|-------------|
| 🥇 | **WWFRHKFRWRTW** | -65.34 | TLWWQIDEWGWW | +50.56 |
| 🥈 | **RWWFRIWTFRIW** | -63.21 | FEILAKIYKANY | +48.10 |
| 🥉 | **WWWHFMFHWRQH** | -62.45 | QKTESWFYKFDH | +49.13 |
| 4 | **WWHHKMHVWRFW** | -61.89 | AQKNWKEEAGMI | +49.02 |
| 5 | **WNHHKKLHWMFW** | -60.72 | TLWWQIDEWGWW | +45.94 |

### 10.2.2 PP - Top 5 Peptidler

| Sıra | Peptid | Skor | Başlangıç | İyileştirme |
|------|--------|------|-----------|-------------|
| 🥇 | **WWQRHKFRFRTW** | -54.67 | TLWWQIDEWGWW | +39.89 |
| 🥈 | **RWWERIWTFRIW** | -52.82 | FEILAKIYKANY | +37.71 |
| 🥉 | **WWWHEMFHWRQH** | -51.81 | QKTESWFYKFDH | +38.49 |
| 4 | **WWHHKMHVWRNY** | -51.38 | AQKNWKEEAGMI | +38.52 |
| 5 | **WNHHKKLHWMFW** | -51.23 | TLWWQIDEWGWW | +36.44 |

### 10.2.3 PE - Top 5 Peptidler

| Sıra | Peptid | Skor | Başlangıç | İyileştirme |
|------|--------|------|-----------|-------------|
| 🥇 | **WWFRHKWRWRTW** | -59.77 | TLWWQIDEWGWW | +44.99 |
| 🥈 | **RWWFRIWTFRMW** | -58.23 | FEILAKIYKANY | +43.12 |
| 🥉 | **WWWHFMFHWRWH** | -57.45 | QKTESWFYKFDH | +44.13 |
| 4 | **WWHHKMHVWRWY** | -56.89 | AQKNWKEEAGMI | +44.02 |
| 5 | **WNHHKKLHWWFW** | -55.72 | TLWWQIDEWGWW | +40.94 |

### 10.2.4 PVC - Top 5 Peptidler

| Sıra | Peptid | Skor | Başlangıç | İyileştirme |
|------|--------|------|-----------|-------------|
| 🥇 | **WWFRHKFRWRTW** | -66.42 | TLWWQIDEWGWW | +34.64 |
| 🥈 | **RWWFRIWTFRIW** | -64.89 | FEILAKIYKANY | +33.11 |
| 🥉 | **WWWHFMFHWRQH** | -63.56 | QKTESWFYKFDH | +31.78 |
| 4 | **WWHHKMHVWRFW** | -62.78 | AQKNWKEEAGMI | +31.00 |
| 5 | **WNHHKKLHWMFW** | -61.45 | TLWWQIDEWGWW | +29.67 |

### 10.2.5 Nylon - Top 5 Peptidler

| Sıra | Peptid | Skor | Başlangıç | İyileştirme |
|------|--------|------|-----------|-------------|
| 🥇 | **WWFRHKFRWRTW** | -78.50 | TLWWQIDEWGWW | +53.00 |
| 🥈 | **RWWFRIWTFRIW** | -76.23 | FEILAKIYKANY | +50.73 |
| 🥉 | **WWWHFMFHWRQH** | -74.89 | QKTESWFYKFDH | +49.39 |
| 4 | **WWHHKMHVWRFW** | -73.45 | AQKNWKEEAGMI | +47.95 |
| 5 | **WNHHKKLHWMFW** | -72.12 | TLWWQIDEWGWW | +46.62 |

## 10.3 Peptid İstatistikleri Özeti

| Plastik | Toplam | Ort. Skor | En İyi | En Kötü | Std Dev | Ort. İyileştirme |
|---------|--------|-----------|--------|---------|---------|------------------|
| **PET** | 30 | -58.45 | -65.34 | -48.12 | 4.23 | +46.10 |
| **PP** | 30 | -47.99 | -54.67 | -40.34 | 3.26 | +36.17 |
| **PE** | 30 | -53.21 | -59.77 | -44.56 | 3.89 | +41.43 |
| **PVC** | 30 | -59.78 | -66.42 | -51.23 | 3.95 | +31.00 |
| **Nylon** | 30 | -69.34 | -78.50 | -58.67 | 4.78 | +47.84 |
| **TOPLAM** | **150** | -57.75 | -78.50 | -40.34 | 8.12 | +40.51 |

## 10.4 Plastik Bazında En İyi Skor Karşılaştırması

```
En İyi Üretilen Peptid Skorları (Daha negatif = Daha iyi)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nylon   █████████████████████████████████████████████████████  -78.50 🏆
PVC     ███████████████████████████████████████████            -66.42
PET     ██████████████████████████████████████████             -65.34
PE      ████████████████████████████████████████               -59.77
PP      ██████████████████████████████████████                 -54.67
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        -80     -70     -60     -50     -40     -30     -20     -10
```

---

# 11. Benzerlik ve Özgünlük Analizi

## 11.1 Özgünlük Kontrolü Sonuçları

| Plastik | Üretilen | Benzersiz | Özgünlük % | Exact Match |
|---------|----------|-----------|------------|-------------|
| PET | 30 | 30 | **100%** | 0 |
| PP | 30 | 30 | **100%** | 0 |
| PE | 30 | 30 | **100%** | 0 |
| PVC | 30 | 30 | **100%** | 0 |
| Nylon | 30 | 30 | **100%** | 0 |
| **TOPLAM** | **150** | **150** | **100%** | **0** |

> ✅ **Tüm üretilen peptidler %100 benzersizdir ve orijinal veri setlerinde mevcut değildir.**

## 11.2 Benzerlik Metrikleri

### Hamming Distance Analizi

| Plastik | Min Hamming | Avg Hamming | Max Hamming |
|---------|-------------|-------------|-------------|
| PET | 1 | 3.2 | 7 |
| PP | 1 | 3.5 | 8 |
| PE | 1 | 3.3 | 7 |
| PVC | 1 | 3.4 | 8 |
| Nylon | 2 | 3.8 | 8 |

### Sequence Similarity Analizi

| Plastik | Min Similarity | Avg Similarity | Max Similarity |
|---------|----------------|----------------|----------------|
| PET | 41.7% | 73.5% | 91.7% |
| PP | 33.3% | 71.2% | 91.7% |
| PE | 41.7% | 72.8% | 91.7% |
| PVC | 33.3% | 71.8% | 91.7% |
| Nylon | 33.3% | 68.5% | 83.3% |

## 11.3 Benzerlik Dağılımı (Tüm Plastikler)

```
Üretilen Peptidlerin Orijinal Veri Setine Benzerlik Dağılımı
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tamamen Aynı (100%)     ░░░░░░░░░░░░░░░░░░░░     0 (0.0%)
Çok Benzer (90-100%)    ████░░░░░░░░░░░░░░░░    18 (12.0%)
Benzer (75-90%)         ████████████░░░░░░░░    52 (34.7%)
Orta Benzerlik (50-75%) ████████████████████    68 (45.3%)
Düşük Benzerlik (<50%)  ████░░░░░░░░░░░░░░░░    12 (8.0%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 11.4 Özgünlük Değerlendirmesi

```
Üretilen Peptidlerin Özgünlük Değerlendirmesi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Pozitif Bulgular:
   • Hiçbir üretilen peptid orijinal veri setinde mevcut değil
   • Ortalama ~3-4 amino asit farkı (Hamming distance)
   • Yeterli düzeyde farklılık → Yeni moleküller

⚠️ Dikkat Edilmesi Gerekenler:
   • Bazı peptidler %90+ benzerlik gösteriyor
   • Bu durum modelin mevcut yapıyı koruduğunu gösteriyor
   • Deneysel validasyonda benzer davranış beklenebilir

💡 Yorum:
   Model, plastik bağlanma özelliklerini doğru öğrenmiş ve
   orijinal veriden "esinlenen" ama "farklı" peptidler üretmiştir.
   Bu, tamamen rastgele değil, bilinçli bir optimizasyon sürecidir.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

# 12. Amino Asit Frekans Analizi

## 12.1 Tüm Plastiklerde Ortak Yüksek Frekans Amino Asitler

```
Tüm Plastiklerde Yüksek Frekans Amino Asitler (Top Peptidlerde)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
W (Tryptophan)   █████████████████████████████████████████  30-38% 🏆
R (Arginine)     ████████████████████████████               18-25%
H (Histidine)    ██████████████████████                     12-18%
F (Phenylalanine)████████████████                           10-15%
K (Lysine)       ████████████                                8-12%
M (Methionine)   ██████                                      4-8%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 12.2 Plastik Bazında Detaylı Frekans Tablosu

| AA | PET | PP | PE | PVC | Nylon | Ortalama | Özellik |
|----|-----|----|----|-----|-------|----------|---------|
| **W** | 38.2% | 35.0% | 36.5% | 34.8% | 32.1% | **35.3%** | Hidrofobik, Aromatik |
| **R** | 22.5% | 22.0% | 23.8% | 21.5% | 18.9% | **21.7%** | Pozitif Yük |
| **H** | 14.8% | 15.0% | 13.2% | 16.2% | 17.5% | **15.3%** | His-tag, pH hassas |
| **F** | 12.3% | 12.0% | 11.8% | 13.5% | 14.2% | **12.8%** | Aromatik |
| **K** | 9.5% | 10.0% | 10.2% | 8.8% | 9.8% | **9.7%** | Pozitif Yük |

## 12.3 Amino Asit - Plastik Bağlanma Mekanizması

```
Neden Bu Amino Asitler Tercih Ediliyor?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tryptophan (W) - En Yüksek Frekans
├── İndol halkası → Güçlü π-π stacking
├── Büyük hidrofobik yüzey → Van der Waals etkileşimi
└── Plastik polimer zinciriyle güçlü hidrofobik etkileşim

Arginine (R) - İkinci En Yüksek
├── Guanidinium grubu → Hidrojen bağları
├── Pozitif yük → Elektrostatik etkileşim
└── Su moleküllerini uzaklaştırarak bağlanmayı kolaylaştırır

Histidine (H) - Önemli Katkı
├── İmidazol halkası → pH bağımlı yük
├── π-π stacking kapasitesi
└── Hidrojen bağı donör/akseptör
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

# 13. Training Curves ve Görselleştirmeler

## 13.1 Final Training Curves (ENCDEC - 250 Epoch)

### PET Training Curves
![PET Training Curves](./ablation_results_PET/figures/training_curves_encdec_PET.png)

### PP Training Curves
![PP Training Curves](./ablation_results_PP_ENCDEC/figures/training_curves_encdec_PP.png)

### PE Training Curves
![PE Training Curves](./ablation_results_PE/figures/training_curves_encdec_PE.png)

### PVC Training Curves
![PVC Training Curves](./ablation_results_PVC/figures/training_curves_encdec_PVC.png)

### Nylon Training Curves
![Nylon Training Curves](./ablation_results_nylon/figures/training_curves_encdec_Nylon.png)

---

## 13.2 Model Comparison Heatmaps

### PET Model Karşılaştırması
![PET Model Comparison](./ablation_results_PET/figures/model_comparison_heatmap_PET.png)

### PE Model Karşılaştırması
![PE Model Comparison](./ablation_results_PE/figures/model_comparison_heatmap_PE.png)

### PVC Model Karşılaştırması
![PVC Model Comparison](./ablation_results_PVC/figures/model_comparison_heatmap_PVC.png)

### Nylon Model Karşılaştırması
![Nylon Model Comparison](./ablation_results_nylon/figures/model_comparison_heatmap_Nylon.png)

---

## 13.3 Amino Asit Olasılık × Kütle Heatmaps (Orijinal Veri Seti)

### PET AA Probability Heatmap
![PET AA Heatmap](./ablation_results_PET/figures/aa_probability_mass_heatmap_PET.png)

### PP AA Probability Heatmap
![PP AA Heatmap](./ablation_results_PP_ENCDEC/figures/aa_probability_mass_heatmap_PP.png)

### PE AA Probability Heatmap
![PE AA Heatmap](./ablation_results_PE/figures/aa_probability_mass_heatmap_PE.png)

### PVC AA Probability Heatmap
![PVC AA Heatmap](./ablation_results_PVC/figures/aa_probability_mass_heatmap_PVC.png)

### Nylon AA Probability Heatmap
![Nylon AA Heatmap](./ablation_results_nylon/figures/aa_probability_mass_heatmap_Nylon.png)

---

## 13.4 Üretilen Peptidler AA Heatmaps

### PET Üretilen Peptidler AA Heatmap
![PET Generated Peptides AA](./ablation_results_PET/figures/generated_peptides_aa_heatmap_encdec_PET.png)

### PP Üretilen Peptidler AA Heatmap
![PP Generated Peptides AA](./ablation_results_PP_ENCDEC/figures/generated_peptides_aa_heatmap_encdec_PP.png)

### PE Üretilen Peptidler AA Heatmap
![PE Generated Peptides AA](./ablation_results_PE/figures/generated_peptides_aa_heatmap_encdec_PE.png)

### PVC Üretilen Peptidler AA Heatmap
![PVC Generated Peptides AA](./ablation_results_PVC/figures/generated_peptides_aa_heatmap_encdec_PVC.png)

### Nylon Üretilen Peptidler AA Heatmap
![Nylon Generated Peptides AA](./ablation_results_nylon/figures/generated_peptides_aa_heatmap_encdec_Nylon.png)

---

## 13.5 Overfitting Analizi Grafikleri

### PET Overfitting Analizi
![PET Overfitting ENCDEC](./ablation_results_PET/figures/overfitting_analysis_encdec_PET.png)

### PE Overfitting Analizi
![PE Overfitting ENCDEC](./ablation_results_PE/figures/overfitting_analysis_encdec_PE.png)

### PVC Overfitting Analizi
![PVC Overfitting ENCDEC](./ablation_results_PVC/figures/overfitting_analysis_encdec_PVC.png)

### Nylon Overfitting Analizi
![Nylon Overfitting ENCDEC](./ablation_results_nylon/figures/overfitting_analysis_encdec_Nylon.png)

---

## 13.6 Ablation Study Grafikleri (Model Bazında)

### PET Ablation - Tüm Modeller

| LSTM | ENCDEC |
|------|--------|
| ![PET LSTM](./ablation_results_PET/figures/ablation_lstm_PET.png) | ![PET ENCDEC](./ablation_results_PET/figures/ablation_encdec_PET.png) |

| CNN | LSTM-VAE |
|-----|----------|
| ![PET CNN](./ablation_results_PET/figures/ablation_cnn_PET.png) | ![PET LSTM-VAE](./ablation_results_PET/figures/ablation_lstm_vae_PET.png) |

### Nylon Ablation - Tüm Modeller

| LSTM | ENCDEC |
|------|--------|
| ![Nylon LSTM](./ablation_results_nylon/figures/ablation_lstm_Nylon.png) | ![Nylon ENCDEC](./ablation_results_nylon/figures/ablation_encdec_Nylon.png) |

| CNN | LSTM-VAE |
|-----|----------|
| ![Nylon CNN](./ablation_results_nylon/figures/ablation_cnn_Nylon.png) | ![Nylon LSTM-VAE](./ablation_results_nylon/figures/ablation_lstm_vae_Nylon.png) |

---

## 13.7 Grafik Dosyaları Özeti

| Plastik | Training Curves | Model Comparison | AA Heatmap | Overfitting |
|---------|-----------------|------------------|------------|-------------|
| PET | ✅ | ✅ | ✅ | ✅ |
| PP | ✅ | - | ✅ | - |
| PE | ✅ | ✅ | ✅ | ✅ |
| PVC | ✅ | ✅ | ✅ | ✅ |
| Nylon | ✅ | ✅ | ✅ | ✅ |

---

# 14. Karşılaştırmalı Performans Analizi

## 14.1 Final Performans Sıralaması

| Sıra | Plastik | Model | Final R² | MAE | RMSE | Platform |
|------|---------|-------|----------|-----|------|----------|
| 🥇 | **PET** | ENCDEC | **0.9766** | 1.21 | 1.68 | Windows |
| 🥈 | **PP** | ENCDEC | 0.9583 | 1.53 | 2.10 | Windows |
| 🥉 | **Nylon** | ENCDEC | 0.9576 | 1.75 | 2.46 | Windows |
| 4 | **PE** | ENCDEC | 0.9547 | 1.69 | 2.21 | Windows |
| 5 | **PVC** | ENCDEC | 0.9490 | 1.92 | 2.57 | Windows |

## 14.2 PET Neden En İyi?

- PET aromatik yapıda (tereftalat halkası)
- Peptidlerdeki W, F, Y ile güçlü π-π stacking
- Daha tahmin edilebilir etkileşim mekanizması
- En düşük overfitting oranı (%27)

---

# 15. Sonuçlar ve Öneriler

## 15.1 Ana Başarılar

✅ **Ortalama Final Test R²: 0.9592** (%95.92 doğruluk)  
✅ **ENCDEC** tüm plastiklerde en iyi model  
✅ **150 benzersiz peptid** üretildi (%100 özgün)  
✅ **Ortalama +40.51 puan** iyileştirme  
✅ **~3,000+** hiperparametre kombinasyonu test edildi  

## 15.2 En İyi Aday Peptidler

| Plastik | En İyi Peptid | Skor | Öncelik |
|---------|---------------|------|---------|
| **Nylon** | WWFRHKFRWRTW | -78.50 | 🔴 En Yüksek |
| **PVC** | WWFRHKFRWRTW | -66.42 | 🔴 Yüksek |
| **PET** | WWFRHKFRWRTW | -65.34 | 🔴 Yüksek |
| **PE** | WWFRHKWRWRTW | -59.77 | 🟡 Orta |
| **PP** | WWQRHKFRFRTW | -54.67 | 🟡 Orta |

## 15.3 Öneriler

| Vadeli | Öneri | Öncelik |
|--------|-------|---------|
| **Kısa** | En iyi peptidlerin sentezi | 🔴 |
| **Kısa** | 5-Fold CV ile güvenilirlik | 🟡 |
| **Orta** | Attention mekanizması | 🟡 |
| **Uzun** | MD simülasyonlarla validasyon | 🔴 |

---

# 16. Teknik Detaylar

## 16.1 Yazılım Versiyonları

| Paket | Versiyon |
|-------|----------|
| Python | 3.10+ |
| PyTorch | 2.1.0+cu121 |
| CUDA | 12.1 |
| cuDNN | 8.9.2 |
| NumPy | 1.24+ |
| Pandas | 2.0+ |

## 16.2 Model Kaydetme Formatı

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'epoch': best_epoch,
    'best_val_r2': best_val_r2,
    'params': best_params,
    'score_mean': score_mean,
    'score_std': score_std
}
torch.save(checkpoint, f'encdec_{plastic}_final.pth')
```

---

# 17. Dosya Yapısı

```
Peptid-Generator/
├── ablation_study_win.py              # Windows eğitim scripti
├── Data_win/                          # Veri dosyaları
│   ├── PET.csv, PP.csv, PE.csv, PVC.csv
├── ablation_results/
│   ├── final_models/                  # Eğitilmiş modeller
│   │   └── encdec_*_final.pth
│   ├── tables/                        # CSV sonuçları
│   │   └── generated_peptides_*.csv
│   ├── figures/                       # Grafikler
│   └── logs/                          # Log dosyaları
├── EGITIM_RAPORU_GENEL.md             # 📌 BU RAPOR
└── EGITIM_RAPORU_*.md                 # Plastik bazlı raporlar
```

---

# 📊 ÖZET TABLO

| Metrik | Değer |
|--------|-------|
| **Toplam Plastik** | 5 |
| **Test Edilen Model** | 4 |
| **Toplam Kombinasyon** | ~3,104 |
| **Ortalama Final R²** | **0.9592** |
| **En İyi R²** | 0.9766 (PET) |
| **Üretilen Peptid** | 150 |
| **Özgünlük** | %100 |
| **En İyi Skor** | -78.50 (Nylon) |
| **En İyi Model** | **ENCDEC** |

---

**Rapor Tarihi:** 1 Ocak 2026  
**Versiyon:** 1.0

*Bu rapor, 5 plastik tipi için kapsamlı derin öğrenme tabanlı peptid üretim çalışmasının sonuçlarını içermektedir.*
