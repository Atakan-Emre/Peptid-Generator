📊 İçindekiler

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
11. [Öneriler ve Sonraki Adımlar](#öneriler-ve-sonraki-adımlar)
12. [Dosya Yapısı](#dosya-yapısı)
13. [Teknik Detaylar](#teknik-detaylar)

---

## li kapsamı bir şekldet edilmiştir. RTX 4080 Super GPU ile hızlandırılmış eğiim sürecinde **624 kombinasyon**değerlnir 📈 Performans Özeti

```
                    Ablation R²    Final R²    İyileşme
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCDEC              0.9415    →   0.9583      +1.68%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Önemli Not: yşamaktadı(0.9583 vs 0.9562) mişr Bu nedenle PP için **ENCDEC modeli** tercih edilmiştir.� Donanım ve Optimizasyon

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
| LSTM | 2048 | ~6 GB |
| CNN | 4096 | ~6-8 GB |
| LSTM-VAE | 1024 | ~10-12 GB |
| ENCDEC | 1024 | ~10-12 GB |

---

## �Bst Epoch | Kombinasyon | Eğim Süres|------|---------------------🥇 | 50 192|N/A🥈******5** | 0 128|~4:17:15🥉 | 49 | 256N/A | 49 | 48 | N/A

### Model Performans Grafiği

```
Val R²Karşılaştırması (Ablation Study)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LSTM     █████████████████████████████████████████████████  94.39%
ENCEC   ████████████████████████████████████████████████   94.24%
LSTM_VAE ████████████████████████████████████████████       91.69%
CNN      ██████████████████████████████████████            82.86%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
``` Final Test R²Karşılaştırması

```
 ile Final Eğitim SonrasıDetayları

###  (Encoder-Decoder) Modeli# Açıklama         │  LayerNorm:True│
     
                    ↓
         ┌─────────────────────┐
         │   Score Predictor   │
         │   FC: 256 → 128 → 1 ││ReLUActivation   │
     └─────────────────────┘
                 Predicted 
Loss = MSE(score) + λ × Reconstruction_Loss
     (λ = 1.0)
# Açıklama |---|------- | Gizli katman boyutu (Encoder/Decoderhidden size)  | LSTM katman sayısı(derinlik)  Dropout oranı (regularization) | Adam optimizer öğrenme hızı | | Mini-batchboyutu  | Skor loss ağırlığı L2 regularization (AdamW) |e | Layer normalization aktif |

#### Normalizasyon Parametrleri

| Parametre | Değer | Açıklama-----------|-------|----------|
| Ortalama skor (normalizasyon için) | | Standart sapma (normalizasyoniçin) #### Loss Fonksiyonu

```python
# EncoderDecoder Combined Loss
Loss = MSELoss(predicted_score, actual_score) + λ × ReconstructionLoss

# λ (lambda_score) = 1.0
# Reconstruction Loss: Cross-Entropy for peptide sequence reconstruction
# Optimizer: AdamW with weight_decay=0.0001
# Scheduler: ReduceLROnPlateau (optional)
```

---

## 🔬 Ablation Study Detayları

### ENCDEC Model için Top 5 Kombinasyon

| Sıra | Val R² | Test R² | Epoch | hidden_dim | num_layers | dropout | lr | batch | λ_score | weight_decay | layernorm |
|------|--------|---------|-------|------------|------------|---------|-----|-------|---------|--------------|-----------|
| 🥇 | **0.9424** | **0.9415** | 50 | 256 | 2 | 0.1 | 0.001 | 1024 | 1.0 | 0.0001 | True |
| 🥈 | 0.9422 | 0.9420 | 50 | 256 | 2 | 0.1 | 0.001 | 1024 | 1.0 | 0.01 | True |
| 🥉 | 0.9416 | 0.9404 | 50 | 256 | 3 | 0.1 | 0.001 | 1024 | 1.0 | 0.0001 | True |
| 4 | 0.9408 | 0.9404 | 50 | 256 | 3 | 0.1 | 0.001 | 1024 | 1.0 | 0.01 | True |
| 5 | 0.9398 | 0.9389 | 50 | 256 | 2 | 0.2 | 0.001 | 1024 | 1.0 | 0.01 | True |

### Hiperparametre Etki Analizi

#### Hidden Dimension Etkisi
```
hidden_dim=256  █████████████████████████████████████████  ~94% Val R²
hidden_dim=128  █████████████████████████████████████      ~91% Val R²
hidden_dim=64   ██████████████████████████████             ~86% Val R²
```

#### Num Layers Etkisi
```
num_layers=2    █████████████████████████████████████████  ~94% Val R² (Optimal!)
num_layers=3    ███████████████████████████████████████    ~94% Val R²
num_layers=1    ██████████████████████████████████         ~89% Val R²
```

#### Dropout Etkisi
```
dropout=0.1     █████████████████████████████████████████  ~94% Val R² (Optimal)
dropout=0.2     ███████████████████████████████████████    ~94% Val R²
dropout=0.3     █████████████████████████████████████      ~92% Val R²
```

### Kombinasyon Özeti

| Model | Toplam Kombinasyon | Tamamlanan | Başarı Oranı |
|-------|-------------------|------------|--------------|
| LSTM | 192 | 192 | 100% |
| CNN | 48 | 48 | 100% |
| LSTM_VAE | 256 | 256 | 100% |
| ENCDEC | 128 | 128 | 100% |
| **Toplam** | **624** | **624** | **100%** |

- Final (250 Epoch Planlı → 185 Epoch Earlytop) |
| Son Validation Loss 0.043068 || Patience | 10 epoch |
Cuv ÖzetiVal R²  | Duru━━━━━━━━━━━━━━   1   |  0.30    0.2100  |820  |  Başlangıç
  50   |    0.0940 |  Hızlı düşüş9530 |  Stabilizasyon9560  |  İyiperformans    |0.9583 EN İYİMODL0.9575  |   Early Stop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### GPU Kullanımı (Eğitimüresince)

```
GU Memory Kullanımı━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 50:   █░░░░░░░░░░░░░░░  0.08 GB / 16 GB (0.5%)
Epoch 100:  █░░░░░░░░░░░░░░░  0.11 GB / 16 GB (0.7%)
Epoch 150:  █░░░░░░░░░░░░░░░  0.14 GB / 16 GB (0.9%)
Epoch 175:  █░░░░░░░░░░░░░░░  0.14 GB / 16 GB (0.9%)
 | Açıklama-------|--- | Varyansın %95.83'ü açıklanıyor | Ortalama mutlakhata (puan cinsinden)  Kök ortalama kare hatası |

### R² Skor Yorumu

```
R² = 0.9583 anlamı:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Model, peptid skorlarındaki varyasyonun %95.83'ünü açıklıyor
✅ Tahmin edilen skorlar gerçek skorlarla çok yüksek korelasyon
✅ Bu değer, biyoinformatik uygulamaları için çok iyi kabul edilir
✅ LSTM'den daha yüksek (+0.21%) genelleme başarısı

Karşılaştırma:
  R² = 1.00  → Mükemmel (pratik olarak imkansız)  R² > 0.95  → Çok İyi ✓ (Mevcut ENCDEC model)  R² > 0.90  → İyi
  R² > 0.80  → Orta
  R² < 0.70  → Zayıf
```

En Yüksek SkrluSekansı Peptid Başlangıç Skoru | --------------------------------|-14.78 | -15.11 | -13.32 | **** | -12.87**** | -14.78**** | -12.87**** | -15.28**** | -9.29**** | -12.33**** | -3.22 Peptid6 |
| Ortalama İyileştirme | +3.17 puan Analizi
Yüksek Frekans (PP-Tercih Edilen Amino Asitler):━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
.0  ← En yüksek.0.0.0.0
M (Methionine)   ██████                                 6.0%

Düşük Frekans (PP-Kaçınılan Amino Asitler):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
D (Aspartic Acid)   █                           1.0%
E (Glutamic Acid)   ██                          2.0%
S (Serine)          █                           1.0%
T (Threonine)       █                           1.0%
G (Glycine)         ██                          2.0%
```

### Önemli Motif Örüntüleri

```
Tespit Edilen Güçlü Motifler:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. "WW" başlangıcı  → 7/10 top peptitte (PP ile güçlü hidrofobik etkileşim)
2. "RW" / "IW" sonu → 8/10 top peptitte (pozitif yük + aromatik)
3. "HK" / "HH" orta → 6/10 top peptitte (histidine cluster)
4. "FW" dizisi      → 5/10 top peptitte (aromatik etkileşim)

Hipotetik Mekanizma:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Tryptophan (W): Hidrofobik etkileşim (PP metil gruplarıyla)• Arginine (R): Elektrostatik stabilizasyon
• Histidine (H): Hidrojen bağları ve π-stacking
• Phenylalanine (F): Aromatik etkileşim (PP polimer zinciriyle)
• Lysine (K): Pozitif yük, elektrostatik etkileşim


---

## 🔍 Benzerlik Analizi

### Üretilen Peptidlerin Özgünlüğü

| Metrik | Değer | Yorum |
|--------|-------|-------|
| **Benzersiz Peptid Oranı** | 100% (5/5) | Tüm test edilen peptidler özgün |
| **Ortalama Maks. Benzerlik** | ~72% | Orijinal veri setine orta benzerlik |
| **Ortalama Min. Hamming Distance** | ~3.5/12 | Ortalama 3-4 amino asit farkı |

### Detaylı Benzerlik Analizi (Top 5)

| Üretilen Peptid | Benzersiz | En Yakın Orijinal | Benzerlik | Hamming |
|-----------------|-----------|-------------------|-----------|---------|
| **WWQRHKFRFRTW** | ✅ Evet | WWQRHMFNFRTW | 83.3% | 2/12 |
| **RWWERIWTFRIW** | ✅ Evet | RWWERHWMFRIW | 75.0% | 3/12 |
| **WWWHEMFHWRQH** | ✅ Evet | WWWHEMFHWRQF | 91.7% | 1/12 || **WWHHKMHVWRNY** | ✅ Evet | WWHHKQHVWRNY | 91.7% | 1/12 || **WNHHKKLHWMFW** | ✅ Evet | WNHHKMLHWMFW | 91.7% | 1/12 |

### ✅ Benzerlik Sonuç Değerlendirmesi

> **Sonuç:** Üretilen peptidler orijinal veri setinde **mevcut değil** ve yüksek benzerlik gösteriyor.
> Bu durum, modelin:
> 1. PP bağlanma özelliklerini doğru öğrendiğini
> 2. Orijinal veriden **esinlenen** ama **farklı** peptidler ürettiğini
> 3. Tamamen rastgele değil, bilinçli değişiklikler yaptığını gösteriyor.

Model Oerfitting Durumu

| Metrik | Değer | Yorum |
|--------|-------|-------|
| Toplam Kombinayon| 128 | Test edilen hiperparametre seti |
| Overfitting Olan | 51 (39.8%) | Train < Val loss |
| Overfitting Olmayan | 77 (60.2%) | Train ≥ Val loss |

### ENCDEC vs Overfitting  Sonuç |-------|~55% | ENCEC diyiENCDEC daha iyi g
|Tran-ValGap DüşükYüksk| daha stabil |
| Rconstuton Loss | ✅ Var | ❌ Yok |k regularzaton |### Overfitting MetrikTnımlrı

```pthon
#Ovrfittig Ratio Hsapaa
ovrfittg_rtio =(ran_loss-vl_loss) /val_los

# Yorum:
# Ngatifatio→ (Train <Va)
# Pozitif R → UTrain > Val)
# ıfıra Yakın  → İalurum
```
### En İyiKombinasyonlnOvitting Duuu

| Sıra | Vl R² | Test R² | Overfittig Ratio |G | Durum |
|------|--------|---------|-------------------|---------------|-------|
| 1 | 0.9424 | 0.9415 | -0.495 | -0.030 |Hfif Overfit ✓ |
| 2 | 0.9422 | 0.9420 | -0.481 | -0.029 | Hfif Overfit ✓|| 3 | 0.9416 | 0.940 | -0.398 | -0025 | Minimal Overfit ✅ |

### Önemli Gözlem

> **ENCDEC içinAvantaj:  
> ENCDEC modeli rlss ayeindeeation sağlar.
> 
> **Bu ne anlam geliyor?**
> 1. Reconstruclos, modelin dah genel özellikler örenmesini teşvik eder
> 2. Encoder latent space'i daha anlamlı temsiller üretir
> 3. Bu durum, overfitting'i azatr ve genellemei atırır> 4. Final eğitimde ENCDEC'in LSTM'i geçmesinin nedeni budur
 ve GörselleştirmelerGrafik ı Konum |-------|Final e `figures/` |model_comparison_heatmap_PP.pn` | 4 modl karşılaştırma heatmap | `figures/` |
| `ablatio_ncdec_PP.png` | ENCDEC ablation sonuç gfikleri | `figures/` |
| `overfiting_analysis_ncecPP.ng` | ENCDEC ovrfitting analizi | `figures/` |
| `generated_epÜrtilen ep `figures/` |olasılık  `figures/` |
---

## 💡 Öneriler ve SonrakiAdılar

### 1. Deneysel Validasyn Önclikeri

| Önelik | Peptid | Skr | Öne |
|---------|--------|------|-------|
| 🔴 Yüksek | WWQRHKFRFRTW | -54.67 | İlk test adayı (en yüksek kr) |
| 🔴 Yüksek | RWWERIWTFRIW | -52.82 | Alteratif ady |
| 🟡 Or | WWWHEMFHWRQH | -5181 | Yüksek bezerlik, test edilebilir

### 2.del İyileştirme Önerileri

#### Kısa Vaeli (Hemn Uygulanabiir)
- [ ]Farlı random seed'lerle ensemble model
- [ ] 5-Fold Cross-vlidation
- [ ] Data augmentation

#### Ota Vadeli (1-2 Hafta)
- [ ] Attention mekanizmas ekeme
- [ ] Multi-tsk learning
- [ ] Negatif örnekleme stratejileri

#### Uzun Vadeli (1+ Ay)
- [ ] Üretilen peptidlerin sentezi ve deneysel tes
- [ ] Moleküle dinamik siülasyonlarlvalidasyon
Peptid-Generator/
├── ablation_study_win.py              ← Ana eğitim scripti
├── run_encdec_pp_final.py             ← ENCDEC final eğitim scripti
├── EGITIM_RAPORU_PP_ENCDEC.md         ← Bu rapor (ENCDEC - SEÇİLEN)├── EGITIM_RAPORU_PP_LSTM.md           ← LSTM raporu (referns)
├── Data_win/
│   └── PP.csv                         ← Eğitim verisi (433,487 peptid)
└── a        ├ 🏆SEÇİLEN model
    │   └── lstm_PP_final.pth         ← LSTM  (overfitting)                            └        json
```

---

## 🔬 Teknik Detaylar

### Kullanılan Teknoloiler

| Bileşen | Versiyon/Detay |
|---------|----------------|
| Python | 3.10+ |
| PyTorch | 2.x (CUDA backend) |
| CUDA | 12.x |
| cuDNN | 8.x |

### Eğitim Konfigürayu
python# Windows + RTX 4080 Super OptimizasyonNUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
USE_AMP = True

# CUDA Optimizasyonları
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

### Model Kaydetme Formatı

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_val_r2': best_val_r2,
    'params': best_params,
    'score_mean': score_mean,
    'score_std': score_std
}
torch.save(checkpoint, 'encdec_PP_final.pth')
```

� Notlar ve Uyarılar

### ⚠️ Dikkat Edilmesi Gerekenler

1. **Model Tipi:** PP için **ENCDEC** modeli kullanılmalıdır (LSTM değil!)

2. **Model Yükleme:** Model yüklerken aynı normalizasyon parametreleri kullanılmalı:
   - `score_mean = -19.4196`
   - `score_std = 10.2381`

3. **Peptid Uzunluğu:** Model 12 amino asitlik peptidler için eğitilmiştir.

4. **Skor Yorumu:** Negatif skorlar daha iyi bağlanma potansiyeli gösterir:
   - Daha negatif = Daha iyi
   - En iyi üretilen: -54.67
   - Orijinal veri ortalaması: -19.42

### ✅ Başarılar

- ✓ 4 farklı model mimarisi başarıyla karşılaştırıldı
- ✓ 624 hiperparametre kombinasyonu değerlendirildi
- ✓**%9.83 Test R²** ile çok yüksek doğruluk elde edildi
- ✓ **ENCDEC modeli**P için en iyi genelleme başarısı gösterdi
- ✓ 30 potansiyel peptid adayı üretildi
- ✓ Ortaama **+36.17 pun** iyileştirme ağlandı
- ✓ Tüm üretilen pepidler **%100 benzersz**

### 📊alı Özet (PP vs Diğer Pltikler)**PP********83**54.6Nylon7678.0

**Son Güncelleme:** 1 Ocak 2026  
**Platform:** Windows 11 + RTX 4080 Super ablation'da yüksek görünse deedilmi. ENCDEC fna eğitide %95.83 Test R² il aha iy genelleme yaptı