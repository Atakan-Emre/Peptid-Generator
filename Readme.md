# 🧬 Peptid Generator - Mikroplastik Bağlayıcı Peptid Tasarımı

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Derin öğrenme tabanlı mikroplastik bağlayıcı peptid tasarım sistemi. 6 farklı plastik tipi için optimize edilmiş ENCDEC (Encoder-Decoder) modeli ile yüksek afinite peptidler üretir.

---

## 📊 Proje Özeti

| Metrik | Değer |
|--------|-------|
| **Desteklenen Plastikler** | PET, PP, PE, PVC, PS, Nylon |
| **Model Mimarileri** | LSTM, CNN, LSTM-VAE, ENCDEC |
| **Test Edilen Kombinasyon** | 4,056 |
| **Ortalama Test R²** | **0.9588** (%95.88) |
| **En İyi R²** | 0.9766 (PET) |
| **Üretilen Peptid** | 180 (30 × 6 plastik) |
| **Peptid Özgünlüğü** | %100 |

---

## 📁 Proje Yapısı

```
Peptid-Generator/
│
├── 📂 src/                          # Kaynak Kodlar
│   ├── ablation_study_win.py        # Windows ana eğitim scripti (RTX 4080)
│   ├── ablation_study_mlx.py        # Mac MLX eğitim scripti
│   ├── ablation_study_mac.py        # Mac PyTorch scripti
│   ├── peptide_generation_comparison.py # SA, ILS, MOCO-CEM karşılaştırması
│   ├── analiz_raporu.py             # Analiz raporu oluşturma
│   ├── durum_raporu.py              # Durum raporu oluşturma
│   └── optimization/                # ILS ve MOCO-CEM optimizasyon kodları
│
├── 📂 data/                         # Veri Dosyaları
│   ├── Data_win/                    # Windows uyumluluk/veri klasörü
│   └── sortingData/                 # PET, PP, PE, PVC, PS, Nylon CSV veri setleri
│
├── 📂 results/                      # Eğitim Sonuçları
│   ├── ablation_results_PET/        # PET sonuçları
│   │   ├── figures/                 # Grafikler
│   │   ├── tables/                  # CSV tabloları
│   │   ├── logs/                    # Eğitim logları
│   │   └── final_models/            # Eğitilmiş modeller
│   ├── ablation_results_PP_ENCDEC/  # PP ENCDEC sonuçları
│   ├── ablation_results_PP_LSTM/    # PP LSTM sonuçları
│   ├── ablation_results_PE/         # PE sonuçları
│   ├── ablation_results_PVC/        # PVC sonuçları
│   ├── ablation_results_PS/         # PS sonuçları
│   ├── ablation_results_NYLON/      # Nylon sonuçları
│   └── peptide_generation_comparison/ # Üretim yöntemi karşılaştırmaları
│
├── 📂 reports/                      # Eğitim Raporları
│   ├── EGITIM_RAPORU_GENEL.md       # 📌 Kapsamlı birleşik rapor
│   ├── EGITIM_RAPORU_PET.md         # PET detaylı raporu
│   ├── EGITIM_RAPORU_PP_ENCDEC.md   # PP ENCDEC raporu
│   ├── EGITIM_RAPORU_PP_LSTM.md     # PP LSTM raporu
│   ├── EGITIM_RAPORU_PE.md          # PE detaylı raporu
│   ├── EGITIM_RAPORU_PVC.md         # PVC detaylı raporu
│   ├── EGITIM_RAPORU_PS.md          # PS detaylı raporu
│   └── EGITIM_RAPORU_NYLON.md       # Nylon detaylı raporu
│
├── README.md                        # 📌 Bu dosya
├── .gitignore                       # Git ignore kuralları
└── LICENSE                          # Lisans dosyası
```

---

## 🏆 Sonuçlar

### Plastik Bazında Performans

| Plastik | Model | Test R² | MAE | RMSE | Veri Boyutu |
|---------|-------|---------|-----|------|-------------|
| **PET** | ENCDEC | **0.9766** | 1.48 | 2.20 | 441,978 |
| **PP** | ENCDEC | 0.9583 | 1.53 | 2.10 | 433,487 |
| **Nylon** | ENCDEC | 0.9576 | 1.75 | 2.46 | 142,614 |
| **PS** | ENCDEC | 0.9569 | 1.30 | 1.85 | 405,827 |
| **PE** | ENCDEC | 0.9547 | 1.51 | 2.17 | 715,508 |
| **PVC** | ENCDEC | 0.9490 | 1.71 | 2.40 | 208,608 |

### Veri Seti Yapısı

| Plastik | CSV Dosyası | Satır Sayısı | Benzersiz Sequence | Tekrarlı Sequence |
|---------|-------------|--------------|--------------------|-------------------|
| **PET** | `data/sortingData/PET.csv` | 441,978 | 381,065 | 60,913 |
| **PP** | `data/sortingData/PP.csv` | 433,487 | 433,487 | 0 |
| **PE** | `data/sortingData/PE.csv` | 715,508 | 715,508 | 0 |
| **PVC** | `data/sortingData/PVC.csv` | 208,608 | 208,608 | 0 |
| **PS** | `data/sortingData/PS.csv` | 405,827 | 405,827 | 0 |
| **Nylon** | `data/sortingData/Nylon.csv` | 142,614 | 142,614 | 0 |

> 📝 **Not:** Her CSV dosyası `Sequence` ve `Score` kolonlarından oluşur. PET veri setinde aynı sequence için tekrarlı satırlar bulunmaktadır.

### En İyi Üretilen Peptidler

| Plastik | Peptid | Skor | İyileştirme |
|---------|--------|------|-------------|
| **Nylon** | YWRMMDWWLRWW | -78.79 | +55.62 |
| **PVC** | HWNWIQNFMFIR | -69.17 | +49.92 |
| **PET** | WEWWFGFHHRLR | -66.11 | +54.89 |
| **PE** | AHYHFLWHQMRW | -60.77 | +51.38 |
| **PP** | WWQRRAFFWRMW | -55.53 | +38.44 |
| **PS** | WHWQREIWQSMR | -45.34 | +36.34 |

> 📝 **Not:** Daha negatif skor = Daha iyi bağlanma afinitesi

---

## 🚀 Kurulum

### Gereksinimler

```bash
# Python 3.10+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

### Donanım Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---------|---------|----------|
| GPU | NVIDIA GTX 1080 | RTX 4080 Super |
| VRAM | 8 GB | 16 GB |
| RAM | 16 GB | 64 GB |
| CPU | 4 Core | 8+ Core |

---

## 💻 Kullanım

### 1. Ablation Study Çalıştırma

```bash
cd src
python ablation_study_win.py
```

### 2. Peptid Üretim Karşılaştırması

```bash
cd src
python peptide_generation_comparison.py
```

### 3. Analiz Raporu Oluşturma

```bash
cd src
python analiz_raporu.py
```

---

## 🧠 Model Mimarileri

### ENCDEC (Encoder-Decoder) - En İyi Model

```
[Input: One-Hot Peptide (12 × 18)]
            ↓
    ┌─────────────┐
    │   Encoder   │ LSTM 2-3 Layer, hidden=256
    │   LayerNorm │ Dropout=0.1
    └─────────────┘
            ↓
      [Latent Vector]
            ↓
    ┌───────┴───────┐
    ↓               ↓
┌─────────┐   ┌───────────┐
│ Decoder │   │   Score   │
│  LSTM   │   │ Predictor │
└─────────┘   └───────────┘
    ↓               ↓
[Reconstruction] [Predicted Score]

Loss = MSE(score) + λ·CrossEntropy(reconstruction)
```

### Neden ENCDEC En İyi?

1. **Reconstruction Loss** - Overfitting'i doğal olarak azaltır
2. **Dual-Task Learning** - Genelleme kapasitesini artırır
3. **Information Bottleneck** - Sadece önemli özellikleri korur
4. **LayerNorm** - Eğitim stabilitesi sağlar

---

## 📈 Eğitim Süreci

### Ablation Study (50 Epoch)
- 4 model × hiperparametre kombinasyonları
- Toplam 4,056 aktif kombinasyon
- Val R² bazında en iyi model seçimi

### Final Training (250 Epoch)
- Early stopping (patience=10)
- Mixed precision training (AMP)
- Best model checkpoint kaydetme

---

## 📊 Görselleştirmeler

Tüm grafikler `results/ablation_results_*/figures/` klasöründe:

| Grafik Türü | Açıklama |
|-------------|----------|
| `training_curves_*.png` | Epoch vs Loss/R² |
| `model_comparison_heatmap_*.png` | Model performans karşılaştırması |
| `aa_probability_mass_heatmap_*.png` | AA × Kütle olasılık dağılımı |
| `generated_peptides_aa_heatmap_*.png` | Üretilen peptidlerin AA dağılımı |
| `overfitting_analysis_*.png` | Overfitting metrik analizi |
| `ablation_*.png` | Ablation study sonuçları |

---

## 🔬 Amino Asit Frekans Analizi

Üretilen peptidlerde en sık görülen amino asitler:

| AA | Frekans | Özellik | Neden Tercih Ediliyor? |
|----|---------|---------|------------------------|
| **W** | ~35% | Aromatik | π-π stacking, büyük hidrofobik yüzey |
| **R** | ~22% | Pozitif | Hidrojen bağları, elektrostatik |
| **H** | ~15% | His-tag | pH hassas, π stacking |
| **F** | ~13% | Aromatik | Hidrofobik etkileşim |
| **K** | ~10% | Pozitif | Elektrostatik bağlanma |

---

## 📚 Raporlar

| Rapor | Açıklama | Konum |
|-------|----------|-------|
| **Kapsamlı Rapor** | Tüm plastiklerin birleşik analizi | [`reports/EGITIM_RAPORU_GENEL.md`](reports/EGITIM_RAPORU_GENEL.md) |
| PET Raporu | PET detaylı analizi | [`reports/EGITIM_RAPORU_PET.md`](reports/EGITIM_RAPORU_PET.md) |
| PP ENCDEC Raporu | PP ENCDEC modeli | [`reports/EGITIM_RAPORU_PP_ENCDEC.md`](reports/EGITIM_RAPORU_PP_ENCDEC.md) |
| PE Raporu | PE detaylı analizi | [`reports/EGITIM_RAPORU_PE.md`](reports/EGITIM_RAPORU_PE.md) |
| PVC Raporu | PVC detaylı analizi | [`reports/EGITIM_RAPORU_PVC.md`](reports/EGITIM_RAPORU_PVC.md) |
| PS Raporu | PS detaylı analizi | [`reports/EGITIM_RAPORU_PS.md`](reports/EGITIM_RAPORU_PS.md) |
| Nylon Raporu | Nylon detaylı analizi | [`reports/EGITIM_RAPORU_NYLON.md`](reports/EGITIM_RAPORU_NYLON.md) |

---

## 🔧 Hiperparametreler

### ENCDEC Optimal Parametreler

| Parametre | PET | PP | PE | PVC | PS | Nylon |
|-----------|-----|----|----|-----|----|-------|
| `hidden_dim` | 256 | 256 | 256 | 256 | 256 | 256 |
| `num_layers` | 3 | 2 | 3 | 3 | 3 | 3 |
| `dropout` | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| `lambda_score` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `batch_size` | 1024 | 1024 | 1024 | 1024 | 1024 | 128 |
| `learning_rate` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |

---

## 🖥️ Sistem Bilgileri

### Eğitim Platformu

| Bileşen | Değer |
|---------|-------|
| **OS** | Windows 11 Pro |
| **GPU** | NVIDIA RTX 4080 Super (16GB) |
| **CPU** | AMD Ryzen 9 9700X |
| **RAM** | 64GB DDR5-6000 |
| **CUDA** | 12.1 |
| **cuDNN** | 8.9.2 |
| **PyTorch** | 2.1.0+cu121 |

### GPU Optimizasyonları

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()
```

---

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 👨‍💻 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'i push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

---

## 📞 İletişim

- **GitHub:** [Atakan-Emre](https://github.com/Atakan-Emre)
- **Proje:** [Peptid-Generator](https://github.com/Atakan-Emre/Peptid-Generator)

---

## 🙏 Teşekkürler

Bu proje, plastik kirliliğiyle mücadele için biyolojik çözümler geliştirme amacıyla yapılmıştır.

---

*Son Güncelleme: 27 Nisan 2026*
