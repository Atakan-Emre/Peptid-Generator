# Reviewer compliance report

Each row is checked against the artefact that answers it: the file is opened and its contents verified, not merely its existence.

| # | Reviewer request | Status | Evidence | What the artefact shows |
| --- | --- | --- | --- | --- |
| R1.1 | Yorumlanabilirlik ayrintili tartisilmali | **met** | `analysis/interpretability.json`, `paper_tables/T8_position_importance.md`, `figures/F7_interpretability.png` | 6 polimer, her biri 12 pozisyon x 18 residu taramasi; en duyarli: PET P1 (|dR| = 5.91) |
| R1.2 | Onceki araclarla sistematik karsilastirma | **met** | `jain_baseline/jain_karsilastirma.json`, `paper_tables/T9_jain_benchmark.md`, `figures/F3_jain_benchmark.png` | 5 polimer x 3 kosul x 2 mimari, 3 tohum; ustunluk A kosulunda +0.1428, C kosulunda +0.0784 (%45 azalma) |
| R1.3 | Fizikokimyasal temel ve biyolojik anlam | **met** | `analysis/physicochemical.json`, `paper_tables/T7_physicochemical.md`, `figures/F8_physicochemical.png` | 6 polimer x 4+ ozellik, skorla korelasyon; aromatiklik korelasyonu polimere gore isaret degistiriyor (-0.47 ... +0.75) |
| R1.4 | Secili peptitlerin in vitro dogrulanmasi | answered in text | - | Deneysel olcum bu calismanin kapsami disinda. Cevapta kapsam olarak beyan edildi; yerine gecen hesapsal kanit R2.2 altinda |
| R2.1 | Jain et al. uzerine bilimsel katki + nicel kiyas | **met** | `jain_baseline/jain_karsilastirma.json`, `paper_tables/T9_jain_benchmark.md`, `figures/F3_jain_benchmark.png` | 5 polimer x 3 kosul x 2 mimari, 3 tohum; ustunluk A kosulunda +0.1428, C kosulunda +0.0784 (%45 azalma) |
| R2.2 | Bagimsiz dogrulama ve baseline karsilastirmasi | **met** | `analysis/independent_validation.json`, `analysis/generator_comparison.md`, `figures/F4_independent_validation.png` | 6/6 polimerde rastgele ve olculen en iyi %1 asiliyor; optimizasyonda kullanilmayan lstm modeli 6/6 dogruluyor; secilen uretici: encdec |
| R2.3a | Dizi kimligine gore kumeleme ve bolme | **met** | `prepared/leakage_summary.json`, `paper_tables/T2_split_comparison.md`, `figures/F2_split_effect.png` | kume bazli bolme uygulandi ve olculdu; rastgele bolmede test dizilerinin %37-%89'i egitime <=2 mutasyon uzaklikta; sisme 0.070-0.128 R2 |
| R2.3b | Coklu tohum, belirsizlik ve anlamlilik | **met** | `results/summary_clustered.json`, `results/architecture_stats_clustered.json`, `paper_tables/T4_architecture_significance.md`, `figures/F1_architecture_comparison.png` | her yapilandirma 5 tohum; 18 esli karsilastirmanin 18'i anlamli, hepsinde %95 GA ve Cohen d raporlaniyor; referans mimari: cnn |
| R2.3c | Her mimari/plastik icin train-val-test R2, RMSE, MAE | **met** | `paper_tables/T3_full_metrics_clustered.md`, `paper_tables/T3b_full_metrics_random.md` | 24+24 = 48 satir (6 polimer x 4 mimari x 2 bolme); her satirda train/val/test icin R2, RMSE, MAE ve standart sapma |
| R2.3d | Normalizasyon kaynagi + PS parametreleri | **met** | `prepared/<polimer>_clustered_seed42.json`, `pbp/data/prepare.py:181`, `paper_tables/T1_dataset_statistics.md` | normalizasyon yalnizca egitim bolmesinden; 6/6 polimerde .npz'den yeniden hesaplanarak dogrulandi. Hakemin eksik dedigi PS dahil hepsi var (PS: ort -15.3918, ss 8.8599) |
| R2.4 | Docking protokolu ve ML-docking korelasyonu | answered in text | `analysis/independent_validation.json` | Docking bagimsiz dogrulama olarak SUNULMUYOR; iddialar yumusatildi. Hakemin korelasyon talebi bu kosula bagliydi, dolayisiyla dusuyor. Bagimsiz dogrulamayi R2.2 tasiyor (6/6 polimerde bagimsiz model dogruluyor). Makaledeki docking protokol ayrintilari yazarlarca doldurulacak |
| R2.5 | Yenilik iddiasinin olculmesi / yumusatilmasi | **met** | `analysis/novelty.json`, `paper_tables/T6_novelty.md`, `figures/F6_novelty.png` | ikili birebir-esles testi yerine kimlik DAGILIMI raporlaniyor; en yakin komsu kimligi ortalama %55.6-%62.8, %90 uzerinde dizi orani %0.0 |
| R2.6 | Alti plastige karsi secicilik | **met** | `analysis/selectivity.json`, `paper_tables/T5_selectivity.md`, `figures/F5_selectivity.png` | 6x6 capraz skor matrisi; 6/6 peptit seti kendi hedefinde en iyi; secicilik indeksi 9.5 (PS) - 42.7 |

11 met, 0 awaiting input, 2 answered in text, 0 not met.
