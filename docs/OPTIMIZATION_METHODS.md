# Peptid Optimizasyonu için Metaheuristik Yöntemler

## Teknik Dokümantasyon

**Versiyon:** 1.0  
**Tarih:** Ocak 2026  
**Proje:** PeptidGenerator - Plastik Bağlayıcı Peptid Tasarımı

---

## 1. Giriş

Bu dokümantasyon, plastik yüzeylere bağlanma potansiyeli yüksek peptid dizileri üretmek için kullanılan üç farklı metaheuristik optimizasyon algoritmasını detaylı olarak açıklamaktadır.

### 1.1 Problem Tanımı

**Amaç:** 12 amino asitlik (12-mer) peptid dizileri için bağlanma skorunu minimize etmek.

**Arama Uzayı:**
- Alfabe boyutu: |Σ| = 18 amino asit (ADEFGHIKLMNQRSTVWY)
- Dizi uzunluğu: L = 12
- Toplam olası kombinasyon: 18¹² ≈ 1.15 × 10¹⁵

**Amaç Fonksiyonu:**
```
f(x) = ENCDEC_model(x) → ℝ
```
Burada x ∈ Σ¹² bir peptid dizisi ve f(x) tahmin edilen bağlanma skorudur (düşük = daha iyi).

### 1.2 Yöntem Karşılaştırma Özeti

| Özellik | SA | ILS | MOCO-CEM |
|---------|-----|-----|----------|
| **Paradigma** | Trajectory-based | Trajectory-based | Population-based |
| **Arama Stratejisi** | Stokastik yerel arama | Yerel arama + Pertürbasyon | Örnekleme + Elitizm |
| **Kabul Kriteri** | Metropolis | Greedy | Elite selection |
| **Escape Mekanizması** | Sıcaklık | Kick pertürbasyonu | Olasılık dağılımı |
| **Parallelizasyon** | Düşük | Orta | Yüksek |
| **Model Çağrısı/Peptid** | ~3,000 | ~48,000 | ~300,000 |

---

## 2. Simulated Annealing (SA)

### 2.1 Algoritma Teorisi

Simulated Annealing, metalurjideki tavlama işleminden esinlenen bir optimizasyon algoritmasıdır. Yüksek sıcaklıkta kötü çözümleri kabul ederek yerel minimumlardan kaçar, sıcaklık düştükçe daha seçici hale gelir.

### 2.2 Matematiksel Formülasyon

**Kabul Olasılığı (Metropolis Kriteri):**

```
P(accept) = {  1,                    if Δf < 0
            {  exp(-Δf / T),         if Δf ≥ 0
```

Burada:
- Δf = f(x') - f(x) : Enerji farkı
- T : Mevcut sıcaklık

**Soğutma Programı (Geometrik):**

```
T(k+1) = α × T(k)
```

Burada α ∈ (0, 1) soğutma katsayısı.

### 2.3 Hareket Operatörleri

```
┌─────────────────────────────────────────────────────────────┐
│                    MOVE OPERATOR                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   75% Substitution (Tek nokta mutasyonu)                    │
│   ═══════════════════════════════════════                   │
│                                                              │
│   Orijinal:  A E F G H I K L M N Q R                        │
│                    ↓                                         │
│   Mutant:    A E F W H I K L M N Q R                        │
│                  [pos=3, G→W]                                │
│                                                              │
│   25% Swap (İki pozisyon değişimi)                          │
│   ════════════════════════════════                          │
│                                                              │
│   Orijinal:  A E F G H I K L M N Q R                        │
│                ↓           ↓                                 │
│   Swapped:   A E F L H I K G M N Q R                        │
│                [pos1=3, pos2=7 swap]                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Algoritma Akış Diyagramı

```
                    ┌─────────────────┐
                    │  Başlangıç      │
                    │  x₀, T₀ = 0.5   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  x' = move(x)   │
                    │  Δf = f(x')-f(x)│
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │    Δf < 0 veya              │
              │    rand() < exp(-Δf/T) ?    │
              └──────────────┬──────────────┘
                     Evet    │    Hayır
                    ┌────────┴────────┐
                    │                 │
           ┌────────▼────────┐        │
           │   x = x'        │        │
           │   best güncelle │        │
           └────────┬────────┘        │
                    │                 │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │   iter % 75 == 0 ?          │
              │   T = T × 0.9               │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   T < 0.01 veya             │
              │   iter > 3000 ?             │
              └──────────────┬──────────────┘
                     Hayır   │    Evet
                    ┌────────┴────────┐
                    │                 │
                    │        ┌────────▼────────┐
                    │        │  Return best    │
                    │        └─────────────────┘
                    │
                    └────────────────────────────┐
                                                 │
                    ┌────────────────────────────┘
                    │
           ┌────────▼────────┐
           │  Sonraki iter   │
           └─────────────────┘
```

### 2.5 Hiperparametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `initial_temp` | 0.5 | Başlangıç sıcaklığı |
| `cooling_rate` | 0.9 | Geometrik soğutma katsayısı (α) |
| `max_iterations` | 3000 | Maksimum iterasyon sayısı |
| `cooling_interval` | 75 | Soğutma aralığı (iterasyon) |
| `min_temp` | 0.01 | Minimum sıcaklık (durma kriteri) |

### 2.6 Zaman Karmaşıklığı

```
O(max_iterations × model_inference_time)
≈ O(3000 × 1ms) ≈ 3 saniye/peptid
```

---

## 3. Iterated Local Search (ILS)

### 3.1 Algoritma Teorisi

ILS, yerel arama ile pertürbasyon mekanizmasını birleştiren bir metaheuristiktir. Yerel minimum bulunduğunda, "kick" operatörü ile çözüm uzayının farklı bölgelerine atlanır.

### 3.2 Algoritma Bileşenleri

```
┌───────────────────────────────────────────────────────────────────┐
│                         ILS FRAMEWORK                              │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐         │
│   │   Initial   │────▶│   Local     │────▶│   Accept    │         │
│   │   Solution  │     │   Search    │     │   Criterion │         │
│   └─────────────┘     └─────────────┘     └──────┬──────┘         │
│                                                   │                │
│                              ┌────────────────────┘                │
│                              │                                     │
│                       ┌──────▼──────┐                              │
│                       │    Kick     │                              │
│                       │ Perturbation│                              │
│                       └─────────────┘                              │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

### 3.3 Yerel Arama (Local Search)

**Greedy Hill Climbing:**

```python
def local_search(peptide, score):
    for step in range(local_steps):  # 60 adım
        candidate = move_operator(peptide)
        candidate_score = predict(candidate)
        if candidate_score < score:  # Sadece iyileşme kabul
            peptide = candidate
            score = candidate_score
    return peptide, score
```

### 3.4 Pertürbasyon Operatörleri (Kick)

#### 3.4.1 Segment Rearrangement Kick (%60)

Double-bridge benzeri pertürbasyon - dizi segmentlerini yeniden düzenler:

```
┌─────────────────────────────────────────────────────────────────┐
│              SEGMENT REARRANGEMENT KICK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Orijinal dizi:                                                 │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐             │
│   │ A │ E │ F │ G │ H │ I │ K │ L │ M │ N │ Q │ R │             │
│   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘             │
│     S1    │  S2   │   S3  │   S4  │     S5                      │
│           p1      p2      p3      p4                             │
│                                                                  │
│   Yeniden düzenleme: S1 + S3 + S2 + S4 + S5                     │
│                                                                  │
│   Sonuç:                                                         │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐             │
│   │ A │ E │ H │ I │ F │ G │ K │ L │ M │ N │ Q │ R │             │
│   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.4.2 Point Mutation Kick (%40)

Birden fazla (2-3) rastgele mutasyon:

```
┌─────────────────────────────────────────────────────────────────┐
│                POINT MUTATION KICK                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Orijinal:  A E F G H I K L M N Q R                            │
│                ↓     ↓       ↓                                   │
│   Mutant:    A W F G Y I K L W N Q R                            │
│              [3 rastgele mutasyon]                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Adaptif Kick Strength

```
┌──────────────────────────────────────────────────────────────────┐
│                 ADAPTIVE KICK MECHANISM                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   İyileşme var → kick_strength = max(1, kick - 1)                │
│                                                                   │
│   Stagnation (50 iter) → kick_strength = min(4, kick + 1)        │
│                                                                   │
│   Restart (250 iter) → kick_max kadar kick + local search        │
│                                                                   │
│                                                                   │
│   kick_strength:  1 ──────▶ 2 ──────▶ 3 ──────▶ 4                │
│                   │         │         │         │                 │
│                   ▼         ▼         ▼         ▼                 │
│   Etki:        Hafif    Orta     Güçlü    Çok Güçlü              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.6 Algoritma Akış Diyagramı

```
          ┌─────────────────────────────────┐
          │      Başlangıç x₀               │
          │      x = LocalSearch(x₀)        │
          │      best = x                   │
          └───────────────┬─────────────────┘
                          │
          ┌───────────────▼─────────────────┐
          │         Kick(x)                 │
          │         x' = perturb(x)         │
          └───────────────┬─────────────────┘
                          │
          ┌───────────────▼─────────────────┐
          │      LocalSearch(x')            │
          │      x'' = improved x'          │
          └───────────────┬─────────────────┘
                          │
          ┌───────────────▼─────────────────┐
          │      f(x'') < f(x) ?            │
          └───────────────┬─────────────────┘
                  Evet    │    Hayır
          ┌───────────────┴───────────────┐
          │                               │
   ┌──────▼──────┐               ┌────────▼────────┐
   │   x = x''   │               │  no_improve++   │
   │   best?     │               │  Adaptif kick   │
   └──────┬──────┘               └────────┬────────┘
          │                               │
          └───────────────┬───────────────┘
                          │
          ┌───────────────▼─────────────────┐
          │    iter < 800 ?                 │
          └───────────────┬─────────────────┘
                  Evet    │    Hayır
                  │       └──────────────────┐
                  │                          │
          ┌───────▼───────┐         ┌────────▼────────┐
          │  Sonraki iter │         │  Return best    │
          └───────────────┘         └─────────────────┘
```

### 3.7 Hiperparametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `max_iterations` | 800 | Ana döngü iterasyon sayısı |
| `local_steps` | 60 | Yerel arama adım sayısı |
| `kick_strength` | 1 | Başlangıç kick gücü |
| `kick_max` | 4 | Maksimum kick gücü |
| `p_segment_kick` | 0.60 | Segment kick olasılığı |
| `stagnation_iters` | 50 | Stagnation tespiti için iterasyon |
| `restart_iters` | 250 | Restart için iterasyon eşiği |

### 3.8 Zaman Karmaşıklığı

```
O(max_iterations × local_steps × model_inference_time)
≈ O(800 × 60 × 1ms) ≈ 48 saniye/peptid
```

---

## 4. MOCO-CEM (Cross-Entropy Method)

### 4.1 Algoritma Teorisi

MOCO-CEM, çözüm uzayı üzerinde bir olasılık dağılımı (θ) tutar ve bu dağılımı "elite" çözümlerden öğrenerek günceller. Cross-Entropy Method'un meta-optimizasyon versiyonudur.

### 4.2 Matematiksel Formülasyon

**Olasılık Matrisi (Heatmap):**

```
θ ∈ ℝ^(L×|Σ|) = ℝ^(12×18)
```

Her pozisyon için amino asit olasılıkları:

```
P(aᵢ = a | θ) = softmax(θᵢ / τ)ₐ = exp(θᵢₐ/τ) / Σⱼ exp(θᵢⱼ/τ)
```

**Sıcaklık Programı (Temperature Annealing):**

```
τ(t) = τ_start + (τ_end - τ_start) × (t-1) / (K-1)
```

**EMA Güncellemesi (Exponential Moving Average):**

```
θ_new = (1 - α) × θ_old + α × θ_elite
```

### 4.3 Algoritma Yapısı

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MOCO-CEM FRAMEWORK                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Iteration t:                                                       │
│                                                                      │
│   1. SAMPLE: θ(t) → {x₁, x₂, ..., x_batch}                          │
│              (128 peptid örnekle)                                    │
│                                                                      │
│   2. EVALUATE: f(xᵢ) for all i                                       │
│                                                                      │
│   3. LOCAL SEARCH: Top 25% elite'lere yerel arama                   │
│                                                                      │
│   4. SELECT ELITE: Top 12% (≈15 peptid)                             │
│                                                                      │
│   5. UPDATE θ: EMA ile olasılık matrisi güncelle                    │
│                                                                      │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│   │ Sample  │───▶│Evaluate │───▶│ Local   │───▶│ Update  │          │
│   │  θ→X    │    │  f(X)   │    │ Search  │    │   θ     │          │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘          │
│        ▲                                            │                │
│        └────────────────────────────────────────────┘                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.4 Heatmap (θ) Görselleştirmesi

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROBABILITY HEATMAP θ                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│        A   D   E   F   G   H   I   K   L   M   N   Q   R   S   T   V   W   Y │
│      ┌───────────────────────────────────────────────────────────────┐│
│  P1  │ ░░░ ░░░ ░░░ ███ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ▓▓▓ ░░░ ││
│  P2  │ ░░░ ░░░ ▓▓▓ ░░░ ███ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ││
│  P3  │ ░░░ ░░░ ░░░ ░░░ ░░░ ███ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ▓▓▓ ░░░ ░░░ ░░░ ░░░ ░░░ ││
│  ... │ ...                                                           ││
│  P12 │ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░ ███ ░░░ ░░░ ░░░ ░░░ ░░░ ││
│      └───────────────────────────────────────────────────────────────┘│
│                                                                      │
│   ███ = Yüksek olasılık (>20%)                                       │
│   ▓▓▓ = Orta olasılık (10-20%)                                       │
│   ░░░ = Düşük olasılık (<10%)                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.5 Elite Seçimi ve Ağırlıklandırma

```
Elite ağırlıkları (rank-based):

w_i = 1/rank_i / Σⱼ(1/rank_j)

Örnek (15 elite için):
┌──────┬──────┬──────────┬────────────┐
│ Rank │ Skor │ 1/rank   │ Ağırlık    │
├──────┼──────┼──────────┼────────────┤
│  1   │-66.8 │ 1.000    │ 0.303      │
│  2   │-66.1 │ 0.500    │ 0.152      │
│  3   │-65.5 │ 0.333    │ 0.101      │
│  4   │-65.0 │ 0.250    │ 0.076      │
│  5   │-64.5 │ 0.200    │ 0.061      │
│ ...  │ ...  │ ...      │ ...        │
│ 15   │-60.0 │ 0.067    │ 0.020      │
└──────┴──────┴──────────┴────────────┘
```

### 4.6 Algoritma Akış Diyagramı

```
          ┌─────────────────────────────────────┐
          │   θ₀ = init_from_peptide(x₀)       │
          │   best = x₀, best_score = f(x₀)    │
          └─────────────────┬───────────────────┘
                            │
          ┌─────────────────▼───────────────────┐
          │         τ = τ(iteration)            │
          │   X = sample_batch(θ, τ, 128)       │
          └─────────────────┬───────────────────┘
                            │
          ┌─────────────────▼───────────────────┐
          │      scores = [f(x) for x in X]     │
          └─────────────────┬───────────────────┘
                            │
          ┌─────────────────▼───────────────────┐
          │   Local search on top 32 (25%)      │
          └─────────────────┬───────────────────┘
                            │
          ┌─────────────────▼───────────────────┐
          │      Update best if improved        │
          └─────────────────┬───────────────────┘
                            │
          ┌─────────────────▼───────────────────┐
          │   elite = top 15 (12%) peptides     │
          │   weights = rank_based_weights      │
          └─────────────────┬───────────────────┘
                            │
          ┌─────────────────▼───────────────────┐
          │   freq = weighted_aa_frequency      │
          │   θ_elite = normalize(freq)         │
          └─────────────────┬───────────────────┘
                            │
          ┌─────────────────▼───────────────────┐
          │   θ = (1-α)×θ + α×θ_elite           │
          │   (α = 0.18)                        │
          └─────────────────┬───────────────────┘
                            │
          ┌─────────────────▼───────────────────┐
          │      iter < 300 ?                   │
          └─────────────────┬───────────────────┘
                    Evet    │    Hayır
                    │       └────────────────────┐
          ┌─────────▼─────────┐         ┌────────▼────────┐
          │   Sonraki iter    │         │  Return best    │
          └───────────────────┘         └─────────────────┘
```

### 4.7 Hiperparametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `max_iterations` | 300 | Ana döngü iterasyon sayısı |
| `batch_size` | 128 | Her iterasyonda örneklenen peptid |
| `elite_frac` | 0.12 | Elite oranı (%12) |
| `alpha` | 0.18 | EMA güncelleme katsayısı |
| `tau_start` | 1.8 | Başlangıç sıcaklığı |
| `tau_end` | 0.65 | Bitiş sıcaklığı |
| `use_local_search` | True | Yerel arama aktif |
| `local_steps` | 30 | Yerel arama adım sayısı |

### 4.8 Zaman Karmaşıklığı

```
O(max_iterations × (batch_size + elite_ls × local_steps) × model_inference_time)
≈ O(300 × (128 + 32×30) × 1ms)
≈ O(300 × 1088 × 1ms) ≈ 5-6 dakika/peptid
```

---

## 5. Karşılaştırmalı Analiz

### 5.1 Performans Metrikleri (PET Örneği)

| Metrik | SA | ILS | MOCO-CEM |
|--------|-----|-----|----------|
| **En İyi Skor** | -63.14 | -66.11 | -66.76 |
| **Ortalama Skor** | -57.99 | -63.20 | -64.14 |
| **Standart Sapma** | 3.56 | 2.08 | 2.27 |
| **Ortalama İyileşme** | 46.12 | 51.33 | 52.27 |
| **Süre (10 peptid)** | ~30s | ~8dk | ~55dk |

### 5.2 Convergence Karakteristikleri

```
Skor
  │
  │    SA (hızlı yakınsama, erken durma)
  │    ╲
-50 ┤     ╲───────────────────────────
  │       ╲
  │         ILS (kademeli iyileşme)
-55 ┤         ╲
  │           ╲──────────────────────
  │             ╲
-60 ┤               MOCO-CEM (sürekli iyileşme)
  │                 ╲
  │                   ╲──────────────
-65 ┤                     ╲
  │                       ╲────────
  │
  └──┬──────┬──────┬──────┬──────┬───▶ İterasyon
     0     500   1000   1500   2000
```

### 5.3 Keşif vs Sömürü (Exploration vs Exploitation)

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   EXPLORATION ◄──────────────────────────────────────► EXPLOITATION│
│                                                                    │
│        │                    │                    │                 │
│        ▼                    ▼                    ▼                 │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐           │
│   │  MOCO-  │          │   ILS   │          │   SA    │           │
│   │   CEM   │          │         │          │         │           │
│   │         │          │         │          │         │           │
│   │ Popülasyon│        │ Kick +  │          │ Soğutma │           │
│   │ tabanlı  │        │ Local   │          │ ile     │           │
│   │ örnekleme│        │ Search  │          │ yerel   │           │
│   └─────────┘          └─────────┘          └─────────┘           │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 5.4 Amino Asit Tercih Farkları

| Yöntem | Top 5 AA | Karakteristik |
|--------|----------|---------------|
| **SA** | W(21%), R(14%), M(12%), H(9%), F(6%) | Hidrofobik + bazik |
| **ILS** | W(26%), H(15%), R(13%), F(10%), G(6%) | Aromatik ağırlıklı |
| **MOCO-CEM** | W(31%), G(23%), R(16%), Y(8%), Q(7%) | Triptofan dominant |

---

## 6. Uygulama Önerileri

### 6.1 Yöntem Seçimi Karar Ağacı

```
                    ┌─────────────────┐
                    │   Başla         │
                    └────────┬────────┘
                             │
               ┌─────────────▼─────────────┐
               │  Zaman kısıtlı mı?        │
               └─────────────┬─────────────┘
                     Evet    │    Hayır
              ┌──────────────┴──────────────┐
              │                             │
       ┌──────▼──────┐            ┌─────────▼─────────┐
       │     SA      │            │ Kalite öncelikli? │
       │   (~3s)     │            └─────────┬─────────┘
       └─────────────┘                 Evet │    Hayır
                                ┌──────────┴──────────┐
                                │                     │
                        ┌───────▼───────┐     ┌───────▼───────┐
                        │   MOCO-CEM    │     │     ILS       │
                        │   (~5-6dk)    │     │   (~48s)      │
                        └───────────────┘     └───────────────┘
```

### 6.2 Hibrit Yaklaşım Önerisi

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HİBRİT PİPELİNE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Aşama 1: SA ile hızlı tarama (100 başlangıç noktası)              │
│            → En iyi 10 peptidi seç                                   │
│                                                                      │
│   Aşama 2: ILS ile orta derinlikte arama (10 peptid)                │
│            → En iyi 5 peptidi seç                                    │
│                                                                      │
│   Aşama 3: MOCO-CEM ile derin arama (5 peptid)                      │
│            → Final optimum                                           │
│                                                                      │
│   Toplam süre: ~30dk (vs. saf MOCO-CEM: ~5 saat)                    │
│   Kalite: Saf MOCO-CEM'e yakın                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Çıktı Klasör Yapısı

Script çalıştırıldığında her plastik tipi için ayrı bir klasör oluşturulur:

```
results/peptide_generation_comparison/
├── PET/
│   ├── model/
│   │   └── encdec_PET.pth          # Eğitilmiş model
│   ├── figures/
│   │   ├── training_curve.png      # Eğitim loss/R² grafiği
│   │   ├── method_comparison.png   # SA/ILS/MOCO karşılaştırma
│   │   ├── score_distribution.png  # Skor dağılımı histogramı
│   │   ├── aa_heatmap.png          # Pozisyon bazlı AA frekansı
│   │   └── trajectory.png          # Optimizasyon trajectory
│   ├── peptides/
│   │   ├── SA_peptides.csv         # SA ile üretilen peptidler
│   │   ├── ILS_peptides.csv        # ILS ile üretilen peptidler
│   │   ├── MOCO-CEM_peptides.csv   # MOCO-CEM ile üretilen peptidler
│   │   └── all_peptides.csv        # Tüm peptidler birleşik
│   ├── logs/
│   │   ├── training_log.json       # Eğitim metrikleri
│   │   └── peptide_results.json    # Peptid üretim sonuçları
│   └── RAPOR_PET.md                # Detaylı markdown rapor
├── PP/
│   └── ... (aynı yapı)
├── PE/
├── PVC/
├── PS/
├── Nylon/
├── summary/
│   └── genel_ozet.csv              # Tüm plastikler özet tablo
└── checkpoint.json                  # Resume için checkpoint
```

### Kullanım

```bash
# Tüm plastikler için çalıştır
python peptide_generation_comparison.py

# Tek plastik için çalıştır
python peptide_generation_comparison.py --plastic PET

# Peptid sayısını ayarla (varsayılan: 10)
python peptide_generation_comparison.py --n_peptides 20
```

---

## 8. Sonuç

Bu çalışmada, peptid optimizasyonu için üç farklı metaheuristik yöntem uygulanmıştır:

1. **SA (Simulated Annealing):** Hızlı, basit, iyi baseline sonuçlar
2. **ILS (Iterated Local Search):** Orta hız, iyi kalite, adaptif pertürbasyon
3. **MOCO-CEM (Cross-Entropy Method):** Yavaş ama en iyi sonuçlar, popülasyon tabanlı

**Temel Bulgular:**
- MOCO-CEM, PET için en düşük skoru (-66.76) elde etmiştir
- ILS ve MOCO-CEM peptidleri arasında yüksek benzerlik (Hamming mesafesi: 2)
- Triptofan (W) tüm yöntemlerde en sık tercih edilen amino asittir

**Önerilen Kullanım:**
- Hızlı tarama: SA
- Dengeli performans: ILS
- En iyi kalite: MOCO-CEM veya Hibrit yaklaşım

---

## Referanslar

1. Kirkpatrick, S., et al. (1983). "Optimization by Simulated Annealing." Science, 220(4598), 671-680.
2. Lourenço, H. R., et al. (2003). "Iterated Local Search." Handbook of Metaheuristics, 320-353.
3. De Boer, P. T., et al. (2005). "A Tutorial on the Cross-Entropy Method." Annals of Operations Research, 134(1), 19-67.

---

*Bu dokümantasyon peptide_generation_comparison.py scripti için hazırlanmıştır.*
