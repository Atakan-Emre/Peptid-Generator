# 🧬 Peptid Generator — Microplastic-Binding Peptide Design

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+cu128-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Deep-learning design of microplastic-binding peptides. Affinity models are
trained per polymer for six plastics and then optimised with three
metaheuristics to propose high-affinity 12-mer candidates.

---

## 📊 Project Summary

| Metric | Value |
|--------|-------|
| **Polymers** | PET, PP, PE, PVC, PS, Nylon |
| **Architectures** | CNN, LSTM, LSTM-VAE, Encoder-Decoder |
| **Hyperparameter configurations** | 624 (104 per polymer) |
| **Final training runs** | 240 (4 architectures × 5 seeds × 2 splits × 6 polymers) |
| **Best architecture** | **CNN** — best on all six polymers, 18/18 comparisons significant |
| **Mean test R² (identity-aware split)** | **0.8509** |
| **Mean test R² (random split)** | 0.9474 |
| **Generated peptides** | 180 (30 × 6 polymers) |
| **Exact matches in training data** | 0 |

Two test R² figures are reported because the two splitting strategies measure
different things; see [Evaluation protocol](#-evaluation-protocol).

---

## 📁 Project Structure

```
Peptid-Generator/
│
├── 📂 pbp/                          # Library
│   ├── encoding.py                  # 18-letter alphabet, 12-mer encoding
│   ├── device.py                    # GPU detection, capability check, batch sizing
│   ├── config.py                    # Configuration loader
│   ├── data/
│   │   ├── cluster.py               # Greedy clustering with pigeonhole banding
│   │   └── prepare.py               # Deduplication, splitting, leakage report
│   ├── models/architectures.py      # CNN, LSTM, LSTM-VAE, EncDec
│   ├── train/                       # Training loop, experiment runner, metrics
│   ├── optimize/algorithms.py       # SA, ILS, MOCO-CEM
│   ├── analysis/                    # Scoring, selectivity, novelty, physchem,
│   │                                # interpretability, statistics
│   └── reporting/                   # Shared figure style, paper figures
│
├── 📂 scripts/                      # Pipeline steps 00-11 + helpers
│   ├── 00_check_device.py           # GPU and architecture smoke test
│   ├── 01_prepare_data.py           # Deduplicate, cluster, split, leakage report
│   ├── 02_estimate_budget.py        # Training-time estimate
│   ├── 03_search_hyperparams.py     # Stage A: hyperparameter search
│   ├── 04_train_final.py            # Stage B: multi-seed final training
│   ├── 05_generate_peptides.py      # SA / ILS / MOCO-CEM generation
│   ├── 06_compare_generators.py     # Surrogate comparison and selection
│   ├── 07_run_analyses.py           # Selectivity, novelty, physchem, positions
│   ├── 08_jain_baseline.py          # Benchmark against Jain et al. (2025)
│   ├── 09_build_paper_tables.py     # T1-T10
│   ├── 10_build_figures.py          # F1-F8
│   ├── 11_review_compliance.py      # Reviewer-request audit
│   ├── 12_select_docking_set.py     # Stratified peptide set for docking
│   ├── 13_make_peptide_summary.py   # Peptide summary document for the lab
│   ├── make_review_documents.py     # Response letters + revision notes
│   └── run_pipeline.py              # Runs all thirteen steps in order
│
├── 📂 configs/active.json           # Local configuration, not tracked
│
├── 📂 data/sortingData/             # Raw data: PET, PP, PE, PVC, PS, Nylon
│
├── 📂 prepared/                     # Splits (.npz, untracked) + metadata (.json)
│   └── leakage_summary.json         # Duplicate and nearest-neighbour report
│
├── 📂 results/                      # Current results
│   ├── search/                      # Best configuration per polymer × architecture
│   ├── final/                       # Per-seed metrics, 240 runs
│   ├── models/                      # Checkpoints (untracked, 352 MB)
│   ├── summary_*.json               # Aggregated metrics per split
│   ├── architecture_stats_*.json    # Paired significance tests
│   ├── generated/                   # Peptides per surrogate + selection record
│   ├── analysis/                    # Selectivity, novelty, physchem, positions
│   ├── jain_baseline/               # Prior-work benchmark
│   ├── paper_tables/ALL_TABLES.md   # 📌 T1-T10, the tables as reported
│   ├── figures/                     # 📌 F1-F8, PNG (300 dpi) + PDF
│   ├── review_compliance.md         # 📌 Reviewer-request audit
│   └── analysis/docking_selection.* # 📌 Peptides to dock, with rationale
│
├── diyagram.drawio                  # Workflow diagram (manuscript Figure 1)
├── Readme.md                        # 📌 This file
├── .gitignore
├── .gitattributes
└── LICENSE
```

---

## 🏆 Results

### Performance per Polymer

Best architecture (CNN), five seeds, mean ± standard deviation.

| Polymer | Test R² (identity-aware) | Test R² (random) | MAE | RMSE | Dataset size |
|---------|--------------------------|------------------|-----|------|--------------|
| **PET** | **0.8912** ± 0.0008 | 0.9615 | 3.35 | 4.67 | 441,978 |
| **PP** | 0.8639 ± 0.0011 | 0.9497 | 2.77 | 3.79 | 433,487 |
| **PE** | 0.8556 ± 0.0007 | 0.9446 | 2.77 | 3.81 | 715,508 |
| **Nylon** | 0.8520 ± 0.0023 | 0.9559 | 3.31 | 4.49 | 142,614 |
| **PS** | 0.8373 ± 0.0017 | 0.9391 | 2.70 | 3.65 | 405,827 |
| **PVC** | 0.8057 ± 0.0015 | 0.9334 | 3.53 | 4.77 | 208,608 |

MAE and RMSE are for the identity-aware split.

### Architecture Comparison

Paired across seeds — same split, same data, only the architecture differs.
CNN is best on every polymer under both splits; all 18 pairwise comparisons
are significant (paired *t*, p < 1e-6 in most cases).

| Polymer | CNN | EncDec | LSTM | LSTM-VAE |
|---------|-----|--------|------|----------|
| PET | **0.8912** | 0.8480 | 0.8243 | 0.7779 |
| PP | **0.8639** | 0.8155 | 0.8007 | 0.7493 |
| PE | **0.8556** | 0.8034 | 0.7884 | 0.7424 |
| Nylon | **0.8520** | 0.7560 | 0.7079 | 0.7176 |
| PS | **0.8373** | 0.7745 | 0.7545 | 0.6940 |
| PVC | **0.8057** | 0.7234 | 0.6770 | 0.6769 |

Full statistics in `results/paper_tables/T4_architecture_significance.md`.

### Dataset Structure

| Polymer | CSV file | Rows | Unique sequences | Duplicates | Train | Val | Test |
|---------|----------|------|------------------|------------|-------|-----|------|
| **PET** | `data/sortingData/PET.csv` | 441,978 | 381,065 | 60,913 | 304,851 | 38,107 | 38,107 |
| **PP** | `data/sortingData/PP.csv` | 433,487 | 433,487 | 0 | 346,789 | 43,349 | 43,349 |
| **PE** | `data/sortingData/PE.csv` | 715,508 | 715,508 | 0 | 572,406 | 71,551 | 71,551 |
| **PVC** | `data/sortingData/PVC.csv` | 208,608 | 208,608 | 0 | 166,886 | 20,861 | 20,861 |
| **PS** | `data/sortingData/PS.csv` | 405,827 | 405,827 | 0 | 324,661 | 40,583 | 40,583 |
| **Nylon** | `data/sortingData/Nylon.csv` | 142,614 | 142,614 | 0 | 114,090 | 14,262 | 14,262 |

> 📝 Each CSV has `Sequence` and `Score` columns. Only PET contains repeated
> sequences; these are removed before splitting. Split sizes are for the
> identity-aware split.

### Best Generated Peptides

| Polymer | Peptide | Score | Method |
|---------|---------|-------|--------|
| **Nylon** | RWMLWHWRLRRW | **-83.05** | SA |
| **PVC** | KWTMRVNMRRRR | -78.70 | SA |
| **PET** | WWYWEWRYMRWW | -73.76 | SA |
| **PE** | TFLMTMKWRMLF | -72.41 | SA |
| **PP** | FWLWQIFERRLW | -64.77 | MOCO-CEM |
| **PS** | IFWRYAVLQHQM | -51.55 | SA |

> 📝 More negative score = stronger predicted binding affinity.

---

## 🔬 Evaluation Protocol

### Why two test scores

A random split is not a valid holdout for this data. The libraries contain
dense families of near neighbours, so under a random split a large share of
test sequences are one or two mutations away from a training sequence. A
model can then score well by recognising variants it has already seen.

Sequences are therefore clustered before splitting, and the split is made at
the level of cluster representatives. Clustering is greedy and incremental
(CD-HIT style) at a Hamming radius of 3, using pigeonhole banding: with 12
positions divided into *d*+1 blocks, two sequences within Hamming distance
*d* must share at least one identical block, so only candidates sharing a
block are compared. All six polymers cluster in under 20 seconds and the
largest cluster holds ≤ 0.3 % of sequences.

| Polymer | Split | Mean NN identity | Test sequences ≤ 2 mutations from train |
|---------|-------|------------------|------------------------------------------|
| PET | random | 77.53 % | 52.67 % |
| PET | identity-aware | 68.31 % | 21.67 % |
| PE | random | 72.19 % | 37.00 % |
| PE | identity-aware | 64.17 % | 13.67 % |
| PP | random | 76.36 % | 50.33 % |
| PP | identity-aware | 65.83 % | 15.67 % |
| PS | random | 74.33 % | 44.33 % |
| PS | identity-aware | 65.36 % | 17.67 % |
| PVC | random | 83.08 % | 73.67 % |
| PVC | identity-aware | 72.92 % | 34.00 % |
| Nylon | random | 87.36 % | 89.00 % |
| Nylon | identity-aware | 75.72 % | 38.00 % |

The resulting difference in measured performance is 0.070 (PET) to 0.128
(PVC) R², and it is largest exactly where the random split leaves the most
near neighbours in the test set.

### Other protocol details

* Normalisation statistics (z-score) are computed from the **training split
  only** and stored in each checkpoint.
* Every configuration is trained with five seeds; architecture comparisons
  use paired tests across seeds with Cohen's *d* and 95 % confidence
  intervals.
* Learning rate follows cosine annealing from 1e-3 to 1e-5.

---

## 📊 Benchmark against Jain et al. (2025)

The closest prior method (Jain et al., *Chem. Sci.* 2025, **16**, 20823) is
re-implemented as described there — unidirectional LSTM, 2 layers, hidden
size 512, one-hot input — and run under three conditions that differ only in
how the data is split:

* **A** — raw data, random split (their setup)
* **B** — duplicates removed, random split
* **C** — duplicates removed, identity-aware split (this work)

The same three conditions are also run with this work's CNN, using the
hyperparameters selected in Stage A, so the comparison isolates the
architecture rather than the tuning effort. Three seeds per cell.

| Polymer | Condition A: prior | Condition A: CNN | Condition C: prior | Condition C: CNN | Reported by Jain et al. |
|---------|--------------------|------------------|--------------------|------------------|--------------------------|
| PET | 0.8890 | **0.9704** | 0.8396 | **0.8918** | 0.9755 |
| PE | 0.8183 | **0.9444** | 0.7930 | **0.8561** | 0.9517 |
| PP | 0.8323 | **0.9497** | 0.8077 | **0.8635** | 0.9640 |
| PVC | 0.7276 | **0.9329** | 0.6956 | **0.8070** | 0.9554 |
| Nylon | 0.7719 | **0.9557** | 0.7419 | **0.8516** | 0.9774 |

The CNN is better on all five polymers in every condition, and all fifteen
comparisons are significant. The **size** of the advantage falls from +0.1428
on average in condition A to +0.0784 in condition C, a reduction of 45 %:
the architectural difference is real under both protocols, but a random split
inflates it, because part of the measured gap reflects memorisation of near
variants.

Within the prior architecture, moving from condition A to condition C costs
0.0246 (PP) to 0.0494 (PET) R², all significant at p < 0.01, with data
splitting as the only changing factor.

Full table: `results/jain_baseline/jain_karsilastirma.md`.

---

## 🧪 Surrogate Comparison for Generation

Optimising a learned surrogate carries a specific risk: the search can find
sequences that exploit the model's error rather than genuinely bind the
target. Generation is therefore run independently with each candidate
surrogate and the resulting peptide sets are compared on three axes —
selectivity, agreement with a third model used in neither optimisation, and
sequence degeneracy.

| Surrogate | Train–test R² gap | Target best | Mean selectivity index | Independent gain | Top AA share | Positional entropy | Degenerate |
|-----------|-------------------|-------------|------------------------|------------------|--------------|--------------------|------------|
| CNN | 0.1385 | 4/6 | 20.99 | 29.24 | 0.436 | 0.491 | 1/6 |
| **EncDec** | 0.0916 | **6/6** | **28.51** | **30.92** | 0.303 | 0.612 | 0/6 |

The CNN is the stronger predictor but the weaker generator: with the wider
train–test gap, the optimiser collapses PET sequences onto tryptophan
(62.5 % W, against 36.7 % for the encoder-decoder) and loses target specificity for PP and PS. The encoder–decoder
passes on all six polymers, so it supplies the peptides reported here. Both
sets are kept in `results/generated/` and the selection record, including
the rejected candidate's full results, is in
`results/analysis/generator_comparison.md`.

### Selectivity

Each peptide set is scored against all six polymer models. The selectivity
index is the mean score on the other five polymers minus the score on the
target; positive means target-specific.

| Polymer | PVC | Nylon | PET | PE | PP | PS |
|---------|-----|-------|-----|----|----|-----|
| **Selectivity index** | 42.74 | 40.33 | 30.57 | 26.27 | 21.64 | 9.53 |

Every set scores best on its own target. PS is the narrowest margin.

### Novelty and Independent Validation

| Polymer | Exact matches in train | Mean NN identity | ≥ 90 % identity | Optimised | Random baseline | Best 1 % of training data |
|---------|------------------------|------------------|-----------------|-----------|-----------------|---------------------------|
| PET | 0 | 55.6 % | 0 % | -64.84 | -14.66 | -56.87 |
| PE | 0 | 62.8 % | 0 % | -60.52 | -15.35 | -51.23 |
| PP | 0 | 58.3 % | 0 % | -57.10 | -11.36 | -45.64 |
| PS | 0 | 58.1 % | 0 % | -43.89 | -9.94 | -39.60 |
| PVC | 0 | 55.6 % | 0 % | -69.23 | -24.23 | -58.65 |
| Nylon | 0 | 62.2 % | 0 % | -75.97 | -26.01 | -68.20 |

The generated peptides outscore both random sequences and the best 1 % of
the measured training data on every polymer.

---

## 🚀 Installation

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # Linux / macOS

# RTX 50-series (Blackwell, sm_120) requires the CUDA 12.8 wheel
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scipy matplotlib
```

`matplotlib` is needed from step 10 onward (figures). The helper that writes
the reviewer-response documents additionally needs `python-docx`, `reportlab`
and `pypdf`; it is not part of the pipeline, so install those only if you want
to regenerate those documents.

### Hardware

| Component | Minimum | Used here |
|-----------|---------|-----------|
| GPU | NVIDIA GTX 1080 | RTX 5070 Ti (16 GB) |
| VRAM | 8 GB | 17.1 GB |
| RAM | 16 GB | 64 GB |
| CPU | 4 cores | 8+ cores |

---

## 💻 Usage

### Full pipeline

```bash
python scripts/run_pipeline.py              # all thirteen steps, resumable
python scripts/run_pipeline.py --from 07    # from a given step onward
```

Each step writes a success stamp to `logs/`; re-running skips completed work.
A full run from scratch takes roughly **12 hours** on an RTX 5070 Ti:
~4 h hyperparameter search, ~6.5 h final training, ~1 h prior-work benchmark,
~0.5 h generation and analysis.

### Individual steps

```bash
python scripts/01_prepare_data.py                        # splits + leakage report
python scripts/04_train_final.py --strategy clustered    # final training
python scripts/05_generate_peptides.py                   # peptide generation
python scripts/06_compare_generators.py                  # surrogate selection
python scripts/07_run_analyses.py                        # characterisation
python scripts/09_build_paper_tables.py                  # T1-T10
python scripts/10_build_figures.py                       # F1-F8
python scripts/11_review_compliance.py                   # reviewer audit
python scripts/12_select_docking_set.py                  # docking work list
python scripts/13_make_peptide_summary.py                # lab summary (.docx)
```

Figures and tables are built from the same JSON outputs and never recompute a
model, so the two cannot drift apart. `10_build_figures.py` skips any figure
whose input is missing and names the step that produces it.

On Windows, calling `.venv\Scripts\python.exe` directly avoids PowerShell's
script-execution policy.

---

## 🧠 Model Architectures

Four architectures are trained under identical conditions. The CNN is best on
every polymer.

### CNN — best predictor

```
[Input: one-hot peptide, 12 × 18]
            ↓
   ┌──────────────────┐
   │  Conv1d × 3      │  256 filters, kernel 5
   │  LayerNorm       │  dropout 0.1
   │  ReLU            │
   └──────────────────┘
            ↓
   [Global pooling → dense]
            ↓
     [Predicted score]

Loss = MSE(score)
```

A kernel width of 5 over a 12-residue sequence captures local motifs
directly. The advantage is largest on the smallest datasets (Nylon +0.096,
PVC +0.082 R² over the encoder–decoder), which is consistent with a
convolutional model being more data-efficient than recurrent ones here.

### Encoder-Decoder — used for generation

```
[Input: one-hot peptide, 12 × 18]
            ↓
   ┌─────────────┐
   │   Encoder   │  LSTM 2-3 layers, hidden 256
   │   LayerNorm │  dropout 0.1
   └─────────────┘
            ↓
      [Latent vector]
            ↓
    ┌───────┴───────┐
    ↓               ↓
┌─────────┐   ┌───────────┐
│ Decoder │   │   Score   │
│  LSTM   │   │ Predictor │
└─────────┘   └───────────┘
    ↓               ↓
[Reconstruction] [Predicted score]

Loss = MSE(score) + λ · CrossEntropy(reconstruction)
```

The reconstruction term narrows the train–test gap (0.092 versus 0.139 for
the CNN), which is what makes this model the safer target for optimisation
even though it predicts less accurately.

---

## 📈 Training Protocol

### Stage A — hyperparameter search
- 104 configurations per polymer, 624 in total
- 200,000-sample subsample, 25 epochs, patience 6
- Selection on validation R²

### Stage B — final training
- Full data, 5 seeds, 60 epochs, patience 15
- Cosine annealing, 1e-3 → 1e-5
- Batch size 2048, GPU-resident data (uint8 indices, 12 B/sample)
- Both splitting strategies, all four architectures: 240 runs

---

## 🔬 Amino Acid Frequency

Most frequent residues in the generated peptides:

| AA | Frequency | Property | Likely role |
|----|-----------|----------|-------------|
| **W** | 28.0 % | Aromatic | π–π stacking, large hydrophobic surface |
| **R** | 17.1 % | Positive | Hydrogen bonding, electrostatics |
| **M** | 9.5 % | Hydrophobic | Flexible non-polar contact |
| **L** | 5.9 % | Hydrophobic | Packing against the polymer surface |
| **F** | 5.8 % | Aromatic | Hydrophobic interaction |
| **H** | 5.6 % | Aromatic / pH-sensitive | π stacking |

Tryptophan enrichment is a property of the generated sequences, not an
imposed constraint; its share is one of the degeneracy measures used to
reject a surrogate (see above).

---

## 🔧 Hyperparameters

### CNN — selected in Stage A

| Parameter | PET | PP | PE | PVC | PS | Nylon |
|-----------|-----|----|----|-----|----|-------|
| `hidden_dim` | 256 | 256 | 256 | 256 | 256 | 256 |
| `num_layers` | 3 | 3 | 3 | 3 | 3 | 3 |
| `n_filters` | 256 | 256 | 256 | 256 | 256 | 256 |
| `kernel_size` | 5 | 5 | 5 | 5 | 5 | 5 |
| `dropout` | 0.1 | 0.1 | 0.2 | 0.1 | 0.1 | 0.1 |
| `batch_size` | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| `learning_rate` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |

All of these, plus polymers, architectures, seeds and epoch counts, live in
`configs/active.json`. That file is **not tracked**, because it also carries
local paths and device settings. Everything needed to recreate it is in this
file: the polymers and architectures listed above, the five seeds and two
splitting strategies of the training protocol, and the hyperparameters in this
table. Moving to different hardware normally needs only `batch_size` and
`search.subsample_n`.

---

## 🖥️ System Information

| Component | Value |
|-----------|-------|
| **OS** | Windows 11 |
| **GPU** | NVIDIA RTX 5070 Ti (16 GB, sm_120) |
| **VRAM reported** | 17.1 GB |
| **CUDA** | 12.8 |
| **PyTorch** | 2.11.0+cu128 |
| **Python** | 3.12 |

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

> 📝 RTX 50-series GPUs report compute capability sm_120, which older PyTorch
> builds do not support. `00_check_device.py` checks this and reports the
> required wheel if the installed build is incompatible.

---

## 📚 Output Files

### Figures

Each figure is written as a 300 dpi PNG and as a vector PDF in
`results/figures/`, and each answers one question raised in review.

| Figure | Shows |
|--------|-------|
| **F1** `architecture_comparison` | Test R² for all four architectures × six polymers × both splits, with seed SD |
| **F2** `split_effect` | Measured performance by split, near-neighbour leakage, and the relation between the two |
| **F3** `jain_benchmark` | Prior work vs this work under conditions A, B and C |
| **F4** `independent_validation` | Optimised peptides against random sequences and the best 1 % of measured data, scored by the target model and by a model used in neither optimisation nor selection |
| **F5** `selectivity` | 6 × 6 cross-polymer score matrix and the selectivity index |
| **F6** `novelty` | Nearest-neighbour identity distribution against the training data |
| **F7** `interpretability` | Positional sensitivity and residue preference per polymer |
| **F8** `physicochemical` | Property–affinity relationships, which are polymer-specific |

### Files

| File | Contents |
|------|----------|
| [`results/paper_tables/ALL_TABLES.md`](results/paper_tables/ALL_TABLES.md) | 📌 T1–T10, all tables in one file |
| [`results/figures/`](results/figures/) | 📌 F1–F8, PNG (300 dpi) and PDF |
| [`results/review_compliance.md`](results/review_compliance.md) | 📌 Reviewer-request audit: which artefact answers which comment |
| [`prepared/leakage_summary.json`](prepared/leakage_summary.json) | Duplicates and nearest-neighbour identity per polymer and split |
| [`results/summary_clustered.json`](results/summary_clustered.json) | Per-seed metrics, identity-aware split |
| [`results/summary_random.json`](results/summary_random.json) | Per-seed metrics, random split |
| [`results/architecture_stats_clustered.json`](results/architecture_stats_clustered.json) | Paired significance tests |
| [`results/jain_baseline/jain_karsilastirma.md`](results/jain_baseline/jain_karsilastirma.md) | Prior-work benchmark, all three conditions |
| [`results/analysis/generator_comparison.md`](results/analysis/generator_comparison.md) | Surrogate comparison and selection record |
| [`results/analysis/selectivity_matrix.md`](results/analysis/selectivity_matrix.md) | 6 × 6 cross-polymer score matrix |
| [`results/generated/`](results/generated/) | Generated peptides, per surrogate |

Model checkpoints (`results/models/`, 352 MB) and prepared splits
(`prepared/*.npz`, 66 MB) are not tracked; both are reproduced by the
pipeline from the raw data.

---

## 📝 License

MIT — see [LICENSE](LICENSE).
