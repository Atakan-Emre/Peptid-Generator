"""Veri temizleme, bolme ve normalizasyon (leakage-controlled data preparation).

Uretilen her veri bolmesi diske bir "manifest" ile yazilir: hangi dizinin
hangi bolmeye dustugu, kume kimlikleri, normalizasyon parametreleri ve
temizlik istatistikleri. Boylece tum egitim kosulari (tum tohumlar, tum
mimariler) birebir ayni bolmeleri kullanir ve sonuclar tekrarlanabilir olur.
"""
from __future__ import annotations
import json, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

from ..encoding import encode_many, validate_lengths, PEPTIDE_LENGTH
from .cluster import greedy_cluster, cluster_summary


@dataclass
class Normalization:
    """Z-skor parametreleri. YALNIZCA egitim bolmesinden hesaplanir."""
    mean: float
    std: float
    fitted_on: str = "train"

    def apply(self, y):
        return (y - self.mean) / self.std

    def invert(self, z):
        return z * self.std + self.mean


@dataclass
class DedupReport:
    n_raw: int
    n_unique: int
    n_removed: int
    duplicate_pct: float
    score_spread_of_duplicates: float | None


def deduplicate(df: pd.DataFrame, keep: str = "mean") -> tuple[pd.DataFrame, DedupReport]:
    """Tekrarlanan dizileri birlestirir.

    PET veri setinde dizilerin %13.8'i tekrarli ve bu, rastgele bolmede
    test setinin %20.7'sinin egitim setinde birebir yer almasina yol
    aciyordu. keep="mean" ayni dizinin skorlarinin ortalamasini alir;
    "first" ilk kaydi tutar.
    """
    n_raw = len(df)
    dup_mask = df.duplicated("Sequence", keep=False)
    spread = None
    if dup_mask.any():
        g = df[dup_mask].groupby("Sequence")["Score"]
        spread = float((g.max() - g.min()).mean())
    if keep == "mean":
        out = df.groupby("Sequence", as_index=False, sort=False)["Score"].mean()
    else:
        out = df.drop_duplicates("Sequence", keep="first").reset_index(drop=True)
    rep = DedupReport(
        n_raw=n_raw,
        n_unique=len(out),
        n_removed=n_raw - len(out),
        duplicate_pct=round(100 * (n_raw - len(out)) / n_raw, 3),
        score_spread_of_duplicates=None if spread is None else round(spread, 4),
    )
    return out, rep


def random_split(n: int, seed: int, fracs=(0.8, 0.1, 0.1)) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(fracs[0] * n)
    n_va = int(fracs[1] * n)
    return {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}


def clustered_split(labels: np.ndarray, seed: int, fracs=(0.8, 0.1, 0.1)) -> dict[str, np.ndarray]:
    """Kumeleri butun olarak bolmelere dagitir.

    Kumeler buyukten kucuge siralanip hedef boyutlara gore aç gözlü
    yerlestirilir; bu, bolme oranlarinin hedefe yakin kalmasini saglar
    (kume boyutlari esit olmadigi icin tam oran garanti edilemez).
    """
    rng = np.random.default_rng(seed)
    uniq, sizes = np.unique(labels, return_counts=True)
    order = np.argsort(-sizes)
    # Esit boyutlu kumeler arasinda sirayi tohum belirlesin
    jitter = rng.permutation(len(order))
    order = order[np.lexsort((jitter, -sizes[order]))] if len(order) else order
    n = len(labels)
    targets = {"train": fracs[0] * n, "val": fracs[1] * n, "test": fracs[2] * n}
    current = {k: 0 for k in targets}
    assign: dict[int, str] = {}
    for oi in order:
        c, sz = uniq[oi], sizes[oi]
        # En cok geride kalan bolmeye ver
        part = max(targets, key=lambda k: (targets[k] - current[k]) / max(targets[k], 1))
        assign[int(c)] = part
        current[part] += sz
    part_of = np.array([assign[int(c)] for c in labels])
    return {k: np.flatnonzero(part_of == k) for k in ("train", "val", "test")}


def leakage_report(X: np.ndarray, splits: dict[str, np.ndarray],
                   sample: int = 300, seed: int = 0) -> dict:
    """Test bolmesinin egitim bolmesine ne kadar yakin oldugunu olcer.

    Makalede raporlanacak temel sayi budur: rastgele bolme ile kume bazli
    bolmenin farkini gosteren tek metrik.
    """
    rng = np.random.default_rng(seed)
    tr, te = X[splits["train"]], X[splits["test"]]
    exact = 0
    tr_set = set(map(bytes, tr))
    for r in te:
        if bytes(r) in tr_set:
            exact += 1
    k = min(sample, len(te))
    q = te[rng.choice(len(te), k, replace=False)]
    m = min(60_000, len(tr))
    ref = tr[rng.choice(len(tr), m, replace=False)]
    nn = np.array([int((ref != row).sum(axis=1).min()) for row in q])
    return {
        "exact_test_in_train_pct": round(100 * exact / len(te), 3),
        "nn_hamming_mean": round(float(nn.mean()), 3),
        "nn_hamming_median": float(np.median(nn)),
        "nn_identity_mean_pct": round(100 * (1 - nn.mean() / PEPTIDE_LENGTH), 2),
        "pct_nn_le_2": round(100 * float((nn <= 2).mean()), 2),
        "pct_nn_le_3": round(100 * float((nn <= 3).mean()), 2),
    }


def prepare_plastic(
    csv_path: str | Path,
    out_dir: str | Path,
    plastic: str,
    split_strategy: str = "clustered",
    max_hamming: int = 3,
    seed: int = 42,
    dedup: bool = True,
    force: bool = False,
    tag_suffix: str = "",
) -> dict:
    """Bir plastik icin tum hazirlik adimlarini calistirip diske yazar.

    tag_suffix, ayni plastik icin alternatif bir hazirligi (ornegin tekrar
    temizligi YAPILMAMIS bir surumu) ayri dosya adiyla saklamayi saglar.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{plastic}_{split_strategy}_seed{seed}{tag_suffix}"
    npz_path = out_dir / f"{tag}.npz"
    man_path = out_dir / f"{tag}.json"
    if npz_path.exists() and man_path.exists() and not force:
        return json.loads(man_path.read_text())

    df = pd.read_csv(csv_path)
    validate_lengths(df["Sequence"].astype(str).tolist())
    dedup_rep = None
    if dedup:
        df, rep = deduplicate(df)
        dedup_rep = asdict(rep)

    seqs = df["Sequence"].astype(str).values
    y = df["Score"].astype(np.float32).values
    X = encode_many(seqs)

    labels = None
    clus = None
    if split_strategy == "clustered":
        labels = greedy_cluster(X, max_hamming=max_hamming, verbose=False)
        clus = cluster_summary(labels)
        splits = clustered_split(labels, seed=seed)
    elif split_strategy == "random":
        splits = random_split(len(X), seed=seed)
    else:
        raise ValueError(f"Bilinmeyen bolme stratejisi: {split_strategy}")

    # Normalizasyon YALNIZCA egitim bolmesinden
    y_tr = y[splits["train"]]
    norm = Normalization(mean=float(y_tr.mean()), std=float(y_tr.std() + 1e-8))

    np.savez_compressed(
        npz_path,
        X=X, y=y,
        train=splits["train"], val=splits["val"], test=splits["test"],
        cluster_labels=(labels if labels is not None else np.zeros(0, dtype=np.int32)),
    )

    manifest = {
        "plastic": plastic,
        "split_strategy": split_strategy,
        "max_hamming": max_hamming if split_strategy == "clustered" else None,
        "identity_threshold_pct": (
            round(100 * (1 - max_hamming / PEPTIDE_LENGTH), 1)
            if split_strategy == "clustered" else None
        ),
        "seed": seed,
        "n_after_prep": int(len(X)),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "split_pct": {k: round(100 * len(v) / len(X), 2) for k, v in splits.items()},
        "dedup": dedup_rep,
        "clustering": clus,
        "normalization": asdict(norm),
        "leakage": leakage_report(X, splits, seed=seed),
        "npz": str(npz_path.name),
        "data_sha1": hashlib.sha1(X.tobytes()).hexdigest()[:16],
    }
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def load_prepared(out_dir, plastic, split_strategy, seed, tag_suffix=""):
    out_dir = Path(out_dir)
    tag = f"{plastic}_{split_strategy}_seed{seed}{tag_suffix}"
    z = np.load(out_dir / f"{tag}.npz")
    man = json.loads((out_dir / f"{tag}.json").read_text())
    return z, man
