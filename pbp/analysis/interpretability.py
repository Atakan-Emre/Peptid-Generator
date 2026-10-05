"""Yorumlanabilirlik (model interpretability).

Iki tamamlayici analiz:

1. Pozisyon onemi - maskeleme yoluyla. Her pozisyon sirayla alfabedeki tum
   aminoasitlerle degistirilir ve tahmin edilen skordaki ortalama mutlak
   degisim olculur. Yuksek deger, o pozisyonun modelin kararinda baskin
   oldugunu gosterir. Yeniden egitim gerektirmez.

2. Pozisyon x aminoasit tercih haritasi. Her (pozisyon, aminoasit) cifti
   icin skordaki ortalama degisim; negatif deger o aminoasidin o pozisyonda
   baglanmayi guclendirdigini gosterir. Bu harita dogrudan makalenin
   girisindeki hidrofobik / aromatik / elektrostatik mekanizma anlatisiyla
   karsilastirilabilir.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from ..encoding import N_AA, PEPTIDE_LENGTH, ALPHABET, encode_many
from .scoring import ScoredModel


def position_importance(model: ScoredModel, X: np.ndarray,
                        max_samples: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    base = model.score_encoded(X)

    importance = np.zeros(PEPTIDE_LENGTH)
    pref = np.zeros((PEPTIDE_LENGTH, N_AA))
    for pos in range(PEPTIDE_LENGTH):
        deltas = np.zeros((N_AA, len(X)))
        for aa in range(N_AA):
            Xm = X.copy()
            Xm[:, pos] = aa
            deltas[aa] = model.score_encoded(Xm) - base
        importance[pos] = float(np.abs(deltas).mean())
        pref[pos] = deltas.mean(axis=1)

    return {
        "plastic": model.plastic,
        "architecture": model.architecture,
        "n_samples": int(len(X)),
        "position_importance": [round(float(v), 4) for v in importance],
        "most_important_positions": [int(p) + 1 for p in np.argsort(-importance)[:4]],
        "preference_map": {
            f"pos{p+1}": {ALPHABET[a]: round(float(pref[p, a]), 4) for a in range(N_AA)}
            for p in range(PEPTIDE_LENGTH)
        },
        "best_residue_per_position": {
            f"pos{p+1}": ALPHABET[int(np.argmin(pref[p]))] for p in range(PEPTIDE_LENGTH)
        },
        "consensus_sequence": "".join(
            ALPHABET[int(np.argmin(pref[p]))] for p in range(PEPTIDE_LENGTH)),
    }


def save(results: list[dict], out_dir) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "interpretability.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    rows = ["| Polymer | " + " | ".join(f"P{i+1}" for i in range(PEPTIDE_LENGTH))
            + " | Most important | Consensus |",
            "| --- |" + " --- |" * (PEPTIDE_LENGTH + 2)]
    for r in results:
        rows.append(f"| {r['plastic']} | " + " | ".join(
            str(v) for v in r["position_importance"]) +
            f" | {r['most_important_positions']} | {r['consensus_sequence']} |")
    (out_dir / "position_importance.md").write_text("\n".join(rows))
