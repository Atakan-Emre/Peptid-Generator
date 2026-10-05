"""Fizikokimyasal karakterizasyon (physicochemical characterisation).

Uretilen peptitlerin hidrofobiklik, aromatiklik, net yuk ve izoelektrik
nokta ozellikleri hesaplanir ve tahmin edilen afinite skoruyla iliskisi
raporlanir. Amac, modelin ogrendigi dizi tercihlerini makalenin girisinde
anlatilan hidrofobik / pi-pi / elektrostatik mekanizmalara baglamaktir.

Harici bagimlilik yoktur; Kyte-Doolittle ve pKa degerleri modul icinde
tanimlidir (Biopython kurulu olmayabilir).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
AROMATIC = set("FWY")
POSITIVE = set("KR")
NEGATIVE = set("DE")
# Sadece yan zincir pKa degerleri (+ N/C ucu)
PKA_SIDE = {"D": 3.65, "E": 4.25, "H": 6.0, "C": 8.18, "Y": 10.07,
            "K": 10.53, "R": 12.48}
PKA_NTERM, PKA_CTERM = 9.69, 2.34


def net_charge_at_ph(seq: str, ph: float = 7.4) -> float:
    q = 1.0 / (1.0 + 10 ** (ph - PKA_NTERM)) - 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))
    for aa in seq:
        pka = PKA_SIDE.get(aa)
        if pka is None:
            continue
        if aa in ("K", "R", "H"):
            q += 1.0 / (1.0 + 10 ** (ph - pka))
        else:
            q -= 1.0 / (1.0 + 10 ** (pka - ph))
    return q


def isoelectric_point(seq: str) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if net_charge_at_ph(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def describe(seq: str) -> dict:
    n = len(seq)
    return {
        "gravy": round(sum(KYTE_DOOLITTLE.get(a, 0.0) for a in seq) / n, 4),
        "aromatic_fraction": round(sum(a in AROMATIC for a in seq) / n, 4),
        "positive_fraction": round(sum(a in POSITIVE for a in seq) / n, 4),
        "negative_fraction": round(sum(a in NEGATIVE for a in seq) / n, 4),
        "net_charge_ph7_4": round(net_charge_at_ph(seq), 4),
        "isoelectric_point": isoelectric_point(seq),
        "tryptophan_count": seq.count("W"),
        "arginine_count": seq.count("R"),
    }


FEATURES = ["gravy", "aromatic_fraction", "positive_fraction", "negative_fraction",
            "net_charge_ph7_4", "isoelectric_point", "tryptophan_count",
            "arginine_count"]


def analyse(sequences, scores, plastic: str) -> dict:
    from ..train.metrics import pearson, spearman
    props = [describe(s) for s in sequences]
    scores = np.asarray(scores, float)
    corr = {}
    for f in FEATURES:
        v = np.array([p[f] for p in props], float)
        corr[f] = {"pearson_with_score": pearson(v, scores),
                   "spearman_with_score": spearman(v, scores),
                   "mean": round(float(v.mean()), 4),
                   "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)}
    return {"plastic": plastic, "n": len(sequences), "correlations": corr,
            "per_peptide": [{"sequence": s, "score": round(float(sc), 4), **p}
                            for s, sc, p in zip(sequences, scores, props)]}


def save(results: list[dict], out_dir) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "physicochemical.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    head = "| Polymer | " + " | ".join(FEATURES) + " |"
    rows = [head, "| --- |" + " --- |" * len(FEATURES)]
    for r in results:
        rows.append(f"| {r['plastic']} | " + " | ".join(
            f"{r['correlations'][f]['mean']} (r={r['correlations'][f]['pearson_with_score']})"
            for f in FEATURES) + " |")
    (out_dir / "physicochemical_table.md").write_text("\n".join(rows))
