"""ML skoru - docking skoru korelasyonu (orthogonal structural validation).

If docking is presented as an independent line of validation, the relation
between predicted affinity and docking score has to be quantified rather
than asserted. This module computes that correlation.

Bu modul docking sonuclarini bir CSV'den okur ve korelasyon + siralama
uyumu raporlar. Beklenen CSV sutunlari:
    plastic, sequence, method, docking_score
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ..train.metrics import pearson, spearman
from .scoring import ScoredModel

MIN_N_FOR_CORRELATION = 8


def analyse(docking_csv, models: dict[str, ScoredModel]) -> dict:
    df = pd.read_csv(docking_csv)
    need = {"plastic", "sequence", "docking_score"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Docking CSV'sinde eksik sutun: {missing}")

    per_plastic, rows = {}, []
    for plastic, g in df.groupby("plastic"):
        if plastic not in models:
            continue
        ml = models[plastic].score(g["sequence"].astype(str).tolist())
        dk = g["docking_score"].to_numpy(float)
        for s, a, b, m in zip(g["sequence"], ml, dk, g.get("method", [""] * len(g))):
            rows.append({"plastic": plastic, "sequence": s, "method": m,
                         "ml_score": round(float(a), 4), "docking_score": float(b)})
        entry = {"n": int(len(g))}
        if len(g) >= MIN_N_FOR_CORRELATION:
            entry.update({
                "pearson": pearson(ml, dk),
                "spearman": spearman(ml, dk),
                "rank_agreement_top3": _top_k_overlap(ml, dk, 3),
            })
        else:
            entry["note"] = (f"n={len(g)} < {MIN_N_FOR_CORRELATION}; korelasyon "
                             "raporlanmadi. Plastik basina en az 10-15 peptit "
                             "docking'e sokulmalidir.")
        per_plastic[plastic] = entry

    all_ml = np.array([r["ml_score"] for r in rows])
    all_dk = np.array([r["docking_score"] for r in rows])
    pooled = {"n": len(rows)}
    if len(rows) >= MIN_N_FOR_CORRELATION:
        pooled.update({"pearson": pearson(all_ml, all_dk),
                       "spearman": spearman(all_ml, all_dk)})
    return {"per_plastic": per_plastic, "pooled": pooled, "pairs": rows,
            "caveat": ("Kisa oligomerler genisletilmis polimer yuzeyindeki "
                       "adsorpsiyonu birebir temsil etmez; docking sonuclari "
                       "baglanmanin dogrulanmasi degil, niteliksel destekleyici "
                       "kanit olarak sunulmalidir.")}


def _top_k_overlap(a, b, k: int) -> float:
    k = min(k, len(a))
    sa = set(np.argsort(a)[:k]); sb = set(np.argsort(b)[:k])
    return round(len(sa & sb) / k, 3)


def template_csv(path, peptides_by_plastic: dict[str, list[str]],
                 methods=("SA", "ILS", "MOCO-CEM")) -> None:
    """Docking calismasi icin doldurulmaya hazir CSV sablonu uretir."""
    rows = []
    for plastic, seqs in peptides_by_plastic.items():
        for s in seqs:
            rows.append({"plastic": plastic, "sequence": s, "method": "",
                         "docking_score": "", "oligomer_length": "",
                         "n_runs": "", "notes": ""})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def table_md(result: dict) -> str:
    """Makale tablosu: plastik basina korelasyon ve siralama uyumu."""
    rows = ["| Polymer | n | Pearson r | Spearman rho | Top-3 rank agreement |",
            "| --- | --- | --- | --- | --- |"]
    for plastic, e in sorted(result["per_plastic"].items()):
        if "spearman" in e:
            rows.append(f"| {plastic} | {e['n']} | {e['pearson']:.3f} | "
                        f"{e['spearman']:.3f} | {e['rank_agreement_top3']:.2f} |")
        else:
            rows.append(f"| {plastic} | {e['n']} | n/a | n/a | n/a |")
    po = result.get("pooled", {})
    if "spearman" in po:
        rows.append(f"| **Pooled** | {po['n']} | {po['pearson']:.3f} | "
                    f"{po['spearman']:.3f} | - |")
    note = ("\n\nCorrelations are reported only for polymers with at least "
            f"{MIN_N_FOR_CORRELATION} docked peptides.\n\n> {result['caveat']}\n")
    return "\n".join(rows) + note


def save(result: dict, out_dir) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "docking_correlation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    (out_dir / "docking_correlation.md").write_text(table_md(result), encoding="utf-8")
    pd.DataFrame(result["pairs"]).to_csv(out_dir / "ml_vs_docking.csv", index=False)
