"""Paired statistical comparison of architectures across seeds.

A difference in mean R2 is not evidence on its own: with a handful of seeds,
run-to-run variation can be larger than the gap between two architectures.
This module therefore compares architectures seed by seed -- same split, same
data, only the architecture differs: ayni bolme, ayni
veri, yalnizca mimari farkli. Eslesmis t-testi ve Wilcoxon isaretli siralar
testi ile birlikte etki buyuklugu (Cohen's d) ve guven araligi verilir.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def paired_comparison(a_scores, b_scores, name_a="A", name_b="B",
                      alpha: float = 0.05) -> dict:
    from scipy import stats
    a, b = np.asarray(a_scores, float), np.asarray(b_scores, float)
    if len(a) != len(b):
        raise ValueError("Eslesmis karsilastirma icin ayni sayida tohum gerekir")
    d = a - b
    n = len(d)
    out = {"comparison": f"{name_a} vs {name_b}", "n_seeds": n,
           f"{name_a}_mean": round(float(a.mean()), 6),
           f"{name_a}_std": round(float(a.std(ddof=1)) if n > 1 else 0.0, 6),
           f"{name_b}_mean": round(float(b.mean()), 6),
           f"{name_b}_std": round(float(b.std(ddof=1)) if n > 1 else 0.0, 6),
           "mean_difference": round(float(d.mean()), 6)}
    if n < 3:
        out["note"] = "Anlamlilik testi icin en az 3 tohum gerekir"
        return out

    t = stats.ttest_rel(a, b)
    out["paired_t_statistic"] = round(float(t.statistic), 4)
    out["paired_t_pvalue"] = round(float(t.pvalue), 6)
    try:
        w = stats.wilcoxon(a, b)
        out["wilcoxon_pvalue"] = round(float(w.pvalue), 6)
    except ValueError:
        out["wilcoxon_pvalue"] = None

    sd = d.std(ddof=1)
    out["cohens_d"] = round(float(d.mean() / sd), 4) if sd > 0 else None
    se = sd / np.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    out["difference_ci95"] = [round(float(d.mean() - crit * se), 6),
                              round(float(d.mean() + crit * se), 6)]
    out["significant_at_0.05"] = bool(out["paired_t_pvalue"] < alpha)
    out["verdict"] = (
        f"{name_a}, {name_b}'den anlamli olarak farkli (p={out['paired_t_pvalue']})"
        if out["significant_at_0.05"] else
        f"Fark istatistiksel olarak anlamli degil (p={out['paired_t_pvalue']}); "
        f"iki mimari bu veride es deger kabul edilmelidir")
    return out


def best_architecture(summaries: list[dict], metric="test_r2") -> str:
    """Architecture with the highest mean score across all plastics."""
    agg = {}
    for s in summaries:
        agg.setdefault(s["architecture"], []).append(s[f"{metric}_mean"])
    return max(agg, key=lambda a: sum(agg[a]) / len(agg[a]))


def compare_architectures(summaries: list[dict], metric="test_r2",
                          reference=None) -> dict:
    # Reference defaults to the empirically best architecture, so the table
    # never hard-codes an assumption about which model wins.
    if reference is None:
        reference = best_architecture(summaries, metric)
    """Her plastik icin referans mimariyi digerleriyle karsilastirir."""
    by_plastic: dict[str, dict[str, list]] = {}
    for s in summaries:
        key = "per_seed_" + ("test_r2" if metric == "test_r2" else "val_r2")
        by_plastic.setdefault(s["plastic"], {})[s["architecture"]] = s[key]

    out = {"metric": metric, "reference": reference, "per_plastic": {}}
    for plastic, archs in by_plastic.items():
        if reference not in archs:
            continue
        out["per_plastic"][plastic] = {
            other: paired_comparison(archs[reference], archs[other], reference, other)
            for other in archs if other != reference
        }
    sig = [(p, o) for p, d in out["per_plastic"].items()
           for o, r in d.items() if r.get("significant_at_0.05")]
    total = sum(len(d) for d in out["per_plastic"].values())
    out["summary"] = {
        "n_comparisons": total, "n_significant": len(sig),
        "significant_pairs": [f"{p}: {reference} vs {o}" for p, o in sig],
    }
    return out


def save(result: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
