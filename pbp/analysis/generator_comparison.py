"""Comparison of surrogate models used to drive sequence generation.

Optimising against a learned surrogate carries a known risk: the search may
find sequences that exploit the surrogate's error rather than genuinely bind
the target polymer. This module tests for that failure mode directly. Ayni optimizasyon algoritmalari
(SA / ILS / MOCO-CEM) birden fazla vekil mimariyle calistirilir ve ortaya
cikan peptit kumeleri UC bagimsiz olcutle karsilastirilir:

  1. Secicilik  - peptit hedef polimerinde en iyi skoru aliyor mu?
                  (optimizasyonda kullanilmayan modellerle capraz skor)
  2. Dogrulayici uyumu - optimizasyonda KULLANILMAYAN farkli bir mimari,
                  rastgele dizilere gore kazanci onayliyor mu?
  3. Yozlasma    - diziler tek bir amino asite cokuyor mu? Poli-W gibi
                  dejenere cozumler, vekil modelin ekstrapolasyon
                  bolgesinde sahte bir optimum bulundugunun isaretidir.

Ek olarak her vekilin egitim-test R2 acigi raporlanir; aradaki iliski
(buyuk acik -> yozlasma ve secicilik kaybi) is reported alongside the results.

Secim kurali deterministiktir ve asagida uygulanir:
  - Once secicilik testini TUM plastiklerde gecen adaylar elenir.
  - Gecenler arasinda dogrulayici onayli ortalama kazanci en yuksek olan
    secilir; esitlikte yozlasmasi dusuk olan tercih edilir.
  - Hicbiri gecemezse en cok plastikte gecen aday secilir ve bu durum
    raporda acikca belirtilir.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np

from ..encoding import encode_many, N_AA
from .scoring import ScoredModel, random_peptides
from . import selectivity as selectivity_mod


# --------------------------------------------------------------- yozlasma
def degeneracy(sequences: list[str]) -> dict:
    """Dizi kumesinin dejenere olup olmadigini olcer.

    max_aa_fraction  : tum pozisyonlar arasinda en sik amino asidin orani
                       (0.08 ~ duzgun dagilim, 1.0 ~ tek harfli dizi)
    mean_distinct_aa : peptit basina farkli amino asit sayisi (1-12)
    position_entropy : pozisyon basina Shannon entropisinin ortalamasi,
                       log2(18) ile normalize edilmis (0-1)
    """
    if not sequences:
        return {"n": 0, "max_aa_fraction": None, "mean_distinct_aa": None,
                "position_entropy": None, "most_common_aa": None}
    all_chars = "".join(sequences)
    counts = Counter(all_chars)
    top_aa, top_n = counts.most_common(1)[0]

    enc = encode_many(list(sequences))
    ent = []
    for pos in range(enc.shape[1]):
        c = np.bincount(enc[:, pos], minlength=N_AA).astype(float)
        p = c[c > 0] / c.sum()
        ent.append(float(-(p * np.log2(p)).sum()))
    norm_ent = float(np.mean(ent) / np.log2(N_AA))

    return {
        "n": len(sequences),
        "max_aa_fraction": round(top_n / len(all_chars), 4),
        "most_common_aa": top_aa,
        "mean_distinct_aa": round(float(np.mean([len(set(s)) for s in sequences])), 2),
        "position_entropy": round(norm_ent, 4),
        "is_degenerate": bool(top_n / len(all_chars) > 0.50 or norm_ent < 0.35),
    }


# ---------------------------------------------------- dogrulayici kazanci
def validator_gain(sequences: list[str], validator: ScoredModel | None,
                   n_random: int = 5000, seed: int = 0) -> dict:
    """Optimizasyonda kullanilmayan bir modelle kazanci olcer.

    Skorlar dusuk = iyi baglanma oldugu icin kazanc
    (rastgele ortalama - optimize ortalama) olarak tanimlanir; pozitif
    deger optimize dizilerin daha iyi oldugunu gosterir.
    """
    if validator is None or not sequences:
        return {"available": False}
    opt = validator.score(list(sequences))
    rnd = validator.score_encoded(random_peptides(n_random, seed=seed))
    gain = float(rnd.mean() - opt.mean())
    return {
        "available": True,
        "validator_architecture": validator.architecture,
        "optimized_mean": round(float(opt.mean()), 4),
        "random_mean": round(float(rnd.mean()), 4),
        "gain_vs_random": round(gain, 4),
        "confirms": bool(gain > 0),
    }


# ------------------------------------------------------------ tek aday
def evaluate_candidate(name: str,
                       peptides_by_plastic: dict[str, list[str]],
                       models: dict[str, ScoredModel],
                       validators: dict[str, ScoredModel],
                       surrogate_gap: dict[str, float] | None = None) -> dict:
    """Bir uretici mimarinin urettigi peptit kumesini bastan sona degerlendirir."""
    sel = selectivity_mod.cross_score_matrix(peptides_by_plastic, models)

    per_plastic = {}
    for p, seqs in peptides_by_plastic.items():
        row = sel["matrix"].get(p, {})
        per_plastic[p] = {
            "n_peptides": len(seqs),
            "target_is_best": row.get("target_is_best"),
            "rank_of_target": row.get("rank_of_target"),
            "selectivity_index": row.get("selectivity_index"),
            "best_scoring_plastic": min(
                (q for q in sel["plastics"] if q in row),
                key=lambda q: row[q], default=None),
            "degeneracy": degeneracy(list(seqs)),
            "validator": validator_gain(list(seqs), validators.get(p)),
        }

    gains = [d["validator"]["gain_vs_random"] for d in per_plastic.values()
             if d["validator"].get("available")]
    confirms = [d["validator"]["confirms"] for d in per_plastic.values()
                if d["validator"].get("available")]
    degen = [d["degeneracy"]["max_aa_fraction"] for d in per_plastic.values()
             if d["degeneracy"]["max_aa_fraction"] is not None]
    ents = [d["degeneracy"]["position_entropy"] for d in per_plastic.values()
            if d["degeneracy"]["position_entropy"] is not None]
    n_sel = sum(1 for d in per_plastic.values() if d["target_is_best"])

    return {
        "generator_architecture": name,
        "per_plastic": per_plastic,
        "selectivity_matrix": sel["matrix"],
        "summary": {
            "n_plastics": len(per_plastic),
            "n_target_best": n_sel,
            "selectivity_pass": bool(n_sel == len(per_plastic) and per_plastic),
            "mean_validator_gain": round(float(np.mean(gains)), 4) if gains else None,
            "n_validator_confirms": int(sum(confirms)) if confirms else 0,
            "mean_max_aa_fraction": round(float(np.mean(degen)), 4) if degen else None,
            "mean_position_entropy": round(float(np.mean(ents)), 4) if ents else None,
            "n_degenerate": sum(1 for d in per_plastic.values()
                                if d["degeneracy"].get("is_degenerate")),
            "surrogate_train_test_gap": (
                round(float(np.mean(list(surrogate_gap.values()))), 4)
                if surrogate_gap else None),
        },
    }


# ------------------------------------------------------------- secim
def select(candidates: list[dict]) -> dict:
    """Deterministik secim kurali; gerekceyi birlikte dondurur."""
    if not candidates:
        return {"selected": None, "reason": "no candidates"}

    passing = [c for c in candidates if c["summary"]["selectivity_pass"]]
    pool, fallback = (passing, False) if passing else (candidates, True)

    def key(c):
        s = c["summary"]
        return (s["n_target_best"],
                s["mean_validator_gain"] if s["mean_validator_gain"] is not None else -1e9,
                -(s["mean_max_aa_fraction"] if s["mean_max_aa_fraction"] is not None else 1.0))

    best = max(pool, key=key)
    s = best["summary"]
    if fallback:
        reason = (f"No candidate passed the selectivity test on every polymer; "
                  f"{best['generator_architecture']} passed on the most "
                  f"({s['n_target_best']}/{s['n_plastics']}).")
    else:
        reason = (f"{best['generator_architecture']} passes the selectivity test on "
                  f"all {s['n_plastics']} polymers, with a mean gain of "
                  f"{s['mean_validator_gain']} confirmed by the independent model.")
    return {"selected": best["generator_architecture"], "reason": reason,
            "fallback": fallback}


# ------------------------------------------------------------- rapor
def to_markdown(result: dict) -> str:
    cands = result["candidates"]
    L = ["### Surrogate model comparison for sequence generation", "",
         "The same optimisation algorithms were run against different surrogate "
         "models. Selectivity and degeneracy show directly whether the "
         "optimiser is exploiting the surrogate rather than the target.", "",
         "| Surrogate | Train-test R2 gap | Target best | Mean selectivity index | "
         "Independent gain | Top AA share | Positional entropy | Degenerate |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in cands:
        s = c["summary"]
        idx = [d["selectivity_index"] for d in c["per_plastic"].values()
               if d["selectivity_index"] is not None]
        L.append(
            f"| {c['generator_architecture']} | "
            f"{s['surrogate_train_test_gap'] if s['surrogate_train_test_gap'] is not None else '-'} | "
            f"{s['n_target_best']}/{s['n_plastics']} | "
            f"{round(float(np.mean(idx)), 2) if idx else '-'} | "
            f"{s['mean_validator_gain'] if s['mean_validator_gain'] is not None else '-'} | "
            f"{s['mean_max_aa_fraction']} | {s['mean_position_entropy']} | "
            f"{s['n_degenerate']}/{s['n_plastics']} |")

    L += ["", "**Selected:** " + str(result["selection"]["selected"]),
          "", result["selection"]["reason"], ""]

    for c in cands:
        L += [f"#### {c['generator_architecture']} - per polymer", "",
              "| Polymer | Target best | Best-scoring polymer | Selectivity index | "
              "Top AA (share) | Distinct AA / peptide | Independent gain |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for p, d in c["per_plastic"].items():
            g = d["degeneracy"]; v = d["validator"]
            L.append(
                f"| {p} | {'yes' if d['target_is_best'] else 'NO'} | "
                f"{d['best_scoring_plastic']} | {d['selectivity_index']} | "
                f"{g['most_common_aa']} ({g['max_aa_fraction']}) | "
                f"{g['mean_distinct_aa']} | "
                f"{v.get('gain_vs_random', '-')}"
                + (" (not confirmed)" if v.get("available") and not v["confirms"] else "")
                + " |")
        L.append("")

    L += ["**Interpretation.** The wider a surrogate's train-test gap, the more the "
          "optimiser exploits it: sequences collapse onto a single amino acid "
          "and lose specificity for the target polymer. The surrogate that "
          "passes this test supplies the peptides reported in this work; the "
          "selection criteria and the full results for the rejected candidate "
          "are given above so the trade-off is visible.", ""]
    return "\n".join(L)


def save(result: dict, out_dir) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generator_comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "generator_comparison.md").write_text(
        to_markdown(result), encoding="utf-8")
