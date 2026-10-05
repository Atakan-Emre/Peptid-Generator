"""Cross-polymer selectivity analysis.

Her plastik icin optimize edilmis peptitler, ALTI plastigin modelinin
hepsiyle skorlanir. Sonuc 6x6 matris: satir = peptidin optimize edildigi
plastik, sutun = skorlamayi yapan model.

Secicilik indeksi = (diger plastiklerin ortalama skoru) - (hedef plastik
skoru). Pozitif deger hedefe ozgulugu, sifira yakin deger genis spektrumlu
baglanmayi gosterir.

Not: alti plastigin veri setleri neredeyse tamamen ayriktir (Nylon-PVC
arasinda 37 ortak dizi, diger tum ciftlerde 0), dolayisiyla capraz
tahminler gercek bir genelleme testidir.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from ..encoding import encode_many
from .scoring import ScoredModel


def cross_score_matrix(peptides_by_plastic: dict[str, list[str]],
                       models: dict[str, ScoredModel]) -> dict:
    plastics = sorted(models)
    matrix, per_peptide = {}, []

    for target in plastics:
        seqs = peptides_by_plastic.get(target, [])
        if not seqs:
            continue
        enc = encode_many(list(seqs))
        row = {}
        scores_by_model = {}
        for scorer in plastics:
            s = models[scorer].score_encoded(enc)
            scores_by_model[scorer] = s
            row[scorer] = round(float(s.mean()), 4)
        others = [row[p] for p in plastics if p != target]
        # Tek plastikli kosularda karsilastirilacak baska model yok
        row["selectivity_index"] = (
            round(float(np.mean(others) - row[target]), 4) if others else None)
        row["rank_of_target"] = int(
            1 + sum(1 for p in plastics if p != target and row[p] < row[target]))
        row["target_is_best"] = bool(row["rank_of_target"] == 1)
        matrix[target] = row

        for i, seq in enumerate(seqs):
            rec = {"optimized_for": target, "sequence": seq}
            for p in plastics:
                rec[f"score_{p}"] = round(float(scores_by_model[p][i]), 4)
            oth = [scores_by_model[p][i] for p in plastics if p != target]
            rec["selectivity_index"] = (
                round(float(np.mean(oth) - scores_by_model[target][i]), 4)
                if oth else None)
            rec["best_scoring_plastic"] = min(plastics, key=lambda p: scores_by_model[p][i])
            per_peptide.append(rec)

    n_sel = sum(1 for r in matrix.values() if r["target_is_best"])
    return {
        "plastics": plastics,
        "matrix": matrix,
        "per_peptide": per_peptide,
        "summary": {
            "n_target_best": n_sel,
            "n_targets": len(matrix),
            "mean_selectivity_index": (
                round(float(np.mean([r["selectivity_index"]
                                     for r in matrix.values()
                                     if r["selectivity_index"] is not None])), 4)
                if any(r["selectivity_index"] is not None for r in matrix.values())
                else None),
            "interpretation": (
                "Every peptide set scores best on the polymer it was optimised for"
                if n_sel == len(matrix) else
                f"for {len(matrix) - n_sel}/{len(matrix)} polymers the peptides score "
                "better on a different polymer, indicating broad-spectrum binding"),
        },
    }


def to_markdown_table(result: dict) -> str:
    ps = result["plastics"]
    lines = ["| Optimised for \\ Scored by | " + " | ".join(ps) + " | Selectivity index |",
             "| --- |" + " --- |" * (len(ps) + 1)]
    for t, row in result["matrix"].items():
        cells = [f"**{row[p]}**" if p == t else str(row[p]) for p in ps]
        lines.append(f"| {t} | " + " | ".join(cells) + f" | {row['selectivity_index']} |")
    return "\n".join(lines)


def save(result: dict, out_dir) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selectivity.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (out_dir / "selectivity_matrix.md").write_text(to_markdown_table(result))
    import csv
    if result["per_peptide"]:
        with open(out_dir / "selectivity_per_peptide.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result["per_peptide"][0]))
            w.writeheader(); w.writerows(result["per_peptide"])
