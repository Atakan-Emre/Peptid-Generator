#!/usr/bin/env python3
"""Adim 12: laboratuvarda docking yapilacak peptit kumesini secer.

    python scripts/12_select_docking_set.py [--per-polymer 12]

Neden rastgele secmiyoruz
-------------------------
Docking, hakemin istedigi gibi ML skoruyla KORELE edilecekse, secilen
kumenin tahmin skoru genis bir araliga yayilmasi gerekir. Yalnizca en iyi
peptitler docking'e sokulursa skorlar dar bir bantta toplanir ve korelasyon
katsayisi aralik kisitlamasi yuzunden anlamsizlasir; orijinal gonderimdeki
sorun tam olarak buydu.

Bu yuzden her polimer icin uc grup secilir:

  designed  Yeni uretilen 30 peptitten, uretilen skor araligini tarayacak
            sekilde secilenler (en iyisi + ceyreklikler).
  training  Egitim bolmesinden OLCULMUS skoru bilinen diziler. Bunlar hem
            arligi asagi dogru genisletir hem de docking'i tahmine degil
            olculen degere karsi dogrulama imkani verir.
  random    Rastgele 12-mer'ler: null uc. Hakem 2'nin ikinci maddesindeki
            "uygun baseline/rastgele diziler" talebini docking tarafinda da
            karsilar.

Cikti: results/analysis/docking_selection.csv  (laboratuvara verilecek dosya)
       results/analysis/docking_selection.md   (gerekce + ozet tablo)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit
from pbp.config import load_config
from pbp.encoding import ALPHABET, PEPTIDE_LENGTH
from pbp.analysis.scoring import load_all_models

# Olculen egitim dizilerinin secilecegi yuzdelikler. Guclu uctan zayif uca
# dogru: korelasyon icin gereken dinamik araligi bunlar saglar.
TRAIN_PERCENTILES = [0.01, 1.0, 25.0, 50.0, 90.0]


def designed_peptides(gen_path: Path, k: int,
                      selective: set[str] | None = None) -> list[tuple[str, float, str]]:
    """Uretilen 30 peptitten skor araligini tarayan k tanesini secer.

    `selective` verilirse, once kendi hedefinde en iyi skorlayan peptitler
    arasindan secilir. Hedefinde en iyi olmayan bir peptidi o polimere karsi
    docking'e sokmak laboratuvar zamanini bosa harcar; hakem de bunu sorar.
    """
    data = json.loads(gen_path.read_text(encoding="utf-8"))
    rows = {}
    for method, block in data.items():
        for r in block["runs"]:
            # Ayni dizi birden fazla yontemden gelebilir; en iyi skoru tut.
            s, m = r["score"], method
            if r["sequence"] not in rows or s < rows[r["sequence"]][0]:
                rows[r["sequence"]] = (s, m)
    ordered = sorted(((s, seq, m) for seq, (s, m) in rows.items()))
    if selective:
        kept = [t for t in ordered if t[1] in selective]
        # Secici peptit sayisi yeterliyse yalnizca onlari kullan.
        if len(kept) >= k:
            ordered = kept
    if k >= len(ordered):
        picks = ordered
    else:
        # En iyisi her zaman icerde; kalanlar araliga esit yayilir.
        idx = [0] + [round(i * (len(ordered) - 1) / (k - 1)) for i in range(1, k)]
        picks = [ordered[i] for i in sorted(set(idx))]
    # Tasarlanan peptitlerin OLCULMUS skoru yoktur; uretim skoru tahmindir
    # ve asagida yeniden hesaplanir. measured alani bos birakilir.
    return [(seq, float("nan"), f"designed, {m}") for s, seq, m in picks]


def training_peptides(csv_path: Path, npz_path: Path, pcts) -> list[tuple[str, float, str]]:
    """Egitim bolmesinden, olculen skor dagiliminin yuzdeliklerinde diziler."""
    df = pd.read_csv(csv_path)
    df["Sequence"] = df["Sequence"].astype(str).str.strip().str.upper()
    df = df.groupby("Sequence", as_index=False)["Score"].mean()

    # Yalnizca egitim bolmesindeki diziler: dogrulama/test sizmasin.
    z = np.load(npz_path)
    from pbp.encoding import decode_many
    train_seqs = set(decode_many(z["X"][z["train"]]))
    df = df[df["Sequence"].isin(train_seqs)]

    df = df.sort_values("Score").reset_index(drop=True)
    out, used = [], set()
    for p in pcts:
        i = min(len(df) - 1, max(0, int(round((p / 100.0) * (len(df) - 1)))))
        while i in used and i + 1 < len(df):
            i += 1
        used.add(i)
        r = df.iloc[i]
        out.append((r["Sequence"], float(r["Score"]),
                    f"training, measured p{p:g}"))
    return out


def random_peptides(k: int, seed: int) -> list[tuple[str, float, str]]:
    rng = np.random.default_rng(seed)
    letters = list(ALPHABET)
    seen, out = set(), []
    while len(out) < k:
        s = "".join(rng.choice(letters, size=PEPTIDE_LENGTH))
        if s in seen:
            continue
        seen.add(s)
        out.append((s, float("nan"), "random control"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-polymer", type=int, default=12,
                    help="polimer basina toplam peptit (varsayilan 12)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = load_config()
    out_root = Path(cfg["out_dir"])
    prep = Path(cfg["prep_dir"])
    seed_cfg = cfg["prepare"]["split_seed"]
    arch = json.loads((out_root / "analysis" / "generator_comparison.json")
                      .read_text(encoding="utf-8"))["selection"]["selected"]

    # Gruplarin buyuklugu: tasarlanan ve olculen agirlikli, birkac rastgele.
    n_rand = max(2, round(a.per_polymer * 0.25))
    n_train = min(len(TRAIN_PERCENTILES), round(a.per_polymer * 0.42))
    n_des = a.per_polymer - n_rand - n_train

    print(f"Uretici mimari: {arch} | polimer basina {a.per_polymer} peptit "
          f"({n_des} tasarlanan + {n_train} olculen + {n_rand} rastgele)\n")

    models = load_all_models(out_root / "models", architecture=arch,
                             split_strategy="clustered")

    # Peptit bazinda secicilik: hedefinde en iyi skorlayanlar ve indeksler.
    import csv as _csv
    with (out_root / "analysis" / "selectivity_per_peptide.csv").open(encoding="utf-8") as fh:
        per_pep = {(r["optimized_for"], r["sequence"]): r for r in _csv.DictReader(fh)}
    selective = {p: {seq for (tgt, seq), r in per_pep.items()
                     if tgt == p and r["best_scoring_plastic"] == p}
                 for p in cfg["plastics"]}
    for p in cfg["plastics"]:
        print(f"  {p:6} hedefinde en iyi olan tasarim: {len(selective[p])}/30")
    print()

    rows = []
    for p in cfg["plastics"]:
        picks = []
        picks += designed_peptides(
            out_root / "generated" / f"{p}_{arch}_clustered.json", n_des,
            selective=selective[p])
        picks += training_peptides(
            Path(cfg["data_dir"]) / f"{p}.csv",
            prep / f"{p}_clustered_seed{seed_cfg}.npz",
            TRAIN_PERCENTILES[:n_train])
        picks += random_peptides(n_rand, a.seed + hash(p) % 1000)

        seqs = [s for s, _, _ in picks]
        pred = models[p].score(seqs)
        for (seq, measured, why), pr in zip(picks, pred):
            rows.append({
                "plastic": p,
                "sequence": seq,
                "group": why.split(",")[0],
                "selection_reason": why,
                "predicted_score": round(float(pr), 4),
                "measured_score": ("" if measured != measured else round(measured, 4)),
                "selectivity_index": (per_pep[(p, seq)]["selectivity_index"]
                                      if (p, seq) in per_pep else ""),
                "target_selective": ("yes" if (p, seq) in per_pep
                                     and per_pep[(p, seq)]["best_scoring_plastic"] == p
                                     else ("no" if (p, seq) in per_pep else "")),
                # Laboratuvarin dolduracagi sutunlar:
                "docking_score": "", "oligomer_length": "", "n_runs": "",
                "software": "", "notes": "",
            })
        lo, hi = pred.min(), pred.max()
        print(f"  {p:6} {len(picks):>2} peptit | tahmin araligi "
              f"{lo:7.2f} .. {hi:7.2f}  (genislik {hi - lo:.1f})")

    an = out_root / "analysis"
    an.mkdir(parents=True, exist_ok=True)
    csv_path = an / "docking_selection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Ozet / gerekce
    df = pd.DataFrame(rows)
    md = ["# Peptides selected for molecular docking", "",
          f"Surrogate used for the predicted score: **{arch}**, identity-aware split.",
          f"{len(rows)} peptides, {a.per_polymer} per polymer.", "",
          "## Why this composition", "",
          "Docking is being repeated so that the relationship between the "
          "predicted score and the docking score can be reported, as requested "
          "in review. That correlation is only interpretable if the docked set "
          "spans a wide range of predicted affinity. Docking only the "
          "best-scoring designs would restrict the range and make any "
          "coefficient meaningless.", "",
          "| Group | n per polymer | Role |", "| --- | --- | --- |",
          f"| Designed | {n_des} | New peptides, spanning the generated score range |",
          f"| Training | {n_train} | Sequences with a **measured** affinity, spanning the "
          "measured distribution; these let docking be compared against "
          "experiment, not only against the model |",
          f"| Random | {n_rand} | Null end of the range; also the baseline control "
          "asked for in review |", "",
          "## Dynamic range achieved", "",
          "| Polymer | n | Predicted score range | Width |",
          "| --- | --- | --- | --- |"]
    for p in cfg["plastics"]:
        g = df[df["plastic"] == p]["predicted_score"]
        md.append(f"| {p} | {len(g)} | {g.min():.2f} to {g.max():.2f} | "
                  f"{g.max() - g.min():.1f} |")
    md += ["", "## The list", "",
           "| Polymer | Sequence | Group | Predicted | Measured | Why |",
           "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(f"| {r['plastic']} | `{r['sequence']}` | {r['group']} | "
                  f"{r['predicted_score']:.2f} | {r['measured_score'] or '-'} | "
                  f"{r['selection_reason']} |")
    md += ["", "## What the laboratory returns", "",
           "Fill `docking_score`, `oligomer_length`, `n_runs`, `software` and "
           "`notes` in `docking_selection.csv`, keeping one row per "
           "peptide-polymer pair, then run:", "",
           "```bash",
           "python scripts/07_run_analyses.py --docking-csv "
           "results/analysis/docking_selection.csv",
           "python scripts/10_build_figures.py",
           "python scripts/11_review_compliance.py",
           "```", "",
           "Report the docking score as the best (most negative) binding energy "
           "of the run, in kcal/mol, and keep the oligomer length constant "
           "within a polymer so the scores stay comparable.", ""]
    (an / "docking_selection.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n{len(rows)} peptit secildi")
    print(f"  {csv_path}")
    print(f"  {an / 'docking_selection.md'}")
    return 0


if __name__ == "__main__":
    clean_exit(main())
