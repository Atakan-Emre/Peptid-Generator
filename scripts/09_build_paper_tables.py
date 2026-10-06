#!/usr/bin/env python3
"""Adim 9: makaleye girecek tablolari uretir.

    python scripts/09_build_paper_tables.py
"""
import json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit

from pbp.config import load_config


def table_dataset_stats(cfg) -> str:
    """Tablo 2 ve 4'un yerine gecer (recomputed from the data)."""
    rows = ["| Polymer | Raw samples | After deduplication | Train | Val | Test | "
            "Score mean (train) | Score SD (train) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in cfg["plastics"]:
        f = Path(cfg["prep_dir"]) / f"{p}_clustered_seed{cfg['prepare']['split_seed']}.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        d, s, n = m["dedup"], m["split_sizes"], m["normalization"]
        rows.append(f"| {p} | {d['n_raw']:,} | {m['n_after_prep']:,} | {s['train']:,} | "
                    f"{s['val']:,} | {s['test']:,} | {n['mean']:.4f} | {n['std']:.4f} |")
    return "\n".join(rows) + ("\n\nNormalisation statistics are computed from the "
                              "training split only.\n")


def table_split_comparison(cfg) -> str:
    """Yeni tablo: rastgele vs kume bazli bolme (random vs identity-aware splitting)."""
    f = Path(cfg["prep_dir"]) / "leakage_summary.json"
    if not f.exists():
        return "_leakage_summary.json missing; run 01_prepare_data.py first._\n"
    data = json.loads(f.read_text())
    rows = ["| Polymer | Split | Exact test-in-train (%) | "
            "Mean NN identity (%) | NN <= 2 mutations (%) |",
            "| --- | --- | --- | --- | --- |"]
    for r in sorted(data, key=lambda x: (x["plastic"], x["strategy"])):
        rows.append(f"| {r['plastic']} | {r['strategy']} | {r['exact_test_in_train_pct']} | "
                    f"{r['nn_identity_mean_pct']} | {r['pct_nn_le_2']} |")
    return "\n".join(rows)


def table_full_metrics(cfg, strategy) -> str:
    """Her mimari x plastik icin train/val/test setlerinin R2, RMSE ve MAE'si.

    Hakem 2, ucuncu maddede bu dokuz metrigin tamamini istemisti; tablo
    kisaltilmadan verilir.
    """
    f = Path(cfg["out_dir"]) / f"summary_{strategy}.json"
    if not f.exists():
        return f"_summary_{strategy}.json missing._\n"
    data = json.loads(f.read_text())
    sets = ("train", "val", "test")
    head = ["Polymer", "Architecture"]
    for st in sets:
        lbl = {"train": "Train", "val": "Validation", "test": "Test"}[st]
        head += [f"{lbl} R2", f"{lbl} RMSE", f"{lbl} MAE"]
    head.append("Seeds")
    rows = ["| " + " | ".join(head) + " |",
            "| " + " | ".join("---" for _ in head) + " |"]
    for s in sorted(data, key=lambda x: (x["plastic"], x["architecture"])):
        cells = [s["plastic"], s["architecture"]]
        for st in sets:
            cells.append(f"{s[f'{st}_r2_mean']:.4f} ± {s[f'{st}_r2_std']:.4f}")
            cells.append(f"{s[f'{st}_rmse_mean']:.3f} ± {s[f'{st}_rmse_std']:.3f}")
            cells.append(f"{s[f'{st}_mae_mean']:.3f} ± {s[f'{st}_mae_std']:.3f}")
        cells.append(str(s["n_seeds"]))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + (
        "\n\nMean ± standard deviation over the seeds. RMSE and MAE are in "
        "the units of the binding score.\n")


def table_arch_stats(cfg, strategy) -> str:
    f = Path(cfg["out_dir"]) / f"architecture_stats_{strategy}.json"
    if not f.exists():
        return "_statistics file missing._\n"
    st = json.loads(f.read_text())
    ref = st.get("reference", "?")
    rows = [f"Reference architecture: **{ref}** (highest mean test R2 across "
            "all polymers). Paired across seeds: same split, same data, only "
            "the architecture differs.", "",
            "| Polymer | Comparison | Difference | 95% CI | p (paired t) | Significant |",
            "| --- | --- | --- | --- | --- | --- |"]
    for p, d in st["per_plastic"].items():
        for other, r in d.items():
            ci = r.get("difference_ci95", ["-", "-"])
            pv = r.get("paired_t_pvalue")
            pv = ("-" if pv is None else
                  "<1e-6" if pv < 1e-6 else f"{pv:.2e}" if pv < 1e-3 else f"{pv:.4f}")
            rows.append(f"| {p} | {ref} vs {other} | {r['mean_difference']:+.5f} | "
                        f"[{ci[0]}, {ci[1]}] | {pv} | "
                        f"{'yes' if r.get('significant_at_0.05') else 'NO'} |")
    return "\n".join(rows)


def main():
    cfg = load_config()
    out = Path(cfg["out_dir"]) / "paper_tables"; out.mkdir(parents=True, exist_ok=True)
    an = Path(cfg["out_dir"]) / "analysis"

    parts = {
        "T1_dataset_statistics.md": table_dataset_stats(cfg),
        "T2_split_comparison.md": table_split_comparison(cfg),
        "T3_full_metrics_clustered.md": table_full_metrics(cfg, "clustered"),
        "T3b_full_metrics_random.md": table_full_metrics(cfg, "random"),
        "T4_architecture_significance.md": table_arch_stats(cfg, "clustered"),
    }
    jb = Path(cfg["out_dir"]) / "jain_baseline" / "jain_karsilastirma.md"
    parts["T9_jain_benchmark.md"] = (jb.read_text(encoding="utf-8")
                                     if jb.exists() else "_Prior-work benchmark missing._\n")
    gcp = an / "generator_comparison.md"
    parts["T10_generator_comparison.md"] = (
        gcp.read_text(encoding="utf-8") if gcp.exists()
        else "_Surrogate comparison missing._\n")

    for src, dst in [("selectivity_matrix.md", "T5_selectivity.md"),
                     ("novelty_table.md", "T6_novelty.md"),
                     ("physicochemical_table.md", "T7_physicochemical.md"),
                     ("position_importance.md", "T8_position_importance.md")]:
        p = an / src
        parts[dst] = p.read_text(encoding="utf-8") if p.exists() else f"_{src} missing._\n"


    # T11 yalnizca laboratuvar docking skorlari geldiginde olusur.
    dc = an / "docking_correlation.md"
    if dc.exists():
        parts["T11_docking_correlation.md"] = dc.read_text(encoding="utf-8")

    # T1..T11 sirasiyla yaz (sozluk ekleme sirasi degil, tablo numarasi).
    parts = dict(sorted(parts.items(),
                        key=lambda kv: int(re.match(r"T(\d+)", kv[0]).group(1))))

    for name, content in parts.items():
        (out / name).write_text(content, encoding="utf-8")
        print(f"  {name}")
    def demote(text: str) -> str:
        """Gomulu parcanin baslik hiyerarsisini bolum basliginin altina tasir.

        Parcanin en ust basligi "###" olacak sekilde hepsi ayni miktarda
        kaydirilir; aksi halde bir tablonun kendi basligi, ALL_TABLES icinde
        o tablonun kardesi gibi gorunur.
        """
        levels = [len(ln) - len(ln.lstrip("#"))
                  for ln in text.splitlines() if ln.startswith("#")]
        if not levels:
            return text
        shift = max(0, 3 - min(levels))
        return "\n".join(("#" * shift + ln) if ln.startswith("#") else ln
                          for ln in text.splitlines())

    (out / "ALL_TABLES.md").write_text(
        "# Paper tables\n\n" + "\n\n".join(
            f"## {k.replace('.md','')}\n\n{demote(v)}" for k, v in parts.items()),
        encoding="utf-8")
    print(f"\nTablolar: {out}")


if __name__ == "__main__":
    main()
    clean_exit(0)
