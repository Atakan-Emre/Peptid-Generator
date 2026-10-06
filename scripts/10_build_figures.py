#!/usr/bin/env python3
"""Adim 10: makaleye girecek figurleri uretir.

    python scripts/10_build_figures.py

Figurler yalnizca onceki adimlarin yazdigi JSON/CSV dosyalarindan uretilir;
hicbir model yeniden calistirilmaz. Eksik bir girdi varsa o figur atlanir ve
hangi adimin calistirilmasi gerektigi yazilir.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit
from pbp.config import load_config
from pbp.reporting import style, figures

# (ad, cagri, eksikse hangi adim)
FIGURES = [
    ("F1 architecture comparison", "architecture_comparison", "04/05_train_final"),
    ("F2 split effect",            "split_effect",            "01_prepare_data + 04/05"),
    ("F3 Jain benchmark",          "jain_benchmark",          "08_jain_baseline"),
    ("F4 independent validation",  "independent_validation",  "07_run_analyses"),
    ("F5 selectivity",             "selectivity_heatmap",     "07_run_analyses"),
    ("F6 novelty",                 "novelty",                 "07_run_analyses"),
    ("F7 interpretability",        "interpretability",        "07_run_analyses"),
    ("F8 physicochemical",         "physicochemical",         "07_run_analyses"),
    ("F9 docking correlation",     "docking_correlation",
     "07_run_analyses --docking-csv <laboratuvar sonucu>"),
]


def main() -> int:
    cfg = load_config()
    out_root = Path(cfg["out_dir"])
    prep_dir = Path(cfg["prep_dir"])
    fig_dir = out_root / "figures"

    style.apply()

    # Her figurun kendi imza ihtiyaci var; tek yerden cagirmak icin eslestirme.
    calls = {
        "architecture_comparison": lambda: figures.architecture_comparison(
            out_root, fig_dir, cfg),
        "split_effect": lambda: figures.split_effect(out_root, prep_dir, fig_dir, cfg),
        "jain_benchmark": lambda: figures.jain_benchmark(out_root, fig_dir),
        "independent_validation": lambda: figures.independent_validation(
            out_root, fig_dir, cfg),
        "selectivity_heatmap": lambda: figures.selectivity_heatmap(out_root, fig_dir),
        "novelty": lambda: figures.novelty(out_root, fig_dir, cfg),
        "interpretability": lambda: figures.interpretability(out_root, fig_dir, cfg),
        "physicochemical": lambda: figures.physicochemical(out_root, fig_dir, cfg),
        "docking_correlation": lambda: figures.docking_correlation(out_root, fig_dir),
    }

    made, skipped = [], []
    for label, key, needs in FIGURES:
        try:
            path = calls[key]()
        except Exception as exc:                      # tek figur hatasi digerlerini durdurmasin
            print(f"  [HATA ] {label}: {type(exc).__name__}: {exc}", flush=True)
            skipped.append((label, f"hata: {exc}"))
            continue
        if path is None:
            print(f"  [atlan] {label}  <- once {needs} calistirilmali", flush=True)
            skipped.append((label, needs))
        else:
            print(f"  [tamam] {label}  -> {path.relative_to(out_root.parent)}", flush=True)
            made.append(label)

    print(f"\n{len(made)} figur uretildi, {len(skipped)} atlandi -> {fig_dir}")
    if skipped:
        print("Atlananlar:")
        for label, why in skipped:
            print(f"  - {label}: {why}")
    # Atlanan figur bir hata degil; eksik girdi pipeline'in onceki adimindan gelir.
    return 0


if __name__ == "__main__":
    clean_exit(main())
