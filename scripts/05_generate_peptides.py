#!/usr/bin/env python3
"""Adim 4: SA / ILS / MOCO-CEM ile aday peptit uretimi.

Uretim, yapilandirmada listelenen TUM aday vekil mimarileriyle ayri ayri
calistirilir (varsayilan: cnn ve encdec). Boylece bir sonraki adim,
ogrenilmis vekilin optimizasyon tarafindan somurulup somurulmedigini
karsilastirmali olarak test edebilir (surrogate exploitation check).

    python scripts/05_generate_peptides.py
    python scripts/05_generate_peptides.py --architecture cnn
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit
from pbp.config import load_config
from pbp.device import configure
from pbp.analysis.scoring import ScoredModel
from pbp.optimize.algorithms import run_all


def generate_for(arch, plastics, cfg, strategy, models_dir, out_dir):
    g = cfg["generation"]
    all_out = {}
    for p in plastics:
        ck = models_dir / f"{p}_{arch}_{strategy}.pt"
        if not ck.exists():
            print(f"  ! {ck} yok, atlaniyor (once 04_train_final.py calistirin)")
            continue
        m = ScoredModel(ck)
        print(f"\n=== URETIM {p} ({m.architecture}/{strategy}) ===", flush=True)
        res = run_all(m.score_encoded, n_peptides=g["n_peptides_per_method"],
                      seed=0, max_iterations=g["max_iterations"])
        for name, r in res.items():
            print(f"  {name:9s} en iyi {r['best']['score']:8.2f}  {r['best']['sequence']}"
                  f"  ort {r['mean_score']:8.2f}  cesitlilik {r['diversity_ratio']}")
        all_out[p] = res
        (out_dir / f"{p}_{arch}_{strategy}.json").write_text(json.dumps(res, indent=2))

    flat = {p: sorted({r["sequence"] for m in res.values() for r in m["runs"]})
            for p, res in all_out.items()}
    path = out_dir / f"peptides_{arch}_{strategy}.json"
    path.write_text(json.dumps(flat, indent=2))
    print(f"\n{arch}: toplam {sum(len(v) for v in flat.values())} benzersiz peptit -> {path.name}")
    return flat


def main():
    cfg = load_config()
    an = cfg.get("analysis", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--plastic", default="ALL")
    ap.add_argument("--architecture", default=None,
                    help="tek mimari; verilmezse yapilandirmadaki tum adaylar")
    ap.add_argument("--strategy", default="clustered")
    a = ap.parse_args()

    print("cihaz:", configure(), flush=True)
    plastics = cfg["plastics"] if a.plastic == "ALL" else [a.plastic]
    archs = ([a.architecture] if a.architecture else
             an.get("generator_candidates",
                    [an.get("best_architecture", "encdec")]))
    models_dir = Path(cfg["out_dir"]) / "models"
    out_dir = Path(cfg["out_dir"]) / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    for arch in archs:
        print(f"\n{'#' * 60}\n# URETICI VEKIL: {arch}\n{'#' * 60}", flush=True)
        generate_for(arch, plastics, cfg, a.strategy, models_dir, out_dir)

    (out_dir / f"_generated_architectures_{a.strategy}.json").write_text(
        json.dumps(archs, indent=2))


if __name__ == "__main__":
    main()
    clean_exit(0)
