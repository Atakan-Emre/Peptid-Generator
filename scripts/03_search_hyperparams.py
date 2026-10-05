#!/usr/bin/env python3
"""Adim 2 (Asama A): hiperparametre aramasi.

Alt-orneklenmis veri ve tek tohum ile en iyi konfigurasyonu bulur.
Yarida kesilirse ayni komut kaldigi yerden devam eder.

    python scripts/03_search_hyperparams.py --plastic PET
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit

from pbp.config import load_config

from pbp.device import configure
from pbp.train.experiment import ExperimentRunner


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--plastic", default="ALL")
    ap.add_argument("--architecture", default="ALL")
    ap.add_argument("--strategy", default="clustered")
    a = ap.parse_args()

    print("cihaz:", configure(), flush=True)
    plastics = cfg["plastics"] if a.plastic == "ALL" else [a.plastic]
    archs = cfg["architectures"] if a.architecture == "ALL" else [a.architecture]

    runner = ExperimentRunner(cfg["prep_dir"], cfg["out_dir"])
    s = cfg["search"]
    for p in plastics:
        for arch in archs:
            print(f"\n=== ARAMA {p} / {arch} / {a.strategy} ===", flush=True)
            r = runner.search(p, arch, strategy=a.strategy,
                              split_seed=cfg["prepare"]["split_seed"],
                              subsample_n=s["subsample_n"],
                              max_epochs=s["max_epochs"], patience=s["patience"])
            print(f"  en iyi val R2 = {r['best_val_r2']:.5f}")
            print(f"  konfig: { {k: v for k, v in r['best_config'].items() if k in ('hidden_dim','num_layers','dropout','latent_dim','n_filters','kernel_size','lambda_recon')} }")


if __name__ == "__main__":
    main()
    clean_exit(0)
