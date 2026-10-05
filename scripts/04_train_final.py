#!/usr/bin/env python3
"""Adim 3 (Asama B): tam veri, cok tohumlu nihai egitim.

Makaleye girecek ortalama+-ss sayilari ve istatistik testler buradan cikar.

    python scripts/04_train_final.py --strategy clustered
    python scripts/04_train_final.py --strategy random    # karsilastirma icin
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit

from pbp.config import load_config

from pbp.device import configure
from pbp.train.experiment import ExperimentRunner
from pbp.train.loop import TrainConfig
from pbp.analysis.stats import compare_architectures, save as save_stats


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--plastic", default="ALL")
    ap.add_argument("--architecture", default="ALL")
    ap.add_argument("--strategy", default="clustered")
    a = ap.parse_args()

    print("cihaz:", configure(), flush=True)
    f = cfg["final"]
    plastics = cfg["plastics"] if a.plastic == "ALL" else [a.plastic]
    if a.strategy == "random" and a.plastic == "ALL" and f.get("random_split_plastics"):
        # Rastgele bolme kolu yalnizca bolme stratejisinin mimari siralamasini
        # nasil degistirdigini gostermek icin; iki temsili polimer yeterli.
        plastics = f["random_split_plastics"]
    if a.architecture != "ALL":
        archs = [a.architecture]
    elif a.strategy == "random" and f.get("random_split_architectures"):
        # Rastgele bolme yalnizca bolme karsilastirma tablosu icin gerekli;
        # tum mimarileri tekrar egitmek butcenin dortte birini bosa harcardi.
        archs = f["random_split_architectures"]
    else:
        archs = cfg["architectures"]
    print(f"  mimariler: {archs}")

    runner = ExperimentRunner(cfg["prep_dir"], cfg["out_dir"])
    summaries = []
    for p in plastics:
        for arch in archs:
            sp = Path(cfg["out_dir"]) / "search" / f"{p}_{arch}_{a.strategy}.json"
            if sp.exists():
                best = TrainConfig(**json.loads(sp.read_text())["best_config"])
            else:
                # Rastgele bolme kolunun amaci YALNIZCA bolme stratejisini
                # degistirmek. Hiperparametreleri de degistirirsek iki etki
                # birbirine karisir; bu yuzden kume bazli aramanin kazanani
                # oldugu gibi kullanilir.
                alt = Path(cfg["out_dir"]) / "search" / f"{p}_{arch}_clustered.json"
                if alt.exists():
                    best = TrainConfig(**json.loads(alt.read_text())["best_config"])
                    print(f"  {p}/{arch}: kume bazli aramanin konfigurasyonu "
                          f"kullaniliyor (karsilastirma icin ayni hiperparametreler)")
                else:
                    print(f"  ! {p}/{arch}: arama sonucu yok, varsayilan konfig")
                    best = TrainConfig(architecture=arch)
            best.batch_size = f["batch_size"]
            best.scheduler = f.get("scheduler", "cosine")
            # Tam tohum seti yalnizca istatistik testine giren mimariler ve
            # kume bazli bolme icin. Rastgele bolme sadece bolme karsilastirma
            # tablosuna girdigi icin kisa tohum seti yeterli.
            if a.strategy == "random":
                seeds = f.get("seeds_secondary", f["seeds"])
            elif arch in f.get("primary_architectures", cfg["architectures"]):
                seeds = f["seeds"]
            else:
                seeds = f.get("seeds_secondary", f["seeds"])
            print(f"\n=== NIHAI {p} / {arch} / {a.strategy} "
                  f"({len(seeds)} tohum) ===", flush=True)
            s = runner.final(p, arch, best, strategy=a.strategy,
                             split_seed=cfg["prepare"]["split_seed"],
                             seeds=tuple(seeds), max_epochs=f["max_epochs"],
                             patience=f["patience"])
            summaries.append(s)
            print(f"  test R2 = {s['test_r2_mean']:.5f} +- {s['test_r2_std']:.5f} "
                  f"({s['n_seeds']} tohum)")

    out = Path(cfg["out_dir"]) / f"summary_{a.strategy}.json"
    out.write_text(json.dumps(summaries, indent=2))
    if len({s["architecture"] for s in summaries}) > 1:
        stats = compare_architectures(summaries, metric="test_r2")
        save_stats(stats, Path(cfg["out_dir"]) / f"architecture_stats_{a.strategy}.json")
        print("\n=== ISTATISTIK ===")
        print(json.dumps(stats["summary"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
    clean_exit(0)
