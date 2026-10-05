#!/usr/bin/env python3
"""Adim 0: M4 uzerinde egitim suresini olcup toplam butceyi tahmin eder.

Gercek bir kalibrasyon kosusu yapar (kucuk bir alt orneklem, birkac epoch),
epoch basina sureyi olcer ve tum protokolun ne kadar surecegini hesaplar.
Uzun kosuya baslamadan once calistirin.

    python scripts/02_estimate_budget.py
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit

from pbp.config import load_config

from pbp.device import configure, get_device
from pbp.data.prepare import load_prepared, Normalization
from pbp.train.loop import TrainConfig, train_one
from pbp.train.experiment import SEARCH_GRID, subsample
import itertools

CALIB_N = 20_000
CALIB_EPOCHS = 2


def main():
    cfg = load_config()
    print("cihaz:", configure(), flush=True)
    dev = get_device()
    if dev.type == "cpu":
        print("  ! GPU yok; asagidaki sureler GPU icin temsili degil")

    probe = cfg["plastics"][-1]
    try:
        z, man = load_prepared(cfg["prep_dir"], probe,
                               cfg["prepare"]["split_strategies"][0],
                               cfg["prepare"]["split_seed"])
    except FileNotFoundError:
        sys.exit("Once 01_prepare_data.py calistirin.")

    X, y = z["X"], z["y"]
    norm = Normalization(**man["normalization"])
    full_train = len(z["train"])
    sp = subsample({k: z[k] for k in ("train", "val", "test")}, CALIB_N)

    per_epoch = {}
    for arch in cfg["architectures"]:
        c = TrainConfig(architecture=arch, max_epochs=CALIB_EPOCHS, patience=99,
                        batch_size=cfg["final"]["batch_size"])
        t = time.time()
        train_one(X, y, sp, norm, c, dev, progress=False).pop("model")
        sec = (time.time() - t) / CALIB_EPOCHS
        # Olculen sure alt orneklem icin; tam veriye olceklendir
        per_epoch[arch] = sec * full_train / len(sp["train"])
        print(f"  {arch:9s} ~{per_epoch[arch]:6.1f} sn/epoch (tam veri, {probe})")

    # Her plastigin kendi egitim boyutuna gore olcekle. Tum plastikleri
    # olculen plastik boyutunda varsaymak maliyeti ciddi olcude dusuk gosterir.
    train_sizes = {}
    for p in cfg["plastics"]:
        try:
            zz, _ = load_prepared(cfg["prep_dir"], p,
                                  cfg["prepare"]["split_strategies"][0],
                                  cfg["prepare"]["split_seed"])
            train_sizes[p] = len(zz["train"])
        except FileNotFoundError:
            train_sizes[p] = full_train
    units = sum(train_sizes.values()) / full_train   # probe plastigi cinsinden
    print(f"\n  toplam egitim ornegi: {sum(train_sizes.values()):,} "
          f"({units:.1f}x {probe})")

    n_strategies = len(cfg["prepare"]["split_strategies"])
    s, f = cfg["search"], cfg["final"]
    n_plastics = len(cfg["plastics"])
    search_scale = s["subsample_n"] / full_train
    primary = f.get("primary_architectures", cfg["architectures"])
    rnd_archs = f.get("random_split_architectures", cfg["architectures"])
    seeds_2nd = f.get("seeds_secondary", f["seeds"])
    total_search = total_final = 0.0
    for arch in cfg["architectures"]:
        n_cfg = len(list(itertools.product(*SEARCH_GRID[arch].values())))
        # Arama sabit boyutlu alt orneklemde: plastik boyutundan bagimsiz
        total_search += (per_epoch[arch] * search_scale * s["max_epochs"]
                         * n_cfg * n_plastics)
        n_seed = len(f["seeds"]) if arch in primary else len(seeds_2nd)
        total_final += per_epoch[arch] * units * f["max_epochs"] * n_seed
        if n_strategies > 1 and arch in rnd_archs:
            total_final += (per_epoch[arch] * units * f["max_epochs"]
                            * len(seeds_2nd))

    # Erken durdurma tipik olarak butcenin ~%60'inda devreye girer
    est = lambda x: (x / 3600 * 0.6, x / 3600)
    print(f"\n  Asama A (arama, {n_plastics} plastik): "
          f"{est(total_search)[0]:.1f}-{est(total_search)[1]:.1f} saat")
    print(f"  Asama B (nihai, {len(f['seeds'])} tohum x {n_strategies} strateji): "
          f"{est(total_final)[0]:.1f}-{est(total_final)[1]:.1f} saat")
    tot = total_search + total_final
    print(f"  TOPLAM: {est(tot)[0]:.1f}-{est(tot)[1]:.1f} saat")
    print("\n  Butce fazlaysa: final.seeds'i 3'e dusurun, prepare.split_strategies'i")
    print("  ['clustered'] yapin (rastgele bolme yalnizca karsilastirma tablosu icin)")
    print("  veya search.subsample_n'i azaltin.")
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["out_dir"], "budget_estimate.json").write_text(json.dumps(
        {"seconds_per_epoch_full_data": {k: round(v, 1) for k, v in per_epoch.items()},
         "search_hours": round(total_search / 3600, 2),
         "final_hours": round(total_final / 3600, 2),
         "device": str(dev)}, indent=2))


if __name__ == "__main__":
    main()
    clean_exit(0)
