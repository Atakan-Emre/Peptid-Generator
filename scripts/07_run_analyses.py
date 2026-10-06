#!/usr/bin/env python3
"""Step 7: characterisation of the generated peptides.

    python scripts/07_run_analyses.py --strategy clustered

Produces:
  selectivity      6x6 cross-polymer score matrix and selectivity index
  validation       independent check (random baseline, best training
                   sequences, and a model not used in the optimisation)
  novelty          exact matches and nearest-neighbour identity distribution
  physicochemical  property-score correlations
  interpretability positional importance and consensus sequence
  docking          correlation with docking scores when a CSV is supplied
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit

from pbp.config import load_config

from pbp.device import configure
from pbp.analysis import (independent_validation, selectivity, novelty,
                          physchem, interpretability, docking_corr)
from pbp.analysis.scoring import ScoredModel, load_all_models
from pbp.encoding import decode_many


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="clustered")
    ap.add_argument("--docking-csv",
                    default="results/analysis/docking_selection.csv")
    an = cfg.get("analysis", {})
    ap.add_argument("--architecture", default=None,
                    help="verilmezse 06_compare_generators.py'nin sectigi mimari")
    ap.add_argument("--validator-architecture",
                    default=an.get("validator_architecture", "lstm"),
                    help="Optimizasyonda KULLANILMAYAN bagimsiz model")
    a = ap.parse_args()

    print("cihaz:", configure(), flush=True)
    out_root = Path(cfg["out_dir"]) / "analysis"; out_root.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg["out_dir"]) / "models"
    gen_dir = Path(cfg["out_dir"]) / "generated"

    # Uretici mimari, 06_compare_generators.py'nin deterministik seciminden
    # gelir; boylece analizler her zaman secicilik testini gecen kumeye
    # uygulanir (not used in the optimisation).
    if a.architecture is None:
        sel_path = gen_dir / f"_selected_generator_{a.strategy}.json"
        if sel_path.exists():
            picked = json.loads(sel_path.read_text())
            a.architecture = picked["architecture"]
            print(f"uretici mimari (karsilastirmadan secildi): {a.architecture}")
            print(f"  gerekce: {picked['reason']}")
        else:
            a.architecture = an.get("best_architecture", "encdec")
            print(f"  ! secim dosyasi yok; yapilandirmadaki {a.architecture} kullanilyor")

    gen_path = gen_dir / f"peptides_{a.strategy}.json"
    if not gen_path.exists():
        sys.exit(f"{gen_path} yok. Once 05_generate_peptides.py ve "
                 "06_compare_generators.py calistirin.")
    peptides = json.loads(gen_path.read_text())

    models = load_all_models(models_dir, a.architecture, a.strategy)
    try:
        validators = load_all_models(models_dir, a.validator_architecture, a.strategy)
    except FileNotFoundError:
        validators = {}
        print(f"  ! {a.validator_architecture} modelleri yok; capraz-model "
              "dogrulamasi atlanacak")

    # --- cross-polymer selectivity ---
    print("\n=== capraz-plastik secicilik ===", flush=True)
    sel = selectivity.cross_score_matrix(peptides, models)
    selectivity.save(sel, out_root)
    print(selectivity.to_markdown_table(sel))
    print(sel["summary"]["interpretation"])

    # --- independent validation ---
    print("\n=== bagimsiz dogrulama ===", flush=True)
    iv_all = []
    for p, seqs in peptides.items():
        npz = Path(cfg["prep_dir"]) / f"{p}_{a.strategy}_seed{cfg['prepare']['split_seed']}.npz"
        r = independent_validation.run(p, seqs, npz, models[p], validators.get(p))
        iv_all.append(r)
        print(f"  {p}: optimize {r['optimized']['mean']:8.2f} | "
              f"rastgele {r['random_baseline']['mean']:8.2f} | "
              f"egitim-en iyi%1 {r['train_top1pct_measured']['mean']:8.2f} | "
              f"kazanc {r['gain_vs_random']:7.2f}"
              + (f" | dogrulayici onayi: {r['validator_confirms_gain']}"
                 if "validator_model" in r else ""))
    independent_validation.save(iv_all, out_root / "independent_validation.json")

    # --- novelty ---
    print("\n=== yenilik ===", flush=True)
    nov = []
    for p, seqs in peptides.items():
        npz = Path(cfg["prep_dir"]) / f"{p}_{a.strategy}_seed{cfg['prepare']['split_seed']}.npz"
        z = np.load(npz)
        ref = decode_many(z["X"][z["train"]])
        r = novelty.analyse(seqs, ref, p)
        nov.append(r)
        print(f"  {p}: birebir esles {r['exact_matches']}  "
              f"NN kimlik ort %{r['nn_identity_mean_pct']}  "
              f"%90+ kimlik orani %{r['pct_above_90_identity']}")
    novelty.save(nov, out_root)

    # --- physicochemical ---
    print("\n=== fizikokimyasal ===", flush=True)
    phys = []
    for p, seqs in peptides.items():
        sc = models[p].score(seqs)
        r = physchem.analyse(list(seqs), sc, p)
        phys.append(r)
        c = r["correlations"]
        print(f"  {p}: aromatik {c['aromatic_fraction']['mean']} "
              f"(r={c['aromatic_fraction']['pearson_with_score']})  "
              f"net yuk {c['net_charge_ph7_4']['mean']} "
              f"(r={c['net_charge_ph7_4']['pearson_with_score']})")
    physchem.save(phys, out_root)

    # --- interpretability ---
    print("\n=== yorumlanabilirlik ===", flush=True)
    interp = []
    for p in peptides:
        npz = Path(cfg["prep_dir"]) / f"{p}_{a.strategy}_seed{cfg['prepare']['split_seed']}.npz"
        z = np.load(npz)
        r = interpretability.position_importance(models[p], z["X"][z["test"]])
        interp.append(r)
        print(f"  {p}: en onemli pozisyonlar {r['most_important_positions']}  "
              f"uzlasim {r['consensus_sequence']}")
    interpretability.save(interp, out_root)

    # --- docking korelasyonu ---
    # Laboratuvar sonuclari geldiginde calisir; CSV yoksa veya skor sutunu
    # hala bossa sessizce atlanir, boylece adim her durumda tamamlanir.
    dp = Path(a.docking_csv)
    if dp.exists():
        import pandas as _pd
        _df = _pd.read_csv(dp)
        if "docking_score" in _df.columns and _df["docking_score"].notna().any():
            print("\n=== docking korelasyonu ===", flush=True)
            dres = docking_corr.analyse(dp, models)
            docking_corr.save(dres, out_root / "analysis")
            print(json.dumps(dres["per_plastic"], indent=1, ensure_ascii=False))
        else:
            print(f"\n  Docking skorlari henuz girilmemis: {dp}")

    print(f"\nTum analizler: {out_root}")


if __name__ == "__main__":
    main()
    clean_exit(0)
