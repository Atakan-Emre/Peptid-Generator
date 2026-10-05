#!/usr/bin/env python3
"""Adim 1: veri temizleme, kumeleme ve bolme.

Kullanim:
    python scripts/01_prepare_data.py                  # tum plastikler, her iki strateji
    python scripts/01_prepare_data.py --plastic PET
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit

from pbp.config import load_config

from pbp.data.prepare import prepare_plastic


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--plastic", default="ALL")
    ap.add_argument("--strategy", default="ALL", choices=["ALL", "clustered", "random"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    plastics = cfg["plastics"] if a.plastic == "ALL" else [a.plastic]
    strategies = cfg["prepare"]["split_strategies"] if a.strategy == "ALL" else [a.strategy]

    summary = []
    for p in plastics:
        for s in strategies:
            print(f"\n=== {p} / {s} ===", flush=True)
            m = prepare_plastic(
                Path(cfg["data_dir"]) / f"{p}.csv", cfg["prep_dir"], p,
                split_strategy=s, max_hamming=cfg["prepare"]["max_hamming"],
                seed=cfg["prepare"]["split_seed"], dedup=cfg["prepare"]["dedup"],
                force=a.force)
            lk = m["leakage"]
            print(f"  ornek: {m['n_after_prep']:,}  bolme: {m['split_pct']}")
            if m["dedup"]:
                print(f"  tekrar temizligi: {m['dedup']['n_removed']:,} "
                      f"(%{m['dedup']['duplicate_pct']})")
            if m["clustering"]:
                c = m["clustering"]
                print(f"  kume: {c['n_clusters']:,}  en buyuk %{c['largest_cluster_pct']}")
            print(f"  SIZINTI: birebir %{lk['exact_test_in_train_pct']}  "
                  f"NN kimlik %{lk['nn_identity_mean_pct']}  "
                  f"NN<=2 oran %{lk['pct_nn_le_2']}")
            summary.append({"plastic": p, "strategy": s, **lk,
                            "n": m["n_after_prep"]})

    out = Path(cfg["prep_dir"]) / "leakage_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nOzet yazildi: {out}")
    print("\n| Plastik | Strateji | Birebir sizinti % | NN kimlik % | NN<=2 % |")
    print("| --- | --- | --- | --- | --- |")
    for r in summary:
        print(f"| {r['plastic']} | {r['strategy']} | {r['exact_test_in_train_pct']} "
              f"| {r['nn_identity_mean_pct']} | {r['pct_nn_le_2']} |")


if __name__ == "__main__":
    main()
    clean_exit(0)
