#!/usr/bin/env python3
"""Kosunun durumunu ozetler (Windows ve Linux)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit
from pbp.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = load_config()
    out = ROOT / cfg["out_dir"]
    logs = ROOT / "logs"

    n_pl = len(cfg["plastics"]); n_ar = len(cfg["architectures"])
    f = cfg["final"]
    expected = (n_pl * n_ar * len(f["seeds"])
                + len(f.get("random_split_plastics", cfg["plastics"]))
                * len(f.get("random_split_architectures", cfg["architectures"]))
                * len(f.get("seeds_secondary", f["seeds"])))
    done = len(list((out / "final").glob("*.json"))) if (out / "final").exists() else 0
    search = len(list((out / "search").glob("*.json"))) if (out / "search").exists() else 0
    print(f"arama   : {search}/{n_pl * n_ar}")
    print(f"egitim  : {done}/{expected}")

    newest, age = None, None
    for p in logs.glob("*.log"):
        if newest is None or p.stat().st_mtime > newest.stat().st_mtime:
            newest = p
    if newest:
        age = (time.time() - newest.stat().st_mtime) / 60
        print(f"son log : {newest.name}, {age:.0f} dk once")
        if age > 60:
            print("  ! 60 dakikadir yeni cikti yok - kosu durmus olabilir")
        tail = newest.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
        for line in tail:
            print("    " + line)
    else:
        print("son log : yok (kosu hic baslamamis)")
    sel = out / "generated" / "_selected_generator_clustered.json"
    if sel.exists():
        d = json.loads(sel.read_text(encoding="utf-8"))
        print(f"uretici : {d['architecture']}"
              + ("  (yedek secim - hicbir aday tam gecmedi)" if d.get("fallback") else ""))

    print("\nadimlar :")
    for st in sorted(logs.glob(".ok_*")):
        print(f"  tamam  {st.name[4:]}")

    tbl = out / "paper_tables" / "ALL_TABLES.md"
    if tbl.exists():
        print(f"\nTABLOLAR HAZIR: {tbl}")


if __name__ == "__main__":
    main()
    clean_exit(0)
