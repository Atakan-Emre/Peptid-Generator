"""Aktif yapilandirmanin yuklenmesi.

Scriptler sabit bir dosya adina bagli kalmasin diye tek yerden okunur.
Secim sirasi:
  1. PBP_CONFIG ortam degiskeni (tam yol)
  2. configs/active.json
  3. configs/ altindaki tek .json dosya (yalnizca bir tane varsa)
"""
from __future__ import annotations
import json, os
from pathlib import Path


def config_path() -> Path:
    env = os.environ.get("PBP_CONFIG")
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"PBP_CONFIG gosterilen dosya yok: {p}")
        return p
    active = Path("configs/active.json")
    if active.exists():
        return active
    found = sorted(Path("configs").glob("*.json"))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        "configs/active.json bulunamadi. Donanima uygun yapilandirmayi "
        "kopyalayin, ornegin: copy configs\\rtx5070ti.json configs\\active.json")


def load_config() -> dict:
    return json.loads(config_path().read_text(encoding="utf-8"))
