"""Deney koordinasyonu: mimari x plastik x bolme x tohum.

Iki asamali protokol (M4 uzerinde gercekci sure icin)
-----------------------------------------------------
Asama A - hiperparametre arama: tek tohum, alt-orneklenmis veri, kucultulmus
  izgara. Amac en iyi konfigurasyonu bulmak.
Asama B - nihai egitim: Asama A'nin kazanani, TAM veri, COK TOHUM, her iki
  bolme stratejisi. Makaleye giren ortalama+-ss sayilari buradan gelir.

Onceki surumdeki ~624 konfigurasyon x 6 plastik izgarasi tek tohumluydu ve
zaten M4'te gunlerce surerdi; cok tohumlu tekrari imkansizdi. Kucultulmus
izgara, leaves budget for multi-seed repetition.
"""
from __future__ import annotations
import itertools, json, time
from pathlib import Path
import numpy as np
import torch

from ..data.prepare import load_prepared, Normalization
from ..device import get_device, clamp_batch_size, free_memory
from .loop import TrainConfig, train_one

PLASTICS = ["PET", "PE", "PP", "PS", "PVC", "Nylon"]
ARCHITECTURES = ["lstm", "cnn", "lstm_vae", "encdec"]

# Asama A icin kucultulmus fakat savunulabilir izgara.
# Onceki izgarada duyarlilik analizi zaten hidden_dim ve num_layers'in
# baskin oldugunu, ogrenme oraninin 1e-3'te sabitlendigini gostermisti.
SEARCH_GRID = {
    "lstm":     {"hidden_dim": [128, 256], "num_layers": [2, 3],
                 "dropout": [0.1, 0.2, 0.3], "weight_decay": [1e-4, 1e-2]},
    "encdec":   {"hidden_dim": [128, 256], "num_layers": [2, 3],
                 "dropout": [0.1, 0.2], "lambda_recon": [0.5, 1.0],
                 "use_layernorm": [True, False]},
    "lstm_vae": {"hidden_dim": [128, 256], "latent_dim": [32, 64, 128],
                 "num_layers": [2, 3], "dropout": [0.1, 0.2]},
    "cnn":      {"n_filters": [64, 128, 256], "kernel_size": [3, 5],
                 "dropout": [0.1, 0.2], "learning_rate": [1e-3, 5e-4]},
}


def grid_configs(architecture: str, base: dict | None = None) -> list[TrainConfig]:
    grid = SEARCH_GRID[architecture]
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, combo))
        cfg = TrainConfig(architecture=architecture, **(base or {}), **kw)
        cfg.batch_size = clamp_batch_size(architecture, cfg.batch_size)
        out.append(cfg)
    return out


def subsample(splits: dict, n_max: int, seed: int = 0) -> dict:
    """Asama A icin bolmeleri oransal olarak kucultur (bolme butunlugu korunur)."""
    rng = np.random.default_rng(seed)
    total = sum(len(v) for v in splits.values())
    if total <= n_max:
        return splits
    frac = n_max / total
    return {k: rng.choice(v, max(1, int(len(v) * frac)), replace=False)
            for k, v in splits.items()}


class ExperimentRunner:
    def __init__(self, prep_dir, out_dir, device=None, progress=True):
        self.prep_dir = Path(prep_dir)
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or get_device()
        self.progress = progress
        self.state_path = self.out_dir / "_runner_state.json"
        self.done = set(json.loads(self.state_path.read_text())) if self.state_path.exists() else set()

    def _mark(self, key):
        self.done.add(key)
        self.state_path.write_text(json.dumps(sorted(self.done)))

    def _load(self, plastic, strategy, seed):
        z, man = load_prepared(self.prep_dir, plastic, strategy, seed)
        norm = Normalization(**man["normalization"])
        splits = {k: z[k] for k in ("train", "val", "test")}
        return z["X"], z["y"], splits, norm, man

    def search(self, plastic, architecture, strategy="clustered", split_seed=42,
               subsample_n=150_000, max_epochs=25, patience=5):
        """Asama A: en iyi hiperparametreleri bul."""
        key = f"search|{plastic}|{architecture}|{strategy}"
        res_path = self.out_dir / "search" / f"{plastic}_{architecture}_{strategy}.json"
        if key in self.done and res_path.exists():
            return json.loads(res_path.read_text())

        X, y, splits, norm, _ = self._load(plastic, strategy, split_seed)
        splits = subsample(splits, subsample_n)
        rows = []
        cfgs = grid_configs(architecture)
        for i, cfg in enumerate(cfgs):
            cfg.max_epochs, cfg.patience = max_epochs, patience
            if self.progress:
                print(f"  [{plastic}/{architecture}] konfig {i+1}/{len(cfgs)}: "
                      f"h={cfg.hidden_dim} L={cfg.num_layers} do={cfg.dropout}", flush=True)
            r = train_one(X, y, splits, norm, cfg, self.device, progress=False)
            r.pop("model", None); r.pop("history", None)
            rows.append(r)
            free_memory(self.device)
        best = max(rows, key=lambda r: r["metrics"]["val"]["r2"])
        out = {"plastic": plastic, "architecture": architecture,
               "split_strategy": strategy, "n_configs": len(cfgs),
               "subsample_n": subsample_n, "best_config": best["config"],
               "best_val_r2": best["metrics"]["val"]["r2"], "all_runs": rows}
        res_path.parent.mkdir(parents=True, exist_ok=True)
        res_path.write_text(json.dumps(out, indent=2))
        self._mark(key)
        return out

    def final(self, plastic, architecture, config: TrainConfig, strategy="clustered",
              split_seed=42, seeds=(0, 1, 2, 3, 4), max_epochs=60, patience=8,
              save_model_for_seed=0):
        """Asama B: tam veri, cok tohum. Ortalama+-ss buradan cikar."""
        runs = []
        for s in seeds:
            key = f"final|{plastic}|{architecture}|{strategy}|{s}"
            rp = self.out_dir / "final" / f"{plastic}_{architecture}_{strategy}_seed{s}.json"
            if key in self.done and rp.exists():
                runs.append(json.loads(rp.read_text())); continue

            X, y, splits, norm, _ = self._load(plastic, strategy, split_seed)
            cfg = TrainConfig(**{**config.__dict__}) if isinstance(config, TrainConfig) \
                else TrainConfig(**config)
            cfg.seed, cfg.max_epochs, cfg.patience = s, max_epochs, patience
            cfg.batch_size = clamp_batch_size(architecture, cfg.batch_size,
                                              self.device)
            if self.progress:
                print(f"  [{plastic}/{architecture}/{strategy}] tohum {s}", flush=True)
            r = train_one(X, y, splits, norm, cfg, self.device, progress=self.progress)
            model = r.pop("model")
            if s == save_model_for_seed:
                mdir = self.out_dir / "models"; mdir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__,
                            "normalization": {"mean": norm.mean, "std": norm.std},
                            "plastic": plastic, "split_strategy": strategy},
                           mdir / f"{plastic}_{architecture}_{strategy}.pt")
            del model; free_memory(self.device)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(r, indent=2))
            runs.append(r); self._mark(key)

        return summarize_seeds(runs, plastic, architecture, strategy)


def summarize_seeds(runs, plastic, architecture, strategy) -> dict:
    """Tohumlar uzerinden ortalama+-ss (multi-seed significance testing)."""
    out = {"plastic": plastic, "architecture": architecture,
           "split_strategy": strategy, "n_seeds": len(runs)}
    for part in ("train", "val", "test"):
        for metric in ("r2", "rmse", "mae"):
            vals = np.array([r["metrics"][part][metric] for r in runs], dtype=float)
            out[f"{part}_{metric}_mean"] = round(float(vals.mean()), 6)
            out[f"{part}_{metric}_std"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 6)
    out["per_seed_test_r2"] = [r["metrics"]["test"]["r2"] for r in runs]
    out["per_seed_val_r2"] = [r["metrics"]["val"]["r2"] for r in runs]
    out["mean_train_seconds"] = round(float(np.mean([r["train_seconds"] for r in runs])), 1)
    return out
