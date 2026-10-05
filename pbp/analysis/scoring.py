"""Kayitli modelleri yukleyip peptit skorlamak icin ortak yardimcilar."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from ..encoding import encode_many
from ..models.architectures import build_model
from ..train.loop import TrainConfig
from ..device import get_device


class ScoredModel:
    """Egitilmis bir model + normalizasyonu; ham skor doner."""

    def __init__(self, ckpt_path, device=None):
        self.device = device or get_device()
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = TrainConfig(**ck["config"])
        self.model = build_model(cfg.architecture, **cfg.model_kwargs()).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.mean = ck["normalization"]["mean"]
        self.std = ck["normalization"]["std"]
        self.plastic = ck["plastic"]
        self.architecture = cfg.architecture
        self.split_strategy = ck.get("split_strategy")
        self.batch_size = cfg.batch_size * 2

    @torch.no_grad()
    def score_encoded(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy(np.asarray(X, dtype=np.uint8))
        out = []
        for i in range(0, len(Xt), self.batch_size):
            xb = Xt[i:i + self.batch_size].to(self.device)
            out.append(self.model.predict_score(xb).float().cpu().numpy())
        z = np.concatenate(out) if out else np.zeros(0, np.float32)
        return z * self.std + self.mean

    def score(self, sequences) -> np.ndarray:
        return self.score_encoded(encode_many(list(sequences)))


def load_all_models(models_dir, architecture="encdec", split_strategy="clustered",
                    device=None) -> dict[str, ScoredModel]:
    """Her plastik icin kayitli modeli yukler: {plastik: ScoredModel}."""
    models_dir = Path(models_dir)
    out = {}
    for p in sorted(models_dir.glob(f"*_{architecture}_{split_strategy}.pt")):
        m = ScoredModel(p, device=device)
        out[m.plastic] = m
    if not out:
        raise FileNotFoundError(
            f"{models_dir} altinda *_{architecture}_{split_strategy}.pt bulunamadi")
    return out


def random_peptides(n: int, seed: int = 0) -> np.ndarray:
    """Alfabeden duzgun dagilimla rastgele 12-mer uretir (referans hatti)."""
    from ..encoding import N_AA, PEPTIDE_LENGTH
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_AA, size=(n, PEPTIDE_LENGTH), dtype=np.uint8)
