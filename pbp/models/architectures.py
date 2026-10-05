"""Dort mimarinin tek kaynaktan tanimi.

Onceki surumde model tanimlari ablation_study_{mac,win,mlx}.py ve
peptide_generation_comparison.py icinde tekrarlaniyordu; bu, egitilen model
ile peptit uretiminde kullanilan modelin sessizce ayrisma riskini tasiyordu.
Burada tek tanim vardir ve her yerden bu import edilir.

Girdi sozlesmesi: modeller (B, 12) uint8/long indeks tensoru alir ve one-hot
donusumunu kendi icinde, cihaz uzerinde yapar. Boylece veri bellekte 12
bayt/ornek olarak kalir (M4'un 16 GB birlesik bellegi icin kritik).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoding import N_AA, PEPTIDE_LENGTH


class _Base(nn.Module):
    returns_reconstruction = False

    def one_hot(self, idx: torch.Tensor) -> torch.Tensor:
        return F.one_hot(idx.long(), num_classes=N_AA).float()

    def predict_score(self, idx: torch.Tensor) -> torch.Tensor:
        out = self(idx)
        return out[0] if isinstance(out, tuple) else out


class LSTMRegressor(_Base):
    """Varsayilan olarak cift yonlu. bidirectional=False, Jain ve ark.'nin
    tek yonlu LSTM'ini birebir kurmak icin kullanilir (onlar BiLSTM'i
    denemis ve tek yonlu LSTM'in daha iyi oldugunu raporlamislar)."""

    def __init__(self, hidden_dim=256, num_layers=3, dropout=0.1,
                 bidirectional=True, **_):
        super().__init__()
        self.lstm = nn.LSTM(N_AA, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, idx):
        x = self.one_hot(idx)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class CNNRegressor(_Base):
    def __init__(self, n_filters=128, kernel_size=3, dropout=0.1, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(N_AA, n_filters, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters * 2, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(n_filters * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(n_filters * 2, n_filters), nn.ReLU(),
            nn.Linear(n_filters, 1))

    def forward(self, idx):
        x = self.one_hot(idx).transpose(1, 2)
        return self.head(self.net(x)).squeeze(-1)


class LSTMVAE(_Base):
    """Regresyon + yeniden yapilandirma + KL duzenlilestirme."""
    returns_reconstruction = True

    def __init__(self, hidden_dim=256, latent_dim=64, num_layers=2,
                 dropout=0.2, **_):
        super().__init__()
        self.encoder = nn.LSTM(N_AA, hidden_dim, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.recon_head = nn.Linear(hidden_dim, N_AA)
        self.score_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, idx):
        x = self.one_hot(idx)
        enc, _ = self.encoder(x)
        h = enc[:, -1]
        mu, logvar = self.to_mu(h), self.to_logvar(h).clamp(-10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        dec_in = self.from_latent(z).unsqueeze(1).expand(-1, PEPTIDE_LENGTH, -1)
        dec, _ = self.decoder(dec_in)
        return self.score_head(z).squeeze(-1), self.recon_head(dec), mu, logvar


class LSTMEncDec(_Base):
    """Ortak gizli gosterimden hem skor hem dizi yeniden yapilandirmasi."""
    returns_reconstruction = True

    def __init__(self, hidden_dim=256, num_layers=3, dropout=0.1,
                 use_layernorm=True, **_):
        super().__init__()
        self.encoder = nn.LSTM(N_AA, hidden_dim, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.recon_head = nn.Linear(hidden_dim, N_AA)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, idx):
        x = self.one_hot(idx)
        enc, _ = self.encoder(x)
        h = self.norm(enc[:, -1])
        dec_in = h.unsqueeze(1).expand(-1, PEPTIDE_LENGTH, -1)
        dec, _ = self.decoder(dec_in)
        return self.score_head(h).squeeze(-1), self.recon_head(dec), None, None


REGISTRY = {
    "lstm": LSTMRegressor,
    "cnn": CNNRegressor,
    "lstm_vae": LSTMVAE,
    "encdec": LSTMEncDec,
}


def build_model(name: str, **kwargs) -> _Base:
    if name not in REGISTRY:
        raise ValueError(f"Bilinmeyen mimari: {name}. Secenekler: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def compute_loss(name, out, y_norm, idx, lambda_recon=1.0, beta_kl=0.01):
    """Mimariye gore kayip; skor kaybi her zaman MSE."""
    if not isinstance(out, tuple):
        loss = F.mse_loss(out, y_norm)
        return loss, {"score": float(loss.detach())}
    score, recon, mu, logvar = out
    loss_s = F.mse_loss(score, y_norm)
    loss_r = F.cross_entropy(recon.reshape(-1, N_AA), idx.long().reshape(-1))
    total = loss_s + lambda_recon * loss_r
    parts = {"score": float(loss_s.detach()), "recon": float(loss_r.detach())}
    if mu is not None:
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = total + beta_kl * kl
        parts["kl"] = float(kl.detach())
    return total, parts
