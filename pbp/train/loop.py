"""Tek bir egitim kosusu.

Tasarim tercihleri
------------------
* DataLoader kullanilmaz. Veri uint8 indeks olarak tutuldugu icin cok
  kucuk (715k ornek = 8.6 MB); indeks karistirip dilim almak, worker
  surecleri arasinda veri kopyalamaktan belirgin olarak hizlidir ve
  Windows'ta multiprocessing baslatma maliyetini tamamen ortadan kaldirir.
* CUDA'da veri setinin TAMAMI bir kez cihaza tasinir (bkz. device.py,
  can_hold_data). Boylece egitim boyunca tek bir host-cihaz transferi bile
  yapilmaz; bu kadar kucuk veride transfer, hesaptan daha pahaliya gelirdi.
* float32. Autocast varsayilan olarak kapali: LSTM'lerde sayisal
  kararliligi korumak, bu butcede hiz kazanmaktan daha degerli.
* Batch tavanlari cihaza gore (bkz. device.py).
"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import torch

from ..device import can_hold_data
from ..models.architectures import build_model, compute_loss
from .metrics import regression_metrics


@dataclass
class TrainConfig:
    architecture: str = "encdec"
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    latent_dim: int = 64
    n_filters: int = 128
    kernel_size: int = 3
    use_layernorm: bool = True
    bidirectional: bool = True
    lambda_recon: float = 1.0
    beta_kl: float = 0.01
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 60
    patience: int = 8
    grad_clip: float = 1.0
    seed: int = 42
    amp: bool = False
    scheduler: str = "cosine"   # "cosine" | "plateau"

    def model_kwargs(self) -> dict:
        return dict(hidden_dim=self.hidden_dim, num_layers=self.num_layers,
                    dropout=self.dropout, latent_dim=self.latent_dim,
                    n_filters=self.n_filters, kernel_size=self.kernel_size,
                    use_layernorm=self.use_layernorm,
                    bidirectional=self.bidirectional)


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, X: torch.Tensor, batch_size: int, device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        if xb.device != device:
            xb = xb.to(device, non_blocking=True)
        out.append(model.predict_score(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train_one(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
    norm,
    cfg: TrainConfig,
    device,
    log_path: Path | None = None,
    progress: bool = True,
) -> dict:
    """Bir mimari + hiperparametre + tohum kombinasyonunu egitir.

    train/val/test uclusunun HEPSI icin R2, RMSE ve MAE dondurur
    (training protocol reporting).
    """
    seed_everything(cfg.seed)
    Xt = torch.from_numpy(X)
    y_norm_all = torch.from_numpy(norm.apply(y).astype(np.float32))

    # Veri siginiyorsa cihazda yerlesik tut: epoch basina binlerce kucuk
    # transfer yerine tek seferlik bir kopya.
    resident = can_hold_data(Xt.numel() + y_norm_all.numel() * 4, device)
    if resident and device.type != "cpu":
        Xt = Xt.to(device)
        y_norm_all = y_norm_all.to(device)

    tr, va, te = splits["train"], splits["val"], splits["test"]
    tr_t = torch.as_tensor(np.asarray(tr), dtype=torch.long, device=Xt.device)
    Xtr, ytr = Xt[tr_t], y_norm_all[tr_t]
    model = build_model(cfg.architecture, **cfg.model_kwargs()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    # Kosinus tavlama sabit butcede gercek yakinsama saglar. Plato tabanli
    # azaltma bu veride devreye girmiyordu: val R2 surekli iyilestigi icin
    # ogrenme orani hic dusmuyor ve modeller epoch tavaninda hala
    # iyilesirken kesiliyordu.
    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.max_epochs, eta_min=cfg.learning_rate * 0.01)
        _step_sched = lambda m: sched.step()
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=max(2, cfg.patience // 3))
        _step_sched = lambda m: sched.step(m)

    n = len(tr)
    steps = math.ceil(n / cfg.batch_size)
    best = {"val_r2": -math.inf, "epoch": -1, "state": None}
    history = []
    rng = np.random.default_rng(cfg.seed)
    t_start = time.time()
    bad_epochs = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.as_tensor(rng.permutation(n), dtype=torch.long,
                               device=Xtr.device)
        running = 0.0
        for s in range(steps):
            sl = perm[s * cfg.batch_size:(s + 1) * cfg.batch_size]
            xb = Xtr[sl]
            yb = ytr[sl]
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss, _ = compute_loss(cfg.architecture, out, yb, xb,
                                   cfg.lambda_recon, cfg.beta_kl)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.detach())

        va_t = torch.as_tensor(np.asarray(va), dtype=torch.long, device=Xt.device)
        val_pred = norm.invert(predict(model, Xt[va_t], cfg.batch_size * 2, device))
        val_m = regression_metrics(y[va], val_pred)
        _step_sched(val_m["r2"])
        history.append({"epoch": epoch, "train_loss": running / steps,
                        "lr": opt.param_groups[0]["lr"],
                        **{f"val_{k}": v for k, v in val_m.items()}})

        if val_m["r2"] > best["val_r2"] + 1e-6:
            best = {"val_r2": val_m["r2"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if progress:
            print(f"      epoch {epoch+1:3d}/{cfg.max_epochs}  loss={running/steps:.4f}"
                  f"  val_R2={val_m['r2']:.5f}  (en iyi {best['val_r2']:.5f} @ {best['epoch']+1})",
                  flush=True)
        if bad_epochs >= cfg.patience:
            if progress:
                print(f"      erken durdurma: {cfg.patience} epoch iyilesme yok", flush=True)
            break


    if best["state"] is not None:
        model.load_state_dict(best["state"])

    result = {"config": asdict(cfg), "best_epoch": best["epoch"], "epochs_run": len(history),
              "train_seconds": round(time.time() - t_start, 1), "history": history,
              "metrics": {}}
    for part, idxs in (("train", tr), ("val", va), ("test", te)):
        ix = torch.as_tensor(np.asarray(idxs), dtype=torch.long, device=Xt.device)
        pred = norm.invert(predict(model, Xt[ix], cfg.batch_size * 2, device))
        result["metrics"][part] = regression_metrics(y[idxs], pred)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(result, indent=2))
    result["model"] = model
    return result
