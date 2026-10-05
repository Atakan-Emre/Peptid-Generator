"""Regresyon metrikleri - her bolme icin R2, RMSE, MAE birlikte."""
from __future__ import annotations
import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "r2": round(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), 6),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 6),
        "mae": round(float(np.abs(err).mean()), 6),
        "n": int(len(y_true)),
    }


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return round(float(np.corrcoef(a, b)[0, 1]), 6)


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return round(float(spearmanr(a, b).statistic), 6)
