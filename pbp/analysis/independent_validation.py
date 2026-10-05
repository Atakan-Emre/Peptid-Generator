"""Bagimsiz dogrulama (validation independent of the optimised surrogate).

The problem: the model is both trained on the affinity scores and
optimizasyonun amac fonksiyonu olarak kullaniliyor. Bu durumda giderek daha
negatif skorlar bulmak, peptitlerin gercekten daha yuksek afiniteye sahip
oldugunu degil, optimize edicinin modelin dusuk skor tahmin ettigi bolgeleri
buldugunu gosterir.

Bu modul dort bagimsiz kontrol uretir:
  1. Rastgele 12-mer referans hatti
  2. Egitim setinin en iyi %1'i (ulasilabilir en iyi gercek olculmus deger)
  3. Capraz-model dogrulama: optimizasyonda KULLANILMAYAN, farkli mimari ve
     farkli tohumla egitilmis bir model ile yeniden skorlama
  4. (docking_corr modulu) molekuler docking ile korelasyon
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from ..encoding import encode_many, decode_many
from ..train.metrics import pearson, spearman
from .scoring import ScoredModel, random_peptides


def summarize(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4),
            "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
            "best": round(float(v.min()), 4), "median": round(float(np.median(v)), 4),
            "p05": round(float(np.percentile(v, 5)), 4)}


def run(plastic, optimized_sequences, prepared_npz, target_model: ScoredModel,
        validator_model: ScoredModel | None = None, n_random=2000, seed=0) -> dict:
    """Optimize peptitleri referans hatlariyla karsilastirir."""
    z = np.load(prepared_npz)
    X_all, y_all, tr = z["X"], z["y"], z["train"]

    opt_enc = encode_many(list(optimized_sequences))
    opt_scores = target_model.score_encoded(opt_enc)

    rnd_enc = random_peptides(n_random, seed=seed)
    rnd_scores = target_model.score_encoded(rnd_enc)

    # Egitim setinin en iyi %1'i: gercek olculmus skorlar
    y_tr = y_all[tr]
    k = max(1, int(0.01 * len(y_tr)))
    top_idx = tr[np.argsort(y_tr)[:k]]
    top_true = y_all[top_idx]
    top_pred = target_model.score_encoded(X_all[top_idx])

    out = {
        "plastic": plastic,
        "target_model": {"architecture": target_model.architecture,
                         "split_strategy": target_model.split_strategy},
        "optimized": summarize(opt_scores),
        "random_baseline": summarize(rnd_scores),
        "train_top1pct_measured": summarize(top_true),
        "train_top1pct_predicted": summarize(top_pred),
    }
    out["gain_vs_random"] = round(float(opt_scores.mean() - rnd_scores.mean()), 4)
    out["gain_vs_train_top1pct_measured"] = round(
        float(opt_scores.mean() - top_true.mean()), 4)
    out["beats_best_training_sequence"] = bool(opt_scores.min() < y_tr.min())
    out["best_training_score"] = round(float(y_tr.min()), 4)

    if validator_model is not None:
        v_opt = validator_model.score_encoded(opt_enc)
        v_rnd = validator_model.score_encoded(rnd_enc)
        out["validator_model"] = {
            "architecture": validator_model.architecture,
            "split_strategy": validator_model.split_strategy,
            "optimized": summarize(v_opt),
            "random_baseline": summarize(v_rnd),
            "gain_vs_random": round(float(v_opt.mean() - v_rnd.mean()), 4),
            "agreement_pearson": pearson(opt_scores, v_opt),
            "agreement_spearman": spearman(opt_scores, v_opt),
        }
        # En onemli sayi: bagimsiz model de kazanc goruyor mu?
        out["validator_confirms_gain"] = bool(v_opt.mean() < v_rnd.mean())

    out["sequences"] = [
        {"sequence": s, "target_score": round(float(a), 4),
         **({"validator_score": round(float(b), 4)} if validator_model is not None else {})}
        for s, a, b in zip(decode_many(opt_enc), opt_scores,
                           (v_opt if validator_model is not None else opt_scores))
    ]
    return out


def save(result: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
