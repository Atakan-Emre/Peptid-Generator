"""Yenilik analizi (novelty of generated sequences).

The problem: for 12-mers the mean Hamming distance is 2-3.8, i.e.
%68-83 dizi kimligi. Egitim setinde birebir esles bulunmamasi, dizilerin
yapisal ya da islevsel anlamda "yeni" oldugunu gostermez.

Bu modul "birebir esles var mi" ikili sorusu yerine en yakin komsu kimlik
DAGILIMINI raporlar ve terminoloji icin somut bir dayanak uretir.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from ..encoding import encode_many, decode_one, PEPTIDE_LENGTH


def nearest_neighbours(query: np.ndarray, reference: np.ndarray, top_k: int = 3,
                       block: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Her sorgu dizisi icin referanstaki en yakin top_k diziyi bulur."""
    n_q = len(query)
    best_d = np.full((n_q, top_k), PEPTIDE_LENGTH + 1, dtype=np.int16)
    best_i = np.full((n_q, top_k), -1, dtype=np.int64)
    for start in range(0, len(reference), block):
        ref = reference[start:start + block]
        # (n_q, block) Hamming
        d = (query[:, None, :] != ref[None, :, :]).sum(axis=2).astype(np.int16)
        cat_d = np.concatenate([best_d, d], axis=1)
        cat_i = np.concatenate([best_i, np.arange(start, start + len(ref))[None, :].repeat(n_q, 0)], axis=1)
        order = np.argsort(cat_d, axis=1, kind="stable")[:, :top_k]
        best_d = np.take_along_axis(cat_d, order, axis=1)
        best_i = np.take_along_axis(cat_i, order, axis=1)
    return best_d, best_i


def analyse(generated_sequences, reference_sequences, plastic: str,
            top_k: int = 3) -> dict:
    q = encode_many(list(generated_sequences))
    r = encode_many(list(reference_sequences))
    d, i = nearest_neighbours(q, r, top_k=top_k)
    nn = d[:, 0].astype(float)
    identity = 100 * (1 - nn / PEPTIDE_LENGTH)

    records = []
    for k in range(len(q)):
        records.append({
            "plastic": plastic,
            "sequence": decode_one(q[k]),
            "nn_hamming": int(d[k, 0]),
            "nn_identity_pct": round(float(identity[k]), 2),
            "exact_match": bool(d[k, 0] == 0),
            "neighbours": [
                {"sequence": decode_one(r[i[k, j]]),
                 "hamming": int(d[k, j]),
                 "identity_pct": round(100 * (1 - d[k, j] / PEPTIDE_LENGTH), 2)}
                for j in range(top_k) if i[k, j] >= 0
            ],
        })

    hist = {f"identity_{lo}_{hi}": int(((identity >= lo) & (identity < hi)).sum())
            for lo, hi in [(0, 60), (60, 70), (70, 80), (80, 90), (90, 100)]}
    hist["identity_100"] = int((identity >= 100).sum())

    return {
        "plastic": plastic,
        "n_generated": len(q),
        "n_reference": len(r),
        "exact_matches": int((d[:, 0] == 0).sum()),
        "nn_hamming_min": int(nn.min()), "nn_hamming_max": int(nn.max()),
        "nn_hamming_mean": round(float(nn.mean()), 3),
        "nn_identity_mean_pct": round(float(identity.mean()), 2),
        "nn_identity_min_pct": round(float(identity.min()), 2),
        "nn_identity_max_pct": round(float(identity.max()), 2),
        "pct_above_90_identity": round(100 * float((identity >= 90).mean()), 2),
        "identity_histogram": hist,
        "records": records,
        "recommended_wording": (
            "Uretilen diziler egitim setinde birebir yer almamakla birlikte en yakin "
            f"komsularina ortalama %{identity.mean():.1f} kimlik tasimaktadir; bu nedenle "
            "metinde 'novel' yerine 'optimizasyonla iyilestirilmis varyantlar' "
            "(optimization-refined variants) ifadesi kullanilmalidir."),
    }


def save(results: list[dict], out_dir) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "novelty.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    rows = ["| Polymer | n | Exact matches | NN Hamming (mean) | NN identity (mean) | >=90% identity |",
            "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        rows.append(f"| {r['plastic']} | {r['n_generated']} | {r['exact_matches']} | "
                    f"{r['nn_hamming_mean']} | %{r['nn_identity_mean_pct']} | "
                    f"%{r['pct_above_90_identity']} |")
    (out_dir / "novelty_table.md").write_text("\n".join(rows))
