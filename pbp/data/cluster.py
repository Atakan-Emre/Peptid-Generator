"""Dizi kimligi bazli kumeleme (identity-aware data splitting).

Neden gerekli
-------------
Rastgele %80/10/10 bolmede test dizilerinin cogu, egitim setindeki bir
diziden 1-3 aminoasit uzakta kaliyor (PET'te ayrica %20.7 birebir sizinti
vardi). Model bu durumda transfer edilebilir dizi-afinite iliskisi ogrenmek
yerine yakin komsular arasinda interpolasyon yapiyor olabilir.

Yontem
------
CD-HIT'in acgozlu artimli yaklasiminin sabit uzunluklu diziler icin
uyarlanmasi. Iki dizi arasindaki uzaklik Hamming uzakligidir; 12-mer icin
kimlik = 1 - hamming/12.

Aday arama guvercin yuvasi (pigeonhole) ilkesine dayanir: Hamming uzakligi
<= d olan iki dizi, 12 pozisyon d+1 bloga bolundugunde en az bir blokta
birebir ayni olmak zorundadir. Boylece her dizi icin yalnizca kendi
bloklarini paylasan temsilcilere bakilir; tum ciftler karsilastirilmaz.

Acgozlu (temsilciye uyelik) yaklasimi bilincli bir tercihtir: baglanti
bileseni (transitif) kumeleme 12-mer gibi yogun bir uzayda zincirlenip tek
bir dev kumeye cokerdi. Temsilciye uyelik bu zincirlenmeyi engeller.
"""
from __future__ import annotations
import numpy as np
from ..encoding import PEPTIDE_LENGTH


def _blocks(n_blocks: int) -> list[np.ndarray]:
    """12 pozisyonu n_blocks parcaya (mumkun oldugunca esit) boler."""
    return [np.array(b) for b in np.array_split(np.arange(PEPTIDE_LENGTH), n_blocks)]


def _band_keys(X: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Bir blogun icerigini tek bir int64 anahtara paketler (taban 18)."""
    key = np.zeros(len(X), dtype=np.int64)
    for p in block:
        key = key * 18 + X[:, p].astype(np.int64)
    return key


def greedy_cluster(
    X: np.ndarray,
    max_hamming: int = 3,
    order: str | np.ndarray = "given",
    log_every: int = 100_000,
    verbose: bool = True,
) -> np.ndarray:
    """Dizileri kumeleyip her dizi icin kume kimligi dondurur.

    Parametreler
    ------------
    X : (n, 12) uint8 indeks matrisi
    max_hamming : ayni kumeye girmek icin izin verilen en buyuk Hamming
        uzakligi. 3 -> %75 kimlik esigi, 2 -> %83.3, 1 -> %91.7
    order : "given" | "random" | acik indeks dizisi. Temsilci secimi isleme
        sirasina baglidir; tekrarlanabilirlik icin sabit tutulur.

    Doner
    -----
    labels : (n,) int32, her dizinin kume kimligi
    """
    n = len(X)
    if isinstance(order, str):
        if order == "random":
            rng = np.random.default_rng(0)
            idx_order = rng.permutation(n)
        else:
            idx_order = np.arange(n)
    else:
        idx_order = np.asarray(order)

    n_blocks = max_hamming + 1
    blocks = _blocks(n_blocks)
    band_keys = [_band_keys(X, b) for b in blocks]

    # Her bant icin: anahtar -> o anahtari tasiyan temsilcilerin listesi
    band_index: list[dict[int, list[int]]] = [dict() for _ in range(n_blocks)]

    labels = np.full(n, -1, dtype=np.int32)
    # Temsilciler onceden ayrilmis bir dizide tutulur ve amortize buyur;
    # her adimda listeden matris kurmak O(n_rep) maliyet getirirdi.
    cap = 1024
    rep_rows = np.empty((cap, PEPTIDE_LENGTH), dtype=np.uint8)
    n_rep = 0

    for count, i in enumerate(idx_order):
        row = X[i]
        cand: set[int] = set()
        for b in range(n_blocks):
            lst = band_index[b].get(int(band_keys[b][i]))
            if lst:
                cand.update(lst)

        best_rep = -1
        if cand:
            cand_arr = np.fromiter(cand, dtype=np.int64, count=len(cand))
            dist = (rep_rows[cand_arr] != row).sum(axis=1)
            j = int(dist.argmin())
            if dist[j] <= max_hamming:
                best_rep = int(cand_arr[j])

        if best_rep >= 0:
            labels[i] = best_rep
        else:
            if n_rep == cap:
                cap *= 2
                grown = np.empty((cap, PEPTIDE_LENGTH), dtype=np.uint8)
                grown[:n_rep] = rep_rows[:n_rep]
                rep_rows = grown
            rep_rows[n_rep] = row
            labels[i] = n_rep
            for b in range(n_blocks):
                band_index[b].setdefault(int(band_keys[b][i]), []).append(n_rep)
            n_rep += 1

        if verbose and log_every and (count + 1) % log_every == 0:
            print(f"    {count + 1:,}/{n:,} islendi, {n_rep:,} kume", flush=True)

    return labels


def cluster_summary(labels: np.ndarray) -> dict:
    _, sizes = np.unique(labels, return_counts=True)
    return {
        "n_sequences": int(len(labels)),
        "n_clusters": int(len(sizes)),
        "largest_cluster": int(sizes.max()),
        "largest_cluster_pct": round(100 * float(sizes.max()) / len(labels), 3),
        "mean_cluster_size": round(float(sizes.mean()), 2),
        "median_cluster_size": float(np.median(sizes)),
        "singletons": int((sizes == 1).sum()),
        "singleton_pct": round(100 * float((sizes == 1).sum()) / len(sizes), 2),
    }
