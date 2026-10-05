"""Peptit dizisi kodlama ve alfabe tanimlari.

Veri setindeki 18 aminoasit (sistein C ve prolin P haric) sabittir.
Diziler bellekte uint8 indeks dizisi olarak tutulur (12 bayt/ornek);
one-hot gosterim egitim sirasinda cihaz uzerinde uretilir. 715k ornek
icin bu 618 MB yerine 8.6 MB demektir - M4'un birlesik bellegi icin kritik.
"""
from __future__ import annotations
import numpy as np

ALPHABET = "ADEFGHIKLMNQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(ALPHABET)}
IDX_TO_AA = {i: a for a, i in AA_TO_IDX.items()}
N_AA = len(ALPHABET)          # 18
PEPTIDE_LENGTH = 12

_LUT = np.full(256, 255, dtype=np.uint8)
for _a, _i in AA_TO_IDX.items():
    _LUT[ord(_a)] = _i


def encode_many(sequences) -> np.ndarray:
    """Dizi listesini (n, 12) uint8 indeks matrisine cevirir."""
    arr = np.asarray(sequences, dtype="U%d" % PEPTIDE_LENGTH)
    raw = arr.view(np.uint32).reshape(len(arr), -1)[:, :PEPTIDE_LENGTH]
    if raw.max() > 255:
        raise ValueError("Dizilerde ASCII disi karakter var")
    out = _LUT[raw.astype(np.uint8)]
    if (out == 255).any():
        bad = arr[(out == 255).any(axis=1)][:5]
        raise ValueError(f"Alfabede olmayan karakter iceren diziler: {list(bad)}")
    return out


def decode_one(idx_row) -> str:
    return "".join(IDX_TO_AA[int(i)] for i in idx_row)


def decode_many(mat) -> list[str]:
    return [decode_one(r) for r in np.asarray(mat)]


def validate_lengths(sequences) -> None:
    bad = [s for s in sequences[:1000] if len(s) != PEPTIDE_LENGTH]
    if bad:
        raise ValueError(f"12-mer olmayan diziler var: {bad[:5]}")
