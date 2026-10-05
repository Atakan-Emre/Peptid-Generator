"""Windows + CUDA'da temiz surec sonlandirma.

Sorun
-----
Windows'ta CUDA kullanan bir Python sureci, isini bitirdikten SONRA,
yorumlayici kapanirken 0xC0000409 (3221226505, STATUS_STACK_BUFFER_OVERRUN)
ile cokebiliyor. Cikti tamamen yazilmis, dosyalar diske inmis oluyor;
coken sey yalnizca CUDA/cuDNN DLL'lerinin bosaltilma sirasi. Disaridan
bakan bir koordinator bunu basarisizlik sanar.

Cozum
-----
clean_exit() once CUDA'yi bosaltip akislari bosaltir, sonra os._exit ile
cikar. os._exit atexit islemcilerini ve yorumlayici yikimini atladigi icin
cokmenin olustugu asama hic calismaz.

Ayrica basarili bitiste logs/.ok_<script> damgasi birakilir; koordinator
beklenmedik bir cikis kodu gorse bile isin gercekten bittigini bu damgadan
dogrulayabilir.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

STAMP_DIR = Path(__file__).resolve().parents[1] / "logs"


def stamp_name(script: str | None = None) -> str:
    name = Path(script or sys.argv[0]).stem or "script"
    return f".ok_{name}"


def write_stamp(script: str | None = None) -> None:
    try:
        STAMP_DIR.mkdir(exist_ok=True)
        (STAMP_DIR / stamp_name(script)).write_text(str(time.time()))
    except OSError:
        pass


def clean_exit(code: int = 0) -> None:
    """Islem bitti; sureci yorumlayici yikimina sokmadan sonlandir."""
    if code == 0:
        write_stamp()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)
