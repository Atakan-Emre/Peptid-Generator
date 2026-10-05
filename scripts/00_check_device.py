#!/usr/bin/env python3
"""Uzun kosuya baslamadan once GPU'yu dogrular.

Blackwell (RTX 50 serisi) icin en sik hata, PyTorch'un eski bir CUDA
tekerlegiyle kurulmus olmasidir: model yuklenir ama ilk kernel cagrisinda
"no kernel image is available for execution" hatasi gelir. Bu script
gercek bir ileri/geri gecis yaparak bunu simdiden yakalar.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit

import json
import torch
from pbp.device import configure, get_device, clamp_batch_size
from pbp.models.architectures import build_model, compute_loss


def main():
    info = configure()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    dev = get_device()

    if dev.type == "cpu":
        print("\n! GPU bulunamadi. Egitim CPU'da cok yavas olacak.")
        print("  PyTorch'u CUDA destegiyle kurun:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 1

    print(f"\nGercek hesap testi ({dev})...")
    ok = True
    for arch in ("lstm", "cnn", "lstm_vae", "encdec"):
        try:
            bs = min(512, clamp_batch_size(arch, 4096, dev))
            m = build_model(arch, hidden_dim=128, num_layers=2).to(dev)
            x = torch.randint(0, 18, (bs, 12), device=dev)
            y = torch.randn(bs, device=dev)
            m.train()
            loss, _ = compute_loss(arch, m(x), y, x)
            loss.backward()
            loss_val = float(loss.detach())
            if dev.type == "cuda":
                torch.cuda.synchronize()
            print(f"  {arch:9s} ileri+geri gecis OK  (loss {loss_val:.4f})")
            del m, x, y
        except Exception as e:
            ok = False
            print(f"  {arch:9s} BASARISIZ: {type(e).__name__}: {e}")
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if not ok:
        print("\n! En az bir mimari calismadi. Yukaridaki hata 'no kernel image'")
        print("  diyorsa PyTorch derlemesi bu GPU'yu desteklemiyor demektir:")
        print("  pip uninstall -y torch")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 1
    print("\nHazir. scripts/01_prepare_data.py ile devam edebilirsiniz.")
    return 0


if __name__ == "__main__":
    clean_exit(main() or 0)
