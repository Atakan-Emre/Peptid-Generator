"""Cihaz secimi ve donanima gore bellek/batch ayarlari.

Desteklenen: NVIDIA CUDA (birincil), Apple MPS, CPU.

RTX 5070 Ti notu: Blackwell mimarisi (sm_120) CUDA 12.8 ve uzerini
gerektirir. Daha eski bir PyTorch tekerlegi kurulursa model yuklenir ama
"no kernel image is available" hatasi verir. RUN_ALL.bat cu128 tekerlegini
kurar; 00_check_device.py bunu dogrular.
"""
from __future__ import annotations
import os
import torch

# Batch tavanlari. Modeller kucuk (en fazla ~4M parametre, dizi uzunlugu 12),
# bu nedenle sinir bellek degil GPU doluluk oranidir: buyuk batch, LSTM'in
# sirali 12 adimini daha iyi amorti eder.
MAX_BATCH = {
    "cuda": {"lstm": 4096, "encdec": 4096, "lstm_vae": 4096, "cnn": 8192},
    "mps":  {"lstm": 1024, "encdec": 1024, "lstm_vae": 1024, "cnn": 2048},
    "cpu":  {"lstm": 512,  "encdec": 512,  "lstm_vae": 512,  "cnn": 1024},
}

# Veri setinin tamami bu esigin altindaysa cihaza bir kez tasinir ve
# egitim boyunca host-cihaz transferi hic yapilmaz. 715k ornek uint8
# olarak 8.6 MB, skorlar 2.9 MB - hepsi rahatlikla siginiyor.
RESIDENT_DATA_LIMIT_BYTES = 2 * 1024**3


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure(prefer: str = "auto") -> dict:
    """Cihazi secer, hizlandirma ayarlarini yapar ve ozet dondurur."""
    dev = get_device(prefer)
    info = {"device": str(dev)}
    torch.set_float32_matmul_precision("high")   # TF32 matmul

    if dev.type == "cuda":
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        info.update({
            "gpu": props.name,
            "vram_gb": round(props.total_memory / 1e9, 1),
            "capability": f"sm_{props.major}{props.minor}",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        })
        # Girdi boyutlari sabit (B, 12, 18) oldugu icin cuDNN en hizli
        # algoritmayi bir kez secip tekrar kullanabilir.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
        want = f"sm_{props.major}{props.minor}"
        if arch_list and want not in arch_list:
            info["WARNING"] = (
                f"Bu PyTorch derlemesi {want} desteklemiyor (destekledikleri: "
                f"{', '.join(arch_list)}). Blackwell icin cu128 tekerlegi gerekir: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu128")
    elif dev.type == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        info["torch"] = torch.__version__
    return info


# Geriye donuk uyumluluk (eski scriptler configure_for_m4 cagiriyordu)
configure_for_m4 = configure


def clamp_batch_size(architecture: str, requested: int, device=None) -> int:
    dev = (device or get_device()).type
    cap = MAX_BATCH.get(dev, MAX_BATCH["cpu"]).get(architecture, 1024)
    return min(requested, cap)


def can_hold_data(n_bytes: int, device) -> bool:
    """Veri setinin tamami cihaz belleginde tutulabilir mi?"""
    if device.type == "cpu":
        return True
    if n_bytes > RESIDENT_DATA_LIMIT_BYTES:
        return False
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        return n_bytes < free * 0.25     # modele ve aktivasyonlara yer birak
    return True


def free_memory(device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
