"""Ortak figur stili (shared figure style for every paper figure).

Tek yerden yonetilen renk/yazitipi/dpi ayarlari: butun figurler ayni
gorsel dile sahip olsun diye. Renkler acik zeminde ayirt edilebilir ve
gri tonlamada da siralanabilir sekilde secilmistir.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300

# Mimariler icin sabit renk atamasi; butun figurlerde ayni mimari ayni renk.
ARCH_COLOR = {
    "cnn":      "#1b4f72",
    "encdec":   "#2e86c1",
    "lstm":     "#85c1e9",
    "lstm_vae": "#d5dbdb",
}
ARCH_LABEL = {
    "cnn": "CNN", "encdec": "EncDec", "lstm": "LSTM", "lstm_vae": "LSTM-VAE",
}

# Bolme stratejileri: kume bazli olan vurgulu, rastgele olan soluk.
SPLIT_COLOR = {"clustered": "#1b4f72", "random": "#aab7b8"}
SPLIT_LABEL = {"clustered": "Identity-aware split", "random": "Random split"}

ACCENT = "#c0392b"      # vurgu / referans cizgileri
NEUTRAL = "#566573"
GRID = "#d5d8dc"

SEQ_CMAP = "YlGnBu"     # tek yonlu buyukluk
DIV_CMAP = "RdBu_r"     # isaretli (korelasyon vb.)


def apply() -> None:
    """Global matplotlib ayarlarini uygular."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": NEUTRAL,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.autolayout": False,
    })


def save(fig, out_dir: Path, name: str) -> Path:
    """Figuru PNG (raster) ve PDF (vektor) olarak kaydeder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    return png
