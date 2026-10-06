"""Makale figurleri (paper figures), dogrudan results ciktilarindan.

Her fonksiyon bir hakem talebine karsilik gelir ve yalnizca pipeline'in
yazdigi JSON/CSV dosyalarini okur; model yeniden calistirilmaz. Boylece
figurler tablolarla ayni sayilardan uretilir ve ikisi ayrisamaz.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import style

ARCH_ORDER = ["cnn", "encdec", "lstm", "lstm_vae"]


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _by(rows, *keys):
    """Kayit listesini anahtar demetine gore sozluge cevirir."""
    return {tuple(r[k] for k in keys): r for r in rows}


def _polymer_order(cfg) -> list[str]:
    return list(cfg["plastics"])


# --------------------------------------------------------------------------
# F1 - mimari karsilastirmasi (Reviewer 2.3b, 2.3c)
# --------------------------------------------------------------------------
def architecture_comparison(out_root: Path, fig_dir: Path, cfg) -> Path | None:
    polymers = _polymer_order(cfg)
    panels = []
    for strat in ("clustered", "random"):
        f = out_root / f"summary_{strat}.json"
        if f.exists():
            panels.append((strat, _by(_load(f), "plastic", "architecture")))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 3.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(polymers))
    width = 0.8 / len(ARCH_ORDER)

    for ax, (strat, data) in zip(axes, panels):
        for j, arch in enumerate(ARCH_ORDER):
            mean = [data.get((p, arch), {}).get("test_r2_mean", np.nan) for p in polymers]
            err = [data.get((p, arch), {}).get("test_r2_std", 0.0) for p in polymers]
            ax.bar(x + j * width - 0.4 + width / 2, mean, width,
                   yerr=err, capsize=2, label=style.ARCH_LABEL[arch],
                   color=style.ARCH_COLOR[arch],
                   edgecolor="white", linewidth=0.5,
                   error_kw={"elinewidth": 0.8, "ecolor": style.NEUTRAL})
        ax.set_xticks(x)
        ax.set_xticklabels(polymers)
        ax.set_title(style.SPLIT_LABEL[strat])
        ax.set_ylim(0.6, 1.0)
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
    axes[0].set_ylabel("Test $R^2$ (mean +/- SD, 5 seeds)")
    axes[-1].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.0, -0.09))
    fig.suptitle("Architecture comparison across polymers and splitting strategies",
                 fontsize=10.5, y=1.00)
    return style.save(fig, fig_dir, "F1_architecture_comparison")


# --------------------------------------------------------------------------
# F2 - bolme stratejisinin etkisi (Reviewer 2.3a)
# --------------------------------------------------------------------------
def split_effect(out_root: Path, prep_dir: Path, fig_dir: Path, cfg) -> Path | None:
    leak_f = prep_dir / "leakage_summary.json"
    clu_f = out_root / "summary_clustered.json"
    rnd_f = out_root / "summary_random.json"
    if not (leak_f.exists() and clu_f.exists() and rnd_f.exists()):
        return None

    polymers = _polymer_order(cfg)
    leak = _by(_load(leak_f), "plastic", "strategy")
    clu = _by(_load(clu_f), "plastic", "architecture")
    rnd = _by(_load(rnd_f), "plastic", "architecture")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 3.8))
    x = np.arange(len(polymers))
    w = 0.38

    # (a) ayni mimari, iki bolme
    r_clu = [clu[(p, "cnn")]["test_r2_mean"] for p in polymers]
    r_rnd = [rnd[(p, "cnn")]["test_r2_mean"] for p in polymers]
    ax1.bar(x - w / 2, r_rnd, w, label=style.SPLIT_LABEL["random"],
            color=style.SPLIT_COLOR["random"], edgecolor="white", linewidth=0.5)
    ax1.bar(x + w / 2, r_clu, w, label=style.SPLIT_LABEL["clustered"],
            color=style.SPLIT_COLOR["clustered"], edgecolor="white", linewidth=0.5)
    for xi, (a, b) in enumerate(zip(r_rnd, r_clu)):
        ax1.annotate(f"-{a - b:.3f}", (xi, max(a, b) + 0.008), ha="center",
                     fontsize=7, color=style.ACCENT)
    ax1.set_ylim(0.6, 1.02)
    ax1.set_ylabel("Test $R^2$ (CNN)")
    ax1.set_title("(a) Measured performance by split")
    ax1.legend(loc="lower right")

    # (b) test dizilerinin egitime yakinligi
    le2_r = [leak[(p, "random")]["pct_nn_le_2"] for p in polymers]
    le2_c = [leak[(p, "clustered")]["pct_nn_le_2"] for p in polymers]
    ax2.bar(x - w / 2, le2_r, w, color=style.SPLIT_COLOR["random"],
            edgecolor="white", linewidth=0.5)
    ax2.bar(x + w / 2, le2_c, w, color=style.SPLIT_COLOR["clustered"],
            edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Test sequences <= 2 mutations from train (%)")
    ax2.set_title("(b) Near-neighbour leakage")

    # (c) sizinti ile sisirilen performans arasindaki iliski
    delta = np.array(r_rnd) - np.array(r_clu)
    extra = np.array(le2_r) - np.array(le2_c)
    ax3.scatter(extra, delta, s=46, color=style.SPLIT_COLOR["clustered"], zorder=3)
    for xi, yi, p in zip(extra, delta, polymers):
        ax3.annotate(p, (xi, yi), textcoords="offset points", xytext=(5, 3),
                     fontsize=7.5)
    if len(polymers) > 2:
        k, b = np.polyfit(extra, delta, 1)
        xs = np.linspace(extra.min(), extra.max(), 10)
        r = np.corrcoef(extra, delta)[0, 1]
        ax3.plot(xs, k * xs + b, "--", color=style.ACCENT, linewidth=1.1,
                 label=f"Pearson r = {r:.2f}")
        ax3.legend(loc="upper left")
    ax3.set_xlabel("Extra near neighbours under random split (pp)")
    ax3.set_ylabel("$R^2$ inflation (random - identity-aware)")
    ax3.set_title("(c) Inflation tracks leakage")

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(polymers)
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
    fig.suptitle("Effect of the data-splitting strategy", fontsize=10.5, y=1.02)
    return style.save(fig, fig_dir, "F2_split_effect")


# --------------------------------------------------------------------------
# F3 - onceki calisma ile kiyas (Reviewer 2.1)
# --------------------------------------------------------------------------
# Kisa etiket: uzun kosul adlari dar panellerde ust uste biniyor, aciklama
# figurun altinda tek satirda veriliyor.
COND = [("A_ham_rastgele", "A"),
        ("B_temiz_rastgele", "B"),
        ("C_temiz_kumelenmis", "C")]
COND_NOTE = ("A: raw data, random split (setup of Jain et al.)   "
             "B: duplicates removed, random split   "
             "C: duplicates removed, identity-aware split (this work)")


def jain_benchmark(out_root: Path, fig_dir: Path) -> Path | None:
    f = out_root / "jain_baseline" / "jain_karsilastirma.json"
    if not f.exists():
        return None
    data = _load(f)
    polymers = list(data)

    fig, axes = plt.subplots(1, len(polymers), figsize=(2.35 * len(polymers), 3.7),
                             sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(COND))
    w = 0.38

    for ax, p in zip(axes, polymers):
        prior = [data[p].get(c, {}).get("test_r2_mean", np.nan) for c, _ in COND]
        prior_e = [data[p].get(c, {}).get("test_r2_std", 0.0) for c, _ in COND]
        ours = [data[p].get("OURS_" + c, {}).get("test_r2_mean", np.nan) for c, _ in COND]
        ours_e = [data[p].get("OURS_" + c, {}).get("test_r2_std", 0.0) for c, _ in COND]
        ax.bar(x - w / 2, prior, w, yerr=prior_e, capsize=2,
               color=style.ARCH_COLOR["lstm"], edgecolor="white", linewidth=0.5,
               label="Jain et al. (LSTM)",
               error_kw={"elinewidth": 0.8, "ecolor": style.NEUTRAL})
        ax.bar(x + w / 2, ours, w, yerr=ours_e, capsize=2,
               color=style.ARCH_COLOR["cnn"], edgecolor="white", linewidth=0.5,
               label="This work (CNN)",
               error_kw={"elinewidth": 0.8, "ecolor": style.NEUTRAL})
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in COND])
        ax.set_xlabel("Data condition")
        ax.set_title(p)
        ax.set_ylim(0.6, 1.0)
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
    axes[0].set_ylabel("Test $R^2$ (mean +/- SD, 3 seeds)")
    axes[0].legend(ncol=2, loc="upper center",
                   bbox_to_anchor=(len(polymers) / 2.0, -0.17))
    fig.text(0.5, -0.10, COND_NOTE, ha="center", fontsize=7.5,
             color=style.NEUTRAL)
    fig.suptitle("Benchmark against Jain et al. (2025) under three data conditions",
                 fontsize=10.5, y=1.01)
    return style.save(fig, fig_dir, "F3_jain_benchmark")


# --------------------------------------------------------------------------
# F4 - bagimsiz dogrulama (Reviewer 2.2)
# --------------------------------------------------------------------------
def independent_validation(out_root: Path, fig_dir: Path, cfg) -> Path | None:
    f = out_root / "analysis" / "independent_validation.json"
    if not f.exists():
        return None
    recs = {r["plastic"]: r for r in _load(f)}
    polymers = [p for p in _polymer_order(cfg) if p in recs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    x = np.arange(len(polymers))
    w = 0.26
    series = [("random_baseline", "Random 12-mers", "#aab7b8"),
              ("train_top1pct_measured", "Best 1 % of measured training data", "#5d8aa8"),
              ("optimized", "Optimised peptides (this work)", style.ARCH_COLOR["cnn"])]

    for j, (key, lbl, col) in enumerate(series):
        m = [recs[p][key]["mean"] for p in polymers]
        s = [recs[p][key]["std"] for p in polymers]
        ax1.bar(x + (j - 1) * w, m, w, yerr=s, capsize=2, label=lbl, color=col,
                edgecolor="white", linewidth=0.5,
                error_kw={"elinewidth": 0.8, "ecolor": style.NEUTRAL})
    ax1.set_xticks(x)
    ax1.set_xticklabels(polymers)
    ax1.set_ylabel("Predicted binding score (lower = stronger)")
    ax1.set_title("(a) Target model")
    ax1.legend(loc="lower left", fontsize=7.5)
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(False)

    # (b) optimizasyonda kullanilmayan ucuncu model ile ayni kiyas.
    # Dogrulayici model yalnizca optimize edilen set ile rastgele temeli skorlar;
    # egitim verisinin en iyi %1'i hedef model altinda raporlanir.
    vm = [recs[p].get("validator_model") for p in polymers]
    if all(v for v in vm):
        present = [(k, lbl, col) for k, lbl, col in series if k in vm[0]]
        vw = 0.8 / (len(present) + 1)
        for j, (key, lbl, col) in enumerate(present):
            m = [v[key]["mean"] for v in vm]
            s = [v[key]["std"] for v in vm]
            ax2.bar(x + (j - (len(present) - 1) / 2) * vw, m, vw, yerr=s, capsize=2,
                    label=lbl, color=col, edgecolor="white", linewidth=0.5,
                    error_kw={"elinewidth": 0.8, "ecolor": style.NEUTRAL})
        arch = vm[0]["architecture"]
        ax2.set_title(f"(b) Independent {style.ARCH_LABEL.get(arch, arch)} model, "
                      "used in neither optimisation nor selection", fontsize=9.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(polymers)
        ax2.set_axisbelow(True)
        ax2.xaxis.grid(False)
        ax2.legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Optimised sequences against baselines, scored by two independent models",
                 fontsize=10.5, y=1.02)
    return style.save(fig, fig_dir, "F4_independent_validation")


# --------------------------------------------------------------------------
# F5 - secicilik (Reviewer 2.6)
# --------------------------------------------------------------------------
def selectivity_heatmap(out_root: Path, fig_dir: Path) -> Path | None:
    f = out_root / "analysis" / "selectivity.json"
    if not f.exists():
        return None
    d = _load(f)
    polymers = d["plastics"]
    m = np.array([[d["matrix"][t][s] for s in polymers] for t in polymers])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    im = ax.imshow(m, cmap=style.SEQ_CMAP + "_r", aspect="auto")
    ax.set_xticks(range(len(polymers)))
    ax.set_xticklabels(polymers)
    ax.set_yticks(range(len(polymers)))
    ax.set_yticklabels(polymers)
    ax.set_xlabel("Scored against polymer model")
    ax.set_ylabel("Peptide set optimised for")
    ax.grid(False)
    lo, hi = m.min(), m.max()
    for i in range(len(polymers)):
        for j in range(len(polymers)):
            v = m[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v < lo + 0.45 * (hi - lo) else "#1c2833",
                    fontweight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=style.ACCENT, linewidth=1.8))
    fig.colorbar(im, ax=ax, label="Predicted score", shrink=0.85)
    ax.set_title("(a) Cross-polymer score matrix")

    idx = [d["matrix"][t]["selectivity_index"] for t in polymers]
    order = np.argsort(idx)[::-1]
    ax2.barh([polymers[i] for i in order], [idx[i] for i in order],
             color=style.ARCH_COLOR["encdec"], edgecolor="white", linewidth=0.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Selectivity index\n(mean of other five - target)")
    ax2.set_title("(b) Target specificity")
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(False)
    fig.suptitle("Every peptide set scores best on the polymer it was optimised for",
                 fontsize=10.5, y=1.00)
    return style.save(fig, fig_dir, "F5_selectivity")


# --------------------------------------------------------------------------
# F6 - yenilik (Reviewer 2.5)
# --------------------------------------------------------------------------
BINS = [("identity_0_60", "< 60 %"), ("identity_60_70", "60-70 %"),
        ("identity_70_80", "70-80 %"), ("identity_80_90", "80-90 %"),
        ("identity_90_100", "90-100 %"), ("identity_100", "exact match")]


def novelty(out_root: Path, fig_dir: Path, cfg) -> Path | None:
    f = out_root / "analysis" / "novelty.json"
    if not f.exists():
        return None
    recs = {r["plastic"]: r for r in _load(f)}
    polymers = [p for p in _polymer_order(cfg) if p in recs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 3.9),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    colors = plt.get_cmap(style.SEQ_CMAP)(np.linspace(0.25, 0.95, len(BINS)))
    bottom = np.zeros(len(polymers))
    for (key, lbl), col in zip(BINS, colors):
        vals = np.array([recs[p]["identity_histogram"].get(key, 0) for p in polymers],
                        dtype=float)
        ax1.bar(polymers, vals, 0.62, bottom=bottom, label=lbl, color=col,
                edgecolor="white", linewidth=0.5)
        bottom += vals
    ax1.set_ylabel("Generated peptides (n = 30 per polymer)")
    ax1.set_title("(a) Nearest-neighbour identity to the training data")
    # Legend grafigin altinda: yan tarafa konunca komsu paneli ortuyor.
    ax1.legend(title="Identity to the closest training sequence", ncol=3,
               fontsize=7.5, title_fontsize=7.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.09))
    ax1.set_ylim(0, 34)
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(False)

    mean = [recs[p]["nn_identity_mean_pct"] for p in polymers]
    lo = [recs[p]["nn_identity_mean_pct"] - recs[p]["nn_identity_min_pct"]
          for p in polymers]
    hi = [recs[p]["nn_identity_max_pct"] - recs[p]["nn_identity_mean_pct"]
          for p in polymers]
    ax2.errorbar(mean, range(len(polymers)), xerr=[lo, hi], fmt="o",
                 color=style.ARCH_COLOR["cnn"], capsize=3, markersize=5)
    ax2.axvline(91.7, color=style.ACCENT, linestyle="--", linewidth=1.1)
    ax2.annotate("91.7 % = one mutation\nfrom a training sequence",
                 xy=(0.965, 0.03), xycoords="axes fraction", fontsize=7,
                 color=style.ACCENT, ha="right", va="bottom")
    ax2.set_yticks(range(len(polymers)))
    ax2.set_yticklabels(polymers)
    ax2.invert_yaxis()
    ax2.set_xlim(40, 100)
    ax2.set_xlabel("Identity to nearest training sequence (%)\nmean, with min-max range")
    ax2.set_title("(b) Distance from the training set")
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(False)
    fig.suptitle("Sequence novelty reported as an identity distribution, "
                 "not as absence of exact matches", fontsize=10.5, y=1.02)
    return style.save(fig, fig_dir, "F6_novelty")


# --------------------------------------------------------------------------
# F7 - yorumlanabilirlik (Reviewer 1.1)
# --------------------------------------------------------------------------
def interpretability(out_root: Path, fig_dir: Path, cfg) -> Path | None:
    f = out_root / "analysis" / "interpretability.json"
    if not f.exists():
        return None
    recs = {r["plastic"]: r for r in _load(f)}
    polymers = [p for p in _polymer_order(cfg) if p in recs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    imp = np.array([recs[p]["position_importance"] for p in polymers])
    im = ax1.imshow(imp, cmap=style.SEQ_CMAP, aspect="auto")
    ax1.set_xticks(range(imp.shape[1]))
    ax1.set_xticklabels(range(1, imp.shape[1] + 1))
    ax1.set_yticks(range(len(polymers)))
    ax1.set_yticklabels(polymers)
    ax1.set_xlabel("Residue position")
    ax1.grid(False)
    ax1.set_title("(a) Positional sensitivity of the model")
    fig.colorbar(im, ax=ax1, label="Mean |delta score| on mutation", shrink=0.85)

    # Ortalama amino asit tercihi: butun pozisyonlar uzerinden
    aas = sorted(recs[polymers[0]]["preference_map"]["pos1"])
    pref = np.zeros((len(polymers), len(aas)))
    for i, p in enumerate(polymers):
        pm = recs[p]["preference_map"]
        for j, aa in enumerate(aas):
            pref[i, j] = np.mean([pm[k][aa] for k in pm])
    order = np.argsort(pref.mean(axis=0))
    pref = pref[:, order]
    aas = [aas[j] for j in order]
    lim = np.abs(pref).max()
    im2 = ax2.imshow(pref, cmap=style.DIV_CMAP, aspect="auto", vmin=-lim, vmax=lim)
    ax2.set_xticks(range(len(aas)))
    ax2.set_xticklabels(aas)
    ax2.set_yticks(range(len(polymers)))
    ax2.set_yticklabels(polymers)
    ax2.set_xlabel("Amino acid (sorted by mean effect)")
    ax2.grid(False)
    ax2.set_title("(b) Residue preference, averaged over positions")
    fig.colorbar(im2, ax=ax2, shrink=0.85,
                 label="Mean predicted score when substituted\n(lower = stronger binding)")
    fig.suptitle("What the model has learned: position sensitivity and residue preference",
                 fontsize=10.5, y=1.02)
    return style.save(fig, fig_dir, "F7_interpretability")


# --------------------------------------------------------------------------
# F8 - fizikokimyasal temel (Reviewer 1.3)
# --------------------------------------------------------------------------
PROPS = [("aromatic_fraction", "Aromatic fraction"),
         ("gravy", "GRAVY (hydropathy)"),
         ("positive_fraction", "Positive fraction"),
         ("negative_fraction", "Negative fraction"),
         ("net_charge_ph7_4", "Net charge (pH 7.4)"),
         ("isoelectric_point", "Isoelectric point")]


def physicochemical(out_root: Path, fig_dir: Path, cfg) -> Path | None:
    f = out_root / "analysis" / "physicochemical.json"
    if not f.exists():
        return None
    recs = {r["plastic"]: r for r in _load(f)}
    polymers = [p for p in _polymer_order(cfg) if p in recs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    avail = recs[polymers[0]]["correlations"]
    keys = [k for k, _ in PROPS if k in avail]
    labels = [lbl for k, lbl in PROPS if k in avail]
    m = np.array([[recs[p]["correlations"][k]["pearson_with_score"] for p in polymers]
                  for k in keys])
    lim = np.abs(m).max()
    im = ax1.imshow(m, cmap=style.DIV_CMAP, vmin=-lim, vmax=lim, aspect="auto")
    ax1.set_xticks(range(len(polymers)))
    ax1.set_xticklabels(polymers)
    ax1.set_yticks(range(len(keys)))
    ax1.set_yticklabels(labels)
    ax1.grid(False)
    for i in range(len(keys)):
        for j in range(len(polymers)):
            ax1.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="#1c2833" if abs(m[i, j]) < 0.6 * lim else "white")
    fig.colorbar(im, ax=ax1, label="Pearson r with predicted score", shrink=0.85)
    ax1.set_title("(a) Property-affinity relationships")

    # En guclu iliski: aromatiklik ve skor
    cmap = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(polymers)))
    for p, col in zip(polymers, cmap):
        pp = recs[p]["per_peptide"]
        ax2.scatter([r["aromatic_fraction"] for r in pp], [r["score"] for r in pp],
                    s=18, alpha=0.75, label=p, color=col, edgecolor="white",
                    linewidth=0.4)
    ax2.set_xlabel("Aromatic residue fraction (F, W, Y, H)")
    ax2.set_ylabel("Predicted binding score (lower = stronger)")
    # Isaretin polimere gore degistigi (a) panelinde gorulyor; baslik bu yuzden
    # tek yonlu bir iliski iddia etmiyor.
    ax2.set_title("(b) Aromaticity against affinity, polymer by polymer")
    ax2.legend(ncol=2, fontsize=7.5, loc="upper right")
    fig.suptitle("Physicochemical basis of the designed peptides: "
                 "property-affinity relationships are polymer-specific",
                 fontsize=10.5, y=1.02)
    return style.save(fig, fig_dir, "F8_physicochemical")


# --------------------------------------------------------------------------
# F9 - ML skoru ile docking skoru iliskisi (Reviewer 2.4)
# --------------------------------------------------------------------------
def docking_correlation(out_root: Path, fig_dir: Path) -> Path | None:
    f = out_root / "analysis" / "docking_correlation.json"
    if not f.exists():
        return None
    d = _load(f)
    pairs = d.get("pairs") or []
    if not pairs:
        return None
    polymers = sorted({r["plastic"] for r in pairs})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    cmap = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(polymers)))
    for p, col in zip(polymers, cmap):
        sel = [r for r in pairs if r["plastic"] == p]
        ax1.scatter([r["ml_score"] for r in sel], [r["docking_score"] for r in sel],
                    s=22, alpha=0.8, label=p, color=col, edgecolor="white",
                    linewidth=0.4)
    ml = np.array([r["ml_score"] for r in pairs])
    dk = np.array([r["docking_score"] for r in pairs])
    if len(pairs) > 2:
        k, b = np.polyfit(ml, dk, 1)
        xs = np.linspace(ml.min(), ml.max(), 20)
        ax1.plot(xs, k * xs + b, "--", color=style.ACCENT, linewidth=1.2)
    po = d.get("pooled", {})
    note = []
    if "pearson" in po:
        note.append(f"pooled Pearson r = {po['pearson']:.2f}")
    if "spearman" in po:
        note.append(f"Spearman rho = {po['spearman']:.2f}")
    if note:
        ax1.annotate("   ".join(note), (0.03, 0.04), xycoords="axes fraction",
                     fontsize=8, color=style.ACCENT)
    ax1.set_xlabel("Predicted (surrogate) score")
    ax1.set_ylabel("Docking score (kcal/mol)")
    ax1.set_title("(a) Surrogate score vs docking score")
    ax1.legend(ncol=2, fontsize=7.5)

    per = d.get("per_plastic", {})
    have = [p for p in polymers if "spearman" in per.get(p, {})]
    if have:
        ax2.barh(have, [per[p]["spearman"] for p in have],
                 color=style.ARCH_COLOR["encdec"], edgecolor="white", linewidth=0.5)
        ax2.axvline(0, color=style.NEUTRAL, linewidth=0.8)
        ax2.invert_yaxis()
        ax2.set_xlabel("Spearman rho (surrogate vs docking)")
        ax2.set_title("(b) Rank agreement per polymer")
        ax2.set_axisbelow(True)
        ax2.yaxis.grid(False)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Per-polymer correlation not reported:\n"
                           "fewer than 8 docked peptides per polymer.",
                 ha="center", va="center", fontsize=8, color=style.NEUTRAL)
    fig.suptitle("Docking as supporting evidence, quantified rather than asserted",
                 fontsize=10.5, y=1.02)
    return style.save(fig, fig_dir, "F9_docking_correlation")
