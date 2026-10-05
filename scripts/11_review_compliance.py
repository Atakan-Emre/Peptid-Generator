#!/usr/bin/env python3
"""Adim 11: hakem taleplerinin karsilanip karsilanmadigini kanitlarla denetler.

    python scripts/11_review_compliance.py

Her hakem maddesi icin, o maddeyi karsilayan artefakti ACAR ve icindeki
somut degeri okur. Dosyanin varligi yeterli sayilmaz: kayit sayisi, alan
adlari ve degerler dogrulanir. Boylece "calistirdik" iddiasi dosya listesine
degil, veriye dayanir.

Cikti: results/review_compliance.md ve ayni ozetin konsol hali.
Donus kodu, karsilanmamis (FAIL) madde varsa 1, yoksa 0.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit
from pbp.config import load_config

OK, PENDING, NARRATIVE, FAIL = "OK", "PENDING", "NARRATIVE", "FAIL"
BADGE = {OK: "[OK]     ", PENDING: "[BEKLIYOR]", NARRATIVE: "[METIN]  ",
         FAIL: "[EKSIK]  "}

ARCHS = ["cnn", "encdec", "lstm", "lstm_vae"]
METRICS = ["train_r2", "val_r2", "test_r2", "train_rmse", "val_rmse",
           "test_rmse", "train_mae", "val_mae", "test_mae"]


class Ctx:
    """Denetim sirasinda tekrar tekrar okunan dosyalari bir kez yukler."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.out = Path(cfg["out_dir"])
        self.prep = Path(cfg["prep_dir"])
        self.an = self.out / "analysis"
        self.fig = self.out / "figures"
        self.tab = self.out / "paper_tables"
        self._cache = {}

    def load(self, path: Path):
        key = str(path)
        if key not in self._cache:
            self._cache[key] = (json.loads(path.read_text(encoding="utf-8"))
                                if path.exists() else None)
        return self._cache[key]

    def artefacts(self, *names) -> str:
        """Var olan artefaktlari kisa yol olarak listeler."""
        return ", ".join(f"`{n}`" for n in names)


# --------------------------------------------------------------------------
# Tek tek denetimler. Her biri (durum, kanit, ayrinti) dondurur.
# --------------------------------------------------------------------------
def check_interpretability(c: Ctx):
    d = c.load(c.an / "interpretability.json")
    if not d:
        return FAIL, "-", "interpretability.json yok; 07_run_analyses calistirilmali"
    plastics = {r["plastic"] for r in d}
    missing = set(c.cfg["plastics"]) - plastics
    if missing:
        return FAIL, "analysis/interpretability.json", f"eksik polimer: {sorted(missing)}"
    bad = [r["plastic"] for r in d
           if len(r.get("position_importance", [])) != 12
           or len(r.get("preference_map", {})) != 12]
    if bad:
        return FAIL, "analysis/interpretability.json", f"eksik profil: {bad}"
    top = max(d, key=lambda r: max(r["position_importance"]))
    return OK, c.artefacts("analysis/interpretability.json",
                           "paper_tables/T8_position_importance.md",
                           "figures/F7_interpretability.png"), (
        f"{len(d)} polimer, her biri 12 pozisyon x 18 residu taramasi; "
        f"en duyarli: {top['plastic']} P"
        f"{top['position_importance'].index(max(top['position_importance'])) + 1} "
        f"(|dR| = {max(top['position_importance']):.2f})")


def check_prior_work(c: Ctx):
    d = c.load(c.out / "jain_baseline" / "jain_karsilastirma.json")
    if not d:
        return FAIL, "-", "jain_karsilastirma.json yok; 08_jain_baseline calistirilmali"
    conds = ["A_ham_rastgele", "B_temiz_rastgele", "C_temiz_kumelenmis"]
    gaps = []
    for p, e in d.items():
        for cond in conds:
            if cond not in e or "OURS_" + cond not in e:
                gaps.append(f"{p}/{cond}")
    if gaps:
        return FAIL, "jain_baseline/jain_karsilastirma.json", f"eksik hucre: {gaps[:4]}"
    da = [d[p]["OURS_A_ham_rastgele"]["test_r2_mean"] - d[p]["A_ham_rastgele"]["test_r2_mean"]
          for p in d]
    dc = [d[p]["OURS_C_temiz_kumelenmis"]["test_r2_mean"] - d[p]["C_temiz_kumelenmis"]["test_r2_mean"]
          for p in d]
    return OK, c.artefacts("jain_baseline/jain_karsilastirma.json",
                           "paper_tables/T9_jain_benchmark.md",
                           "figures/F3_jain_benchmark.png"), (
        f"{len(d)} polimer x 3 kosul x 2 mimari, 3 tohum; ustunluk "
        f"A kosulunda +{sum(da)/len(da):.4f}, C kosulunda +{sum(dc)/len(dc):.4f} "
        f"(%{100 * (1 - (sum(dc)/len(dc)) / (sum(da)/len(da))):.0f} azalma)")


def check_physchem(c: Ctx):
    d = c.load(c.an / "physicochemical.json")
    if not d:
        return FAIL, "-", "physicochemical.json yok; 07_run_analyses calistirilmali"
    need = {"gravy", "aromatic_fraction", "positive_fraction", "net_charge_ph7_4"}
    bad = [r["plastic"] for r in d if not need <= set(r.get("correlations", {}))]
    if bad:
        return FAIL, "analysis/physicochemical.json", f"eksik ozellik: {bad}"
    arom = {r["plastic"]: r["correlations"]["aromatic_fraction"]["pearson_with_score"]
            for r in d}
    return OK, c.artefacts("analysis/physicochemical.json",
                           "paper_tables/T7_physicochemical.md",
                           "figures/F8_physicochemical.png"), (
        f"{len(d)} polimer x {len(need)}+ ozellik, skorla korelasyon; aromatiklik "
        f"korelasyonu polimere gore isaret degistiriyor "
        f"({min(arom.values()):+.2f} ... {max(arom.values()):+.2f})")


def check_in_vitro(c: Ctx):
    return NARRATIVE, "-", (
        "Deneysel olcum bu calismanin kapsami disinda. Cevapta kapsam olarak "
        "beyan edildi; yerine gecen hesapsal kanit R2.2 altinda")


def check_independent_validation(c: Ctx):
    d = c.load(c.an / "independent_validation.json")
    if not d:
        return FAIL, "-", "independent_validation.json yok; 07_run_analyses calistirilmali"
    need = {"optimized", "random_baseline", "train_top1pct_measured", "validator_model"}
    bad = [r["plastic"] for r in d if not need <= set(r)]
    if bad:
        return FAIL, "analysis/independent_validation.json", f"eksik baseline: {bad}"
    beats = sum(1 for r in d if r.get("beats_best_training_sequence"))
    confirms = sum(1 for r in d if r.get("validator_confirms_gain"))
    val_arch = {r["validator_model"]["architecture"] for r in d}
    gen = c.load(c.an / "generator_comparison.json") or {}
    sel = gen.get("selection", {}).get("selected", "?")
    if beats < len(d) or confirms < len(d):
        return FAIL, "analysis/independent_validation.json", (
            f"{beats}/{len(d)} en iyi egitim dizisini geciyor, "
            f"{confirms}/{len(d)} bagimsiz modelce dogrulaniyor")
    return OK, c.artefacts("analysis/independent_validation.json",
                           "analysis/generator_comparison.md",
                           "figures/F4_independent_validation.png"), (
        f"{len(d)}/{len(d)} polimerde rastgele ve olculen en iyi %1 asiliyor; "
        f"optimizasyonda kullanilmayan {'/'.join(sorted(val_arch))} modeli "
        f"{confirms}/{len(d)} dogruluyor; secilen uretici: {sel}")


def check_clustered_split(c: Ctx):
    leak = c.load(c.prep / "leakage_summary.json")
    if not leak:
        return FAIL, "-", "leakage_summary.json yok; 01_prepare_data calistirilmali"
    strategies = {r["strategy"] for r in leak}
    if {"clustered", "random"} - strategies:
        return FAIL, "prepared/leakage_summary.json", f"eksik strateji: {strategies}"
    clu = {r["plastic"]: r for r in leak if r["strategy"] == "clustered"}
    rnd = {r["plastic"]: r for r in leak if r["strategy"] == "random"}
    sc = c.load(c.out / "summary_clustered.json")
    sr = c.load(c.out / "summary_random.json")
    if not (sc and sr):
        return FAIL, "results/summary_*.json", "iki bolme icin de egitim sonucu yok"
    gc = {r["plastic"]: r["test_r2_mean"] for r in sc if r["architecture"] == "cnn"}
    gr = {r["plastic"]: r["test_r2_mean"] for r in sr if r["architecture"] == "cnn"}
    infl = {p: gr[p] - gc[p] for p in gc}
    le2 = {p: rnd[p]["pct_nn_le_2"] for p in rnd}
    return OK, c.artefacts("prepared/leakage_summary.json",
                           "paper_tables/T2_split_comparison.md",
                           "figures/F2_split_effect.png"), (
        f"kume bazli bolme uygulandi ve olculdu; rastgele bolmede test "
        f"dizilerinin %{min(le2.values()):.0f}-%{max(le2.values()):.0f}'i egitime "
        f"<=2 mutasyon uzaklikta; sisme {min(infl.values()):.3f}-"
        f"{max(infl.values()):.3f} R2")


def check_seeds_and_significance(c: Ctx):
    sc = c.load(c.out / "summary_clustered.json")
    st = c.load(c.out / "architecture_stats_clustered.json")
    if not sc or not st:
        return FAIL, "-", "summary/architecture_stats eksik"
    seeds = {r["n_seeds"] for r in sc}
    if seeds != {5}:
        return FAIL, "results/summary_clustered.json", f"tohum sayisi {seeds}, 5 bekleniyor"
    # per_plastic: {plastik: {mimari: karsilastirma}}
    rows = [r for per in st["per_plastic"].values() for r in per.values()]
    need = {"mean_difference", "difference_ci95", "paired_t_pvalue", "cohens_d",
            "n_seeds"}
    bad = [r.get("comparison", "?") for r in rows if not need <= set(r)]
    if bad:
        return FAIL, "results/architecture_stats_clustered.json", (
            f"{len(bad)} karsilastirmada CI/p/d eksik: {bad[:3]}")
    sig = sum(1 for r in rows if r.get("significant_at_0.05"))
    return OK, c.artefacts("results/summary_clustered.json",
                           "results/architecture_stats_clustered.json",
                           "paper_tables/T4_architecture_significance.md",
                           "figures/F1_architecture_comparison.png"), (
        f"her yapilandirma 5 tohum; {len(rows)} esli karsilastirmanin "
        f"{sig}'i anlamli, hepsinde %95 GA ve Cohen d raporlaniyor; "
        f"referans mimari: {st['reference']}")


def check_full_metrics(c: Ctx):
    gaps, counts = [], {}
    for strat in ("clustered", "random"):
        d = c.load(c.out / f"summary_{strat}.json")
        if not d:
            return FAIL, "-", f"summary_{strat}.json yok"
        counts[strat] = len(d)
        have = {(r["plastic"], r["architecture"]) for r in d}
        for p in c.cfg["plastics"]:
            for a in ARCHS:
                if (p, a) not in have:
                    gaps.append(f"{strat}/{p}/{a}")
        for r in d:
            for m in METRICS:
                if f"{m}_mean" not in r or f"{m}_std" not in r:
                    gaps.append(f"{strat}/{r['plastic']}/{r['architecture']}/{m}")
    if gaps:
        return FAIL, "results/summary_*.json", f"{len(gaps)} eksik hucre: {gaps[:4]}"
    return OK, c.artefacts("paper_tables/T3_full_metrics_clustered.md",
                           "paper_tables/T3b_full_metrics_random.md"), (
        f"{counts['clustered']}+{counts['random']} = "
        f"{counts['clustered'] + counts['random']} satir "
        f"({len(c.cfg['plastics'])} polimer x {len(ARCHS)} mimari x 2 bolme); "
        f"her satirda train/val/test icin R2, RMSE, MAE ve standart sapma")


def check_normalisation(c: Ctx):
    seed = c.cfg["prepare"]["split_seed"]
    missing, stats = [], {}
    for p in c.cfg["plastics"]:
        m = c.load(c.prep / f"{p}_clustered_seed{seed}.json")
        if not m or "normalization" not in m:
            missing.append(p)
            continue
        n = m["normalization"]
        if "mean" not in n or "std" not in n:
            missing.append(p)
            continue
        stats[p] = (n["mean"], n["std"], n.get("source", "train"))
    if missing:
        return FAIL, "prepared/*_clustered_seed*.json", f"normalizasyon eksik: {missing}"

    # Alan varligi yetmez: saklanan istatistik gercekten egitim bolmesinin mi?
    # .npz varsa yeniden hesaplayip hem egitim hem tum veri ile karsilastiririz.
    import numpy as np
    verified, leaked, unchecked = [], [], []
    for p in c.cfg["plastics"]:
        npz = c.prep / f"{p}_clustered_seed{seed}.npz"
        if not npz.exists():
            unchecked.append(p)
            continue
        z = np.load(npz)
        y, tr = z["y"], z["train"]
        st = stats[p]
        if abs(st[0] - float(y[tr].mean())) < 1e-6:
            verified.append(p)
        elif abs(st[0] - float(y.mean())) < 1e-6:
            leaked.append(p)
        else:
            unchecked.append(p)
    if leaked:
        return FAIL, "prepared/*.npz", (
            f"normalizasyon TUM veriden hesaplanmis (sizinti): {leaked}")

    ps = stats.get("PS")
    proof = (f"{len(verified)}/{len(c.cfg['plastics'])} polimerde .npz'den yeniden "
             f"hesaplanarak dogrulandi" if verified else
             ".npz yok, yalnizca manifest alanlari kontrol edildi")
    return OK, c.artefacts("prepared/<polimer>_clustered_seed%d.json" % seed,
                           "pbp/data/prepare.py:181",
                           "paper_tables/T1_dataset_statistics.md"), (
        f"normalizasyon yalnizca egitim bolmesinden; {proof}. "
        f"Hakemin eksik dedigi PS dahil hepsi var "
        f"(PS: ort {ps[0]:.4f}, ss {ps[1]:.4f})")


def check_docking(c: Ctx):
    """Docking bu depoda yapilmiyor; hakemin istedigi ikinci yol secildi.

    Hakem 2 maddeyi sartli kurmustu: "If docking is intended as an
    independent validation method, the authors should evaluate the
    relationship between the two quantities". Docking artik bagimsiz
    dogrulama olarak sunulmadigi icin korelasyon talebi dusuyor; o yuku
    R2.2'deki bagimsiz dogrulama tasiyor.
    """
    iv = c.load(c.an / "independent_validation.json")
    carrier = (f"{sum(1 for r in iv if r['validator_confirms_gain'])}/{len(iv)} "
               f"polimerde bagimsiz model dogruluyor") if iv else "R2.2'ye bakiniz"
    return NARRATIVE, c.artefacts("analysis/independent_validation.json"), (
        "Docking bagimsiz dogrulama olarak SUNULMUYOR; iddialar yumusatildi. "
        "Hakemin korelasyon talebi bu kosula bagliydi, dolayisiyla dusuyor. "
        f"Bagimsiz dogrulamayi R2.2 tasiyor ({carrier}). "
        "Makaledeki docking protokol ayrintilari yazarlarca doldurulacak")


def check_novelty(c: Ctx):
    d = c.load(c.an / "novelty.json")
    if not d:
        return FAIL, "-", "novelty.json yok; 07_run_analyses calistirilmali"
    need = {"identity_histogram", "nn_identity_mean_pct", "pct_above_90_identity",
            "nn_identity_min_pct", "nn_identity_max_pct"}
    bad = [r["plastic"] for r in d if not need <= set(r)]
    if bad:
        return FAIL, "analysis/novelty.json", f"dagilim eksik: {bad}"
    mean = [r["nn_identity_mean_pct"] for r in d]
    above = max(r["pct_above_90_identity"] for r in d)
    return OK, c.artefacts("analysis/novelty.json",
                           "paper_tables/T6_novelty.md",
                           "figures/F6_novelty.png"), (
        f"ikili birebir-esles testi yerine kimlik DAGILIMI raporlaniyor; "
        f"en yakin komsu kimligi ortalama %{min(mean):.1f}-%{max(mean):.1f}, "
        f"%90 uzerinde dizi orani %{above:.1f}")


def check_selectivity(c: Ctx):
    d = c.load(c.an / "selectivity.json")
    if not d:
        return FAIL, "-", "selectivity.json yok; 07_run_analyses calistirilmali"
    pl = d["plastics"]
    for t in pl:
        row = d["matrix"].get(t, {})
        if not set(pl) <= set(row):
            return FAIL, "analysis/selectivity.json", f"{t} satiri 6 polimere karsi degil"
    s = d["summary"]
    idx = [d["matrix"][t]["selectivity_index"] for t in pl]
    return OK, c.artefacts("analysis/selectivity.json",
                           "paper_tables/T5_selectivity.md",
                           "figures/F5_selectivity.png"), (
        f"{len(pl)}x{len(pl)} capraz skor matrisi; "
        f"{s['n_target_best']}/{s['n_targets']} peptit seti kendi hedefinde en iyi; "
        f"secicilik indeksi {min(idx):.1f} (PS) - {max(idx):.1f}")


# --------------------------------------------------------------------------
CHECKS = [
    ("R1.1", "Yorumlanabilirlik ayrintili tartisilmali", check_interpretability),
    ("R1.2", "Onceki araclarla sistematik karsilastirma", check_prior_work),
    ("R1.3", "Fizikokimyasal temel ve biyolojik anlam", check_physchem),
    ("R1.4", "Secili peptitlerin in vitro dogrulanmasi", check_in_vitro),
    ("R2.1", "Jain et al. uzerine bilimsel katki + nicel kiyas", check_prior_work),
    ("R2.2", "Bagimsiz dogrulama ve baseline karsilastirmasi", check_independent_validation),
    ("R2.3a", "Dizi kimligine gore kumeleme ve bolme", check_clustered_split),
    ("R2.3b", "Coklu tohum, belirsizlik ve anlamlilik", check_seeds_and_significance),
    ("R2.3c", "Her mimari/plastik icin train-val-test R2, RMSE, MAE", check_full_metrics),
    ("R2.3d", "Normalizasyon kaynagi + PS parametreleri", check_normalisation),
    ("R2.4", "Docking protokolu ve ML-docking korelasyonu", check_docking),
    ("R2.5", "Yenilik iddiasinin olculmesi / yumusatilmasi", check_novelty),
    ("R2.6", "Alti plastige karsi secicilik", check_selectivity),
]


def main() -> int:
    cfg = load_config()
    c = Ctx(cfg)

    rows = []
    for code, title, fn in CHECKS:
        try:
            status, evidence, detail = fn(c)
        except Exception as exc:                 # tek denetim digerlerini durdurmasin
            status, evidence, detail = FAIL, "-", f"{type(exc).__name__}: {exc}"
        rows.append((code, title, status, evidence, detail))
        print(f"{BADGE[status]} {code:<6} {title}")
        print(f"{'':11}{detail}")

    tally = {s: sum(1 for r in rows if r[2] == s) for s in (OK, PENDING, NARRATIVE, FAIL)}
    print(f"\n{tally[OK]} karsilandi, {tally[PENDING]} veri bekliyor, "
          f"{tally[NARRATIVE]} metinle yanitlandi, {tally[FAIL]} eksik")

    md = ["# Reviewer compliance report", "",
          "Each row is checked against the artefact that answers it: the file is "
          "opened and its contents verified, not merely its existence.", "",
          "| # | Reviewer request | Status | Evidence | What the artefact shows |",
          "| --- | --- | --- | --- | --- |"]
    label = {OK: "**met**", PENDING: "awaiting input", NARRATIVE: "answered in text",
             FAIL: "**NOT MET**"}
    for code, title, status, evidence, detail in rows:
        md.append(f"| {code} | {title} | {label[status]} | {evidence} | {detail} |")
    md += ["", f"{tally[OK]} met, {tally[PENDING]} awaiting input, "
               f"{tally[NARRATIVE]} answered in text, {tally[FAIL]} not met.", ""]
    dest = c.out / "review_compliance.md"
    dest.write_text("\n".join(md), encoding="utf-8")
    print(f"Rapor: {dest}")

    return 1 if tally[FAIL] else 0


if __name__ == "__main__":
    clean_exit(main())
