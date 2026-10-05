#!/usr/bin/env python3
"""Step 6: comparison of the surrogate models used for generation.

Optimising a learned surrogate can produce sequences that exploit the
model's error instead of binding the target. This step tests for that: ayni optimizasyon algoritmalari farkli
vekillerle kosulur, sonuclar secicilik / dogrulayici uyumu / yozlasma
olcutleriyle karsilastirilir ve kazanan deterministik bir kuralla secilir.

Secilen kumenin peptitleri, sonraki analizlerin okudugu
`generated/peptides_<strateji>.json` dosyasina yazilir.

    python scripts/05_compare_generators.py --strategy clustered
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit
from pbp.config import load_config
from pbp.device import configure
from pbp.analysis import generator_comparison as gc
from pbp.analysis.scoring import load_all_models


def surrogate_gaps(cfg, arch, strategy):
    """summary_<strateji>.json'dan mimarinin egitim-test R2 acigini okur."""
    f = Path(cfg["out_dir"]) / f"summary_{strategy}.json"
    if not f.exists():
        return None
    out = {}
    for s in json.loads(f.read_text()):
        if s["architecture"] == arch:
            out[s["plastic"]] = s["train_r2_mean"] - s["test_r2_mean"]
    return out or None


def main():
    cfg = load_config()
    an = cfg.get("analysis", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="clustered")
    ap.add_argument("--validator-architecture",
                    default=an.get("validator_architecture", "lstm"))
    a = ap.parse_args()

    print("cihaz:", configure(), flush=True)
    gen_dir = Path(cfg["out_dir"]) / "generated"
    out_root = Path(cfg["out_dir"]) / "analysis"
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg["out_dir"]) / "models"

    archs = an.get("generator_candidates", [an.get("best_architecture", "encdec")])
    available = [x for x in archs
                 if (gen_dir / f"peptides_{x}_{a.strategy}.json").exists()]
    if not available:
        sys.exit(f"{gen_dir} altinda uretim yok. Once 05_generate_peptides.py calistirin.")
    if len(available) < len(archs):
        print(f"  ! eksik uretim: {sorted(set(archs) - set(available))}")

    # Secicilik ve dogrulama, HER ZAMAN uretimde kullanilmayan bagimsiz bir
    # mimariyle yapilir; boylece karsilastirma adaylardan birine yanli olmaz.
    validators = {}
    try:
        validators = load_all_models(models_dir, a.validator_architecture, a.strategy)
        print(f"bagimsiz dogrulayici: {a.validator_architecture}")
    except FileNotFoundError:
        print(f"  ! {a.validator_architecture} modelleri yok; dogrulayici testi atlanacak")

    candidates = []
    for arch in available:
        print(f"\n=== DEGERLENDIRME: {arch} ===", flush=True)
        peptides = json.loads((gen_dir / f"peptides_{arch}_{a.strategy}.json").read_text())
        # Capraz skorlama, uretici mimarinin KENDI modelleriyle yapilir ki
        # secicilik testi uretimin yapildigi uzayda olculsun.
        models = load_all_models(models_dir, arch, a.strategy)
        res = gc.evaluate_candidate(arch, peptides, models, validators,
                                    surrogate_gaps(cfg, arch, a.strategy))
        s = res["summary"]
        print(f"  hedef-en-iyi {s['n_target_best']}/{s['n_plastics']}  "
              f"dogrulayici kazanci {s['mean_validator_gain']}  "
              f"en sik AA orani {s['mean_max_aa_fraction']}  "
              f"dejenere {s['n_degenerate']}/{s['n_plastics']}")
        for p, d in res["per_plastic"].items():
            if not d["target_is_best"]:
                print(f"    ! {p}: en iyi skoru {d['best_scoring_plastic']} aliyor")
            if d["degeneracy"].get("is_degenerate"):
                print(f"    ! {p}: yozlasmis diziler "
                      f"({d['degeneracy']['most_common_aa']} "
                      f"%{d['degeneracy']['max_aa_fraction'] * 100:.0f})")
        candidates.append(res)

    result = {"strategy": a.strategy,
              "validator_architecture": a.validator_architecture,
              "candidates": candidates,
              "selection": gc.select(candidates)}
    gc.save(result, out_root)

    chosen = result["selection"]["selected"]
    print(f"\n=== SECIM: {chosen} ===")
    print(result["selection"]["reason"])

    src = gen_dir / f"peptides_{chosen}_{a.strategy}.json"
    (gen_dir / f"peptides_{a.strategy}.json").write_text(src.read_text())
    (gen_dir / f"_selected_generator_{a.strategy}.json").write_text(
        json.dumps({"architecture": chosen,
                    "reason": result["selection"]["reason"],
                    "fallback": result["selection"]["fallback"]}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print(f"Secilen kume -> generated/peptides_{a.strategy}.json")
    print(f"Rapor -> {out_root / 'generator_comparison.md'}")


if __name__ == "__main__":
    main()
    clean_exit(0)
