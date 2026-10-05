#!/usr/bin/env python3
"""Tum hatti sirayla calistirir; Windows ve Linux'ta ayni sekilde.

Her adim ayri bir surec olarak kosar ve kendi log dosyasina yazar. Bir adim
beklenmedik sekilde olurse (surucu cokmesi, gecici bellek hatasi) 60 saniye
sonra kaldigi yerden tekrar denenir; tamamlanan egitim kosulari diske
kaydedildigi icin tekrarlanmaz.

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --from 03      # belirli adimdan devam
"""
import argparse, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
MAX_ATTEMPTS = 10

# Windows'ta CUDA kullanan bir surec, isini bitirdikten sonra yorumlayici
# kapanirken bu kodla cokebiliyor (STATUS_STACK_BUFFER_OVERRUN). pbp/exit.py
# bunu os._exit ile onluyor; yine de olusursa, adim basari damgasini
# birakmissa is gercekten tamamlanmis demektir.
WINDOWS_TEARDOWN_CRASH = 3221226505

STEPS = [
    ("00", "cihaz dogrulama",          ["scripts/00_check_device.py"]),
    ("01", "veri hazirligi + sizinti", ["scripts/01_prepare_data.py"]),
    ("02", "sure butcesi",             ["scripts/02_estimate_budget.py"]),
    ("03", "Asama A: hiperparametre",  ["scripts/03_search_hyperparams.py"]),
    ("04", "Asama B: kume bazli",      ["scripts/04_train_final.py", "--strategy", "clustered"]),
    ("05", "Asama B: rastgele",        ["scripts/04_train_final.py", "--strategy", "random"]),
    ("06", "Jain karsilastirmasi",     ["scripts/08_jain_baseline.py",
                                        "--our-architecture", "auto"]),
    ("07", "peptit uretimi",           ["scripts/05_generate_peptides.py"]),
    ("08", "uretici karsilastirmasi",  ["scripts/06_compare_generators.py"]),
    ("09", "peptit analizleri",         ["scripts/07_run_analyses.py"]),
    ("10", "makale tablolari",         ["scripts/09_build_paper_tables.py"]),
    ("11", "makale figurleri",         ["scripts/10_build_figures.py"]),
    ("12", "hakem uyum denetimi",      ["scripts/11_review_compliance.py"]),
]


def stamp() -> str:
    return datetime.now().strftime("%d %b %H:%M:%S")


def run_step(num, desc, argv) -> bool:
    script = argv[0].split("/")[-1].replace(".py", "")
    log = LOGS / f"{num}_{script}.log"
    stamp_file = LOGS / f".ok_{script}"
    before = stamp_file.stat().st_mtime if stamp_file.exists() else 0.0

    print(f"\n>>> [{num}] {desc}  ({stamp()})", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== {stamp()} =====\n")
        fh.flush()
        rc = subprocess.call([sys.executable, "-u"] + argv, cwd=ROOT,
                             stdout=fh, stderr=subprocess.STDOUT)
    mins = (time.time() - t0) / 60

    # Adim bu kosuda yeni bir basari damgasi biraktiysa is tamamlanmistir.
    finished = stamp_file.exists() and stamp_file.stat().st_mtime > before

    if rc == 0:
        print(f"    tamam ({mins:.0f} dk) -> {log.name}", flush=True)
        return True
    if finished:
        extra = (" (Windows CUDA kapanis cokmesi; is tamamlandi)"
                 if rc == WINDOWS_TEARDOWN_CRASH else
                 f" (cikis {rc}, ama is tamamlandi)")
        print(f"    tamam ({mins:.0f} dk){extra} -> {log.name}", flush=True)
        return True
    print(f"    HATA (cikis {rc}, {mins:.0f} dk) -> {log.name}", flush=True)
    try:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        print("    son satirlar:", flush=True)
        for line in tail:
            print("      " + line, flush=True)
    except OSError:
        pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="00",
                    help="bu adim numarasindan itibaren calistir")
    a = ap.parse_args()
    LOGS.mkdir(exist_ok=True)

    steps = [s for s in STEPS if int(s[0]) >= int(a.start)]
    print("=" * 50)
    print(f" Peptid-Generator hatti - {len(steps)} adim")
    print(f" baslangic: {stamp()}")
    print("=" * 50, flush=True)

    for num, desc, argv in steps:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if run_step(num, desc, argv):
                break
            if attempt == MAX_ATTEMPTS:
                print(f"\n{MAX_ATTEMPTS} denemeden sonra vazgecildi: [{num}] {desc}")
                return 1
            print(f"    60 sn sonra tekrar denenecek (deneme {attempt + 1}/{MAX_ATTEMPTS})",
                  flush=True)
            time.sleep(60)

    print("\n" + "=" * 50)
    print(f" TAMAMLANDI: {stamp()}")
    print(" Tablolar: results/paper_tables/ALL_TABLES.md")
    print(" Jain karsilastirmasi: results/jain_baseline/jain_karsilastirma.md")
    print(" Uretici karsilastirmasi: results/analysis/generator_comparison.md")
    print("=" * 50, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
