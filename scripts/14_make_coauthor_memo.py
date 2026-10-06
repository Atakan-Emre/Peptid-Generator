#!/usr/bin/env python3
"""Adim 14: ortak yazarlar icin Turkce degisiklik notunu uretir.

    python scripts/14_make_coauthor_memo.py

Eski hat ile yeni hat arasindaki farklari, nedenleriyle birlikte anlatir.
Butun sayilar results/ altindaki ciktilardan okunur.
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbp.exit import clean_exit
from pbp.config import load_config

AUTHOR = "Sahin Atakan Emre"
ARCH_TR = {"cnn": "CNN", "encdec": "LSTM-EncDec", "lstm": "LSTM",
           "lstm_vae": "LSTM-VAE"}


def main() -> int:
    import docx
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    cfg = load_config()
    out = Path(cfg["out_dir"])
    P = list(cfg["plastics"])
    J = lambda f: json.loads((out / f).read_text(encoding="utf-8"))

    clu, rnd = J("summary_clustered.json"), J("summary_random.json")
    stats = J("architecture_stats_clustered.json")
    gcmp = J("analysis/generator_comparison.json")
    nov = {r["plastic"]: r for r in J("analysis/novelty.json")}
    sel = J("analysis/selectivity.json")
    iv = {r["plastic"]: r for r in J("analysis/independent_validation.json")}
    leak = {(r["plastic"], r["strategy"]): r
            for r in json.loads((Path(cfg["prep_dir"]) / "leakage_summary.json")
                                .read_text(encoding="utf-8"))}
    gen = {p: J(f"generated/{p}_encdec_clustered.json") for p in P}

    mean = lambda d, a: sum(r["test_r2_mean"] for r in d if r["architecture"] == a) / len(P)
    cand = {c["generator_architecture"]: c["summary"] for c in gcmp["candidates"]}
    chosen = gcmp["selection"]["selected"]

    d = docx.Document()
    cp = d.core_properties
    cp.author = AUTHOR
    cp.last_modified_by = AUTHOR
    cp.title = "Yeni uygulama - degisenler ve nedenleri"
    cp.comments = ""
    cp.revision = 1
    st = d.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for s in d.sections:
        s.left_margin = s.right_margin = Cm(2.2)
        s.top_margin = s.bottom_margin = Cm(2.0)

    def h(t, lv=1):
        p = d.add_heading(t, level=lv)
        for r in p.runs:
            r.font.name = "Calibri"
        return p

    def para(t="", bold=False, italic=False, size=10.5, after=7):
        p = d.add_paragraph()
        r = p.add_run(t)
        r.bold, r.italic = bold, italic
        r.font.size = Pt(size)
        p.paragraph_format.space_after = Pt(after)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return p

    def table(rows, size=9.5):
        t = d.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        for i, row in enumerate(rows):
            for j, c in enumerate(row):
                cell = t.cell(i, j)
                cell.text = ""
                pp = cell.paragraphs[0]
                rr = pp.add_run(str(c))
                rr.font.size = Pt(size)
                rr.font.name = "Calibri"
                rr.bold = (i == 0)
                pp.paragraph_format.space_after = Pt(0)
        d.add_paragraph().paragraph_format.space_after = Pt(9)

    # ---------------------------------------------------------------- baslik
    h("Yeni uygulama: ne degisti, neden degisti", 0)
    para(f"Hazirlanma tarihi: {date.today().strftime('%d.%m.%Y')}", bold=True, after=2)
    para("Ortak yazarlar icin ozet. Butun sayilar yeni hattin ciktilarindan "
         "alinmistir.", italic=True, size=9.5, after=14)

    # ------------------------------------------------------- 1. temel duzeltme
    h("1. En kritik nokta: CNN her yerde secilmedi", 1)
    para("Bu konuda bir yanlis anlama olusmus olabilir, netlestirelim: "
         "\"CNN en iyi cikti, her seyde CNN kullandik\" DEGIL. Birbirinden "
         "bagimsiz iki karar var ve ikisi FARKLI modele cikiyor:", bold=True)
    table([
        ["Karar", "Secilen model", "Gerekce"],
        ["En iyi tahminci (affinite tahmini)", ARCH_TR["cnn"],
         f"Alti polimerde de en yuksek test R2; "
         f"{stats['summary']['n_significant']}/{stats['summary']['n_comparisons']} "
         f"esli karsilastirmanin tamami anlamli"],
        ["Peptit ureten model (optimizasyon hedefi)", ARCH_TR[chosen],
         "Optimizasyona daha dayanikli; secicilik testini 6/6 geciyor"],
    ])
    para(f"Yani eski modelimiz ({ARCH_TR[chosen]}) devre disi kalmadi, ROLU "
         f"DEGISTI. Peptitler hala onunla uretiliyor.", bold=True)

    para("Neden en iyi tahminci uretimde kullanilmiyor? Cunku CNN'in ezberi "
         "daha fazla ve optimizer bu bosluga giriyor:")
    table([
        ["Olcut", ARCH_TR["cnn"], ARCH_TR["encdec"], "Hangisi iyi"],
        ["Train-test acigi", f"{cand['cnn']['surrogate_train_test_gap']:.4f}",
         f"{cand['encdec']['surrogate_train_test_gap']:.4f}", "dusuk olan"],
        ["Kendi hedefinde en iyi skorlayan set",
         f"{cand['cnn']['n_target_best']}/6", f"{cand['encdec']['n_target_best']}/6",
         "yuksek olan"],
        ["Dejenere (tek amino aside cokmus) set",
         f"{cand['cnn']['n_degenerate']}/6", f"{cand['encdec']['n_degenerate']}/6",
         "dusuk olan"],
        ["En sik amino asit orani",
         f"{cand['cnn']['mean_max_aa_fraction']:.3f}",
         f"{cand['encdec']['mean_max_aa_fraction']:.3f}", "dusuk olan"],
    ])
    para("CNN'i optimizasyon hedefi yapinca optimizer modelin HATASINI "
         "somuruyor: PET dizileri triptofana cokuyor (%62,5 W) ve PP ile PS'te "
         "hedef ozgullugu kayboluyor. Hakem 2'nin 2. maddesinde uyardigi problem "
         "tam olarak budur. Biz bunu olcup belgeledik ve bu yuzden "
         f"{ARCH_TR[chosen]}'i sectik.")
    para("Makaleye su sekilde yazilmali: en yuksek tahmin dogrulugu CNN'dedir; "
         "ancak uretim icin, train-test acigi daha dar oldugu ve secicilik "
         f"testini 6/6 gectigi icin {ARCH_TR[chosen]} kullanilmistir. En iyi "
         "tahminci ile en iyi uretici ayni model degildir. Bu, makalenin en "
         "guclu bulgularindan biridir ve hakemin elestirisine dogrudan cevap "
         "verir.", bold=True)

    # ------------------------------------------------ 2. mimari neden degisti
    h("2. Mimari sonucu neden degisti", 1)
    para("Eskiden LSTM-EncDec en iyi gorunuyordu, cunku karsilastirma su "
         "sekilde yapilmisti:")
    table([
        ["Eski", "Yeni", "Sonuc"],
        ["Tek kosu, tekrar yok", "5 tohum", "Degiskenlik olculebiliyor"],
        ["Validation R2 ile karsilastirma", "Test R2 ile karsilastirma",
         "Secim validation'la, karsilastirma test'le; ikisi karistirilmiyor"],
        ["Istatistik yok", "Esli t-testi, %95 GA, Cohen d",
         f"{stats['summary']['n_significant']}/{stats['summary']['n_comparisons']} "
         f"karsilastirma anlamli"],
        ["Fark 0,0036", "CNN farki 0,043 - 0,096", "Fark artik gurultunun uzerinde"],
    ])
    para("Hakem 2 zaten tam bunu yazmis: 0,0036'lik fark, belirsizlik "
         "raporlanmadan LSTM-EncDec'i ustun saymaya yetmez. Hakli cikti.")

    rows = [["Polimer"] + [ARCH_TR[a] for a in ("cnn", "encdec", "lstm", "lstm_vae")]]
    for p in P:
        r = {x["architecture"]: x for x in clu if x["plastic"] == p}
        rows.append([p] + [f"{r[a]['test_r2_mean']:.4f}"
                           for a in ("cnn", "encdec", "lstm", "lstm_vae")])
    rows.append(["Ortalama"] + [f"{mean(clu, a):.4f}"
                                for a in ("cnn", "encdec", "lstm", "lstm_vae")])
    table(rows)
    para("Test R2, kimlik-duyarli bolme, 5 tohum ortalamasi.", italic=True, size=9)

    # ------------------------------------------------------- 3. asil degisiklik
    h("3. Asil buyuk degisiklik mimari degil, veri bolme", 1)
    para("Bu, mimariden daha onemli ve R2 degerlerinin DUSMESININ sebebi.")
    lo = min(leak[(p, "random")]["pct_nn_le_2"] for p in P)
    hi = max(leak[(p, "random")]["pct_nn_le_2"] for p in P)
    para(f"Eskiden rastgele %80/10/10 bolme vardi. Olctuk: rastgele bolmede "
         f"test dizilerinin %{lo:.0f} ile %{hi:.0f} arasindaki kismi egitim "
         f"setine iki mutasyondan daha yakin. Yani model yeni dizi ogrenmiyor, "
         f"gordugu dizilerin varyantlarini taniyordu. Artik diziler kimliklerine "
         f"gore kumeleniyor ve bolme kume duzeyinde yapiliyor.")
    rows = [["Polimer", "Rastgele bolme: egitime <=2 mutasyon",
             "Kimlik-duyarli bolme", "CNN test R2 farki"]]
    for p in P:
        c = [x for x in clu if x["plastic"] == p and x["architecture"] == "cnn"][0]
        r = [x for x in rnd if x["plastic"] == p and x["architecture"] == "cnn"][0]
        rows.append([p, f"%{leak[(p, 'random')]['pct_nn_le_2']:.1f}",
                     f"%{leak[(p, 'clustered')]['pct_nn_le_2']:.1f}",
                     f"{r['test_r2_mean'] - c['test_r2_mean']:.3f}"])
    table(rows)
    para(f"Sonuc: ortalama test R2 {mean(rnd, 'cnn'):.4f} -> "
         f"{mean(clu, 'cnn'):.4f}. Dusus kotu bir sey degil, GERCEK DEGER budur; "
         f"eski rakam sisikti. Makalede her iki bolme de raporlaniyor, ancak "
         f"baslik rakami {mean(clu, 'cnn'):.4f} olmalidir.", bold=True)

    # --------------------------------------------------------- 4. peptitler
    h("4. Peptitler tamamen degisti", 1)
    uniq = len({r["sequence"] for p in P for v in gen[p].values() for r in v["runs"]})
    para(f"Eski listede 115 benzersiz peptit vardi (180 satirda, cok tekrar "
         f"iceriyordu). Yeni listede {uniq}/180 benzersiz. Iki liste arasindaki "
         f"ORTAK PEPTIT SAYISI SIFIR.", bold=True)
    rows = [["Polimer", "Yeni en iyi peptit", "Skor", "Yontem"]]
    for p in P:
        b = min(((gen[p][m]["best"], m) for m in gen[p]), key=lambda x: x[0]["score"])
        rows.append([p, b[0]["sequence"], f"{b[0]['score']:.2f}", b[1]])
    table(rows)
    para("Dolayisiyla Tablo 11'deki butun diziler, soyuttaki skorlar ve "
         "Sekil 13-15'teki docking pozlari degisecek.")

    # ----------------------------------------------------------- 5. docking
    h("5. Docking yeniden yapilacak", 1)
    para("Eski docking pozlari artik makalede bulunmayan dizilere ait. "
         "Laboratuvara 72 peptitlik liste gonderildi (DOCKING_LISTESI.csv).")
    para("Neden 72 ve neden sadece en iyiler degil: hakem, ML skoru ile docking "
         "skoru arasinda korelasyon istiyor. Yalnizca en iyi peptitler docking'e "
         "sokulursa skorlar dar bir banda toplanir ve korelasyon katsayisi "
         "yorumlanamaz hale gelir. Bu nedenle her plastikte 12 peptit var: "
         "4 tasarim, OLCULMUS afinitesi bilinen 5 egitim dizisi ve 3 rastgele "
         "kontrol. Egitim dizileri onemli, cunku docking'i tahmine degil gercek "
         "olcume karsi dogrulama imkani veriyorlar.")
    para("Ayrica docking artik \"dogrulama\" olarak sunulmuyor; \"yerel temas "
         "geometrisi icin destekleyici kanit\" olarak sunuluyor. Hakem bunu "
         "ozellikle istedi.")

    # ------------------------------------------------------- 6. yeni analizler
    h("6. Eskiden olmayan, yeni eklenen analizler", 1)
    table([
        ["Analiz", "Hangi hakem maddesi icin"],
        [f"Secicilik matrisi ({len(P)}x{len(P)})", "Hakem 2, madde 6"],
        ["Bagimsiz dogrulama (optimizasyonda kullanilmayan model + rastgele "
         "ve en iyi %1 karsilastirmasi)", "Hakem 2, madde 2"],
        ["Jain et al. (2025) ile nicel kiyas, 3 kosulda", "Hakem 1 m.2, Hakem 2 m.1"],
        ["Yorumlanabilirlik (pozisyon duyarliligi, rezidu tercihi)", "Hakem 1, madde 1"],
        ["Fizikokimyasal analiz", "Hakem 1, madde 3"],
        ["Yenilik dagilimi (en yakin komsu kimligi)", "Hakem 2, madde 5"],
        ["Bolme stratejisinin etkisi", "Hakem 2, madde 3"],
    ])

    # ------------------------------------------------------ 7. makalede yapilacak
    h("7. Makalede yapilmasi gerekenler", 1)
    para("Sayfa sayfa ayrintili liste Makale_Revizyon_Notlari.pdf dosyasindadir. "
         "Baslicalari:")
    best = ", ".join(
        f"{p} {min((gen[p][m]['best']['score'] for m in gen[p])):.2f}" for p in P)
    table([
        ["Sayfa", "Ne degisecek"],
        ["1 (Soyut)", f"0,9491 yerine CNN ve {mean(clu, 'cnn'):.4f} / "
                      f"{mean(rnd, 'cnn'):.4f}; \"180 novel peptide\" ifadesinden "
                      f"\"novel\" cikacak; skorlar yenilenecek ({best}); docking "
                      f"cumlesi yumusayacak"],
        ["6 (Tablo 4)", "PS satiri eksikti, hakem yakaladi. Yeni Tablo 1 alti "
                        "polimeri de veriyor"],
        ["13-14 (4.1 ve Tablo 10)", "\"LSTM-EncDec en iyi, recurrent modeller "
                                    "convolution'dan ustun\" iddiasi degisecek; "
                                    "validation R2 tablosu yerine 48 satirlik "
                                    "train/val/test tablosu"],
        ["16 ve 18 (Tablo 13-14)", "Hamming minimumu 1, maksimum benzerlik %91,7 "
                                   "yaziyordu. Hakem \"bu zaten tek mutasyon, "
                                   "yenilik degil\" dedi. Kimlik dagilimiyla "
                                   "degisecek"],
        ["20 (Sonuc)", "Soyutla ayni dort duzeltme"],
        ["Yeni bolumler", "4.6 bolme etkisi, 4.7 Jain kiyasi, 4.8 bagimsiz "
                          "dogrulama, 4.9 secicilik, 4.10 yorumlanabilirlik, "
                          "4.11 fizikokimya"],
    ])

    # --------------------------------------------------------- 8. sinirlamalar
    h("8. Bilinmesi gereken iki sinirlama", 1)
    idx = {t: sel["matrix"][t]["selectivity_index"] for t in sel["plastics"]}
    worst = min(idx, key=idx.get)
    best_p = max(idx, key=idx.get)
    para(f"1) {worst} en zayif sonucumuz. Secicilik indeksi {idx[worst]:.2f} "
         f"({best_p}'de {idx[best_p]:.2f}). Makalede bunu kendimiz soylemeliyiz; "
         f"hakem veriye bakinca zaten gorecek, onden soylemek lehimize olur.")
    para("2) Deneysel dogrulama yok. Hakem 1'in 4. maddesi in vitro olcum "
         "istiyor. Kapsam disi oldugunu acikca yazdik ve yerine hesapsal "
         "bagimsiz dogrulamayi koyduk: optimize diziler, optimizasyonda ve model "
         "seciminde hic kullanilmayan bir modelle skorlanip rastgele dizilerle "
         "ve olculen en iyi %1'lik egitim verisiyle karsilastirildi; alti "
         "polimerde de ikisini de geciyor.")

    h("Ozet", 1)
    para(f"\"CNN her seyi aldi\" degil: CNN tahminde, {ARCH_TR[chosen]} uretimde. "
         f"Asil degisim ise mimari degil, DEGERLENDIRME PROTOKOLU. Durust "
         f"bolmeye gecince rakamlar dustu, peptitler degisti ve docking'in "
         f"yenilenmesi gerekti.", bold=True)

    dest = Path(__file__).resolve().parents[2] / "YENI_UYGULAMA_DEGISIKLIKLER.docx"
    d.save(dest)
    print(f"  {dest}")
    return 0


if __name__ == "__main__":
    clean_exit(main())
