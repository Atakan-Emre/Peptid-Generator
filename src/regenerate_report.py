"""Regenerate detailed report for any plastic type"""
import os, json, numpy as np
from datetime import datetime

AMINO_ACIDS = "ADEFGHIKLMNQRSTVWY"
PROJECT_ROOT = r"d:\GoogleDrive\Projeler\PeptidGenerator"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "peptide_generation_comparison")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

def calculate_aa_frequency(peptides):
    freq = {aa: 0 for aa in AMINO_ACIDS}
    total = 0
    for pep in peptides:
        for aa in pep:
            if aa in freq:
                freq[aa] += 1
                total += 1
    return {aa: (count / total * 100) if total > 0 else 0 for aa, count in freq.items()}

def regenerate_report(plastic_type):
    log_file = os.path.join(LOGS_DIR, f"log_{plastic_type}.json")
    if not os.path.exists(log_file):
        print(f"Log dosyasi bulunamadi: {log_file}")
        return
    
    with open(log_file, "r") as f:
        data = json.load(f)
    
    pep_res = {}
    for method, peps in data["peptides"].items():
        pep_res[method] = [{"best_peptide": p["peptide"], "best_score": p["score"], 
                            "initial_peptide": "N/A", "initial_score": 0, 
                            "improvement": p["improvement"], "method": method} for p in peps]
    
    train_res = data["training"]
    params = train_res.get("params", {})
    
    report_file = os.path.join(REPORTS_DIR, f"RAPOR_{plastic_type}.md")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# 🧬 Peptid Üretim Karşılaştırma Raporu - {plastic_type}\n\n")
        f.write(f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Platform:** Windows 11, RTX 4080 Super\n")
        f.write("**Model:** LSTM Encoder-Decoder (ENCDEC)\n\n")
        
        best_method = min(pep_res.keys(), key=lambda m: min(r["best_score"] for r in pep_res[m]))
        best_score = min(min(r["best_score"] for r in pep_res[m]) for m in pep_res)
        best_pep = min([r for m in pep_res for r in pep_res[m]], key=lambda x: x["best_score"])
        
        f.write("---\n\n## 📋 Özet\n\n")
        f.write(f"- **En İyi Yöntem:** {best_method}\n")
        f.write(f"- **En İyi Skor:** {best_score:.2f}\n")
        f.write(f"- **En İyi Peptid:** `{best_pep['best_peptide']}`\n")
        f.write(f"- **Model R²:** {train_res['test_r2']:.4f}\n\n")
        
        f.write("---\n\n## 📊 Model Eğitim Sonuçları\n\n")
        f.write("### Performans Metrikleri\n\n")
        f.write("| Metrik | Değer | Açıklama |\n|--------|-------|----------|\n")
        f.write(f"| **Test R²** | **{train_res['test_r2']:.4f}** | Model açıklayıcılığı |\n")
        f.write(f"| Test MAE | {train_res['test_mae']:.4f} | Ortalama mutlak hata |\n")
        f.write(f"| Test RMSE | {train_res['test_rmse']:.4f} | Kök ortalama kare hata |\n")
        f.write(f"| Best Epoch | {train_res['best_epoch']} / 250 | Early stopping |\n")
        f.write(f"| Eğitim Süresi | {train_res['time']:.0f}s | ~{train_res['time']/60:.1f} dakika |\n\n")
        
        f.write("### Model Hiperparametreleri\n\n")
        f.write("| Parametre | Değer |\n|-----------|-------|\n")
        f.write(f"| Hidden Dim | {params.get('hidden_dim', 256)} |\n")
        f.write(f"| Num Layers | {params.get('num_layers', 2)} |\n")
        f.write(f"| Dropout | {params.get('dropout', 0.1)} |\n")
        f.write(f"| Learning Rate | {params.get('learning_rate', 0.001)} |\n")
        f.write(f"| Batch Size | {params.get('batch_size', 1024)} |\n")
        f.write(f"| Lambda Score | {params.get('lambda_score', 1.0)} |\n")
        f.write(f"| Weight Decay | {params.get('weight_decay', 0.0001)} |\n\n")
        
        f.write("### Eğitim Grafiği\n\n")
        f.write(f"![Training Curve](../figures/training_{plastic_type}.png)\n\n")
        
        f.write("---\n\n## 🧬 Peptid Üretim Sonuçları\n\n")
        f.write("### Yöntem Karşılaştırması\n\n")
        f.write("| Yöntem | En İyi Skor | Ort. Skor | Std | İyileşme |\n")
        f.write("|--------|-------------|-----------|-----|----------|\n")
        
        method_stats = {}
        for m in pep_res:
            scores = [r["best_score"] for r in pep_res[m]]
            imps = [r["improvement"] for r in pep_res[m]]
            method_stats[m] = {"best": min(scores), "avg": np.mean(scores), "std": np.std(scores), "imp": np.mean(imps)}
            f.write(f"| **{m}** | {method_stats[m]['best']:.2f} | {method_stats[m]['avg']:.2f} | {method_stats[m]['std']:.2f} | {method_stats[m]['imp']:.2f} |\n")
        
        f.write(f"\n**🏆 En İyi Yöntem: {best_method}**\n\n")
        
        f.write("### Karşılaştırma Grafikleri\n\n")
        f.write(f"![Method Comparison](../figures/comparison_{plastic_type}.png)\n\n")
        f.write(f"![Score Distribution](../figures/score_dist_{plastic_type}.png)\n\n")
        f.write(f"![Optimization Trajectory](../figures/trajectory_{plastic_type}.png)\n\n")
        
        f.write("---\n\n## 📝 Üretilen Peptidler\n\n")
        for m in pep_res:
            f.write(f"### {m} Peptidleri\n\n")
            f.write("| # | Peptid | Skor | İyileşme |\n")
            f.write("|---|--------|------|----------|\n")
            sorted_res = sorted(pep_res[m], key=lambda x: x["best_score"])
            for i, r in enumerate(sorted_res, 1):
                f.write(f"| {i} | `{r['best_peptide']}` | {r['best_score']:.2f} | {r['improvement']:.2f} |\n")
            f.write("\n")
        
        f.write("---\n\n## 🔬 Amino Asit Frekans Analizi\n\n")
        f.write("### Genel AA Dağılımı\n\n")
        all_peptides = [r["best_peptide"] for m in pep_res for r in pep_res[m]]
        aa_freq = calculate_aa_frequency(all_peptides)
        sorted_aa = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)
        f.write("| AA | Frekans (%) | Görsel |\n|----|-----------:|--------|\n")
        for aa, freq in sorted_aa[:10]:
            bar = "█" * int(freq / 2) + "░" * (25 - int(freq / 2))
            f.write(f"| **{aa}** | {freq:.1f}% | {bar} |\n")
        
        f.write("\n### Yönteme Göre En Sık AA\n\n")
        f.write("| Yöntem | Top 5 AA |\n|--------|----------|\n")
        for m in pep_res:
            m_peps = [r["best_peptide"] for r in pep_res[m]]
            m_freq = calculate_aa_frequency(m_peps)
            top5 = sorted(m_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            top5_str = ", ".join([f"{aa}({f:.0f}%)" for aa, f in top5])
            f.write(f"| {m} | {top5_str} |\n")
        
        f.write(f"\n### AA Heatmap\n\n")
        f.write(f"![AA Heatmap](../figures/aa_heatmap_{plastic_type}.png)\n\n")
        
        f.write("---\n\n## 🔗 Peptid Benzerlik Analizi\n\n")
        f.write("### En İyi Peptidler Arası Hamming Mesafesi\n\n")
        best_peps = {m: min(pep_res[m], key=lambda x: x["best_score"])["best_peptide"] for m in pep_res}
        methods = list(best_peps.keys())
        f.write("| | " + " | ".join(methods) + " |\n")
        f.write("|" + "---|" * (len(methods) + 1) + "\n")
        for m1 in methods:
            row = f"| **{m1}** |"
            for m2 in methods:
                if m1 == m2:
                    row += " - |"
                else:
                    dist = sum(1 for a, b in zip(best_peps[m1], best_peps[m2]) if a != b)
                    row += f" {dist} |"
            f.write(row + "\n")
        
        f.write("\n### En İyi 3 Peptid\n\n")
        top3 = sorted([r for m in pep_res for r in pep_res[m]], key=lambda x: x["best_score"])[:3]
        for i, r in enumerate(top3, 1):
            f.write(f"{i}. `{r['best_peptide']}` ({r['method']}, Skor: {r['best_score']:.2f})\n")
        
        f.write("\n---\n\n## 📌 Sonuç\n\n")
        if train_res["test_r2"] >= 0.95:
            f.write("✅ **Model Performansı:** Mükemmel (R² ≥ 0.95)\n\n")
        elif train_res["test_r2"] >= 0.90:
            f.write("✅ **Model Performansı:** Çok İyi (R² ≥ 0.90)\n\n")
        
        best_m = min(method_stats.keys(), key=lambda m: method_stats[m]["best"])
        f.write(f"**En İyi Yöntem:** {best_m} (Skor: {method_stats[best_m]['best']:.2f})\n\n")
        f.write("---\n*Rapor: peptide_generation_comparison.py*\n")
    
    print(f"✓ RAPOR_{plastic_type}.md yeniden oluşturuldu!")

if __name__ == "__main__":
    import sys
    plastic = sys.argv[1] if len(sys.argv) > 1 else "PP"
    regenerate_report(plastic)
