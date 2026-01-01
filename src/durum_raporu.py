"""
Ablation Study Durum Raporu
Kullanım: python durum_raporu.py
"""
import os
import pandas as pd
import glob
from datetime import datetime

def detect_plastic_types_from_csv(data_dir):
    """
    Data klasöründeki CSV dosyalarını tarayıp plastik tiplerini otomatik tespit eder.
    
    Args:
        data_dir: Veri klasörü yolu
        
    Returns:
        List[str]: Tespit edilen plastik tipleri (CSV dosya adlarından .csv uzantısı çıkarılarak)
    """
    if not os.path.exists(data_dir):
        return []
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    plastic_types = [os.path.splitext(f)[0] for f in csv_files]  # .csv uzantısını kaldır
    
    return sorted(plastic_types)  # Alfabetik sırala

def get_status_report():
    """Detaylı durum raporu oluştur"""
    
    print("=" * 80)
    print("ABLATION STUDY DURUM RAPORU")
    print("=" * 80)
    print(f"Rapor Zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Veri klasörünü dinamik olarak bul (newDate veya Data)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "newDate")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(script_dir, "Data")
    
    # Plastik tiplerini dinamik olarak tespit et
    detected_plastics = detect_plastic_types_from_csv(data_dir)
    
    if not detected_plastics:
        print(f"[UYARI] Veri klasöründe CSV dosyasi bulunamadi: {data_dir}")
        print("   Mevcut dosyalar:", os.listdir(data_dir) if os.path.exists(data_dir) else "Klasör yok")
        return
    
    print(f"[BILGI] Tespit edilen plastik tipleri: {', '.join(detected_plastics)}")
    print()
    
    # Tablo dosyalarını bul
    tables_dir = "ablation_results/tables"
    ablation_files = glob.glob(os.path.join(tables_dir, "ablation_*.csv"))
    
    if not ablation_files:
        print("[UYARI] Henuz tamamlanmis ablation study yok.")
        return
    
    # Plastik tipine göre grupla
    plastic_status = {}
    
    print("HÜCRE DURUMLARI:")
    print("-" * 80)
    
    completed_models = {}
    for file in ablation_files:
        try:
            df = pd.read_csv(file)
            filename = os.path.basename(file)
            # Model ve plastik tipini çıkar
            # Format: "ablation_lstm_vae_PET.csv" -> model="LSTM_VAE", plastic="PET"
            # Format: "ablation_lstm_PET.csv" -> model="LSTM", plastic="PET"
            name_part = filename.replace("ablation_", "").replace(".csv", "")
            parts = name_part.split("_")
            
            if len(parts) >= 2:
                # Son kısım her zaman plastik tipi
                plastic_type = parts[-1]
                
                # Geri kalanı model tipi
                if len(parts) == 3:  # lstm_vae_PET
                    model_type = f"{parts[0].upper()}_{parts[1].upper()}"
                elif len(parts) == 2:  # lstm_PET
                    model_type = parts[0].upper()
                else:
                    # Çok parça varsa son hariç hepsi model
                    model_type = "_".join(parts[:-1]).upper()
                
                if plastic_type not in plastic_status:
                    plastic_status[plastic_type] = {
                        'models': [],
                        'completed_count': 0,
                        'total_models': 4  # lstm, cnn, lstm_vae, encdec (dinamik olabilir)
                    }
                
                best_row = df.loc[df['val_r2'].idxmax()]
                
                completed_models[f"{model_type}_{plastic_type}"] = {
                    'model': model_type,
                    'plastic': plastic_type,
                    'total': len(df),
                    'best_val_r2': best_row['val_r2'],
                    'best_test_r2': best_row['test_r2'],
                    'best_params': best_row.to_dict()
                }
                
                plastic_status[plastic_type]['models'].append({
                    'model': model_type,
                    'status': 'TAMAMLANDI',
                    'val_r2': best_row['val_r2'],
                    'test_r2': best_row['test_r2']
                })
                plastic_status[plastic_type]['completed_count'] += 1
        except Exception as e:
            print(f"[HATA] {file} okunamadi: {e}")
    
    # Plastik bazında durum göster (dinamik olarak tespit edilen plastikler için)
    for plastic_type in detected_plastics:
        if plastic_type in plastic_status:
            status = plastic_status[plastic_type]
            completed = status['completed_count']
            total = status['total_models']
            progress = (completed / total * 100) if total > 0 else 0
            
            print(f"\n{plastic_type.upper()}:")
            print(f"   Ilerleme: {completed}/{total} model tamamlandi ({progress:.0f}%)")
            
            for model_info in status['models']:
                print(f"     - {model_info['model']}: TAMAMLANDI (Val R2: {model_info['val_r2']:.4f})")
            
            # Devam eden modelleri kontrol et
            if completed < total:
                print(f"     - Kalan: {total - completed} model devam ediyor veya bekliyor")
        else:
            print(f"\n{plastic_type.upper()}:")
            print(f"   Durum: Henuz baslamadi")
    
    print()
    print("=" * 80)
    print("DETAYLI MODEL SONUCLARI:")
    print("-" * 80)
    
    for key, info in completed_models.items():
        print(f"\n[TAMAMLANDI] {info['plastic']} - {info['model']}:")
        print(f"   Tamamlanan kombinasyon: {info['total']}")
        print(f"   En iyi Val R2: {info['best_val_r2']:.6f}")
        print(f"   En iyi Test R2: {info['best_test_r2']:.6f}")
        print(f"   En iyi parametreler:")
        for param_key in ['hidden_dim', 'num_layers', 'dropout', 'learning_rate', 'batch_size', 
                          'latent_dim', 'beta_kl', 'gamma_score', 'lambda_score']:
            if param_key in info['best_params'] and pd.notna(info['best_params'][param_key]):
                print(f"     - {param_key}: {info['best_params'][param_key]}")
    
    print()
    print("=" * 80)
    print("DEVAM EDEN MODELLER:")
    print("-" * 80)
    
    # Log dosyalarını kontrol et
    logs_dir = "ablation_results/logs"
    log_files = glob.glob(os.path.join(logs_dir, "ablation_log_*.txt"))
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        parts = filename.replace("ablation_log_", "").replace(".txt", "").split("_")
        
        if len(parts) >= 2:
            # Model tipi ve plastik tipini doğru çıkar
            # Format: "lstm_vae_PET" -> model="LSTM_VAE", plastic="PET"
            if len(parts) == 3:  # lstm_vae_PET gibi
                model_type = f"{parts[0].upper()}_{parts[1].upper()}"
                plastic_type = parts[2]
            else:  # lstm_PET gibi
                model_type = parts[0].upper()
                plastic_type = parts[1]
            key = f"{model_type}_{plastic_type}"
            
            # Eğer tamamlanmışsa atla
            if key in completed_models:
                continue
            
            # Log dosyasını oku
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Toplam kombinasyon sayısını bul
                import re
                total_match = re.search(r'Toplam Kombinasyon: (\d+)', content)
                total = int(total_match.group(1)) if total_match else 0
                
                # Mevcut kombinasyonu bul
                combo_matches = re.findall(r'KOMB[İI]NASYON (\d+)/(\d+)', content)
                if combo_matches:
                    current = int(combo_matches[-1][0])
                    total = int(combo_matches[-1][1]) if total > 0 else int(combo_matches[-1][1])
                    progress = (current / total * 100) if total > 0 else 0
                else:
                    current = 0
                    progress = 0
                
                # En iyi Val R²'yi bul
                best_r2_match = re.findall(r'Best Val R[²2]: ([\d.]+)', content)
                if not best_r2_match:
                    best_r2_match = re.findall(r'Best Val R.*?([\d.]+)', content)
                best_val_r2 = float(best_r2_match[-1]) if best_r2_match else 0.0
                
                print(f"\n[DEVAM] {plastic_type} - {model_type}:")
                print(f"   Ilerleme: {current}/{total} kombinasyon ({progress:.1f}%)")
                print(f"   Su anki en iyi Val R2: {best_val_r2:.6f}")
                print(f"   Log dosyasi: {filename}")
                
            except Exception as e:
                print(f"[HATA] {filename} okunamadi: {e}")
    
    print()
    print("=" * 80)
    print("ÜRETİLEN DOSYALAR:")
    print("-" * 80)
    
    # Final modeller
    models_dir = "ablation_results/final_models"
    if os.path.exists(models_dir):
        model_files = glob.glob(os.path.join(models_dir, "*.pth"))
        if model_files:
            print(f"\nFinal Modeller ({len(model_files)} adet):")
            for mf in model_files:
                size_mb = os.path.getsize(mf) / (1024 * 1024)
                print(f"   - {os.path.basename(mf)} ({size_mb:.2f} MB)")
        else:
            print("\nFinal Modeller: Henuz uretilmedi")
    
    # Grafikler
    figures_dir = "ablation_results/figures"
    if os.path.exists(figures_dir):
        fig_files = glob.glob(os.path.join(figures_dir, "*.png"))
        if fig_files:
            print(f"\nGrafikler ({len(fig_files)} adet):")
            for ff in fig_files[:5]:  # İlk 5'i göster
                print(f"   - {os.path.basename(ff)}")
            if len(fig_files) > 5:
                print(f"   ... ve {len(fig_files) - 5} adet daha")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    get_status_report()

