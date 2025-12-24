#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study Sonuç Analizi ve Training Curves Görselleştirme
==============================================================
Bu script, tamamlanmış ablation study sonuçlarını analiz eder ve
en iyi modellerin training curves grafiklerini çizer.

Kullanım: python3 analiz_raporu.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI gerektirmez
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Proje dizinleri
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.join(PROJECT_ROOT, "ablation_results")
LOGS_DIR = os.path.join(ABLATION_DIR, "logs")
TABLES_DIR = os.path.join(ABLATION_DIR, "tables")
FIGURES_DIR = os.path.join(ABLATION_DIR, "figures")
REPORT_DIR = os.path.join(ABLATION_DIR, "analysis_reports")

os.makedirs(REPORT_DIR, exist_ok=True)

print("="*80)
print("ABLATION STUDY SONUÇ ANALİZİ VE TRAINING CURVES")
print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

def load_completed_results():
    """Tamamlanmış ablation sonuçlarını yükle"""
    results = {}
    
    # CSV dosyalarını tara
    if os.path.exists(TABLES_DIR):
        csv_files = [f for f in os.listdir(TABLES_DIR) if f.endswith('.csv') and f.startswith('ablation_')]
        for csv_file in csv_files:
            # ablation_lstm_Nylon.csv -> lstm, Nylon
            parts = csv_file.replace('ablation_', '').replace('.csv', '').split('_')
            if len(parts) >= 2:
                model_type = parts[0]
                plastic_type = '_'.join(parts[1:])
                
                csv_path = os.path.join(TABLES_DIR, csv_file)
                json_path = os.path.join(LOGS_DIR, f'ablation_log_{model_type}_{plastic_type}.json')
                
                try:
                    df = pd.read_csv(csv_path)
                    
                    # JSON log varsa yükle
                    json_data = None
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                    
                    results[f"{model_type}_{plastic_type}"] = {
                        'model_type': model_type,
                        'plastic_type': plastic_type,
                        'csv_df': df,
                        'json_data': json_data,
                        'csv_path': csv_path,
                        'json_path': json_path
                    }
                    print(f"✓ Yüklendi: {model_type.upper()} - {plastic_type} ({len(df)} kombinasyon)")
                except Exception as e:
                    print(f"✗ Hata: {csv_file} - {e}")
    
    return results


def get_best_combination_training_data(json_data, best_combo_id):
    """En iyi kombinasyonun training verilerini al"""
    if not json_data or 'all_combinations' not in json_data:
        return None
    
    for combo in json_data['all_combinations']:
        if combo.get('combination_id') == best_combo_id:
            return combo
    return None


def plot_training_curves_for_best(result_data, save_dir):
    """En iyi kombinasyon için training curves çiz"""
    model_type = result_data['model_type']
    plastic_type = result_data['plastic_type']
    df = result_data['csv_df']
    json_data = result_data['json_data']
    
    # En iyi kombinasyonu bul (val_r2'ye göre)
    best_row = df.loc[df['val_r2'].idxmax()]
    best_combo_id = int(best_row['combination_id'])
    best_val_r2 = best_row['val_r2']
    best_test_r2 = best_row['test_r2']
    best_epoch = int(best_row['best_epoch'])
    
    print(f"\n📊 {model_type.upper()} - {plastic_type}:")
    print(f"   En İyi Kombinasyon ID: {best_combo_id}")
    print(f"   Val R²: {best_val_r2:.6f}")
    print(f"   Test R²: {best_test_r2:.6f}")
    print(f"   Best Epoch: {best_epoch}")
    
    # Parametreleri göster
    param_cols = [c for c in df.columns if c not in ['combination_id', 'test_r2', 'test_mae', 
                                                       'test_rmse', 'val_r2', 'best_epoch', 
                                                       'total_epochs', 'is_best', 'final_train_loss',
                                                       'final_val_loss', 'train_val_gap', 'overfitting_ratio',
                                                       'best_epoch_gap', 'best_epoch_overfitting_ratio', 'val_r2_trend']]
    print(f"   Parametreler:")
    for col in param_cols:
        if pd.notna(best_row[col]):
            print(f"      {col}: {best_row[col]}")
    
    # Training verilerini al
    combo_data = get_best_combination_training_data(json_data, best_combo_id)
    
    if combo_data and 'epochs' in combo_data:
        epochs_data = combo_data['epochs']
        train_losses = [e['train_loss'] for e in epochs_data]
        val_losses = [e['val_loss'] for e in epochs_data]
        val_r2_list = [e.get('val_r2', 0) for e in epochs_data]
        epochs_list = list(range(1, len(train_losses) + 1))
        
        # 2x2 subplot oluştur
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Curves: {model_type.upper()} - {plastic_type}\n'
                    f'En İyi Kombinasyon (ID: {best_combo_id}) | Val R²: {best_val_r2:.4f} | Test R²: {best_test_r2:.4f}', 
                    fontsize=14, fontweight='bold')
        
        # 1. Train/Val Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs_list, train_losses, label='Train Loss', color='#2E86AB', linewidth=2)
        ax1.plot(epochs_list, val_losses, label='Validation Loss', color='#E94F37', linewidth=2)
        ax1.axvline(best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch: {best_epoch}')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Val R² Progress
        ax2 = axes[0, 1]
        ax2.plot(epochs_list, val_r2_list, label='Validation R²', color='#28A745', linewidth=2)
        ax2.axvline(best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch: {best_epoch}')
        ax2.axhline(best_val_r2, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Validation R²', fontsize=11)
        ax2.set_title('Validation R² Progress', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Loss Gap (Overfitting analizi)
        ax3 = axes[1, 0]
        loss_gap = [t - v for t, v in zip(train_losses, val_losses)]
        colors = ['red' if g < 0 else 'green' for g in loss_gap]
        ax3.bar(epochs_list, loss_gap, color=colors, alpha=0.7, width=0.8)
        ax3.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3.axvline(best_epoch, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch: {best_epoch}')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Train - Val Loss Gap', fontsize=11)
        ax3.set_title('Overfitting Analysis (Negatif = Overfitting)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Log Scale Loss
        ax4 = axes[1, 1]
        ax4.semilogy(epochs_list, train_losses, label='Train Loss', color='#2E86AB', linewidth=2)
        ax4.semilogy(epochs_list, val_losses, label='Validation Loss', color='#E94F37', linewidth=2)
        ax4.axvline(best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch: {best_epoch}')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Loss (Log Scale)', fontsize=11)
        ax4.set_title('Loss (Log Scale)', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        # Kaydet
        fig_path = os.path.join(save_dir, f'best_training_curves_{model_type}_{plastic_type}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Training curves kaydedildi: {fig_path}")
        
        return {
            'model_type': model_type,
            'plastic_type': plastic_type,
            'combo_id': best_combo_id,
            'val_r2': best_val_r2,
            'test_r2': best_test_r2,
            'best_epoch': best_epoch,
            'total_epochs': len(epochs_list),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'params': {col: best_row[col] for col in param_cols if pd.notna(best_row[col])}
        }
    else:
        print(f"   ⚠ Training verisi bulunamadı (JSON log eksik)")
        return {
            'model_type': model_type,
            'plastic_type': plastic_type,
            'combo_id': best_combo_id,
            'val_r2': best_val_r2,
            'test_r2': best_test_r2,
            'best_epoch': best_epoch,
            'params': {col: best_row[col] for col in param_cols if pd.notna(best_row[col])}
        }


def plot_model_comparison(all_best_results, save_dir):
    """Tüm modelleri karşılaştır"""
    if not all_best_results:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Karşılaştırması - Tamamlanmış Ablation Sonuçları', fontsize=14, fontweight='bold')
    
    # Veri hazırla
    models = [f"{r['model_type'].upper()}" for r in all_best_results]
    val_r2s = [r['val_r2'] for r in all_best_results]
    test_r2s = [r['test_r2'] for r in all_best_results]
    
    x = np.arange(len(models))
    width = 0.35
    
    # 1. Val vs Test R²
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, val_r2s, width, label='Validation R²', color='#2E86AB', edgecolor='black')
    bars2 = ax1.bar(x + width/2, test_r2s, width, label='Test R²', color='#28A745', edgecolor='black')
    
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('Validation vs Test R²', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Değerleri üzerine yaz
    for bar, val in zip(bars1, val_r2s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, test_r2s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Best Epoch karşılaştırması
    ax2 = axes[1]
    best_epochs = [r.get('best_epoch', 0) for r in all_best_results]
    total_epochs = [r.get('total_epochs', 50) for r in all_best_results]
    
    bars3 = ax2.bar(x, best_epochs, width*2, label='Best Epoch', color='#E94F37', edgecolor='black')
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('Epoch', fontsize=11)
    ax2.set_title('Best Epoch (Early Stopping)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, best_epochs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, 'model_comparison_summary.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Model karşılaştırma grafiği kaydedildi: {fig_path}")


def generate_text_report(all_best_results, save_dir):
    """Detaylı text raporu oluştur"""
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ABLATION STUDY SONUÇ RAPORU")
    report_lines.append(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Özet
    report_lines.append("📊 ÖZET")
    report_lines.append("-"*80)
    report_lines.append(f"Tamamlanmış Model Sayısı: {len(all_best_results)}")
    
    if all_best_results:
        best_overall = max(all_best_results, key=lambda x: x['test_r2'])
        report_lines.append(f"En İyi Model: {best_overall['model_type'].upper()}")
        report_lines.append(f"En İyi Test R²: {best_overall['test_r2']:.6f}")
        report_lines.append(f"En İyi Val R²: {best_overall['val_r2']:.6f}")
    report_lines.append("")
    
    # Her model için detay
    report_lines.append("📋 MODEL DETAYLARI")
    report_lines.append("-"*80)
    
    for result in sorted(all_best_results, key=lambda x: x['test_r2'], reverse=True):
        report_lines.append(f"\n🏆 {result['model_type'].upper()} - {result['plastic_type']}")
        report_lines.append(f"   Kombinasyon ID: {result['combo_id']}")
        report_lines.append(f"   Validation R²: {result['val_r2']:.6f}")
        report_lines.append(f"   Test R²: {result['test_r2']:.6f}")
        report_lines.append(f"   Best Epoch: {result.get('best_epoch', 'N/A')}")
        report_lines.append(f"   Total Epochs: {result.get('total_epochs', 'N/A')}")
        
        if 'final_train_loss' in result:
            report_lines.append(f"   Final Train Loss: {result['final_train_loss']:.6f}")
            report_lines.append(f"   Final Val Loss: {result['final_val_loss']:.6f}")
        
        report_lines.append(f"   Parametreler:")
        for key, value in result.get('params', {}).items():
            report_lines.append(f"      {key}: {value}")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("RAPOR SONU")
    report_lines.append("="*80)
    
    # Kaydet
    report_text = "\n".join(report_lines)
    report_path = os.path.join(save_dir, 'ablation_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ Text raporu kaydedildi: {report_path}")
    print("\n" + report_text)


def main():
    """Ana fonksiyon"""
    print("\n📂 Tamamlanmış ablation sonuçları yükleniyor...")
    results = load_completed_results()
    
    if not results:
        print("\n⚠ Tamamlanmış ablation sonucu bulunamadı!")
        print("   Lütfen önce ablation_study_mac.py'yi çalıştırın.")
        return
    
    print(f"\n✓ Toplam {len(results)} tamamlanmış model bulundu")
    
    # Her model için en iyi kombinasyonun training curves'ünü çiz
    print("\n" + "="*80)
    print("EN İYİ KOMBİNASYONLARIN TRAINING CURVES'LERİ")
    print("="*80)
    
    all_best_results = []
    for key, result_data in results.items():
        best_result = plot_training_curves_for_best(result_data, REPORT_DIR)
        if best_result:
            all_best_results.append(best_result)
    
    # Model karşılaştırma grafiği
    if len(all_best_results) > 1:
        print("\n" + "="*80)
        print("MODEL KARŞILAŞTIRMASI")
        print("="*80)
        plot_model_comparison(all_best_results, REPORT_DIR)
    
    # Text raporu oluştur
    print("\n" + "="*80)
    print("DETAYLI RAPOR")
    print("="*80)
    generate_text_report(all_best_results, REPORT_DIR)
    
    print("\n" + "="*80)
    print("ANALİZ TAMAMLANDI!")
    print(f"Sonuçlar: {REPORT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()

