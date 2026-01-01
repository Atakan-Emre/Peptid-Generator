"""
PP için ENCDEC Final Eğitimi ve Peptid Üretimi
LSTM overfitting yaptığı için ENCDEC kullanılacak
"""

import os
import sys

# Ana script'i import etmeden önce çalışma dizinini ayarla
os.chdir(r"D:\GoogleDrive\Projeler\PeptidGenerator")

# Ablation study script'inden gerekli fonksiyonları import et
from ablation_study_win import (
    train_final_model,
    generate_peptides_with_best_model,
    plot_generated_peptides_amino_acid_heatmap,
    analyze_generated_peptides_similarity,
    plot_training_curves,
    plot_amino_acid_probability_mass_heatmap,
    create_comparison_table_and_heatmap,
    DATA_DIR,
    ABLATION_DIR as RESULTS_DIR,
    ABLATION_FIGURES_DIR as FIGURES_DIR
)

def main():
    plastic_type = "PP"
    model_type = "encdec"
    
    # ENCDEC için en iyi parametreler (ablation_encdec_PP.csv'den)
    best_params = {
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 1024,
        'lambda_score': 1.0,
        'weight_decay': 0.0001,
        'use_layernorm': True,
        'best_val_r2': 0.9424,
        'best_test_r2': 0.9415,
        'score_mean': -19.41959788664366,
        'score_std': 10.238143903952345
    }
    
    print("="*80)
    print(f"PP İÇİN ENCDEC FINAL EĞİTİMİ (250 EPOCH)")
    print("="*80)
    print(f"\nParametreler:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    print()
    
    # ADIM 1: Final eğitim (250 epoch)
    print(f"\n{'='*80}")
    print(f"ADIM 1: FINAL EĞİTİM - ENCDEC (250 EPOCH)")
    print(f"{'='*80}")
    
    metrics, train_losses, val_losses = train_final_model(
        model_type=model_type,
        plastic_type=plastic_type,
        best_params=best_params,
        data_dir=DATA_DIR,
        final_epochs=250
    )
    
    test_r2 = metrics.get('test_r2', 0)
    test_mae = metrics.get('test_mae', 0)
    test_rmse = metrics.get('test_rmse', 0)
    print(f"\n✓ Final Test R²: {test_r2:.4f}")
    print(f"✓ Final Test MAE: {test_mae:.4f}")
    print(f"✓ Final Test RMSE: {test_rmse:.4f}")
    
    # Training curves grafiği
    print(f"\n{'='*80}")
    print(f"ADIM 1.5: TRAINING CURVES GRAFİĞİ")
    print(f"{'='*80}")
    
    try:
        plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            model_type=model_type,
            plastic_type=plastic_type,
            best_params=best_params
        )
        print(f"✓ Training curves kaydedildi: {FIGURES_DIR}/training_curves_{model_type}_{plastic_type}.png")
    except Exception as e:
        print(f"⚠ Training curves hatası: {e}")
    
    # ADIM 2: Yeni peptid üretimi
    print(f"\n{'='*80}")
    print(f"ADIM 2: YENİ PEPTİD ÜRETİMİ - ENCDEC")
    print(f"{'='*80}")
    
    try:
        generated = generate_peptides_with_best_model(
            plastic_type=plastic_type,
            model_type=model_type,
            best_params=best_params,
            n_samples=10,
            n_optimizations=3,
            max_iterations=2000
        )
        
        if generated:
            print(f"\n✓ Toplam {len(generated)} peptid üretildi")
            
            # En iyi 5 peptidi göster
            sorted_peptides = sorted(generated, key=lambda x: x['optimized_score'], reverse=False)
            print(f"\n🏆 EN İYİ 5 PEPTİD:")
            for i, pep in enumerate(sorted_peptides[:5], 1):
                print(f"   {i}. {pep['optimized_peptide']} (Skor: {pep['optimized_score']:.2f})")
            
            # ADIM 3: Amino acid heatmap
            print(f"\n{'='*80}")
            print(f"ADIM 3: ÜRETİLEN PEPTİDLER AMINO ACID HEATMAP")
            print(f"{'='*80}")
            
            try:
                plot_generated_peptides_amino_acid_heatmap(
                    generated_peptides_list=generated,
                    plastic_type=plastic_type,
                    model_type=model_type
                )
            except Exception as e:
                print(f"⚠ Heatmap hatası: {e}")
            
            # ADIM 4: Benzerlik analizi
            print(f"\n{'='*80}")
            print(f"ADIM 4: BENZERLİK ANALİZİ")
            print(f"{'='*80}")
            
            try:
                all_generated = {plastic_type: {model_type: generated}}
                analyze_generated_peptides_similarity(all_generated, DATA_DIR)
            except Exception as e:
                print(f"⚠ Benzerlik analizi hatası: {e}")
                
    except Exception as e:
        print(f"✗ Peptid üretim hatası: {e}")
        import traceback
        traceback.print_exc()
    
    # ADIM 5: Amino Acid Probability x Mass Heatmap
    print(f"\n{'='*80}")
    print(f"ADIM 5: AMINO ACID PROBABILITY X MASS HEATMAP")
    print(f"{'='*80}")
    
    try:
        plot_amino_acid_probability_mass_heatmap(plastic_type=plastic_type, data_dir=DATA_DIR)
        print(f"✓ AA probability x mass heatmap kaydedildi")
    except Exception as e:
        print(f"⚠ AA heatmap hatası: {e}")
    
    # Sonuç özeti
    print(f"\n{'='*80}")
    print(f"PP - ENCDEC TAMAMLANDI!")
    print(f"{'='*80}")
    print(f"\n📊 SONUÇ ÖZETİ:")
    print(f"   Model: ENCDEC")
    print(f"   Final Test R²: {metrics.get('test_r2', 'N/A'):.4f}")
    print(f"   Final Test MAE: {metrics.get('test_mae', 'N/A'):.4f}")
    print(f"   Final Test RMSE: {metrics.get('test_rmse', 'N/A'):.4f}")
    print(f"\n📁 Kaydedilen dosyalar:")
    print(f"   - Final model: {RESULTS_DIR}/final_models/encdec_PP_final.pth")
    print(f"   - Training curves: {FIGURES_DIR}/training_curves_encdec_PP.png")
    print(f"   - Üretilen peptidler: {RESULTS_DIR}/tables/generated_peptides_encdec_PP.csv")
    print(f"   - AA heatmap: {FIGURES_DIR}/generated_peptides_aa_heatmap_encdec_PP.png")

if __name__ == "__main__":
    main()
