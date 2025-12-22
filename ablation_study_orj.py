# ============================================================================
# ABLATION STUDY - WINDOWS YEREL ORTAM VERSİYONU
# ============================================================================
# Bu script Windows yerel ortam için optimize edilmiştir
# GPU: RTX 4080 Super (16GB VRAM)
# CPU: Ryzen 9700X
# RAM: 64GB
# 
# Özellikler:
# - Colab bağımlılıkları kaldırıldı
# - Windows dosya yolları kullanılıyor
# - RTX 4080 Super için optimize edilmiş batch size ve AMP
# - Windows multiprocessing için optimize edilmiş num_workers
# ============================================================================

# ============================================================================
# HÜCRE 1: Setup ve Windows Yerel Ortam Yapılandırması
# ============================================================================
print("="*80)
print("HÜCRE 1: WINDOWS YEREL ORTAM SETUP")
print("="*80)

import os

# Windows yerel ortam - Colab desteği yok
IS_COLAB = False
print("✓ Windows yerel PC ortamı tespit edildi")

# Proje kök dizini (script'in bulunduğu klasör)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

# Yerel PC için proje dizini
DRIVE_PROJECT_DIR = PROJECT_ROOT
print(f"✓ Proje kök dizini: {PROJECT_ROOT}")

RESULTS_DIR = os.path.join(DRIVE_PROJECT_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
NLP_MODELS_DIR = os.path.join(RESULTS_DIR, "nlp_models")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NLP_MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

print(f"✓ Kalıcı kayıt klasörleri oluşturuldu (Yerel: {DRIVE_PROJECT_DIR})")
print("="*80)

# ============================================================================
# HÜCRE 2: Import'lar ve Setup
# ============================================================================
print("\n" + "="*80)
print("HÜCRE 2: IMPORT'LAR VE SETUP")
print("="*80)

print("📦 Kütüphaneler yükleniyor...")
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Windows için backend ayarı (GUI gerektirmez)
import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()  # Interactive mode kapalı (Windows için)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import random
import math
import re
from typing import Dict, List, Tuple, Optional
import warnings
import itertools
import copy
warnings.filterwarnings('ignore')
print("✓ Tüm kütüphaneler yüklendi")

# Veri klasörü - önce newDate, yoksa Data klasörünü dene
DATA_DIR = os.path.join(DRIVE_PROJECT_DIR, "newDate")
if not os.path.exists(DATA_DIR):
    # Alternatif olarak Data klasörünü dene
    DATA_DIR = os.path.join(DRIVE_PROJECT_DIR, "Data")
    if not os.path.exists(DATA_DIR):
        print(f"⚠ UYARI: Veri klasörü bulunamadı!")
        print(f"   Denenen yollar:")
        print(f"   - {os.path.join(DRIVE_PROJECT_DIR, 'newDate')}")
        print(f"   - {os.path.join(DRIVE_PROJECT_DIR, 'Data')}")
        raise FileNotFoundError(f"Veri klasörü bulunamadı! Lütfen 'newDate' veya 'Data' klasörünün proje dizininde olduğundan emin olun.")
    else:
        print(f"✓ Veri klasörü bulundu (Data): {DATA_DIR}")
else:
    print(f"✓ Veri klasörü bulundu (newDate): {DATA_DIR}")

# CSV dosyası kontrolü
test_csv = os.path.join(DATA_DIR, "PET.csv")
if os.path.exists(test_csv):
    print(f"✓ CSV dosyaları doğrulandı: {DATA_DIR}")
else:
    print(f"⚠ UYARI: PET.csv bulunamadı: {DATA_DIR}")
    print(f"   Mevcut dosyalar: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else 'Klasör yok'}")

# Ablation study için özel klasörler
# DRIVE_PROJECT_DIR HÜCRE 1'de tanımlanmış olmalı
if 'DRIVE_PROJECT_DIR' not in globals():
    DRIVE_PROJECT_DIR = PROJECT_ROOT
    print(f"⚠ DRIVE_PROJECT_DIR tanımlı değildi, proje kök dizini kullanılıyor: {DRIVE_PROJECT_DIR}")

ABLATION_DIR = os.path.join(DRIVE_PROJECT_DIR, "ablation_results")
ABLATION_TABLES_DIR = os.path.join(ABLATION_DIR, "tables")
ABLATION_FIGURES_DIR = os.path.join(ABLATION_DIR, "figures")
FINAL_MODELS_DIR = os.path.join(ABLATION_DIR, "final_models")
ABLATION_LOGS_DIR = os.path.join(ABLATION_DIR, "logs")  # Detaylı loglar için
CHECKPOINT_DIR = os.path.join(ABLATION_DIR, "checkpoints")  # Checkpoint'ler için

os.makedirs(ABLATION_TABLES_DIR, exist_ok=True)
os.makedirs(ABLATION_FIGURES_DIR, exist_ok=True)
os.makedirs(FINAL_MODELS_DIR, exist_ok=True)
os.makedirs(ABLATION_LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# DataLoader için num_workers ayarı (Windows için optimize)
# Windows'ta multiprocessing sorunları nedeniyle 0 kullanıyoruz (tek thread ama güvenli)
# Batch size artışı ile performans kaybı telafi edilecek
NUM_WORKERS = 0  # Windows'ta multiprocessing sorunlarını önlemek için 0
PREFETCH_FACTOR = 2  # num_workers=0 olduğunda kullanılmaz ama tanımlı tutuyoruz
PIN_MEMORY = True  # GPU'ya veri transferini hızlandırır

print(f"\n⚙️  PERFORMANS AYARLARI (Windows Optimize):")
print(f"   NUM_WORKERS: {NUM_WORKERS} (Windows multiprocessing sorunlarını önlemek için)")
print(f"   PIN_MEMORY: {PIN_MEMORY} (GPU transfer hızlandırma)")
print(f"   Batch size artırıldı (performans telafisi)")

# GPU kontrolü ve optimizasyon
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Using device: {device}")

# GPU optimizasyon ayarları
if torch.cuda.is_available():
    # CUDA optimizasyonları
    torch.backends.cudnn.benchmark = True  # Maksimum performans için
    torch.backends.cudnn.deterministic = False  # Hız için deterministic=False
    torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 (RTX 4080 Super için)
    torch.backends.cudnn.allow_tf32 = True  # TensorFloat-32
    
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   GPU: {gpu_name}")
    print(f"   GPU Memory: {gpu_memory:.2f} GB")
    
    # RTX 4080 Super optimizasyonları
    if '4080' in gpu_name or 'RTX' in gpu_name:
        print("   ✓ RTX 4080 Super GPU tespit edildi - Optimizasyonlar aktif:")
        print("     - Mixed Precision Training (AMP) aktif")
        print("     - Optimize edilmiş batch size aralığı (yüksek batch size)")
        print("     - Windows için optimize edilmiş num_workers")
        
        # Mixed Precision için scaler (RTX 4080 Super için önerilir)
        USE_AMP = True  # Automatic Mixed Precision
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            print("     - GradScaler hazır")
        except ImportError:
            USE_AMP = False
            scaler = None
            print("     ⚠ AMP mevcut değil, normal precision kullanılacak")
    else:
        # Diğer GPU'lar için de AMP kullanabiliriz
        USE_AMP = True
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            print(f"   ✓ GPU tespit edildi - Mixed Precision Training aktif")
        except ImportError:
            USE_AMP = False
            scaler = None
            print("   ⚠ AMP mevcut değil, normal precision kullanılacak")
else:
    USE_AMP = False
    scaler = None
    print("   ⚠ CPU kullanılıyor (GPU bulunamadı)")

print(f"\n📁 Veri klasörü: {DATA_DIR}")
print(f"📁 Ablation sonuçları: {ABLATION_DIR}")
print(f"📁 Log klasörü: {ABLATION_LOGS_DIR}")
print("="*80)

# ============================================================================
# HÜCRE 3: Utility Fonksiyonlar
# ============================================================================
print("\n" + "="*80)
print("HÜCRE 3: UTILITY FONKSİYONLAR")
print("="*80)

AMINO_ACIDS = "ADEFGHIKLMNQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Amino acid masses (in Daltons)
AA_MASSES = {
    'A': 89, 'D': 133, 'E': 147, 'F': 165, 'G': 75,
    'H': 155, 'I': 131, 'K': 146, 'L': 131, 'M': 149,
    'N': 132, 'Q': 146, 'R': 174, 'S': 105, 'T': 119,
    'V': 117, 'W': 204, 'Y': 181, 'C': 121, 'P': 115
}

print(f"✓ Amino asitler tanımlandı: {len(AMINO_ACIDS)} amino asit")
print(f"✓ Amino asit kütleleri tanımlandı")
print("="*80)

def _onehotencode(s, vocab=None):
    """One-hot encode peptide sequence."""
    if not vocab:
        vocab = list(AMINO_ACIDS)
    to_one_hot = {}
    for i, a in enumerate(vocab):
        v = np.zeros(len(vocab))
        v[i] = 1
        to_one_hot[a] = v
    result = []
    for l in s:
        result.append(to_one_hot[l])
    result = np.array(result)
    return np.reshape(result, (1, result.shape[0], result.shape[1])), to_one_hot, vocab

def one_hot_encode_sequences(seqs: List[str]) -> np.ndarray:
    """Vectorize sequences to (N, 12, 18)."""
    encoded = []
    for seq in seqs:
        x, _, _ = _onehotencode(seq, vocab=list(AMINO_ACIDS))
        encoded.append(x[0])
    return np.stack(encoded, axis=0)

def seed_everything(seed=42):
    """Seed all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Model Mimarileri
# ============================================================================

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.head(out).squeeze(-1)
        return out

class CNNRegressor(nn.Module):
    def __init__(self, input_channels=18):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.conv(x)
        h = self.pool(h)
        out = self.head(h).squeeze(-1)
        return out

class LSTMVAE(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=256, latent_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec_init = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.score_head = nn.Sequential(nn.Tanh(), nn.Linear(latent_dim, 1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        h_last = enc_out[:, -1, :]
        mu = self.mu(h_last)
        logvar = self.logvar(h_last)
        z = self.reparameterize(mu, logvar)

        batch_size, seq_len, _ = x.shape
        dec_input = torch.zeros(batch_size, seq_len, x.size(-1), device=x.device)
        h0 = torch.tanh(self.dec_init(z)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        dec_out, _ = self.decoder(dec_input, (h0, c0))
        recon_logits = self.out_proj(dec_out)
        score_pred = self.score_head(z).squeeze(-1)
        return recon_logits, mu, logvar, score_pred

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dec_init = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.score_head = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        h_last = enc_out[:, -1, :]
        score_pred = self.score_head(h_last).squeeze(-1)
        
        batch_size, seq_len, _ = x.shape
        dec_input = torch.zeros(batch_size, seq_len, x.size(-1), device=x.device)
        h0 = torch.tanh(self.dec_init(h_last)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        dec_out, _ = self.decoder(dec_input, (h0, c0))
        recon_logits = self.out_proj(dec_out)
        return recon_logits, score_pred

print("✓ Model mimarileri tanımlandı: LSTM, CNN, LSTM-VAE, Encoder-Decoder")
print("="*80)

# ============================================================================
# HÜCRE 5: Dataset
# ============================================================================
print("\n" + "="*80)
print("HÜCRE 5: DATASET")
print("="*80)

class PeptideDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.class_targets = torch.tensor(np.argmax(X, axis=2), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.class_targets[idx]

# ============================================================================
# Ablation Study Fonksiyonu
# ============================================================================

def run_ablation_study(model_type, plastic_type, data_dir=None, 
                       ablation_epochs=50, seed=42):
    """
    50 epoch ile hiperparametre taraması yapar.
    En iyi kombinasyonu bulur ve döndürür.
    """
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Detaylı log dosyası oluştur
    log_file_path = os.path.join(ABLATION_LOGS_DIR, f'ablation_log_{model_type}_{plastic_type}.txt')
    json_log_path = os.path.join(ABLATION_LOGS_DIR, f'ablation_log_{model_type}_{plastic_type}.json')
    output_path = os.path.join(ABLATION_TABLES_DIR, f'ablation_{model_type}_{plastic_type}.csv')
    
    # Eğer sonuç dosyası zaten varsa, bu model tamamlanmış demektir - atla
    if os.path.exists(output_path):
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY: {model_type.upper()} - {plastic_type}")
        print(f"{'='*80}")
        print(f"✓ Bu model zaten tamamlanmış! Sonuç dosyası mevcut: {output_path}")
        print(f"  Sonuçları yükleniyor...")
        
        # Mevcut sonuçları yükle
        results_df = pd.read_csv(output_path)
        
        # En iyi parametreleri bul - CSV'deki TÜM parametreleri oku
        best_row = results_df.loc[results_df['val_r2'].idxmax()]
        
        # Temel parametreler (her zaman gerekli)
        best_params = {
            'hidden_dim': int(best_row.get('hidden_dim', 128)),
            'learning_rate': float(best_row.get('learning_rate', 0.001)),
            'batch_size': int(best_row.get('batch_size', 512)),
            'best_val_r2': float(best_row['val_r2']),
            'best_test_r2': float(best_row['test_r2'])
        }
        
        # Model tipine göre gerekli parametreleri ekle
        # num_layers ve dropout (lstm, lstm_vae, encdec için)
        if 'num_layers' in best_row and pd.notna(best_row['num_layers']):
            best_params['num_layers'] = int(best_row['num_layers'])
        else:
            best_params['num_layers'] = 2  # Varsayılan
        
        if 'dropout' in best_row and pd.notna(best_row['dropout']):
            best_params['dropout'] = float(best_row['dropout'])
        else:
            best_params['dropout'] = 0.1  # Varsayılan
        
        # lstm_vae için özel parametreler
        if model_type == 'lstm_vae':
            if 'latent_dim' in best_row and pd.notna(best_row['latent_dim']):
                best_params['latent_dim'] = int(best_row['latent_dim'])
            else:
                best_params['latent_dim'] = 64  # Varsayılan
            
            if 'beta_kl' in best_row and pd.notna(best_row['beta_kl']):
                best_params['beta_kl'] = float(best_row['beta_kl'])
            else:
                best_params['beta_kl'] = 0.1  # Varsayılan
            
            if 'gamma_score' in best_row and pd.notna(best_row['gamma_score']):
                best_params['gamma_score'] = float(best_row['gamma_score'])
            else:
                best_params['gamma_score'] = 1.0  # Varsayılan
        
        # encdec için özel parametreler
        if model_type == 'encdec':
            if 'lambda_score' in best_row and pd.notna(best_row['lambda_score']):
                best_params['lambda_score'] = float(best_row['lambda_score'])
            else:
                best_params['lambda_score'] = 0.7  # Varsayılan
        
        # Score mean/std'yi veriyi tekrar yükleyerek hesapla (güvenilir yöntem)
        csv_path = os.path.join(data_dir, f"{plastic_type}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['Sequence', 'Score'])
            df['Sequence'] = df['Sequence'].str.strip().str.upper()
            
            # Aynı seed ile veri böl (tutarlılık için)
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n = len(df)
            n_train = int(0.8 * n)
            train_df = df.iloc[:n_train]
            y_train = train_df['Score'].values
            
            # Normalize değerlerini hesapla
            score_mean = np.mean(y_train)
            score_std = np.std(y_train) + 1e-8
            best_params['score_mean'] = float(score_mean)
            best_params['score_std'] = float(score_std)
        else:
            # Fallback: log dosyasından oku
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                    score_mean_match = re.search(r'score_mean[:\s=]+([\d.]+)', log_content)
                    score_std_match = re.search(r'score_std[:\s=]+([\d.]+)', log_content)
                    if score_mean_match:
                        best_params['score_mean'] = float(score_mean_match.group(1))
                    if score_std_match:
                        best_params['score_std'] = float(score_std_match.group(1))
        
        # Eğer hala yoksa varsayılan değerler (hata önleme)
        if 'score_mean' not in best_params:
            print(f"  ⚠ score_mean bulunamadı, varsayılan değerler kullanılıyor")
            best_params['score_mean'] = 0.0
            best_params['score_std'] = 1.0
        
        print(f"  ✓ En iyi Val R²: {best_params['best_val_r2']:.6f}")
        print(f"  ✓ En iyi Test R²: {best_params['best_test_r2']:.6f}")
        print(f"  ✓ Score mean: {best_params['score_mean']:.4f}, std: {best_params['score_std']:.4f}")
        print(f"\n  📋 CSV'den yüklenen parametreler (Final eğitimde kullanılacak):")
        for key in ['hidden_dim', 'num_layers', 'dropout', 'learning_rate', 'batch_size', 
                    'latent_dim', 'beta_kl', 'gamma_score', 'lambda_score']:
            if key in best_params:
                print(f"     {key}: {best_params[key]}")
        print(f"  ✓ Model atlandı, devam ediliyor...\n")
        
        return best_params, results_df
    
    # Log dosyasını aç (append mode - temp silinmeyecek)
    log_file = open(log_file_path, 'a', encoding='utf-8')
    all_logs = []  # JSON için tüm logları topla
    
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY: {model_type.upper()} - {plastic_type}")
    print(f"{'='*80}")
    print(f"Veri klasörü: {data_dir}")
    
    # Veri yükle
    csv_path = os.path.join(data_dir, f"{plastic_type}.csv")
    
    # Dosya varlığını kontrol et
    if not os.path.exists(csv_path):
        print(f"✗ CSV dosyası bulunamadı: {csv_path}")
        if os.path.exists(data_dir):
            print(f"   Mevcut dosyalar: {os.listdir(data_dir)}")
        else:
            print(f"   Veri klasörü mevcut değil: {data_dir}")
        raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")
    
    print(f"✓ CSV dosyası bulundu: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Sequence', 'Score'])
    df['Sequence'] = df['Sequence'].str.strip().str.upper()
    
    # Veri böl (80/10/10)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    # One-hot encode
    X_train = one_hot_encode_sequences(train_df['Sequence'].tolist())
    X_val = one_hot_encode_sequences(val_df['Sequence'].tolist())
    X_test = one_hot_encode_sequences(test_df['Sequence'].tolist())
    
    y_train = train_df['Score'].values
    y_val = val_df['Score'].values
    y_test = test_df['Score'].values
    
    # Normalize
    score_mean = np.mean(y_train)
    score_std = np.std(y_train) + 1e-8
    
    y_train_norm = (y_train - score_mean) / score_std
    y_val_norm = (y_val - score_mean) / score_std
    y_test_norm = (y_test - score_mean) / score_std
    
    # DataLoader
    train_ds = PeptideDataset(X_train, y_train_norm)
    val_ds = PeptideDataset(X_val, y_val_norm)
    test_ds = PeptideDataset(X_test, y_test_norm)
    
    # Hiperparametre grid'leri (optimize edilmiş - daha az kombinasyon)
    # Her model için ~50-100 kombinasyon hedefleniyor
    if model_type == 'lstm':
        param_grid = {
            'hidden_dim': [256, 512],  # 2 değer
            'num_layers': [2, 3],  # 2 değer
            'dropout': [0.0, 0.1],  # 2 değer
            'learning_rate': [1e-4, 1e-3],  # 2 değer
            'batch_size': [512, 1024, 2048, 4096]  # RTX 4080 Super için maksimum performans (16GB VRAM)
        }
        # Toplam: 2^5 = 32 kombinasyon
    elif model_type == 'cnn':
        param_grid = {
            'learning_rate': [1e-4, 1e-3, 5e-3],  # 3 değer
            'batch_size': [512, 1024, 2048, 4096]  # RTX 4080 Super için maksimum performans (16GB VRAM)
        }
        # Toplam: 3^2 = 9 kombinasyon
    elif model_type == 'lstm_vae':
        # LSTM-VAE için bazı parametreleri sabitleyerek kombinasyon sayısını azaltıyoruz
        param_grid = {
            'hidden_dim': [256, 512],  # 2 değer
            'num_layers': [2, 3],  # 2 değer
            'dropout': [0.0, 0.1],  # 2 değer
            'latent_dim': [64],  # 1 değer (sabitlendi)
            'learning_rate': [1e-4, 1e-3],  # 2 değer
            'batch_size': [256, 512, 1024],  # RTX 4080 Super için optimize edildi (yüksek batch size)
            'beta_kl': [0.1, 1.0],  # 2 değer
            'gamma_score': [1.0, 2.0]  # 2 değer
        }
        # Toplam: 2^7 = 128 kombinasyon
    elif model_type == 'encdec':
        param_grid = {
            'hidden_dim': [256, 512],  # 2 değer
            'num_layers': [2, 3],  # 2 değer
            'dropout': [0.0, 0.1],  # 2 değer
            'learning_rate': [1e-4, 1e-3],  # 2 değer
            'batch_size': [256, 512, 1024],  # RTX 4080 Super için optimize edildi (yüksek batch size)
            'lambda_score': [0.7, 1.0]  # 2 değer
        }
        # Toplam: 2^6 = 64 kombinasyon
    
    # Grid search
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))
    
    print(f"\n📊 Toplam {len(all_combinations)} kombinasyon test edilecek...")
    
    # Ana kombinasyon progress bar
    combo_pbar = tqdm(total=len(all_combinations), 
                     desc="Kombinasyonlar",
                     ncols=100,
                     position=0,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    # Log başlığı
    log_file.write("\n" + "="*80 + "\n")
    log_file.write(f"ABLATION STUDY: {model_type.upper()} - {plastic_type}\n")
    log_file.write(f"Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Toplam Kombinasyon: {len(all_combinations)}\n")
    log_file.write(f"Ablation Epochs: {ablation_epochs}\n")
    log_file.write(f"Seed: {seed}\n")
    log_file.write("="*80 + "\n\n")
    log_file.flush()
    
    results = []
    best_r2 = -float('inf')
    best_params = None
    
    for idx, combo in enumerate(all_combinations):
        # Kombinasyon progress bar güncelle
        combo_pbar.update(1)
        combo_pbar.set_description(f"Kombinasyon {idx+1}/{len(all_combinations)}")
        
        # GPU memory temizleme (her kombinasyon başında)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        params = dict(zip(keys, combo))
        
        # Default değerler
        if model_type == 'lstm':
            hidden_dim = params['hidden_dim']
            num_layers = params['num_layers']
            dropout = params['dropout']
            lr = params['learning_rate']
            batch_size = params['batch_size']
            latent_dim = 64  # Not used
            beta_kl = 0.1  # Not used
            gamma_score = 1.0  # Not used
            lambda_score = 0.7  # Not used
        elif model_type == 'cnn':
            hidden_dim = 256  # Not used
            num_layers = 2  # Not used
            dropout = 0.1  # Not used
            lr = params['learning_rate']
            batch_size = params['batch_size']
            latent_dim = 64  # Not used
            beta_kl = 0.1  # Not used
            gamma_score = 1.0  # Not used
            lambda_score = 0.7  # Not used
        elif model_type == 'lstm_vae':
            hidden_dim = params['hidden_dim']
            num_layers = params['num_layers']
            dropout = params['dropout']
            latent_dim = params['latent_dim']
            lr = params['learning_rate']
            batch_size = params['batch_size']
            beta_kl = params['beta_kl']
            gamma_score = params['gamma_score']
            lambda_score = 0.7  # Not used
        elif model_type == 'encdec':
            hidden_dim = params['hidden_dim']
            num_layers = params['num_layers']
            dropout = params['dropout']
            lr = params['learning_rate']
            batch_size = params['batch_size']
            latent_dim = 64  # Not used
            beta_kl = 0.1  # Not used
            gamma_score = 1.0  # Not used
            lambda_score = params['lambda_score']
        
        seed_everything(seed)
        
        # Windows için DataLoader ayarları (multiprocessing sorunlarını önlemek için)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        
        # Model oluştur
        if model_type == 'lstm':
            model = LSTMRegressor(input_dim=18, hidden_dim=hidden_dim, 
                                 num_layers=num_layers, dropout=dropout)
        elif model_type == 'cnn':
            model = CNNRegressor(input_channels=18)
        elif model_type == 'lstm_vae':
            model = LSTMVAE(input_dim=18, hidden_dim=hidden_dim, latent_dim=latent_dim,
                           num_layers=num_layers, dropout=dropout)
        elif model_type == 'encdec':
            model = LSTMEncoderDecoder(input_dim=18, hidden_dim=hidden_dim,
                                      num_layers=num_layers, dropout=dropout)
        
        model = model.to(device)
        
        # Loss ve optimizer
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        # Mixed Precision Training için scaler (RTX 4080 Super optimizasyonu)
        if USE_AMP and scaler is not None:
            from torch.cuda.amp import autocast
        else:
            autocast = None
        
        # Checkpoint dosyası yolu
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 
                                     f'ablation_{model_type}_{plastic_type}_combo_{idx+1}.pth')
        
        # Checkpoint'ten devam et (varsa)
        start_epoch = 0
        best_val_r2 = -float('inf')
        best_state = None
        patience_ctr = 0
        patience = 10  # Ablation study için daha kısa patience
        best_epoch = 0
        
        train_losses_epoch = []
        val_losses_epoch = []
        val_r2_epoch = []  # Her epoch için Val R²
        
        if os.path.exists(checkpoint_path):
            print(f"  🔄 Checkpoint bulundu! Kaldığı yerden devam ediliyor: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
            best_epoch = checkpoint.get('best_epoch', 0)
            patience_ctr = checkpoint.get('patience_ctr', 0)
            if 'best_state' in checkpoint:
                best_state = checkpoint['best_state']
            train_losses_epoch = checkpoint.get('train_losses_epoch', [])
            val_losses_epoch = checkpoint.get('val_losses_epoch', [])
            val_r2_epoch = checkpoint.get('val_r2_epoch', [])
            print(f"  ✓ Checkpoint yüklendi: Epoch {start_epoch}/{ablation_epochs}, Best Val R²: {best_val_r2:.6f}")
        
        # Kombinasyon başlangıç logu
        combo_log = {
            'combination_id': idx + 1,
            'total_combinations': len(all_combinations),
            'parameters': params.copy(),
            'epochs': []
        }
        
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"KOMBİNASYON {idx + 1}/{len(all_combinations)}\n")
        log_file.write(f"Parametreler: {params}\n")
        if start_epoch > 0:
            log_file.write(f"🔄 Checkpoint'ten devam: Epoch {start_epoch}/{ablation_epochs}\n")
        log_file.write(f"{'='*80}\n")
        log_file.flush()
        
        # Canlı progress bar için tqdm
        epoch_pbar = tqdm(range(start_epoch, ablation_epochs), 
                         desc=f"Komb {idx+1}/{len(all_combinations)} | Epoch",
                         leave=False,
                         initial=start_epoch,
                         total=ablation_epochs,
                         ncols=100)
        
        for epoch in epoch_pbar:
            model.train()
            train_total = 0.0
            train_count = 0
            
            # Train loop için progress bar
            train_pbar = tqdm(train_loader, 
                             desc=f"  Train", 
                             leave=False,
                             ncols=80)
            
            for xb, yb, cls in train_pbar:
                xb, yb, cls = xb.to(device), yb.to(device), cls.to(device)
                optimizer.zero_grad()
                
                # Mixed Precision Training (RTX 4080 Super optimizasyonu)
                if USE_AMP and scaler is not None and autocast is not None:
                    with autocast():
                        if model_type in ('lstm', 'cnn'):
                            pred = model(xb)
                            loss = mse_loss(pred, yb)
                        elif model_type == 'lstm_vae':
                            recon_logits, mu, logvar, score_pred = model(xb)
                            recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                            score_loss = mse_loss(score_pred, yb)
                            loss = recon_loss + beta_kl * kl + gamma_score * score_loss
                        else:  # encdec
                            recon_logits, score_pred = model(xb)
                            recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                            score_loss = mse_loss(score_pred, yb)
                            loss = recon_loss + lambda_score * score_loss
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Normal precision training
                    if model_type in ('lstm', 'cnn'):
                        pred = model(xb)
                        loss = mse_loss(pred, yb)
                    elif model_type == 'lstm_vae':
                        recon_logits, mu, logvar, score_pred = model(xb)
                        recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        score_loss = mse_loss(score_pred, yb)
                        loss = recon_loss + beta_kl * kl + gamma_score * score_loss
                    else:  # encdec
                        recon_logits, score_pred = model(xb)
                        recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                        score_loss = mse_loss(score_pred, yb)
                        loss = recon_loss + lambda_score * score_loss
                    
                    loss.backward()
                    optimizer.step()
                train_total += loss.item() * xb.size(0)
                train_count += xb.size(0)
                
                # Canlı loss gösterimi
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_pbar.close()
            avg_train_loss = train_total / train_count if train_count > 0 else 0.0
            
            # Validation
            model.eval()
            val_preds = []
            val_total = 0.0
            val_count = 0
            
            val_pbar = tqdm(val_loader, 
                           desc=f"  Val  ", 
                           leave=False,
                           ncols=80)
            
            with torch.no_grad():
                for xb, yb, cls in val_pbar:
                    xb, yb, cls = xb.to(device), yb.to(device), cls.to(device)
                    if model_type in ('lstm', 'cnn'):
                        pred = model(xb)
                        loss = mse_loss(pred, yb)
                    elif model_type == 'lstm_vae':
                        recon_logits, mu, logvar, score_pred = model(xb)
                        pred = score_pred  # Score prediction'ı kullan
                        recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        score_loss = mse_loss(score_pred, yb)
                        loss = recon_loss + beta_kl * kl + gamma_score * score_loss
                    else:  # encdec
                        recon_logits, score_pred = model(xb)
                        pred = score_pred  # Score prediction'ı kullan
                        recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                        score_loss = mse_loss(score_pred, yb)
                        loss = recon_loss + lambda_score * score_loss
                    
                    val_preds.append(pred.cpu().numpy())
                    val_total += loss.item() * xb.size(0)
                    val_count += xb.size(0)
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            val_pbar.close()
            avg_val_loss = val_total / val_count if val_count > 0 else 0.0
            train_losses_epoch.append(avg_train_loss)
            val_losses_epoch.append(avg_val_loss)
            
            val_preds = np.concatenate(val_preds) * score_std + score_mean
            val_r2 = r2_score(y_val, val_preds)
            val_r2_epoch.append(val_r2)
            
            # Canlı epoch progress güncelleme
            epoch_pbar.set_postfix({
                'Train': f'{avg_train_loss:.4f}',
                'Val': f'{avg_val_loss:.4f}',
                'Val R²': f'{val_r2:.4f}'
            })
            
            # Her epoch için detaylı log
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': float(avg_train_loss),
                'val_loss': float(avg_val_loss),
                'val_r2': float(val_r2)
            }
            combo_log['epochs'].append(epoch_log)
            
            # Her 10 epoch'ta bir veya önemli epoch'larda log yaz
            if (epoch + 1) % 10 == 0 or epoch == 0 or val_r2 > best_val_r2:
                log_file.write(f"  Epoch {epoch + 1:3d}/{ablation_epochs}: "
                             f"Train Loss: {avg_train_loss:.6f}, "
                             f"Val Loss: {avg_val_loss:.6f}, "
                             f"Val R²: {val_r2:.6f}\n")
                log_file.flush()
                # Canlı konsol çıktısı
                print(f"\n  Epoch {epoch + 1:3d}/{ablation_epochs}: "
                      f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, Val R²: {val_r2:.6f}")
            
            # Early stopping kontrolü (val_r2'ye göre)
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_epoch = epoch + 1
                patience_ctr = 0
                # Deep copy kullanarak ağırlıkları güvenli şekilde kaydet
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    # Early stopping tetiklendi
                    if best_state:
                        model.load_state_dict(best_state)
                    break
            
            # Her epoch'ta checkpoint kaydet (kaldığı yerden devam için)
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_r2': best_val_r2,
                'best_epoch': best_epoch,
                'patience_ctr': patience_ctr,
                'train_losses_epoch': train_losses_epoch,
                'val_losses_epoch': val_losses_epoch,
                'val_r2_epoch': val_r2_epoch,
                'params': params,
                'model_type': model_type,
                'plastic_type': plastic_type,
                'combination_id': idx + 1
            }
            if best_state is not None:
                checkpoint_data['best_state'] = best_state
            
            # Her 5 epoch'ta bir checkpoint kaydet (disk I/O optimizasyonu)
            if (epoch + 1) % 5 == 0 or epoch == ablation_epochs - 1:
                torch.save(checkpoint_data, checkpoint_path)
        
        # Test (en iyi model ile)
        if best_state:
            model.load_state_dict(best_state)
        
        model.eval()
        test_preds = []
        with torch.no_grad():
            for xb, _, cls in test_loader:
                xb = xb.to(device)
                if model_type in ('lstm', 'cnn'):
                    pred = model(xb)
                elif model_type == 'lstm_vae':
                    _, _, _, pred = model(xb)
                else:
                    _, pred = model(xb)
                test_preds.append(pred.cpu().numpy())
        
        test_preds = np.concatenate(test_preds) * score_std + score_mean
        
        test_r2 = r2_score(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        
        # Kombinasyon sonuç logu
        combo_log['final_results'] = {
            'best_val_r2': float(best_val_r2),
            'test_r2': float(test_r2),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
            'best_epoch': int(best_epoch),
            'total_epochs': int(epoch + 1),
            'early_stopped': epoch + 1 < ablation_epochs
        }
        all_logs.append(combo_log)
        
        # Kombinasyon özet logu (TÜM HİPERPARAMETRELER İLE)
        log_file.write(f"\n  ✓ Kombinasyon {idx + 1} tamamlandı:\n")
        log_file.write(f"     Parametreler: {params}\n")  # Tüm hiperparametreler
        log_file.write(f"     Best Val R²: {best_val_r2:.6f} (Epoch {best_epoch})\n")
        log_file.write(f"     Test R²: {test_r2:.6f}\n")
        log_file.write(f"     Test MAE: {test_mae:.6f}\n")
        log_file.write(f"     Test RMSE: {test_rmse:.6f}\n")
        log_file.write(f"     Toplam Epoch: {epoch + 1}/{ablation_epochs}\n")
        log_file.flush()
        
        result = {
            'combination_id': idx + 1,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'val_r2': best_val_r2,
            'best_epoch': best_epoch,  # Early stopping epoch'u
            'total_epochs': epoch + 1,  # Toplam çalışan epoch sayısı
            **params
        }
        results.append(result)
        
        # KRİTİK DÜZELTME: En iyi hiperparametre seçimi VAL R²'ye göre yapılmalı (test seti sızıntısını önlemek için)
        if best_val_r2 > best_r2:
            best_r2 = best_val_r2  # Val R² kullan
            best_params = params.copy()
            best_params['score_mean'] = score_mean
            best_params['score_std'] = score_std
            best_params['best_val_r2'] = best_val_r2  # Val R²'yi de kaydet
            best_params['best_test_r2'] = test_r2  # Test R²'yi de kaydet (raporlama için)
        
        # Detaylı ilerleme gösterimi (her kombinasyonda)
        epoch_pbar.close()  # Epoch progress bar'ı kapat
        progress_pct = ((idx + 1) / len(all_combinations)) * 100
        bar_length = 50
        filled = int(bar_length * (idx + 1) / len(all_combinations))
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # GPU memory kullanımı
        if torch.cuda.is_available():
            gpu_mem_used = torch.cuda.memory_allocated(0) / 1024**3
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_mem_pct = (gpu_mem_used / gpu_mem_total) * 100
            gpu_info = f"GPU: {gpu_mem_used:.1f}/{gpu_mem_total:.1f}GB ({gpu_mem_pct:.0f}%)"
        else:
            gpu_info = "GPU: N/A"
        
        print(f"\n  [{bar}] {idx + 1}/{len(all_combinations)} ({progress_pct:.1f}%)")
        print(f"     Val R²: {best_val_r2:.6f} | Test R²: {test_r2:.6f} | "
              f"Epoch: {best_epoch}/{epoch + 1} | {gpu_info}")
        print(f"     Params: hidden={params.get('hidden_dim', 'N/A')}, "
              f"lr={params.get('learning_rate', 'N/A')}, "
              f"batch={params.get('batch_size', 'N/A')}")
        
        # Her 10 kombinasyonda bir veya son kombinasyonda detaylı özet
        if (idx + 1) % 10 == 0 or (idx + 1) == len(all_combinations):
            combo_pbar.set_postfix({
                'Best Val R²': f'{best_r2:.4f}',
                'Best Test R²': f"{best_params.get('best_test_r2', 0):.4f}" if best_params else 'N/A'
            })
            print(f"\n  ✓ {idx + 1}/{len(all_combinations)} kombinasyon tamamlandı. "
                  f"En iyi Val R²: {best_r2:.4f} (Test R²: {best_params.get('best_test_r2', 'N/A')})")
    
    # Kombinasyon progress bar'ı kapat
    combo_pbar.close()
    
    # Sonuçları DataFrame'e çevir
    results_df = pd.DataFrame(results)
    
    # En iyi kombinasyonu işaretle (VAL R²'ye göre - test seti sızıntısını önlemek için)
    results_df['is_best'] = results_df['val_r2'] == best_r2
    
    # Sıralama: Önce Val R²'ye göre (yüksekten düşüğe), sonra Test R²'ye göre
    results_df = results_df.sort_values(['val_r2', 'test_r2'], ascending=[False, False]).reset_index(drop=True)
    
    # Kaydet
    output_path = os.path.join(ABLATION_TABLES_DIR, f'ablation_{model_type}_{plastic_type}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Ablation sonuçları kaydedildi: {output_path}")
    
    # Ablation tamamlandı, checkpoint'leri temizle (disk alanı için)
    for idx in range(len(all_combinations)):
        combo_checkpoint = os.path.join(CHECKPOINT_DIR, 
                                      f'ablation_{model_type}_{plastic_type}_combo_{idx+1}.pth')
        if os.path.exists(combo_checkpoint):
            try:
                os.remove(combo_checkpoint)
            except:
                pass  # Silinemezse sorun değil
    
    # En iyi 5 kombinasyonu göster (VAL R²'ye göre sıralanmış)
    print(f"\n🏆 EN İYİ 5 KOMBİNASYON (Val R²'ye göre sıralanmış):")
    print("="*80)
    top5 = results_df.head(5)
    for i, row in top5.iterrows():
        print(f"\n{i+1}. Val R²: {row['val_r2']:.4f}, Test R²: {row['test_r2']:.4f}, "
              f"Epoch: {row['best_epoch']}/{row['total_epochs']}")
        param_str = ", ".join([f"{k}={v}" for k, v in row.items() 
                               if k not in ['combination_id', 'test_r2', 'test_mae', 'test_rmse', 
                                           'val_r2', 'best_epoch', 'total_epochs', 'is_best']])
        print(f"   Parametreler: {param_str}")
    
    print(f"\n🏆 EN İYİ KOMBİNASYON (Val R² = {best_r2:.4f}, Test R² = {best_params.get('best_test_r2', 'N/A')}):")
    for key, value in best_params.items():
        if key not in ['score_mean', 'score_std', 'best_val_r2', 'best_test_r2']:
            print(f"   {key}: {value}")
    
    # Ablation sonuçlarını görselleştir
    plot_ablation_results(results_df, model_type, plastic_type)
    
    # Final özet logu
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"ABLATION STUDY TAMAMLANDI\n")
    log_file.write(f"En İyi Val R²: {best_r2:.6f}\n")
    log_file.write(f"En İyi Test R²: {best_params.get('best_test_r2', 'N/A')}\n")
    log_file.write(f"En İyi Parametreler: {best_params}\n")
    log_file.write(f"{'='*80}\n")
    log_file.close()
    
    # JSON logunu kaydet (kalıcı - yerel diskte)
    with open(json_log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_type': model_type,
            'plastic_type': plastic_type,
            'ablation_epochs': ablation_epochs,
            'seed': seed,
            'total_combinations': len(all_combinations),
            'best_params': {k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                           for k, v in best_params.items()},
            'best_val_r2': float(best_r2),
            'best_test_r2': float(best_params.get('best_test_r2', 0)),
            'all_combinations': all_logs,
            'summary_df_path': output_path
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detaylı loglar kaydedildi:")
    print(f"   - Text log: {log_file_path}")
    print(f"   - JSON log: {json_log_path}")
    
    print("\n" + "="*80)
    print(f"✓ ABLATION STUDY TAMAMLANDI: {model_type.upper()} - {plastic_type}")
    print("="*80)

    return best_params, results_df

# ============================================================================
# Final Eğitim Fonksiyonu
# ============================================================================

def train_final_model(model_type, plastic_type, best_params, data_dir=None,
                      final_epochs=250, seed=42):
    """
    En iyi hiperparametrelerle 250 epoch final eğitim yapar.
    """
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    print(f"\n{'='*80}")
    print(f"FINAL EĞİTİM: {model_type.upper()} - {plastic_type} ({final_epochs} epoch)")
    print(f"{'='*80}")
    print(f"Veri klasörü: {data_dir}")
    
    # Veri yükle
    csv_path = os.path.join(data_dir, f"{plastic_type}.csv")
    
    # Dosya varlığını kontrol et
    if not os.path.exists(csv_path):
        print(f"✗ CSV dosyası bulunamadı: {csv_path}")
        if os.path.exists(data_dir):
            print(f"   Mevcut dosyalar: {os.listdir(data_dir)}")
        else:
            print(f"   Veri klasörü mevcut değil: {data_dir}")
        raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")
    
    print(f"✓ CSV dosyası bulundu: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Sequence', 'Score'])
    df['Sequence'] = df['Sequence'].str.strip().str.upper()
    
    # Veri böl (80/10/10)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    # One-hot encode
    X_train = one_hot_encode_sequences(train_df['Sequence'].tolist())
    X_val = one_hot_encode_sequences(val_df['Sequence'].tolist())
    X_test = one_hot_encode_sequences(test_df['Sequence'].tolist())
    
    y_train = train_df['Score'].values
    y_val = val_df['Score'].values
    y_test = test_df['Score'].values
    
    # Normalize (best_params'tan al)
    score_mean = best_params['score_mean']
    score_std = best_params['score_std']
    
    y_train_norm = (y_train - score_mean) / score_std
    y_val_norm = (y_val - score_mean) / score_std
    y_test_norm = (y_test - score_mean) / score_std
    
    # DataLoader
    train_ds = PeptideDataset(X_train, y_train_norm)
    val_ds = PeptideDataset(X_val, y_val_norm)
    test_ds = PeptideDataset(X_test, y_test_norm)
    
    # Parametreleri al (güvenli okuma - varsayılan değerlerle)
    if model_type == 'lstm':
        hidden_dim = best_params.get('hidden_dim', 128)
        num_layers = best_params.get('num_layers', 2)
        dropout = best_params.get('dropout', 0.1)
        lr = best_params.get('learning_rate', 0.001)
        batch_size = best_params.get('batch_size', 512)
        latent_dim = 64
        beta_kl = 0.1
        gamma_score = 1.0
        lambda_score = 0.7
    elif model_type == 'cnn':
        hidden_dim = 256
        num_layers = 2
        dropout = 0.1
        lr = best_params.get('learning_rate', 0.001)
        batch_size = best_params.get('batch_size', 512)
        latent_dim = 64
        beta_kl = 0.1
        gamma_score = 1.0
        lambda_score = 0.7
    elif model_type == 'lstm_vae':
        hidden_dim = best_params.get('hidden_dim', 128)
        num_layers = best_params.get('num_layers', 2)
        dropout = best_params.get('dropout', 0.1)
        latent_dim = best_params.get('latent_dim', 64)
        lr = best_params.get('learning_rate', 0.001)
        batch_size = best_params.get('batch_size', 512)
        beta_kl = best_params.get('beta_kl', 0.1)
        gamma_score = best_params.get('gamma_score', 1.0)
        lambda_score = 0.7
    elif model_type == 'encdec':
        hidden_dim = best_params.get('hidden_dim', 128)
        num_layers = best_params.get('num_layers', 2)
        dropout = best_params.get('dropout', 0.1)
        lr = best_params.get('learning_rate', 0.001)
        batch_size = best_params.get('batch_size', 512)
        latent_dim = 64
        beta_kl = 0.1
        gamma_score = 1.0
        lambda_score = best_params.get('lambda_score', 0.7)
    
    seed_everything(seed)
    
    # DEBUG: Kullanılan parametreleri göster (doğrulama için)
    print(f"\n{'='*80}")
    print(f"🔍 FINAL EĞİTİM PARAMETRELERİ (CSV'den yüklenen):")
    print(f"{'='*80}")
    print(f"   Model Tipi: {model_type.upper()}")
    print(f"   Plastik Tipi: {plastic_type}")
    if model_type == 'lstm':
        print(f"   hidden_dim: {hidden_dim}")
        print(f"   num_layers: {num_layers}")
        print(f"   dropout: {dropout}")
        print(f"   learning_rate: {lr}")
        print(f"   batch_size: {batch_size}")
    elif model_type == 'cnn':
        print(f"   learning_rate: {lr}")
        print(f"   batch_size: {batch_size}")
    elif model_type == 'lstm_vae':
        print(f"   hidden_dim: {hidden_dim}")
        print(f"   num_layers: {num_layers}")
        print(f"   dropout: {dropout}")
        print(f"   latent_dim: {latent_dim}")
        print(f"   learning_rate: {lr}")
        print(f"   batch_size: {batch_size}")
        print(f"   beta_kl: {beta_kl}")
        print(f"   gamma_score: {gamma_score}")
    elif model_type == 'encdec':
        print(f"   hidden_dim: {hidden_dim}")
        print(f"   num_layers: {num_layers}")
        print(f"   dropout: {dropout}")
        print(f"   learning_rate: {lr}")
        print(f"   batch_size: {batch_size}")
        print(f"   lambda_score: {lambda_score}")
    print(f"   score_mean: {score_mean:.4f}")
    print(f"   score_std: {score_std:.4f}")
    print(f"{'='*80}\n")
    
    # Windows için DataLoader ayarları (multiprocessing sorunlarını önlemek için)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # Model oluştur
    if model_type == 'lstm':
        model = LSTMRegressor(input_dim=18, hidden_dim=hidden_dim, 
                             num_layers=num_layers, dropout=dropout)
    elif model_type == 'cnn':
        model = CNNRegressor(input_channels=18)
    elif model_type == 'lstm_vae':
        model = LSTMVAE(input_dim=18, hidden_dim=hidden_dim, latent_dim=latent_dim,
                       num_layers=num_layers, dropout=dropout)
    elif model_type == 'encdec':
        model = LSTMEncoderDecoder(input_dim=18, hidden_dim=hidden_dim,
                                  num_layers=num_layers, dropout=dropout)
    
    model = model.to(device)
    
    # Loss ve optimizer
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Mixed Precision Training için scaler (RTX 4080 Super optimizasyonu)
    if USE_AMP and scaler is not None:
        from torch.cuda.amp import autocast
    else:
        autocast = None
    
    # Detaylı log dosyası oluştur
    log_file_path = os.path.join(ABLATION_LOGS_DIR, f'final_training_log_{model_type}_{plastic_type}.txt')
    json_log_path = os.path.join(ABLATION_LOGS_DIR, f'final_training_log_{model_type}_{plastic_type}.json')
    
    # Log dosyasını aç (append mode - temp silinmeyecek)
    log_file = open(log_file_path, 'a', encoding='utf-8')
    training_logs = []  # JSON için tüm epoch loglarını topla
    
    # Log başlığı
    log_file.write("\n" + "="*80 + "\n")
    log_file.write(f"FINAL EĞİTİM: {model_type.upper()} - {plastic_type}\n")
    log_file.write(f"Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Final Epochs: {final_epochs}\n")
    log_file.write(f"Seed: {seed}\n")
    log_file.write(f"En İyi Parametreler (Ablation'dan):\n")
    for key, value in best_params.items():
        if key not in ['score_mean', 'score_std', 'best_val_r2', 'best_test_r2']:
            log_file.write(f"  {key}: {value}\n")
    log_file.write("="*80 + "\n\n")
    log_file.flush()
    
    # Checkpoint dosyası yolu
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 
                                 f'final_{model_type}_{plastic_type}.pth')
    
    # Checkpoint'ten devam et (varsa)
    start_epoch = 0
    best_val = float('inf')
    patience_ctr = 0
    best_state = None
    train_losses = []
    val_losses = []
    patience = 20  # Final eğitimde daha uzun patience
    
    if os.path.exists(checkpoint_path):
        print(f"  🔄 Checkpoint bulundu! Kaldığı yerden devam ediliyor: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint.get('best_val', float('inf'))
        patience_ctr = checkpoint.get('patience_ctr', 0)
        if 'best_state' in checkpoint:
            best_state = checkpoint['best_state']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"  ✓ Checkpoint yüklendi: Epoch {start_epoch}/{final_epochs}, Best Val Loss: {best_val:.6f}")
        log_file.write(f"🔄 Checkpoint'ten devam: Epoch {start_epoch}/{final_epochs}\n")
        log_file.flush()
    
    # Canlı progress bar
    epoch_pbar = tqdm(range(start_epoch, final_epochs), 
                     desc=f"Final Training {model_type.upper()}",
                     initial=start_epoch,
                     total=final_epochs,
                     ncols=100,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for epoch in epoch_pbar:
        model.train()
        train_total = 0.0
        for xb, yb, cls in train_loader:
            xb, yb, cls = xb.to(device), yb.to(device), cls.to(device)
            optimizer.zero_grad()
            
            # Mixed Precision Training (RTX 4080 Super optimizasyonu)
            if USE_AMP and scaler is not None and autocast is not None:
                with autocast():
                    if model_type in ('lstm', 'cnn'):
                        pred = model(xb)
                        loss = mse_loss(pred, yb)
                    elif model_type == 'lstm_vae':
                        recon_logits, mu, logvar, score_pred = model(xb)
                        recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        score_loss = mse_loss(score_pred, yb)
                        loss = recon_loss + beta_kl * kl + gamma_score * score_loss
                    else:  # encdec
                        recon_logits, score_pred = model(xb)
                        recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                        score_loss = mse_loss(score_pred, yb)
                        loss = recon_loss + lambda_score * score_loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Normal precision training
                if model_type in ('lstm', 'cnn'):
                    pred = model(xb)
                    loss = mse_loss(pred, yb)
                elif model_type == 'lstm_vae':
                    recon_logits, mu, logvar, score_pred = model(xb)
                    recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    score_loss = mse_loss(score_pred, yb)
                    loss = recon_loss + beta_kl * kl + gamma_score * score_loss
                else:  # encdec
                    recon_logits, score_pred = model(xb)
                    recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                    score_loss = mse_loss(score_pred, yb)
                    loss = recon_loss + lambda_score * score_loss
                
                loss.backward()
                optimizer.step()
            train_total += loss.item() * xb.size(0)
        
        # Validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb, cls in val_loader:
                xb, yb, cls = xb.to(device), yb.to(device), cls.to(device)
                if model_type in ('lstm', 'cnn'):
                    pred = model(xb)
                    loss = mse_loss(pred, yb)
                elif model_type == 'lstm_vae':
                    recon_logits, mu, logvar, score_pred = model(xb)
                    recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    score_loss = mse_loss(score_pred, yb)
                    loss = recon_loss + beta_kl * kl + gamma_score * score_loss
                else:
                    recon_logits, score_pred = model(xb)
                    recon_loss = ce_loss(recon_logits.view(-1, recon_logits.size(-1)), cls.view(-1))
                    score_loss = mse_loss(score_pred, yb)
                    loss = recon_loss + lambda_score * score_loss
                val_total += loss.item() * xb.size(0)
        
        avg_train = train_total / len(train_loader.dataset)
        avg_val = val_total / len(val_loader.dataset)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        # Canlı progress güncelleme
        epoch_pbar.set_postfix({
            'Train': f'{avg_train:.4f}',
            'Val': f'{avg_val:.4f}',
            'Best': f'{best_val:.4f}'
        })
        
        # Her epoch için detaylı log
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': float(avg_train),
            'val_loss': float(avg_val)
        }
        training_logs.append(epoch_log)
        
        # Her epoch'ta log yaz (detaylı takip için)
        if (epoch + 1) % 10 == 0 or epoch == 0 or avg_val < best_val:
            log_file.write(f"Epoch {epoch + 1:3d}/{final_epochs}: "
                         f"Train Loss: {avg_train:.6f}, "
                         f"Val Loss: {avg_val:.6f}\n")
            log_file.flush()
        
        if avg_val < best_val:
            best_val = avg_val
            patience_ctr = 0
            # Deep copy kullanarak ağırlıkları güvenli şekilde kaydet
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Her epoch'ta checkpoint kaydet (kaldığı yerden devam için)
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val': best_val,
            'patience_ctr': patience_ctr,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_type': model_type,
            'plastic_type': plastic_type,
            'final_epochs': final_epochs
        }
        if best_state is not None:
            checkpoint_data['best_state'] = best_state
        
        # Her 10 epoch'ta bir checkpoint kaydet (disk I/O optimizasyonu)
        if (epoch + 1) % 10 == 0 or epoch == final_epochs - 1:
            torch.save(checkpoint_data, checkpoint_path)
        
        if (epoch + 1) % 50 == 0:
            # GPU memory bilgisi
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                print(f"\n  Epoch {epoch + 1}/{final_epochs} - Train: {avg_train:.4f}, Val: {avg_val:.4f}, GPU Mem: {gpu_mem:.2f}GB")
            else:
                print(f"\n  Epoch {epoch + 1}/{final_epochs} - Train: {avg_train:.4f}, Val: {avg_val:.4f}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Test
    model.eval()
    test_preds = []
    with torch.no_grad():
        for xb, _, cls in test_loader:
            xb = xb.to(device)
            if model_type in ('lstm', 'cnn'):
                pred = model(xb)
            elif model_type == 'lstm_vae':
                _, _, _, pred = model(xb)
            else:
                _, pred = model(xb)
            test_preds.append(pred.cpu().numpy())
    
    test_preds = np.concatenate(test_preds) * score_std + score_mean
    
    # Metrikler
    metrics = {
        'test': {
            'rmse': math.sqrt(mean_squared_error(y_test, test_preds)),
            'mae': mean_absolute_error(y_test, test_preds),
            'r2': r2_score(y_test, test_preds)
        }
    }
    
    # Kaydet
    model_path = os.path.join(FINAL_MODELS_DIR, f'{model_type}_{plastic_type}_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'score_mean': score_mean,
        'score_std': score_std,
        'config': {
            'model_type': model_type,
            'plastic_type': plastic_type,
            'hidden_dim': hidden_dim if model_type != 'cnn' else None,
            'num_layers': num_layers if model_type != 'cnn' else None,
            'dropout': dropout if model_type != 'cnn' else None,
            'latent_dim': latent_dim if model_type == 'lstm_vae' else None,
            'beta_kl': beta_kl if model_type == 'lstm_vae' else None,
            'gamma_score': gamma_score if model_type == 'lstm_vae' else None,
            'lambda_score': lambda_score if model_type == 'encdec' else None,
            'learning_rate': lr,
            'batch_size': batch_size,
            'final_epochs': final_epochs,
            'test_r2': metrics['test']['r2'],
            'test_mae': metrics['test']['mae'],
            'test_rmse': metrics['test']['rmse']
        }
    }, model_path)
    
    print(f"\n✓ Final model kaydedildi: {model_path}")
    print(f"✓ Test R²: {metrics['test']['r2']:.4f}")
    print(f"✓ Test MAE: {metrics['test']['mae']:.4f}")
    print(f"✓ Test RMSE: {metrics['test']['rmse']:.4f}")
    
    # Final model kaydedildi, checkpoint'i temizle (disk alanı için)
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print(f"  ✓ Checkpoint temizlendi: {checkpoint_path}")
        except:
            pass  # Silinemezse sorun değil
    
    # Train/validation loss grafiklerini çiz
    plot_training_curves(train_losses, val_losses, model_type, plastic_type, final_epochs)
    
    # Final sonuç logu
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"FINAL EĞİTİM TAMAMLANDI\n")
    log_file.write(f"Test R²: {metrics['test']['r2']:.6f}\n")
    log_file.write(f"Test MAE: {metrics['test']['mae']:.6f}\n")
    log_file.write(f"Test RMSE: {metrics['test']['rmse']:.6f}\n")
    log_file.write(f"Best Val Loss: {best_val:.6f}\n")
    log_file.write(f"Toplam Epoch: {len(train_losses)}\n")
    log_file.write(f"{'='*80}\n")
    log_file.close()
    
    # JSON logunu kaydet (kalıcı - yerel diskte)
    with open(json_log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_type': model_type,
            'plastic_type': plastic_type,
            'final_epochs': final_epochs,
            'seed': seed,
            'best_params': {k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                           for k, v in best_params.items()},
            'final_metrics': {
                'test_r2': float(metrics['test']['r2']),
                'test_mae': float(metrics['test']['mae']),
                'test_rmse': float(metrics['test']['rmse'])
            },
            'best_val_loss': float(best_val),
            'total_epochs': len(train_losses),
            'training_curves': {
                'train_losses': [float(x) for x in train_losses],
                'val_losses': [float(x) for x in val_losses]
            },
            'all_epochs': training_logs,
            'model_path': model_path
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detaylı final eğitim logları kaydedildi:")
    print(f"   - Text log: {log_file_path}")
    print(f"   - JSON log: {json_log_path}")
    
    return metrics, train_losses, val_losses

# ============================================================================
# Görselleştirme Fonksiyonları
# ============================================================================

def plot_ablation_results(results_df, model_type, plastic_type):
    """Ablation study sonuçlarını görselleştir"""
    
    if len(results_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Ablation Study Results: {model_type.upper()} - {plastic_type}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Test R² dağılımı
    ax1 = axes[0, 0]
    ax1.hist(results_df['test_r2'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    best_r2 = results_df['test_r2'].max()
    ax1.axvline(best_r2, color='red', linestyle='--', linewidth=2, label=f'Best R²: {best_r2:.4f}')
    ax1.set_xlabel('Test R² Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Test R² Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation vs Test R²
    ax2 = axes[0, 1]
    scatter = ax2.scatter(results_df['val_r2'], results_df['test_r2'], 
                         c=results_df['test_r2'], cmap='viridis', 
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    best_idx = results_df['test_r2'].idxmax()
    ax2.scatter(results_df.loc[best_idx, 'val_r2'], results_df.loc[best_idx, 'test_r2'],
               color='red', s=200, marker='*', edgecolors='black', linewidth=2,
               label='Best Combination', zorder=5)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax2.set_xlabel('Validation R²', fontsize=12)
    ax2.set_ylabel('Test R²', fontsize=12)
    ax2.set_title('Validation vs Test R²', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Test R²')
    
    # 3. Hiperparametre önem analizi
    ax3 = axes[1, 0]
    param_importance = {}
    for col in results_df.columns:
        if col not in ['combination_id', 'test_r2', 'test_mae', 'test_rmse', 'val_r2', 'is_best']:
            if results_df[col].dtype in ['int64', 'float64']:
                corr = results_df[col].corr(results_df['test_r2'])
                if not np.isnan(corr):
                    param_importance[col] = abs(corr)
    
    if param_importance:
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        params, importances = zip(*sorted_params[:10])  # Top 10
        ax3.barh(range(len(params)), importances, color='coral', edgecolor='black')
        ax3.set_yticks(range(len(params)))
        ax3.set_yticklabels(params)
        ax3.set_xlabel('Absolute Correlation with Test R²', fontsize=12)
        ax3.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. En iyi 10 kombinasyon
    ax4 = axes[1, 1]
    top_10 = results_df.nlargest(10, 'test_r2')
    ax4.barh(range(len(top_10)), top_10['test_r2'], color='green', edgecolor='black', alpha=0.7)
    ax4.set_yticks(range(len(top_10)))
    ax4.set_yticklabels([f"#{i+1}" for i in range(len(top_10))])
    ax4.set_xlabel('Test R² Score', fontsize=12)
    ax4.set_title('Top 10 Combinations', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # En iyi kombinasyonu işaretle
    best_idx_in_top10 = top_10['test_r2'].idxmax()
    best_pos = list(top_10.index).index(best_idx_in_top10)
    ax4.scatter([top_10.loc[best_idx_in_top10, 'test_r2']], [best_pos],
               color='red', s=200, marker='*', edgecolors='black', linewidth=2, zorder=5)
    
    plt.tight_layout()
    
    # Kaydet
    fig_path = os.path.join(ABLATION_FIGURES_DIR, f'ablation_{model_type}_{plastic_type}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Ablation grafikleri kaydedildi: {fig_path}")
    # Windows'ta plt.show() yerine sadece kaydet (GUI gerektirmez)
    # plt.show()  # İsterseniz bu satırı açabilirsiniz (GUI backend gerekir)
    plt.close()

def plot_training_curves(train_losses, val_losses, model_type, plastic_type, epochs):
    """Train/validation loss grafiklerini çiz"""
    
    if not train_losses or not val_losses:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Training Curves: {model_type.upper()} - {plastic_type} ({epochs} epochs)', 
                 fontsize=16, fontweight='bold')
    
    epochs_list = range(1, len(train_losses) + 1)
    
    # 1. Loss grafiği
    ax1 = axes[0]
    ax1.plot(epochs_list, train_losses, label='Train Loss', color='steelblue', linewidth=2)
    ax1.plot(epochs_list, val_losses, label='Validation Loss', color='coral', linewidth=2)
    
    # En iyi epoch'u işaretle
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax1.axvline(best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch: {best_epoch}')
    ax1.scatter([best_epoch], [best_val_loss], color='red', s=100, marker='*', zorder=5, edgecolors='black', linewidth=1)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Log scale loss grafiği
    ax2 = axes[1]
    ax2.semilogy(epochs_list, train_losses, label='Train Loss', color='steelblue', linewidth=2)
    ax2.semilogy(epochs_list, val_losses, label='Validation Loss', color='coral', linewidth=2)
    ax2.axvline(best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch: {best_epoch}')
    ax2.scatter([best_epoch], [best_val_loss], color='red', s=100, marker='*', zorder=5, edgecolors='black', linewidth=1)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Loss (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Kaydet
    fig_path = os.path.join(ABLATION_FIGURES_DIR, f'training_curves_{model_type}_{plastic_type}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training grafikleri kaydedildi: {fig_path}")
    # Windows'ta plt.show() yerine sadece kaydet (GUI gerektirmez)
    # plt.show()  # İsterseniz bu satırı açabilirsiniz (GUI backend gerekir)
    plt.close()

def plot_model_comparison(all_results_summary):
    """Tüm modellerin karşılaştırma grafiklerini çiz"""
    
    if not all_results_summary:
        return
    
    summary_df = pd.DataFrame(all_results_summary)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison Across All Plastics and Models', 
                 fontsize=16, fontweight='bold')
    
    # 1. Model tipine göre ortalama R²
    ax1 = axes[0, 0]
    model_avg = summary_df.groupby('Model_Type')['Test_R2'].mean().sort_values(ascending=False)
    bars1 = ax1.bar(range(len(model_avg)), model_avg.values, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xticks(range(len(model_avg)))
    ax1.set_xticklabels(model_avg.index, rotation=45, ha='right')
    ax1.set_ylabel('Average Test R²', fontsize=12, fontweight='bold')
    ax1.set_title('Average R² by Model Type', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Değerleri üzerine yaz
    for i, (bar, val) in enumerate(zip(bars1, model_avg.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Plastik tipine göre ortalama R²
    ax2 = axes[0, 1]
    plastic_avg = summary_df.groupby('Plastic_Type')['Test_R2'].mean().sort_values(ascending=False)
    bars2 = ax2.bar(range(len(plastic_avg)), plastic_avg.values, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xticks(range(len(plastic_avg)))
    ax2.set_xticklabels(plastic_avg.index, rotation=45, ha='right')
    ax2.set_ylabel('Average Test R²', fontsize=12, fontweight='bold')
    ax2.set_title('Average R² by Plastic Type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars2, plastic_avg.values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Heatmap: Model x Plastik
    ax3 = axes[1, 0]
    pivot = summary_df.pivot_table(values='Test_R2', index='Model_Type', columns='Plastic_Type', aggfunc='mean')
    im = ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=pivot.values.min(), vmax=pivot.values.max())
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels(pivot.columns)
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(pivot.index)
    ax3.set_title('R² Heatmap: Model x Plastic', fontsize=14, fontweight='bold')
    
    # Değerleri yaz
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax3.text(j, i, f'{pivot.iloc[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    plt.colorbar(im, ax=ax3, label='Test R²')
    
    # 4. Box plot: Model tipine göre R² dağılımı
    ax4 = axes[1, 1]
    model_list = summary_df['Model_Type'].unique()
    data_for_box = [summary_df[summary_df['Model_Type'] == model]['Test_R2'].values for model in model_list]
    bp = ax4.boxplot(data_for_box, labels=model_list, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('black')
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Test R²', fontsize=12, fontweight='bold')
    ax4.set_title('R² Distribution by Model Type', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Kaydet
    fig_path = os.path.join(ABLATION_FIGURES_DIR, 'model_comparison_all.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Model karşılaştırma grafikleri kaydedildi: {fig_path}")
    # Windows'ta plt.show() yerine sadece kaydet (GUI gerektirmez)
    # plt.show()  # İsterseniz bu satırı açabilirsiniz (GUI backend gerekir)
    plt.close()

def create_comparison_table_and_heatmap(all_ablation_results, plastic_type, best_model_type):
    """
    Tüm modellerin parametrelerini ve sonuçlarını gösteren tablo ve heatmap oluşturur.
    Best model/parametreleri vurgular.
    """
    # Tüm modellerin sonuçlarını topla
    all_model_data = []
    
    for model_type, results in all_ablation_results.items():
        ablation_df = results['ablation_df']
        best_params = results['best_params']
        best_val_r2 = results['best_val_r2']
        best_test_r2 = results['best_test_r2']
        
        # En iyi kombinasyonu bul
        best_row = ablation_df.loc[ablation_df['val_r2'].idxmax()]
        
        row_data = {
            'Model': model_type.upper(),
            'Val_R2': best_val_r2,
            'Test_R2': best_test_r2,
            'Best_Epoch': int(best_row['best_epoch']),
        }
        
        # Model tipine göre parametreleri ekle
        if model_type == 'lstm':
            row_data['hidden_dim'] = int(best_params.get('hidden_dim', 0))
            row_data['num_layers'] = int(best_params.get('num_layers', 0))
            row_data['dropout'] = float(best_params.get('dropout', 0))
            row_data['learning_rate'] = float(best_params.get('learning_rate', 0))
            row_data['batch_size'] = int(best_params.get('batch_size', 0))
            row_data['latent_dim'] = None
            row_data['beta_kl'] = None
            row_data['gamma_score'] = None
            row_data['lambda_score'] = None
        elif model_type == 'cnn':
            row_data['hidden_dim'] = None
            row_data['num_layers'] = None
            row_data['dropout'] = None
            row_data['learning_rate'] = float(best_params.get('learning_rate', 0))
            row_data['batch_size'] = int(best_params.get('batch_size', 0))
            row_data['latent_dim'] = None
            row_data['beta_kl'] = None
            row_data['gamma_score'] = None
            row_data['lambda_score'] = None
        elif model_type == 'lstm_vae':
            row_data['hidden_dim'] = int(best_params.get('hidden_dim', 0))
            row_data['num_layers'] = int(best_params.get('num_layers', 0))
            row_data['dropout'] = float(best_params.get('dropout', 0))
            row_data['learning_rate'] = float(best_params.get('learning_rate', 0))
            row_data['batch_size'] = int(best_params.get('batch_size', 0))
            row_data['latent_dim'] = int(best_params.get('latent_dim', 0))
            row_data['beta_kl'] = float(best_params.get('beta_kl', 0))
            row_data['gamma_score'] = float(best_params.get('gamma_score', 0))
            row_data['lambda_score'] = None
        elif model_type == 'encdec':
            row_data['hidden_dim'] = int(best_params.get('hidden_dim', 0))
            row_data['num_layers'] = int(best_params.get('num_layers', 0))
            row_data['dropout'] = float(best_params.get('dropout', 0))
            row_data['learning_rate'] = float(best_params.get('learning_rate', 0))
            row_data['batch_size'] = int(best_params.get('batch_size', 0))
            row_data['latent_dim'] = None
            row_data['beta_kl'] = None
            row_data['gamma_score'] = None
            row_data['lambda_score'] = float(best_params.get('lambda_score', 0))
        
        row_data['is_best'] = (model_type == best_model_type)
        all_model_data.append(row_data)
    
    comparison_df = pd.DataFrame(all_model_data)
    
    # Tabloyu göster
    print(f"\n{'='*80}")
    print(f"📊 TÜM MODELLER KARŞILAŞTIRMA TABLOSU - {plastic_type}")
    print(f"{'='*80}")
    
    # Best model'i vurgula
    def highlight_best(row):
        if row['is_best']:
            return ['background-color: #FFD700; font-weight: bold'] * len(row)
        return [''] * len(row)
    
    # Tabloyu yazdır (best model koyu/kalın göster)
    display_df = comparison_df.copy()
    display_df = display_df.drop('is_best', axis=1)  # Gösterim için is_best'i kaldır
    
    # Best model'i işaretle
    print("\n🏆 = EN İYİ MODEL (Koyu/Altın Renk)")
    print("\n" + display_df.to_string(index=False))
    
    # Styled tabloyu kaydet
    styled_df = comparison_df.style.apply(highlight_best, axis=1)
    
    # HTML olarak kaydet
    html_path = os.path.join(ABLATION_TABLES_DIR, f'model_comparison_table_{plastic_type}.html')
    styled_df.to_html(html_path)
    print(f"\n✓ Karşılaştırma tablosu kaydedildi: {html_path}")
    
    # CSV olarak da kaydet
    csv_path = os.path.join(ABLATION_TABLES_DIR, f'model_comparison_table_{plastic_type}.csv')
    display_df.to_csv(csv_path, index=False)
    print(f"✓ Karşılaştırma tablosu (CSV) kaydedildi: {csv_path}")
    
    # Heatmap oluştur (Model x Parametreler)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Heatmap için uygun veri hazırla (sadece numeric ve best model'i vurgula)
    heatmap_data = comparison_df[['Val_R2', 'Test_R2', 'Best_Epoch', 
                                   'hidden_dim', 'num_layers', 'dropout', 
                                   'learning_rate', 'batch_size']].copy()
    
    # NaN değerleri 0 ile doldur (görselleştirme için)
    heatmap_data = heatmap_data.fillna(0)
    
    # Best model'i belirle
    best_idx = comparison_df[comparison_df['is_best']].index[0]
    
    # Heatmap çiz
    im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', 
                   vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
    
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels(comparison_df['Model'].values)
    ax.set_title(f'Model Comparison Heatmap: {plastic_type} (Best Model Highlighted)', 
                 fontsize=14, fontweight='bold')
    
    # Best model satırını vurgula
    ax.add_patch(plt.Rectangle((-0.5, best_idx - 0.5), len(heatmap_data.columns), 1,
                               fill=False, edgecolor='gold', linewidth=3))
    
    # Değerleri yaz
    for i in range(len(heatmap_data)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.iloc[i, j]
            if val != 0:  # Sadece 0 olmayan değerleri göster
                text_color = 'white' if i == best_idx else 'black'
                text_weight = 'bold' if i == best_idx else 'normal'
                ax.text(j, i, f'{val:.3f}' if val < 1 else f'{int(val)}',
                       ha="center", va="center", color=text_color, 
                       fontweight=text_weight, fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Normalized Value')
    plt.tight_layout()
    
    # Kaydet
    fig_path = os.path.join(ABLATION_FIGURES_DIR, f'model_comparison_heatmap_{plastic_type}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Model karşılaştırma heatmap kaydedildi: {fig_path}")
    plt.show()
    plt.close()
    
    return comparison_df

def plot_amino_acid_probability_mass_heatmap(plastic_type, data_dir=None):
    """
    Her plastik için amino acid probability x mass heatmap'i oluşturur.
    Orijinal veri setindeki peptidlerin amino acid dağılımını analiz eder.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Veri yükle
    csv_path = os.path.join(data_dir, f"{plastic_type}.csv")
    if not os.path.exists(csv_path):
        print(f"⚠ {plastic_type} CSV dosyası bulunamadı, heatmap oluşturulamadı")
        return None
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Sequence'])
    df['Sequence'] = df['Sequence'].str.strip().str.upper()
    
    sequences = df['Sequence'].tolist()
    
    # Pozisyon bazlı amino acid probability hesapla
    max_len = 12  # Peptid uzunluğu
    position_probs = np.zeros((max_len, len(AMINO_ACIDS)))
    
    for seq in sequences:
        if len(seq) == max_len:
            for pos, aa in enumerate(seq):
                if aa in AA_TO_IDX:
                    position_probs[pos, AA_TO_IDX[aa]] += 1
    
    # Normalize et (probability)
    for pos in range(max_len):
        total = position_probs[pos].sum()
        if total > 0:
            position_probs[pos] = position_probs[pos] / total
    
    # Mass'e göre düzenle
    aa_masses = [AA_MASSES[aa] for aa in AMINO_ACIDS]
    
    # Heatmap için veri hazırla: Position x Amino Acid (mass'e göre sıralı)
    # Amino asitleri mass'e göre sırala
    aa_mass_pairs = [(aa, AA_MASSES[aa]) for aa in AMINO_ACIDS]
    aa_mass_pairs.sort(key=lambda x: x[1])
    sorted_aas = [aa for aa, _ in aa_mass_pairs]
    sorted_masses = [mass for _, mass in aa_mass_pairs]
    
    # Sıralanmış probability matrix
    sorted_probs = np.zeros((max_len, len(AMINO_ACIDS)))
    for i, aa in enumerate(sorted_aas):
        orig_idx = AA_TO_IDX[aa]
        sorted_probs[:, i] = position_probs[:, orig_idx]
    
    # Heatmap çiz
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(sorted_probs.T, cmap='YlOrRd', aspect='auto', 
                   vmin=0, vmax=sorted_probs.max())
    
    ax.set_xticks(range(max_len))
    ax.set_xticklabels([f'Pos {i+1}' for i in range(max_len)])
    ax.set_yticks(range(len(sorted_aas)))
    ax.set_yticklabels([f'{aa}\n({m}Da)' for aa, m in zip(sorted_aas, sorted_masses)])
    ax.set_xlabel('Peptide Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amino Acid (Mass in Daltons)', fontsize=12, fontweight='bold')
    ax.set_title(f'Amino Acid Probability x Mass Heatmap: {plastic_type}', 
                 fontsize=14, fontweight='bold')
    
    # Değerleri yaz (sadece > 0.05 olanlar)
    for i in range(len(sorted_aas)):
        for j in range(max_len):
            val = sorted_probs[j, i]
            if val > 0.05:  # Sadece önemli değerleri göster
                text_color = 'black' if val < 0.5 else 'white'
                ax.text(j, i, f'{val:.2f}',
                       ha="center", va="center", color=text_color, 
                       fontweight='bold', fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Probability')
    plt.tight_layout()
    
    # Kaydet
    fig_path = os.path.join(ABLATION_FIGURES_DIR, f'aa_probability_mass_heatmap_{plastic_type}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Amino acid probability x mass heatmap kaydedildi: {fig_path}")
    plt.show()
    plt.close()
    
    return sorted_probs, sorted_aas, sorted_masses

def plot_generated_peptides_amino_acid_heatmap(generated_peptides_list, plastic_type, model_type):
    """
    Üretilen peptidler için amino acid probability x mass heatmap'i oluşturur.
    Üretilen peptidlerin pozisyon bazlı amino acid dağılımını ve skorlarını analiz eder.
    
    Args:
        generated_peptides_list: generate_peptides_with_best_model'den dönen liste
        plastic_type: Plastik tipi
        model_type: Model tipi
    """
    if not generated_peptides_list or len(generated_peptides_list) == 0:
        print(f"⚠ {plastic_type} için üretilen peptid bulunamadı, heatmap oluşturulamadı")
        return None
    
    # Üretilen peptidleri ve skorlarını al
    sequences = []
    scores = []
    
    for result in generated_peptides_list:
        optimized_peptide = result.get('optimized_peptide', '')
        optimized_score = result.get('optimized_score', 0.0)
        if optimized_peptide and len(optimized_peptide) == 12:
            sequences.append(optimized_peptide)
            scores.append(optimized_score)
    
    if not sequences:
        print(f"⚠ {plastic_type} için geçerli üretilen peptid bulunamadı")
        return None
    
    # Pozisyon bazlı amino acid probability hesapla
    max_len = 12
    position_probs = np.zeros((max_len, len(AMINO_ACIDS)))
    position_scores = []  # Her pozisyon için skor ortalaması
    
    for seq, score in zip(sequences, scores):
        if len(seq) == max_len:
            for pos, aa in enumerate(seq):
                if aa in AA_TO_IDX:
                    position_probs[pos, AA_TO_IDX[aa]] += 1
    
    # Normalize et (probability)
    for pos in range(max_len):
        total = position_probs[pos].sum()
        if total > 0:
            position_probs[pos] = position_probs[pos] / total
    
    # Mass'e göre düzenle
    aa_mass_pairs = [(aa, AA_MASSES[aa]) for aa in AMINO_ACIDS]
    aa_mass_pairs.sort(key=lambda x: x[1])
    sorted_aas = [aa for aa, _ in aa_mass_pairs]
    sorted_masses = [mass for _, mass in aa_mass_pairs]
    
    # Sıralanmış probability matrix
    sorted_probs = np.zeros((max_len, len(AMINO_ACIDS)))
    for i, aa in enumerate(sorted_aas):
        orig_idx = AA_TO_IDX[aa]
        sorted_probs[:, i] = position_probs[:, orig_idx]
    
    # Heatmap çiz (2 panel: probability ve score distribution)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Amino Acid Probability x Mass Heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(sorted_probs.T, cmap='YlOrRd', aspect='auto', 
                    vmin=0, vmax=sorted_probs.max())
    
    ax1.set_xticks(range(max_len))
    ax1.set_xticklabels([f'Pos {i+1}' for i in range(max_len)])
    ax1.set_yticks(range(len(sorted_aas)))
    ax1.set_yticklabels([f'{aa}\n({m}Da)' for aa, m in zip(sorted_aas, sorted_masses)])
    ax1.set_xlabel('Peptide Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amino Acid (Mass in Daltons)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Generated Peptides - Amino Acid Probability x Mass: {plastic_type} ({model_type.upper()})', 
                  fontsize=14, fontweight='bold')
    
    # Değerleri yaz (sadece > 0.05 olanlar)
    for i in range(len(sorted_aas)):
        for j in range(max_len):
            val = sorted_probs[j, i]
            if val > 0.05:
                text_color = 'black' if val < 0.5 else 'white'
                ax1.text(j, i, f'{val:.2f}',
                        ha="center", va="center", color=text_color, 
                        fontweight='bold', fontsize=8)
    
    plt.colorbar(im1, ax=ax1, label='Probability')
    
    # 2. Score Distribution
    ax2 = axes[1]
    ax2.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(scores):.2f}')
    ax2.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(scores):.2f}')
    ax2.set_xlabel('Optimized Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Generated Peptides Score Distribution: {plastic_type} ({model_type.upper()})', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Kaydet
    fig_path = os.path.join(ABLATION_FIGURES_DIR, 
                           f'generated_peptides_aa_heatmap_{model_type}_{plastic_type}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Üretilen peptidler amino acid heatmap kaydedildi: {fig_path}")
    plt.show()
    plt.close()
    
    # İstatistikler
    print(f"\n📊 Üretilen Peptidler İstatistikleri ({plastic_type} - {model_type.upper()}):")
    print(f"   Toplam üretilen peptid: {len(sequences)}")
    print(f"   Ortalama skor: {np.mean(scores):.4f}")
    print(f"   Medyan skor: {np.median(scores):.4f}")
    print(f"   En iyi skor: {np.min(scores):.4f}")
    print(f"   En kötü skor: {np.max(scores):.4f}")
    print(f"   Skor standart sapma: {np.std(scores):.4f}")
    
    return sorted_probs, sorted_aas, sorted_masses, scores

def create_detailed_report(summary_df, all_results_summary):
    """Detaylı istatistik raporu oluştur"""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("DETAYLI ABLATION STUDY RAPORU")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Genel istatistikler
    report_lines.append("📊 GENEL İSTATİSTİKLER")
    report_lines.append("-"*80)
    report_lines.append(f"Toplam Model-Plastik Kombinasyonu: {len(summary_df)}")
    report_lines.append(f"En Yüksek Test R²: {summary_df['Test_R2'].max():.4f}")
    report_lines.append(f"En Düşük Test R²: {summary_df['Test_R2'].min():.4f}")
    report_lines.append(f"Ortalama Test R²: {summary_df['Test_R2'].mean():.4f}")
    report_lines.append(f"Test R² Standart Sapma: {summary_df['Test_R2'].std():.4f}")
    report_lines.append("")
    
    # Model tipine göre
    report_lines.append("📊 MODEL TİPİNE GÖRE PERFORMANS")
    report_lines.append("-"*80)
    model_stats = summary_df.groupby('Model_Type')['Test_R2'].agg(['mean', 'std', 'min', 'max', 'count'])
    for model, stats in model_stats.iterrows():
        report_lines.append(f"{model.upper()}:")
        report_lines.append(f"  Ortalama R²: {stats['mean']:.4f} ± {stats['std']:.4f}")
        report_lines.append(f"  Aralık: [{stats['min']:.4f}, {stats['max']:.4f}]")
        report_lines.append(f"  Örnek Sayısı: {int(stats['count'])}")
        report_lines.append("")
    
    # Plastik tipine göre
    report_lines.append("📊 PLASTİK TİPİNE GÖRE PERFORMANS")
    report_lines.append("-"*80)
    plastic_stats = summary_df.groupby('Plastic_Type')['Test_R2'].agg(['mean', 'std', 'min', 'max', 'count'])
    for plastic, stats in plastic_stats.iterrows():
        report_lines.append(f"{plastic}:")
        report_lines.append(f"  Ortalama R²: {stats['mean']:.4f} ± {stats['std']:.4f}")
        report_lines.append(f"  Aralık: [{stats['min']:.4f}, {stats['max']:.4f}]")
        report_lines.append(f"  Örnek Sayısı: {int(stats['count'])}")
        report_lines.append("")
    
    # En iyi kombinasyonlar
    report_lines.append("🏆 EN İYİ KOMBİNASYONLAR")
    report_lines.append("-"*80)
    best_per_plastic = summary_df.loc[summary_df.groupby('Plastic_Type')['Test_R2'].idxmax()]
    for idx, row in best_per_plastic.iterrows():
        report_lines.append(f"{row['Plastic_Type']} - {row['Model_Type']}: R² = {row['Test_R2']:.4f}")
    report_lines.append("")
    
    # Raporu kaydet
    report_text = "\n".join(report_lines)
    report_path = os.path.join(ABLATION_TABLES_DIR, 'detailed_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ Detaylı rapor kaydedildi: {report_path}")
    print("\n" + report_text)

# ============================================================================
# Peptid Optimizasyonu (Simulated Annealing) - PLASTİK BLOKLARINDAN ÖNCE TANIMLANMALI
# ============================================================================

class PeptideOptimizer:
    """Simulated Annealing ile peptit optimizasyonu - Her plastik için en iyi model ile."""
    
    def __init__(self, model, model_type, score_mean, score_std, 
                 initial_temp=0.5, cooling_rate=0.9, max_iterations=3000):
        """
        Args:
            model: Eğitilmiş model (PyTorch)
            model_type: Model tipi ('lstm', 'cnn', 'lstm_vae', 'encdec')
            score_mean: Score normalizasyon ortalaması
            score_std: Score normalizasyon standart sapması
            initial_temp: Başlangıç sıcaklığı
            cooling_rate: Soğutma oranı
            max_iterations: Maksimum iterasyon sayısı
        """
        self.model = model
        self.model_type = model_type
        self.score_mean = score_mean
        self.score_std = score_std
        self.temperature = initial_temp
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.amino_acids = list(AMINO_ACIDS)
        self.model.eval()  # Evaluation mode
        
    def peptide_to_one_hot(self, peptide):
        """Peptit dizisini one-hot encoding'e çevir."""
        one_hot = np.zeros((12, len(AMINO_ACIDS)))
        for i, aa in enumerate(peptide):
            if aa in AA_TO_IDX:
                one_hot[i, AA_TO_IDX[aa]] = 1
        return one_hot[np.newaxis, :, :]
    
    def predict_score(self, peptide):
        """Peptit için skor tahmin et (normalize edilmemiş)."""
        one_hot = self.peptide_to_one_hot(peptide)
        X = torch.tensor(one_hot, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if self.model_type == 'lstm_vae':
                _, _, _, pred_norm = self.model(X)
            elif self.model_type == 'encdec':
                _, pred_norm = self.model(X)
            else:  # lstm, cnn
                pred_norm = self.model(X)
            
            # Denormalize et
            pred = pred_norm.cpu().numpy()[0] * self.score_std + self.score_mean
            return float(pred)
    
    def move_operator(self, peptide):
        """Yeni komşu peptit üret (substitution veya swap)."""
        if random.random() < 0.75:  # %75 substitution
            pos = random.randint(0, len(peptide) - 1)
            current_aa = peptide[pos]
            possible_aas = [aa for aa in self.amino_acids if aa != current_aa]
            new_aa = random.choice(possible_aas)
            return peptide[:pos] + new_aa + peptide[pos + 1:]
        else:  # %25 swap
            pos1, pos2 = random.sample(range(len(peptide)), 2)
            peptide_list = list(peptide)
            peptide_list[pos1], peptide_list[pos2] = peptide_list[pos2], peptide_list[pos1]
            return ''.join(peptide_list)
    
    def optimize(self, initial_peptide, n_samples=1):
        """
        Simulated annealing optimizasyonu.
        
        Args:
            initial_peptide: Başlangıç peptit dizisi
            n_samples: Kaç farklı başlangıç noktasından optimizasyon yapılacak
        
        Returns:
            List of (best_peptide, best_score, trajectory) tuples
        """
        results = []
        
        for sample_idx in range(n_samples):
            # Her sample için farklı başlangıç noktası (eğer n_samples > 1)
            if sample_idx == 0:
                current_peptide = initial_peptide
            else:
                # Rastgele başlangıç peptit
                current_peptide = ''.join(np.random.choice(list(AMINO_ACIDS), size=12))
            
            best_peptide = current_peptide
            current_score = self.predict_score(current_peptide)
            best_score = current_score
            
            trajectory = [(0, current_score)]
            self.temperature = self.initial_temp
            
            for iteration in range(self.max_iterations):
                new_peptide = self.move_operator(current_peptide)
                new_score = self.predict_score(new_peptide)
                
                delta = new_score - current_score
                
                # Metropolis kriteri (düşük skor = daha iyi)
                if delta < 0 or random.random() < np.exp(-delta / self.temperature):
                    current_peptide = new_peptide
                    current_score = new_score
                    trajectory.append((iteration + 1, current_score))
                    
                    if current_score < best_score:
                        best_peptide = current_peptide
                        best_score = current_score
                
                # Soğutma (her 75 iterasyonda bir)
                if (iteration + 1) % 75 == 0 and iteration > 0:
                    self.temperature *= self.cooling_rate
                
                if self.temperature < 0.1:
                    break
            
            results.append({
                'best_peptide': best_peptide,
                'best_score': best_score,
                'initial_peptide': current_peptide if sample_idx == 0 else initial_peptide,
                'initial_score': self.predict_score(initial_peptide) if sample_idx == 0 else self.predict_score(current_peptide),
                'trajectory': trajectory,
                'improvement': self.predict_score(initial_peptide) - best_score if sample_idx == 0 else current_score - best_score
            })
        
        return results

def generate_peptides_with_best_model(plastic_type, model_type, best_params, 
                                      n_samples=10, n_optimizations=5,
                                      initial_temp=0.5, cooling_rate=0.9, 
                                      max_iterations=3000):
    """
    Her plastik için en iyi model ile yeni peptidler üretir.
    
    Args:
        plastic_type: Plastik tipi ('PET', 'PP', vb.)
        model_type: Model tipi ('lstm', 'cnn', 'lstm_vae', 'encdec')
        best_params: Ablation study'den gelen en iyi parametreler
        n_samples: Kaç farklı başlangıç peptitinden başlanacak
        n_optimizations: Her başlangıç için kaç optimizasyon yapılacak
        initial_temp: Simulated annealing başlangıç sıcaklığı
        cooling_rate: Soğutma oranı
        max_iterations: Maksimum iterasyon
    
    Returns:
        List of generated peptides with scores
    """
    print(f"\n{'='*80}")
    print(f"YENİ PEPTİD ÜRETİMİ: {model_type.upper()} - {plastic_type}")
    print(f"{'='*80}")
    
    # Model yükle
    model_path = os.path.join(FINAL_MODELS_DIR, f'{model_type}_{plastic_type}_final.pth')
    if not os.path.exists(model_path):
        print(f"✗ Model bulunamadı: {model_path}")
        return []
    
    print(f"✓ Model yükleniyor: {model_path}")
    
    # Model oluştur ve yükle (güvenli okuma - varsayılan değerlerle)
    if model_type == 'lstm':
        model = LSTMRegressor(
            input_dim=len(AMINO_ACIDS),
            hidden_dim=best_params.get('hidden_dim', 128),
            num_layers=best_params.get('num_layers', 2),
            dropout=best_params.get('dropout', 0.1)
        )
    elif model_type == 'cnn':
        model = CNNRegressor(input_dim=len(AMINO_ACIDS))
    elif model_type == 'lstm_vae':
        model = LSTMVAE(
            input_dim=len(AMINO_ACIDS),
            hidden_dim=best_params.get('hidden_dim', 128),
            num_layers=best_params.get('num_layers', 2),
            latent_dim=best_params.get('latent_dim', 64),
            dropout=best_params.get('dropout', 0.1)
        )
    elif model_type == 'encdec':
        model = LSTMEncoderDecoder(
            input_dim=len(AMINO_ACIDS),
            hidden_dim=best_params.get('hidden_dim', 128),
            num_layers=best_params.get('num_layers', 2),
            dropout=best_params.get('dropout', 0.1)
        )
    else:
        print(f"✗ Bilinmeyen model tipi: {model_type}")
        return []
    
    # Model ağırlıklarını yükle
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    score_mean = best_params.get('score_mean', 0.0)
    score_std = best_params.get('score_std', 1.0)
    
    # Optimizer oluştur
    optimizer = PeptideOptimizer(
        model=model,
        model_type=model_type,
        score_mean=score_mean,
        score_std=score_std,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        max_iterations=max_iterations
    )
    
    # Rastgele başlangıç peptitleri oluştur
    all_results = []
    
    print(f"\n🔬 {n_samples} farklı başlangıç peptitinden optimizasyon başlıyor...")
    
    for sample_idx in range(n_samples):
        # Rastgele başlangıç peptit
        initial_peptide = ''.join(np.random.choice(list(AMINO_ACIDS), size=12))
        initial_score = optimizer.predict_score(initial_peptide)
        
        print(f"\n  Örnek {sample_idx + 1}/{n_samples}: {initial_peptide} (Skor: {initial_score:.4f})")
        
        # Optimizasyon yap
        opt_results = optimizer.optimize(initial_peptide, n_samples=n_optimizations)
        
        for opt_idx, opt_result in enumerate(opt_results):
            all_results.append({
                'sample_id': sample_idx + 1,
                'optimization_id': opt_idx + 1,
                'initial_peptide': opt_result['initial_peptide'],
                'initial_score': opt_result['initial_score'],
                'optimized_peptide': opt_result['best_peptide'],
                'optimized_score': opt_result['best_score'],
                'improvement': opt_result['improvement'],
                'trajectory': opt_result['trajectory']
            })
            
            print(f"    Optimizasyon {opt_idx + 1}: {opt_result['best_peptide']} "
                  f"(Skor: {opt_result['best_score']:.4f}, "
                  f"İyileştirme: {opt_result['improvement']:.4f})")
    
    print(f"\n✓ Toplam {len(all_results)} yeni peptid üretildi")
    
    # Sonuçları kaydet
    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(ABLATION_TABLES_DIR, 
                               f'generated_peptides_{model_type}_{plastic_type}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"✓ Sonuçlar kaydedildi: {output_path}")
    
    return all_results

# ============================================================================
# Çalıştırma Parametreleri (Script başında değiştirilebilir)
# ============================================================================
# Çalıştırılacak plastik ve model listelerini parametreleştir
# Sadece belirtilen plastik/model kombinasyonları çalıştırılacak
# TÜM PLASTİKLER İÇİN ÇALIŞTIRMAK İÇİN:
PLASTICS_TO_RUN = ['PET', 'PP', 'PE', 'PVC', 'Nylon', 'PMMA', 'PS']  # Tüm plastikler
# Sadece belirli plastikler için: PLASTICS_TO_RUN = ['PET', 'PP']  # Örnek

MODEL_TYPES_TO_RUN = ['lstm', 'cnn', 'lstm_vae', 'encdec']  # Tüm modeller
# Sadece belirli modeller için: MODEL_TYPES_TO_RUN = ['lstm', 'cnn']  # Örnek

# Sonuç değişkenlerini başlat (NameError'ı önlemek için)
pet_results = {}
pp_results = {}
pe_results = {}
pvc_results = {}
nylon_results = {}
pmma_results = {}
ps_results = {}

# ============================================================================
# HÜCRE 1: PET için Ablation Study ve Final Eğitim
# ============================================================================

# Windows yerel ortam - direkt çalıştır
# Ana kod bloğu - Windows multiprocessing için gerekli
def main():
    """Ana çalıştırma fonksiyonu - Windows multiprocessing için"""
    global pet_results, pp_results, pe_results, pvc_results, nylon_results, pmma_results, ps_results
    
    if 'PET' in PLASTICS_TO_RUN:
        print("\n" + "="*80)
        print("HÜCRE 1: PET PLASTİK TİPİ İÇİN ABLATION STUDY VE FINAL EĞİTİM")
        print("="*80)
    
    # Veri klasörünü kontrol et
    print(f"\n📁 Veri klasörü kontrolü:")
    print(f"   DATA_DIR: {DATA_DIR}")
    if os.path.exists(DATA_DIR):
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        print(f"   ✓ Klasör mevcut, {len(csv_files)} CSV dosyası bulundu")
        if csv_files:
            print(f"   CSV dosyaları: {', '.join(csv_files[:5])}{'...' if len(csv_files) > 5 else ''}")
        
        # PET.csv var mı kontrol et
        pet_csv = os.path.join(DATA_DIR, "PET.csv")
        if not os.path.exists(pet_csv):
            print(f"   ⚠ PET.csv bulunamadı! Mevcut CSV dosyaları: {csv_files}")
    else:
        print(f"   ✗ Klasör bulunamadı: {DATA_DIR}")
        print("   Lütfen veri klasörünün doğru yolda olduğundan emin olun.")
        print("   Veri klasörünü manuel olarak belirtin:")
        print("   DATA_DIR = '/content/drive/MyDrive/.../newDate'")
        raise FileNotFoundError(f"Veri klasörü bulunamadı: {DATA_DIR}")

    plastic_type = 'PET'
    model_types = [m for m in MODEL_TYPES_TO_RUN]  # Filtrelenmiş model listesi

    # ADIM 1: Tüm modellerin ablation study'lerini yap (50 epoch)
    print(f"\n{'='*80}")
    print("ADIM 1: TÜM MODELLER İÇİN ABLATION STUDY (50 EPOCH)")
    print(f"{'='*80}")
    
    all_ablation_results = {}  # Tüm modellerin sonuçlarını topla
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"ABLATION: {model_type.upper()} - {plastic_type}")
        print(f"{'='*80}")
        
        # Ablation study (50 epoch)
        best_params, ablation_df = run_ablation_study(
            model_type=model_type,
            plastic_type=plastic_type,
            data_dir=DATA_DIR,
            ablation_epochs=50,
            seed=42
        )
        
        # Sonuçları topla
        all_ablation_results[model_type] = {
            'best_params': best_params,
            'ablation_df': ablation_df,
            'best_val_r2': best_params.get('best_val_r2', -float('inf')),
            'best_test_r2': best_params.get('best_test_r2', -float('inf'))
        }
        
        print(f"✓ {model_type.upper()} tamamlandı - Val R²: {all_ablation_results[model_type]['best_val_r2']:.6f}, Test R²: {all_ablation_results[model_type]['best_test_r2']:.6f}")
    
    # ADIM 2: En iyi model+parametre kombinasyonunu bul (Val R²'ye göre)
    print(f"\n{'='*80}")
    print("ADIM 2: EN İYİ MODEL VE PARAMETRE SEÇİMİ")
    print(f"{'='*80}")
    
    best_model_type = None
    best_val_r2 = -float('inf')
    best_params_overall = None
    
    for model_type, results in all_ablation_results.items():
        val_r2 = results['best_val_r2']
        test_r2 = results['best_test_r2']
        print(f"  {model_type.upper()}: Val R² = {val_r2:.6f}, Test R² = {test_r2:.6f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_type = model_type
            best_params_overall = results['best_params']
    
    print(f"\n🏆 EN İYİ MODEL: {best_model_type.upper()}")
    print(f"   Val R²: {best_val_r2:.6f}")
    print(f"   Test R²: {best_params_overall.get('best_test_r2', 'N/A')}")
    print(f"   Parametreler:")
    for key, value in best_params_overall.items():
        if key not in ['score_mean', 'score_std', 'best_val_r2', 'best_test_r2']:
            print(f"     {key}: {value}")
    
    # ADIM 2.5: Tüm modellerin karşılaştırma tablosu ve heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.5: TÜM MODELLER KARŞILAŞTIRMA TABLOSU VE HEATMAP")
    print(f"{'='*80}")
    
    comparison_df = create_comparison_table_and_heatmap(
        all_ablation_results, plastic_type, best_model_type
    )
    
    # ADIM 2.6: Amino Acid Probability x Mass Heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.6: AMINO ACID PROBABILITY X MASS HEATMAP")
    print(f"{'='*80}")
    
    plot_amino_acid_probability_mass_heatmap(plastic_type='PET', data_dir=DATA_DIR)
    
    # ADIM 3: Sadece en iyi model için final eğitim (250 epoch)
    print(f"\n{'='*80}")
    print(f"ADIM 3: FINAL EĞİTİM - {best_model_type.upper()} (250 EPOCH)")
    print(f"{'='*80}")
    
    metrics, train_losses, val_losses = train_final_model(
        model_type=best_model_type,
        plastic_type=plastic_type,
        best_params=best_params_overall,
        data_dir=DATA_DIR,
        final_epochs=250,
        seed=42
    )
    
    # Sonuçları kaydet (sadece en iyi model)
    pet_results[best_model_type] = {
        'best_params': best_params_overall,
        'ablation_df': all_ablation_results[best_model_type]['ablation_df'],
        'final_metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'is_best_model': True
    }
    
    # Diğer modellerin ablation sonuçlarını da kaydet (karşılaştırma için)
    for model_type, results in all_ablation_results.items():
        if model_type != best_model_type:
            pet_results[model_type] = {
                'best_params': results['best_params'],
                'ablation_df': results['ablation_df'],
                'is_best_model': False
            }
    
    # ADIM 4: Sadece en iyi model ile yeni peptid üretimi
    print(f"\n{'='*80}")
    print(f"ADIM 4: YENİ PEPTİD ÜRETİMİ - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    pet_generated_peptides = {}
    try:
        generated = generate_peptides_with_best_model(
            plastic_type='PET',
            model_type=best_model_type,
            best_params=best_params_overall,
            n_samples=10,  # 10 farklı başlangıç peptitinden
            n_optimizations=3,  # Her biri için 3 optimizasyon
            max_iterations=2000
        )
        if generated:
            pet_generated_peptides[best_model_type] = generated
    except Exception as e:
        print(f"✗ PET - {best_model_type} için peptid üretim hatası: {e}")
    
    # ADIM 4.5: Üretilen peptidler için amino acid probability x mass heatmap
    print(f"\n{'='*80}")
    print(f"ADIM 4.5: ÜRETİLEN PEPTİDLER İÇİN AMINO ACID HEATMAP - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    if best_model_type in pet_generated_peptides and pet_generated_peptides[best_model_type]:
        try:
            plot_generated_peptides_amino_acid_heatmap(
                generated_peptides_list=pet_generated_peptides[best_model_type],
                plastic_type='PET',
                model_type=best_model_type
            )
        except Exception as e:
            print(f"⚠ Üretilen peptidler heatmap hatası: {e}")
    else:
        print(f"⚠ Üretilen peptid bulunamadı, heatmap oluşturulamadı")
    
    print(f"\n{'='*80}")
    print(f"PET İÇİN TAMAMLANDI! En iyi model: {best_model_type.upper()}")
    print(f"{'='*80}")

# ============================================================================
# HÜCRE 2: PP için Ablation Study ve Final Eğitim
# ============================================================================

    if 'PP' in PLASTICS_TO_RUN:
        print("\n" + "="*80)
        print("HÜCRE 2: PP PLASTİK TİPİ İÇİN ABLATION STUDY VE FINAL EĞİTİM")
        print("="*80)

    plastic_type = 'PP'
    model_types = [m for m in MODEL_TYPES_TO_RUN]  # Filtrelenmiş model listesi

    # ADIM 1: Tüm modellerin ablation study'lerini yap (50 epoch)
    print(f"\n{'='*80}")
    print("ADIM 1: TÜM MODELLER İÇİN ABLATION STUDY (50 EPOCH)")
    print(f"{'='*80}")
    
    all_ablation_results = {}  # Tüm modellerin sonuçlarını topla
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"ABLATION: {model_type.upper()} - {plastic_type}")
        print(f"{'='*80}")
        
        # Ablation study (50 epoch)
        best_params, ablation_df = run_ablation_study(
            model_type=model_type,
            plastic_type=plastic_type,
            data_dir=DATA_DIR,
            ablation_epochs=50,
            seed=42
        )
        
        # Sonuçları topla
        all_ablation_results[model_type] = {
            'best_params': best_params,
            'ablation_df': ablation_df,
            'best_val_r2': best_params.get('best_val_r2', -float('inf')),
            'best_test_r2': best_params.get('best_test_r2', -float('inf'))
        }
        
        print(f"✓ {model_type.upper()} tamamlandı - Val R²: {all_ablation_results[model_type]['best_val_r2']:.6f}, Test R²: {all_ablation_results[model_type]['best_test_r2']:.6f}")
    
    # ADIM 2: En iyi model+parametre kombinasyonunu bul (Val R²'ye göre)
    print(f"\n{'='*80}")
    print("ADIM 2: EN İYİ MODEL VE PARAMETRE SEÇİMİ")
    print(f"{'='*80}")
    
    best_model_type = None
    best_val_r2 = -float('inf')
    best_params_overall = None
    
    for model_type, results in all_ablation_results.items():
        val_r2 = results['best_val_r2']
        test_r2 = results['best_test_r2']
        print(f"  {model_type.upper()}: Val R² = {val_r2:.6f}, Test R² = {test_r2:.6f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_type = model_type
            best_params_overall = results['best_params']
    
    print(f"\n🏆 EN İYİ MODEL: {best_model_type.upper()}")
    print(f"   Val R²: {best_val_r2:.6f}")
    print(f"   Test R²: {best_params_overall.get('best_test_r2', 'N/A')}")
    print(f"   Parametreler:")
    for key, value in best_params_overall.items():
        if key not in ['score_mean', 'score_std', 'best_val_r2', 'best_test_r2']:
            print(f"     {key}: {value}")
    
    # ADIM 2.5: Tüm modellerin karşılaştırma tablosu ve heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.5: TÜM MODELLER KARŞILAŞTIRMA TABLOSU VE HEATMAP")
    print(f"{'='*80}")
    
    comparison_df = create_comparison_table_and_heatmap(
        all_ablation_results, plastic_type, best_model_type
    )
    
    # ADIM 2.6: Amino Acid Probability x Mass Heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.6: AMINO ACID PROBABILITY X MASS HEATMAP")
    print(f"{'='*80}")
    
    plot_amino_acid_probability_mass_heatmap(plastic_type='PP', data_dir=DATA_DIR)
    
    # ADIM 3: Sadece en iyi model için final eğitim (250 epoch)
    print(f"\n{'='*80}")
    print(f"ADIM 3: FINAL EĞİTİM - {best_model_type.upper()} (250 EPOCH)")
    print(f"{'='*80}")
    
    metrics, train_losses, val_losses = train_final_model(
        model_type=best_model_type,
        plastic_type=plastic_type,
        best_params=best_params_overall,
        data_dir=DATA_DIR,
        final_epochs=250,
        seed=42
    )
    
    # Sonuçları kaydet (sadece en iyi model)
    pp_results[best_model_type] = {
        'best_params': best_params_overall,
        'ablation_df': all_ablation_results[best_model_type]['ablation_df'],
        'final_metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'is_best_model': True
    }
    
    # Diğer modellerin ablation sonuçlarını da kaydet (karşılaştırma için)
    for model_type, results in all_ablation_results.items():
        if model_type != best_model_type:
            pp_results[model_type] = {
                'best_params': results['best_params'],
                'ablation_df': results['ablation_df'],
                'is_best_model': False
            }
    
    # ADIM 4: Sadece en iyi model ile yeni peptid üretimi
    print(f"\n{'='*80}")
    print(f"ADIM 4: YENİ PEPTİD ÜRETİMİ - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    pp_generated_peptides = {}
    try:
        generated = generate_peptides_with_best_model(
            plastic_type='PP',
            model_type=best_model_type,
            best_params=best_params_overall,
            n_samples=10,
            n_optimizations=3,
            max_iterations=2000
        )
        if generated:
            pp_generated_peptides[best_model_type] = generated
    except Exception as e:
        print(f"✗ PP - {best_model_type} için peptid üretim hatası: {e}")
    
    # ADIM 4.5: Üretilen peptidler için amino acid probability x mass heatmap
    print(f"\n{'='*80}")
    print(f"ADIM 4.5: ÜRETİLEN PEPTİDLER İÇİN AMINO ACID HEATMAP - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    if best_model_type in pp_generated_peptides and pp_generated_peptides[best_model_type]:
        try:
            plot_generated_peptides_amino_acid_heatmap(
                generated_peptides_list=pp_generated_peptides[best_model_type],
                plastic_type='PP',
                model_type=best_model_type
            )
        except Exception as e:
            print(f"⚠ Üretilen peptidler heatmap hatası: {e}")
    else:
        print(f"⚠ Üretilen peptid bulunamadı, heatmap oluşturulamadı")
    
    print(f"\n{'='*80}")
    print(f"PP İÇİN TAMAMLANDI! En iyi model: {best_model_type.upper()}")
    print(f"{'='*80}")

# ============================================================================
# HÜCRE 3: PE için Ablation Study ve Final Eğitim
# ============================================================================

    if 'PE' in PLASTICS_TO_RUN:
        print("\n" + "="*80)
        print("HÜCRE 3: PE PLASTİK TİPİ İÇİN ABLATION STUDY VE FINAL EĞİTİM")
        print("="*80)

    plastic_type = 'PE'
    model_types = [m for m in MODEL_TYPES_TO_RUN]  # Filtrelenmiş model listesi

    # ADIM 1: Tüm modellerin ablation study'lerini yap (50 epoch)
    print(f"\n{'='*80}")
    print("ADIM 1: TÜM MODELLER İÇİN ABLATION STUDY (50 EPOCH)")
    print(f"{'='*80}")
    
    all_ablation_results = {}  # Tüm modellerin sonuçlarını topla
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"ABLATION: {model_type.upper()} - {plastic_type}")
        print(f"{'='*80}")
        
        # Ablation study (50 epoch)
        best_params, ablation_df = run_ablation_study(
            model_type=model_type,
            plastic_type=plastic_type,
            data_dir=DATA_DIR,
            ablation_epochs=50,
            seed=42
        )
        
        # Sonuçları topla
        all_ablation_results[model_type] = {
            'best_params': best_params,
            'ablation_df': ablation_df,
            'best_val_r2': best_params.get('best_val_r2', -float('inf')),
            'best_test_r2': best_params.get('best_test_r2', -float('inf'))
        }
        
        print(f"✓ {model_type.upper()} tamamlandı - Val R²: {all_ablation_results[model_type]['best_val_r2']:.6f}, Test R²: {all_ablation_results[model_type]['best_test_r2']:.6f}")
    
    # ADIM 2: En iyi model+parametre kombinasyonunu bul (Val R²'ye göre)
    print(f"\n{'='*80}")
    print("ADIM 2: EN İYİ MODEL VE PARAMETRE SEÇİMİ")
    print(f"{'='*80}")
    
    best_model_type = None
    best_val_r2 = -float('inf')
    best_params_overall = None
    
    for model_type, results in all_ablation_results.items():
        val_r2 = results['best_val_r2']
        test_r2 = results['best_test_r2']
        print(f"  {model_type.upper()}: Val R² = {val_r2:.6f}, Test R² = {test_r2:.6f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_type = model_type
            best_params_overall = results['best_params']
    
    print(f"\n🏆 EN İYİ MODEL: {best_model_type.upper()}")
    print(f"   Val R²: {best_val_r2:.6f}")
    print(f"   Test R²: {best_params_overall.get('best_test_r2', 'N/A')}")
    print(f"   Parametreler:")
    for key, value in best_params_overall.items():
        if key not in ['score_mean', 'score_std', 'best_val_r2', 'best_test_r2']:
            print(f"     {key}: {value}")
    
    # ADIM 2.5: Tüm modellerin karşılaştırma tablosu ve heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.5: TÜM MODELLER KARŞILAŞTIRMA TABLOSU VE HEATMAP")
    print(f"{'='*80}")
    
    comparison_df = create_comparison_table_and_heatmap(
        all_ablation_results, plastic_type, best_model_type
    )
    
    # ADIM 2.6: Amino Acid Probability x Mass Heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.6: AMINO ACID PROBABILITY X MASS HEATMAP")
    print(f"{'='*80}")
    
    plot_amino_acid_probability_mass_heatmap(plastic_type='PE', data_dir=DATA_DIR)
    
    # ADIM 3: Sadece en iyi model için final eğitim (250 epoch)
    print(f"\n{'='*80}")
    print(f"ADIM 3: FINAL EĞİTİM - {best_model_type.upper()} (250 EPOCH)")
    print(f"{'='*80}")
    
    metrics, train_losses, val_losses = train_final_model(
        model_type=best_model_type,
        plastic_type=plastic_type,
        best_params=best_params_overall,
        data_dir=DATA_DIR,
        final_epochs=250,
        seed=42
    )
    
    # Sonuçları kaydet (sadece en iyi model)
    pe_results[best_model_type] = {
        'best_params': best_params_overall,
        'ablation_df': all_ablation_results[best_model_type]['ablation_df'],
        'final_metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'is_best_model': True
    }
    
    # Diğer modellerin ablation sonuçlarını da kaydet (karşılaştırma için)
    for model_type, results in all_ablation_results.items():
        if model_type != best_model_type:
            pe_results[model_type] = {
                'best_params': results['best_params'],
                'ablation_df': results['ablation_df'],
                'is_best_model': False
            }
    
    # ADIM 4: Sadece en iyi model ile yeni peptid üretimi
    print(f"\n{'='*80}")
    print(f"ADIM 4: YENİ PEPTİD ÜRETİMİ - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    pe_generated_peptides = {}
    try:
        generated = generate_peptides_with_best_model(
            plastic_type='PE',
            model_type=best_model_type,
            best_params=best_params_overall,
            n_samples=10,
            n_optimizations=3,
            max_iterations=2000
        )
        if generated:
            pe_generated_peptides[best_model_type] = generated
    except Exception as e:
        print(f"✗ PE - {best_model_type} için peptid üretim hatası: {e}")
    
    # ADIM 4.5: Üretilen peptidler için amino acid probability x mass heatmap
    print(f"\n{'='*80}")
    print(f"ADIM 4.5: ÜRETİLEN PEPTİDLER İÇİN AMINO ACID HEATMAP - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    if best_model_type in pe_generated_peptides and pe_generated_peptides[best_model_type]:
        try:
            plot_generated_peptides_amino_acid_heatmap(
                generated_peptides_list=pe_generated_peptides[best_model_type],
                plastic_type='PE',
                model_type=best_model_type
            )
        except Exception as e:
            print(f"⚠ Üretilen peptidler heatmap hatası: {e}")
    else:
        print(f"⚠ Üretilen peptid bulunamadı, heatmap oluşturulamadı")
    
    print(f"\n{'='*80}")
    print(f"PE İÇİN TAMAMLANDI! En iyi model: {best_model_type.upper()}")
    print(f"{'='*80}")

# ============================================================================
# HÜCRE 4: PVC için Ablation Study ve Final Eğitim
# ============================================================================

    if 'PVC' in PLASTICS_TO_RUN:
        print("\n" + "="*80)
        print("HÜCRE 4: PVC PLASTİK TİPİ İÇİN ABLATION STUDY VE FINAL EĞİTİM")
        print("="*80)

    plastic_type = 'PVC'
    model_types = [m for m in MODEL_TYPES_TO_RUN]  # Filtrelenmiş model listesi

    # ADIM 1: Tüm modellerin ablation study'lerini yap (50 epoch)
    print(f"\n{'='*80}")
    print("ADIM 1: TÜM MODELLER İÇİN ABLATION STUDY (50 EPOCH)")
    print(f"{'='*80}")
    
    all_ablation_results = {}  # Tüm modellerin sonuçlarını topla
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"ABLATION: {model_type.upper()} - {plastic_type}")
        print(f"{'='*80}")
        
        # Ablation study (50 epoch)
        best_params, ablation_df = run_ablation_study(
            model_type=model_type,
            plastic_type=plastic_type,
            data_dir=DATA_DIR,
            ablation_epochs=50,
            seed=42
        )
        
        # Sonuçları topla
        all_ablation_results[model_type] = {
            'best_params': best_params,
            'ablation_df': ablation_df,
            'best_val_r2': best_params.get('best_val_r2', -float('inf')),
            'best_test_r2': best_params.get('best_test_r2', -float('inf'))
        }
        
        print(f"✓ {model_type.upper()} tamamlandı - Val R²: {all_ablation_results[model_type]['best_val_r2']:.6f}, Test R²: {all_ablation_results[model_type]['best_test_r2']:.6f}")
    
    # ADIM 2: En iyi model+parametre kombinasyonunu bul (Val R²'ye göre)
    print(f"\n{'='*80}")
    print("ADIM 2: EN İYİ MODEL VE PARAMETRE SEÇİMİ")
    print(f"{'='*80}")
    
    best_model_type = None
    best_val_r2 = -float('inf')
    best_params_overall = None
    
    for model_type, results in all_ablation_results.items():
        val_r2 = results['best_val_r2']
        test_r2 = results['best_test_r2']
        print(f"  {model_type.upper()}: Val R² = {val_r2:.6f}, Test R² = {test_r2:.6f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_type = model_type
            best_params_overall = results['best_params']
    
    print(f"\n🏆 EN İYİ MODEL: {best_model_type.upper()}")
    print(f"   Val R²: {best_val_r2:.6f}")
    print(f"   Test R²: {best_params_overall.get('best_test_r2', 'N/A')}")
    print(f"   Parametreler:")
    for key, value in best_params_overall.items():
        if key not in ['score_mean', 'score_std', 'best_val_r2', 'best_test_r2']:
            print(f"     {key}: {value}")
    
    # ADIM 2.5: Tüm modellerin karşılaştırma tablosu ve heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.5: TÜM MODELLER KARŞILAŞTIRMA TABLOSU VE HEATMAP")
    print(f"{'='*80}")
    
    comparison_df = create_comparison_table_and_heatmap(
        all_ablation_results, plastic_type, best_model_type
    )
    
    # ADIM 2.6: Amino Acid Probability x Mass Heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.6: AMINO ACID PROBABILITY X MASS HEATMAP")
    print(f"{'='*80}")
    
    plot_amino_acid_probability_mass_heatmap(plastic_type='PVC', data_dir=DATA_DIR)
    
    # ADIM 3: Sadece en iyi model için final eğitim (250 epoch)
    print(f"\n{'='*80}")
    print(f"ADIM 3: FINAL EĞİTİM - {best_model_type.upper()} (250 EPOCH)")
    print(f"{'='*80}")
    
    metrics, train_losses, val_losses = train_final_model(
        model_type=best_model_type,
        plastic_type=plastic_type,
        best_params=best_params_overall,
        data_dir=DATA_DIR,
        final_epochs=250,
        seed=42
    )
    
    # Sonuçları kaydet (sadece en iyi model)
    pvc_results[best_model_type] = {
        'best_params': best_params_overall,
        'ablation_df': all_ablation_results[best_model_type]['ablation_df'],
        'final_metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'is_best_model': True
    }
    
    # Diğer modellerin ablation sonuçlarını da kaydet (karşılaştırma için)
    for model_type, results in all_ablation_results.items():
        if model_type != best_model_type:
            pvc_results[model_type] = {
                'best_params': results['best_params'],
                'ablation_df': results['ablation_df'],
                'is_best_model': False
            }
    
    # ADIM 4: Sadece en iyi model ile yeni peptid üretimi
    print(f"\n{'='*80}")
    print(f"ADIM 4: YENİ PEPTİD ÜRETİMİ - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    pvc_generated_peptides = {}
    try:
        generated = generate_peptides_with_best_model(
            plastic_type='PVC',
            model_type=best_model_type,
            best_params=best_params_overall,
            n_samples=10,
            n_optimizations=3,
            max_iterations=2000
        )
        if generated:
            pvc_generated_peptides[best_model_type] = generated
    except Exception as e:
        print(f"✗ PVC - {best_model_type} için peptid üretim hatası: {e}")
    
    # ADIM 4.5: Üretilen peptidler için amino acid probability x mass heatmap
    print(f"\n{'='*80}")
    print(f"ADIM 4.5: ÜRETİLEN PEPTİDLER İÇİN AMINO ACID HEATMAP - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    if best_model_type in pvc_generated_peptides and pvc_generated_peptides[best_model_type]:
        try:
            plot_generated_peptides_amino_acid_heatmap(
                generated_peptides_list=pvc_generated_peptides[best_model_type],
                plastic_type='PVC',
                model_type=best_model_type
            )
        except Exception as e:
            print(f"⚠ Üretilen peptidler heatmap hatası: {e}")
    else:
        print(f"⚠ Üretilen peptid bulunamadı, heatmap oluşturulamadı")
    
    print(f"\n{'='*80}")
    print(f"PVC İÇİN TAMAMLANDI! En iyi model: {best_model_type.upper()}")
    print(f"{'='*80}")

# ============================================================================
# HÜCRE 5: Nylon için Ablation Study ve Final Eğitim
# ============================================================================

# HÜCRE 5: Nylon için (Windows yerel ortam - direkt çalışır)
    if 'Nylon' in PLASTICS_TO_RUN:
        print("\n" + "="*80)
        print("HÜCRE 5: NYLON PLASTİK TİPİ İÇİN ABLATION STUDY VE FINAL EĞİTİM")
        print("="*80)

    plastic_type = 'Nylon'
    model_types = [m for m in MODEL_TYPES_TO_RUN]  # Filtrelenmiş model listesi

    # ADIM 1: Tüm modellerin ablation study'lerini yap (50 epoch)
    print(f"\n{'='*80}")
    print("ADIM 1: TÜM MODELLER İÇİN ABLATION STUDY (50 EPOCH)")
    print(f"{'='*80}")
    
    all_ablation_results = {}  # Tüm modellerin sonuçlarını topla
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"ABLATION: {model_type.upper()} - {plastic_type}")
        print(f"{'='*80}")
        
        # Ablation study (50 epoch)
        best_params, ablation_df = run_ablation_study(
            model_type=model_type,
            plastic_type=plastic_type,
            data_dir=DATA_DIR,
            ablation_epochs=50,
            seed=42
        )
        
        # Sonuçları topla
        all_ablation_results[model_type] = {
            'best_params': best_params,
            'ablation_df': ablation_df,
            'best_val_r2': best_params.get('best_val_r2', -float('inf')),
            'best_test_r2': best_params.get('best_test_r2', -float('inf'))
        }
        
        print(f"✓ {model_type.upper()} tamamlandı - Val R²: {all_ablation_results[model_type]['best_val_r2']:.6f}, Test R²: {all_ablation_results[model_type]['best_test_r2']:.6f}")
    
    # ADIM 2: En iyi model+parametre kombinasyonunu bul (Val R²'ye göre)
    print(f"\n{'='*80}")
    print("ADIM 2: EN İYİ MODEL VE PARAMETRE SEÇİMİ")
    print(f"{'='*80}")
    
    best_model_type = None
    best_val_r2 = -float('inf')
    best_params_overall = None
    
    for model_type, results in all_ablation_results.items():
        val_r2 = results['best_val_r2']
        test_r2 = results['best_test_r2']
        print(f"  {model_type.upper()}: Val R² = {val_r2:.6f}, Test R² = {test_r2:.6f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_type = model_type
            best_params_overall = results['best_params']
    
    print(f"\n🏆 EN İYİ MODEL: {best_model_type.upper()}")
    print(f"   Val R²: {best_val_r2:.6f}")
    print(f"   Test R²: {best_params_overall.get('best_test_r2', 'N/A')}")
    print(f"   Parametreler:")
    for key, value in best_params_overall.items():
        if key not in ['score_mean', 'score_std', 'best_val_r2', 'best_test_r2']:
            print(f"     {key}: {value}")
    
    # ADIM 2.5: Tüm modellerin karşılaştırma tablosu ve heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.5: TÜM MODELLER KARŞILAŞTIRMA TABLOSU VE HEATMAP")
    print(f"{'='*80}")
    
    comparison_df = create_comparison_table_and_heatmap(
        all_ablation_results, plastic_type, best_model_type
    )
    
    # ADIM 2.6: Amino Acid Probability x Mass Heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.6: AMINO ACID PROBABILITY X MASS HEATMAP")
    print(f"{'='*80}")
    
    plot_amino_acid_probability_mass_heatmap(plastic_type='Nylon', data_dir=DATA_DIR)
    
    # ADIM 3: Sadece en iyi model için final eğitim (250 epoch)
    print(f"\n{'='*80}")
    print(f"ADIM 3: FINAL EĞİTİM - {best_model_type.upper()} (250 EPOCH)")
    print(f"{'='*80}")
    
    metrics, train_losses, val_losses = train_final_model(
        model_type=best_model_type,
        plastic_type=plastic_type,
        best_params=best_params_overall,
        data_dir=DATA_DIR,
        final_epochs=250,
        seed=42
    )
    
    # Sonuçları kaydet (sadece en iyi model)
    nylon_results[best_model_type] = {
        'best_params': best_params_overall,
        'ablation_df': all_ablation_results[best_model_type]['ablation_df'],
        'final_metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'is_best_model': True
    }
    
    # Diğer modellerin ablation sonuçlarını da kaydet (karşılaştırma için)
    for model_type, results in all_ablation_results.items():
        if model_type != best_model_type:
            nylon_results[model_type] = {
                'best_params': results['best_params'],
                'ablation_df': results['ablation_df'],
                'is_best_model': False
            }
    
    # ADIM 4: Sadece en iyi model ile yeni peptid üretimi
    print(f"\n{'='*80}")
    print(f"ADIM 4: YENİ PEPTİD ÜRETİMİ - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    nylon_generated_peptides = {}
    try:
        generated = generate_peptides_with_best_model(
            plastic_type='Nylon',
            model_type=best_model_type,
            best_params=best_params_overall,
            n_samples=10,
            n_optimizations=3,
            max_iterations=2000
        )
        if generated:
            nylon_generated_peptides[best_model_type] = generated
    except Exception as e:
        print(f"✗ Nylon - {best_model_type} için peptid üretim hatası: {e}")
    
    # ADIM 4.5: Üretilen peptidler için amino acid probability x mass heatmap
    print(f"\n{'='*80}")
    print(f"ADIM 4.5: ÜRETİLEN PEPTİDLER İÇİN AMINO ACID HEATMAP - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    if best_model_type in nylon_generated_peptides and nylon_generated_peptides[best_model_type]:
        try:
            plot_generated_peptides_amino_acid_heatmap(
                generated_peptides_list=nylon_generated_peptides[best_model_type],
                plastic_type='Nylon',
                model_type=best_model_type
            )
        except Exception as e:
            print(f"⚠ Üretilen peptidler heatmap hatası: {e}")
    else:
        print(f"⚠ Üretilen peptid bulunamadı, heatmap oluşturulamadı")
    
    print(f"\n{'='*80}")
    print(f"NYLON İÇİN TAMAMLANDI! En iyi model: {best_model_type.upper()}")
    print(f"{'='*80}")

# ============================================================================
# HÜCRE 6: PMMA için Ablation Study ve Final Eğitim
# ============================================================================

    if 'PMMA' in PLASTICS_TO_RUN:
        print("\n" + "="*80)
        print("HÜCRE 6: PMMA PLASTİK TİPİ İÇİN ABLATION STUDY VE FINAL EĞİTİM")
        print("="*80)

    plastic_type = 'PMMA'
    model_types = [m for m in MODEL_TYPES_TO_RUN]

    # ADIM 1: Tüm modellerin ablation study'lerini yap (50 epoch)
    print(f"\n{'='*80}")
    print("ADIM 1: TÜM MODELLER İÇİN ABLATION STUDY (50 EPOCH)")
    print(f"{'='*80}")
    
    all_ablation_results = {}  # Tüm modellerin sonuçlarını topla
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"ABLATION: {model_type.upper()} - {plastic_type}")
        print(f"{'='*80}")
        
        # Ablation study (50 epoch)
        best_params, ablation_df = run_ablation_study(
            model_type=model_type,
            plastic_type=plastic_type,
            data_dir=DATA_DIR,
            ablation_epochs=50,
            seed=42
        )
        
        # Sonuçları topla
        all_ablation_results[model_type] = {
            'best_params': best_params,
            'ablation_df': ablation_df,
            'best_val_r2': best_params.get('best_val_r2', -float('inf')),
            'best_test_r2': best_params.get('best_test_r2', -float('inf'))
        }
        
        print(f"✓ {model_type.upper()} tamamlandı - Val R²: {all_ablation_results[model_type]['best_val_r2']:.6f}, Test R²: {all_ablation_results[model_type]['best_test_r2']:.6f}")
    
    # ADIM 2: En iyi model+parametre kombinasyonunu bul (Val R²'ye göre)
    print(f"\n{'='*80}")
    print("ADIM 2: EN İYİ MODEL VE PARAMETRE SEÇİMİ")
    print(f"{'='*80}")
    
    best_model_type = None
    best_val_r2 = -float('inf')
    best_params_overall = None
    
    for model_type, results in all_ablation_results.items():
        val_r2 = results['best_val_r2']
        test_r2 = results['best_test_r2']
        print(f"  {model_type.upper()}: Val R² = {val_r2:.6f}, Test R² = {test_r2:.6f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_type = model_type
            best_params_overall = results['best_params']
    
    print(f"\n🏆 EN İYİ MODEL: {best_model_type.upper()}")
    print(f"   Val R²: {best_val_r2:.6f}")
    print(f"   Test R²: {best_params_overall.get('best_test_r2', 'N/A')}")
    print(f"   Parametreler:")
    for key, value in best_params_overall.items():
        if key not in ['score_mean', 'score_std', 'best_val_r2', 'best_test_r2']:
            print(f"     {key}: {value}")
    
    # ADIM 2.5: Tüm modellerin karşılaştırma tablosu ve heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.5: TÜM MODELLER KARŞILAŞTIRMA TABLOSU VE HEATMAP")
    print(f"{'='*80}")
    
    comparison_df = create_comparison_table_and_heatmap(
        all_ablation_results, plastic_type, best_model_type
    )
    
    # ADIM 2.6: Amino Acid Probability x Mass Heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.6: AMINO ACID PROBABILITY X MASS HEATMAP")
    print(f"{'='*80}")
    
    plot_amino_acid_probability_mass_heatmap(plastic_type='PMMA', data_dir=DATA_DIR)
    
    # ADIM 3: Sadece en iyi model için final eğitim (250 epoch)
    print(f"\n{'='*80}")
    print(f"ADIM 3: FINAL EĞİTİM - {best_model_type.upper()} (250 EPOCH)")
    print(f"{'='*80}")
    
    metrics, train_losses, val_losses = train_final_model(
        model_type=best_model_type,
        plastic_type=plastic_type,
        best_params=best_params_overall,
        data_dir=DATA_DIR,
        final_epochs=250,
        seed=42
    )
    
    # Sonuçları kaydet (sadece en iyi model)
    pmma_results[best_model_type] = {
        'best_params': best_params_overall,
        'ablation_df': all_ablation_results[best_model_type]['ablation_df'],
        'final_metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'is_best_model': True
    }
    
    # Diğer modellerin ablation sonuçlarını da kaydet (karşılaştırma için)
    for model_type, results in all_ablation_results.items():
        if model_type != best_model_type:
            pmma_results[model_type] = {
                'best_params': results['best_params'],
                'ablation_df': results['ablation_df'],
                'is_best_model': False
            }
    
    # ADIM 4: Sadece en iyi model ile yeni peptid üretimi
    print(f"\n{'='*80}")
    print(f"ADIM 4: YENİ PEPTİD ÜRETİMİ - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    pmma_generated_peptides = {}
    try:
        generated = generate_peptides_with_best_model(
            plastic_type='PMMA',
            model_type=best_model_type,
            best_params=best_params_overall,
            n_samples=10,
            n_optimizations=3,
            max_iterations=2000
        )
        if generated:
            pmma_generated_peptides[best_model_type] = generated
    except Exception as e:
        print(f"✗ PMMA - {best_model_type} için peptid üretim hatası: {e}")
    
    # ADIM 4.5: Üretilen peptidler için amino acid probability x mass heatmap
    print(f"\n{'='*80}")
    print(f"ADIM 4.5: ÜRETİLEN PEPTİDLER İÇİN AMINO ACID HEATMAP - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    if best_model_type in pmma_generated_peptides and pmma_generated_peptides[best_model_type]:
        try:
            plot_generated_peptides_amino_acid_heatmap(
                generated_peptides_list=pmma_generated_peptides[best_model_type],
                plastic_type='PMMA',
                model_type=best_model_type
            )
        except Exception as e:
            print(f"⚠ Üretilen peptidler heatmap hatası: {e}")
    else:
        print(f"⚠ Üretilen peptid bulunamadı, heatmap oluşturulamadı")
    
    print(f"\n{'='*80}")
    print(f"PMMA İÇİN TAMAMLANDI! En iyi model: {best_model_type.upper()}")
    print(f"{'='*80}")

# ============================================================================
# HÜCRE 7: PS için Ablation Study ve Final Eğitim
# ============================================================================

    if 'PS' in PLASTICS_TO_RUN:
        print("\n" + "="*80)
        print("HÜCRE 7: PS PLASTİK TİPİ İÇİN ABLATION STUDY VE FINAL EĞİTİM")
        print("="*80)

    plastic_type = 'PS'
    model_types = [m for m in MODEL_TYPES_TO_RUN]

    # ADIM 1: Tüm modellerin ablation study'lerini yap (50 epoch)
    print(f"\n{'='*80}")
    print("ADIM 1: TÜM MODELLER İÇİN ABLATION STUDY (50 EPOCH)")
    print(f"{'='*80}")
    
    all_ablation_results = {}  # Tüm modellerin sonuçlarını topla
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"ABLATION: {model_type.upper()} - {plastic_type}")
        print(f"{'='*80}")
        
        # Ablation study (50 epoch)
        best_params, ablation_df = run_ablation_study(
            model_type=model_type,
            plastic_type=plastic_type,
            data_dir=DATA_DIR,
            ablation_epochs=50,
            seed=42
        )
        
        # Sonuçları topla
        all_ablation_results[model_type] = {
            'best_params': best_params,
            'ablation_df': ablation_df,
            'best_val_r2': best_params.get('best_val_r2', -float('inf')),
            'best_test_r2': best_params.get('best_test_r2', -float('inf'))
        }
        
        print(f"✓ {model_type.upper()} tamamlandı - Val R²: {all_ablation_results[model_type]['best_val_r2']:.6f}, Test R²: {all_ablation_results[model_type]['best_test_r2']:.6f}")
    
    # ADIM 2: En iyi model+parametre kombinasyonunu bul (Val R²'ye göre)
    print(f"\n{'='*80}")
    print("ADIM 2: EN İYİ MODEL VE PARAMETRE SEÇİMİ")
    print(f"{'='*80}")
    
    best_model_type = None
    best_val_r2 = -float('inf')
    best_params_overall = None
    
    for model_type, results in all_ablation_results.items():
        val_r2 = results['best_val_r2']
        test_r2 = results['best_test_r2']
        print(f"  {model_type.upper()}: Val R² = {val_r2:.6f}, Test R² = {test_r2:.6f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_type = model_type
            best_params_overall = results['best_params']
    
    print(f"\n🏆 EN İYİ MODEL: {best_model_type.upper()}")
    print(f"   Val R²: {best_val_r2:.6f}")
    print(f"   Test R²: {best_params_overall.get('best_test_r2', 'N/A')}")
    print(f"   Parametreler:")
    for key, value in best_params_overall.items():
        if key not in ['score_mean', 'score_std', 'best_val_r2', 'best_test_r2']:
            print(f"     {key}: {value}")
    
    # ADIM 2.5: Tüm modellerin karşılaştırma tablosu ve heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.5: TÜM MODELLER KARŞILAŞTIRMA TABLOSU VE HEATMAP")
    print(f"{'='*80}")
    
    comparison_df = create_comparison_table_and_heatmap(
        all_ablation_results, plastic_type, best_model_type
    )
    
    # ADIM 2.6: Amino Acid Probability x Mass Heatmap
    print(f"\n{'='*80}")
    print("ADIM 2.6: AMINO ACID PROBABILITY X MASS HEATMAP")
    print(f"{'='*80}")
    
    plot_amino_acid_probability_mass_heatmap(plastic_type='PS', data_dir=DATA_DIR)
    
    # ADIM 3: Sadece en iyi model için final eğitim (250 epoch)
    print(f"\n{'='*80}")
    print(f"ADIM 3: FINAL EĞİTİM - {best_model_type.upper()} (250 EPOCH)")
    print(f"{'='*80}")
    
    metrics, train_losses, val_losses = train_final_model(
        model_type=best_model_type,
        plastic_type=plastic_type,
        best_params=best_params_overall,
        data_dir=DATA_DIR,
        final_epochs=250,
        seed=42
    )
    
    # Sonuçları kaydet (sadece en iyi model)
    ps_results[best_model_type] = {
        'best_params': best_params_overall,
        'ablation_df': all_ablation_results[best_model_type]['ablation_df'],
        'final_metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'is_best_model': True
    }
    
    # Diğer modellerin ablation sonuçlarını da kaydet (karşılaştırma için)
    for model_type, results in all_ablation_results.items():
        if model_type != best_model_type:
            ps_results[model_type] = {
                'best_params': results['best_params'],
                'ablation_df': results['ablation_df'],
                'is_best_model': False
            }
    
    # ADIM 4: Sadece en iyi model ile yeni peptid üretimi
    print(f"\n{'='*80}")
    print(f"ADIM 4: YENİ PEPTİD ÜRETİMİ - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    ps_generated_peptides = {}
    try:
        generated = generate_peptides_with_best_model(
            plastic_type='PS',
            model_type=best_model_type,
            best_params=best_params_overall,
            n_samples=10,
            n_optimizations=3,
            max_iterations=2000
        )
        if generated:
            ps_generated_peptides[best_model_type] = generated
    except Exception as e:
        print(f"✗ PS - {best_model_type} için peptid üretim hatası: {e}")
    
    # ADIM 4.5: Üretilen peptidler için amino acid probability x mass heatmap
    print(f"\n{'='*80}")
    print(f"ADIM 4.5: ÜRETİLEN PEPTİDLER İÇİN AMINO ACID HEATMAP - {best_model_type.upper()}")
    print(f"{'='*80}")
    
    if best_model_type in ps_generated_peptides and ps_generated_peptides[best_model_type]:
        try:
            plot_generated_peptides_amino_acid_heatmap(
                generated_peptides_list=ps_generated_peptides[best_model_type],
                plastic_type='PS',
                model_type=best_model_type
            )
        except Exception as e:
            print(f"⚠ Üretilen peptidler heatmap hatası: {e}")
    else:
        print(f"⚠ Üretilen peptid bulunamadı, heatmap oluşturulamadı")
    
    print(f"\n{'='*80}")
    print(f"PS İÇİN TAMAMLANDI! En iyi model: {best_model_type.upper()}")
    print(f"{'='*80}")

# ============================================================================
# HÜCRE 8: Özet ve Karşılaştırma
# ============================================================================

    # HÜCRE 6: Özet ve Karşılaştırma (Windows yerel ortam - direkt çalışır)
    print("\n" + "="*80)
    print("HÜCRE 6: TÜM SONUÇLARIN ÖZETİ VE KARŞILAŞTIRMASI")
    print("="*80)

    # Tüm sonuçları topla (sadece çalıştırılmış plastikler için)
    all_results_summary = []

    # Plastik tipi ve result dict eşleştirmesi
    plastic_results_map = {
        'PET': pet_results,
        'PP': pp_results,
        'PE': pe_results,
        'PVC': pvc_results,
        'Nylon': nylon_results,
        'PMMA': pmma_results,
        'PS': ps_results
    }

    # Sadece çalıştırılmış plastikler için sonuçları topla
    for plastic_type in PLASTICS_TO_RUN:
        results_dict = plastic_results_map.get(plastic_type, {})

        # Eğer bu plastik için sonuç varsa işle
        if results_dict:
            for model_type, result_data in results_dict.items():
                try:
                    all_results_summary.append({
                        'Plastic_Type': plastic_type,
                        'Model_Type': model_type,
                        'Test_R2': result_data['final_metrics']['test']['r2'],
                        'Test_MAE': result_data['final_metrics']['test']['mae'],
                        'Test_RMSE': result_data['final_metrics']['test']['rmse'],
                        **{k: v for k, v in result_data['best_params'].items()
                           if k not in ['score_mean', 'score_std', 'best_val_r2', 'best_test_r2']}
                    })
                except KeyError as e:
                    print(f"⚠ UYARI: {plastic_type} - {model_type} için eksik veri: {e}")
                    continue
        else:
            print(f"⚠ {plastic_type} için sonuç bulunamadı (çalıştırılmamış olabilir)")

    # Sonuçları kontrol et
    if not all_results_summary:
        print("\n⚠ UYARI: Hiç sonuç bulunamadı!")
        print("   Lütfen önce plastik/model eğitimlerini çalıştırdığınızdan emin olun.")
        print(f"   Çalıştırılan plastikler: {PLASTICS_TO_RUN}")
        print(f"   Çalıştırılan modeller: {MODEL_TYPES_TO_RUN}")
    else:
        summary_df = pd.DataFrame(all_results_summary)
        summary_path = os.path.join(ABLATION_TABLES_DIR, 'ablation_final_summary.csv')
        summary_df.to_csv(summary_path, index=False)

        print(f"\n✓ Özet tablosu kaydedildi: {summary_path}")
        print(f"   Toplam {len(summary_df)} sonuç kaydedildi")
        
        print("\n📊 Model Tipine Göre Ortalama R² Skorları:")
        model_avg = summary_df.groupby('Model_Type')['Test_R2'].mean().sort_values(ascending=False)
        print(model_avg)
        
        print("\n📊 Plastik Tipine Göre Ortalama R² Skorları:")
        plastic_avg = summary_df.groupby('Plastic_Type')['Test_R2'].mean().sort_values(ascending=False)
        print(plastic_avg)

        print("\n🏆 En İyi Kombinasyonlar:")
        best_per_plastic = summary_df.loc[summary_df.groupby('Plastic_Type')['Test_R2'].idxmax()]
        print(best_per_plastic[['Plastic_Type', 'Model_Type', 'Test_R2']].to_string(index=False))
        
        # Genel karşılaştırma tablosu oluştur (tüm plastikler için)
        print(f"\n{'='*80}")
        print("GENEL KARŞILAŞTIRMA TABLOSU (TÜM PLASTİKLER)")
        print(f"{'='*80}")
        
        # Genel tabloyu göster ve kaydet
        display_summary = summary_df.copy()
        display_summary = display_summary[['Plastic_Type', 'Model_Type', 'Test_R2', 'Test_MAE', 'Test_RMSE']]
        print("\n" + display_summary.to_string(index=False))
        
        # Genel HTML tablo (best modelleri vurgula)
        def highlight_best_overall(row):
            best_per_plastic = summary_df.groupby('Plastic_Type')['Test_R2'].max()
            if row['Plastic_Type'] in best_per_plastic.index:
                if row['Test_R2'] == best_per_plastic[row['Plastic_Type']]:
                    return ['background-color: #FFD700; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        styled_summary = summary_df.style.apply(highlight_best_overall, axis=1)
        summary_html_path = os.path.join(ABLATION_TABLES_DIR, 'all_plastics_comparison_table.html')
        styled_summary.to_html(summary_html_path)
        print(f"\n✓ Genel karşılaştırma tablosu kaydedildi: {summary_html_path}")
        
        # Model karşılaştırma grafiklerini çiz
        try:
            plot_model_comparison(all_results_summary)
        except Exception as e:
            print(f"⚠ Grafik oluşturma hatası: {e}")
        
        # Tüm plastikler için amino acid probability x mass heatmap (karşılaştırma)
        print(f"\n{'='*80}")
        print("TÜM PLASTİKLER İÇİN AMINO ACID PROBABILITY X MASS HEATMAP (KARŞILAŞTIRMA)")
        print(f"{'='*80}")
        
        try:
            # Her plastik için heatmap oluştur
            all_plastic_heatmaps = {}
            for plastic_type in PLASTICS_TO_RUN:
                try:
                    probs, aas, masses = plot_amino_acid_probability_mass_heatmap(
                        plastic_type=plastic_type, data_dir=DATA_DIR
                    )
                    if probs is not None:
                        all_plastic_heatmaps[plastic_type] = {'probs': probs, 'aas': aas, 'masses': masses}
                except Exception as e:
                    print(f"⚠ {plastic_type} için amino acid heatmap hatası: {e}")
            
            # Karşılaştırmalı heatmap (tüm plastikler yan yana)
            if all_plastic_heatmaps:
                n_plastics = len(all_plastic_heatmaps)
                fig, axes = plt.subplots(2, (n_plastics + 1) // 2, figsize=(6 * ((n_plastics + 1) // 2), 12))
                if n_plastics == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                for idx, (plastic_type, data) in enumerate(all_plastic_heatmaps.items()):
                    ax = axes[idx]
                    im = ax.imshow(data['probs'].T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
                    ax.set_xticks(range(12))
                    ax.set_xticklabels([f'P{i+1}' for i in range(12)], fontsize=8)
                    ax.set_yticks(range(len(data['aas'])))
                    ax.set_yticklabels([f'{aa}' for aa in data['aas']], fontsize=7)
                    ax.set_title(f'{plastic_type}', fontsize=12, fontweight='bold')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Fazla axes'leri gizle
                for idx in range(len(all_plastic_heatmaps), len(axes)):
                    axes[idx].axis('off')
                
                plt.suptitle('Amino Acid Probability x Mass Heatmap - All Plastics Comparison', 
                            fontsize=16, fontweight='bold', y=0.995)
                plt.tight_layout()
                
                comparison_heatmap_path = os.path.join(ABLATION_FIGURES_DIR, 'aa_probability_mass_heatmap_all_plastics_comparison.png')
                plt.savefig(comparison_heatmap_path, dpi=300, bbox_inches='tight')
                print(f"✓ Tüm plastikler karşılaştırma heatmap kaydedildi: {comparison_heatmap_path}")
                # plt.show()  # İsterseniz bu satırı açabilirsiniz (GUI backend gerekir)
                plt.close()
        except Exception as e:
            print(f"⚠ Amino acid heatmap karşılaştırması hatası: {e}")
        
        # Detaylı istatistik raporu oluştur
        try:
            create_detailed_report(summary_df, all_results_summary)
        except Exception as e:
            print(f"⚠ Rapor oluşturma hatası: {e}")

# ============================================================================
# HÜCRE 7: Üretilen Peptitlerin Orijinal Veri Setine Benzerlik Analizi
# ============================================================================

def calculate_hamming_distance(seq1, seq2):
    """İki peptit dizisi arasındaki Hamming distance (farklı pozisyon sayısı)"""
    if len(seq1) != len(seq2):
        return len(seq1)  # Farklı uzunlukta ise maksimum distance
    return sum(a != b for a, b in zip(seq1, seq2))

def calculate_edit_distance(seq1, seq2):
    """Levenshtein distance (edit distance) - minimum değişiklik sayısı"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

def calculate_sequence_similarity(seq1, seq2):
    """İki peptit dizisi arasındaki benzerlik yüzdesi (0-100)"""
    if len(seq1) != len(seq2):
        return 0.0
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return (matches / len(seq1)) * 100

def find_closest_sequences(generated_seq, original_sequences, top_k=5):
    """Orijinal veri setindeki en yakın peptitleri bul"""
    similarities = []
    for orig_seq in original_sequences:
        hamming = calculate_hamming_distance(generated_seq, orig_seq)
        edit = calculate_edit_distance(generated_seq, orig_seq)
        similarity = calculate_sequence_similarity(generated_seq, orig_seq)
        similarities.append({
            'original_sequence': orig_seq,
            'hamming_distance': hamming,
            'edit_distance': edit,
            'similarity_percent': similarity
        })
    
    # Similarity'ye göre sırala (yüksek = daha benzer)
    similarities.sort(key=lambda x: x['similarity_percent'], reverse=True)
    return similarities[:top_k]


def analyze_generated_peptides_similarity(generated_peptides, data_dir=None, 
                                         plastic_types=['PET', 'PP', 'PE', 'PVC', 'Nylon', 'PMMA', 'PS'],
                                         verbose=None):
    """
    Üretilen peptitlerin orijinal veri setindeki peptitlere benzerliğini analiz eder.
    
    Args:
        generated_peptides: Dict {plastic_type: [list of generated sequences]}
        data_dir: Orijinal veri klasörü
        plastic_types: Analiz edilecek plastik tipleri
        verbose: Print çıktıları göster (None ise __name__ == '__main__' kontrolü yapılır)
    
    Returns:
        Analiz sonuçları DataFrame
    """
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Verbose parametresi: eğer None ise __name__ == '__main__' kontrolü yap
    if verbose is None:
        verbose = __name__ == '__main__'
    
    all_results = []
    
    if verbose:
        print("\n" + "="*80)
        print("ÜRETİLEN PEPTİTLERİN BENZERLİK ANALİZİ")
        print("="*80)
        print(f"Veri klasörü: {data_dir}")
    
    for plastic_type in plastic_types:
        if verbose:
            print(f"\n{'='*60}")
            print(f"PLASTİK TİPİ: {plastic_type}")
            print(f"{'='*60}")
        
        # Orijinal veri setini yükle
        csv_path = os.path.join(data_dir, f"{plastic_type}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠ {plastic_type} CSV dosyası bulunamadı, atlanıyor...")
            continue
        
        df_original = pd.read_csv(csv_path)
        df_original = df_original.dropna(subset=['Sequence', 'Score'])
        df_original['Sequence'] = df_original['Sequence'].str.strip().str.upper()
        
        original_sequences = df_original['Sequence'].tolist()
        if verbose:
            print(f"✓ Orijinal veri seti yüklendi: {len(original_sequences)} peptit")
        
        # Üretilen peptitleri al
        if plastic_type not in generated_peptides:
            if verbose:
                print(f"⚠ {plastic_type} için üretilen peptit bulunamadı, atlanıyor...")
            continue
        
        generated_seqs = generated_peptides[plastic_type]
        if verbose:
            print(f"✓ Analiz edilecek üretilen peptit sayısı: {len(generated_seqs)}")
        
        # Her üretilen peptit için analiz
        for gen_seq in generated_seqs:
            # En yakın 5 peptit bul
            closest = find_closest_sequences(gen_seq, original_sequences, top_k=5)
            
            # Benzersizlik kontrolü
            is_unique = gen_seq not in original_sequences
            exact_matches = sum(1 for orig in original_sequences if orig == gen_seq)
            
            # İstatistikler
            min_hamming = min(c['hamming_distance'] for c in closest)
            max_similarity = max(c['similarity_percent'] for c in closest)
            avg_similarity = np.mean([c['similarity_percent'] for c in closest])
            
            # Sonuç kaydet
            result = {
                'plastic_type': plastic_type,
                'generated_sequence': gen_seq,
                'is_unique': is_unique,
                'exact_matches_in_original': exact_matches,
                'min_hamming_distance': min_hamming,
                'max_similarity_percent': max_similarity,
                'avg_similarity_percent': avg_similarity,
                'closest_sequence_1': closest[0]['original_sequence'],
                'closest_similarity_1': closest[0]['similarity_percent'],
                'closest_hamming_1': closest[0]['hamming_distance'],
                'closest_sequence_2': closest[1]['original_sequence'] if len(closest) > 1 else None,
                'closest_similarity_2': closest[1]['similarity_percent'] if len(closest) > 1 else None,
                'closest_sequence_3': closest[2]['original_sequence'] if len(closest) > 2 else None,
                'closest_similarity_3': closest[2]['similarity_percent'] if len(closest) > 2 else None,
            }
            all_results.append(result)
            
            # Özet yazdır (sadece verbose modda)
            if verbose:
                print(f"\n  Üretilen: {gen_seq}")
                print(f"    ✓ Benzersiz: {'Evet' if is_unique else 'Hayır'}")
                print(f"    ✓ En yakın benzerlik: {max_similarity:.1f}%")
                print(f"    ✓ Minimum Hamming distance: {min_hamming}/12")
                print(f"    ✓ En yakın peptit: {closest[0]['original_sequence']} ({closest[0]['similarity_percent']:.1f}% benzer)")
    
    # DataFrame oluştur
    if all_results:
        similarity_df = pd.DataFrame(all_results)
        
        # Kaydet
        similarity_path = os.path.join(ABLATION_TABLES_DIR, 'generated_peptides_similarity_analysis.csv')
        similarity_df.to_csv(similarity_path, index=False)
        print(f"\n✓ Benzerlik analizi kaydedildi: {similarity_path}")
        
        # Özet istatistikler
        print("\n" + "="*80)
        print("BENZERLİK ANALİZİ ÖZET İSTATİSTİKLERİ")
        print("="*80)
        
        print(f"\n📊 Genel İstatistikler:")
        print(f"   Toplam üretilen peptit: {len(similarity_df)}")
        print(f"   Benzersiz peptit sayısı: {similarity_df['is_unique'].sum()}")
        print(f"   Benzersizlik oranı: {similarity_df['is_unique'].mean()*100:.1f}%")
        print(f"   Ortalama maksimum benzerlik: {similarity_df['max_similarity_percent'].mean():.1f}%")
        print(f"   Ortalama minimum Hamming distance: {similarity_df['min_hamming_distance'].mean():.2f}/12")
        
        print(f"\n📊 Plastik Tipine Göre İstatistikler:")
        for plastic in similarity_df['plastic_type'].unique():
            plastic_df = similarity_df[similarity_df['plastic_type'] == plastic]
            print(f"\n   {plastic}:")
            print(f"      Üretilen peptit: {len(plastic_df)}")
            print(f"      Benzersiz: {plastic_df['is_unique'].sum()}/{len(plastic_df)} ({plastic_df['is_unique'].mean()*100:.1f}%)")
            print(f"      Ort. maks. benzerlik: {plastic_df['max_similarity_percent'].mean():.1f}%")
            print(f"      Ort. min. Hamming: {plastic_df['min_hamming_distance'].mean():.2f}/12")
        
        # Benzerlik dağılımı
        print(f"\n📊 Benzerlik Dağılımı:")
        similarity_ranges = [
            (100, "Tamamen aynı (100%)"),
            (90, 100, "Çok benzer (90-100%)"),
            (75, 90, "Benzer (75-90%)"),
            (50, 75, "Orta benzerlik (50-75%)"),
            (0, 50, "Düşük benzerlik (0-50%)")
        ]
        
        for range_info in similarity_ranges:
            if len(range_info) == 2:
                label, desc = range_info
                count = (similarity_df['max_similarity_percent'] == label).sum()
            else:
                min_val, max_val, desc = range_info
                count = ((similarity_df['max_similarity_percent'] >= min_val) & 
                        (similarity_df['max_similarity_percent'] < max_val)).sum()
            print(f"   {desc}: {count} peptit ({count/len(similarity_df)*100:.1f}%)")
        
        return similarity_df
    else:
        print("\n⚠ Hiç üretilen peptit bulunamadı, analiz yapılamadı.")
        return None

# ============================================================================
# R² Skorunun Açıklaması ve Doğrulama
# ============================================================================

    # R² Skorunun Açıklaması ve Doğrulama
    print("\n" + "="*80)
    print("R² SKORU AÇIKLAMASI VE DOĞRULAMA")
    print("="*80)

    print("""
📊 R² SKORU NE ANLAMA GELİYOR?

R² (Coefficient of Determination) skoru, modelin tahmin doğruluğunu ölçer:

1. R² = 1.0: Mükemmel tahmin (tahminler gerçek değerlerle tamamen eşleşiyor)
2. R² = 0.0: Model, basit ortalama kadar iyi (tahminler ortalama değere eşit)
3. R² < 0.0: Model, basit ortalamadan daha kötü

R² skoru şu formülle hesaplanır:
    R² = 1 - (SS_res / SS_tot)
    
    SS_res = Σ(y_gerçek - y_tahmin)²  (Residual sum of squares)
    SS_tot = Σ(y_gerçek - y_ortalama)²  (Total sum of squares)

⚠ ÖNEMLİ: R² skoru sadece modelin TAHMİN DOĞRULUĞUNU gösterir.
           R² skoru, üretilen peptitlerin ORİJİNAL VERİ SETİNE BENZERLİĞİNİ göstermez!

🔍 ÜRETİLEN PEPTİTLERİN BENZERLİK KONTROLÜ:

1. Hamming Distance: İki peptit arasındaki farklı pozisyon sayısı
   - 0/12: Tamamen aynı
   - 12/12: Tamamen farklı

2. Edit Distance (Levenshtein): Bir peptidi diğerine çevirmek için gereken minimum değişiklik sayısı

3. Sequence Similarity: İki peptit arasındaki benzerlik yüzdesi
   - 100%: Tamamen aynı
   - 0%: Tamamen farklı

4. Benzersizlik Kontrolü: Üretilen peptit orijinal veri setinde var mı?

✅ İYİ BİR MODEL:
   - Yüksek R² skoru (tahmin doğruluğu iyi)
   - Üretilen peptitler benzersiz (orijinal veri setinde yok)
   - Üretilen peptitler orijinal veri setinden farklı (düşük benzerlik)
   - Üretilen peptitler daha iyi skorlara sahip (optimizasyon başarılı)
""")

    # Örnek: Eğer üretilen peptitler varsa analiz yap
    # Not: Bu kısım optimizasyon sonuçlarından üretilen peptitleri alır
    # Şimdilik placeholder olarak bırakıyoruz, gerçek kullanımda optimizasyon sonuçlarından alınacak

    # HÜCRE 7: Üretilen peptitlerin benzerlik analizi
    print("\n" + "="*80)
    print("HÜCRE 7: ÜRETİLEN PEPTİTLERİN BENZERLİK ANALİZİ")
    print("="*80)
    
    # Tüm üretilen peptitleri topla
    all_generated_peptides = {}
    
    for plastic_type in PLASTICS_TO_RUN:
        generated_dict_name = f"{plastic_type.lower()}_generated_peptides"
        if generated_dict_name in globals():
            generated_dict = globals()[generated_dict_name]
            # Her model için en iyi peptitleri topla
            best_peptides = []
            for model_type, results in generated_dict.items():
                if results:
                    # En iyi skorlu peptitleri al (top 5)
                    sorted_results = sorted(results, key=lambda x: x['optimized_score'])
                    best_peptides.extend([r['optimized_peptide'] for r in sorted_results[:5]])
            
            if best_peptides:
                all_generated_peptides[plastic_type] = best_peptides
    
    # Benzerlik analizi yap
    if all_generated_peptides:
        print(f"\n📊 {len(all_generated_peptides)} plastik tipi için benzerlik analizi yapılıyor...")
        similarity_df = analyze_generated_peptides_similarity(
            all_generated_peptides,
            data_dir=DATA_DIR,
            plastic_types=list(all_generated_peptides.keys()),
            verbose=True
        )
        
        if similarity_df is not None and not similarity_df.empty:
            print(f"\n✓ Benzerlik analizi tamamlandı!")
            print(f"   Toplam analiz edilen peptit: {len(similarity_df)}")
            print(f"\n📋 Özet:")
            print(similarity_df.groupby('plastic_type').agg({
                'max_similarity_percent': ['mean', 'min', 'max'],
                'min_hamming_distance': ['mean', 'min', 'max']
            }).round(2))
    else:
        print("⚠ Üretilen peptit bulunamadı. Önce eğitim ve optimizasyon yapılmalı.")
    
    print("\n" + "="*80)
    print("TÜM ABLATION STUDY, FINAL EĞİTİMLER VE PEPTİD ÜRETİMİ TAMAMLANDI!")
    print("="*80)
    print("\n📁 Kaydedilen dosyalar:")
    print(f"   - Ablation sonuçları: {ABLATION_TABLES_DIR}")
    print(f"   - Final modeller: {FINAL_MODELS_DIR}")
    print(f"   - Üretilen peptitler: {ABLATION_TABLES_DIR}/generated_peptides_*.csv")
    print(f"   - Benzerlik analizi: {ABLATION_TABLES_DIR}/generated_peptides_similarity_analysis.csv")
    print(f"   - Log dosyaları: {ABLATION_LOGS_DIR}")

# Windows multiprocessing için ana kontrol
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Windows için gerekli
    main()

