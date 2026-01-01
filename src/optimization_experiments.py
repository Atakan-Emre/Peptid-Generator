# ============================================================================
# OPTIMIZATION EXPERIMENTS - Farklı Optimizasyon Yöntemleri Karşılaştırması
# ============================================================================
# Bu script ENCDEC modeli için farklı optimizasyon yöntemlerini test eder
# Her optimizer için 250 epoch eğitim yapılır
# Detaylı loglar, grafikler ve peptid üretimi yapılır
#
# Desteklenen Optimizerler:
# - Adam, AdamW, NAdam, RAdam
# - SGD (momentum ile), SGD + Nesterov
# - RMSprop, Adagrad, Adadelta
# - LBFGS (büyük veri için uygun değil - opsiyonel)
# ============================================================================

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import json
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Proje dizinleri
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # src'nin üst dizini

# Veri ve sonuç dizinleri
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sortingData")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "optimization_experiments")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Klasörleri oluştur
for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# WINDOWS PC AYARLARI - RTX 4080 Super (16GB VRAM), Ryzen 9 9700X, 64GB RAM
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    # CUDA Optimizasyonları (RTX 4080 Super için)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32
    torch.backends.cudnn.allow_tf32 = True
    
    # GPU bilgisi
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f} GB)")

# Mixed Precision Training (RTX 4080 Super için optimal)
USE_AMP = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

# DataLoader ayarları (Windows + RTX 4080 Super optimize)
NUM_WORKERS = 4  # Ryzen 9 9700X için
PIN_MEMORY = True  # GPU transfer hızlandırma
PERSISTENT_WORKERS = True  # Worker overhead azaltma
PREFETCH_FACTOR = 2  # Veri ön yükleme

# Amino asit alfabesi
AMINO_ACIDS = "ADEFGHIKLMNQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# ============================================================================
# ENCDEC MODEL - En İyi Model
# ============================================================================

class LSTMEncoderDecoder(nn.Module):
    """ENCDEC Model - Tüm plastiklerde en iyi performansı gösteren model"""
    def __init__(self, input_dim=18, hidden_dim=256, num_layers=2, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.dec_init = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.score_head = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        h_last = enc_out[:, -1, :]
        h_last = self.ln(h_last)
        score_pred = self.score_head(h_last).squeeze(-1)
        
        batch_size, seq_len, _ = x.shape
        dec_input = torch.zeros(batch_size, seq_len, x.size(-1), device=x.device)
        h0 = torch.tanh(self.dec_init(h_last)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        dec_out, _ = self.decoder(dec_input, (h0, c0))
        recon_logits = self.out_proj(dec_out)
        return recon_logits, score_pred

# ============================================================================
# DATASET
# ============================================================================

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
# UTILITY FUNCTIONS
# ============================================================================

def seed_everything(seed=42):
    """Reproducibility için seed ayarla"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def one_hot_encode_sequences(sequences):
    """Peptid sekanslarını one-hot encode et"""
    encoded = []
    for seq in sequences:
        one_hot = np.zeros((12, len(AMINO_ACIDS)))
        for i, aa in enumerate(seq[:12]):
            if aa in AA_TO_IDX:
                one_hot[i, AA_TO_IDX[aa]] = 1
        encoded.append(one_hot)
    return np.stack(encoded, axis=0)

def decode_one_hot(one_hot):
    """One-hot'tan peptid sekansına dönüştür"""
    indices = np.argmax(one_hot, axis=-1)
    return ''.join([IDX_TO_AA[i] for i in indices])

# ============================================================================
# OPTIMIZER CONFIGURATIONS
# ============================================================================

def get_optimizer_configs():
    """Farklı optimizer konfigürasyonları"""
    return {
        # Adam ailesi
        'Adam': {
            'class': optim.Adam,
            'params': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8},
            'description': 'Adaptive Moment Estimation - Varsayılan'
        },
        'AdamW': {
            'class': optim.AdamW,
            'params': {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.0001},
            'description': 'Adam with Decoupled Weight Decay'
        },
        'NAdam': {
            'class': optim.NAdam,
            'params': {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.0001},
            'description': 'Nesterov-accelerated Adam'
        },
        'RAdam': {
            'class': optim.RAdam,
            'params': {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.0001},
            'description': 'Rectified Adam - Variance düzeltmeli'
        },
        
        # SGD ailesi
        'SGD': {
            'class': optim.SGD,
            'params': {'lr': 0.01, 'momentum': 0.9},
            'description': 'Stochastic Gradient Descent + Momentum'
        },
        'SGD_Nesterov': {
            'class': optim.SGD,
            'params': {'lr': 0.01, 'momentum': 0.9, 'nesterov': True},
            'description': 'SGD with Nesterov Momentum'
        },
        
        # Diğer adaptif yöntemler
        'RMSprop': {
            'class': optim.RMSprop,
            'params': {'lr': 0.001, 'alpha': 0.99, 'eps': 1e-8},
            'description': 'Root Mean Square Propagation'
        },
        'Adagrad': {
            'class': optim.Adagrad,
            'params': {'lr': 0.01, 'eps': 1e-10},
            'description': 'Adaptive Gradient - Sparse veriler için'
        },
        'Adadelta': {
            'class': optim.Adadelta,
            'params': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6},
            'description': 'Adadelta - Learning rate gerektirmez'
        },
        
        # Learning rate scheduler ile Adam
        'Adam_CosineAnnealing': {
            'class': optim.Adam,
            'params': {'lr': 0.001},
            'scheduler': 'cosine',
            'description': 'Adam + Cosine Annealing LR Scheduler'
        },
        'AdamW_OneCycle': {
            'class': optim.AdamW,
            'params': {'lr': 0.0001, 'weight_decay': 0.0001},  # max_lr için başlangıç düşük
            'scheduler': 'onecycle',
            'description': 'AdamW + OneCycleLR (Super-Convergence)'
        },
    }

# ============================================================================
# BEST MODEL PARAMETERS - Her plastik için en yüksek R² veren ENCDEC ayarları
# ============================================================================
# Kaynak: Ablation Study sonuçları (reports/*.md)
# Platform: Windows 11, RTX 4080 Super (16GB VRAM), Ryzen 9 9700X, 64GB RAM

BEST_MODEL_PARAMS = {
    # PET: Final Test R² = 0.9766 (EN YÜKSEK)
    'PET': {
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 1024,  # RTX 4080 Super için optimize
        'lambda_score': 1.0,
        'weight_decay': 0.0001,
        'use_layernorm': True
    },
    # PP: Final Test R² = 0.9583
    'PP': {
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 1024,
        'lambda_score': 1.0,
        'weight_decay': 0.0001,
        'use_layernorm': True
    },
    # PE: Final Test R² = 0.9547
    'PE': {
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 1024,
        'lambda_score': 1.0,
        'weight_decay': 0.0001,
        'use_layernorm': True
    },
    # PVC: Final Test R² = 0.9490
    'PVC': {
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 1024,
        'lambda_score': 1.0,
        'weight_decay': 0.0001,
        'use_layernorm': True
    },
    # Nylon: Final Test R² = 0.9576
    'Nylon': {
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.2,  # Küçük veri seti için daha yüksek dropout
        'learning_rate': 0.001,
        'batch_size': 128,  # Küçük veri seti (142K) için optimize
        'lambda_score': 1.0,
        'weight_decay': 0.0001,
        'use_layernorm': False  # Nylon için False daha iyi sonuç verdi
    },
}

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_with_optimizer(plastic_type, optimizer_name, optimizer_config, epochs=250, seed=42):
    """
    Belirli bir optimizer ile ENCDEC modelini eğit
    
    Args:
        plastic_type: Plastik tipi (PET, PP, PE, PVC, Nylon)
        optimizer_name: Optimizer adı
        optimizer_config: Optimizer konfigürasyonu
        epochs: Eğitim epoch sayısı
        seed: Random seed
    
    Returns:
        dict: Eğitim sonuçları ve metrikleri
    """
    
    print(f"\n{'='*80}")
    print(f"🚀 OPTIMIZER EXPERIMENT: {optimizer_name} - {plastic_type}")
    print(f"   {optimizer_config['description']}")
    print(f"{'='*80}")
    
    seed_everything(seed)
    
    # Model parametrelerini al
    model_params = BEST_MODEL_PARAMS.get(plastic_type, BEST_MODEL_PARAMS['PET'])
    
    # Veri yükle
    csv_path = os.path.join(DATA_DIR, f"{plastic_type}.csv")
    if not os.path.exists(csv_path):
        print(f"❌ CSV dosyası bulunamadı: {csv_path}")
        return None
    
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
    
    print(f"📊 Veri: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
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
    batch_size = model_params['batch_size']
    train_ds = PeptideDataset(X_train, y_train_norm)
    val_ds = PeptideDataset(X_val, y_val_norm)
    test_ds = PeptideDataset(X_test, y_test_norm)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
                             prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                           persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
                           prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
                            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None)
    
    # Model oluştur
    model = LSTMEncoderDecoder(
        input_dim=18,
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout'],
        use_layernorm=model_params['use_layernorm']
    ).to(device)
    
    # Loss fonksiyonları
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    lambda_score = model_params['lambda_score']
    
    # Optimizer oluştur
    optimizer = optimizer_config['class'](model.parameters(), **optimizer_config['params'])
    
    # Scheduler (varsa)
    scheduler = None
    if 'scheduler' in optimizer_config:
        if optimizer_config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        elif optimizer_config['scheduler'] == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader),
                pct_start=0.3, anneal_strategy='cos'
            )
    
    # Eğitim logları
    history = {
        'train_loss': [], 'val_loss': [],
        'train_r2': [], 'val_r2': [],
        'learning_rates': [], 'epoch_times': []
    }
    
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    best_epoch = 0
    best_state = None
    patience = 15
    patience_ctr = 0
    
    # Eğitim döngüsü
    start_time = datetime.now()
    
    epoch_pbar = tqdm(range(epochs), desc=f"{optimizer_name}", ncols=100)
    
    for epoch in epoch_pbar:
        epoch_start = datetime.now()
        
        # Training
        model.train()
        train_loss_sum = 0
        train_preds = []
        train_targets = []
        
        for X_batch, y_batch, class_targets in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            class_targets = class_targets.to(device)
            
            optimizer.zero_grad()
            
            if USE_AMP and scaler is not None:
                with torch.cuda.amp.autocast():
                    recon_logits, score_pred = model(X_batch)
                    recon_loss = ce_loss(recon_logits.view(-1, 18), class_targets.view(-1))
                    score_loss = mse_loss(score_pred, y_batch)
                    loss = recon_loss + lambda_score * score_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon_logits, score_pred = model(X_batch)
                recon_loss = ce_loss(recon_logits.view(-1, 18), class_targets.view(-1))
                score_loss = mse_loss(score_pred, y_batch)
                loss = recon_loss + lambda_score * score_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # OneCycle scheduler her step'te güncellenir
            if scheduler is not None and optimizer_config.get('scheduler') == 'onecycle':
                scheduler.step()
            
            train_loss_sum += loss.item() * X_batch.size(0)
            train_preds.extend(score_pred.detach().cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())
        
        train_loss = train_loss_sum / len(train_ds)
        train_r2 = r2_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_loss_sum = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch, class_targets in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                class_targets = class_targets.to(device)
                
                recon_logits, score_pred = model(X_batch)
                recon_loss = ce_loss(recon_logits.view(-1, 18), class_targets.view(-1))
                score_loss = mse_loss(score_pred, y_batch)
                loss = recon_loss + lambda_score * score_loss
                
                val_loss_sum += loss.item() * X_batch.size(0)
                val_preds.extend(score_pred.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
        
        val_loss = val_loss_sum / len(val_ds)
        val_r2 = r2_score(val_targets, val_preds)
        
        # Epoch sonrası scheduler (Cosine için)
        if scheduler is not None and optimizer_config.get('scheduler') != 'onecycle':
            scheduler.step()
        
        # Learning rate kaydet
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        # History güncelle
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        
        # Best model kontrol
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        
        # Progress bar güncelle
        epoch_pbar.set_postfix({
            'TrL': f'{train_loss:.4f}',
            'VaL': f'{val_loss:.4f}',
            'VaR²': f'{val_r2:.4f}',
            'LR': f'{current_lr:.6f}'
        })
        
        # Early stopping
        if patience_ctr >= patience:
            print(f"\n⏹ Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    # En iyi modeli yükle
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Test değerlendirmesi
    model.eval()
    test_preds = []
    test_targets_list = []
    
    with torch.no_grad():
        for X_batch, y_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            _, score_pred = model(X_batch)
            test_preds.extend(score_pred.cpu().numpy())
            test_targets_list.extend(y_batch.numpy())
    
    # Denormalize
    test_preds_denorm = np.array(test_preds) * score_std + score_mean
    test_targets_denorm = np.array(test_targets_list) * score_std + score_mean
    
    test_r2 = r2_score(test_targets_denorm, test_preds_denorm)
    test_mae = mean_absolute_error(test_targets_denorm, test_preds_denorm)
    test_rmse = np.sqrt(mean_squared_error(test_targets_denorm, test_preds_denorm))
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n📊 SONUÇLAR ({optimizer_name} - {plastic_type}):")
    print(f"   Best Epoch: {best_epoch+1}")
    print(f"   Best Val Loss: {best_val_loss:.6f}")
    print(f"   Best Val R²: {best_val_r2:.6f}")
    print(f"   Test R²: {test_r2:.6f}")
    print(f"   Test MAE: {test_mae:.4f}")
    print(f"   Test RMSE: {test_rmse:.4f}")
    print(f"   Toplam Süre: {total_time:.1f}s")
    
    # Sonuçları kaydet
    results = {
        'optimizer': optimizer_name,
        'plastic_type': plastic_type,
        'description': optimizer_config['description'],
        'best_epoch': best_epoch + 1,
        'total_epochs': len(history['train_loss']),
        'best_val_loss': best_val_loss,
        'best_val_r2': best_val_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'total_time_seconds': total_time,
        'history': history,
        'score_mean': score_mean,
        'score_std': score_std,
        'model_params': model_params,
        'optimizer_params': optimizer_config['params']
    }
    
    # Model kaydet
    model_path = os.path.join(MODELS_DIR, f"encdec_{plastic_type}_{optimizer_name}.pth")
    torch.save({
        'model_state_dict': best_state if best_state else model.state_dict(),
        'optimizer_name': optimizer_name,
        'plastic_type': plastic_type,
        'best_epoch': best_epoch,
        'best_val_r2': best_val_r2,
        'test_r2': test_r2,
        'score_mean': score_mean,
        'score_std': score_std,
        'model_params': model_params
    }, model_path)
    
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(results_list, plastic_type):
    """Tüm optimizer'lar için training curves karşılaştırması"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for i, result in enumerate(results_list):
        name = result['optimizer']
        history = result['history']
        color = colors[i]
        
        # Train Loss
        axes[0, 0].plot(history['train_loss'], label=name, color=color, alpha=0.8)
        
        # Val Loss
        axes[0, 1].plot(history['val_loss'], label=name, color=color, alpha=0.8)
        
        # Val R²
        axes[1, 0].plot(history['val_r2'], label=name, color=color, alpha=0.8)
        
        # Learning Rate
        axes[1, 1].plot(history['learning_rates'], label=name, color=color, alpha=0.8)
    
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Validation R²', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].legend(loc='lower right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Optimizer Comparison - {plastic_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, f'training_curves_comparison_{plastic_type}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves kaydedildi: {save_path}")

def plot_final_comparison(results_list, plastic_type):
    """Final metrik karşılaştırma grafiği"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r['optimizer'] for r in results_list]
    test_r2 = [r['test_r2'] for r in results_list]
    test_mae = [r['test_mae'] for r in results_list]
    times = [r['total_time_seconds'] for r in results_list]
    
    # Renk: en iyi yeşil, en kötü kırmızı
    colors_r2 = ['green' if v == max(test_r2) else 'red' if v == min(test_r2) else 'steelblue' for v in test_r2]
    colors_mae = ['green' if v == min(test_mae) else 'red' if v == max(test_mae) else 'steelblue' for v in test_mae]
    colors_time = ['green' if v == min(times) else 'red' if v == max(times) else 'steelblue' for v in times]
    
    # Test R²
    bars1 = axes[0].barh(names, test_r2, color=colors_r2, alpha=0.8)
    axes[0].set_xlabel('Test R²')
    axes[0].set_title('Test R² (↑ iyi)', fontweight='bold')
    axes[0].set_xlim(min(test_r2) * 0.99, max(test_r2) * 1.01)
    for bar, val in zip(bars1, test_r2):
        axes[0].text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
    
    # Test MAE
    bars2 = axes[1].barh(names, test_mae, color=colors_mae, alpha=0.8)
    axes[1].set_xlabel('Test MAE')
    axes[1].set_title('Test MAE (↓ iyi)', fontweight='bold')
    for bar, val in zip(bars2, test_mae):
        axes[1].text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=9)
    
    # Training Time
    bars3 = axes[2].barh(names, times, color=colors_time, alpha=0.8)
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Training Time (↓ iyi)', fontweight='bold')
    for bar, val in zip(bars3, times):
        axes[2].text(val, bar.get_y() + bar.get_height()/2, f'{val:.0f}s', va='center', fontsize=9)
    
    plt.suptitle(f'Optimizer Performance Comparison - {plastic_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, f'optimizer_comparison_{plastic_type}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison chart kaydedildi: {save_path}")

def create_summary_table(all_results):
    """Özet tablo oluştur"""
    
    rows = []
    for result in all_results:
        rows.append({
            'Plastic': result['plastic_type'],
            'Optimizer': result['optimizer'],
            'Description': result['description'],
            'Best_Epoch': result['best_epoch'],
            'Total_Epochs': result['total_epochs'],
            'Best_Val_R2': result['best_val_r2'],
            'Test_R2': result['test_r2'],
            'Test_MAE': result['test_mae'],
            'Test_RMSE': result['test_rmse'],
            'Time_Seconds': result['total_time_seconds']
        })
    
    df = pd.DataFrame(rows)
    
    # CSV kaydet
    csv_path = os.path.join(TABLES_DIR, 'optimizer_comparison_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Summary table kaydedildi: {csv_path}")
    
    return df

# ============================================================================
# PEPTIDE GENERATION (Simulated Annealing)
# ============================================================================

def generate_peptides_for_optimizer(model_path, plastic_type, optimizer_name, n_peptides=30):
    """Belirli bir optimizer ile eğitilmiş model için peptid üret"""
    
    print(f"\n🧬 Peptid üretimi: {optimizer_name} - {plastic_type}")
    
    # Model yükle
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_params = checkpoint['model_params']
    score_mean = checkpoint['score_mean']
    score_std = checkpoint['score_std']
    
    model = LSTMEncoderDecoder(
        input_dim=18,
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout'],
        use_layernorm=model_params['use_layernorm']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    def predict_score(seq):
        """Tek sekans için skor tahmin et"""
        one_hot = np.zeros((1, 12, 18))
        for i, aa in enumerate(seq[:12]):
            if aa in AA_TO_IDX:
                one_hot[0, i, AA_TO_IDX[aa]] = 1
        
        with torch.no_grad():
            X = torch.tensor(one_hot, dtype=torch.float32).to(device)
            _, score_pred = model(X)
            score_norm = score_pred.item()
            return score_norm * score_std + score_mean
    
    def simulated_annealing(start_seq, T_init=1.0, T_min=0.01, alpha=0.995, max_iter=500):
        """Simulated Annealing ile peptid optimize et"""
        current_seq = list(start_seq)
        current_score = predict_score(''.join(current_seq))
        best_seq = current_seq.copy()
        best_score = current_score
        T = T_init
        
        for _ in range(max_iter):
            # Rastgele mutasyon
            pos = random.randint(0, 11)
            old_aa = current_seq[pos]
            new_aa = random.choice(AMINO_ACIDS)
            
            current_seq[pos] = new_aa
            new_score = predict_score(''.join(current_seq))
            
            # Kabul kriteri (daha negatif = daha iyi)
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_seq = current_seq.copy()
            else:
                current_seq[pos] = old_aa
            
            T *= alpha
            if T < T_min:
                break
        
        return ''.join(best_seq), best_score
    
    # Rastgele başlangıç sekansları oluştur
    generated = []
    for i in range(n_peptides):
        start_seq = ''.join(random.choices(AMINO_ACIDS, k=12))
        opt_seq, opt_score = simulated_annealing(start_seq)
        generated.append({
            'Sequence': opt_seq,
            'Score': opt_score,
            'Start_Sequence': start_seq,
            'Start_Score': predict_score(start_seq),
            'Optimizer': optimizer_name,
            'Plastic': plastic_type
        })
    
    # Skora göre sırala (en düşük = en iyi)
    generated = sorted(generated, key=lambda x: x['Score'])
    
    # CSV kaydet
    df = pd.DataFrame(generated)
    csv_path = os.path.join(TABLES_DIR, f'generated_peptides_{optimizer_name}_{plastic_type}.csv')
    df.to_csv(csv_path, index=False)
    print(f"   ✓ {n_peptides} peptid üretildi, en iyi skor: {generated[0]['Score']:.2f}")
    
    return generated

# ============================================================================
# MARKDOWN RAPOR OLUŞTURMA
# ============================================================================

def generate_markdown_report(all_results, plastic_type):
    """Optimizer deneyleri için detaylı Markdown raporu oluştur"""
    
    report_path = os.path.join(RESULTS_DIR, f'OPTIMIZER_RAPORU_{plastic_type}.md')
    
    # Sonuçları R² sırasına göre sırala
    sorted_results = sorted(all_results, key=lambda x: x['test_r2'], reverse=True)
    best_result = sorted_results[0]
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Başlık
        f.write(f"# 🔬 Optimizer Karşılaştırma Raporu - {plastic_type}\n\n")
        f.write(f"**Tarih:** {datetime.now().strftime('%d %B %Y, %H:%M')}\n")
        f.write(f"**Model:** ENCDEC (Encoder-Decoder)\n")
        f.write(f"**Platform:** Windows 11, RTX 4080 Super (16GB VRAM)\n")
        f.write(f"**Eğitim Epoch:** 250\n\n")
        f.write("---\n\n")
        
        # Özet
        f.write("## 📊 Özet\n\n")
        f.write("| Metrik | Değer |\n")
        f.write("|--------|-------|\n")
        f.write(f"| **Plastik Tipi** | {plastic_type} |\n")
        f.write(f"| **Test Edilen Optimizer** | {len(all_results)} |\n")
        f.write(f"| **En İyi Optimizer** | **{best_result['optimizer']}** |\n")
        f.write(f"| **En İyi Test R²** | **{best_result['test_r2']:.6f}** |\n")
        f.write(f"| **En İyi Test MAE** | {best_result['test_mae']:.4f} |\n")
        f.write(f"| **En İyi Test RMSE** | {best_result['test_rmse']:.4f} |\n\n")
        
        # Model parametreleri
        model_params = best_result['model_params']
        f.write("## 🧠 ENCDEC Model Parametreleri\n\n")
        f.write("| Parametre | Değer |\n")
        f.write("|-----------|-------|\n")
        for key, value in model_params.items():
            f.write(f"| `{key}` | {value} |\n")
        f.write("\n")
        
        # Optimizer karşılaştırma tablosu
        f.write("---\n\n")
        f.write("## 📈 Optimizer Karşılaştırma Tablosu\n\n")
        f.write("| Sıra | Optimizer | Test R² | Test MAE | Test RMSE | Best Epoch | Süre (s) |\n")
        f.write("|------|-----------|---------|----------|-----------|------------|----------|\n")
        
        for i, result in enumerate(sorted_results):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}"
            f.write(f"| {medal} | **{result['optimizer']}** | {result['test_r2']:.6f} | ")
            f.write(f"{result['test_mae']:.4f} | {result['test_rmse']:.4f} | ")
            f.write(f"{result['best_epoch']} | {result['total_time_seconds']:.1f} |\n")
        f.write("\n")
        
        # Optimizer açıklamaları
        f.write("## 📝 Optimizer Açıklamaları\n\n")
        for result in sorted_results:
            f.write(f"- **{result['optimizer']}**: {result['description']}\n")
        f.write("\n")
        
        # En iyi 3 optimizer detayları
        f.write("---\n\n")
        f.write("## 🏆 En İyi 3 Optimizer Detayları\n\n")
        
        for i, result in enumerate(sorted_results[:3]):
            medal = ["🥇", "🥈", "🥉"][i]
            f.write(f"### {medal} {result['optimizer']}\n\n")
            f.write(f"**Açıklama:** {result['description']}\n\n")
            f.write("| Metrik | Değer |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Test R² | **{result['test_r2']:.6f}** |\n")
            f.write(f"| Test MAE | {result['test_mae']:.4f} |\n")
            f.write(f"| Test RMSE | {result['test_rmse']:.4f} |\n")
            f.write(f"| Best Epoch | {result['best_epoch']} / {result['total_epochs']} |\n")
            f.write(f"| Best Val R² | {result['best_val_r2']:.6f} |\n")
            f.write(f"| Eğitim Süresi | {result['total_time_seconds']:.1f} saniye |\n\n")
            
            # Optimizer parametreleri
            f.write("**Optimizer Parametreleri:**\n```python\n")
            for key, value in result['optimizer_params'].items():
                f.write(f"{key} = {value}\n")
            f.write("```\n\n")
        
        # Training curves referansı
        f.write("---\n\n")
        f.write("## 📉 Training Curves\n\n")
        f.write(f"![Training Curves](./figures/training_curves_comparison_{plastic_type}.png)\n\n")
        f.write(f"![Optimizer Comparison](./figures/optimizer_comparison_{plastic_type}.png)\n\n")
        
        # Overfitting analizi
        f.write("---\n\n")
        f.write("## 🔍 Overfitting Analizi\n\n")
        f.write("| Optimizer | Train R² (son) | Val R² (son) | Gap | Durum |\n")
        f.write("|-----------|----------------|--------------|-----|-------|\n")
        
        for result in sorted_results:
            history = result['history']
            if history['train_r2'] and history['val_r2']:
                train_r2_last = history['train_r2'][-1]
                val_r2_last = history['val_r2'][-1]
                gap = train_r2_last - val_r2_last
                status = "✅ İyi" if gap < 0.02 else "⚠️ Hafif" if gap < 0.05 else "❌ Yüksek"
                f.write(f"| {result['optimizer']} | {train_r2_last:.4f} | {val_r2_last:.4f} | {gap:.4f} | {status} |\n")
        f.write("\n")
        
        # Sonuçlar ve öneriler
        f.write("---\n\n")
        f.write("## 💡 Sonuçlar ve Öneriler\n\n")
        f.write(f"### En İyi Optimizer: **{best_result['optimizer']}**\n\n")
        
        # Otomatik analiz
        if best_result['optimizer'] in ['Adam', 'AdamW', 'NAdam', 'RAdam']:
            f.write("- ✅ Adam ailesi optimizer'ları bu veri seti için optimal performans gösterdi\n")
            f.write("- ✅ Adaptive learning rate yaklaşımı ENCDEC modeli ile iyi uyum sağlıyor\n")
        elif 'SGD' in best_result['optimizer']:
            f.write("- ✅ SGD tabanlı optimizer bu veri seti için en iyi sonuç verdi\n")
            f.write("- ✅ Momentum ile birlikte kullanımı kritik öneme sahip\n")
        
        f.write(f"\n### Performans Özeti\n\n")
        f.write(f"- **En yüksek R²:** {best_result['test_r2']:.6f} ({best_result['optimizer']})\n")
        f.write(f"- **En düşük R²:** {sorted_results[-1]['test_r2']:.6f} ({sorted_results[-1]['optimizer']})\n")
        f.write(f"- **R² Farkı:** {best_result['test_r2'] - sorted_results[-1]['test_r2']:.6f}\n")
        
        avg_time = np.mean([r['total_time_seconds'] for r in all_results])
        f.write(f"- **Ortalama eğitim süresi:** {avg_time:.1f} saniye\n\n")
        
        # Dosya yapısı
        f.write("---\n\n")
        f.write("## 📁 Oluşturulan Dosyalar\n\n")
        f.write("```\n")
        f.write(f"results/optimization_experiments/\n")
        f.write(f"├── figures/\n")
        f.write(f"│   ├── training_curves_comparison_{plastic_type}.png\n")
        f.write(f"│   └── optimizer_comparison_{plastic_type}.png\n")
        f.write(f"├── tables/\n")
        f.write(f"│   ├── optimizer_comparison_summary.csv\n")
        f.write(f"│   └── generated_peptides_*_{plastic_type}.csv\n")
        f.write(f"├── models/\n")
        f.write(f"│   └── encdec_{plastic_type}_*.pth\n")
        f.write(f"├── logs/\n")
        f.write(f"│   └── optimization_experiment_{plastic_type}.json\n")
        f.write(f"└── OPTIMIZER_RAPORU_{plastic_type}.md  ← Bu rapor\n")
        f.write("```\n\n")
        
        # Footer
        f.write("---\n\n")
        f.write(f"*Bu rapor `optimization_experiments.py` tarafından otomatik oluşturulmuştur.*\n")
        f.write(f"*Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"✓ Markdown raporu oluşturuldu: {report_path}")
    return report_path

def generate_all_plastics_report(all_results):
    """Tüm plastikler için genel özet raporu"""
    
    report_path = os.path.join(RESULTS_DIR, 'OPTIMIZER_RAPORU_GENEL.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🔬 Optimizer Karşılaştırma Raporu - Tüm Plastikler\n\n")
        f.write(f"**Tarih:** {datetime.now().strftime('%d %B %Y, %H:%M')}\n")
        f.write(f"**Model:** ENCDEC (Encoder-Decoder)\n")
        f.write(f"**Platform:** Windows 11, RTX 4080 Super\n\n")
        f.write("---\n\n")
        
        f.write("## 📊 Plastik Bazında En İyi Optimizer\n\n")
        f.write("| Plastik | En İyi Optimizer | Test R² | Test MAE | Model Params |\n")
        f.write("|---------|------------------|---------|----------|---------------|\n")
        
        for plastic, results in all_results.items():
            if results:
                best = max(results, key=lambda x: x['test_r2'])
                params = best['model_params']
                f.write(f"| **{plastic}** | {best['optimizer']} | **{best['test_r2']:.4f}** | ")
                f.write(f"{best['test_mae']:.2f} | layers={params['num_layers']}, drop={params['dropout']} |\n")
        
        f.write("\n---\n\n")
        f.write("## 📈 Optimizer Başarı Sıralaması (Tüm Plastikler)\n\n")
        
        # Her optimizer için ortalama performans
        optimizer_scores = {}
        for plastic, results in all_results.items():
            for r in results:
                if r['optimizer'] not in optimizer_scores:
                    optimizer_scores[r['optimizer']] = []
                optimizer_scores[r['optimizer']].append(r['test_r2'])
        
        avg_scores = [(opt, np.mean(scores)) for opt, scores in optimizer_scores.items()]
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        
        f.write("| Sıra | Optimizer | Ortalama R² | Plastik Sayısı |\n")
        f.write("|------|-----------|-------------|----------------|\n")
        for i, (opt, avg) in enumerate(avg_scores):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}"
            f.write(f"| {medal} | **{opt}** | {avg:.6f} | {len(optimizer_scores[opt])} |\n")
        
        f.write(f"\n---\n\n*Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"✓ Genel rapor oluşturuldu: {report_path}")
    return report_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_optimization_experiments(plastic_type='PP', optimizers=None):
    """
    Belirli bir plastik için tüm optimizer deneylerini çalıştır
    
    Args:
        plastic_type: Test edilecek plastik tipi
        optimizers: Test edilecek optimizer listesi (None=hepsi)
    """
    
    print("="*80)
    print("🔬 OPTIMIZATION EXPERIMENTS")
    print(f"   Plastik: {plastic_type}")
    print(f"   Device: {device}")
    print(f"   Data Dir: {DATA_DIR}")
    print(f"   Results Dir: {RESULTS_DIR}")
    print("="*80)
    
    # Optimizer konfigürasyonları
    all_configs = get_optimizer_configs()
    
    if optimizers is None:
        optimizers = list(all_configs.keys())
    
    print(f"\n📋 Test edilecek optimizer'lar ({len(optimizers)}):")
    for opt in optimizers:
        print(f"   - {opt}: {all_configs[opt]['description']}")
    
    # Her optimizer için eğitim
    all_results = []
    
    for optimizer_name in optimizers:
        config = all_configs[optimizer_name]
        result = train_with_optimizer(
            plastic_type=plastic_type,
            optimizer_name=optimizer_name,
            optimizer_config=config,
            epochs=250,
            seed=42
        )
        if result:
            all_results.append(result)
    
    # Görselleştirmeler
    if all_results:
        plot_training_curves(all_results, plastic_type)
        plot_final_comparison(all_results, plastic_type)
        summary_df = create_summary_table(all_results)
        
        # En iyi optimizer ile peptid üret
        best_result = max(all_results, key=lambda x: x['test_r2'])
        best_optimizer = best_result['optimizer']
        print(f"\n🏆 En iyi optimizer: {best_optimizer} (R²={best_result['test_r2']:.4f})")
        
        model_path = os.path.join(MODELS_DIR, f"encdec_{plastic_type}_{best_optimizer}.pth")
        if os.path.exists(model_path):
            generate_peptides_for_optimizer(model_path, plastic_type, best_optimizer, n_peptides=30)
        
        # JSON log kaydet
        log_path = os.path.join(LOGS_DIR, f'optimization_experiment_{plastic_type}.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            # History'yi JSON serializable yap
            results_for_json = []
            for r in all_results:
                r_copy = r.copy()
                r_copy['history'] = {k: [float(v) for v in vals] for k, vals in r['history'].items()}
                results_for_json.append(r_copy)
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        print(f"✓ Detaylı log kaydedildi: {log_path}")
        
        # Markdown rapor oluştur
        generate_markdown_report(all_results, plastic_type)
        
        print("\n" + "="*80)
        print("✅ DENEYLER TAMAMLANDI")
        print("="*80)
        print(f"\n📊 Sonuç Dosyaları:")
        print(f"   - Figures: {FIGURES_DIR}")
        print(f"   - Tables: {TABLES_DIR}")
        print(f"   - Models: {MODELS_DIR}")
        print(f"   - Logs: {LOGS_DIR}")
        print(f"   - Rapor: OPTIMIZER_RAPORU_{plastic_type}.md")
    
    return all_results

# ============================================================================
# TÜM PLASTİKLER İÇİN ÇALIŞTIR
# ============================================================================

def run_all_plastics(optimizers=None):
    """Tüm plastik tipleri için optimizer deneylerini çalıştır"""
    
    all_plastics = ['PET', 'PP', 'PE', 'PVC', 'Nylon']
    all_results = {}
    
    print("="*80)
    print("🔬 TÜM PLASTİKLER İÇİN OPTİMİZASYON DENEYLERİ")
    print(f"   Plastikler: {', '.join(all_plastics)}")
    print("="*80)
    
    for plastic in all_plastics:
        print(f"\n{'#'*80}")
        print(f"# {plastic} BAŞLIYOR")
        print(f"{'#'*80}")
        
        results = run_optimization_experiments(
            plastic_type=plastic,
            optimizers=optimizers
        )
        all_results[plastic] = results
    
    # Genel özet tablo
    print("\n" + "="*80)
    print("📊 GENEL ÖZET - TÜM PLASTİKLER")
    print("="*80)
    
    summary_rows = []
    for plastic, results in all_results.items():
        if results:
            best = max(results, key=lambda x: x['test_r2'])
            summary_rows.append({
                'Plastic': plastic,
                'Best_Optimizer': best['optimizer'],
                'Test_R2': best['test_r2'],
                'Test_MAE': best['test_mae'],
                'Time': best['total_time_seconds']
            })
            print(f"   {plastic}: {best['optimizer']} → R²={best['test_r2']:.4f}")
    
    # Genel özet CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(TABLES_DIR, 'all_plastics_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Genel özet kaydedildi: {summary_path}")
        
        # Genel Markdown rapor oluştur
        generate_all_plastics_report(all_results)
    
    return all_results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Optimization Experiments - ENCDEC Model')
    parser.add_argument('--plastic', type=str, default='PP', 
                       choices=['PET', 'PP', 'PE', 'PVC', 'Nylon', 'ALL'],
                       help='Plastik tipi (ALL = tüm plastikler)')
    parser.add_argument('--optimizers', type=str, nargs='+', default=None,
                       help='Test edilecek optimizer listesi (boş = hepsi)')
    parser.add_argument('--epochs', type=int, default=250,
                       help='Eğitim epoch sayısı')
    args = parser.parse_args()
    
    print("="*80)
    print("🧬 OPTIMIZATION EXPERIMENTS - ENCDEC MODEL")
    print("="*80)
    print(f"   Platform: Windows 11 + RTX 4080 Super")
    print(f"   Device: {device}")
    print(f"   Plastik: {args.plastic}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Optimizers: {args.optimizers if args.optimizers else 'Tümü (11 adet)'}")
    print("="*80)
    
    if args.plastic == 'ALL':
        results = run_all_plastics(optimizers=args.optimizers)
    else:
        results = run_optimization_experiments(
            plastic_type=args.plastic,
            optimizers=args.optimizers
        )
