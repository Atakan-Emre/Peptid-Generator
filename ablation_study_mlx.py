# ============================================================================
# ABLATION STUDY - APPLE SILICON M4 MLX VERSION
# ============================================================================
# M4 Mac Mini 16GB RAM için MLX ile optimize edilmiştir
# MLX: Apple'ın native ML framework'ü - MPS'ten 2-3x daha hızlı
# ============================================================================

import os
import sys
import json
import math
import random
import itertools
import warnings
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# MLX Import
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    MLX_AVAILABLE = True
    print("✓ MLX framework yüklendi - Apple Silicon native")
except ImportError:
    MLX_AVAILABLE = False
    print("⚠ MLX bulunamadı! pip install mlx")
    sys.exit(1)

# ============================================================================
# Konfigürasyon
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

# Dizinler
ABLATION_DIR = os.path.join(PROJECT_ROOT, "ablation_results_mlx")
ABLATION_TABLES_DIR = os.path.join(ABLATION_DIR, "tables")
ABLATION_FIGURES_DIR = os.path.join(ABLATION_DIR, "figures")
ABLATION_TRAINING_CURVES_DIR = os.path.join(ABLATION_FIGURES_DIR, "training_curves")
FINAL_MODELS_DIR = os.path.join(ABLATION_DIR, "final_models")
ABLATION_LOGS_DIR = os.path.join(ABLATION_DIR, "logs")
CHECKPOINT_DIR = os.path.join(ABLATION_DIR, "checkpoints")

for d in [ABLATION_TABLES_DIR, ABLATION_FIGURES_DIR, ABLATION_TRAINING_CURVES_DIR, 
          FINAL_MODELS_DIR, ABLATION_LOGS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# Veri klasörü
DATA_DIR = os.path.join(PROJECT_ROOT, "newDate")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# M4 16GB RAM için optimize ayarlar
@dataclass
class M4Config:
    default_batch_size: int = 256
    ablation_epochs: int = 50
    final_epochs: int = 250
    patience: int = 15
    max_hidden_dim: int = 512

M4_CONFIG = M4Config()

# Sabitler
AMINO_ACIDS = "ADEFGHIKLMNQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS)

# ============================================================================
# Logger ve Overfitting Detection
# ============================================================================
class AblationLogger:
    """Detaylı log sistemi"""
    def __init__(self, model_type: str, plastic_type: str):
        self.model_type = model_type
        self.plastic_type = plastic_type
        self.start_time = datetime.now()
        self.log_path = os.path.join(ABLATION_LOGS_DIR, f'ablation_{model_type}_{plastic_type}.log')
        self.json_path = os.path.join(ABLATION_LOGS_DIR, f'ablation_{model_type}_{plastic_type}.json')
        self.logs = {'model': model_type, 'plastic': plastic_type, 'start': self.start_time.isoformat(),
                    'combinations': [], 'warnings': []}
        with open(self.log_path, 'w') as f:
            f.write(f"{'='*70}\nABLATION LOG: {model_type.upper()} - {plastic_type}\n{self.start_time}\n{'='*70}\n\n")
    
    def log(self, msg, level="INFO"):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}\n"
        with open(self.log_path, 'a') as f: f.write(line)
        if level == "WARNING": print(f"⚠ {msg}")
    
    def log_combo(self, idx, params, metrics, overfitting):
        self.logs['combinations'].append({'id': idx, 'params': params, 'metrics': metrics, 'overfitting': overfitting})
        if overfitting.get('is_overfitting'):
            self.logs['warnings'].append(f"Combo #{idx}: Overfitting (gap={overfitting['gap']:.4f})")
            self.log(f"Combo #{idx}: ⚠ OVERFITTING", "WARNING")
    
    def finalize(self, summary):
        self.logs['summary'] = summary
        self.logs['duration'] = (datetime.now() - self.start_time).total_seconds()
        with open(self.json_path, 'w') as f: json.dump(self.logs, f, indent=2, default=str)
        self.log(f"✓ Tamamlandı: {self.logs['duration']:.1f}s")

def detect_overfitting(train_losses, val_losses) -> dict:
    """Overfitting tespit et"""
    if len(train_losses) < 5: return {'is_overfitting': False, 'gap': 0, 'severity': 'none'}
    last_n = min(10, len(train_losses))
    gap = np.mean(train_losses[-last_n:]) - np.mean(val_losses[-last_n:])
    train_trend = train_losses[-1] - train_losses[-last_n]
    val_trend = val_losses[-1] - val_losses[-last_n]
    is_overfit = (train_trend < -0.01 and val_trend > 0.01) or gap < -0.1
    severity = 'high' if gap < -0.2 else ('medium' if gap < -0.1 else ('low' if gap < -0.05 else 'none'))
    return {'is_overfitting': is_overfit, 'gap': float(gap), 'severity': severity,
            'train_trend': float(train_trend), 'val_trend': float(val_trend)}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

# ============================================================================
# MLX Checkpoint Functions
# ============================================================================
def flatten_params(d, prefix=''):
    """MLX parametrelerini düzleştir"""
    result = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_params(v, key))
        else:
            result[key] = np.array(v)
    return result

def save_checkpoint(checkpoint_path, model, epoch, best_val_r2, best_epoch, patience_ctr,
                   train_losses, val_losses, val_r2s, params, model_type, plastic_type, combo_id):
    """Epoch-level checkpoint kaydet"""
    checkpoint_data = {
        'epoch': epoch,
        'model_params': flatten_params(model.parameters()),
        'best_val_r2': best_val_r2,
        'best_epoch': best_epoch,
        'patience_ctr': patience_ctr,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_r2s': val_r2s,
        'params': params,
        'model_type': model_type,
        'plastic_type': plastic_type,
        'combination_id': combo_id
    }
    np.savez(checkpoint_path, **{k: v if isinstance(v, np.ndarray) else np.array([v], dtype=object) 
                                  for k, v in checkpoint_data.items()})

def load_checkpoint(checkpoint_path):
    """Checkpoint yükle"""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        data = np.load(checkpoint_path, allow_pickle=True)
        checkpoint = {}
        for k in data.files:
            v = data[k]
            if v.dtype == object and v.size == 1:
                checkpoint[k] = v.item()
            else:
                checkpoint[k] = v
        return checkpoint
    except Exception as e:
        print(f"  ⚠ Checkpoint yüklenemedi: {e}")
        return None

def one_hot_encode(seqs: List[str]) -> np.ndarray:
    encoded = []
    for seq in seqs:
        oh = np.zeros((12, NUM_AA), dtype=np.float32)
        for i, aa in enumerate(seq[:12]):
            if aa in AA_TO_IDX:
                oh[i, AA_TO_IDX[aa]] = 1.0
        encoded.append(oh)
    return np.stack(encoded)

# ============================================================================
# MLX Models
# ============================================================================
class MLXLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wxi = nn.Linear(input_dim, hidden_dim)
        self.Whi = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wxf = nn.Linear(input_dim, hidden_dim)
        self.Whf = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wxc = nn.Linear(input_dim, hidden_dim)
        self.Whc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wxo = nn.Linear(input_dim, hidden_dim)
        self.Who = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def __call__(self, x, h, c):
        i = mx.sigmoid(self.Wxi(x) + self.Whi(h))
        f = mx.sigmoid(self.Wxf(x) + self.Whf(h))
        c_tilde = mx.tanh(self.Wxc(x) + self.Whc(h))
        c_new = f * c + i * c_tilde
        o = mx.sigmoid(self.Wxo(x) + self.Who(h))
        h_new = o * mx.tanh(c_new)
        return h_new, c_new

class MLXLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = [MLXLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) 
                      for i in range(num_layers)]
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def __call__(self, x, training: bool = True):
        batch_size, seq_len, _ = x.shape
        h = [mx.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        c = [mx.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(x_t, h[layer_idx], c[layer_idx])
                x_t = h[layer_idx]
                if self.dropout and layer_idx < self.num_layers - 1 and training:
                    x_t = self.dropout(x_t)
            outputs.append(h[-1])
        return mx.stack(outputs, axis=1), (h[-1], c[-1])

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=256, num_layers=2, dropout=0.1, use_layernorm=False):
        super().__init__()
        self.lstm = MLXLSTM(input_dim, hidden_dim, num_layers, dropout)
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else None
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
    
    def __call__(self, x, training=True):
        out, _ = self.lstm(x, training)
        out = out[:, -1, :]
        if self.ln: out = self.ln(out)
        out = mx.tanh(out)
        out = mx.maximum(self.fc1(out), 0)  # relu
        if training: out = self.dropout(out)
        return self.fc2(out).squeeze(-1)

class CNNRegressor(nn.Module):
    """CNN using Linear layers (more stable in MLX)"""
    def __init__(self, input_channels=18, dropout=0.2, use_batchnorm=False):
        super().__init__()
        # Use Linear layers instead of Conv1d for better MLX compatibility
        self.fc1 = nn.Linear(12 * input_channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.use_bn = use_batchnorm
    
    def __call__(self, x, training=True):
        # x: (batch, seq_len=12, channels=18)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Flatten to (batch, 12*18)
        x = mx.maximum(self.fc1(x), 0)  # relu
        if training: x = self.dropout(x)
        x = mx.maximum(self.fc2(x), 0)
        if training: x = self.dropout(x)
        x = mx.maximum(self.fc3(x), 0)
        return self.fc4(x).squeeze(-1)

class LSTMVAE(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=256, latent_dim=64, num_layers=2, dropout=0.1, use_layernorm=False):
        super().__init__()
        self.encoder = MLXLSTM(input_dim, hidden_dim, num_layers, dropout)
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else None
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLXLSTM(input_dim, hidden_dim, num_layers, dropout)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.score_fc = nn.Linear(latent_dim, 1)
    
    def __call__(self, x, training=True):
        batch_size, seq_len, input_dim = x.shape
        enc_out, _ = self.encoder(x, training)
        h = enc_out[:, -1, :]
        if self.ln: h = self.ln(h)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + mx.exp(0.5 * logvar) * mx.random.normal(mu.shape) if training else mu
        dec_input = mx.zeros((batch_size, seq_len, input_dim))
        dec_out, _ = self.decoder(dec_input, training)
        recon = self.out_proj(dec_out)
        score = self.score_fc(mx.tanh(z)).squeeze(-1)
        return recon, mu, logvar, score

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=256, num_layers=2, dropout=0.1, use_layernorm=False):
        super().__init__()
        self.encoder = MLXLSTM(input_dim, hidden_dim, num_layers, dropout)
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else None
        self.decoder = MLXLSTM(input_dim, hidden_dim, num_layers, dropout)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.score_fc = nn.Linear(hidden_dim, 1)
    
    def __call__(self, x, training=True):
        batch_size, seq_len, input_dim = x.shape
        enc_out, _ = self.encoder(x, training)
        h = enc_out[:, -1, :]
        if self.ln: h = self.ln(h)
        score = self.score_fc(mx.tanh(h)).squeeze(-1)
        dec_input = mx.zeros((batch_size, seq_len, input_dim))
        dec_out, _ = self.decoder(dec_input, training)
        recon = self.out_proj(dec_out)
        return recon, score

# ============================================================================
# Data Loading
# ============================================================================
class MLXDataLoader:
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cls = np.argmax(X, axis=2).astype(np.int32)
    
    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        idx = np.arange(len(self.X))
        if self.shuffle: np.random.shuffle(idx)
        for i in range(0, len(self.X), self.batch_size):
            b = idx[i:i + self.batch_size]
            yield mx.array(self.X[b]), mx.array(self.y[b]), mx.array(self.cls[b])

def load_data(plastic_type, data_dir, seed=42):
    df = pd.read_csv(os.path.join(data_dir, f"{plastic_type}.csv"))
    df = df.dropna(subset=['Sequence', 'Score'])
    df['Sequence'] = df['Sequence'].str.strip().str.upper()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    n = len(df)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    
    X_train = one_hot_encode(df['Sequence'][:n_train].tolist())
    X_val = one_hot_encode(df['Sequence'][n_train:n_train+n_val].tolist())
    X_test = one_hot_encode(df['Sequence'][n_train+n_val:].tolist())
    
    y_train = df['Score'][:n_train].values.astype(np.float32)
    y_val = df['Score'][n_train:n_train+n_val].values.astype(np.float32)
    y_test = df['Score'][n_train+n_val:].values.astype(np.float32)
    
    mean, std = y_train.mean(), y_train.std() + 1e-8
    return {
        'X_train': X_train, 'y_train': (y_train - mean) / std, 'y_train_orig': y_train,
        'X_val': X_val, 'y_val': (y_val - mean) / std, 'y_val_orig': y_val,
        'X_test': X_test, 'y_test': (y_test - mean) / std, 'y_test_orig': y_test,
        'score_mean': mean, 'score_std': std
    }

# ============================================================================
# Training
# ============================================================================
def mse_loss(pred, target):
    return mx.mean((pred - target) ** 2)

def ce_loss(logits, targets):
    """Cross entropy loss using MLX nn.losses"""
    # logits: (batch, seq, classes), targets: (batch, seq)
    return nn.losses.cross_entropy(logits, targets, reduction='mean')

def kl_div(mu, logvar):
    return -0.5 * mx.mean(1 + logvar - mu ** 2 - mx.exp(logvar))

def create_model(model_type, params):
    if model_type == 'lstm':
        return LSTMRegressor(NUM_AA, params.get('hidden_dim', 256), 
                            params.get('num_layers', 2), params.get('dropout', 0.1),
                            params.get('use_layernorm', False))
    elif model_type == 'cnn':
        return CNNRegressor(NUM_AA, params.get('dropout', 0.2), params.get('use_batchnorm', False))
    elif model_type == 'lstm_vae':
        return LSTMVAE(NUM_AA, params.get('hidden_dim', 256), params.get('latent_dim', 64),
                      params.get('num_layers', 2), params.get('dropout', 0.1),
                      params.get('use_layernorm', False))
    elif model_type == 'encdec':
        return LSTMEncoderDecoder(NUM_AA, params.get('hidden_dim', 256),
                                  params.get('num_layers', 2), params.get('dropout', 0.1),
                                  params.get('use_layernorm', False))

def get_loss_fn(model_type, beta_kl=0.1, gamma_score=1.0, lambda_score=0.7):
    def lstm_loss(m, X, y, c, t=True): return mse_loss(m(X, t), y)
    def cnn_loss(m, X, y, c, t=True): return mse_loss(m(X, t), y)
    def vae_loss(m, X, y, c, t=True):
        r, mu, lv, s = m(X, t)
        return ce_loss(r, c) + beta_kl * kl_div(mu, lv) + gamma_score * mse_loss(s, y)
    def enc_loss(m, X, y, c, t=True):
        r, s = m(X, t)
        return ce_loss(r, c) + lambda_score * mse_loss(s, y)
    return {'lstm': lstm_loss, 'cnn': cnn_loss, 'lstm_vae': vae_loss, 'encdec': enc_loss}[model_type]

def train_epoch(model, loader, opt, loss_fn):
    loss_grad = nn.value_and_grad(model, loss_fn)
    total, n = 0.0, 0
    for X, y, c in loader:
        loss, grads = loss_grad(model, X, y, c, True)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        total += loss.item()
        n += 1
    return total / n if n else 0

def evaluate(model, loader, loss_fn, mean, std, model_type):
    total, n, preds, targets = 0.0, 0, [], []
    for X, y, c in loader:
        total += loss_fn(model, X, y, c, False).item()
        if model_type in ('lstm', 'cnn'): p = model(X, False)
        elif model_type == 'lstm_vae': _, _, _, p = model(X, False)
        else: _, p = model(X, False)
        preds.append(np.array(p * std + mean))
        targets.append(np.array(y * std + mean))
        n += 1
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    return {'loss': total/n, 'r2': r2_score(targets, preds),
            'mae': mean_absolute_error(targets, preds),
            'rmse': np.sqrt(mean_squared_error(targets, preds))}

# ============================================================================
# Visualization Functions - Bilimsel Makale Kalitesinde
# ============================================================================
def plot_training_curves(train_losses, val_losses, val_r2s, model_type, plastic_type, best_ep, overfit):
    """Training curves grafiği"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Curves: {model_type.upper()} - {plastic_type}', fontsize=14, fontweight='bold')
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    axes[0,0].plot(epochs, train_losses, 'b-', label='Train', lw=2)
    axes[0,0].plot(epochs, val_losses, 'r-', label='Val', lw=2)
    axes[0,0].axvline(best_ep, color='g', ls='--', alpha=0.7, label=f'Best: {best_ep}')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    
    # Loss Log
    axes[0,1].semilogy(epochs, train_losses, 'b-', label='Train', lw=2)
    axes[0,1].semilogy(epochs, val_losses, 'r-', label='Val', lw=2)
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Loss (Log)')
    axes[0,1].set_title('Loss (Log Scale)'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
    
    # Val R²
    axes[1,0].plot(epochs, val_r2s, 'g-', lw=2)
    axes[1,0].axvline(best_ep, color='r', ls='--', alpha=0.7)
    axes[1,0].axhline(max(val_r2s), color='b', ls=':', alpha=0.5, label=f'Max: {max(val_r2s):.4f}')
    axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('R²')
    axes[1,0].set_title('Validation R²'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
    
    # Overfitting
    gap = [t - v for t, v in zip(train_losses, val_losses)]
    colors = ['red' if g < -0.05 else 'green' for g in gap]
    axes[1,1].bar(epochs, gap, color=colors, alpha=0.7)
    axes[1,1].axhline(0, color='k', lw=1); axes[1,1].axhline(-0.05, color='orange', ls='--')
    status = "⚠ OVERFITTING" if overfit['is_overfitting'] else "✓ OK"
    axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Train-Val Gap')
    axes[1,1].set_title(f'Overfitting ({status})'); axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_TRAINING_CURVES_DIR, f'curves_{model_type}_{plastic_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_ablation_results(df, model_type, plastic_type):
    """Ablation sonuçları grafiği"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Ablation Results: {model_type.upper()} - {plastic_type}', fontsize=14, fontweight='bold')
    
    # R² Dağılımı
    axes[0,0].hist(df['test_r2'], bins=20, edgecolor='k', alpha=0.7, color='steelblue')
    axes[0,0].axvline(df['test_r2'].max(), color='r', ls='--', lw=2, label=f'Best: {df["test_r2"].max():.4f}')
    axes[0,0].axvline(df['test_r2'].mean(), color='g', ls='--', lw=2, label=f'Mean: {df["test_r2"].mean():.4f}')
    axes[0,0].set_xlabel('Test R²'); axes[0,0].set_ylabel('Frekans')
    axes[0,0].set_title('Test R² Dağılımı'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    
    # Val vs Test
    sc = axes[0,1].scatter(df['val_r2'], df['test_r2'], c=df['test_r2'], cmap='viridis', alpha=0.6, s=40)
    axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    best_idx = df['test_r2'].idxmax()
    axes[0,1].scatter(df.loc[best_idx, 'val_r2'], df.loc[best_idx, 'test_r2'], c='r', s=150, marker='*', edgecolors='k', zorder=5)
    axes[0,1].set_xlabel('Val R²'); axes[0,1].set_ylabel('Test R²')
    axes[0,1].set_title('Validation vs Test R²'); axes[0,1].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[0,1])
    
    # Top 10
    top10 = df.nlargest(10, 'test_r2')
    colors = ['gold'] + ['steelblue']*9
    axes[1,0].barh(range(10), top10['test_r2'], color=colors, edgecolor='k', alpha=0.8)
    axes[1,0].set_yticks(range(10)); axes[1,0].set_yticklabels([f"#{i+1}" for i in range(10)])
    axes[1,0].set_xlabel('Test R²'); axes[1,0].set_title('Top 10'); axes[1,0].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(top10['test_r2']): axes[1,0].text(v+0.002, i, f'{v:.4f}', va='center', fontsize=8)
    
    # Param Importance
    importance = {}
    for col in df.columns:
        if col not in ['combination_id', 'val_r2', 'test_r2', 'test_mae', 'test_rmse', 'is_best', 'best_epoch', 'overfit_gap', 'is_overfitting']:
            if df[col].dtype in ['int64', 'float64']:
                corr = df[col].corr(df['test_r2'])
                if not np.isnan(corr): importance[col] = abs(corr)
    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]
        params, imps = zip(*sorted_imp)
        axes[1,1].barh(range(len(params)), imps, color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(params))), edgecolor='k')
        axes[1,1].set_yticks(range(len(params))); axes[1,1].set_yticklabels(params)
        axes[1,1].set_xlabel('|Correlation|'); axes[1,1].set_title('Parametre Önemi'); axes[1,1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_FIGURES_DIR, f'ablation_{model_type}_{plastic_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_overfitting_analysis(df, model_type, plastic_type):
    """Overfitting analiz grafiği"""
    if 'overfit_gap' not in df.columns: return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Overfitting Analysis: {model_type.upper()} - {plastic_type}', fontsize=14, fontweight='bold')
    
    # Dağılım
    overfit = df[df['is_overfitting'] == True]
    normal = df[df['is_overfitting'] == False]
    axes[0].bar(['Overfitting', 'Normal'], [len(overfit), len(normal)], color=['red', 'green'], alpha=0.7, edgecolor='k')
    axes[0].set_ylabel('Sayı'); axes[0].set_title(f'Dağılım ({len(overfit)}/{len(df)})')
    
    # Val R² vs Gap
    colors = ['red' if o else 'green' for o in df['is_overfitting']]
    axes[1].scatter(df['val_r2'], df['overfit_gap'], c=colors, alpha=0.6, s=40)
    axes[1].axhline(-0.05, color='orange', ls='--'); axes[1].axhline(0, color='k', alpha=0.3)
    axes[1].set_xlabel('Val R²'); axes[1].set_ylabel('Train-Val Gap'); axes[1].set_title('R² vs Overfitting'); axes[1].grid(True, alpha=0.3)
    
    # Best non-overfitting
    non_overfit = df[df['is_overfitting'] == False].nlargest(10, 'val_r2')
    if len(non_overfit) > 0:
        axes[2].barh(range(len(non_overfit)), non_overfit['val_r2'], color='green', alpha=0.7, edgecolor='k')
        axes[2].set_yticks(range(len(non_overfit)))
        axes[2].set_yticklabels([f"#{int(r['combination_id'])}" for _, r in non_overfit.iterrows()])
        axes[2].set_xlabel('Val R²'); axes[2].set_title('Top Non-Overfitting')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_FIGURES_DIR, f'overfitting_{model_type}_{plastic_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison_heatmap(all_results):
    """Model karşılaştırma heatmap"""
    if not all_results: return
    df = pd.DataFrame(all_results)
    pivot = df.pivot_table(values='Test_R2', index='Model', columns='Plastic', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', linewidths=0.5, ax=ax, vmin=0, vmax=1,
               annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Model Performance Heatmap (Test R²)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_FIGURES_DIR, 'model_comparison_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(preds, targets, model_type, plastic_type):
    """Tahmin vs Gerçek grafiği"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(targets, preds, alpha=0.5, s=20, c='steelblue')
    mn, mx = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2)
    r2 = r2_score(targets, preds)
    axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].set_xlabel('Gerçek'); axes[0].set_ylabel('Tahmin'); axes[0].set_title('Predictions vs Actual'); axes[0].grid(True, alpha=0.3)
    
    residuals = preds - targets
    axes[1].scatter(preds, residuals, alpha=0.5, s=20, c='coral')
    axes[1].axhline(0, color='k', lw=1); axes[1].axhline(residuals.std(), color='r', ls='--', alpha=0.7)
    axes[1].axhline(-residuals.std(), color='r', ls='--', alpha=0.7)
    axes[1].set_xlabel('Tahmin'); axes[1].set_ylabel('Residual'); axes[1].set_title('Residual Analysis'); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_FIGURES_DIR, f'predictions_{model_type}_{plastic_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Ablation Study
# ============================================================================
def get_param_grid(model_type):
    """Mac versiyonuyla AYNI kombinasyonlar (batch_size M4 16GB için optimize)"""
    # M4 16GB için batch size'lar (Mac versiyonuyla aynı)
    bs = [128, 256, 512]
    
    if model_type == 'lstm':
        # Mac: 2*2*3*2*3*2*2 = 144 kombinasyon
        return {
            'hidden_dim': [128, 256],
            'num_layers': [2, 3],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [1e-4, 1e-3],
            'batch_size': bs,
            'weight_decay': [1e-4, 1e-2],
            'use_layernorm': [True, False]
        }
    elif model_type == 'cnn':
        # Mac: 2*2*3*3*2 = 72 kombinasyon
        return {
            'dropout': [0.2, 0.4],
            'learning_rate': [1e-4, 1e-3],
            'batch_size': bs,
            'weight_decay': [1e-4, 1e-3, 1e-2],
            'use_batchnorm': [True, False]
        }
    elif model_type == 'lstm_vae':
        # Mac: 2*2*2*1*2*3*2*1*2*2 = 192 kombinasyon
        return {
            'hidden_dim': [128, 256],
            'num_layers': [2, 3],
            'dropout': [0.1, 0.2],
            'latent_dim': [64],  # Fixed
            'learning_rate': [1e-4, 1e-3],
            'batch_size': bs,
            'beta_kl': [1.0, 2.0],
            'gamma_score': [1.0],  # Fixed
            'weight_decay': [1e-4, 1e-2],
            'use_layernorm': [True, False]
        }
    elif model_type == 'encdec':
        # Mac: 2*2*2*1*3*2*2*2 = 96 kombinasyon
        return {
            'hidden_dim': [128, 256],
            'num_layers': [2, 3],
            'dropout': [0.1, 0.2],
            'learning_rate': [1e-3],  # Fixed
            'batch_size': bs,
            'lambda_score': [0.7, 1.0],
            'weight_decay': [1e-4, 1e-2],
            'use_layernorm': [True, False]
        }
    return {}

def run_ablation(model_type, plastic_type, data_dir=None, epochs=None, seed=42):
    if not data_dir: data_dir = DATA_DIR
    if not epochs: epochs = M4_CONFIG.ablation_epochs
    
    out_path = os.path.join(ABLATION_TABLES_DIR, f'ablation_{model_type}_{plastic_type}.csv')
    if os.path.exists(out_path):
        print(f"✓ {model_type}-{plastic_type} zaten tamamlanmış")
        df = pd.read_csv(out_path)
        best = df.loc[df['val_r2'].idxmax()]
        excl = ['combination_id', 'val_r2', 'test_r2', 'test_mae', 'test_rmse', 'is_best', 'best_epoch', 'overfit_gap', 'is_overfitting']
        params = {c: best[c] for c in df.columns if c not in excl}
        data = load_data(plastic_type, data_dir, seed)
        params['score_mean'], params['score_std'] = data['score_mean'], data['score_std']
        params['best_val_r2'], params['best_test_r2'] = float(best['val_r2']), float(best['test_r2'])
        return params, df
    
    print(f"\n{'='*70}\nABLATION: {model_type.upper()} - {plastic_type}\n{'='*70}")
    
    # Logger başlat
    logger = AblationLogger(model_type, plastic_type)
    data = load_data(plastic_type, data_dir, seed)
    logger.log(f"Veri: Train={len(data['X_train'])}, Val={len(data['X_val'])}, Test={len(data['X_test'])}")
    
    grid = get_param_grid(model_type)
    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    logger.log(f"Toplam {len(combos)} kombinasyon")
    print(f"Toplam {len(combos)} kombinasyon")
    
    # Checkpoint sistemi - kaldığı yerden devam
    checkpoint_path = os.path.join(ABLATION_LOGS_DIR, f'checkpoint_{model_type}_{plastic_type}.csv')
    start_idx = 0
    results = []
    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        results = checkpoint_df.to_dict('records')
        start_idx = len(results)
        print(f"  🔄 Checkpoint bulundu: {start_idx}/{len(combos)} tamamlanmış, devam ediliyor...")
        logger.log(f"Checkpoint'ten devam: {start_idx} kombinasyon atlandı")
    
    best_r2, best_params = -float('inf'), None
    best_history = {'train': [], 'val': [], 'r2': []}
    best_overfit = {}
    overfit_count = 0
    
    # Önceki sonuçlardan best_r2 bul
    if results:
        for r in results:
            if r['val_r2'] > best_r2:
                best_r2 = r['val_r2']
            if r.get('is_overfitting'): overfit_count += 1
    
    for idx, combo in enumerate(tqdm(combos[start_idx:], desc=f"Ablation {model_type}", initial=start_idx, total=len(combos), dynamic_ncols=True, mininterval=1)):
        actual_idx = start_idx + idx
        params = dict(zip(keys, combo))
        seed_everything(seed)
        
        # Her 10 kombinasyonda durum yazdır ve checkpoint kaydet
        if actual_idx > 0 and actual_idx % 10 == 0:
            print(f"\n  [{actual_idx}/{len(combos)}] Best R²: {best_r2:.4f}", flush=True)
            # Checkpoint kaydet
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        
        bs = params.get('batch_size', 128)
        train_l = MLXDataLoader(data['X_train'], data['y_train'], bs, True)
        val_l = MLXDataLoader(data['X_val'], data['y_val'], bs, False)
        test_l = MLXDataLoader(data['X_test'], data['y_test'], bs, False)
        
        model = create_model(model_type, params)
        opt = optim.AdamW(learning_rate=params.get('learning_rate', 1e-3),
                         weight_decay=params.get('weight_decay', 1e-4))
        loss_fn = get_loss_fn(model_type, params.get('beta_kl', 0.1),
                             params.get('gamma_score', 1.0), params.get('lambda_score', 0.7))
        
        # Epoch-level checkpoint path
        combo_checkpoint_path = os.path.join(CHECKPOINT_DIR, f'ablation_{model_type}_{plastic_type}_combo_{actual_idx+1}.npz')
        
        # Epoch history ve checkpoint'ten devam
        train_losses, val_losses, val_r2s = [], [], []
        best_ep_r2, best_ep, patience = -float('inf'), 0, 0
        start_epoch = 0
        
        # Checkpoint'ten devam et (varsa)
        combo_ckpt = load_checkpoint(combo_checkpoint_path)
        if combo_ckpt is not None:
            start_epoch = int(combo_ckpt['epoch']) + 1
            best_ep_r2 = float(combo_ckpt['best_val_r2'])
            best_ep = int(combo_ckpt['best_epoch'])
            patience = int(combo_ckpt['patience_ctr'])
            train_losses = list(combo_ckpt['train_losses'])
            val_losses = list(combo_ckpt['val_losses'])
            val_r2s = list(combo_ckpt['val_r2s'])
            print(f"\n  🔄 Combo #{actual_idx+1} checkpoint: Epoch {start_epoch}/{epochs}, Best R²: {best_ep_r2:.4f}")
            logger.log(f"Combo #{actual_idx+1} checkpoint'ten devam: Epoch {start_epoch}")
        
        for ep in range(start_epoch, epochs):
            tl = train_epoch(model, train_l, opt, loss_fn)
            val_m = evaluate(model, val_l, loss_fn, data['score_mean'], data['score_std'], model_type)
            train_losses.append(tl)
            val_losses.append(val_m['loss'])
            val_r2s.append(val_m['r2'])
            
            if val_m['r2'] > best_ep_r2:
                best_ep_r2, best_ep, patience = val_m['r2'], ep + 1, 0
            else:
                patience += 1
                if patience >= M4_CONFIG.patience: break
            
            # Her 5 epoch'ta checkpoint kaydet
            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                save_checkpoint(combo_checkpoint_path, model, ep, best_ep_r2, best_ep, patience,
                               train_losses, val_losses, val_r2s, params, model_type, plastic_type, actual_idx+1)
        
        # Overfitting analizi
        overfit = detect_overfitting(train_losses, val_losses)
        if overfit['is_overfitting']: overfit_count += 1
        
        test_m = evaluate(model, test_l, loss_fn, data['score_mean'], data['score_std'], model_type)
        
        result = {'combination_id': actual_idx+1, 'val_r2': best_ep_r2, 'test_r2': test_m['r2'],
                  'test_mae': test_m['mae'], 'test_rmse': test_m['rmse'], 'best_epoch': best_ep,
                  'overfit_gap': overfit['gap'], 'is_overfitting': overfit['is_overfitting'], **params}
        results.append(result)
        
        # Kombinasyon tamamlandı - epoch checkpoint'i temizle
        if os.path.exists(combo_checkpoint_path):
            os.remove(combo_checkpoint_path)
        
        # Her kombinasyondan sonra results checkpoint kaydet
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        
        # Logger
        logger.log_combo(actual_idx+1, params, {'val_r2': best_ep_r2, 'test_r2': test_m['r2']}, overfit)
        
        if best_ep_r2 > best_r2:
            best_r2 = best_ep_r2
            best_params = params.copy()
            best_params['score_mean'], best_params['score_std'] = data['score_mean'], data['score_std']
            best_params['best_val_r2'], best_params['best_test_r2'] = best_r2, test_m['r2']
            best_history = {'train': train_losses, 'val': val_losses, 'r2': val_r2s}
            best_overfit = overfit
            best_best_ep = best_ep
        del model, opt
    
    # DataFrame kaydet
    df = pd.DataFrame(results)
    df['is_best'] = df['val_r2'] == best_r2
    df = df.sort_values('val_r2', ascending=False).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    
    # Checkpoint temizle (başarılı tamamlandı)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  ✓ Checkpoint temizlendi")
    
    # Logger finalize
    logger.finalize({'best_val_r2': best_r2, 'best_test_r2': best_params['best_test_r2'],
                    'total': len(combos), 'overfitting': overfit_count})
    
    # Grafikler
    print("\n📊 Grafikler oluşturuluyor...")
    try:
        plot_training_curves(best_history['train'], best_history['val'], best_history['r2'],
                            model_type, plastic_type, best_best_ep, best_overfit)
        plot_ablation_results(df, model_type, plastic_type)
        plot_overfitting_analysis(df, model_type, plastic_type)
    except Exception as e:
        print(f"⚠ Grafik hatası: {e}")
    
    print(f"\n✓ Kaydedildi: {out_path}")
    print(f"🏆 Best Val R²: {best_r2:.4f}, Test R²: {best_params['best_test_r2']:.4f}")
    print(f"⚠ Overfitting: {overfit_count}/{len(combos)} ({100*overfit_count/len(combos):.1f}%)")
    
    return best_params, df

def train_final(model_type, plastic_type, best_params, data_dir=None, epochs=None, seed=42):
    if not data_dir: data_dir = DATA_DIR
    if not epochs: epochs = M4_CONFIG.final_epochs
    
    print(f"\n{'='*70}\nFINAL TRAINING: {model_type.upper()} - {plastic_type}\n{'='*70}")
    data = load_data(plastic_type, data_dir, seed)
    seed_everything(seed)
    
    bs = int(best_params.get('batch_size', 128))
    train_l = MLXDataLoader(data['X_train'], data['y_train'], bs, True)
    val_l = MLXDataLoader(data['X_val'], data['y_val'], bs, False)
    test_l = MLXDataLoader(data['X_test'], data['y_test'], bs, False)
    
    model = create_model(model_type, best_params)
    opt = optim.AdamW(learning_rate=float(best_params.get('learning_rate', 1e-3)),
                     weight_decay=float(best_params.get('weight_decay', 1e-4)))
    loss_fn = get_loss_fn(model_type, float(best_params.get('beta_kl', 0.1)),
                         float(best_params.get('gamma_score', 1.0)),
                         float(best_params.get('lambda_score', 0.7)))
    
    # Final training checkpoint path
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, f'final_{model_type}_{plastic_type}.npz')
    
    train_losses, val_losses, val_r2s = [], [], []
    best_loss, best_ep, patience = float('inf'), 0, 0
    start_epoch = 0
    
    # Checkpoint'ten devam et (varsa)
    final_ckpt = load_checkpoint(final_checkpoint_path)
    if final_ckpt is not None:
        start_epoch = int(final_ckpt['epoch']) + 1
        best_loss = float(final_ckpt.get('best_val_r2', float('inf')))  # Actually best_loss
        best_ep = int(final_ckpt['best_epoch'])
        patience = int(final_ckpt['patience_ctr'])
        train_losses = list(final_ckpt['train_losses'])
        val_losses = list(final_ckpt['val_losses'])
        val_r2s = list(final_ckpt['val_r2s'])
        print(f"  🔄 Final checkpoint: Epoch {start_epoch}/{epochs}, Best Loss: {best_loss:.4f}")
    
    pbar = tqdm(range(start_epoch, epochs), desc=f"Final {model_type}", initial=start_epoch, total=epochs)
    for ep in pbar:
        tl = train_epoch(model, train_l, opt, loss_fn)
        vm = evaluate(model, val_l, loss_fn, data['score_mean'], data['score_std'], model_type)
        train_losses.append(tl)
        val_losses.append(vm['loss'])
        val_r2s.append(vm['r2'])
        pbar.set_postfix({'Train': f'{tl:.4f}', 'Val': f'{vm["loss"]:.4f}', 'R²': f'{vm["r2"]:.4f}'})
        if vm['loss'] < best_loss:
            best_loss, best_ep, patience = vm['loss'], ep + 1, 0
        else:
            patience += 1
            if patience >= M4_CONFIG.patience * 2: break
        
        # Her 10 epoch'ta checkpoint kaydet
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            save_checkpoint(final_checkpoint_path, model, ep, best_loss, best_ep, patience,
                           train_losses, val_losses, val_r2s, best_params, model_type, plastic_type, 0)
    
    # Test evaluation with predictions
    tm = evaluate(model, test_l, loss_fn, data['score_mean'], data['score_std'], model_type)
    
    # Overfitting analizi
    overfit = detect_overfitting(train_losses, val_losses)
    if overfit['is_overfitting']:
        print(f"⚠ UYARI: Overfitting tespit edildi! (Gap: {overfit['gap']:.4f}, Severity: {overfit['severity']})")
    
    print(f"✓ Test R²: {tm['r2']:.4f}, MAE: {tm['mae']:.4f}, RMSE: {tm['rmse']:.4f}")
    
    # Model kaydet
    path = os.path.join(FINAL_MODELS_DIR, f'{model_type}_{plastic_type}_mlx.npz')
    state = flatten_params(model.parameters())
    np.savez(path, **state, score_mean=data['score_mean'], score_std=data['score_std'],
             config=json.dumps({**best_params, 'model_type': model_type, 'plastic_type': plastic_type}))
    print(f"✓ Model kaydedildi: {path}")
    
    # Final checkpoint temizle
    if os.path.exists(final_checkpoint_path):
        os.remove(final_checkpoint_path)
        print("  ✓ Final checkpoint temizlendi")
    
    # Grafikler
    try:
        plot_training_curves(train_losses, val_losses, val_r2s, model_type, plastic_type, best_ep, overfit)
        # Predictions grafiği - test_l'den predictions al
        all_preds, all_targets = [], []
        for X, y, c in test_l:
            if model_type in ('lstm', 'cnn'): p = model(X, False)
            elif model_type == 'lstm_vae': _, _, _, p = model(X, False)
            else: _, p = model(X, False)
            all_preds.append(np.array(p * data['score_std'] + data['score_mean']))
            all_targets.append(np.array(y * data['score_std'] + data['score_mean']))
        plot_predictions(np.concatenate(all_preds), np.concatenate(all_targets), model_type, plastic_type)
    except Exception as e:
        print(f"⚠ Grafik hatası: {e}")
    
    return tm, train_losses, val_losses

# ============================================================================
# Main
# ============================================================================
MODEL_TYPES = ['lstm', 'cnn', 'lstm_vae', 'encdec']

def main():
    print("\n" + "="*80)
    print("🚀 MLX ABLATION STUDY - M4 16GB RAM OPTİMİZE")
    print("="*80)
    print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    plastics = sorted([f[:-4] for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    print(f"\n📋 Plastikler: {plastics}")
    print(f"📋 Modeller: {MODEL_TYPES}")
    print(f"📋 Toplam: {len(plastics)} plastik × {len(MODEL_TYPES)} model = {len(plastics)*len(MODEL_TYPES)} eğitim")
    
    summary = []
    start_time = time.time()
    
    for plastic in plastics:
        for model_type in MODEL_TYPES:
            try:
                best, ablation_df = run_ablation(model_type, plastic)
                metrics, _, _ = train_final(model_type, plastic, best)
                summary.append({
                    'Plastic': plastic, 'Model': model_type,
                    'Test_R2': metrics['r2'], 'MAE': metrics['mae'], 'RMSE': metrics['rmse'],
                    'Best_Val_R2': best.get('best_val_r2', 0)
                })
            except Exception as e:
                print(f"❌ Hata {model_type}-{plastic}: {e}")
                import traceback
                traceback.print_exc()
    
    if summary:
        # Özet DataFrame
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(ABLATION_TABLES_DIR, 'final_summary_mlx.csv'), index=False)
        
        # Model karşılaştırma heatmap
        print("\n📊 Model karşılaştırma heatmap oluşturuluyor...")
        try:
            plot_model_comparison_heatmap(summary)
        except Exception as e:
            print(f"⚠ Heatmap hatası: {e}")
        
        # Özet yazdır
        print("\n" + "="*80)
        print("📊 SONUÇ ÖZETİ")
        print("="*80)
        print(df.to_string(index=False))
        
        # Model bazında ortalama
        print("\n📊 Model Bazında Ortalama R²:")
        model_avg = df.groupby('Model')['Test_R2'].mean().sort_values(ascending=False)
        for m, r in model_avg.items():
            print(f"   {m.upper()}: {r:.4f}")
        
        # Plastik bazında ortalama
        print("\n📊 Plastik Bazında Ortalama R²:")
        plastic_avg = df.groupby('Plastic')['Test_R2'].mean().sort_values(ascending=False)
        for p, r in plastic_avg.items():
            print(f"   {p}: {r:.4f}")
        
        # En iyi kombinasyon
        best_row = df.loc[df['Test_R2'].idxmax()]
        print(f"\n🏆 EN İYİ KOMBİNASYON:")
        print(f"   Plastik: {best_row['Plastic']}")
        print(f"   Model: {best_row['Model']}")
        print(f"   Test R²: {best_row['Test_R2']:.4f}")
        print(f"   MAE: {best_row['MAE']:.4f}")
        print(f"   RMSE: {best_row['RMSE']:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱ Toplam Süre: {elapsed/60:.1f} dakika")
    print(f"\n📁 Sonuçlar: {ABLATION_DIR}")
    print(f"   - Tablolar: {ABLATION_TABLES_DIR}")
    print(f"   - Grafikler: {ABLATION_FIGURES_DIR}")
    print(f"   - Loglar: {ABLATION_LOGS_DIR}")
    print(f"   - Modeller: {FINAL_MODELS_DIR}")
    print("\n✓ ABLATION STUDY TAMAMLANDI!")

if __name__ == '__main__':
    main()
