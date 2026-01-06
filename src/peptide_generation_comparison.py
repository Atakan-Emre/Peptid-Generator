# ============================================================================
# PEPTIDE GENERATION COMPARISON - SA, ILS, MOCO-CEM
# ============================================================================
# 6 plastik tipi için: Eğitim + SA + ILS + MOCO-CEM peptid üretimi
# Platform: Windows 11, RTX 4080 Super (16GB VRAM)
# ============================================================================

import os, sys, torch, torch.nn as nn, torch.optim as optim, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import seaborn as sns; plt.ioff()
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm; import json, random; from datetime import datetime; from dataclasses import dataclass
from typing import Dict, List, Any, Optional; import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sortingData")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "peptide_generation_comparison")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

def get_plastic_dirs(plastic_type):
    """Get directory paths for a specific plastic type"""
    base = os.path.join(RESULTS_DIR, plastic_type)
    return {'base': base, 'model': os.path.join(base, 'model'), 'figures': os.path.join(base, 'figures'),
            'peptides': os.path.join(base, 'peptides'), 'logs': os.path.join(base, 'logs')}

def setup_plastic_dirs(plastic_type):
    """Create directories for a plastic type"""
    dirs = get_plastic_dirs(plastic_type)
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
    return dirs

# Legacy compatibility
FIGURES_DIR, LOGS_DIR = os.path.join(RESULTS_DIR, "figures"), os.path.join(RESULTS_DIR, "logs")
MODELS_DIR, TABLES_DIR = os.path.join(RESULTS_DIR, "models"), os.path.join(RESULTS_DIR, "tables")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
NUM_WORKERS, PIN_MEMORY = 4, True
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True

AMINO_ACIDS = "ADEFGHIKLMNQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

BEST_MODEL_PARAMS = {
    'PET': {'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 1024, 'lambda_score': 1.0, 'weight_decay': 0.0001, 'use_layernorm': True},
    'PP': {'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 1024, 'lambda_score': 1.0, 'weight_decay': 0.0001, 'use_layernorm': True},
    'PE': {'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 1024, 'lambda_score': 1.0, 'weight_decay': 0.0001, 'use_layernorm': True},
    'PVC': {'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 1024, 'lambda_score': 1.0, 'weight_decay': 0.0001, 'use_layernorm': True},
    'PS': {'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 1024, 'lambda_score': 1.0, 'weight_decay': 0.0001, 'use_layernorm': True},
    'Nylon': {'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 128, 'lambda_score': 1.0, 'weight_decay': 0.0001, 'use_layernorm': False},
}

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=256, num_layers=2, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.dec_init = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.score_head = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        enc_out, _ = self.encoder(x); h_last = self.ln(enc_out[:, -1, :])
        score_pred = self.score_head(h_last).squeeze(-1)
        batch_size, seq_len, _ = x.shape
        dec_input = torch.zeros(batch_size, seq_len, x.size(-1), device=x.device)
        h0 = torch.tanh(self.dec_init(h_last)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        dec_out, _ = self.decoder(dec_input, (h0, torch.zeros_like(h0)))
        return self.out_proj(dec_out), score_pred

class PeptideDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        self.class_targets = torch.tensor(np.argmax(X, axis=2), dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.class_targets[idx]

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def one_hot_encode_sequences(sequences):
    encoded = []
    for seq in sequences:
        one_hot = np.zeros((12, len(AMINO_ACIDS)))
        for i, aa in enumerate(seq[:12]):
            if aa in AA_TO_IDX: one_hot[i, AA_TO_IDX[aa]] = 1
        encoded.append(one_hot)
    return np.stack(encoded, axis=0)

# ============================================================================
# 1. SA (SIMULATED ANNEALING) OPTIMIZER
# ============================================================================
class SAPeptideOptimizer:
    def __init__(self, model, model_type, score_mean, score_std, initial_temp=0.5, cooling_rate=0.9, max_iterations=3000):
        self.model, self.model_type = model, model_type
        self.score_mean, self.score_std = score_mean, score_std
        self.temperature, self.initial_temp = initial_temp, initial_temp
        self.cooling_rate, self.max_iterations = cooling_rate, max_iterations
        self.amino_acids = list(AMINO_ACIDS); self.model.eval()
    
    def predict_score(self, peptide):
        one_hot = np.zeros((1, 12, len(AMINO_ACIDS)))
        for i, aa in enumerate(peptide):
            if aa in AA_TO_IDX: one_hot[0, i, AA_TO_IDX[aa]] = 1
        X = torch.tensor(one_hot, dtype=torch.float32).to(device)
        with torch.no_grad():
            _, pred_norm = self.model(X)
            return float(pred_norm.cpu().numpy()[0] * self.score_std + self.score_mean)
    
    def move_operator(self, peptide):
        if random.random() < 0.75:
            pos = random.randint(0, 11)
            new_aa = random.choice([aa for aa in self.amino_acids if aa != peptide[pos]])
            return peptide[:pos] + new_aa + peptide[pos + 1:]
        pos1, pos2 = random.sample(range(12), 2)
        lst = list(peptide); lst[pos1], lst[pos2] = lst[pos2], lst[pos1]
        return ''.join(lst)
    
    def optimize(self, initial_peptide, n_samples=1):
        results = []
        for sample_idx in range(n_samples):
            current = initial_peptide if sample_idx == 0 else ''.join(np.random.choice(self.amino_acids, 12))
            current_score = self.predict_score(current)
            best, best_score, init_score = current, current_score, current_score
            trajectory = [(0, current_score)]; self.temperature = self.initial_temp
            for it in range(self.max_iterations):
                new = self.move_operator(current); new_score = self.predict_score(new)
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / self.temperature):
                    current, current_score = new, new_score
                    trajectory.append((it + 1, current_score))
                    if current_score < best_score: best, best_score = current, current_score
                if (it + 1) % 75 == 0: self.temperature *= self.cooling_rate
                if self.temperature < 0.01: break
            results.append({'best_peptide': best, 'best_score': best_score, 'initial_peptide': initial_peptide,
                           'initial_score': init_score, 'trajectory': trajectory, 'improvement': init_score - best_score, 'method': 'SA'})
        return results

# ============================================================================
# 2. ILS (ITERATED LOCAL SEARCH) OPTIMIZER
# ============================================================================
@dataclass
class ILSOptions:
    max_iterations: int = 800; local_steps: int = 60; kick_strength: int = 1; kick_max: int = 4
    p_segment_kick: float = 0.60; stagnation_iters: int = 50; restart_iters: int = 250; seed: int = 42

class ILSPeptideOptimizer:
    def __init__(self, model, model_type, score_mean, score_std, **kwargs):
        self.opts = ILSOptions()
        for k, v in kwargs.items():
            if hasattr(self.opts, k): setattr(self.opts, k, v)
        self.model, self.model_type = model, model_type
        self.score_mean, self.score_std = float(score_mean), float(score_std)
        self.amino_acids = list(AMINO_ACIDS); self.model.eval()
        random.seed(self.opts.seed); np.random.seed(self.opts.seed)
    
    def predict_score(self, peptide):
        one_hot = np.zeros((1, 12, len(AMINO_ACIDS)), dtype=np.float32)
        for i, aa in enumerate(peptide):
            if aa in AA_TO_IDX: one_hot[0, i, AA_TO_IDX[aa]] = 1
        X = torch.tensor(one_hot, dtype=torch.float32).to(device)
        with torch.no_grad():
            _, pred_norm = self.model(X)
            return float(pred_norm.cpu().numpy()[0] * self.score_std + self.score_mean)
    
    def move_operator(self, peptide):
        if random.random() < 0.75:
            pos = random.randint(0, 11)
            new_aa = random.choice([aa for aa in self.amino_acids if aa != peptide[pos]])
            return peptide[:pos] + new_aa + peptide[pos + 1:]
        pos1, pos2 = random.sample(range(12), 2)
        lst = list(peptide); lst[pos1], lst[pos2] = lst[pos2], lst[pos1]
        return ''.join(lst)
    
    def segment_kick(self, peptide):
        pts = sorted(random.sample(range(1, 12), 4))
        s1, s2, s3, s4, s5 = peptide[:pts[0]], peptide[pts[0]:pts[1]], peptide[pts[1]:pts[2]], peptide[pts[2]:pts[3]], peptide[pts[3]:]
        return s1 + s3 + s2 + s4 + s5
    
    def point_kick(self, peptide):
        for _ in range(random.choice([2, 3])): peptide = self.move_operator(peptide)
        return peptide
    
    def local_search(self, peptide, score):
        for _ in range(self.opts.local_steps):
            cand = self.move_operator(peptide); cs = self.predict_score(cand)
            if cs < score: peptide, score = cand, cs
        return peptide, score
    
    def optimize(self, initial_peptide, n_samples=1):
        results = []
        for sample_idx in range(n_samples):
            start = initial_peptide if sample_idx == 0 else ''.join(np.random.choice(self.amino_acids, 12))
            start_score = self.predict_score(start)
            cur, cur_s = self.local_search(start, start_score)
            best, best_s = cur, cur_s
            trajectory = [(0, best_s)]; kick = self.opts.kick_strength; no_imp = 0
            for it in range(1, self.opts.max_iterations + 1):
                cand = cur
                for _ in range(kick):
                    cand = self.segment_kick(cand) if random.random() < self.opts.p_segment_kick else self.point_kick(cand)
                cand, cand_s = self.local_search(cand, self.predict_score(cand))
                if cand_s < cur_s: cur, cur_s = cand, cand_s
                if cur_s < best_s:
                    best, best_s = cur, cur_s; trajectory.append((it, best_s))
                    no_imp = 0; kick = max(1, kick - 1)
                else:
                    no_imp += 1
                    if no_imp % self.opts.stagnation_iters == 0: kick = min(self.opts.kick_max, kick + 1)
                    if no_imp >= self.opts.restart_iters:
                        cur = best
                        for _ in range(self.opts.kick_max): cur = self.segment_kick(cur)
                        cur, cur_s = self.local_search(cur, self.predict_score(cur))
                        no_imp = 0; kick = self.opts.kick_strength
            results.append({'best_peptide': best, 'best_score': best_s, 'initial_peptide': start,
                           'initial_score': start_score, 'trajectory': trajectory, 'improvement': start_score - best_s, 'method': 'ILS'})
        return results

# ============================================================================
# 3. MOCO-CEM (CROSS-ENTROPY METHOD) OPTIMIZER
# ============================================================================
@dataclass
class MocoOptions:
    max_iterations: int = 300; batch_size: int = 128; elite_frac: float = 0.12; alpha: float = 0.18
    tau_start: float = 1.8; tau_end: float = 0.65; use_local_search: bool = True; local_steps: int = 30; seed: int = 42

class MocoCEMPeptideOptimizer:
    def __init__(self, model, model_type, score_mean, score_std, **kwargs):
        self.opts = MocoOptions()
        for k, v in kwargs.items():
            if hasattr(self.opts, k): setattr(self.opts, k, v)
        self.model, self.model_type = model, model_type
        self.score_mean, self.score_std = float(score_mean), float(score_std)
        self.amino_acids = list(AMINO_ACIDS); self.A = len(self.amino_acids)
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.idx_to_aa = {i: aa for i, aa in enumerate(self.amino_acids)}
        self.model.eval(); random.seed(self.opts.seed); np.random.seed(self.opts.seed)
    
    def predict_score(self, peptide):
        one_hot = np.zeros((1, 12, self.A), dtype=np.float32)
        for i, aa in enumerate(peptide):
            if aa in self.aa_to_idx: one_hot[0, i, self.aa_to_idx[aa]] = 1
        X = torch.tensor(one_hot, dtype=torch.float32).to(device)
        with torch.no_grad():
            _, pred_norm = self.model(X)
            return float(pred_norm.cpu().numpy()[0] * self.score_std + self.score_mean)
    
    def _softmax(self, x): e = np.exp(x - np.max(x)); return e / (np.sum(e) + 1e-12)
    def _tau(self, it):
        K = self.opts.max_iterations
        return self.opts.tau_start + (self.opts.tau_end - self.opts.tau_start) * (it - 1) / max(1, K - 1)
    
    def _sample_peptide(self, theta, tau):
        pep = []
        for pos in range(12):
            p = self._softmax(theta[pos] / tau); p = np.maximum(p, 1e-6); p /= p.sum()
            pep.append(self.idx_to_aa[np.random.choice(self.A, p=p)])
        return ''.join(pep)
    
    def move_operator(self, peptide):
        if random.random() < 0.75:
            pos = random.randint(0, 11)
            new_aa = random.choice([aa for aa in self.amino_acids if aa != peptide[pos]])
            return peptide[:pos] + new_aa + peptide[pos + 1:]
        pos1, pos2 = random.sample(range(12), 2)
        lst = list(peptide); lst[pos1], lst[pos2] = lst[pos2], lst[pos1]
        return ''.join(lst)
    
    def local_search(self, peptide, score):
        for _ in range(self.opts.local_steps):
            cand = self.move_operator(peptide); cs = self.predict_score(cand)
            if cs < score: peptide, score = cand, cs
        return peptide, score
    
    def optimize(self, initial_peptide, n_samples=1):
        results = []
        found_peptides = set()  # Track found peptides for diversity
        elite_count = max(1, int(self.opts.batch_size * self.opts.elite_frac))
        ls_count = max(1, int(self.opts.batch_size * 0.25))
        
        for sample_idx in range(n_samples):
            # Different seed for each sample to ensure diversity
            sample_seed = self.opts.seed + sample_idx * 1000
            random.seed(sample_seed); np.random.seed(sample_seed)
            
            # Different starting point for each sample
            start = initial_peptide if sample_idx == 0 else ''.join(np.random.choice(self.amino_acids, 12))
            
            # Fresh theta initialization for each sample
            theta = np.random.randn(12, self.A) * 0.1  # Increased variance
            for i, aa in enumerate(start):
                if aa in self.aa_to_idx: theta[i, self.aa_to_idx[aa]] += 1.5  # Reduced bias
            
            best, best_score = start, self.predict_score(start)
            start_score = best_score; trajectory = [(0, best_score)]
            
            for it in range(1, self.opts.max_iterations + 1):
                tau = self._tau(it)
                batch = [self._sample_peptide(theta, tau) for _ in range(self.opts.batch_size)]
                scores = np.array([self.predict_score(p) for p in batch])
                
                # Diversity penalty: penalize already found peptides
                for i, pep in enumerate(batch):
                    if pep in found_peptides:
                        scores[i] += 5.0  # Penalty for duplicate
                
                if self.opts.use_local_search:
                    for idx in np.argsort(scores)[:ls_count]:
                        batch[idx], scores[idx] = self.local_search(batch[idx], scores[idx])
                
                min_idx = np.argmin(scores)
                candidate = batch[min_idx]
                # Only accept if different from found or significantly better
                if candidate not in found_peptides or scores[min_idx] < best_score - 1.0:
                    if scores[min_idx] < best_score:
                        best, best_score = candidate, scores[min_idx]
                        trajectory.append((it, best_score))
                
                elite_idx = np.argsort(scores)[:elite_count]
                freq = np.zeros((12, self.A))
                weights = 1.0 / np.arange(1, elite_count + 1); weights /= weights.sum()
                for e, idx in enumerate(elite_idx):
                    for pos, aa in enumerate(batch[idx]):
                        freq[pos, self.aa_to_idx[aa]] += weights[e]
                eliteP = freq / (freq.sum(axis=1, keepdims=True) + 1e-12)
                curP = np.exp(theta); curP /= (curP.sum(axis=1, keepdims=True) + 1e-12)
                Pnew = (1 - self.opts.alpha) * curP + self.opts.alpha * eliteP
                Pnew = np.maximum(Pnew, 1e-6); Pnew /= Pnew.sum(axis=1, keepdims=True)
                theta = np.log(Pnew + 1e-12)
            
            found_peptides.add(best)  # Track this peptide
            results.append({'best_peptide': best, 'best_score': best_score, 'initial_peptide': start,
                           'initial_score': start_score, 'trajectory': trajectory, 'improvement': start_score - best_score, 'method': 'MOCO-CEM'})
        return results

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_model(plastic_type, epochs=250, seed=42):
    dirs = setup_plastic_dirs(plastic_type)  # Create plastic-specific directories
    print(f"\n{'='*80}\n🚀 MODEL EĞİTİMİ: {plastic_type}\n{'='*80}")
    seed_everything(seed)
    params = BEST_MODEL_PARAMS.get(plastic_type, BEST_MODEL_PARAMS['PET'])
    csv_path = os.path.join(DATA_DIR, f"{plastic_type}.csv")
    if not os.path.exists(csv_path): print(f"❌ CSV bulunamadı: {csv_path}"); return None, None, None, None
    
    df = pd.read_csv(csv_path).dropna(subset=['Sequence', 'Score'])
    df['Sequence'] = df['Sequence'].str.strip().str.upper()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df); n_train, n_val = int(0.8 * n), int(0.1 * n)
    train_df, val_df, test_df = df.iloc[:n_train], df.iloc[n_train:n_train+n_val], df.iloc[n_train+n_val:]
    print(f"📊 Veri: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    X_train, X_val, X_test = [one_hot_encode_sequences(d['Sequence'].tolist()) for d in [train_df, val_df, test_df]]
    y_train, y_val, y_test = train_df['Score'].values, val_df['Score'].values, test_df['Score'].values
    score_mean, score_std = np.mean(y_train), np.std(y_train) + 1e-8
    y_train_n, y_val_n, y_test_n = [(y - score_mean) / score_std for y in [y_train, y_val, y_test]]
    
    train_loader = DataLoader(PeptideDataset(X_train, y_train_n), batch_size=params['batch_size'], shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(PeptideDataset(X_val, y_val_n), batch_size=params['batch_size'], shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(PeptideDataset(X_test, y_test_n), batch_size=params['batch_size'], shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    model = LSTMEncoderDecoder(18, params['hidden_dim'], params['num_layers'], params['dropout'], params['use_layernorm']).to(device)
    mse_loss, ce_loss = nn.MSELoss(), nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}
    best_val_loss, best_state, best_epoch, patience_ctr = float('inf'), None, 0, 0
    start_time = datetime.now()
    
    for epoch in tqdm(range(epochs), desc=f"Training {plastic_type}", ncols=100):
        model.train(); train_loss_sum, train_preds, train_targs = 0, [], []
        for X_b, y_b, cls_b in train_loader:
            X_b, y_b, cls_b = X_b.to(device), y_b.to(device), cls_b.to(device)
            optimizer.zero_grad()
            if USE_AMP and scaler:
                with torch.cuda.amp.autocast():
                    recon, score = model(X_b)
                    loss = ce_loss(recon.view(-1, 18), cls_b.view(-1)) + params['lambda_score'] * mse_loss(score, y_b)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            else:
                recon, score = model(X_b)
                loss = ce_loss(recon.view(-1, 18), cls_b.view(-1)) + params['lambda_score'] * mse_loss(score, y_b)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss_sum += loss.item() * X_b.size(0)
            train_preds.extend(score.detach().cpu().numpy()); train_targs.extend(y_b.cpu().numpy())
        
        model.eval(); val_loss_sum, val_preds, val_targs = 0, [], []
        with torch.no_grad():
            for X_b, y_b, cls_b in val_loader:
                X_b, y_b, cls_b = X_b.to(device), y_b.to(device), cls_b.to(device)
                recon, score = model(X_b)
                loss = ce_loss(recon.view(-1, 18), cls_b.view(-1)) + params['lambda_score'] * mse_loss(score, y_b)
                val_loss_sum += loss.item() * X_b.size(0)
                val_preds.extend(score.cpu().numpy()); val_targs.extend(y_b.cpu().numpy())
        
        train_loss, val_loss = train_loss_sum / len(train_df), val_loss_sum / len(val_df)
        train_r2, val_r2 = r2_score(train_targs, train_preds), r2_score(val_targs, val_preds)
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2); history['val_r2'].append(val_r2)
        
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_ctr = 0
        else: patience_ctr += 1
        if patience_ctr >= 15: print(f"\n⏹ Early stopping at epoch {epoch+1}"); break
    
    if best_state: model.load_state_dict(best_state)
    model.eval(); test_preds, test_targs = [], []
    with torch.no_grad():
        for X_b, y_b, _ in test_loader:
            _, score = model(X_b.to(device))
            test_preds.extend(score.cpu().numpy()); test_targs.extend(y_b.numpy())
    test_preds_d = np.array(test_preds) * score_std + score_mean
    test_targs_d = np.array(test_targs) * score_std + score_mean
    test_r2, test_mae = r2_score(test_targs_d, test_preds_d), mean_absolute_error(test_targs_d, test_preds_d)
    test_rmse = np.sqrt(mean_squared_error(test_targs_d, test_preds_d))
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n📊 SONUÇ: Test R²={test_r2:.4f}, MAE={test_mae:.2f}, RMSE={test_rmse:.2f}, Süre={total_time:.0f}s")
    
    # Save to plastic-specific directory
    torch.save({'model_state_dict': best_state, 'score_mean': score_mean, 'score_std': score_std, 'params': params,
                'test_r2': test_r2, 'test_mae': test_mae}, os.path.join(dirs['model'], f"encdec_{plastic_type}.pth"))
    # Also save training log
    with open(os.path.join(dirs['logs'], "training_log.json"), 'w') as f:
        json.dump({'plastic': plastic_type, 'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse,
                   'best_epoch': best_epoch+1, 'time': total_time, 'params': params}, f, indent=2)
    return model, score_mean, score_std, {'plastic': plastic_type, 'test_r2': test_r2, 'test_mae': test_mae, 
            'test_rmse': test_rmse, 'best_epoch': best_epoch+1, 'time': total_time, 'history': history, 'params': params, 'dirs': dirs}

# ============================================================================
# PEPTIDE GENERATION WITH ALL METHODS
# ============================================================================
def generate_peptides_all_methods(model, plastic_type, score_mean, score_std, n_peptides=10):
    dirs = get_plastic_dirs(plastic_type)
    print(f"\n{'='*80}\n🧬 PEPTİD ÜRETİMİ: {plastic_type} ({n_peptides} adet x 3 yöntem)\n{'='*80}")
    all_results = {'SA': [], 'ILS': [], 'MOCO-CEM': []}
    start_peptides = [''.join(random.choices(AMINO_ACIDS, k=12)) for _ in range(n_peptides)]
    
    print(f"\n🔥 SA ile peptid üretiliyor...")
    sa = SAPeptideOptimizer(model, 'encdec', score_mean, score_std, initial_temp=0.5, cooling_rate=0.9, max_iterations=3000)
    for pep in tqdm(start_peptides, desc="SA"): all_results['SA'].extend(sa.optimize(pep, 1))
    
    print(f"\n🔄 ILS ile peptid üretiliyor...")
    ils = ILSPeptideOptimizer(model, 'encdec', score_mean, score_std, max_iterations=800, local_steps=60)
    for pep in tqdm(start_peptides, desc="ILS"): all_results['ILS'].extend(ils.optimize(pep, 1))
    
    print(f"\n📊 MOCO-CEM ile peptid üretiliyor...")
    moco = MocoCEMPeptideOptimizer(model, 'encdec', score_mean, score_std, max_iterations=300, batch_size=128)
    for pep in tqdm(start_peptides, desc="MOCO-CEM"): all_results['MOCO-CEM'].extend(moco.optimize(pep, 1))
    
    # Save to plastic-specific peptides directory
    for method, res in all_results.items():
        df = pd.DataFrame([{'Peptide': r['best_peptide'], 'Score': r['best_score'], 'Initial': r['initial_peptide'],
                           'Init_Score': r['initial_score'], 'Improvement': r['improvement']} for r in res])
        df = df.sort_values('Score'); df.to_csv(os.path.join(dirs['peptides'], f'{method}_peptides.csv'), index=False)
        print(f"   ✓ {method}: En iyi={df['Score'].min():.2f}, Ort={df['Score'].mean():.2f}")
    
    # Save all peptides combined
    all_peps = [{'Method': m, 'Peptide': r['best_peptide'], 'Score': r['best_score'], 'Improvement': r['improvement']} 
                for m in all_results for r in all_results[m]]
    pd.DataFrame(all_peps).to_csv(os.path.join(dirs['peptides'], 'all_peptides.csv'), index=False)
    return all_results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_training_curve(history, plastic_type):
    dirs = get_plastic_dirs(plastic_type)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train', color='blue'); axes[0].plot(history['val_loss'], label='Val', color='red')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title(f'Loss - {plastic_type}'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['train_r2'], label='Train', color='blue'); axes[1].plot(history['val_r2'], label='Val', color='red')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('R²'); axes[1].set_title(f'R² - {plastic_type}'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(dirs['figures'], 'training_curve.png'), dpi=150); plt.close()

def plot_method_comparison(all_results, plastic_type):
    dirs = get_plastic_dirs(plastic_type)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    methods = list(all_results.keys()); colors = ['steelblue', 'forestgreen', 'darkorange']
    best = [min(r['best_score'] for r in all_results[m]) for m in methods]
    avg = [np.mean([r['best_score'] for r in all_results[m]]) for m in methods]
    imp = [np.mean([r['improvement'] for r in all_results[m]]) for m in methods]
    for ax, data, title in zip(axes, [best, avg, imp], ['En İyi Skor (↓)', 'Ortalama Skor (↓)', 'Ort. İyileşme (↑)']):
        bars = ax.bar(methods, data, color=colors, alpha=0.8)
        ax.set_title(title, fontweight='bold')
        for bar, val in zip(bars, data): ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom')
    plt.suptitle(f'Yöntem Karşılaştırması - {plastic_type}', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(dirs['figures'], 'method_comparison.png'), dpi=150); plt.close()

def plot_aa_heatmap(all_results, plastic_type):
    dirs = get_plastic_dirs(plastic_type)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (method, ax) in enumerate(zip(all_results.keys(), axes)):
        peptides = [r['best_peptide'] for r in all_results[method]]
        freq = np.zeros((12, len(AMINO_ACIDS)))
        for pep in peptides:
            for pos, aa in enumerate(pep):
                if aa in AA_TO_IDX: freq[pos, AA_TO_IDX[aa]] += 1
        freq = freq / len(peptides) * 100
        sns.heatmap(freq, ax=ax, cmap='YlOrRd', xticklabels=list(AMINO_ACIDS), yticklabels=range(1, 13), cbar_kws={'label': '%'})
        ax.set_xlabel('Amino Asit'); ax.set_ylabel('Pozisyon'); ax.set_title(method, fontweight='bold')
    plt.suptitle(f'AA Frekans Analizi - {plastic_type}', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(dirs['figures'], 'aa_heatmap.png'), dpi=150); plt.close()

def plot_score_distribution(all_results, plastic_type):
    dirs = get_plastic_dirs(plastic_type)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'SA': 'steelblue', 'ILS': 'forestgreen', 'MOCO-CEM': 'darkorange'}
    for method in all_results: ax.hist([r['best_score'] for r in all_results[method]], bins=15, alpha=0.5, label=method, color=colors[method])
    ax.set_xlabel('Score'); ax.set_ylabel('Frequency'); ax.set_title(f'Skor Dağılımı - {plastic_type}', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(dirs['figures'], 'score_distribution.png'), dpi=150); plt.close()

def plot_trajectory(all_results, plastic_type):
    dirs = get_plastic_dirs(plastic_type)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, method in zip(axes, all_results.keys()):
        for r in all_results[method][:5]:
            traj = r['trajectory']; ax.plot([t[0] for t in traj], [t[1] for t in traj], alpha=0.6)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Score'); ax.set_title(method, fontweight='bold'); ax.grid(True, alpha=0.3)
    plt.suptitle(f'Optimizasyon Trajectory - {plastic_type}', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(dirs['figures'], 'trajectory.png'), dpi=150); plt.close()

# ============================================================================
# REPORT GENERATION
# ============================================================================
def calculate_aa_frequency(peptides):
    """Calculate amino acid frequency for a list of peptides"""
    freq = {aa: 0 for aa in AMINO_ACIDS}
    total = 0
    for pep in peptides:
        for aa in pep:
            if aa in freq:
                freq[aa] += 1
                total += 1
    return {aa: (count / total * 100) if total > 0 else 0 for aa, count in freq.items()}

def calculate_position_freq(peptides):
    """Calculate position-wise AA frequency"""
    pos_freq = {i: {aa: 0 for aa in AMINO_ACIDS} for i in range(12)}
    for pep in peptides:
        for i, aa in enumerate(pep[:12]):
            if aa in pos_freq[i]:
                pos_freq[i][aa] += 1
    return pos_freq

def generate_report(plastic_type, train_res, pep_res):
    dirs = get_plastic_dirs(plastic_type)
    params = train_res.get('params', BEST_MODEL_PARAMS.get(plastic_type, {}))
    
    with open(os.path.join(dirs['base'], f'RAPOR_{plastic_type}.md'), 'w', encoding='utf-8') as f:
        f.write(f"# 🧬 Peptid Üretim Karşılaştırma Raporu - {plastic_type}\n\n")
        f.write(f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Platform:** Windows 11, RTX 4080 Super\n")
        f.write(f"**Model:** LSTM Encoder-Decoder (ENCDEC)\n\n")
        
        # Özet
        f.write("---\n\n## 📋 Özet\n\n")
        best_method = min(pep_res.keys(), key=lambda m: min(r['best_score'] for r in pep_res[m]))
        best_score = min(min(r['best_score'] for r in pep_res[m]) for m in pep_res)
        best_pep = min([r for m in pep_res for r in pep_res[m]], key=lambda x: x['best_score'])
        f.write(f"- **En İyi Yöntem:** {best_method}\n")
        f.write(f"- **En İyi Skor:** {best_score:.2f}\n")
        f.write(f"- **En İyi Peptid:** `{best_pep['best_peptide']}`\n")
        f.write(f"- **Model R²:** {train_res['test_r2']:.4f}\n\n")
        
        # Eğitim Sonuçları
        f.write("---\n\n## 📊 Model Eğitim Sonuçları\n\n")
        f.write("### Performans Metrikleri\n\n")
        f.write("| Metrik | Değer | Açıklama |\n|--------|-------|----------|\n")
        f.write(f"| **Test R²** | **{train_res['test_r2']:.4f}** | Model açıklayıcılığı (1.0 = mükemmel) |\n")
        f.write(f"| Test MAE | {train_res['test_mae']:.4f} | Ortalama mutlak hata |\n")
        f.write(f"| Test RMSE | {train_res['test_rmse']:.4f} | Kök ortalama kare hata |\n")
        f.write(f"| Best Epoch | {train_res['best_epoch']} / 250 | Early stopping epoch |\n")
        f.write(f"| Eğitim Süresi | {train_res['time']:.0f} saniye | ~{train_res['time']/60:.1f} dakika |\n\n")
        
        # Hiperparametreler
        f.write("### Model Hiperparametreleri\n\n")
        f.write("| Parametre | Değer |\n|-----------|-------|\n")
        f.write(f"| Hidden Dim | {params.get('hidden_dim', 256)} |\n")
        f.write(f"| Num Layers | {params.get('num_layers', 2)} |\n")
        f.write(f"| Dropout | {params.get('dropout', 0.1)} |\n")
        f.write(f"| Learning Rate | {params.get('learning_rate', 0.001)} |\n")
        f.write(f"| Batch Size | {params.get('batch_size', 1024)} |\n")
        f.write(f"| Lambda Score | {params.get('lambda_score', 1.0)} |\n")
        f.write(f"| Weight Decay | {params.get('weight_decay', 0.0001)} |\n")
        f.write(f"| Layer Norm | {params.get('use_layernorm', True)} |\n\n")
        
        # Eğitim Grafiği
        f.write("### Eğitim Grafiği\n\n")
        f.write(f"![Training Curve](figures/training_curve.png)\n\n")
        
        # Peptid Üretim Sonuçları
        f.write("---\n\n## 🧬 Peptid Üretim Sonuçları\n\n")
        f.write("### Yöntem Karşılaştırması\n\n")
        f.write("| Yöntem | En İyi Skor | Ort. Skor | Std | İyileşme | Süre/Peptid |\n")
        f.write("|--------|-------------|-----------|-----|----------|-------------|\n")
        
        method_stats = {}
        for m in pep_res:
            scores = [r['best_score'] for r in pep_res[m]]
            imps = [r['improvement'] for r in pep_res[m]]
            method_stats[m] = {
                'best': min(scores), 'avg': np.mean(scores), 'std': np.std(scores),
                'imp': np.mean(imps), 'count': len(scores)
            }
            time_est = {'SA': '~3s', 'ILS': '~48s', 'MOCO-CEM': '~5dk'}
            f.write(f"| **{m}** | {method_stats[m]['best']:.2f} | {method_stats[m]['avg']:.2f} | {method_stats[m]['std']:.2f} | {method_stats[m]['imp']:.2f} | {time_est.get(m, '?')} |\n")
        
        # En iyi yöntem analizi
        f.write(f"\n**🏆 En İyi Yöntem: {best_method}**\n\n")
        
        # Karşılaştırma Grafiği
        f.write("### Karşılaştırma Grafikleri\n\n")
        f.write(f"![Method Comparison](figures/method_comparison.png)\n\n")
        f.write(f"![Score Distribution](figures/score_distribution.png)\n\n")
        f.write(f"![Optimization Trajectory](figures/trajectory.png)\n\n")
        
        # Tüm üretilen peptidler
        f.write("---\n\n## 📝 Üretilen Peptidler\n\n")
        for m in pep_res:
            f.write(f"### {m} Peptidleri\n\n")
            f.write("| # | Peptid | Skor | Başlangıç | Başlangıç Skor | İyileşme |\n")
            f.write("|---|--------|------|-----------|----------------|----------|\n")
            sorted_res = sorted(pep_res[m], key=lambda x: x['best_score'])
            for i, r in enumerate(sorted_res, 1):
                f.write(f"| {i} | `{r['best_peptide']}` | {r['best_score']:.2f} | `{r['initial_peptide']}` | {r['initial_score']:.2f} | {r['improvement']:.2f} |\n")
            f.write("\n")
        
        # Amino Asit Frekans Analizi
        f.write("---\n\n## 🔬 Amino Asit Frekans Analizi\n\n")
        f.write("### Genel AA Dağılımı (Tüm Yöntemler)\n\n")
        all_peptides = [r['best_peptide'] for m in pep_res for r in pep_res[m]]
        aa_freq = calculate_aa_frequency(all_peptides)
        sorted_aa = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)
        f.write("| AA | Frekans (%) | Görsel |\n|----|-----------:|--------|\n")
        for aa, freq in sorted_aa[:10]:
            bar = '█' * int(freq / 2) + '░' * (25 - int(freq / 2))
            f.write(f"| **{aa}** | {freq:.1f}% | {bar} |\n")
        
        # Yönteme göre AA dağılımı
        f.write("\n### Yönteme Göre En Sık Kullanılan AA'ler\n\n")
        f.write("| Yöntem | Top 5 AA |\n|--------|----------|\n")
        for m in pep_res:
            m_peps = [r['best_peptide'] for r in pep_res[m]]
            m_freq = calculate_aa_frequency(m_peps)
            top5 = sorted(m_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            top5_str = ', '.join([f"{aa}({f:.0f}%)" for aa, f in top5])
            f.write(f"| {m} | {top5_str} |\n")
        
        # AA Heatmap
        f.write(f"\n### Pozisyon Bazlı AA Heatmap\n\n")
        f.write(f"![AA Heatmap](figures/aa_heatmap.png)\n\n")
        
        # Peptid Benzerlik Analizi
        f.write("---\n\n## 🔗 Peptid Benzerlik Analizi\n\n")
        f.write("### En İyi Peptidler Arası Hamming Mesafesi\n\n")
        best_peps = {m: min(pep_res[m], key=lambda x: x['best_score'])['best_peptide'] for m in pep_res}
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
        
        # Ortak motifler
        f.write("\n### Ortak Motifler (En İyi 3 Peptid)\n\n")
        top3 = sorted([r for m in pep_res for r in pep_res[m]], key=lambda x: x['best_score'])[:3]
        for i, r in enumerate(top3, 1):
            f.write(f"{i}. `{r['best_peptide']}` ({r['method']}, Skor: {r['best_score']:.2f})\n")
        
        # Dosya yapısı
        f.write("\n---\n\n## 📁 Oluşturulan Dosyalar\n\n")
        f.write("```\n")
        f.write(f"{plastic_type}/\n")
        f.write(f"├── model/\n")
        f.write(f"│   └── encdec_{plastic_type}.pth\n")
        f.write(f"├── figures/\n")
        f.write(f"│   ├── training_curve.png\n")
        f.write(f"│   ├── method_comparison.png\n")
        f.write(f"│   ├── aa_heatmap.png\n")
        f.write(f"│   ├── score_distribution.png\n")
        f.write(f"│   └── trajectory.png\n")
        f.write(f"├── peptides/\n")
        f.write(f"│   ├── SA_peptides.csv\n")
        f.write(f"│   ├── ILS_peptides.csv\n")
        f.write(f"│   ├── MOCO-CEM_peptides.csv\n")
        f.write(f"│   └── all_peptides.csv\n")
        f.write(f"├── logs/\n")
        f.write(f"│   ├── training_log.json\n")
        f.write(f"│   └── peptide_results.json\n")
        f.write(f"└── RAPOR_{plastic_type}.md\n")
        f.write("```\n\n")
        
        # Sonuç
        f.write("---\n\n## 📌 Sonuç ve Öneriler\n\n")
        if train_res['test_r2'] >= 0.95:
            f.write("✅ **Model Performansı:** Mükemmel (R² ≥ 0.95)\n\n")
        elif train_res['test_r2'] >= 0.90:
            f.write("✅ **Model Performansı:** Çok İyi (R² ≥ 0.90)\n\n")
        else:
            f.write("⚠️ **Model Performansı:** İyileştirme gerekebilir\n\n")
        
        # Yöntem karşılaştırma özeti
        best_m = min(method_stats.keys(), key=lambda m: method_stats[m]['best'])
        f.write(f"**En İyi Yöntem:** {best_m} (En düşük skor: {method_stats[best_m]['best']:.2f})\n\n")
        
        f.write("### Yöntem Değerlendirmesi\n\n")
        f.write("- **SA (Simulated Annealing):** Hızlı, basit, iyi baseline\n")
        f.write("- **ILS (Iterated Local Search):** Daha derin arama, daha iyi sonuçlar\n")
        f.write("- **MOCO-CEM (Cross-Entropy):** En kapsamlı arama, en iyi sonuçlar ama yavaş\n\n")
        
        f.write("---\n\n*Rapor otomatik olarak `peptide_generation_comparison.py` tarafından oluşturulmuştur.*\n")
    
    print(f"✓ Rapor: RAPOR_{plastic_type}.md")

# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "checkpoint.json")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'results': {}}

def save_checkpoint(completed_plastics, results):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'completed': completed_plastics, 'results': results}, f, indent=2)

def is_plastic_completed(plastic_type):
    """Check if plastic has all required output files in new folder structure"""
    dirs = get_plastic_dirs(plastic_type)
    required_files = [
        os.path.join(dirs['model'], f"encdec_{plastic_type}.pth"),
        os.path.join(dirs['peptides'], "SA_peptides.csv"),
        os.path.join(dirs['peptides'], "ILS_peptides.csv"),
        os.path.join(dirs['peptides'], "MOCO-CEM_peptides.csv"),
        os.path.join(dirs['logs'], "training_log.json"),
        os.path.join(dirs['base'], f"RAPOR_{plastic_type}.md")
    ]
    return all(os.path.exists(f) for f in required_files)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def run_for_plastic(plastic_type, n_peptides=10):
    dirs = setup_plastic_dirs(plastic_type)
    print(f"\n{'#'*80}\n# {plastic_type} BAŞLIYOR\n{'#'*80}")
    model, score_mean, score_std, train_res = train_model(plastic_type, epochs=250)
    if model is None: return None
    
    plot_training_curve(train_res['history'], plastic_type)
    pep_res = generate_peptides_all_methods(model, plastic_type, score_mean, score_std, n_peptides)
    plot_method_comparison(pep_res, plastic_type)
    plot_aa_heatmap(pep_res, plastic_type)
    plot_score_distribution(pep_res, plastic_type)
    plot_trajectory(pep_res, plastic_type)
    generate_report(plastic_type, train_res, pep_res)
    
    # Save detailed log to plastic-specific directory
    with open(os.path.join(dirs['logs'], 'peptide_results.json'), 'w') as f:
        json.dump({'training': {k: v for k, v in train_res.items() if k not in ['history', 'dirs']},
                   'peptides': {m: [{'peptide': r['best_peptide'], 'score': r['best_score'], 
                                    'initial': r['initial_peptide'], 'improvement': r['improvement']} 
                               for r in pep_res[m]] for m in pep_res}}, f, indent=2)
    return {'training': train_res, 'peptides': pep_res}

def run_all_plastics(n_peptides=10, resume=True):
    plastics = ['PET', 'PP', 'PE', 'PVC', 'PS', 'Nylon']
    print(f"{'='*80}\n🧬 TÜM PLASTİKLER İÇİN PEPTİD ÜRETİM KARŞILAŞTIRMASI\n{'='*80}")
    print(f"Plastikler: {', '.join(plastics)}\nYöntemler: SA, ILS, MOCO-CEM\n")
    
    checkpoint = load_checkpoint() if resume else {'completed': [], 'results': {}}
    all_res = checkpoint.get('results', {})
    
    # Check which plastics are already completed
    completed = []
    for plastic in plastics:
        if is_plastic_completed(plastic):
            completed.append(plastic)
            if plastic not in all_res:
                dirs = get_plastic_dirs(plastic)
                log_file = os.path.join(dirs['logs'], "peptide_results.json")
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        all_res[plastic] = json.load(f)
    
    if completed:
        print(f"✅ Tamamlanmış plastikler: {', '.join(completed)}")
        remaining = [p for p in plastics if p not in completed]
        if remaining:
            print(f"⏳ Devam edilecek plastikler: {', '.join(remaining)}\n")
        else:
            print("🎉 Tüm plastikler zaten tamamlanmış!\n")
    
    for plastic in plastics:
        if plastic in completed:
            print(f"⏭️ {plastic} zaten tamamlanmış, atlanıyor...")
            continue
        
        result = run_for_plastic(plastic, n_peptides)
        if result:
            all_res[plastic] = {'training': {k: v for k, v in result['training'].items() if k != 'history'},
                               'peptides': {m: [{'peptide': r['best_peptide'], 'score': r['best_score'], 'improvement': r['improvement']} 
                                           for r in result['peptides'][m]] for m in result['peptides']}}
            completed.append(plastic)
            save_checkpoint(completed, all_res)
            print(f"💾 Checkpoint kaydedildi: {plastic} tamamlandı\n")
    
    # Genel özet
    summary = []
    for plastic in plastics:
        if plastic in all_res and all_res[plastic]:
            res = all_res[plastic]
            test_r2 = res.get('training', {}).get('test_r2', 0) if isinstance(res.get('training'), dict) else 0
            peps = res.get('peptides', {})
            for m in peps:
                scores = [p['score'] for p in peps[m]] if peps[m] else [0]
                best = min(scores)
                summary.append({'Plastic': plastic, 'Method': m, 'Best_Score': best, 'Test_R2': test_r2})
    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(SUMMARY_DIR, 'genel_ozet.csv'), index=False)
    print(f"\n{'='*80}\n✅ TÜM İŞLEMLER TAMAMLANDI\n{'='*80}")
    return all_res

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Peptide Generation Comparison')
    parser.add_argument('--plastic', type=str, default='ALL', help='Plastik tipi (ALL=hepsi)')
    parser.add_argument('--n_peptides', type=int, default=10, help='Üretilecek peptid sayısı')
    args = parser.parse_args()
    
    print(f"{'='*80}\n🧬 PEPTIDE GENERATION COMPARISON\n{'='*80}")
    if torch.cuda.is_available(): print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}\nPlastik: {args.plastic}\nPeptid sayısı: {args.n_peptides}\n")
    
    if args.plastic == 'ALL': run_all_plastics(args.n_peptides)
    else: run_for_plastic(args.plastic, args.n_peptides)
