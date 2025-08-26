#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: run_set_ranker_F1_external.py
LAST UPDATED: 2025-08-25

INPUT FILES (full paths):
- /Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/T60.xlsx (sheet: T60) → month-t forecasts
- /Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/T2_Optimizer.xlsx (sheet: Monthly_Net_Returns) → month-(t+1) realized returns
- /Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/Extrernal Data.xlsx (sheet: 1MTR) → external market factors

OUTPUT FILES:
- Excel: predictions_f1_external.xlsx (Predictions, Metrics, Metrics_Aggregate, Completeness sheets + baseline sheets)
- PDF:   performance_f1_external.pdf (NDCG@5, Regret@5 time series with baseline overlays)
- Models: models/model_month_XXX.pt (trained model states, metrics, hyperparams for each OOS month, if --save_models)

RULES & HANDLING:
- Alignment: T60 forecasts are already lagged - T60[t] predicts T2[t] (direct alignment)
- Factors: include both _CS and _TS as separate columns
- External factors: 9 market indicators from Extrernal Data.xlsx (1MTR, 3MTR, 12MTR, Bond Yield Change, Advance Decline, Dollar Index, GDP Growth, Inflation, VIX)
- Missing forecasts at t → fill with month-t cross-sectional mean (logged)
- Missing external data → forward-fill + trailing mean imputation
- Missing labels at t+1 → excluded from loss/metrics (logged)
- No winsorization (per user). Plots use matplotlib (no seaborn). Outputs xlsx/pdf. Meters included.

MODEL:
- Per-factor history K=24 months: features per month = [forecast, zscore_t, rank_pct_t, realized_prev]
- External context history K=24 months: 9 external market factors
- Factor temporal encoder: 2-layer GRU (hidden=64)
- External context encoder: 1-layer GRU (hidden=32)
- Combined features: factor_emb (64) + external_context (32) = 96
- Cross-sectional SetTransformer block (self-attention) + MLP scorer
- Loss: LambdaRank targeting NDCG@5 (rank-based relevance)
- Validation: walk-forward (warmup=24), retrain each OOS step; early stop on val NDCG@5
"""

import argparse
import math
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Utilities ----------------

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_saved_model(model_path: str, device: torch.device):
    """
    Load a saved model from disk for backtesting or analysis.
    
    Returns:
        model: Loaded SetRankerWithExternal model
        info: Dictionary with metadata (month, date, metrics, hyperparams, etc.)
    """
    info = torch.load(model_path, map_location=device)
    
    # Reconstruct model from saved hyperparams
    hp = info['hyperparams']
    f_ext = hp.get('f_ext', 0)  # Default to 0 for backward compatibility
    
    model = SetRankerWithExternal(
        n_factors=hp['n_factors'],
        input_size=hp.get('input_size', 4),
        f_ext=f_ext,
        k_history=hp['k_history'],
        gru_hidden=hp['gru_hidden'],
        n_heads=hp['n_heads'],
        d_model=hp['d_model'],
        emb_dim=hp['emb_dim'],
        dropout=0.1
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(info['state_dict'])
    model.eval()
    
    return model, info


# ---------------- Data ----------------

class DataBundle:
    def __init__(self, dates, factor_names, features, labels_next, label_mask, fill_logs, external_features=None):
        self.dates = dates                    # List[pd.Timestamp]
        self.factor_names = factor_names      # List[str]
        self.features = features              # [T, M, F_in] float32
        self.labels_next = labels_next        # [T, M] float32
        self.label_mask = label_mask          # [T, M] bool
        self.fill_logs = fill_logs            # List[dict]
        self.external_features = external_features  # [T, F_ext] float32 or None


def read_sheets(t60_path: str, t2_path: str, ext_path: str = None, s_t60: str = "T60", s_t2: str = "Monthly_Net_Returns", s_ext: str = "1MTR"):
    t60 = pd.read_excel(t60_path, sheet_name=s_t60)
    t2 = pd.read_excel(t2_path, sheet_name=s_t2)
    t60.columns = [str(c).strip() for c in t60.columns]
    t2.columns = [str(c).strip() for c in t2.columns]
    if 'Date' not in t60.columns or 'Date' not in t2.columns:
        raise ValueError("Both sheets must have a 'Date' column")
    t60['Date'] = pd.to_datetime(t60['Date'])
    t2['Date'] = pd.to_datetime(t2['Date'])
    t60 = t60.sort_values('Date').reset_index(drop=True)
    t2 = t2.sort_values('Date').reset_index(drop=True)
    
    ext = None
    if ext_path is not None:
        try:
            ext = pd.read_excel(ext_path, sheet_name=s_ext)
            ext.columns = [str(c).strip() for c in ext.columns]
            
            # Handle case where date column might be named 'Country' instead of 'Date'
            date_col = None
            if 'Date' in ext.columns:
                date_col = 'Date'
            elif 'Country' in ext.columns:
                date_col = 'Country'
                ext = ext.rename(columns={'Country': 'Date'})
                print("Renamed 'Country' column to 'Date' in external data")
            
            if date_col is None:
                print(f"Warning: External data sheet '{s_ext}' missing 'Date' or 'Country' column. External factors disabled.")
                ext = None
            else:
                ext['Date'] = pd.to_datetime(ext['Date'])
                ext = ext.sort_values('Date').reset_index(drop=True)
                print(f"Loaded external data with {len(ext)} rows and {len(ext.columns)-1} factors")
        except Exception as e:
            print(f"Warning: Could not load external data from {ext_path}: {e}. External factors disabled.")
            ext = None
    
    return t60, t2, ext


def build_dataset_with_external(t60: pd.DataFrame, t2: pd.DataFrame, ext: pd.DataFrame = None, include_cols: Optional[List[str]] = None) -> DataBundle:
    fac_t60 = [c for c in t60.columns if c != 'Date']
    fac_t2 = [c for c in t2.columns if c != 'Date']
    if include_cols is None:
        factors = [c for c in fac_t60 if c in fac_t2]
    else:
        factors = [c for c in include_cols if c in fac_t60 and c in fac_t2]
    if len(factors) == 0:
        raise ValueError('No overlapping factor columns between T60 and T2_Optimizer')

    # T60 forecasts are already lagged: T60(t) predicts T2_Optimizer(t)
    # No shift needed - direct alignment
    merged = pd.merge(t60[['Date'] + factors], t2[['Date'] + factors], on='Date', how='inner', suffixes=('_f', '_y'))

    dates = merged['Date'].tolist()
    M = len(factors)
    F_f = merged[[f + '_f' for f in factors]].to_numpy(float)  # [T, M]
    Y = merged[[f + '_y' for f in factors]].to_numpy(float)    # [T, M]
    T = F_f.shape[0]

    fill_logs: List[Dict] = []
    # Fill missing forecasts with month mean and log
    row_means = np.nanmean(F_f, axis=1)
    nan_mask = np.isnan(F_f)
    fill_vals = np.repeat(row_means[:, None], M, axis=1)
    for t_idx in range(T):
        for j in range(M):
            if nan_mask[t_idx, j]:
                fill_logs.append({'Date': dates[t_idx], 'Factor': factors[j], 'Action': 'forecast_fill_mean', 'FillValue': float(fill_vals[t_idx, j])})
    F_f = np.where(nan_mask, fill_vals, F_f)

    # Z-scores and rank percentiles (cross-sectional per month)
    row_means = F_f.mean(axis=1)
    row_stds = F_f.std(axis=1)
    row_stds = np.where(row_stds == 0, 1.0, row_stds)
    z = (F_f - row_means[:, None]) / row_stds[:, None]
    rank_pct = pd.DataFrame(F_f).rank(axis=1, pct=True, method='average').to_numpy(float)

    # Realized previous month (for feature) aligned to merged dates index
    t2_curr_on_merged = pd.merge(pd.DataFrame({'Date': dates}), t2[['Date'] + factors], on='Date', how='left')
    R_curr = t2_curr_on_merged[factors].to_numpy(float)  # shape [T, M]
    realized_prev = np.vstack([np.full((1, M), np.nan), R_curr[:-1, :]])
    realized_prev = np.where(np.isnan(realized_prev), 0.0, realized_prev)

    # Label mask and logs for missing labels
    label_mask = ~np.isnan(Y)
    for t_idx in range(T):
        for j in range(M):
            if not label_mask[t_idx, j]:
                fill_logs.append({'Date': dates[t_idx], 'Factor': factors[j], 'Action': 'label_missing_excluded', 'FillValue': None})

    features = np.stack([F_f, z, rank_pct, realized_prev], axis=2).astype(np.float32)
    
    # Handle external factors
    external_features = None
    if ext is not None:
        # Expected external columns (from plan) - handle column name variations
        ext_cols_map = {
            '1MTR': ['1MTR', '!MTR'],  # Handle potential typo in column name
            '3MTR': ['3MTR', '#MTR'],  # Handle potential typo in column name
            '12MTR': ['12MTR'],
            'Bond Yield Change': ['Bond Yield Change'],
            'Advance Decline': ['Advance Decline'],
            'Dollar Index': ['Dollar Index'],
            'GDP Growth': ['GDP Growth'],
            'Inflation': ['Inflation'],
            'VIX': ['VIX', 'Vix']  # Handle case variations
        }
        
        # Map actual column names to standard names
        available_ext_cols = []
        col_mapping = {}
        for standard_name, possible_names in ext_cols_map.items():
            for possible_name in possible_names:
                if possible_name in ext.columns:
                    available_ext_cols.append(standard_name)
                    col_mapping[possible_name] = standard_name
                    break
        
        # Rename columns to standard names
        if col_mapping:
            ext = ext.rename(columns=col_mapping)
        
        if available_ext_cols:
            print(f"Using external factors: {available_ext_cols}")
            # Merge external data with main timeline
            ext_merged = pd.merge(pd.DataFrame({'Date': dates}), ext[['Date'] + available_ext_cols], on='Date', how='left')
            ext_data = ext_merged[available_ext_cols].to_numpy(float)  # [T, F_ext]
            
            # Handle missing external data with forward-fill + mean imputation
            for j, col in enumerate(available_ext_cols):
                col_data = ext_data[:, j]
                # Forward fill
                mask = ~np.isnan(col_data)
                if mask.any():
                    # Forward fill using pandas method
                    filled_series = pd.Series(col_data).fillna(method='ffill')
                    # Backward fill for any remaining NaNs at the start
                    filled_series = filled_series.fillna(method='bfill')
                    # If still NaN, use overall mean
                    if filled_series.isna().any():
                        overall_mean = filled_series.mean()
                        filled_series = filled_series.fillna(overall_mean)
                        for t_idx in range(T):
                            if np.isnan(col_data[t_idx]):
                                fill_logs.append({'Date': dates[t_idx], 'Factor': f'External_{col}', 'Action': 'external_fill_mean', 'FillValue': float(overall_mean)})
                    ext_data[:, j] = filled_series.values
                else:
                    # All NaN - fill with 0
                    ext_data[:, j] = 0.0
                    for t_idx in range(T):
                        fill_logs.append({'Date': dates[t_idx], 'Factor': f'External_{col}', 'Action': 'external_fill_zero', 'FillValue': 0.0})
            
            external_features = ext_data.astype(np.float32)
        else:
            print("Warning: No expected external factor columns found in external data")
    
    return DataBundle(dates, factors, features, Y.astype(np.float32), label_mask.astype(bool), fill_logs, external_features)


# ---------------- Model ----------------

class TemporalEncoderGRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
    def forward(self, x_seq):  # x_seq: [B*N, K, F]
        out, h = self.gru(x_seq)
        return h[-1]  # [B*N, H]

class SetAttentionBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, tokens):  # [B, N, D]
        B, N, D = tokens.shape
        x = tokens.transpose(0, 1)  # [N, B, D]
        y, _ = self.mha(x, x, x, need_weights=False)
        x = self.ln1(x + y)
        z = x.transpose(0, 1)
        z2 = self.ff(z)
        return self.ln2(z + z2)

class SetRankerWithExternal(nn.Module):
    def __init__(self, n_factors: int, input_size=4, f_ext=9, k_history=24, gru_hidden=64, n_heads=4, d_model=64, emb_dim=32, dropout=0.1, ext_hidden_ratio=0.5, ext_layers=1, factor_layers=2):
        super().__init__()
        self.k_history = k_history
        self.f_ext = f_ext
        
        # Factor temporal encoder (existing)
        self.factor_gru = nn.GRU(input_size, gru_hidden, num_layers=factor_layers, dropout=dropout if factor_layers > 1 else 0.0, batch_first=True)
        
        # NEW: External context encoder
        if f_ext > 0:
            ext_hidden = int(gru_hidden * ext_hidden_ratio)
            self.external_gru = nn.GRU(f_ext, ext_hidden, num_layers=ext_layers, dropout=dropout if ext_layers > 1 else 0.0, batch_first=True)
            self.external_proj = nn.Linear(ext_hidden, ext_hidden)
            # Combined features: factor_emb + external_context
            combined_dim = gru_hidden + ext_hidden
        else:
            self.external_gru = None
            self.external_proj = None
            combined_dim = gru_hidden
            
        self.factor_emb = nn.Embedding(n_factors, emb_dim)
        self.proj = nn.Linear(combined_dim + emb_dim, d_model)
        self.set_block = SetAttentionBlock(d_model, n_heads, dropout)
        self.scorer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        
    def forward(self, x_seq, factor_ids, external_seq=None):  # x_seq: [B, N, K, F], external_seq: [B, K, F_ext]
        B, N, K, F_in = x_seq.shape
        
        # Process factor histories: [B, N, K, F_in] -> [B, N, gru_hidden]
        h_factor = self.factor_gru(x_seq.reshape(B*N, K, F_in))[1][-1].reshape(B, N, -1)  # Take last hidden state
        
        # Process external history if available: [B, K, F_ext] -> [B, ext_hidden]
        if external_seq is not None and self.external_gru is not None:
            h_external = self.external_gru(external_seq)[1][-1]  # [B, ext_hidden]
            h_external = self.external_proj(h_external)  # [B, ext_hidden]
            # Broadcast external context to all factors: [B, ext_hidden] -> [B, N, ext_hidden]
            h_external = h_external.unsqueeze(1).expand(B, N, -1)
            # Concatenate factor and external features
            h_combined = torch.cat([h_factor, h_external], dim=-1)  # [B, N, gru_hidden + ext_hidden]
        else:
            h_combined = h_factor  # [B, N, gru_hidden]
            
        # Add factor embeddings
        emb = self.factor_emb(factor_ids).unsqueeze(0).expand(B, N, -1)
        tok = self.proj(torch.cat([h_combined, emb], dim=-1))
        tok = self.set_block(tok)
        return self.scorer(tok).squeeze(-1)  # [B, N]

# Keep original class for compatibility
class TemporalSetRanker(SetRankerWithExternal):
    def __init__(self, n_factors: int, input_size=3, k_history=24, gru_hidden=64, n_heads=4, d_model=64, emb_dim=32, dropout=0.1):
        super().__init__(n_factors, input_size, f_ext=0, k_history=k_history, gru_hidden=gru_hidden, n_heads=n_heads, d_model=d_model, emb_dim=emb_dim, dropout=dropout)


# ---------------- Loss & Metrics ----------------

def ndcg_at_k(rels: np.ndarray, k: int = 5) -> float:
    rels = np.asarray(rels)
    k = min(k, len(rels))
    if k == 0:
        return 0.0
    gains = (2.0**rels[:k] - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, k+2))
    dcg = float(np.sum(gains * discounts))
    rels_sorted = np.sort(rels)[::-1]
    idcg = float(np.sum((2.0**rels_sorted[:k] - 1.0) * discounts))
    return 0.0 if idcg <= 0 else dcg / idcg


def lambda_rank_loss(scores: torch.Tensor, y: torch.Tensor, valid_mask: torch.Tensor, topk: int = 5) -> torch.Tensor:
    """
    Vectorized LambdaRank loss targeting NDCG@k.
    Replaces O(N²) Python loops with efficient PyTorch operations.
    """
    B, N = scores.shape
    device = scores.device
    total_loss = scores.new_zeros(())
    eps = 1e-8
    
    for b in range(B):
        mask = valid_mask[b]
        n_valid = mask.sum().item()
        if n_valid < 2:
            continue
            
        # Extract valid scores and labels
        s = scores[b][mask]  # [n_valid]
        y_b = y[b][mask]     # [n_valid]
        n = s.shape[0]
        
        # Create pairwise difference matrices
        # s_diff[i,j] = s[i] - s[j]
        s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [n, n]
        
        # y_diff[i,j] = y[i] - y[j] (preference matrix)
        y_diff = y_b.unsqueeze(1) - y_b.unsqueeze(0)  # [n, n]
        
        # Only consider pairs where labels differ significantly
        preference_mask = torch.abs(y_diff) > eps  # [n, n]
        
        # Compute current ranking for NDCG calculation
        _, ord_pred = torch.sort(s, descending=True)
        y_sorted = y_b[ord_pred]
        
        # Compute NDCG weights (Delta NDCG approximation)
        # For efficiency, use a simplified Delta NDCG based on position changes
        ranks = torch.empty_like(ord_pred, dtype=torch.float, device=device)
        ranks[ord_pred] = torch.arange(n, dtype=torch.float, device=device)
        
        # Delta NDCG approximation: weight by rank difference and label difference
        rank_diff = ranks.unsqueeze(1) - ranks.unsqueeze(0)  # [n, n]
        
        # Discount factors for NDCG (1/log2(rank+2))
        discounts = 1.0 / torch.log2(ranks + 2.0)  # [n]
        discount_diff = discounts.unsqueeze(1) - discounts.unsqueeze(0)  # [n, n]
        
        # Delta NDCG weights: combine label difference, rank difference, and discounts
        # Focus on top-k positions
        topk_mask = (ranks.unsqueeze(1) < topk) | (ranks.unsqueeze(0) < topk)
        
        delta_ndcg = torch.abs(y_diff) * torch.abs(discount_diff) * topk_mask.float()
        delta_ndcg = torch.clamp(delta_ndcg, min=eps)
        
        # Apply preference and topk masks
        valid_pairs = preference_mask & topk_mask
        
        # LambdaRank loss: weighted logistic loss
        # For pairs where y[i] > y[j], we want s[i] > s[j]
        preference_sign = torch.sign(y_diff)  # +1 if y[i] > y[j], -1 if y[i] < y[j]
        
        # Logistic loss: log(1 + exp(-preference_sign * s_diff))
        logistic_loss = torch.log(1 + torch.exp(-preference_sign * s_diff))
        
        # Weight by Delta NDCG and apply masks
        weighted_loss = delta_ndcg * logistic_loss * valid_pairs.float()
        
        # Sum over all valid pairs
        batch_loss = weighted_loss.sum()
        
        # Normalize by number of valid pairs to prevent batch size effects
        n_pairs = valid_pairs.sum().item()
        if n_pairs > 0:
            batch_loss = batch_loss / n_pairs
            
        total_loss += batch_loss
    
    return total_loss


def monthly_metrics(scores: np.ndarray, y_next: np.ndarray, valid_mask: np.ndarray, k: int = 5, name_ranks: Optional[np.ndarray] = None) -> Dict[str, float]:
    idx = np.where(valid_mask)[0]
    if idx.size < 2:
        return {'P@5': np.nan, 'NDCG@5': np.nan, 'AvgTop5': np.nan, 'Regret@5': np.nan, 'IC': np.nan}
    s = scores[idx]; y = y_next[idx]
    try:
        ic, _ = spearmanr(s, y)
    except Exception:
        ic = np.nan
    if name_ranks is not None:
        ord_pred = np.lexsort((np.asarray(name_ranks)[idx], -np.asarray(s)))
    else:
        ord_pred = np.argsort(-s)
    topk_idx = ord_pred[:k]
    if name_ranks is not None:
        ord_true = np.lexsort((np.asarray(name_ranks)[idx], -np.asarray(y)))
    else:
        ord_true = np.argsort(-y)
    true_topk = ord_true[:k]
    prec = np.intersect1d(topk_idx, true_topk).size / min(k, idx.size)
    rel = np.empty_like(y, dtype=float); rel[ord_true] = np.arange(1, len(y)+1)
    ndcg = ndcg_at_k(rel[ord_pred], k=k)
    avg_topk = float(np.nanmean(y[topk_idx])) if topk_idx.size else np.nan
    oracle_avg = float(np.nanmean(y[true_topk])) if true_topk.size else np.nan
    regret = oracle_avg - avg_topk if (not np.isnan(oracle_avg) and not np.isnan(avg_topk)) else np.nan
    return {'P@5': float(prec), 'NDCG@5': float(ndcg), 'AvgTop5': avg_topk, 'Regret@5': float(regret), 'IC': float(ic) if ic is not None else np.nan}


# ---------------- Baselines: Isotonic Calibration ----------------

def _pav_isotonic(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Pool-Adjacent-Violators algorithm. Returns non-decreasing fit to y.
    y, w are 1D arrays of the same length. w defaults to 1s.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
    # blocks: list of [sum_w, sum_y, left_idx, right_idx]
    blocks = []
    for i in range(n):
        sum_w = w[i]
        sum_y = w[i] * y[i]
        blocks.append([sum_w, sum_y, i, i])
        # merge while previous block mean > current block mean
        while len(blocks) >= 2:
            w1, y1, l1, r1 = blocks[-2]
            w2, y2, l2, r2 = blocks[-1]
            if (y1 / w1) <= (y2 / w2):
                break
            # merge
            blocks.pop(); blocks.pop()
            blocks.append([w1 + w2, y1 + y2, l1, r2])
    y_fit = np.empty(n, dtype=float)
    for wsum, ysum, l, r in blocks:
        y_fit[l:r+1] = ysum / wsum
    return y_fit


def fit_isotonic_both(x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """Fit both increasing and decreasing isotonic on (x, y); choose lower MSE.
    Returns dict with keys: 'x', 'y' (piecewise-constant values sorted by x).
    Returns None if insufficient data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (~np.isnan(x)) & (~np.isnan(y))
    x = x[m]; y = y[m]
    if x.size < 2:
        return None
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    y_inc = _pav_isotonic(ys)
    y_dec = -_pav_isotonic(-ys)
    mse_inc = float(np.mean((ys - y_inc) ** 2))
    mse_dec = float(np.mean((ys - y_dec) ** 2))
    if mse_inc <= mse_dec:
        return {'x': xs, 'y': y_inc}
    else:
        return {'x': xs, 'y': y_dec}


def predict_isotonic(model: Dict[str, np.ndarray], x0: np.ndarray) -> np.ndarray:
    """Piecewise-constant prediction using fitted isotonic model."""
    xs = model['x']
    ys = model['y']
    x0 = np.asarray(x0, dtype=float)
    idx = np.searchsorted(xs, x0, side='right') - 1
    idx = np.clip(idx, 0, xs.size - 1)
    return ys[idx]


# ---------------- Train / Validate ----------------

def batch_from_months_with_external(features: np.ndarray, labels_next: np.ndarray, label_mask: np.ndarray, external_features: Optional[np.ndarray], months: List[int], k_history: int, device: torch.device):
    if not months:
        return None, None, None, None
    X_list=[]; Y_list=[]; M_list=[]; E_list=[]
    for t in months:
        start = t - k_history + 1
        if start < 0:
            continue
        X_list.append(features[start:t+1])  # [K, N, F]
        Y_list.append(labels_next[t])
        M_list.append(label_mask[t])
        if external_features is not None:
            E_list.append(external_features[start:t+1])  # [K, F_ext]
        else:
            E_list.append(None)
    if not X_list:
        return None, None, None, None
    X = np.stack(X_list, axis=0)             # [B, K, N, F]
    X = np.transpose(X, (0, 2, 1, 3))        # [B, N, K, F]
    Y = np.stack(Y_list, axis=0)
    M = np.stack(M_list, axis=0).astype(bool)
    
    E = None
    if external_features is not None and E_list[0] is not None:
        E = np.stack(E_list, axis=0)         # [B, K, F_ext]
        E = torch.from_numpy(E).float().to(device)
    
    return torch.from_numpy(X).float().to(device), torch.from_numpy(Y).float().to(device), torch.from_numpy(M).to(device), E

# Keep original function for compatibility
def batch_from_months(features: np.ndarray, labels_next: np.ndarray, label_mask: np.ndarray, months: List[int], k_history: int, device: torch.device):
    X, Y, M, _ = batch_from_months_with_external(features, labels_next, label_mask, None, months, k_history, device)
    return X, Y, M


def train_oos_with_external(model: nn.Module, device: torch.device, features, labels_next, label_mask, external_features: Optional[np.ndarray],
              factor_ids: torch.Tensor, train_months: List[int], val_months: List[int], test_month: int,
              k_history: int, lr=1e-3, weight_decay=1e-3, epochs=50, patience=5, topk_loss=5, verbose=True,
              name_ranks: Optional[np.ndarray] = None, val_every: int = 1):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_ndcg = -np.inf; no_improve=0; best_state=None

    Xtr, Ytr, Mtr, Etr = batch_from_months_with_external(features, labels_next, label_mask, external_features, train_months, k_history, device)
    Xval, Yval, Mval, Eval = batch_from_months_with_external(features, labels_next, label_mask, external_features, val_months, k_history, device)
    Xte, Yte, Mte, Ete = batch_from_months_with_external(features, labels_next, label_mask, external_features, [test_month], k_history, device)

    if Xtr is None or Xte is None:
        return np.zeros(labels_next.shape[1], dtype=float), {'P@5': np.nan, 'NDCG@5': np.nan, 'AvgTop5': np.nan, 'Regret@5': np.nan, 'IC': np.nan}

    pbar = tqdm(range(epochs), desc=f"Train OOS t={test_month}", disable=not verbose)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        scores = model(Xtr, factor_ids, Etr)
        loss = lambda_rank_loss(scores, Ytr, Mtr, topk_loss)
        loss.backward()
        opt.step()

        # Validation (only every val_every epochs)
        if epoch % val_every == 0 or epoch == epochs - 1:
            if Xval is not None:
                model.eval()
                with torch.no_grad():
                    val_scores = model(Xval, factor_ids, Eval)
                    val_met = monthly_metrics(val_scores[0].cpu().numpy(), Yval[0].cpu().numpy(), Mval[0].cpu().numpy(), k=5, name_ranks=name_ranks)
                    val_ndcg = val_met['NDCG@5']

                if val_ndcg > best_ndcg:
                    best_ndcg = val_ndcg
                    best_state = model.state_dict().copy()
                    no_improve = 0
                else:
                    no_improve += val_every  # Increment by val_every since we skip epochs

                if verbose and epoch % 10 == 0:
                    pbar.set_postfix(loss=f"{loss.item():.3f}", val_ndcg=f"{val_ndcg:.3f}")
            else:
                # No validation data, just save current state
                best_state = model.state_dict().copy()
                if verbose and epoch % 10 == 0:
                    pbar.set_postfix(loss=f"{loss.item():.3f}")
        else:
            if verbose and epoch % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")
        
        pbar.update(1)

        if no_improve >= patience * val_every:  # Adjust patience threshold
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        s_test = model(Xte, factor_ids, Ete)[0].cpu().numpy()
    met = monthly_metrics(s_test, Yte.cpu().numpy()[0], Mte.cpu().numpy()[0], k=topk_loss, name_ranks=name_ranks)
    return s_test, met


# ---------------- Outputs ----------------

def write_outputs_xlsx(out_path: str, dates: List[pd.Timestamp], factor_names: List[str],
                       all_scores: List[Optional[np.ndarray]], all_metrics: List[Optional[Dict[str, float]]],
                       labels_next: np.ndarray, label_mask: np.ndarray, fill_logs: List[Dict], topk: int = 5,
                       baseline_raw: Optional[Tuple[List[Optional[np.ndarray]], List[Optional[Dict[str, float]]]]] = None,
                       baseline_iso: Optional[Tuple[List[Optional[np.ndarray]], List[Optional[Dict[str, float]]]]] = None):
    pred_rows=[]
    for t_idx, scores in enumerate(all_scores):
        if scores is None:
            continue
        s = scores; valid = label_mask[t_idx]
        # Build stable name ranks once to tie-break by factor name (ascending)
        order_names = np.argsort(np.asarray(factor_names, dtype='U'))
        name_ranks = np.empty(len(factor_names), dtype=int); name_ranks[order_names] = np.arange(len(factor_names))
        ord_pred = np.lexsort((name_ranks, -np.asarray(s)))
        for r, j in enumerate(ord_pred[:topk], start=1):
            pred_rows.append({'Date': dates[t_idx], 'Rank': r, 'Factor': factor_names[j], 'Score': float(s[j]), 'RealizedReturn(t)': float(labels_next[t_idx, j]) if valid[j] else np.nan})
    preds_df = pd.DataFrame(pred_rows)

    met_rows=[]
    for t_idx, m in enumerate(all_metrics):
        if m is None: continue
        row={'Date': dates[t_idx]}; row.update(m); met_rows.append(row)
    mets_df = pd.DataFrame(met_rows)
    agg={}
    for k in ['P@5','NDCG@5','AvgTop5','Regret@5','IC']:
        vals=[r[k] for r in met_rows if not (isinstance(r[k], float) and math.isnan(r[k]))]
        agg['Mean_'+k]=float(np.mean(vals)) if vals else np.nan
    agg_df=pd.DataFrame([agg])

    # Create wide-format scores and ranks
    all_factor_data = []
    for t_idx, scores in enumerate(all_scores):
        if scores is None:
            continue
        s = scores
        order_names = np.argsort(np.asarray(factor_names, dtype='U'))
        name_ranks = np.empty(len(factor_names), dtype=int); name_ranks[order_names] = np.arange(len(factor_names))
        ranks = pd.Series(s).rank(method='first', ascending=False).astype(int)
        for j, factor in enumerate(factor_names):
            all_factor_data.append({'Date': dates[t_idx], 'Factor': factor, 'Score': s[j], 'Rank': ranks[j]})
    
    all_factor_df = pd.DataFrame(all_factor_data)
    scores_wide_df = all_factor_df.pivot(index='Date', columns='Factor', values='Score')
    ranks_wide_df = all_factor_df.pivot(index='Date', columns='Factor', values='Rank')

    comp_df = pd.DataFrame(fill_logs)
    # Helper to build preds/mets/agg for a scores+metrics pair
    def _build_frames(all_scores_x, all_metrics_x, sheet_prefix: str):
        pred_rows_x = []
        for t_idx, scores_x in enumerate(all_scores_x):
            if scores_x is None:
                continue
            s = scores_x; valid = label_mask[t_idx]
            order_names = np.argsort(np.asarray(factor_names, dtype='U'))
            name_ranks = np.empty(len(factor_names), dtype=int); name_ranks[order_names] = np.arange(len(factor_names))
            ord_pred = np.lexsort((name_ranks, -np.asarray(s)))
            for r, j in enumerate(ord_pred[:topk], start=1):
                pred_rows_x.append({'Date': dates[t_idx], 'Rank': r, 'Factor': factor_names[j], 'Score': float(s[j]), 'RealizedReturn(t)': float(labels_next[t_idx, j]) if valid[j] else np.nan})
        preds_x_df = pd.DataFrame(pred_rows_x)
        met_rows_x = []
        for t_idx, m in enumerate(all_metrics_x):
            if m is None: continue
            row={'Date': dates[t_idx]}; row.update(m); met_rows_x.append(row)
        mets_x_df = pd.DataFrame(met_rows_x)
        agg_x = {}
        for k in ['P@5','NDCG@5','AvgTop5','Regret@5','IC']:
            vals=[r[k] for r in met_rows_x if not (isinstance(r[k], float) and math.isnan(r[k]))]
            agg_x['Mean_'+k]=float(np.mean(vals)) if vals else np.nan
        agg_x_df = pd.DataFrame([agg_x])
        return preds_x_df, mets_x_df, agg_x_df

    try:
        with pd.ExcelWriter(out_path, engine='xlsxwriter') as w:
            preds_df.to_excel(w, index=False, sheet_name='Predictions')
            mets_df.to_excel(w, index=False, sheet_name='Metrics')
            agg_df.to_excel(w, index=False, sheet_name='Metrics_Aggregate')
            scores_wide_df.to_excel(w, sheet_name='Scores_Wide')
            ranks_wide_df.to_excel(w, sheet_name='Ranks_Wide')
            comp_df.to_excel(w, index=False, sheet_name='Completeness')
            if baseline_raw is not None:
                preds_r, mets_r, agg_r = _build_frames(baseline_raw[0], baseline_raw[1], 'RawRank')
                preds_r.to_excel(w, index=False, sheet_name='Predictions_RawRank')
                mets_r.to_excel(w, index=False, sheet_name='Metrics_RawRank')
                agg_r.to_excel(w, index=False, sheet_name='Metrics_Aggregate_RawRank')
            if baseline_iso is not None:
                preds_i, mets_i, agg_i = _build_frames(baseline_iso[0], baseline_iso[1], 'Iso')
                preds_i.to_excel(w, index=False, sheet_name='Predictions_Iso')
                mets_i.to_excel(w, index=False, sheet_name='Metrics_Iso')
                agg_i.to_excel(w, index=False, sheet_name='Metrics_Aggregate_Iso')
    except Exception:
        with pd.ExcelWriter(out_path) as w:
            preds_df.to_excel(w, index=False, sheet_name='Predictions')
            mets_df.to_excel(w, index=False, sheet_name='Metrics')
            agg_df.to_excel(w, index=False, sheet_name='Metrics_Aggregate')
            comp_df.to_excel(w, index=False, sheet_name='Completeness')
            if baseline_raw is not None:
                preds_r, mets_r, agg_r = _build_frames(baseline_raw[0], baseline_raw[1], 'RawRank')
                preds_r.to_excel(w, index=False, sheet_name='Predictions_RawRank')
                mets_r.to_excel(w, index=False, sheet_name='Metrics_RawRank')
                agg_r.to_excel(w, index=False, sheet_name='Metrics_Aggregate_RawRank')
            if baseline_iso is not None:
                preds_i, mets_i, agg_i = _build_frames(baseline_iso[0], baseline_iso[1], 'Iso')
                preds_i.to_excel(w, index=False, sheet_name='Predictions_Iso')
                mets_i.to_excel(w, index=False, sheet_name='Metrics_Iso')
                agg_i.to_excel(w, index=False, sheet_name='Metrics_Aggregate_Iso')


def write_plots_pdf(out_path: str, dates: List[pd.Timestamp], all_metrics: List[Optional[Dict[str, float]]],
                    raw_metrics: Optional[List[Optional[Dict[str, float]]]] = None,
                    iso_metrics: Optional[List[Optional[Dict[str, float]]]] = None):
    dts = pd.to_datetime(dates)
    ndcg=[(m['NDCG@5'] if m is not None else np.nan) for m in all_metrics]
    regret=[(m['Regret@5'] if m is not None else np.nan) for m in all_metrics]
    if raw_metrics is not None:
        ndcg_raw=[(m['NDCG@5'] if m is not None else np.nan) for m in raw_metrics]
        regret_raw=[(m['Regret@5'] if m is not None else np.nan) for m in raw_metrics]
    else:
        ndcg_raw=None; regret_raw=None
    if iso_metrics is not None:
        ndcg_iso=[(m['NDCG@5'] if m is not None else np.nan) for m in iso_metrics]
        regret_iso=[(m['Regret@5'] if m is not None else np.nan) for m in iso_metrics]
    else:
        ndcg_iso=None; regret_iso=None
    with PdfPages(out_path) as pdf:
        plt.figure(figsize=(10,4))
        plt.plot(dts, ndcg, marker='o', label='Model')
        if ndcg_raw is not None: plt.plot(dts, ndcg_raw, marker='o', label='RawRank')
        if ndcg_iso is not None: plt.plot(dts, ndcg_iso, marker='o', label='Iso')
        plt.title('NDCG@5 over time'); plt.xlabel('Date'); plt.ylabel('NDCG@5'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); pdf.savefig(); plt.close()
        plt.figure(figsize=(10,4))
        plt.plot(dts, regret, marker='o', label='Model')
        if regret_raw is not None: plt.plot(dts, regret_raw, marker='o', label='RawRank')
        if regret_iso is not None: plt.plot(dts, regret_iso, marker='o', label='Iso')
        plt.title('Regret@5 over time'); plt.xlabel('Date'); plt.ylabel('Regret@5'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); pdf.savefig(); plt.close()


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description='SetRanker with External Factors (LambdaRank@5) for monthly factor selection')
    ap.add_argument('--t60', type=str, default='/Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/T60.xlsx')
    ap.add_argument('--t2', type=str, default='/Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/T2_Optimizer.xlsx')
    ap.add_argument('--ext', type=str, default='/Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/Extrernal Data.xlsx', help='External factors data file')
    ap.add_argument('--sheet_t60', type=str, default='T60')
    ap.add_argument('--sheet_t2', type=str, default='Monthly_Net_Returns')
    ap.add_argument('--sheet_ext', type=str, default='1MTR', help='External factors sheet name')
    ap.add_argument('--output_dir', type=str, default='output_F1_external_config_3', help='Directory to save all outputs (predictions, plots, models).')
    ap.add_argument('--out_xlsx', type=str, default='predictions_f1_external_config_3.xlsx', help='Output Excel file name (relative to output_dir).')
    ap.add_argument('--out_pdf', type=str, default='performance_f1_external_config_3.pdf', help='Output PDF file name (relative to output_dir).')
    ap.add_argument('--k_history', type=int, default=24)
    ap.add_argument('--warmup', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--emb_dim', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max_oos', type=int, default=0)
    ap.add_argument('--val_every', type=int, default=1, help='Validate every N epochs (default: 1)')
    ap.add_argument('--save_models', action='store_true', help='Save trained models for each OOS month')
    ap.add_argument('--models_dir', type=str, default='models', help='Directory to save models (relative to output_dir).')
    ap.add_argument('--warm_start', action='store_true', help='Use previous month\'s weights as initialization (S2 optimization)')
    ap.add_argument('--warm_epochs', type=int, default=0, help='Reduced epochs for warm-started months (0=use --epochs)')
    args = ap.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_models:
        os.makedirs(os.path.join(args.output_dir, args.models_dir), exist_ok=True)

    set_seeds(args.seed)
    device = get_device()
    print(f"Device: {device}")

    print('[1/5] Reading Excel...')
    t60, t2, ext = read_sheets(args.t60, args.t2, args.ext, args.sheet_t60, args.sheet_t2, args.sheet_ext)
    factors = [c for c in t60.columns if c != 'Date' and c in t2.columns]
    print(f"Found {len(factors)} common factors")

    print('[2/5] Building dataset with external factors...')
    data = build_dataset_with_external(t60, t2, ext)

    # Walk-forward setup
    T, M, F_in = data.features.shape
    k_hist = args.k_history
    factor_ids = torch.arange(M, dtype=torch.long, device=device)
    name_ranks = np.argsort(np.asarray(data.factor_names, dtype='U'))
    name_ranks_full = np.empty(M, dtype=int)
    name_ranks_full[name_ranks] = np.arange(M)

    # Results storage
    all_scores = [None] * T
    all_metrics = [None] * T
    all_scores_raw = [None] * T
    all_metrics_raw = [None] * T
    all_scores_iso = [None] * T
    all_metrics_iso = [None] * T

    test_months = list(range(args.warmup + k_hist, T))
    if args.max_oos > 0:
        test_months = test_months[:args.max_oos]

    # S2: Warm-start tracking
    prev_model_state = None
    warm_start_count = 0

    print('[3/5] Walk-forward training & inference...')
    if args.warm_start:
        print(f"Warm-start enabled: first month from scratch, subsequent months from previous weights")
        if args.warm_epochs > 0:
            print(f"Warm-start epochs: {args.warm_epochs} (vs {args.epochs} for first month)")
    
    pbar_oos = tqdm(test_months, desc="OOS months")
    for test_t in pbar_oos:
        # Train/Val/Test split: ensure at least one training month with full history
        # Eligible months are those with enough history (t >= k_hist - 1)
        eligible = list(range(k_hist - 1, test_t))
        L = len(eligible)
        if L <= 1:
            train_months = eligible
            val_months = []
        else:
            val_span = 6
            # Keep at least one training month
            val_span_eff = min(val_span, max(0, L - 1))
            train_months = eligible[:-val_span_eff] if val_span_eff > 0 else eligible
            val_months = eligible[-val_span_eff:] if val_span_eff > 0 else []

        # S2: Create model with external factors and optionally warm-start from previous month
        f_ext = data.external_features.shape[1] if data.external_features is not None else 0
        model = SetRankerWithExternal(n_factors=M, input_size=4, f_ext=f_ext, k_history=k_hist, gru_hidden=args.hidden, n_heads=args.heads, d_model=args.hidden, emb_dim=args.emb_dim, dropout=0.1).to(device)
        print(f"Model created with {f_ext} external factors")
        
        # Warm-start: load previous month's weights if available
        if args.warm_start and prev_model_state is not None:
            model.load_state_dict(prev_model_state)
            warm_start_count += 1
            # Use reduced epochs for warm-started months
            epochs_to_use = args.warm_epochs if args.warm_epochs > 0 else args.epochs
        else:
            # First month or no warm-start: use full epochs
            epochs_to_use = args.epochs
            
        scores, metrics = train_oos_with_external(model, device, data.features, data.labels_next, data.label_mask, data.external_features, factor_ids, train_months, val_months, test_t, k_hist, lr=args.lr, weight_decay=args.weight_decay, epochs=epochs_to_use, patience=args.patience, topk_loss=5, verbose=args.verbose, name_ranks=name_ranks, val_every=args.val_every)
        
        # S2: Save current model state for next month's warm-start
        if args.warm_start:
            prev_model_state = model.state_dict().copy()
        all_scores[test_t] = scores
        all_metrics[test_t] = metrics

        # Save trained model if requested
        if args.save_models:
            model_path = os.path.join(args.output_dir, args.models_dir, f"model_month_{test_t:03d}.pt")
            model_info = {
                'state_dict': model.state_dict(),
                'month': test_t,
                'date': data.dates[test_t].strftime('%Y-%m-%d'),
                'metrics': metrics,
                'train_months': train_months,
                'val_months': val_months,
                'hyperparams': {
                    'n_factors': M,
                    'input_size': 4,
                    'f_ext': f_ext,
                    'k_history': k_hist,
                    'gru_hidden': args.hidden,
                    'n_heads': args.heads,
                    'd_model': args.hidden,
                    'emb_dim': args.emb_dim,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay
                }
            }
            torch.save(model_info, model_path)

        # Baseline 1: Raw forecast rank
        s_raw = data.features[test_t, :, 0].astype(float)
        all_scores_raw[test_t] = s_raw
        all_metrics_raw[test_t] = monthly_metrics(s_raw, data.labels_next[test_t], data.label_mask[test_t], k=5, name_ranks=name_ranks)

        # Baseline 2: Isotonic calibration per factor (train-only)
        s_iso = np.empty(M, dtype=float)
        for j in range(M):
            xs = []
            ys = []
            for tt in train_months:
                if data.label_mask[tt, j]:
                    xs.append(float(data.features[tt, j, 0]))
                    ys.append(float(data.labels_next[tt, j]))
            if len(xs) < 2:
                s_iso[j] = s_raw[j]
            else:
                model_iso = fit_isotonic_both(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
                if model_iso is None:
                    s_iso[j] = s_raw[j]
                else:
                    s_iso[j] = float(predict_isotonic(model_iso, np.asarray([data.features[test_t, j, 0]], dtype=float))[0])
        all_scores_iso[test_t] = s_iso
        all_metrics_iso[test_t] = monthly_metrics(s_iso, data.labels_next[test_t], data.label_mask[test_t], k=5, name_ranks=name_ranks)

    print('[4/5] Writing XLSX...')
    write_outputs_xlsx(
        os.path.join(args.output_dir, args.out_xlsx), data.dates, data.factor_names,
        all_scores, all_metrics, data.labels_next, data.label_mask, data.fill_logs, topk=5,
        baseline_raw=(all_scores_raw, all_metrics_raw),
        baseline_iso=(all_scores_iso, all_metrics_iso)
    )

    print('[5/5] Writing PDF plots...')
    write_plots_pdf(os.path.join(args.output_dir, args.out_pdf), data.dates, all_metrics, all_metrics_raw, all_metrics_iso)

    # S2: Summary
    if args.warm_start:
        print(f'Warm-start summary: {warm_start_count}/{len(test_months)} months used previous weights')

    print('Done.')


if __name__ == '__main__':
    main()
