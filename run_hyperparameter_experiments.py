#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: run_hyperparameter_experiments.py
LAST UPDATED: 2025-08-25

Hyperparameter optimization experiments for external factor integration.
Tests different GRU architectures, learning rates, and optimization parameters.

INPUT FILES:
- T60.xlsx, T2_Optimizer.xlsx, Extrernal Data.xlsx (same as base model)

OUTPUT FILES:
- hyperparameter_results.xlsx (comparison of all configurations)
- config_X/ folders with individual experiment results
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Hyperparameter configurations to test
CONFIGS = {
    'baseline': {
        'name': 'Baseline (Current)',
        'lr': 1e-3,
        'weight_decay': 1e-3,
        'hidden': 64,
        'ext_hidden_ratio': 0.5,  # external_gru = hidden//2 = 32
        'ext_layers': 1,
        'factor_layers': 2,
        'dropout': 0.1,
        'heads': 4,
        'epochs': 60,
        'patience': 6
    },
    
    'config_1': {
        'name': 'Larger External GRU',
        'lr': 1e-3,
        'weight_decay': 1e-3,
        'hidden': 64,
        'ext_hidden_ratio': 1.0,  # external_gru = hidden = 64
        'ext_layers': 1,
        'factor_layers': 2,
        'dropout': 0.1,
        'heads': 4,
        'epochs': 60,
        'patience': 6
    },
    
    'config_2': {
        'name': 'Deeper External GRU',
        'lr': 1e-3,
        'weight_decay': 1e-3,
        'hidden': 64,
        'ext_hidden_ratio': 0.5,
        'ext_layers': 2,  # 2-layer external GRU
        'factor_layers': 2,
        'dropout': 0.15,  # Higher dropout for deeper model
        'heads': 4,
        'epochs': 60,
        'patience': 6
    },
    
    'config_3': {
        'name': 'Higher Learning Rate',
        'lr': 2e-3,  # 2x learning rate
        'weight_decay': 1e-3,
        'hidden': 64,
        'ext_hidden_ratio': 0.5,
        'ext_layers': 1,
        'factor_layers': 2,
        'dropout': 0.1,
        'heads': 4,
        'epochs': 60,
        'patience': 6
    },
    
    'config_4': {
        'name': 'Lower Learning Rate + Less Regularization',
        'lr': 5e-4,  # 0.5x learning rate
        'weight_decay': 5e-4,  # 0.5x weight decay
        'hidden': 64,
        'ext_hidden_ratio': 0.5,
        'ext_layers': 1,
        'factor_layers': 2,
        'dropout': 0.05,  # Lower dropout
        'heads': 4,
        'epochs': 80,  # More epochs for slower learning
        'patience': 8
    },
    
    'config_5': {
        'name': 'Larger Model + Adaptive LR',
        'lr': 1e-3,
        'weight_decay': 2e-3,  # Higher regularization
        'hidden': 96,  # Larger hidden size
        'ext_hidden_ratio': 0.67,  # external_gru = 64
        'ext_layers': 1,
        'factor_layers': 2,
        'dropout': 0.15,
        'heads': 6,  # More attention heads
        'epochs': 60,
        'patience': 6
    },
    
    'config_6': {
        'name': 'Balanced External-Factor',
        'lr': 1.5e-3,
        'weight_decay': 1e-3,
        'hidden': 64,
        'ext_hidden_ratio': 0.75,  # external_gru = 48
        'ext_layers': 2,
        'factor_layers': 3,  # Deeper factor GRU
        'dropout': 0.12,
        'heads': 4,
        'epochs': 60,
        'patience': 6
    }
}

def create_modified_script(config_name, config, base_script_path, output_script_path):
    """Create a modified version of the base script with specific hyperparameters."""
    
    with open(base_script_path, 'r') as f:
        content = f.read()
    
    # Modify the SetRankerWithExternal class to accept new parameters
    old_init = """    def __init__(self, n_factors: int, input_size=4, f_ext=9, k_history=24, gru_hidden=64, n_heads=4, d_model=64, emb_dim=32, dropout=0.1):"""
    
    new_init = f"""    def __init__(self, n_factors: int, input_size=4, f_ext=9, k_history=24, gru_hidden=64, n_heads=4, d_model=64, emb_dim=32, dropout=0.1, ext_hidden_ratio=0.5, ext_layers=1, factor_layers=2):"""
    
    content = content.replace(old_init, new_init)
    
    # Update the GRU initialization
    old_factor_gru = """        self.factor_gru = nn.GRU(input_size, gru_hidden, num_layers=2, dropout=dropout if 2 > 1 else 0.0, batch_first=True)"""
    new_factor_gru = f"""        self.factor_gru = nn.GRU(input_size, gru_hidden, num_layers=factor_layers, dropout=dropout if factor_layers > 1 else 0.0, batch_first=True)"""
    content = content.replace(old_factor_gru, new_factor_gru)
    
    old_external_gru = """            self.external_gru = nn.GRU(f_ext, gru_hidden//2, num_layers=1, batch_first=True)
            self.external_proj = nn.Linear(gru_hidden//2, gru_hidden//2)
            # Combined features: factor_emb (64) + external_context (32) = 96
            combined_dim = gru_hidden + gru_hidden//2"""
    
    new_external_gru = f"""            ext_hidden = int(gru_hidden * ext_hidden_ratio)
            self.external_gru = nn.GRU(f_ext, ext_hidden, num_layers=ext_layers, dropout=dropout if ext_layers > 1 else 0.0, batch_first=True)
            self.external_proj = nn.Linear(ext_hidden, ext_hidden)
            # Combined features: factor_emb + external_context
            combined_dim = gru_hidden + ext_hidden"""
    
    content = content.replace(old_external_gru, new_external_gru)
    
    # Update the forward pass to handle variable external hidden size
    old_forward_external = """        # Process external history if available: [B, K, F_ext] -> [B, gru_hidden//2]
        if external_seq is not None and self.external_gru is not None:
            h_external = self.external_gru(external_seq)[1][-1]  # [B, gru_hidden//2]
            h_external = self.external_proj(h_external)  # [B, gru_hidden//2]
            # Broadcast external context to all factors: [B, gru_hidden//2] -> [B, N, gru_hidden//2]
            h_external = h_external.unsqueeze(1).expand(B, N, -1)
            # Concatenate factor and external features
            h_combined = torch.cat([h_factor, h_external], dim=-1)  # [B, N, gru_hidden + gru_hidden//2]"""
    
    new_forward_external = """        # Process external history if available: [B, K, F_ext] -> [B, ext_hidden]
        if external_seq is not None and self.external_gru is not None:
            h_external = self.external_gru(external_seq)[1][-1]  # [B, ext_hidden]
            h_external = self.external_proj(h_external)  # [B, ext_hidden]
            # Broadcast external context to all factors: [B, ext_hidden] -> [B, N, ext_hidden]
            h_external = h_external.unsqueeze(1).expand(B, N, -1)
            # Concatenate factor and external features
            h_combined = torch.cat([h_factor, h_external], dim=-1)  # [B, N, gru_hidden + ext_hidden]"""
    
    content = content.replace(old_forward_external, new_forward_external)
    
    # Update model instantiation to pass new parameters
    old_model_creation = """    model = SetRankerWithExternal(
        n_factors=M, 
        input_size=F_in, 
        f_ext=data.external_features.shape[-1] if data.external_features is not None else 0,
        k_history=k_hist, 
        gru_hidden=args.hidden, 
        n_heads=args.heads, 
        d_model=args.hidden, 
        emb_dim=args.emb_dim, 
        dropout=args.dropout
    ).to(device)"""
    
    new_model_creation = f"""    model = SetRankerWithExternal(
        n_factors=M, 
        input_size=F_in, 
        f_ext=data.external_features.shape[-1] if data.external_features is not None else 0,
        k_history=k_hist, 
        gru_hidden=args.hidden, 
        n_heads=args.heads, 
        d_model=args.hidden, 
        emb_dim=args.emb_dim, 
        dropout=args.dropout,
        ext_hidden_ratio={config['ext_hidden_ratio']},
        ext_layers={config['ext_layers']},
        factor_layers={config['factor_layers']}
    ).to(device)"""
    
    content = content.replace(old_model_creation, new_model_creation)
    
    # Add dropout argument
    if 'ap.add_argument(\'--dropout\'' not in content:
        dropout_line = "    ap.add_argument('--emb_dim', type=int, default=32)\n"
        new_dropout_line = dropout_line + "    ap.add_argument('--dropout', type=float, default=0.1)\n"
        content = content.replace(dropout_line, new_dropout_line)
    
    # Update default output directory
    content = content.replace(
        "default='output_F1_external'",
        f"default='output_F1_external_{config_name}'"
    )
    
    # Update default file names
    content = content.replace(
        "default='predictions_f1_external.xlsx'",
        f"default='predictions_f1_external_{config_name}.xlsx'"
    )
    content = content.replace(
        "default='performance_f1_external.pdf'",
        f"default='performance_f1_external_{config_name}.pdf'"
    )
    
    with open(output_script_path, 'w') as f:
        f.write(content)

def run_experiment(config_name, config, max_oos=50):
    """Run a single hyperparameter experiment."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {config['name']} ({config_name})")
    print(f"{'='*60}")
    
    # Create modified script
    base_script = 'run_set_ranker_F1_external.py'
    exp_script = f'run_set_ranker_F1_external_{config_name}.py'
    
    create_modified_script(config_name, config, base_script, exp_script)
    
    # Build command with hyperparameters
    cmd = [
        'python', exp_script,
        '--lr', str(config['lr']),
        '--weight_decay', str(config['weight_decay']),
        '--hidden', str(config['hidden']),
        '--heads', str(config['heads']),
        '--epochs', str(config['epochs']),
        '--patience', str(config['patience']),
        '--dropout', str(config['dropout']),
        '--max_oos', str(max_oos),
        '--verbose'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✓ Experiment {config_name} completed successfully")
            return True, result.stdout, result.stderr
        else:
            print(f"✗ Experiment {config_name} failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"✗ Experiment {config_name} timed out after 1 hour")
        return False, "", "Timeout"
    except Exception as e:
        print(f"✗ Experiment {config_name} failed with exception: {e}")
        return False, "", str(e)

def extract_performance_metrics(output_dir):
    """Extract performance metrics from experiment output."""
    try:
        perf_file = os.path.join(output_dir, 'set_ranker_perf_F1_external.xlsx')
        if not os.path.exists(perf_file):
            return None
            
        # Read performance summary
        df = pd.read_excel(perf_file, sheet_name=0)  # First sheet is summary
        
        # Extract Model (gross) row
        model_row = df[df['Strategy'] == 'Model (gross)'].iloc[0]
        
        return {
            'CAGR': model_row['CAGR'],
            'Volatility': model_row['Vol'],
            'Sharpe': model_row['Sharpe'],
            'Sortino': model_row['Sortino'],
            'MaxDD': model_row['MaxDD'],
            'Calmar': model_row['Calmar'],
            'HitRate': model_row['HitRate'],
            'Best': model_row['Best'],
            'Worst': model_row['Worst']
        }
    except Exception as e:
        print(f"Error extracting metrics from {output_dir}: {e}")
        return None

def main():
    print("Starting Hyperparameter Optimization Experiments")
    print(f"Testing {len(CONFIGS)} configurations")
    
    # Create results directory
    results_dir = 'hyperparameter_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Store experiment results
    results = []
    
    # Run experiments (limit to 50 OOS months for faster testing)
    max_oos = 50
    
    for config_name, config in CONFIGS.items():
        print(f"\nStarting {config_name}: {config['name']}")
        
        success, stdout, stderr = run_experiment(config_name, config, max_oos)
        
        # Extract performance metrics
        output_dir = f"output_F1_external_{config_name}"
        metrics = extract_performance_metrics(output_dir) if success else None
        
        result = {
            'config_name': config_name,
            'config_display_name': config['name'],
            'success': success,
            'lr': config['lr'],
            'weight_decay': config['weight_decay'],
            'hidden': config['hidden'],
            'ext_hidden_ratio': config['ext_hidden_ratio'],
            'ext_layers': config['ext_layers'],
            'factor_layers': config['factor_layers'],
            'dropout': config['dropout'],
            'heads': config['heads'],
            'epochs': config['epochs'],
            'patience': config['patience']
        }
        
        if metrics:
            result.update(metrics)
        else:
            # Fill with NaN for failed experiments
            result.update({
                'CAGR': np.nan,
                'Volatility': np.nan,
                'Sharpe': np.nan,
                'Sortino': np.nan,
                'MaxDD': np.nan,
                'Calmar': np.nan,
                'HitRate': np.nan,
                'Best': np.nan,
                'Worst': np.nan
            })
        
        results.append(result)
        
        # Save intermediate results
        results_df = pd.DataFrame(results)
        results_df.to_excel(os.path.join(results_dir, 'hyperparameter_results_partial.xlsx'), index=False)
    
    # Final results summary
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio (descending)
    results_df = results_df.sort_values('Sharpe', ascending=False, na_last=True)
    
    # Save final results
    results_df.to_excel(os.path.join(results_dir, 'hyperparameter_results_final.xlsx'), index=False)
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    
    # Display top results
    display_cols = ['config_display_name', 'CAGR', 'Sharpe', 'Sortino', 'MaxDD', 'HitRate', 'lr', 'hidden', 'ext_hidden_ratio']
    print(results_df[display_cols].head(10).to_string(index=False, float_format='%.4f'))
    
    print(f"\nFull results saved to: {results_dir}/hyperparameter_results_final.xlsx")
    
    # Find best configuration
    best_config = results_df.iloc[0]
    print(f"\nBest Configuration: {best_config['config_display_name']}")
    print(f"Sharpe Ratio: {best_config['Sharpe']:.4f}")
    print(f"CAGR: {best_config['CAGR']:.4f}")
    print(f"Parameters: lr={best_config['lr']}, hidden={best_config['hidden']}, ext_ratio={best_config['ext_hidden_ratio']}")

if __name__ == '__main__':
    main()
