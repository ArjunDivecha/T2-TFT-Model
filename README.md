# T2-TFT-Model: Temporal SetRanker for Factor Ranking

A high-performance transformer-based model for cross-sectional factor selection and ranking with optimized training pipeline achieving 300x speedup over baseline implementations.

## Overview

This repository contains a complete implementation of a Temporal SetRanker that combines:
- **GRU-based temporal modeling** for factor history (24-month sequences)
- **SetTransformer architecture** for cross-sectional factor interactions
- **Vectorized LambdaRank loss** optimized for NDCG@5 ranking performance
- **Warm-start retraining** for efficient walk-forward evaluation
- **Comprehensive baseline comparisons** (RawRank, Isotonic Calibration)

## Performance Results (21+ years, 257 months)

| Strategy | CAGR | Sharpe | MaxDD | Hit Rate |
|----------|------|--------|-------|----------|
| **Transformer Model** | 2.43% | 0.55 | -8.2% | 59.1% |
| **RawRank Baseline** | 3.00% | 0.68 | -11.3% | 60.3% |
| **Isotonic Calibration** | 1.33% | 0.29 | -9.7% | 50.6% |

## Key Features

### Technical Optimizations
- **300x speedup**: Full evaluation in 35 minutes (vs days originally)
- **Vectorized LambdaRank**: Eliminated O(N²) bottleneck with PyTorch operations
- **Warm-start training**: 99.6% of months use previous weights for faster convergence
- **Model persistence**: All trained models saved for backtesting and analysis

### Architecture
- **Temporal Component**: Per-factor GRU (d=64, 2 layers) over K=24 month sequences
- **Cross-sectional Component**: SetTransformer (1 block, 4-8 heads, d_model=64)
- **Factor Embeddings**: 32-dimensional learned embeddings per factor
- **Loss Function**: Vectorized LambdaRank targeting NDCG@5

### Data Pipeline
- **Input**: T60 forecasts (already lagged) and T2_Optimizer realized returns
- **Features**: Z-score normalized, rank percentiles, historical sequences
- **Missing Data**: Filled with cross-sectional means, fully logged
- **Validation**: Walk-forward expanding window with early stopping

## Files

### Core Implementation
- `run_set_ranker.py` - Main training and evaluation script
- `output.py` - Post-analysis and performance reporting

### Data Requirements
- `T60.xlsx` - Factor forecasts (T60 format)
- `T2_Optimizer.xlsx` - Realized returns (T2 format)

### Outputs
- `set_ranker_outputs.xlsx` - Training results, predictions, metrics
- `set_ranker_plots.pdf` - Performance visualization
- `set_ranker_perf.xlsx` - Post-analysis performance statistics  
- `set_ranker_perf_plots.pdf` - Comprehensive portfolio analysis
- `models/` - Saved model states for each OOS month

## Usage

### Quick Start
```bash
# Basic run with optimizations
python3 run_set_ranker.py --epochs 25 --warm_start --warm_epochs 12 --val_every 3 --save_models

# Fast test (12 months)
python3 run_set_ranker.py --epochs 15 --warm_start --warm_epochs 8 --max_oos 12

# Generate performance reports
python3 output.py
```

### Command Line Arguments
- `--epochs`: Training epochs for first month (default: 50)
- `--warm_start`: Enable warm-start retraining from previous month
- `--warm_epochs`: Epochs for warm-started months (default: 25)
- `--val_every`: Validation frequency (default: 1)
- `--save_models`: Save trained models for each month
- `--models_dir`: Directory for saved models (default: ./models)
- `--max_oos`: Limit number of OOS months for testing

## Architecture Details

### Temporal SetRanker Model
```python
class TemporalSetRanker(nn.Module):
    def __init__(self, n_factors, input_size, k_history=24, 
                 gru_hidden=64, n_heads=4, d_model=64, emb_dim=32):
        # Per-factor GRU for temporal modeling
        self.gru = nn.GRU(input_size, gru_hidden, 2, batch_first=True)
        
        # Factor ID embeddings
        self.factor_emb = nn.Embedding(n_factors, emb_dim)
        
        # SetTransformer for cross-sectional interactions
        self.set_transformer = SetTransformer(
            dim_input=gru_hidden + emb_dim,
            num_outputs=1, dim_output=d_model,
            num_inds=32, dim_hidden=128, num_heads=n_heads
        )
        
        # Final scoring layer
        self.scorer = nn.Linear(d_model, 1)
```

### Vectorized LambdaRank Loss
Optimized implementation using PyTorch tensor operations:
- Computes pairwise score differences in parallel
- Applies Delta NDCG weighting for top-K focus
- Handles batch processing with valid masks
- 10-50x faster than nested Python loops

### Walk-Forward Training
- **Expanding window**: Uses all historical data for training
- **Warm-start**: Initialize from previous month's weights
- **Early stopping**: Based on validation NDCG@5
- **Deterministic**: Fixed seeds and stable tie-breaking

## Performance Optimizations

### S1: Vectorized LambdaRank Loss
- **Before**: O(N²) nested Python loops
- **After**: Vectorized PyTorch operations
- **Speedup**: 10-50x per training epoch

### S2: Warm-Start Retraining
- **Before**: Train from scratch each month
- **After**: Initialize from previous month's weights
- **Efficiency**: 99.6% of months use warm-start
- **Convergence**: Faster training with fewer epochs

## Data Alignment

**Critical Fix Applied**: T60 forecasts are already lagged, so:
- T60(t) predicts T2_Optimizer(t) returns
- No additional shifting required
- Column: `RealizedReturn(t)` (not t+1)

## Requirements

```
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.3.0
xlsxwriter>=3.0.0
tqdm>=4.60.0
scikit-learn>=1.0.0
```

## Development History

### Major Milestones
1. **Initial Implementation**: Basic transformer with standard training
2. **S1 Optimization**: Vectorized LambdaRank loss (300x speedup)
3. **S2 Optimization**: Warm-start retraining (99.6% efficiency)
4. **Data Alignment Fix**: Corrected T60→T2 mapping
5. **Performance Validation**: Full 21-year backtest completed

### Bug Fixes
- **Scaling Issue**: Fixed percentage points to decimal conversion
- **Alignment Issue**: Corrected forecast-to-return mapping
- **Validation Handling**: Fixed empty validation set errors

## Future Enhancements

### Potential Optimizations
- **S4**: Isotonic baseline caching for repeated evaluations
- **S5**: Sampled pairs in LambdaRank for very large factor universes
- **Mixed Precision**: FP16 training for memory efficiency

### Model Extensions
- **Multi-horizon**: Extend to multiple prediction horizons
- **Regime Awareness**: Incorporate market state features
- **Ensemble Methods**: Combine multiple model architectures

## License

MIT License - See LICENSE file for details.

## Citation

```bibtex
@software{temporal_setranker_2025,
  title={T2-TFT-Model: Temporal SetRanker for Factor Ranking},
  author={Divecha, Arjun},
  year={2025},
  url={https://github.com/ArjunDivecha/T2-TFT-Model}
}
```
