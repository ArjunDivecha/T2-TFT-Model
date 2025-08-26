# External Factor Integration Plan for Factor Ranking Model

## Current State
- **Base Model**: `run_set_ranker_F1_detailed.py` - SetTransformer with GRU temporal encoder
- **Performance**: 15.3% average monthly Spearman correlation between F1 and RawRank baseline
- **External Data**: `Extrernal Data.xlsx` with 9 market factors available
- **Architecture**: 2-layer GRU (hidden=64) + SetTransformer + LambdaRank loss

## External Data Available
From `Extrernal Data.xlsx` sheet "1MTR":
- **1MTR, 3MTR, 12MTR**: SP500 trailing returns (1, 3, 12 months)
- **Bond Yield Change**: Interest rate environment indicator
- **Advance Decline**: Market breadth (NYSE advance/decline ratio)
- **Dollar Index**: USD strength (DXY)
- **GDP Growth**: Economic growth rate
- **Inflation**: Inflation rate
- **VIX**: Volatility index

## Implementation Strategy

### 1. Data Integration
**File**: Modify `build_dataset()` function in new `run_set_ranker_F1_external.py`

```python
def build_dataset_with_external(t60, t2, ext, include_cols=None):
    # Merge T60 forecasts, T2 returns, and external data by Date
    # External columns: ['1MTR', '3MTR', '12MTR', 'Bond Yield Change', 
    #                   'Advance Decline', 'Dollar Index', 'GDP Growth', 'Inflation', 'VIX']
    # Handle missing external data with forward-fill + trailing mean
    # Return DataBundle with additional external_features array [T, F_ext]
```

### 2. Model Architecture Enhancement
**Core Change**: Add external context encoder to existing SetTransformer

```python
class SetRankerWithExternal(nn.Module):
    def __init__(self, n_factors, f_in=4, f_ext=9, hidden_dim=64):
        # Factor temporal encoder (existing)
        self.factor_gru = nn.GRU(f_in, hidden_dim, num_layers=2, batch_first=True)
        
        # NEW: External context encoder
        self.external_gru = nn.GRU(f_ext, hidden_dim//2, num_layers=1, batch_first=True)
        self.external_proj = nn.Linear(hidden_dim//2, hidden_dim//2)
        
        # Combined features: factor_emb (64) + external_context (32) = 96
        combined_dim = hidden_dim + hidden_dim//2
        self.set_transformer = SetTransformerBlock(combined_dim, num_heads=4)
        self.scorer = nn.Sequential(nn.Linear(combined_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    
    def forward(self, factor_seq, external_seq):
        # Process factor histories: [batch, seq_len, n_factors, f_in] -> [batch, n_factors, hidden_dim]
        # Process external history: [batch, seq_len, f_ext] -> [batch, hidden_dim//2]
        # Broadcast external context to all factors
        # Concatenate and pass through SetTransformer
        # Return factor scores: [batch, n_factors]
```

### 3. Training Pipeline Modifications
**Walk-Forward Training**: Maintain existing structure but include external features

```python
# For each OOS month t:
# 1. Training data: factor_seq[t-k_hist:t] + external_seq[t-k_hist:t] -> labels[t]
# 2. Train model with LambdaRank loss (unchanged)
# 3. Inference: predict scores for month t using factor + external context
# 4. Compare against RawRank, Iso, and original F1 baselines
```

### 4. Baseline Comparisons
**Three-way comparison**:
- **Model + External**: New model with external factors
- **RawRank**: T60 trailing 60-month mean baseline
- **Iso**: Isotropic (equal weight) baseline
- **F1_Original**: Original F1 model without external factors (for ablation study)

### 5. Expected Benefits
**Regime Awareness**:
- **VIX regimes**: High volatility periods may favor different factors
- **Interest rate cycles**: Bond yield changes affect factor performance
- **Market breadth**: Advance/decline signals market health
- **Economic cycles**: GDP growth and inflation provide fundamental context
- **Currency effects**: Dollar strength impacts international factors

### 6. Implementation Steps

#### Step 1: Copy and Modify Base Model
```bash
cp run_set_ranker_F1_detailed.py run_set_ranker_F1_external.py
```

#### Step 2: Data Loading Changes
- Add external data reading in `read_sheets()`
- Modify `build_dataset()` to merge external factors
- Handle missing external data (forward-fill + mean imputation)
- Add external features to DataBundle class

#### Step 3: Model Architecture
- Create `SetRankerWithExternal` class
- Add external GRU encoder
- Modify forward pass to include external context
- Update feature dimensions (factor: 64, external: 32, combined: 96)

#### Step 4: Training Loop
- Modify data loader to include external sequences
- Update training function signatures
- Maintain existing LambdaRank loss and early stopping

#### Step 5: Output and Comparison
- Generate predictions for all three strategies
- Compute NDCG@5, Regret@5, AvgTop5 metrics
- Create Excel output with baseline comparison sheets
- Generate PDF plots showing performance over time

### 7. File Structure
```
output_F1_external/
├── predictions_f1_external.xlsx
│   ├── Predictions (Model + External)
│   ├── Predictions_RawRank
│   ├── Predictions_Iso
│   ├── Metrics (Model + External)
│   ├── Metrics_RawRank
│   ├── Metrics_Iso
│   └── Metrics_Aggregate (comparison summary)
├── performance_f1_external.pdf
│   ├── NDCG@5 time series (3-way comparison)
│   ├── Regret@5 time series
│   └── AvgTop5 returns comparison
└── models/ (optional model checkpoints)
```

### 8. Success Metrics
**Primary**: Improvement in NDCG@5 vs original F1 model
**Secondary**: 
- Reduced regret@5 (better top-5 selection)
- Higher AvgTop5 returns
- More stable performance across market regimes
- Lower correlation with RawRank baseline (indicating model independence)

### 9. Risk Mitigation
- **Overfitting**: Use same early stopping and validation as base model
- **Data leakage**: Ensure external factors at time t don't include future information
- **Missing data**: Robust handling with forward-fill and mean imputation
- **Regime changes**: External factors should help adapt to changing market conditions

### 10. Testing Strategy
- **Quick test**: Run with `--max_oos 12` for 12 months validation
- **Full backtest**: Run complete time series once validated
- **Ablation study**: Compare with/without each external factor group
- **Regime analysis**: Examine performance in different market conditions (high/low VIX, rising/falling rates)

## Next Actions
1. Implement data integration for external factors
2. Create SetRankerWithExternal model architecture
3. Modify training pipeline to handle external context
4. Run initial validation with 12-month test
5. Analyze results and compare against baselines
6. If successful, run full backtest and generate comprehensive analysis