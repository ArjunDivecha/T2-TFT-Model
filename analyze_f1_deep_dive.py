import pandas as pd
from scipy.stats import spearmanr
import numpy as np

# --- Configuration ---
INPUT_FILE = 'output_F1_detailed/predictions_f1_detailed.xlsx'

def analyze_rank_correlation():
    """
    Analyzes the Spearman rank correlation between the F1 model and the RawRank baseline.
    """
    print('[1/3] Loading data...')
    # Load the F1 model's wide-format ranks
    f1_ranks_wide = pd.read_excel(INPUT_FILE, sheet_name='Ranks_Wide', index_col=0)
    
    # Load the RawRank baseline predictions to construct its wide-format ranks
    raw_preds = pd.read_excel(INPUT_FILE, sheet_name='Predictions_RawRank')
    
    print('[2/3] Processing ranks...')
    # Pivot the RawRank data to get ranks in wide format
    raw_ranks_wide = raw_preds.pivot(index='Date', columns='Factor', values='Rank')
    
    # Align columns and dates between the two dataframes
    common_factors = f1_ranks_wide.columns.intersection(raw_ranks_wide.columns)
    common_dates = f1_ranks_wide.index.intersection(raw_ranks_wide.index)
    
    f1_ranks_aligned = f1_ranks_wide.loc[common_dates, common_factors]
    raw_ranks_aligned = raw_ranks_wide.loc[common_dates, common_factors]
    
    print('[3/3] Calculating monthly correlation...')
    correlations = []
    for date in common_dates:
        f1_rank_series = f1_ranks_aligned.loc[date].dropna()
        raw_rank_series = raw_ranks_aligned.loc[date].dropna()
        
        # Align on common factors for the specific month
        common_monthly_factors = f1_rank_series.index.intersection(raw_rank_series.index)
        if len(common_monthly_factors) < 2:
            continue
            
        f1_rank_series = f1_rank_series[common_monthly_factors]
        raw_rank_series = raw_rank_series[common_monthly_factors]
        
        # Calculate Spearman correlation for the month
        corr, _ = spearmanr(f1_rank_series, raw_rank_series)
        correlations.append(corr)
        
    mean_correlation = np.mean(correlations)
    
    print('\n--- Rank Correlation Analysis ---')
    print(f'Average monthly Spearman correlation between F1 and RawRank: {mean_correlation:.4f}')
    print('---------------------------------')

if __name__ == '__main__':
    analyze_rank_correlation()
