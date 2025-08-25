#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: run_set_ranker_perf.py
LAST UPDATED: 2025-08-23

PURPOSE
- Read set_ranker_outputs.xlsx from the ranker pipeline
- Build monthly portfolio return series for:
    • Model (from 'Metrics' sheet, AvgTop5)
    • RawRank baseline (from 'Metrics_RawRank', AvgTop5)
    • Iso baseline (from 'Metrics_Iso', AvgTop5)
- Compute & export:
    • Cumulative equity curves (gross) and drawdowns
    • Turnover for each strategy (from Predictions sheets)
    • Rolling 12m return & Sharpe
    • Excess vs chosen baseline (default RawRank)
- Output:
    • Excel: set_ranker_perf.xlsx (Summary, Series, Equity, Drawdown, Turnover, Excess)
    • PDF:   set_ranker_perf_plots.pdf (curves, drawdowns, rolling stats, turnover, excess)

USAGE (defaults target your junk5 folder):
python run_set_ranker_perf.py \
  --in_xlsx "/Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/set_ranker_outputs.xlsx" \
  --out_xlsx "/Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/set_ranker_perf.xlsx" \
  --out_pdf  "/Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/set_ranker_perf_plots.pdf" \
  --baseline RawRank

NOTES
- No transaction costs are applied (gross). If you want an optional per-switch cost,
  set --switch_cost_bps > 0 and we’ll compute a “net after turnover” series as well.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ------------------------------ Helpers ------------------------------

@dataclass
class StratSeries:
    name: str
    dates: pd.DatetimeIndex
    r: pd.Series          # monthly returns
    equity: pd.Series     # cumulative product
    dd: pd.Series         # drawdown (negative numbers)
    roll12_ret: pd.Series # rolling 12m total return (not annualized)
    roll12_sh: pd.Series  # rolling 12m Sharpe (monthly rf=0)
    tovr: Optional[pd.Series] = None       # monthly turnover rate (0..1)
    r_net: Optional[pd.Series] = None      # returns net of turnover cost
    equity_net: Optional[pd.Series] = None # equity net of turnover cost
    dd_net: Optional[pd.Series] = None     # drawdown net


def annualize_stats(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if r.empty:
        return dict(CAGR=np.nan, Vol=np.nan, Sharpe=np.nan,
                    Sortino=np.nan, MaxDD=np.nan, Calmar=np.nan,
                    HitRate=np.nan, Best=np.nan, Worst=np.nan)
    n = len(r)
    total = float((1.0 + r).prod())
    yrs = n / 12.0
    cagr = total ** (1.0 / yrs) - 1.0 if yrs > 0 else np.nan
    vol_m = float(r.std())
    vol = vol_m * np.sqrt(12.0)
    sharpe = (cagr / vol) if vol > 0 else np.nan
    neg = r[r < 0.0]
    sortino = (cagr / (float(neg.std()) * np.sqrt(12.0))) if len(neg) > 1 and float(neg.std()) > 0 else np.nan
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    maxdd = float(dd.min()) if not dd.empty else np.nan
    calmar = (cagr / abs(maxdd)) if (maxdd < 0) else np.nan
    hit = float((r > 0).mean())
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, Sortino=sortino,
                MaxDD=maxdd, Calmar=calmar, HitRate=hit,
                Best=float(r.max()), Worst=float(r.min()))


def make_equity_and_dd(r: pd.Series) -> Tuple[pd.Series, pd.Series]:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return eq, dd


def rolling_12m_metrics(r: pd.Series) -> Tuple[pd.Series, pd.Series]:
    # rolling 12m total return (not annualized) and rolling 12m Sharpe (monthly rf=0)
    ret12 = (1.0 + r).rolling(12).apply(np.prod, raw=True) - 1.0
    sh12 = r.rolling(12).mean() / r.rolling(12).std().replace(0, np.nan)
    return ret12, sh12


def turnover_from_predictions(preds_df: pd.DataFrame, topk: int = 5) -> pd.Series:
    """Compute monthly turnover rate = (# names replaced)/topk from Predictions sheet."""
    # preds_df columns: Date, Rank, Factor, Score, RealizedReturn(t+1)
    df = preds_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'Rank'])
    # Build a mapping Date -> set of top-k factors
    sets = df.groupby('Date')['Factor'].apply(lambda s: tuple(s.head(topk))).sort_index()
    dates = sets.index
    tovr = []
    prev_set = None
    for d in dates:
        cur = set(sets.loc[d])
        if prev_set is None:
            tovr.append(np.nan)
        else:
            overlap = len(prev_set & cur)
            changed = max(0, topk - overlap)
            tovr.append(changed / float(topk))
        prev_set = cur
    return pd.Series(tovr, index=dates, name='Turnover')


def apply_turnover_cost(r: pd.Series, tovr: pd.Series, switch_cost_bps: float) -> pd.Series:
    """Subtract per-switch cost (bps per name change) from monthly returns."""
    if switch_cost_bps is None or switch_cost_bps <= 0:
        return r
    # cost per month = (#names changed) * (bps) / 10_000
    # tovr is fraction changed; for top-5, #changed = tovr*5
    cost = tovr.fillna(0.0) * 5.0 * (switch_cost_bps / 10_000.0)
    return r - cost


def read_metrics_series(xlsx_path: str, sheet: str) -> pd.Series:
    """Read 'Metrics' style sheet and return the monthly AvgTop5 series indexed by Date.
    Converts from percentage points to decimal returns (divides by 100)."""
    m = pd.read_excel(xlsx_path, sheet_name=sheet)
    if 'Date' not in m.columns or 'AvgTop5' not in m.columns:
        return pd.Series(dtype=float)
    m['Date'] = pd.to_datetime(m['Date'])
    m = m.sort_values('Date')
    # Convert percentage points to decimal returns (0.89% -> 0.0089)
    return pd.Series(m['AvgTop5'].values / 100.0, index=m['Date'])


def read_predictions_sheet(xlsx_path: str, sheet: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if not {'Date', 'Rank', 'Factor', 'Score'}.issubset(df.columns):
            print(f"Warning: Sheet '{sheet}' is missing required columns (Date, Rank, Factor, Score).")
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Warning: Could not read or process sheet '{sheet}'. Error: {e}")
        return None


def calculate_rank_correlation(p_model: pd.DataFrame, p_raw: pd.DataFrame) -> Optional[pd.Series]:
    """Calculates monthly Spearman rank correlation between Model and RawRank scores."""
    if p_model is None or p_raw is None:
        return None

    # Merge the two prediction dataframes on Date and Factor
    merged = pd.merge(
        p_model[['Date', 'Factor', 'Score']],
        p_raw[['Date', 'Factor', 'Score']],
        on=['Date', 'Factor'],
        suffixes=('_model', '_raw')
    )

    if merged.empty:
        return None

    # Group by date and calculate Spearman correlation
    correlations = merged.groupby('Date').apply(
        lambda g: g[['Score_model', 'Score_raw']].corr(method='spearman').iloc[0, 1]
    )
    correlations.name = 'Spearman_Correlation'
    return correlations


# ------------------------------ Main Pipeline ------------------------------

def build_strategy(name: str,
                   r: pd.Series,
                   preds_df: Optional[pd.DataFrame],
                   switch_cost_bps: float = 0.0) -> StratSeries:
    r = r.sort_index()
    dates = r.index
    tovr = turnover_from_predictions(preds_df) if preds_df is not None else None
    # Net-of-turnover (optional)
    r_net = apply_turnover_cost(r, tovr, switch_cost_bps) if tovr is not None else None

    equity, dd = make_equity_and_dd(r)
    roll_ret, roll_sh = rolling_12m_metrics(r)

    if r_net is not None:
        equity_net, dd_net = make_equity_and_dd(r_net)
    else:
        equity_net, dd_net = None, None

    return StratSeries(
        name=name,
        dates=dates,
        r=r,
        equity=equity,
        dd=dd,
        roll12_ret=roll_ret,
        roll12_sh=roll_sh,
        tovr=tovr,
        r_net=r_net,
        equity_net=equity_net,
        dd_net=dd_net
    )


def main():
    ap = argparse.ArgumentParser(description="Compute performance, drawdowns, turnover, and plots from set_ranker_outputs.xlsx")
    ap.add_argument("--in_xlsx", type=str, default="/Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/set_ranker_outputs.xlsx")
    ap.add_argument("--out_xlsx", type=str, default="/Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/set_ranker_perf.xlsx")
    ap.add_argument("--out_pdf",  type=str, default="/Users/macbook2024/Dropbox/AAA Backup/A Working/junk5/set_ranker_perf_plots.pdf")
    ap.add_argument("--baseline", type=str, default="RawRank", choices=["RawRank", "Iso", "Model"],
                    help="Baseline to compute excess vs. (default RawRank ≈ Top5Trailing60 if T60 are trailing-60 means)")
    ap.add_argument("--switch_cost_bps", type=float, default=0.0,
                    help="Optional per-switch cost in bps per name changed (e.g., 10 = 10bps). Default 0.")
    args = ap.parse_args()

    in_xlsx = args.in_xlsx

    # Read monthly AvgTop5 for strategies
    r_model = read_metrics_series(in_xlsx, "Metrics")
    r_raw   = read_metrics_series(in_xlsx, "Metrics_RawRank")
    r_iso   = read_metrics_series(in_xlsx, "Metrics_Iso")

    # Align all on a common date index (intersection)
    idx = None
    for s in [r_model, r_raw, r_iso]:
        idx = s.index if idx is None else idx.intersection(s.index)
    r_model = r_model.reindex(idx)
    r_raw   = r_raw.reindex(idx)
    r_iso   = r_iso.reindex(idx)

    # Predictions (for turnover)
    p_model = read_predictions_sheet(in_xlsx, "Predictions")
    p_raw   = read_predictions_sheet(in_xlsx, "Predictions_RawRank")
    p_iso   = read_predictions_sheet(in_xlsx, "Predictions_Iso")

    # Calculate rank correlation between model and raw T60 scores
    rank_corr = calculate_rank_correlation(p_model, p_raw)

    # Build strategy objects
    strat_model = build_strategy("Model", r_model, p_model, args.switch_cost_bps)
    strat_raw   = build_strategy("RawRank", r_raw, p_raw, args.switch_cost_bps)
    strat_iso   = build_strategy("Iso", r_iso, p_iso, args.switch_cost_bps)

    # Pick baseline for excess
    baseline_map = {"Model": strat_model, "RawRank": strat_raw, "Iso": strat_iso}
    base = baseline_map[args.baseline]

    # Compute excess series (Model/Raw/Iso vs baseline)
    def excess_series(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
        # arithmetic excess and cumulative relative (ratio of equities)
        # Ensure aligned
        al = a.dropna()
        bl = b.dropna()
        common = al.index.intersection(bl.index)
        a = al.reindex(common)
        b = bl.reindex(common)
        ex = a - b
        # Cumulative relative value: Π (1+a) / Π (1+b)
        cum_rel = ((1.0 + a).cumprod() / (1.0 + b).cumprod()) - 1.0
        return ex, cum_rel

    ex_model, exrel_model = excess_series(strat_model.r, base.r)
    ex_raw,   exrel_raw   = excess_series(strat_raw.r, base.r)
    ex_iso,   exrel_iso   = excess_series(strat_iso.r, base.r)

    # Summaries
    stats_model = annualize_stats(strat_model.r)
    stats_raw   = annualize_stats(strat_raw.r)
    stats_iso   = annualize_stats(strat_iso.r)

    # Optional net-of-turnover stats
    if args.switch_cost_bps > 0 and strat_model.r_net is not None:
        stats_model_net = annualize_stats(strat_model.r_net)
        stats_raw_net   = annualize_stats(strat_raw.r_net)
        stats_iso_net   = annualize_stats(strat_iso.r_net)
    else:
        stats_model_net = stats_raw_net = stats_iso_net = None

    # ---------------- Excel Output ----------------
    out_xlsx = args.out_xlsx
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as w:
        # Summary sheet
        rows = []
        for name, st, stn in [
            ("Model (gross)", strat_model, None if stats_model_net is None else None),
            ("RawRank (gross)", strat_raw, None),
            ("Iso (gross)", strat_iso, None),
        ]:
            s = annualize_stats(st.r)
            rows.append({"Strategy": name, **s})
        if stats_model_net is not None:
            rows.append({"Strategy": "Model (net)", **stats_model_net})
            rows.append({"Strategy": "RawRank (net)", **stats_raw_net})
            rows.append({"Strategy": "Iso (net)", **stats_iso_net})
        summary_df = pd.DataFrame(rows)
        summary_df.to_excel(w, index=False, sheet_name="Summary")

        # Series (monthly returns)
        series_df = pd.DataFrame({
            "Date": strat_model.dates,
            "Model_r": strat_model.r.values,
            "RawRank_r": strat_raw.r.values,
            "Iso_r": strat_iso.r.values
        }).set_index("Date")
        series_df.to_excel(w, sheet_name="Series")

        # Equity (gross)
        equity_df = pd.DataFrame({
            "Date": strat_model.dates,
            "Model": strat_model.equity.values,
            "RawRank": strat_raw.equity.values,
            "Iso": strat_iso.equity.values
        }).set_index("Date")
        equity_df.to_excel(w, sheet_name="Equity")

        # Drawdowns (gross)
        dd_df = pd.DataFrame({
            "Date": strat_model.dates,
            "Model": strat_model.dd.values,
            "RawRank": strat_raw.dd.values,
            "Iso": strat_iso.dd.values
        }).set_index("Date")
        dd_df.to_excel(w, sheet_name="Drawdown")

        # Turnover
        tovr_df = pd.DataFrame(index=equity_df.index)
        if strat_model.tovr is not None:
            tovr_df["Model"] = strat_model.tovr.reindex(tovr_df.index)
        if strat_raw.tovr is not None:
            tovr_df["RawRank"] = strat_raw.tovr.reindex(tovr_df.index)
        if strat_iso.tovr is not None:
            tovr_df["Iso"] = strat_iso.tovr.reindex(tovr_df.index)
        tovr_df.to_excel(w, sheet_name="Turnover")

        # Excess vs baseline (gross)
        ex_df = pd.DataFrame({
            "Date": ex_model.index,
            "Model_ex": ex_model.values,
            "RawRank_ex": ex_raw.reindex(ex_model.index).values,
            "Iso_ex": ex_iso.reindex(ex_model.index).values
        }).set_index("Date")
        exrel_df = pd.DataFrame({
            "Date": exrel_model.index,
            "Model_exrel": exrel_model.values,
            "RawRank_exrel": exrel_raw.reindex(exrel_model.index).values,
            "Iso_exrel": exrel_iso.reindex(exrel_model.index).values
        }).set_index("Date")
        ex_df.to_excel(w, sheet_name=f"Excess_vs_{args.baseline}")
        exrel_df.to_excel(w, sheet_name=f"ExcessRel_vs_{args.baseline}")

        # Rolling 12m metrics (gross)
        roll_df = pd.DataFrame({
            "Date": strat_model.dates,
            "Model_roll12_ret": strat_model.roll12_ret.values,
            "Model_roll12_sh": strat_model.roll12_sh.values,
            "RawRank_roll12_ret": strat_raw.roll12_ret.values,
            "RawRank_roll12_sh": strat_raw.roll12_sh.values,
            "Iso_roll12_ret": strat_iso.roll12_ret.values,
            "Iso_roll12_sh": strat_iso.roll12_sh.values
        }).set_index("Date")
        roll_df.to_excel(w, sheet_name="Rolling12m")

        # Correlation vs RawRank
        if rank_corr is not None:
            rank_corr.to_excel(w, sheet_name="Correlation_vs_RawRank", header=True)

        # Optional Net sheets
        if args.switch_cost_bps > 0 and strat_model.r_net is not None:
            equity_net_df = pd.DataFrame({
                "Date": strat_model.dates,
                "Model_net": strat_model.equity_net.values,
                "RawRank_net": strat_raw.equity_net.values,
                "Iso_net": strat_iso.equity_net.values
            }).set_index("Date")
            equity_net_df.to_excel(w, sheet_name="Equity_Net")

            dd_net_df = pd.DataFrame({
                "Date": strat_model.dates,
                "Model_net": strat_model.dd_net.values,
                "RawRank_net": strat_raw.dd_net.values,
                "Iso_net": strat_iso.dd_net.values
            }).set_index("Date")
            dd_net_df.to_excel(w, sheet_name="Drawdown_Net")

    # ---------------- PDF Plots ----------------
    out_pdf = args.out_pdf
    with PdfPages(out_pdf) as pdf:
        # Equity curves (gross)
        plt.figure(figsize=(10, 5))
        plt.plot(equity_df.index, equity_df["Model"], label="Model")
        plt.plot(equity_df.index, equity_df["RawRank"], label="RawRank")
        plt.plot(equity_df.index, equity_df["Iso"], label="Iso")
        plt.title("Cumulative Growth of $1 (Gross)")
        plt.xlabel("Date"); plt.ylabel("Value")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        pdf.savefig(); plt.close()

        # Drawdowns (gross)
        plt.figure(figsize=(10, 4))
        plt.plot(dd_df.index, dd_df["Model"], label="Model")
        plt.plot(dd_df.index, dd_df["RawRank"], label="RawRank")
        plt.plot(dd_df.index, dd_df["Iso"], label="Iso")
        plt.title("Drawdown (Gross)")
        plt.xlabel("Date"); plt.ylabel("Drawdown")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        pdf.savefig(); plt.close()

        # Rolling 12m return
        plt.figure(figsize=(10, 4))
        plt.plot(roll_df.index, roll_df["Model_roll12_ret"], label="Model")
        plt.plot(roll_df.index, roll_df["RawRank_roll12_ret"], label="RawRank")
        plt.plot(roll_df.index, roll_df["Iso_roll12_ret"], label="Iso")
        plt.title("Rolling 12-Month Total Return")
        plt.xlabel("Date"); plt.ylabel("12m Return")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        pdf.savefig(); plt.close()

        # Rolling 12m Sharpe
        plt.figure(figsize=(10, 4))
        plt.plot(roll_df.index, roll_df["Model_roll12_sh"], label="Model")
        plt.plot(roll_df.index, roll_df["RawRank_roll12_sh"], label="RawRank")
        plt.plot(roll_df.index, roll_df["Iso_roll12_sh"], label="Iso")
        plt.title("Rolling 12-Month Sharpe (rf=0)")
        plt.xlabel("Date"); plt.ylabel("Sharpe")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        pdf.savefig(); plt.close()

        # Monthly Turnover
        if not tovr_df.empty:
            plt.figure(figsize=(10, 4))
            if "Model" in tovr_df:   plt.plot(tovr_df.index, tovr_df["Model"], label="Model")
            if "RawRank" in tovr_df: plt.plot(tovr_df.index, tovr_df["RawRank"], label="RawRank")
            if "Iso" in tovr_df:     plt.plot(tovr_df.index, tovr_df["Iso"], label="Iso")
            plt.title("Monthly Turnover (fraction of names replaced)")
            plt.xlabel("Date"); plt.ylabel("Turnover")
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            pdf.savefig(); plt.close()

        # Excess vs baseline (gross)
        plt.figure(figsize=(10, 4))
        plt.plot(ex_df.index, ex_df["Model_ex"], label="Model - " + args.baseline)
        plt.plot(ex_df.index, ex_df["RawRank_ex"], label="RawRank - " + args.baseline)
        plt.plot(ex_df.index, ex_df["Iso_ex"], label="Iso - " + args.baseline)
        plt.title(f"Monthly Excess Return vs {args.baseline}")
        plt.xlabel("Date"); plt.ylabel("Excess Return")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        pdf.savefig(); plt.close()

        # Cumulative Relative (ExcessRel)
        plt.figure(figsize=(10, 4))
        plt.plot(exrel_df.index, exrel_df["Model_exrel"], label="Model / " + args.baseline)
        plt.plot(exrel_df.index, exrel_df["RawRank_exrel"], label="RawRank / " + args.baseline)
        plt.plot(exrel_df.index, exrel_df["Iso_exrel"], label="Iso / " + args.baseline)
        plt.title(f"Cumulative Relative Performance vs {args.baseline}")
        plt.xlabel("Date"); plt.ylabel("Relative Outperformance")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        pdf.savefig(); plt.close()

        # Model vs RawRank Score Correlation
        if rank_corr is not None:
            plt.figure(figsize=(10, 4))
            rank_corr.plot(label='Monthly Spearman Corr', alpha=0.7, color='teal')
            rank_corr.rolling(12).mean().plot(label='12m Rolling Avg', style='--', color='red')
            plt.title("Model vs. RawRank Score Correlation (Spearman)")
            plt.xlabel("Date"); plt.ylabel("Correlation")
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            pdf.savefig(); plt.close()

        # Optional net-of-turnover curves
        if args.switch_cost_bps > 0 and strat_model.equity_net is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(strat_model.equity_net.index, strat_model.equity_net, label="Model (net)")
            plt.plot(strat_raw.equity_net.index, strat_raw.equity_net, label="RawRank (net)")
            plt.plot(strat_iso.equity_net.index, strat_iso.equity_net, label="Iso (net)")
            plt.title(f"Cumulative Growth of $1 (Net, {args.switch_cost_bps:.0f} bps per switch)")
            plt.xlabel("Date"); plt.ylabel("Value")
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            pdf.savefig(); plt.close()

    print("Done.")
    print(f"Excel: {out_xlsx}")
    print(f"PDF:   {out_pdf}")


if __name__ == "__main__":
    main()