"""
Main script: Parameter sweep with heatmaps for all strategy versions.

Usage:
    python main.py [--source NDX|SPY] [--start 2010-01-01] [--end 2025-12-31]

Architecture (correct order):
    1. Load US data (pure US trading days)
    2. Compute ALL indicators on US trading days  ← rolling calcs here
    3. Shift + merge to Taiwan calendar           ← ffill only copies results
    4. Generate signals & backtest on TW calendar
"""

import argparse
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backtest import backtest, compute_metrics, yearly_breakdown
from data_loader import load_tw_data, load_us_data, merge_us_tw
from indicators import compute_indicators
from strategies import (
    STRATEGIES,
    v1_ma_crossover,
    v4_ma_entry_range_low_exit,
    v5_ma_roc_entry,
    v6_roc_entry_range_low_exit,
)

# ── Configuration ───────────────────────────────────────────────────────────

US_TICKERS = {
    "NDX": "^NDX",
    "SPY": "SPY",
}
TW_TICKER = "00631L.TW"

MA_RANGE = range(5, 31)          # MA periods to sweep
ROC_RANGE = range(3, 21)         # ROC periods to sweep
ROLLING_LOW_PERIOD = 60          # Fixed for range-low exit


def run_single(
    us_data: pd.DataFrame,
    tw_data: pd.DataFrame,
    strategy_name: str,
    ma_period: int,
    roc_period: int = 12,
    rolling_low_period: int = ROLLING_LOW_PERIOD,
) -> dict:
    """
    Run a single backtest for given parameters.

    Key architecture: indicators computed on US days, then merged.
    """
    # Step 1: Compute indicators on pure US trading days
    us_ind = compute_indicators(
        us_data,
        ma_period=ma_period,
        rolling_low_period=rolling_low_period,
        roc_period=roc_period,
    )

    ma_col = f"MA{ma_period}"
    roc_col = f"ROC{roc_period}"
    rolling_low_col = f"RollingLow{rolling_low_period}"

    # Step 2: Shift + merge to Taiwan calendar
    merged = merge_us_tw(us_ind, tw_data, shift_days=1)

    if len(merged) < 100:
        return None

    # Step 3: Generate position signal
    strat = STRATEGIES[strategy_name]
    kwargs = {"df": merged, "ma_col": ma_col}
    if "roc_col" in strat["params"]:
        kwargs["roc_col"] = roc_col
    if "rolling_low_col" in strat["params"]:
        kwargs["rolling_low_col"] = rolling_low_col

    position = strat["func"](**kwargs)

    # Step 4: Backtest
    result = backtest(merged, position)
    metrics = compute_metrics(result)
    metrics["ma_period"] = ma_period
    metrics["roc_period"] = roc_period
    metrics["strategy"] = strategy_name

    return metrics


def sweep_ma_only(
    us_data: pd.DataFrame,
    tw_data: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    """Sweep MA period for non-ROC strategies (V1, V4)."""
    results = []
    for ma in MA_RANGE:
        m = run_single(us_data, tw_data, strategy_name, ma_period=ma)
        if m:
            results.append(m)
    return pd.DataFrame(results)


def sweep_ma_roc(
    us_data: pd.DataFrame,
    tw_data: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    """Sweep MA × ROC grid for momentum strategies (V5, V6)."""
    results = []
    for ma, roc in itertools.product(MA_RANGE, ROC_RANGE):
        m = run_single(us_data, tw_data, strategy_name, ma_period=ma, roc_period=roc)
        if m:
            results.append(m)
    return pd.DataFrame(results)


def plot_heatmap(
    sweep_df: pd.DataFrame,
    title: str,
    metric: str = "sharpe",
    filename: str = None,
):
    """Plot MA × ROC Sharpe heatmap."""
    if "roc_period" not in sweep_df.columns or sweep_df["roc_period"].nunique() <= 1:
        # 1D sweep — bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(sweep_df["ma_period"], sweep_df[metric])
        ax.set_xlabel("MA Period")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        best = sweep_df.loc[sweep_df[metric].idxmax()]
        ax.axvline(best["ma_period"], color="red", linestyle="--", alpha=0.7)
    else:
        # 2D sweep — heatmap
        pivot = sweep_df.pivot_table(
            index="roc_period", columns="ma_period", values=metric
        )
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=pivot.values.mean(),
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("MA Period")
        ax.set_ylabel("ROC Period")

    plt.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"  Saved: {filename}")
    plt.close(fig)


def print_summary_table(all_results: list[dict], source: str):
    """Print a formatted comparison table."""
    df = pd.DataFrame(all_results)
    df = df.sort_values("sharpe", ascending=False)

    print(f"\n{'='*80}")
    print(f"  {source} Strategy Comparison (sorted by Sharpe)")
    print(f"{'='*80}")
    print(
        f"  {'Strategy':<25} {'Sharpe':>7} {'Annual%':>8} {'MDD%':>7} "
        f"{'Calmar':>7} {'Exp%':>5} {'Trades':>7}"
    )
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*5} {'-'*7}")

    for _, row in df.iterrows():
        label = f"{row['strategy']} MA{int(row['ma_period'])}"
        if row.get("roc_period", 0) > 0 and row["strategy"] in ("V5", "V6"):
            label += f" ROC{int(row['roc_period'])}"
        print(
            f"  {label:<25} {row['sharpe']:>7.2f} {row['annual_ret']:>7.1f}% "
            f"{row['mdd']:>6.1f}% {row['calmar']:>7.2f} {row['exposure']:>4.0f}% "
            f"{row['n_trades']:>6d}"
        )

    # Buy & Hold baseline (from any row, they're all the same)
    bh = df.iloc[0]
    print(
        f"  {'B&H 00631L':<25} {bh['bh_sharpe']:>7.2f} {bh['bh_annual_ret']:>7.1f}% "
        f"{bh['bh_mdd']:>6.1f}%  {'---':>6} {'100':>4}%     {'0':>4}"
    )
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="00631L Timing Strategy Backtester"
    )
    parser.add_argument(
        "--source", default="NDX", choices=["NDX", "SPY"],
        help="US signal source (default: NDX)",
    )
    parser.add_argument("--start", default="2010-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2025-12-31", help="Backtest end date")
    parser.add_argument(
        "--strategies", nargs="+", default=["V1", "V4", "V5", "V6"],
        help="Strategies to run (default: all)",
    )
    parser.add_argument(
        "--best-only", action="store_true",
        help="Only show the best parameter set per strategy",
    )
    args = parser.parse_args()

    source = args.source
    ticker = US_TICKERS[source]

    print(f"Loading data: {source} ({ticker}) + 00631L.TW ...")
    us_data = load_us_data(ticker, args.start, args.end)
    tw_data = load_tw_data(TW_TICKER, args.start, args.end)
    print(f"  US data: {len(us_data)} trading days")
    print(f"  TW data: {len(tw_data)} trading days")

    best_per_strategy = []

    for strat_name in args.strategies:
        print(f"\nRunning {strat_name} ({STRATEGIES[strat_name]['desc']}) ...")

        if strat_name in ("V1", "V4"):
            sweep_df = sweep_ma_only(us_data, tw_data, strat_name)
        else:
            sweep_df = sweep_ma_roc(us_data, tw_data, strat_name)

        if sweep_df.empty:
            print(f"  No valid results for {strat_name}")
            continue

        # Best parameters
        best_idx = sweep_df["sharpe"].idxmax()
        best = sweep_df.loc[best_idx].to_dict()
        best_per_strategy.append(best)

        label = f"{source} {strat_name} MA{int(best['ma_period'])}"
        if strat_name in ("V5", "V6"):
            label += f" ROC{int(best['roc_period'])}"
        print(f"  Best: {label} → Sharpe={best['sharpe']:.2f}, "
              f"Annual={best['annual_ret']:.1f}%, MDD={best['mdd']:.1f}%")

        # Plot heatmap
        plot_heatmap(
            sweep_df,
            title=f"{source} {strat_name} Sharpe Heatmap",
            filename=f"heatmap_{source}_{strat_name}.png",
        )

    # Summary
    print_summary_table(best_per_strategy, source)

    # Yearly breakdown for the overall best strategy
    if best_per_strategy:
        overall_best = max(best_per_strategy, key=lambda x: x["sharpe"])
        strat_name = overall_best["strategy"]
        ma = int(overall_best["ma_period"])
        roc = int(overall_best.get("roc_period", 12))

        us_ind = compute_indicators(us_data, ma_period=ma, roc_period=roc)
        merged = merge_us_tw(us_ind, tw_data)

        strat = STRATEGIES[strat_name]
        kwargs = {"df": merged, "ma_col": f"MA{ma}"}
        if "roc_col" in strat["params"]:
            kwargs["roc_col"] = f"ROC{roc}"
        if "rolling_low_col" in strat["params"]:
            kwargs["rolling_low_col"] = f"RollingLow{ROLLING_LOW_PERIOD}"

        position = strat["func"](**kwargs)
        result = backtest(merged, position)
        yearly = yearly_breakdown(result)

        label = f"{source} {strat_name} MA{ma}"
        if strat_name in ("V5", "V6"):
            label += f" ROC{roc}"

        print(f"\nYearly Breakdown: {label}")
        print(f"{'Year':<6} {'Strategy%':>10} {'B&H%':>8} {'Excess%':>9}")
        print(f"{'-'*6} {'-'*10} {'-'*8} {'-'*9}")
        for _, row in yearly.iterrows():
            print(
                f"{int(row['year']):<6} {row['strat_ret']:>9.1f}% "
                f"{row['bh_ret']:>7.1f}% {row['excess']:>8.1f}%"
            )


if __name__ == "__main__":
    main()
