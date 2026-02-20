"""
Backtest runner: loads data (or generates synthetic data if yfinance
is unavailable), computes indicators, runs all strategies, and generates
PNG charts with performance summary.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from indicators import compute_indicators
from strategies import STRATEGIES

# ── Configuration ──────────────────────────────────────────────────────
US_TICKER = "^NDX"
TW_TICKER = "00631L.TW"
START = "2015-01-01"
END = "2026-02-01"
MA_PERIOD = 20
ROLLING_LOW_PERIOD = 60
ROC_PERIOD = 12
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def _try_load_yfinance():
    """Try to load real data via yfinance. Return (us, tw) or None."""
    try:
        from data_loader import load_us_data, load_tw_data, merge_us_tw
        us = load_us_data(US_TICKER, START, END)
        tw = load_tw_data(TW_TICKER, START, END)
        if len(us) == 0 or len(tw) == 0:
            return None
        return us, tw, merge_us_tw
    except Exception:
        return None


def _generate_synthetic_data():
    """
    Generate realistic synthetic US & TW market data when yfinance
    is unavailable. Mimics NDX and 00631L behaviour with:
    - Upward drift, volatility clusters, drawdowns
    - Different trading calendars (US vs TW)
    """
    rng = np.random.default_rng(42)

    # US trading days (Mon-Fri, no holidays simplified)
    us_dates = pd.bdate_range(START, END, freq="B")
    n_us = len(us_dates)

    # Generate US price (NDX-like: start ~4200, end ~20000+)
    drift = 0.0004  # daily drift
    vol = 0.012     # daily vol
    shocks = rng.normal(drift, vol, n_us)

    # Add regime changes (drawdowns in 2018, 2020, 2022)
    for year, severity, duration in [(2018, -0.0015, 60), (2020, -0.004, 30), (2022, -0.001, 200)]:
        yr_start = np.searchsorted(us_dates, pd.Timestamp(f"{year}-01-01"))
        shocks[yr_start:yr_start + duration] -= severity

    us_price = 4200 * np.exp(np.cumsum(shocks))
    us = pd.DataFrame({"US_Close": us_price}, index=us_dates)
    us.index.name = "Date"

    # TW trading days (slightly different calendar)
    tw_dates = pd.bdate_range(START, END, freq="B")
    # Remove some days to simulate TW-specific holidays
    tw_holiday_idx = rng.choice(len(tw_dates), size=50, replace=False)
    tw_dates = tw_dates.delete(tw_holiday_idx)
    n_tw = len(tw_dates)

    # TW price (00631L-like: leveraged, start ~30, correlated with US)
    tw_drift = 0.0003
    tw_vol = 0.018  # higher vol (leveraged)
    tw_shocks = rng.normal(tw_drift, tw_vol, n_tw)
    # Correlate with US
    us_resampled = np.interp(
        np.arange(n_tw), np.linspace(0, n_tw - 1, n_us), shocks
    )
    tw_shocks = 0.6 * tw_shocks + 0.4 * us_resampled * 1.5
    tw_price = 30 * np.exp(np.cumsum(tw_shocks))
    tw = pd.DataFrame({"TW_Close": tw_price}, index=tw_dates)
    tw.index.name = "Date"

    return us, tw


def merge_synthetic(us_with_indicators, tw, shift_days=1):
    """Merge logic identical to data_loader.merge_us_tw."""
    us_shifted = us_with_indicators.shift(shift_days)
    merged = tw.join(us_shifted.reindex(tw.index, method="ffill"), how="left")
    merged.dropna(subset=["US_Close"], inplace=True)
    return merged


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    real_data = _try_load_yfinance()
    if real_data:
        us, tw, merge_fn = real_data
        data_source = "Yahoo Finance (live)"
    else:
        print("yfinance unavailable — using synthetic market data.")
        us, tw = _generate_synthetic_data()
        merge_fn = merge_synthetic
        data_source = "Synthetic (simulated)"

    print(f"  Data source: {data_source}")
    print(f"  US data: {len(us)} trading days  ({us.index[0].date()} ~ {us.index[-1].date()})")
    print(f"  TW data: {len(tw)} trading days  ({tw.index[0].date()} ~ {tw.index[-1].date()})")

    # ── 2. Compute indicators on US calendar ──────────────────────────
    print("Computing indicators...")
    us_ind = compute_indicators(us, MA_PERIOD, ROLLING_LOW_PERIOD, ROC_PERIOD)
    ma_col = us_ind.attrs["ma_col"]
    rlow_col = us_ind.attrs["rolling_low_col"]
    roc_col = us_ind.attrs["roc_col"]

    # ── 3. Merge to Taiwan calendar ───────────────────────────────────
    print("Merging to Taiwan calendar...")
    merged = merge_fn(us_ind, tw, shift_days=1)
    print(f"  Merged: {len(merged)} rows")

    # ── 4. Run each strategy ──────────────────────────────────────────
    param_map = {
        "ma_col": ma_col,
        "roc_col": roc_col,
        "rolling_low_col": rlow_col,
    }

    results = {}
    for name, meta in STRATEGIES.items():
        kwargs = {p: param_map[p] for p in meta["params"]}
        pos = meta["func"](merged, **kwargs)
        results[name] = pos
        print(f"  {name} ({meta['desc']}): {int(pos.sum())} days long / {len(pos)} total")

    # ── 5. Compute performance ────────────────────────────────────────
    tw_ret = merged["TW_Close"].pct_change()
    perf = {}
    for name, pos in results.items():
        # Strategy return: position lagged by 1 (trade next day)
        strat_ret = tw_ret * pos.shift(1)
        cum = (1 + strat_ret).cumprod()
        total_return = cum.iloc[-1] - 1
        years = (merged.index[-1] - merged.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1

        # Max drawdown
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = drawdown.min()

        # Count trades (position changes)
        trades = (pos.diff().abs() > 0).sum()

        # Win rate (days with positive return when in position)
        in_market = strat_ret[pos.shift(1) == 1]
        win_rate = (in_market > 0).sum() / len(in_market) * 100 if len(in_market) > 0 else 0

        # Time in market
        time_in = pos.sum() / len(pos) * 100

        perf[name] = {
            "cum": cum,
            "strat_ret": strat_ret,
            "drawdown": drawdown,
            "total_return": total_return * 100,
            "cagr": cagr * 100,
            "max_dd": max_dd * 100,
            "trades": trades,
            "win_rate": win_rate,
            "time_in_market": time_in,
        }

    # Buy-and-hold benchmark
    bh_cum = (1 + tw_ret).cumprod()
    bh_total = bh_cum.iloc[-1] - 1
    bh_years = (merged.index[-1] - merged.index[0]).days / 365.25
    bh_cagr = (1 + bh_total) ** (1 / bh_years) - 1
    bh_running_max = bh_cum.cummax()
    bh_dd = (bh_cum - bh_running_max) / bh_running_max
    bh_max_dd = bh_dd.min()

    # ── 6. Generate charts ────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)
    colors = {"V1": "#1f77b4", "V4": "#ff7f0e", "V5": "#2ca02c", "V6": "#d62728"}

    # ── Chart 1: Cumulative Returns Comparison ────────────────────────
    fig, ax = plt.subplots(figsize=(16, 8))
    for name, p in perf.items():
        ax.plot(p["cum"].index, p["cum"].values,
                label=f'{name}: {STRATEGIES[name]["desc"]}',
                color=colors[name], linewidth=1.5)
    ax.plot(bh_cum.index, bh_cum.values, label="Buy & Hold", color="gray",
            linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_title(f"Cumulative Returns — {TW_TICKER} with {US_TICKER} Indicators\n"
                 f"({START} ~ {END})  [{data_source}]",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "01_cumulative_returns.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"\n  Saved {path1}")

    # ── Chart 2: Drawdown Comparison ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    for name, p in perf.items():
        ax.fill_between(p["drawdown"].index, p["drawdown"].values * 100,
                        alpha=0.15, color=colors[name])
        ax.plot(p["drawdown"].index, p["drawdown"].values * 100,
                label=f'{name}', color=colors[name], linewidth=1)
    ax.plot(bh_dd.index, bh_dd.values * 100,
            label="Buy & Hold", color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("Drawdown Comparison (%)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Drawdown %")
    ax.legend(loc="lower left", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "02_drawdown.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved {path2}")

    # ── Chart 3: Performance Summary Table ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    col_labels = ["Strategy", "Total Return %", "CAGR %", "Max Drawdown %",
                  "Trades", "Win Rate %", "Time in Market %"]
    table_data = []
    for name, p in perf.items():
        table_data.append([
            f'{name}: {STRATEGIES[name]["desc"]}',
            f'{p["total_return"]:.1f}%',
            f'{p["cagr"]:.2f}%',
            f'{p["max_dd"]:.1f}%',
            f'{p["trades"]}',
            f'{p["win_rate"]:.1f}%',
            f'{p["time_in_market"]:.1f}%',
        ])
    # Add Buy & Hold
    table_data.append([
        "Buy & Hold",
        f'{bh_total * 100:.1f}%',
        f'{bh_cagr * 100:.2f}%',
        f'{bh_max_dd * 100:.1f}%',
        "1",
        "-",
        "100.0%",
    ])

    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight V6 row
    for i, name in enumerate(list(STRATEGIES.keys()) + ["BH"]):
        if name == "V6":
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor("#d5f5e3")

    ax.set_title(f"Strategy Performance Summary  [{data_source}]",
                 fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "03_performance_table.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path3}")

    # ── Chart 4: Position signals for each strategy ───────────────────
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    for idx, (name, pos) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot(merged.index, merged["TW_Close"].values,
                color="black", linewidth=0.8, alpha=0.7)
        # Shade long periods
        long_mask = pos.values == 1
        ax.fill_between(merged.index, merged["TW_Close"].min(), merged["TW_Close"].max(),
                        where=long_mask, alpha=0.2, color=colors[name],
                        label="Long")
        ax.set_ylabel("TW_Close")
        ax.set_title(f'{name}: {STRATEGIES[name]["desc"]}  '
                     f'(CAGR={perf[name]["cagr"]:.2f}%, MDD={perf[name]["max_dd"]:.1f}%)',
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper left")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    fig.suptitle(f"Position Signals — {TW_TICKER}  [{data_source}]",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path4 = os.path.join(OUTPUT_DIR, "04_position_signals.png")
    fig.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path4}")

    # ── Chart 5: Annual Returns Heatmap ───────────────────────────────
    annual = {}
    for name, p in perf.items():
        yearly = p["strat_ret"].groupby(p["strat_ret"].index.year).apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        annual[name] = yearly
    annual["Buy&Hold"] = tw_ret.groupby(tw_ret.index.year).apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    annual_df = pd.DataFrame(annual).T

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(annual_df, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Return %"})
    ax.set_title(f"Annual Returns by Strategy (%)  [{data_source}]",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    path5 = os.path.join(OUTPUT_DIR, "05_annual_returns_heatmap.png")
    fig.savefig(path5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path5}")

    # ── Print summary to console ──────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  STRATEGY PERFORMANCE SUMMARY  [{data_source}]")
    print("=" * 72)
    print(f"  {'Strategy':<30} {'Total':>10} {'CAGR':>8} {'MaxDD':>8} {'Trades':>8} {'WinRate':>8}")
    print("  " + "-" * 70)
    for name, p in perf.items():
        print(f'  {name}: {STRATEGIES[name]["desc"]:<24} '
              f'{p["total_return"]:>9.1f}% {p["cagr"]:>7.2f}% '
              f'{p["max_dd"]:>7.1f}% {p["trades"]:>7} {p["win_rate"]:>7.1f}%')
    print(f'  {"Buy & Hold":<30} {bh_total*100:>9.1f}% {bh_cagr*100:>7.2f}% '
          f'{bh_max_dd*100:>7.1f}%       1       -')
    print("=" * 72)
    print(f"\n  All charts saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
