"""
Backtesting engine for 00631L timing strategies.

Assumes:
- Trading 00631L (Taiwan-listed 2x leveraged NASDAQ ETF).
- Position sizing: 100% in or 100% out.
- Signals are generated on T, executed at T+1 close (next Taiwan trading day).
- No transaction costs modeled (can be added later).
"""

import pandas as pd
import numpy as np


def backtest(
    merged: pd.DataFrame,
    position: pd.Series,
    leverage_etf_col: str = "TW_Close",
) -> pd.DataFrame:
    """
    Run backtest given merged data and position signals.

    Parameters
    ----------
    merged : DataFrame
        Merged US+TW data on Taiwan trading calendar.
    position : Series
        Binary position signal (1=long, 0=flat), same index as merged.
    leverage_etf_col : str
        Column name for the Taiwan ETF price.

    Returns
    -------
    DataFrame with columns: [TW_Close, position, tw_ret, strat_ret, cum_ret, bh_cum_ret]
    """
    result = merged[[leverage_etf_col]].copy()
    result["position"] = position

    # Daily return of the leveraged ETF
    result["tw_ret"] = result[leverage_etf_col].pct_change()

    # Strategy return: use previous day's position signal (execution delay)
    result["strat_ret"] = result["position"].shift(1) * result["tw_ret"]

    # Cumulative returns
    result["cum_ret"] = (1 + result["strat_ret"]).cumprod()
    result["bh_cum_ret"] = (1 + result["tw_ret"]).cumprod()

    result.dropna(subset=["strat_ret"], inplace=True)

    return result


def compute_metrics(result: pd.DataFrame, annual_trading_days: int = 245) -> dict:
    """
    Compute performance metrics from backtest results.

    Parameters
    ----------
    result : DataFrame
        Output from backtest().
    annual_trading_days : int
        Number of trading days per year (Taiwan market ~245).

    Returns
    -------
    dict with keys: sharpe, annual_ret, mdd, calmar, exposure, n_trades,
                    bh_annual_ret, bh_mdd, bh_sharpe
    """
    sr = result["strat_ret"]
    cr = result["cum_ret"]
    bh_cr = result["bh_cum_ret"]
    bh_sr = result["tw_ret"]

    n_years = len(sr) / annual_trading_days

    # --- Strategy metrics ---
    annual_ret = cr.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = sr.std() * np.sqrt(annual_trading_days)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    running_max = cr.cummax()
    drawdown = cr / running_max - 1
    mdd = drawdown.min()

    calmar = annual_ret / abs(mdd) if mdd != 0 else 0

    # Exposure: fraction of days in the market
    pos = result["position"].shift(1).dropna()
    exposure = pos.mean()

    # Number of trades (entries)
    position_changes = result["position"].diff().fillna(0)
    n_trades = int((position_changes == 1).sum())

    # --- Buy & Hold metrics ---
    bh_annual_ret = bh_cr.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    bh_annual_vol = bh_sr.std() * np.sqrt(annual_trading_days)
    bh_sharpe = bh_annual_ret / bh_annual_vol if bh_annual_vol > 0 else 0
    bh_running_max = bh_cr.cummax()
    bh_drawdown = bh_cr / bh_running_max - 1
    bh_mdd = bh_drawdown.min()

    return {
        "sharpe": round(sharpe, 2),
        "annual_ret": round(annual_ret * 100, 1),
        "mdd": round(mdd * 100, 1),
        "calmar": round(calmar, 2),
        "exposure": round(exposure * 100, 0),
        "n_trades": n_trades,
        "bh_sharpe": round(bh_sharpe, 2),
        "bh_annual_ret": round(bh_annual_ret * 100, 1),
        "bh_mdd": round(bh_mdd * 100, 1),
    }


def yearly_breakdown(result: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual performance breakdown for stability analysis.

    Returns DataFrame with one row per year: [year, strat_ret, bh_ret, excess].
    """
    result = result.copy()
    result["year"] = result.index.year

    rows = []
    for year, grp in result.groupby("year"):
        strat = (1 + grp["strat_ret"]).prod() - 1
        bh = (1 + grp["tw_ret"]).prod() - 1
        rows.append(
            {
                "year": year,
                "strat_ret": round(strat * 100, 1),
                "bh_ret": round(bh * 100, 1),
                "excess": round((strat - bh) * 100, 1),
            }
        )

    return pd.DataFrame(rows)
