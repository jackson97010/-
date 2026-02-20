"""
Technical indicators computed on pure US trading days.

All rolling calculations happen here, on the US-only calendar,
so ffill during merge will never dilute the window.
"""

import pandas as pd


def compute_indicators(
    us: pd.DataFrame,
    ma_period: int = 20,
    rolling_low_period: int = 60,
    roc_period: int = 12,
) -> pd.DataFrame:
    """
    Compute all indicators on US trading days.

    Parameters
    ----------
    us : DataFrame
        US market data with 'US_Close' column, indexed by US trading dates.
    ma_period : int
        Moving average lookback period (in US trading days).
    rolling_low_period : int
        Rolling low lookback period for range-low exit.
    roc_period : int
        Rate of Change lookback period for momentum confirmation.

    Returns
    -------
    DataFrame
        Original data plus computed indicator columns.
    """
    df = us.copy()

    # Simple Moving Average
    df[f"MA{ma_period}"] = df["US_Close"].rolling(ma_period).mean()

    # Rolling Low for range-low exit strategy
    df[f"RollingLow{rolling_low_period}"] = (
        df["US_Close"].rolling(rolling_low_period).min()
    )

    # Rate of Change (ROC) = (Close / Close_n_days_ago - 1) * 100
    df[f"ROC{roc_period}"] = (
        df["US_Close"] / df["US_Close"].shift(roc_period) - 1
    ) * 100

    # Store parameter names for downstream reference
    df.attrs["ma_col"] = f"MA{ma_period}"
    df.attrs["rolling_low_col"] = f"RollingLow{rolling_low_period}"
    df.attrs["roc_col"] = f"ROC{roc_period}"

    return df
