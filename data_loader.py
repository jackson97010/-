"""
Data loader for US (NDX/SPY) and Taiwan (00631L) market data.
Uses yfinance to fetch historical prices.
"""

import pandas as pd
import yfinance as yf


def load_us_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load US market daily data (e.g., ^NDX, SPY).
    Returns DataFrame with columns: [Close] indexed by date.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[["Close"]].copy()
    df.columns = ["US_Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def load_tw_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load Taiwan market daily data (e.g., 00631L.TW).
    Returns DataFrame with columns: [Close] indexed by date.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[["Close"]].copy()
    df.columns = ["TW_Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def merge_us_tw(
    us_with_indicators: pd.DataFrame,
    tw: pd.DataFrame,
    shift_days: int = 1,
) -> pd.DataFrame:
    """
    Merge US indicators into Taiwan trading calendar.

    Critical architecture point:
    - All indicators (MA, Rolling Low, ROC) are already computed on
      pure US trading days BEFORE this function is called.
    - We shift US data by `shift_days` to account for the time difference
      (US market closes after Taiwan opens next day).
    - ffill only propagates already-computed indicator values to Taiwan
      trading days that fall on US holidays. This does NOT dilute
      rolling calculations.

    Parameters
    ----------
    us_with_indicators : DataFrame
        US data with all indicators pre-computed on US trading days.
    tw : DataFrame
        Taiwan market data indexed by date.
    shift_days : int
        Number of US trading days to shift forward (default=1).
        This ensures we only use information available before Taiwan opens.

    Returns
    -------
    DataFrame
        Merged data on Taiwan trading calendar with US indicators.
    """
    # Shift US indicators forward by N trading days to avoid look-ahead bias
    us_shifted = us_with_indicators.shift(shift_days)

    # Reindex to Taiwan calendar and forward-fill
    # ffill here only copies already-computed indicator values
    merged = tw.join(us_shifted.reindex(tw.index, method="ffill"), how="left")
    merged.dropna(subset=["US_Close"], inplace=True)

    return merged
