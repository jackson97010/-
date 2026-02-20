"""
Strategy signal generators.

All strategies generate a binary 'position' column (1 = long, 0 = flat)
on the merged (Taiwan calendar) DataFrame.

Architecture:
- Indicators are pre-computed on US trading days (see indicators.py).
- Merged DataFrame already has shifted + ffilled indicator values.
- Strategies only read indicator columns to decide position.

Strategy versions:
    V1: MA crossover (price > MA → long)
    V2: MA crossover with re-entry delay
    V3: MA crossover with trailing stop
    V4: MA crossover entry + rolling range-low exit
    V5: MA crossover + ROC confirmation entry (strict)
    V6: MA + ROC confirmation entry + rolling range-low exit (best combo)
"""

import pandas as pd
import numpy as np


def v1_ma_crossover(df: pd.DataFrame, ma_col: str) -> pd.Series:
    """
    V1: Simple MA crossover.
    Long when US_Close > MA, flat otherwise.
    """
    position = (df["US_Close"] > df[ma_col]).astype(int)
    return position


def v4_ma_entry_range_low_exit(
    df: pd.DataFrame,
    ma_col: str,
    rolling_low_col: str,
) -> pd.Series:
    """
    V4: MA crossover entry + rolling range-low exit.
    Entry: US_Close > MA
    Exit:  US_Close < Rolling Low (breaks below range support)

    This gives a wider exit band — you stay in until price truly breaks
    down below the recent range, tolerating normal pullbacks.
    """
    n = len(df)
    position = np.zeros(n, dtype=int)
    in_position = False

    close = df["US_Close"].values
    ma = df[ma_col].values
    rlow = df[rolling_low_col].values

    for i in range(n):
        if np.isnan(ma[i]) or np.isnan(rlow[i]):
            position[i] = 0
            continue

        if not in_position:
            # Entry: price above MA
            if close[i] > ma[i]:
                in_position = True
                position[i] = 1
            else:
                position[i] = 0
        else:
            # Exit: price drops below rolling low
            if close[i] < rlow[i]:
                in_position = False
                position[i] = 0
            else:
                position[i] = 1

    return pd.Series(position, index=df.index, name="position")


def v5_ma_roc_entry(
    df: pd.DataFrame,
    ma_col: str,
    roc_col: str,
) -> pd.Series:
    """
    V5: MA + ROC confirmation entry.
    Entry: US_Close > MA AND ROC > 0 (positive momentum)
    Exit:  US_Close < MA

    Stricter entry filter — requires both trend AND momentum.
    Can miss early-stage moves where MA just turned but momentum hasn't confirmed.
    """
    n = len(df)
    position = np.zeros(n, dtype=int)
    in_position = False

    close = df["US_Close"].values
    ma = df[ma_col].values
    roc = df[roc_col].values

    for i in range(n):
        if np.isnan(ma[i]) or np.isnan(roc[i]):
            position[i] = 0
            continue

        if not in_position:
            # Entry: price above MA AND positive momentum
            if close[i] > ma[i] and roc[i] > 0:
                in_position = True
                position[i] = 1
            else:
                position[i] = 0
        else:
            # Exit: price drops below MA
            if close[i] < ma[i]:
                in_position = False
                position[i] = 0
            else:
                position[i] = 1

    return pd.Series(position, index=df.index, name="position")


def v6_roc_entry_range_low_exit(
    df: pd.DataFrame,
    ma_col: str,
    roc_col: str,
    rolling_low_col: str,
) -> pd.Series:
    """
    V6: ROC-confirmed entry + range-low exit.
    Entry: US_Close > MA AND ROC > 0 (precise entry with momentum confirmation)
    Exit:  US_Close < Rolling Low (tolerant exit — stays in during pullbacks)

    Best combination: "precise entry + tolerant exit"
    - Short MA lets you enter early
    - Long ROC confirms the momentum is real
    - Range-low exit prevents getting shaken out by minor corrections
    """
    n = len(df)
    position = np.zeros(n, dtype=int)
    in_position = False

    close = df["US_Close"].values
    ma = df[ma_col].values
    roc = df[roc_col].values
    rlow = df[rolling_low_col].values

    for i in range(n):
        if np.isnan(ma[i]) or np.isnan(roc[i]) or np.isnan(rlow[i]):
            position[i] = 0
            continue

        if not in_position:
            # Entry: price above MA AND positive momentum
            if close[i] > ma[i] and roc[i] > 0:
                in_position = True
                position[i] = 1
            else:
                position[i] = 0
        else:
            # Exit: price drops below rolling low (range breakdown)
            if close[i] < rlow[i]:
                in_position = False
                position[i] = 0
            else:
                position[i] = 1

    return pd.Series(position, index=df.index, name="position")


# Registry for easy iteration
STRATEGIES = {
    "V1": {
        "func": v1_ma_crossover,
        "params": ["ma_col"],
        "desc": "MA crossover",
    },
    "V4": {
        "func": v4_ma_entry_range_low_exit,
        "params": ["ma_col", "rolling_low_col"],
        "desc": "MA entry + range-low exit",
    },
    "V5": {
        "func": v5_ma_roc_entry,
        "params": ["ma_col", "roc_col"],
        "desc": "MA + ROC entry",
    },
    "V6": {
        "func": v6_roc_entry_range_low_exit,
        "params": ["ma_col", "roc_col", "rolling_low_col"],
        "desc": "ROC entry + range-low exit",
    },
}
