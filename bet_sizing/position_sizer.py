"""
Bet Sizing and Signal Averaging Module (AFML Chapter 10)
Translates model predicted probabilities into continuous bet sizes [-1, 1],
averages concurrently active positions, and applies discretization to prevent trade jitter.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sample_weights.mp_engine import mpPandasObj


def mpAvgActiveSignals(signals, molecule):
    """
    Computes average active signals across time molecule.
    """
    out = pd.Series(dtype=float)
    for loc in molecule:
        df0 = (signals.index.values <= loc) & (
            (loc < signals["t1"]) | pd.isnull(signals["t1"])
        )
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, "signal"].mean()
        else:
            out[loc] = 0.0
    return out


def avgActiveSignals(signals, numThreads=1):
    """
    Derives average active concurrent signals (Snippets 10.2 - 10.3).
    """
    tPnts = set(signals["t1"].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = sorted(list(tPnts))
    out = mpPandasObj(
        mpAvgActiveSignals,
        ("molecule", tPnts),
        numThreads=numThreads,
        signals=signals,
    )
    return out


def discreteSignal(signal0, stepSize=0.05):
    """
    Discretizes continuous signal to avoid trade jitter (Snippet 10.4).
    """
    signal1 = (signal0 / stepSize).round() * stepSize
    signal1 = signal1.clip(lower=-1.0, upper=1.0)
    return signal1


def getSignal(events, stepSize=0.05, prob=None, pred=None, numClasses=2, numThreads=1):
    """
    Translates predicted probabilities into continuous bet sizes [-1, +1] (Snippet 10.1).
    - events: DataFrame containing 't1' (end timestamp) and optionally 'side' (+1 / -1).
    - prob: pd.Series of predicted probabilities for positive class (meta-label == 1).
    - pred: pd.Series of binary predictions (1 = take trade, 0 = skip).
    """
    if prob is None or len(prob) == 0:
        return pd.Series(dtype=float)

    # 1. Calculate t-value of One-vs-Rest (OvR)
    # Clip probability slightly away from 0 and 1 to prevent division by zero in z-score
    p = prob.clip(lower=1e-5, upper=1.0 - 1e-5)
    z_stat = (p - 1.0 / numClasses) / np.sqrt(p * (1.0 - p))
    bet_size = pd.Series(2.0 * norm.cdf(z_stat) - 1.0, index=prob.index)

    if pred is not None:
        bet_size = bet_size * pred

    if "side" in events.columns:
        bet_size = bet_size * events.loc[bet_size.index, "side"]

    # 2. Average active signals concurrently open
    df0 = bet_size.to_frame("signal")
    if "t1" in events.columns:
        df0 = df0.join(events[["t1"]], how="left")
        df0 = avgActiveSignals(df0, numThreads=numThreads)
    else:
        df0 = df0["signal"]

    # 3. Discretize
    signal_out = discreteSignal(signal0=df0, stepSize=stepSize)
    return signal_out
