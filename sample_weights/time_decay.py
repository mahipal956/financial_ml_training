"""
Piecewise-Linear Time Decay Module (AFML Snippet 4.11)
Applies piecewise-linear decay factors to observed sample weights.
"""
import pandas as pd


def getTimeDecay(tW, clfLastW=1.0):
    """
    Apply piecewise-linear decay to observed uniqueness/sample weights (tW).
    - clfLastW = 1.0 implies no time decay (all samples weighted equally by uniqueness).
    - 0 <= clfLastW < 1.0: newest observation gets weight=1.0, oldest observation gets weight=clfLastW.
    - clfLastW < 0: decays faster than linear.
    """
    clfW = tW.sort_index().cumsum()
    if clfW.iloc[-1] == 0:
        return pd.Series(1.0, index=tW.index)

    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1.0) * clfW.iloc[-1])

    const = 1.0 - slope * clfW.iloc[-1]
    decayedW = const + slope * clfW
    decayedW[decayedW < 0] = 0.0
    return decayedW
