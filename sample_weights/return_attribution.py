"""
Return Attribution Sample Weights (AFML Snippet 4.10)
Derives sample weight by absolute return attribution divided by concurrent events.
"""
import numpy as np
import pandas as pd


def mpSampleW(t1, numCoEvents, close, molecule):
    """
    Derive sample weight by absolute return attribution.
    - t1: pd.Series mapping event start to event end
    - numCoEvents: pd.Series of event concurrency per bar
    - close: pd.Series of underlying asset close prices
    - molecule: subset of event start timestamps
    """
    ret = np.log(close).diff()
    wght = pd.Series(index=molecule, dtype=float)
    for tIn, tOut in t1.loc[wght.index].items():
        sub_ret = ret.loc[tIn:tOut]
        sub_c = numCoEvents.loc[tIn:tOut]
        if len(sub_c) > 0 and (sub_c > 0).any():
            wght.loc[tIn] = (sub_ret / sub_c).sum()
        else:
            wght.loc[tIn] = 0.0
    return wght.abs()
