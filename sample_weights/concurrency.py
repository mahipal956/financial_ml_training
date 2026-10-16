"""
Concurrency Estimation Module (AFML Snippet 4.1)
Computes the number of concurrent active events per bar.
"""
import numpy as np
import pandas as pd


def mpNumCoEvents(closeIdx, t1, molecule):
    """
    Compute the number of concurrent events per bar.
    - closeIdx: DatetimeIndex of the original price bars.
    - t1: pd.Series mapping event start time (index) to event end time (values).
    - molecule: Array or Index of event start timestamps for which weights are computed.
    """
    # Fill NA end times with the last known close index
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]

    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0.0, index=closeIdx[iloc[0]:iloc[1] + 1])

    for tIn, tOut in t1.items():
        count.loc[tIn:tOut] += 1.0

    return count.loc[molecule[0]:t1[molecule].max()]
