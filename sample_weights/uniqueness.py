"""
Average Uniqueness and Sequential Bootstrapping Module (AFML Snippets 4.2 - 4.5)
Derives sample uniqueness over event lifespans and builds IID-approximated samples.
"""
import numpy as np
import pandas as pd


def mpSampleTW(t1, numCoEvents, molecule):
    """
    Derive average uniqueness over the event's lifespan (Snippet 4.2)
    - t1: pd.Series mapping event start to event end
    - numCoEvents: pd.Series of event concurrency per bar
    - molecule: subset of event start timestamps
    """
    wght = pd.Series(index=molecule, dtype=float)
    for tIn, tOut in t1.loc[wght.index].items():
        sub_c = numCoEvents.loc[tIn:tOut]
        if len(sub_c) > 0 and (sub_c > 0).any():
            wght.loc[tIn] = (1.0 / sub_c).mean()
        else:
            wght.loc[tIn] = 1.0
    return wght


def getIndMatrix(barIx, t1):
    """
    Get binary indicator matrix of label spans (Snippet 4.3)
    Rows: price bar timestamps
    Columns: integer event index
    """
    indM = pd.DataFrame(0.0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1_) in enumerate(t1.items()):
        indM.loc[t0:t1_, i] = 1.0
    return indM


def getAvgUniqueness(indM):
    """
    Compute average uniqueness from indicator matrix (Snippet 4.4)
    """
    c = indM.sum(axis=1)  # concurrency
    u = indM.div(c, axis=0)  # uniqueness
    avgU = u[u > 0].mean()  # average uniqueness across non-zero elements
    return avgU


def seqBootstrap(indM, sLength=None):
    """
    Generate an IID-approximated sample via sequential bootstrap (Snippet 4.5)
    indM: binary indicator matrix
    sLength: sample length to draw (defaults to number of columns/events)
    """
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series(dtype=float)
        for i in indM.columns:
            indM_ = indM[phi + [i]]
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()
        chosen = np.random.choice(indM.columns, p=prob.values)
        phi.append(chosen)
    return phi
