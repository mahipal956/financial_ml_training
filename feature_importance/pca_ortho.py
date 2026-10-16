"""
Orthogonal Features (PCA) and Rank Correlation (AFML Snippets 8.5 - 8.6)
Generates unsupervised orthogonal feature sets and calculates rank concordance via weighted Kendall's Tau.
"""
import numpy as np
import pandas as pd
from scipy.stats import weightedtau


def orthoFeats(dfX, varThres=0.95):
    """
    Compute standardized orthogonal features dfP explaining >= varThres variance.
    """
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1).fillna(0.0)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)

    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]
    eVal, eVec = eVal[idx], eVec[:, idx]

    eVal = pd.Series(eVal, index=["PC_" + str(i + 1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)

    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[: dim + 1], eVec.iloc[:, : dim + 1]

    dfP = np.dot(dfZ, eVec)
    return pd.DataFrame(dfP, index=dfX.index, columns=eVal.index), eVal, eVec


def check_pca_rank_correl(featImp_array, pcRank_array):
    """
    Check rank correlation between supervised feature importance and inverse PCA rank.
    Inverts pcRank (pcRank**-1.0) so top-ranked elements receive the highest weights.
    """
    featImp_arr = np.array(featImp_array, dtype=float)
    pcRank_arr = np.array(pcRank_array, dtype=float)
    tau, p_value = weightedtau(featImp_arr, pcRank_arr ** -1.0)
    return tau, p_value
