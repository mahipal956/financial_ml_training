"""
Financial Cross-Validation: PurgedKFold with Embargoing (AFML Chapter 7)
Strictly prevents information leakage across overlapping label intervals and serial correlation.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


def getTrainTimes(t1, testTimes):
    """
    Given testTimes, find the times of the training observations (Snippet 7.1).
    - t1.index: Event start time.
    - t1.values: Event end time.
    - testTimes: pd.Series of testing observation start and end times.
    """
    trn = t1.copy(deep=True)
    for i, j in testTimes.items():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index  # Train starts within test
        df1 = trn[(i <= trn) & (trn <= j)].index              # Train ends within test
        df2 = trn[(trn.index <= i) & (j <= trn)].index        # Train envelops test
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def getEmbargoTimes(times, pctEmbargo):
    """
    Derive post-test embargo window (Snippet 7.2).
    - times: pd.DatetimeIndex of observations.
    - pctEmbargo: float fraction (e.g., 0.01 = 1%).
    """
    step = int(times.shape[0] * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = pd.concat([mbrg, pd.Series(times[-1], index=times[-step:])])
    return mbrg


class PurgedKFold(BaseCrossValidator):
    """
    PurgedKFold cross-validation splitter (Snippet 7.3).
    Extends BaseCrossValidator to work with labels that span intervals.
    The training set is purged of observations overlapping test-label intervals,
    and a post-test embargo is applied to remove post-event autocorrelation.
    """
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 (label through dates) must be a pd.Series")
        self.n_splits = n_splits
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if hasattr(X, "index") and hasattr(self.t1, "index"):
            if (X.index == self.t1.index).sum() != len(self.t1):
                raise ValueError("X and t1 must have the same index")
        elif len(X) != len(self.t1):
            raise ValueError("X and t1 must have the same length")

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]

        for i, j in test_starts:
            t0 = self.t1.index[i]  # Start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 < t0].index)

            if maxT1Idx < X.shape[0]:  # Right train (with embargo applied)
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
