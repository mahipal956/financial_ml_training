"""
Unit Tests for Sample Weights and Uniqueness (AFML Chapter 4)
"""
import pytest
import numpy as np
import pandas as pd
from sample_weights.concurrency import mpNumCoEvents
from sample_weights.uniqueness import mpSampleTW, getIndMatrix, getAvgUniqueness
from sample_weights.time_decay import getTimeDecay


def test_concurrency_and_uniqueness():
    """
    Test concurrency and average uniqueness calculation.
    """
    dates = pd.date_range("2026-01-01", periods=10, freq="1D")
    # 3 events:
    # Event 0: Day 0 to Day 2
    # Event 1: Day 1 to Day 3
    # Event 2: Day 5 to Day 6
    t1 = pd.Series([dates[2], dates[3], dates[6]], index=[dates[0], dates[1], dates[5]])

    numCo = mpNumCoEvents(dates, t1, t1.index)
    assert numCo.loc[dates[0]] == 1.0  # Only event 0 is active on day 0
    assert numCo.loc[dates[1]] == 2.0  # Events 0 and 1 active on day 1
    assert numCo.loc[dates[2]] == 2.0  # Events 0 and 1 active on day 2
    assert numCo.loc[dates[5]] == 1.0  # Only event 2 active on day 5

    tw = mpSampleTW(t1, numCo, t1.index)
    assert len(tw) == 3
    assert 0.0 < tw.loc[dates[0]] <= 1.0
    assert tw.loc[dates[5]] == 1.0  # Completely unique event


def test_indicator_matrix():
    dates = pd.date_range("2026-01-01", periods=5, freq="1D")
    t1 = pd.Series([dates[2], dates[4]], index=[dates[0], dates[2]])
    indM = getIndMatrix(dates, t1)
    assert indM.shape == (5, 2)
    assert indM.iloc[0, 0] == 1.0
    assert indM.iloc[3, 0] == 0.0
    assert indM.iloc[2, 1] == 1.0


def test_time_decay():
    dates = pd.date_range("2026-01-01", periods=5, freq="1D")
    tw = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=dates)
    decay_w = getTimeDecay(tw, clfLastW=0.5)
    assert len(decay_w) == 5
    # Newest observation gets weight 1.0, oldest gets weight 0.6 (const 0.5 + slope 0.1 * 1)
    assert np.isclose(decay_w.iloc[-1], 1.0)
    assert np.isclose(decay_w.iloc[0], 0.6)
