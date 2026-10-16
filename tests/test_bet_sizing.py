"""
Unit Tests for AFML Chapter 10: Bet Sizing Module
Covers: OvR t-statistic, Normal CDF bet sizing [-1, 1], active signal averaging, and step-size discretization.
"""
import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
from bet_sizing.position_sizer import getSignal, avgActiveSignals, discreteSignal


class TestBetSizing(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.idx = pd.date_range("2020-01-01", periods=6, freq="h")
        self.t1 = pd.Series(pd.date_range("2020-01-01 03:00:00", periods=6, freq="h"), index=self.idx)
        self.events = pd.DataFrame({"t1": self.t1, "side": [1, -1, 1, -1, 1, 1]}, index=self.idx)

    def test_ovr_t_stat_and_cdf_mapping(self):
        # When p = 0.5 (random guess for 2 classes), z = 0, bet_size = 0
        prob = pd.Series([0.5], index=[self.idx[0]])
        events = pd.DataFrame({"side": [1]}, index=[self.idx[0]])
        signal = getSignal(events=events, stepSize=0.01, prob=prob, numClasses=2)
        self.assertAlmostEqual(signal.iloc[0], 0.0)

        # When p = 0.9, z = (0.9 - 0.5) / sqrt(0.9 * 0.1) = 0.4 / 0.3 = 1.3333...
        # CDF(1.3333...) ~ 0.908788
        # m = 2 * 0.908788 - 1 ~ 0.8175
        expected_z = (0.9 - 0.5) / np.sqrt(0.9 * 0.1)
        expected_m = 2.0 * norm.cdf(expected_z) - 1.0
        
        prob_high = pd.Series([0.9], index=[self.idx[0]])
        signal_high = getSignal(events=events, stepSize=0.001, prob=prob_high, numClasses=2)
        self.assertAlmostEqual(signal_high.iloc[0], expected_m, places=2)

    def test_meta_labeling_side_direction(self):
        # Positive prediction with negative side (-1) must yield negative bet size
        prob = pd.Series([0.85], index=[self.idx[0]])
        events_short = pd.DataFrame({"side": [-1]}, index=[self.idx[0]])
        signal = getSignal(events=events_short, stepSize=0.01, prob=prob, numClasses=2)
        self.assertLess(signal.iloc[0], 0.0)

    def test_discrete_signal_and_clamping(self):
        raw = pd.Series([-1.5, -0.42, 0.03, 0.47, 1.3])
        step = 0.05
        disc = discreteSignal(raw, stepSize=step)

        # Clamping
        self.assertEqual(disc.iloc[0], -1.0)
        self.assertEqual(disc.iloc[-1], 1.0)

        # Discretization to multiples of 0.05
        self.assertAlmostEqual(disc.iloc[1], -0.40)
        self.assertAlmostEqual(disc.iloc[2], 0.05)
        self.assertAlmostEqual(disc.iloc[3], 0.45)

    def test_active_signal_averaging(self):
        # Two overlapping events with signals +0.8 and +0.4
        t0 = self.idx[0]
        t1_ev = self.idx[2]
        
        signals_df = pd.DataFrame({
            "signal": [0.8, 0.4],
            "t1": [t1_ev, t1_ev],
        }, index=[t0, t0])

        avg_sig = avgActiveSignals(signals_df, numThreads=1)
        self.assertIn(t0, avg_sig.index)
        self.assertAlmostEqual(avg_sig.loc[t0], 0.6)


if __name__ == "__main__":
    unittest.main()
