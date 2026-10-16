"""
Unit Tests for AFML Chapter 4: Sample Weighting & Multiprocessing Engine
Covers: mpNumCoEvents, mpSampleTW, getIndMatrix, getAvgUniqueness, seqBootstrap, mpSampleW, getTimeDecay, mpPandasObj
"""
import unittest
import numpy as np
import pandas as pd
from sample_weights.concurrency import mpNumCoEvents
from sample_weights.uniqueness import mpSampleTW, getIndMatrix, getAvgUniqueness, seqBootstrap
from sample_weights.return_attribution import mpSampleW
from sample_weights.time_decay import getTimeDecay
from sample_weights.mp_engine import mpPandasObj, linParts, nestedParts


class TestSampleWeights(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create 10 daily bars
        self.bar_index = pd.date_range("2020-01-01", periods=10, freq="D")
        self.close = pd.Series([100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 107.0], index=self.bar_index)
        
        # 3 events:
        # Event 0: from bar 0 to bar 3
        # Event 1: from bar 2 to bar 5
        # Event 2: from bar 4 to bar 8
        self.t1 = pd.Series(
            [self.bar_index[3], self.bar_index[5], self.bar_index[8]],
            index=[self.bar_index[0], self.bar_index[2], self.bar_index[4]]
        )

    def test_concurrency_mpNumCoEvents(self):
        co_events = mpNumCoEvents(self.bar_index, self.t1, self.t1.index)
        self.assertIsInstance(co_events, pd.Series)
        # Bar 2 overlaps with Event 0 and Event 1 -> concurrency should be 2
        self.assertEqual(co_events.loc[self.bar_index[2]], 2.0)
        # Bar 4 overlaps with Event 1 and Event 2 -> concurrency should be 2
        self.assertEqual(co_events.loc[self.bar_index[4]], 2.0)
        # Bar 0 only overlaps with Event 0 -> concurrency should be 1
        self.assertEqual(co_events.loc[self.bar_index[0]], 1.0)

    def test_uniqueness_mpSampleTW(self):
        co_events = mpNumCoEvents(self.bar_index, self.t1, self.t1.index)
        u = mpSampleTW(self.t1, co_events, self.t1.index)
        self.assertEqual(len(u), len(self.t1))
        # Uniqueness must be bounded in (0, 1]
        for val in u:
            self.assertGreater(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_indicator_matrix_and_avg_uniqueness(self):
        ind_m = getIndMatrix(self.bar_index, self.t1)
        self.assertEqual(ind_m.shape, (10, 3))
        # Indicator for event 0 should be 1.0 from bar 0 to 3
        self.assertTrue((ind_m.loc[self.bar_index[0]:self.bar_index[3], 0] == 1.0).all())
        self.assertTrue((ind_m.loc[self.bar_index[4]:, 0] == 0.0).all())
        
        avg_u = getAvgUniqueness(ind_m)
        self.assertIsInstance(avg_u, (pd.Series, float))

    def test_sequential_bootstrap(self):
        ind_m = getIndMatrix(self.bar_index, self.t1)
        phi = seqBootstrap(ind_m, sLength=3)
        self.assertEqual(len(phi), 3)
        for idx in phi:
            self.assertIn(idx, [0, 1, 2])

    def test_return_attribution_mpSampleW(self):
        co_events = mpNumCoEvents(self.bar_index, self.t1, self.t1.index)
        weights = mpSampleW(self.t1, co_events, self.close, self.t1.index)
        self.assertEqual(len(weights), len(self.t1))
        # Absolute returns -> weights must be non-negative
        self.assertTrue((weights >= 0.0).all())

    def test_time_decay_piecewise_linear(self):
        # Create weights
        tw = pd.Series([1.0, 1.0, 1.0, 1.0], index=self.bar_index[:4])
        # Case 1: clfLastW = 1.0 (no decay)
        w_flat = getTimeDecay(tw, clfLastW=1.0)
        self.assertTrue(np.allclose(w_flat.values, [1.0, 1.0, 1.0, 1.0]))
        
        # Case 2: clfLastW = 0.5 (decay oldest observation to 0.5)
        w_decay = getTimeDecay(tw, clfLastW=0.5)
        self.assertAlmostEqual(w_decay.iloc[-1], 1.0)
        self.assertAlmostEqual(w_decay.iloc[0], 0.625)  # Const + slope * cumsum
        self.assertTrue((w_decay.diff().dropna() > 0).all())  # Strictly increasing towards newest
        
        # Case 3: clfLastW < 0 (accelerated decay)
        w_acc = getTimeDecay(tw, clfLastW=-0.5)
        self.assertAlmostEqual(w_acc.iloc[-1], 1.0)
        self.assertTrue((w_acc >= 0.0).all())

    def test_multiprocessing_engine(self):
        parts_lin = linParts(100, 4)
        self.assertEqual(len(parts_lin), 5)
        self.assertEqual(parts_lin[0], 0)
        self.assertEqual(parts_lin[-1], 100)

        parts_nested = nestedParts(100, 4)
        self.assertEqual(len(parts_nested), 5)
        self.assertEqual(parts_nested[0], 0)
        
        # mpPandasObj test
        def square_func(molecule, scale=2.0):
            return pd.Series(molecule * scale, index=molecule)

        res = mpPandasObj(square_func, ("molecule", np.arange(10)), numThreads=2, scale=3.0)
        self.assertEqual(len(res), 10)
        self.assertEqual(res.iloc[4], 12.0)


if __name__ == "__main__":
    unittest.main()
