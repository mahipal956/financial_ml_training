"""
Unit Tests for AFML Chapter 8: Feature Importance Suite
Covers: MDI (zero-handling, max_features=1), MDA (Purged CV permutation), SFI, PCA Orthogonal Features, and Weighted Kendall's Tau.
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from feature_importance.mdi import featImpMDI
from feature_importance.mda import featImpMDA
from feature_importance.sfi import featImpSFI
from feature_importance.pca_ortho import orthoFeats, check_pca_rank_correl


class TestFeatureImportance(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.times = pd.date_range("2020-01-01", periods=80, freq="D")
        self.t1 = pd.Series([self.times[min(79, i + 3)] for i in range(80)], index=self.times)
        
        # 3 features: f0 is informative, f1 and f2 are noise
        f0 = np.random.randn(80)
        f1 = np.random.randn(80)
        f2 = np.random.randn(80)
        self.X = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2}, index=self.times)
        self.y = pd.Series((f0 > 0).astype(int), index=self.times)
        self.w = pd.Series(np.ones(80), index=self.times)

    def test_mdi_zero_handling(self):
        rf = RandomForestClassifier(n_estimators=30, max_features=1, random_state=42)
        rf.fit(self.X, self.y, sample_weight=self.w.values)
        
        imp = featImpMDI(rf, list(self.X.columns))
        self.assertEqual(len(imp), 3)
        self.assertIn("mean", imp.columns)
        self.assertIn("std", imp.columns)
        
        # MDI means should sum to 1.0
        self.assertAlmostEqual(imp["mean"].sum(), 1.0, places=4)
        # Informative feature f0 should have highest importance
        self.assertEqual(imp["mean"].idxmax(), "f0")

    def test_mda_purged(self):
        rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
        imp, base_score = featImpMDA(
            clf=rf,
            X=self.X,
            y=self.y,
            cv=3,
            sample_weight=self.w,
            t1=self.t1,
            pctEmbargo=0.01,
            scoring="f1",
        )
        self.assertEqual(len(imp), 3)
        self.assertIn("mean", imp.columns)
        # Shuffling informative feature f0 should drop performance the most
        self.assertEqual(imp["mean"].idxmax(), "f0")

    def test_sfi(self):
        rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
        imp = featImpSFI(
            clf=rf,
            X=self.X,
            y=self.y,
            cv=3,
            sample_weight=self.w,
            t1=self.t1,
            pctEmbargo=0.01,
            scoring="f1",
        )
        self.assertEqual(len(imp), 3)
        # Informative feature f0 must rank #1 in SFI
        self.assertEqual(imp.index[0], "f0")

    def test_ortho_feats_and_kendall_tau(self):
        dfP, eVal, eVec = orthoFeats(self.X, varThres=0.95)
        self.assertGreaterEqual(dfP.shape[1], 1)
        self.assertLessEqual(dfP.shape[1], 3)
        
        # Test Kendall's tau correlation
        feat_imp = np.array([0.60, 0.25, 0.15])
        pc_ranks = np.array([1, 2, 3])
        tau, p_val = check_pca_rank_correl(feat_imp, pc_ranks)
        self.assertAlmostEqual(tau, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
