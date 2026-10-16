"""
Unit Tests for AFML Chapter 9: Hyperparameter Tuning with Purged Cross-Validation
Covers: logUniform distribution generator and clfHyperFit with PurgedKFold.
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from models.pipeline import MyPipeline
from tuning.distributions import logUniform
from tuning.hyper_fit import clfHyperFit


class TestTuning(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.times = pd.date_range("2020-01-01", periods=60, freq="D")
        self.t1 = pd.Series([self.times[min(59, i + 3)] for i in range(60)], index=self.times)
        self.X = pd.DataFrame(np.random.randn(60, 3), index=self.times, columns=["f1", "f2", "f3"])
        self.y_meta = pd.Series(np.random.randint(0, 2, 60), index=self.times)
        self.w = pd.Series(np.random.uniform(0.5, 1.5, 60), index=self.times)

    def test_log_uniform_distribution(self):
        dist = logUniform(a=1e-3, b=1e2)
        samples = dist.rvs(size=500, random_state=42)
        
        self.assertTrue((samples >= 1e-3).all())
        self.assertTrue((samples <= 1e2).all())
        
        # Test analytical CDF
        self.assertAlmostEqual(dist._cdf(1e-3), 0.0)
        self.assertAlmostEqual(dist._cdf(1e2), 1.0)
        # Median of log-uniform on [10^-3, 10^2] is 10^(-0.5) ~ 0.3162
        self.assertAlmostEqual(dist._cdf(np.sqrt(1e-3 * 1e2)), 0.5)

    def test_clf_hyper_fit_purged(self):
        pipe = MyPipeline([("tree", DecisionTreeClassifier(random_state=42))])
        param_grid = {"tree__max_depth": [2, 3]}

        best_pipe, best_params, best_score = clfHyperFit(
            feat=self.X,
            lbl=self.y_meta,
            t1=self.t1,
            pipe_clf=pipe,
            param_grid=param_grid,
            cv=3,
            bagging=[5, 0.8, 1.0],
            rndSearchIter=2,
            n_jobs=1,
            pctEmbargo=0.01,
            sample_weight=self.w.values,
        )

        self.assertIn("tree__max_depth", best_params)
        self.assertIsNotNone(best_score)
        self.assertTrue(hasattr(best_pipe, "predict"))

        preds = best_pipe.predict(self.X)
        self.assertEqual(len(preds), 60)


if __name__ == "__main__":
    unittest.main()
