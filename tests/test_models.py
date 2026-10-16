"""
Unit Tests for AFML Chapter 6 & Chapter 9: Ensemble Architectures & Pipeline Routing
Covers: MyPipeline sample_weight routing, ensemble factory tree constraints, and scalable SVM.
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from models.pipeline import MyPipeline
from models.model_factory import get_model_architectures, get_scalable_svm_bagging


class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(80, 4), columns=[f"f{i}" for i in range(4)])
        self.y = pd.Series(np.random.randint(0, 2, 80))
        self.w = pd.Series(np.random.uniform(0.5, 2.0, 80))

    def test_my_pipeline_sample_weight_routing(self):
        """
        Verifies that MyPipeline intercepts sample_weight and routes it
        into fit_params for the final estimator step.
        """
        pipe = MyPipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ])

        # Fitting with sample_weight directly into fit()
        pipe.fit(self.X, self.y, sample_weight=self.w.values)
        self.assertTrue(hasattr(pipe.named_steps["rf"], "estimators_"))

        # Check prediction
        prob = pipe.predict_proba(self.X)
        self.assertEqual(prob.shape, (80, 2))

    def test_ensemble_architectures_constraints(self):
        avg_u = 0.65
        archs = get_model_architectures(avgU=avg_u)

        # 1. Standard RF
        rf = archs["rf"]
        self.assertEqual(rf.class_weight, "balanced_subsample")
        self.assertEqual(rf.criterion, "entropy")

        # 2. Bagged Decision Tree with max_samples=avgU
        bagged_tree = archs["bagged_tree_avgU"]
        self.assertIsInstance(bagged_tree, BaggingClassifier)
        self.assertAlmostEqual(bagged_tree.max_samples, avg_u)

        # 3. Bagged Single-Tree Forest with max_features=1
        bagged_rf = archs["bagged_rf_avgU"]
        self.assertIsInstance(bagged_rf, BaggingClassifier)
        self.assertAlmostEqual(bagged_rf.max_samples, avg_u)
        base_rf = bagged_rf.estimator
        self.assertEqual(base_rf.max_features, 1)

    def test_scalable_svm_bagging(self):
        svm_bag = get_scalable_svm_bagging(n_estimators=5, max_iter=1000)
        self.assertIsInstance(svm_bag, BaggingClassifier)
        self.assertEqual(svm_bag.estimator.max_iter, 1000)
        self.assertEqual(svm_bag.estimator.tol, 1e-3)
        self.assertTrue(svm_bag.estimator.probability)

        svm_bag.fit(self.X, self.y)
        preds = svm_bag.predict(self.X)
        self.assertEqual(len(preds), 80)


if __name__ == "__main__":
    unittest.main()
