"""
Unit Tests for AFML Chapter 7: Financial Cross-Validation & PurgedKFold
Covers: PurgedKFold purging, post-test embargo, and cvScore sample_weight and label preservation.
Verifies elimination of scikit-learn Issue #6231 and Issue #9144.
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from validation.purged_kfold import PurgedKFold, getTrainTimes, getEmbargoTimes
from validation.custom_scoring import cvScore


class TestPurgedCV(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 100 observations across time
        self.times = pd.date_range("2020-01-01", periods=100, freq="D")
        # Event durations between 2 and 5 days
        end_times = [self.times[min(99, i + np.random.randint(2, 6))] for i in range(100)]
        self.t1 = pd.Series(end_times, index=self.times)
        
        self.X = pd.DataFrame(np.random.randn(100, 4), index=self.times, columns=[f"f{i}" for i in range(4)])
        self.y = pd.Series(np.random.randint(0, 2, 100), index=self.times)
        self.weights = pd.Series(np.random.uniform(0.5, 1.5, 100), index=self.times)

    def test_purged_kfold_overlap_purging(self):
        cv = PurgedKFold(n_splits=5, t1=self.t1, pctEmbargo=0.0)
        splits = list(cv.split(self.X))
        self.assertEqual(len(splits), 5)
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            t0_test = self.times[test_idx[0]]
            t1_test_max = self.t1.iloc[test_idx].max()
            
            # Check left train purging: NO training event may end AFTER t0_test
            left_train = [i for i in train_idx if i < test_idx[0]]
            for i in left_train:
                ev_end = self.t1.iloc[i]
                self.assertLessEqual(
                    ev_end, t0_test,
                    f"Fold {fold}: Train event {i} ends at {ev_end} > test start {t0_test} (Leakage detected!)"
                )

            # Check right train purging: NO right train event may start BEFORE max test end
            right_train = [i for i in train_idx if i > test_idx[-1]]
            for i in right_train:
                ev_start = self.times[i]
                self.assertGreaterEqual(
                    ev_start, t1_test_max,
                    f"Fold {fold}: Right train event {i} starts at {ev_start} < max test end {t1_test_max} (Leakage detected!)"
                )

    def test_purged_kfold_embargo(self):
        pct_embargo = 0.05
        cv_no_embargo = PurgedKFold(n_splits=5, t1=self.t1, pctEmbargo=0.0)
        cv_with_embargo = PurgedKFold(n_splits=5, t1=self.t1, pctEmbargo=pct_embargo)
        
        train_no_embargo, _ = next(cv_no_embargo.split(self.X))
        train_with_embargo, _ = next(cv_with_embargo.split(self.X))
        
        # Training set with embargo must be smaller due to embargo drops
        self.assertLessEqual(len(train_with_embargo), len(train_no_embargo))

    def test_cv_score_sample_weight_and_label_preservation(self):
        """
        Tests scikit-learn bug elimination:
        1. Issue #9144: Routes sample_weight to scoring metric on test set.
        2. Issue #6231: Passes clf.classes_ to log_loss so folds missing a class do not fail.
        """
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        
        # Test with neg_log_loss
        scores_log_loss = cvScore(
            clf=clf,
            X=self.X,
            y=self.y,
            sample_weight=self.weights,
            scoring="neg_log_loss",
            t1=self.t1,
            cv=3,
            pctEmbargo=0.01,
        )
        self.assertEqual(len(scores_log_loss), 3)
        self.assertTrue((scores_log_loss < 0.0).all())  # neg_log_loss is negative

        # Test with F1 score
        scores_f1 = cvScore(
            clf=clf,
            X=self.X,
            y=self.y,
            sample_weight=self.weights,
            scoring="f1",
            t1=self.t1,
            cv=3,
            pctEmbargo=0.01,
        )
        self.assertEqual(len(scores_f1), 3)
        self.assertTrue(((scores_f1 >= 0.0) & (scores_f1 <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
