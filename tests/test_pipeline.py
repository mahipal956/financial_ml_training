"""
Unit Tests for Custom Pipeline and Scoring (AFML Chapter 7 & 9)
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from models.pipeline import MyPipeline
from validation.custom_scoring import cvScore
from validation.purged_kfold import PurgedKFold


def test_mypipeline_sample_weight_routing():
    """
    Test that MyPipeline successfully captures sample_weight and routes it to fit_params.
    """
    X = pd.DataFrame(np.random.randn(50, 4))
    y = pd.Series(np.random.choice([0, 1], size=50))
    weights = pd.Series(np.random.uniform(0.1, 2.0, size=50))

    dt = DecisionTreeClassifier(max_depth=3)
    pipe = MyPipeline([("tree", dt)])

    # Fit with sample_weight
    pipe.fit(X, y, sample_weight=weights.values)
    assert hasattr(pipe.steps[-1][1], "tree_")


def test_cvscore_neg_log_loss_and_f1():
    """
    Test cvScore with both neg_log_loss and f1 metrics.
    """
    dates = pd.date_range("2026-01-01", periods=60, freq="1D")
    t1 = pd.Series(dates[min(len(dates) - 1, i + 3)] for i in range(len(dates)))
    t1.index = dates

    X = pd.DataFrame(np.random.randn(60, 3), index=dates, columns=["f1", "f2", "f3"])
    y = pd.Series(np.random.choice([0, 1], size=60), index=dates)
    weights = pd.Series(np.ones(60), index=dates)

    dt = DecisionTreeClassifier(max_depth=2)

    # Test F1 score
    f1_scores = cvScore(
        clf=dt,
        X=X,
        y=y,
        sample_weight=weights,
        scoring="f1",
        t1=t1,
        cv=3,
        pctEmbargo=0.01,
    )
    assert len(f1_scores) == 3
    assert (f1_scores >= 0.0).all() and (f1_scores <= 1.0).all()

    # Test neg_log_loss
    ll_scores = cvScore(
        clf=dt,
        X=X,
        y=y,
        sample_weight=weights,
        scoring="neg_log_loss",
        t1=t1,
        cv=3,
        pctEmbargo=0.01,
    )
    assert len(ll_scores) == 3
    assert (ll_scores <= 0.0).all()
