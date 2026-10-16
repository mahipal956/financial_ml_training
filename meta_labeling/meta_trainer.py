"""
Meta-Labeling Trainer Module (AFML Chapter 3 & Chapter 6)
Fits a secondary model to learn when to execute primary model trades (binary meta-labels {0, 1}).
"""
import pandas as pd
from models.pipeline import MyPipeline
from validation.custom_scoring import cvScore


class MetaLabelTrainer:
    """
    Trains and validates a secondary classifier on meta-labels.
    Primary model decides side {-1, 1}; meta-model decides trade execution and bet size {0, 1}.
    """
    def __init__(self, base_estimator, cv=5, pct_embargo=0.01, scoring="f1"):
        self.base_estimator = base_estimator
        self.cv = cv
        self.pct_embargo = pct_embargo
        self.scoring = scoring
        self.fitted_model_ = None
        self.cv_scores_ = None

    def fit_and_evaluate(self, X, y, sample_weight, t1):
        """
        Validates meta-model out-of-sample using PurgedKFold, then fits on all data.
        """
        # 1. Evaluate out-of-sample performance
        self.cv_scores_ = cvScore(
            clf=self.base_estimator,
            X=X,
            y=y,
            sample_weight=sample_weight,
            scoring=self.scoring,
            t1=t1,
            cv=self.cv,
            pctEmbargo=self.pct_embargo,
        )

        # 2. Fit final model on full dataset
        pipe = MyPipeline([("model", self.base_estimator)])
        pipe.fit(X, y, sample_weight=sample_weight.values)
        self.fitted_model_ = pipe
        return self

    def predict_proba(self, X):
        if self.fitted_model_ is None:
            raise ValueError("Model must be fitted before predict_proba")
        return self.fitted_model_.predict_proba(X)

    def predict(self, X):
        if self.fitted_model_ is None:
            raise ValueError("Model must be fitted before predict")
        return self.fitted_model_.predict(X)
