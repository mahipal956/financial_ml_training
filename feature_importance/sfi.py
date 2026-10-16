"""
Single Feature Importance (SFI) Module (AFML Snippet 8.4)
Evaluates out-of-sample performance of each feature in isolation to eliminate multi-collinearity and substitution effects.
"""
import pandas as pd
from validation.custom_scoring import cvScore


def featImpSFI(clf, X, y, cv, sample_weight, t1, pctEmbargo=0.0, scoring="f1"):
    """
    Computes Single Feature Importance (SFI).
    Fits the classifier on a single feature column at a time across PurgedKFold splits.
    """
    imp = pd.DataFrame(columns=["mean", "std"])
    for feat in X.columns:
        x_sub = X[[feat]]
        scores = cvScore(
            clf=clf,
            X=x_sub,
            y=y,
            sample_weight=sample_weight,
            scoring=scoring,
            t1=t1,
            cv=cv,
            pctEmbargo=pctEmbargo,
        )
        imp.loc[feat, "mean"] = scores.mean()
        imp.loc[feat, "std"] = scores.std() * (scores.shape[0] ** -0.5)
    return imp.sort_values(by="mean", ascending=False)
