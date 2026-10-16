"""
Hyperparameter Fitting with PurgedKFold Cross-Validation (AFML Snippet 9.3)
Performs Grid or Randomized Search over PurgedKFold splits using appropriate financial scoring.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from validation.purged_kfold import PurgedKFold
from models.pipeline import MyPipeline


def clfHyperFit(
    feat,
    lbl,
    t1,
    pipe_clf,
    param_grid,
    cv=3,
    bagging=None,
    rndSearchIter=0,
    n_jobs=-1,
    pctEmbargo=0.0,
    random_state=42,
    **fit_params,
):
    """
    1. Selects scoring metric: 'f1' for binary meta-labeling {0, 1}, 'neg_log_loss' otherwise.
    2. Runs Purged GridSearchCV or RandomizedSearchCV.
    3. Fits validated model on full dataset (with optional Bagging wrapper).
    """
    if bagging is None:
        bagging = [0, None, 1.0]

    unique_labels = set(lbl.unique())
    if unique_labels == {0, 1}:
        scoring = "f1"  # F1 for meta-labeling
    else:
        scoring = "neg_log_loss"  # Cross-entropy loss for multi-class

    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)

    if rndSearchIter == 0:
        gs = GridSearchCV(
            estimator=pipe_clf,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
        )
    else:
        gs = RandomizedSearchCV(
            estimator=pipe_clf,
            param_distributions=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            n_iter=rndSearchIter,
            random_state=random_state,
        )

    gs.fit(feat, lbl, **fit_params)
    best_estimator = gs.best_estimator_

    # Optional Bagging wrapper around tuned model
    if bagging[0] > 0 and bagging[1] is not None:
        sub_sample = float(bagging[1])
        base_pipe = MyPipeline(best_estimator.steps)
        bagged_clf = BaggingClassifier(
            estimator=base_pipe,
            n_estimators=int(bagging[0]),
            max_samples=sub_sample,
            max_features=float(bagging[2]),
            n_jobs=n_jobs,
            random_state=random_state,
        )
        # Extract sample_weight if present in fit_params
        sample_w_key = f"{best_estimator.steps[-1][0]}__sample_weight"
        sw = fit_params.get(sample_w_key, fit_params.get("sample_weight", None))
        if sw is not None:
            bagged_clf.fit(feat, lbl, sample_weight=sw)
        else:
            bagged_clf.fit(feat, lbl)
        best_estimator = MyPipeline([("bag", bagged_clf)])

    return best_estimator, gs.best_params_, gs.best_score_
