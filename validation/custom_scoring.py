"""
Custom Cross-Validation Scoring Module (AFML Snippet 7.4)
Resolves scikit-learn issues (#6231, #9144): preserves sample_weight in both fit and scoring,
and ensures class label awareness when calculating log-loss.
"""
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, f1_score
from validation.purged_kfold import PurgedKFold


def cvScore(
    clf,
    X,
    y,
    sample_weight,
    scoring="neg_log_loss",
    t1=None,
    cv=None,
    cvGen=None,
    pctEmbargo=0.0,
):
    """
    Cross-validation evaluation wrapper that routes sample_weight directly
    to both fit and scoring functions.
    """
    if scoring not in ["neg_log_loss", "accuracy", "f1"]:
        raise ValueError(f"Unsupported scoring method: {scoring}")

    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)

    scores = []
    for train, test in cvGen.split(X=X):
        # Extract train & test slices
        x_train, y_train = X.iloc[train, :], y.iloc[train]
        w_train = sample_weight.iloc[train].values

        x_test, y_test = X.iloc[test, :], y.iloc[test]
        w_test = sample_weight.iloc[test].values

        fit = clf.fit(X=x_train, y=y_train, sample_weight=w_train)

        if scoring == "neg_log_loss":
            prob = fit.predict_proba(x_test)
            score_ = -log_loss(
                y_test,
                prob,
                sample_weight=w_test,
                labels=clf.classes_,
            )
        elif scoring == "f1":
            pred = fit.predict(x_test)
            score_ = f1_score(
                y_test,
                pred,
                sample_weight=w_test,
                average="weighted",
            )
        else:
            pred = fit.predict(x_test)
            score_ = accuracy_score(
                y_test,
                pred,
                sample_weight=w_test,
            )
        scores.append(score_)

    return np.array(scores)
