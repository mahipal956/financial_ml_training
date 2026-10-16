"""
Mean Decrease Accuracy (MDA / Permutation Importance) (AFML Snippet 8.3)
Measures out-of-sample score reduction when feature columns are randomly shuffled.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, f1_score
from validation.purged_kfold import PurgedKFold


def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo=0.0, scoring="f1"):
    """
    MDA Feature Importance across PurgedKFold cross-validation splits.
    """
    if scoring not in ["neg_log_loss", "accuracy", "f1"]:
        raise ValueError(f"Unsupported scoring method: {scoring}")

    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    scr0 = pd.Series(dtype=float)
    scr1 = pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]

        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)

        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
        elif scoring == "f1":
            pred = fit.predict(X1)
            scr0.loc[i] = f1_score(y1, pred, sample_weight=w1.values, average="weighted")
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            col_arr = X1_[j].to_numpy().copy()
            np.random.shuffle(col_arr)
            X1_[j] = col_arr

            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
            elif scoring == "f1":
                pred = fit.predict(X1_)
                scr1.loc[i, j] = f1_score(y1, pred, sample_weight=w1.values, average="weighted")
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)

    imp = (-scr1).add(scr0, axis=0)
    if scoring == "neg_log_loss":
        imp = imp / (-scr1)
    else:
        imp = imp / (1.0 - scr1 + 1e-9)

    imp = pd.concat(
        {
            "mean": imp.mean(),
            "std": imp.std() * (imp.shape[0] ** -0.5),
        },
        axis=1,
    )
    return imp, scr0.mean()
