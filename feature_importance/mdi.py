"""
Mean Decrease Impurity (MDI) Feature Importance (AFML Snippet 8.2)
In-sample tree impurity reduction. Replaces 0 with NaN to avoid bias from max_features=1.
"""
import numpy as np
import pandas as pd


def featImpMDI(fit, featNames):
    """
    MDI Feature Importance (In-Sample Tree Impurity Reduction).
    - fit: fitted tree-based ensemble (RandomForestClassifier or BaggingClassifier of trees).
    - featNames: list of feature column names.
    """
    estimators = getattr(fit, "estimators_", [])
    if len(estimators) == 0:
        raise ValueError("Model does not have fitted estimators_ attribute")

    # If BaggingClassifier wrapping DecisionTreeClassifier
    trees = []
    for est in estimators:
        if hasattr(est, "tree_"):
            trees.append(est)
        elif hasattr(est, "estimators_"):
            trees.extend(est.estimators_)
        elif hasattr(est, "steps"):
            final_step = est.steps[-1][1]
            if hasattr(final_step, "tree_"):
                trees.append(final_step)

    if len(trees) == 0:
        raise ValueError("Could not extract individual decision trees from model")

    df0 = {i: tree.feature_importances_ for i, tree in enumerate(trees)}
    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # Replace zeros to prevent masking bias

    imp = pd.concat(
        {
            "mean": df0.mean(),
            "std": df0.std() * (df0.shape[0] ** -0.5),
        },
        axis=1,
    )
    imp /= imp["mean"].sum()
    return imp
