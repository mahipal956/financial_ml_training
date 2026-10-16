"""
Model Architecture Factory (AFML Chapter 6)
Builds Random Forests, Bagged Decision Trees constrained by avgU, and Scalable SVMs.
"""
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def get_model_architectures(avgU=1.0, random_state=42):
    """
    Constructs the three core ensemble architectures detailed in Chapter 6:
    1. Standard Random Forest with balanced subsamples.
    2. Bagged Decision Tree with max_samples restricted to average uniqueness (avgU).
    3. Bagged Single-Tree Forest with max_samples=avgU and max_features=1 (preventing masking).
    """
    # 1. Standard Random Forest
    clf0 = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        criterion="entropy",
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    )

    # 2. Bagged Decision Tree with avgU sub-sampling
    clf1_base = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        class_weight="balanced",
        random_state=random_state,
    )
    clf1 = BaggingClassifier(
        estimator=clf1_base,
        n_estimators=200,
        max_samples=float(min(1.0, max(0.01, avgU))),
        random_state=random_state,
        n_jobs=-1,
    )

    # 3. Bagged Forest with max_features=1 to avoid tree feature masking
    clf2_base = RandomForestClassifier(
        n_estimators=1,
        criterion="entropy",
        max_depth=5,
        bootstrap=False,
        class_weight="balanced_subsample",
        max_features=1,
        random_state=random_state,
    )
    clf2 = BaggingClassifier(
        estimator=clf2_base,
        n_estimators=200,
        max_samples=float(min(1.0, max(0.01, avgU))),
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )

    return {
        "rf": clf0,
        "bagged_tree_avgU": clf1,
        "bagged_rf_avgU": clf2,
    }


def get_scalable_svm_bagging(n_estimators=50, max_iter=100000, random_state=42):
    """
    Scalable Bagged Support Vector Classifier with early stopping (AFML Section 6.7).
    """
    base_svc = SVC(
        kernel="rbf",
        probability=True,
        max_iter=int(max_iter),
        tol=1e-3,
        random_state=random_state,
    )
    scalable_bag = BaggingClassifier(
        estimator=base_svc,
        n_estimators=n_estimators,
        max_samples=0.1,
        random_state=random_state,
        n_jobs=-1,
    )
    return scalable_bag
