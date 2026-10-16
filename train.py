"""
Financial Machine Learning: End-to-End Master Training Pipeline
Author: Marcos López de Prado AFML Framework
Dataset: USDJPY Bar History with CUSUM Events and Triple-Barrier Meta-Labels
"""
import sys
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Ensure local modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.paths import (
    DEFAULT_DATASET_PATH,
    OUTPUTS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    LOGS_DIR,
    CBOT_PROJECT_DIR,
)
from data_loader.dataset_loader import load_dataset
from sample_weights.concurrency import mpNumCoEvents
from sample_weights.uniqueness import mpSampleTW, getIndMatrix, getAvgUniqueness
from sample_weights.return_attribution import mpSampleW
from sample_weights.time_decay import getTimeDecay
from validation.purged_kfold import PurgedKFold
from validation.custom_scoring import cvScore
from models.model_factory import get_model_architectures, get_scalable_svm_bagging
from models.pipeline import MyPipeline
from tuning.distributions import logUniform
from tuning.hyper_fit import clfHyperFit
from feature_importance.mdi import featImpMDI
from feature_importance.mda import featImpMDA
from feature_importance.sfi import featImpSFI
from feature_importance.pca_ortho import orthoFeats, check_pca_rank_correl
from meta_labeling.meta_trainer import MetaLabelTrainer
from bet_sizing.position_sizer import getSignal
from export.onnx_exporter import export_model_to_onnx
from export.json_exporter import export_config_to_json


def run_pipeline(dataset_path=None, config_path=None):
    print("=" * 80)
    print("  FINANCIAL MACHINE LEARNING: MASTER TRAINING PIPELINE (AFML)")
    print("=" * 80)

    # 1. Setup Directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    data_file = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    print(f"\n[1/8] Ingesting Dataset: {data_file}")
    data = load_dataset(data_file)
    X = data["X"]
    y = data["y"]
    t1 = data["t1"]
    w = data["sample_weight"]
    close = data["close"]
    full_df = data["full_df"]

    print(f"      Samples: {len(X)} | Features: {list(X.columns)}")
    print(f"      Meta-Label Class Distribution:\n{y.value_counts().to_dict()}")

    # 2. Sample Uniqueness & Concurrency Analysis
    print("\n[2/8] Calculating Concurrency & Average Uniqueness (avgU)...")
    ind_m = getIndMatrix(close.index, t1)
    avg_u = getAvgUniqueness(ind_m)
    if isinstance(avg_u, pd.Series):
        avg_u_val = float(avg_u.iloc[-1])
    else:
        avg_u_val = float(avg_u) if not np.isnan(avg_u) else 0.5
    avg_u_val = max(0.05, min(1.0, avg_u_val))
    print(f"      Calculated avgU: {avg_u_val:.4f}")

    # 3. Model Architecture Selection
    print("\n[3/8] Instantiating Financial Ensemble Architectures...")
    architectures = get_model_architectures(avgU=avg_u_val)
    # Primary model: Bagged trees constrained by avgU
    selected_clf = architectures["bagged_tree_avgU"]
    print(f"      Selected Estimator: BaggingClassifier(max_samples=avgU={avg_u_val:.3f}, n_estimators=200)")

    # 4. Feature Importance Analysis
    print("\n[4/8] Running Marcos' Feature Importance Suite...")
    # A. MDI (In-Sample Impurity Reduction)
    # Fit temporary RF with max_features=1 to avoid tree masking
    temp_rf = architectures["rf"]
    temp_rf.fit(X, y, sample_weight=w.values)
    mdi_imp = featImpMDI(temp_rf, list(X.columns))
    print("      --- MDI Feature Importance ---")
    for feat, row in mdi_imp.iterrows():
        print(f"      {feat:15s}: mean={row['mean']:.4f}, std={row['std']:.4f}")

    # B. MDA (Out-of-Sample Permutation with PurgedKFold)
    print("      --- MDA Feature Importance (Purged Out-of-Sample) ---")
    mda_imp, baseline_score = featImpMDA(
        clf=selected_clf,
        X=X,
        y=y,
        cv=5,
        sample_weight=w,
        t1=t1,
        pctEmbargo=0.01,
        scoring="f1",
    )
    for feat, row in mda_imp.iterrows():
        print(f"      {feat:15s}: mean={row['mean']:.4f}, std={row['std']:.4f}")

    # C. SFI (Single Feature Importance Out-of-Sample)
    print("      --- SFI Feature Importance (Single Feature OOS) ---")
    sfi_imp = featImpSFI(
        clf=selected_clf,
        X=X,
        y=y,
        cv=5,
        sample_weight=w,
        t1=t1,
        pctEmbargo=0.01,
        scoring="f1",
    )
    for feat, row in sfi_imp.iterrows():
        print(f"      {feat:15s}: mean={row['mean']:.4f}, std={row['std']:.4f}")

    # D. PCA Orthogonal Feature & Rank Concordance Check (AFML Section 8.4)
    print("      --- PCA Orthogonal Features & Weighted Kendall's Tau ---")
    dfP, eVal, eVec = orthoFeats(X, varThres=0.95)
    ortho_rf = architectures["rf"]
    ortho_rf.fit(dfP, y, sample_weight=w.values)
    ortho_mdi = featImpMDI(ortho_rf, list(dfP.columns))
    pc_ranks = np.arange(1, len(dfP.columns) + 1)
    tau, p_val = check_pca_rank_correl(ortho_mdi["mean"].values, pc_ranks)
    print(f"      PCA Components explaining >=95% variance: {dfP.shape[1]}")
    print(f"      Weighted Kendall's Tau (Orthogonal MDI vs. Inverse PCA Rank): {tau:.4f}")

    # Generate Feature Importance Bar Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    mdi_imp["mean"].plot(kind="barh", xerr=mdi_imp["std"], ax=axes[0], color="steelblue", title="MDI (In-Sample)")
    axes[0].axvline(1.0 / len(X.columns), color="red", linestyle="--", label="1/N Benchmark")
    axes[0].legend()

    mda_imp["mean"].plot(kind="barh", xerr=mda_imp["std"], ax=axes[1], color="forestgreen", title="MDA (Out-of-Sample)")
    axes[1].axvline(0, color="black", linestyle=":")

    sfi_imp["mean"].plot(kind="barh", xerr=sfi_imp["std"], ax=axes[2], color="darkorange", title="SFI (Single Feature OOS)")

    plt.tight_layout()
    plot_file = PLOTS_DIR / "feature_importance_summary.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"      Saved diagnostic plots to: {plot_file}")

    # 5. Out-of-Sample Purged Validation & Final Model Fitting
    print("\n[5/8] Validating Meta-Label Model via PurgedKFold Cross-Validation...")
    meta_trainer = MetaLabelTrainer(base_estimator=selected_clf, cv=5, pct_embargo=0.01, scoring="f1")
    meta_trainer.fit_and_evaluate(X=X, y=y, sample_weight=w, t1=t1)
    cv_scores = meta_trainer.cv_scores_
    print(f"      PurgedKFold Out-of-Sample F1 Scores: {np.round(cv_scores, 4)}")
    print(f"      Mean Out-of-Sample F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 6. Bet Sizing Calibration
    print("\n[6/8] Calibrating Probability-to-Position Bet Sizing...")
    # Predict positive class probabilities on full set
    pred_prob = pd.Series(meta_trainer.predict_proba(X)[:, 1], index=X.index)
    pred_binary = pd.Series(meta_trainer.predict(X), index=X.index)

    events_df = pd.DataFrame(index=X.index)
    events_df["t1"] = t1
    if "Primary_Model_Side" in full_df.columns:
        events_df["side"] = full_df.loc[X.index, "Primary_Model_Side"]

    bet_signals = getSignal(
        events=events_df,
        stepSize=0.05,
        prob=pred_prob,
        pred=pred_binary,
        numClasses=2,
        numThreads=1,
    )
    print(f"      Calculated Active Bet Sizes:")
    print(f"      Mean Bet Size: {bet_signals.mean():.4f} | Max: {bet_signals.max():.2f} | Min: {bet_signals.min():.2f}")
    print(f"      Zero-Sized Bets (Filtered Out): {(bet_signals == 0).sum()} / {len(bet_signals)} ({(bet_signals == 0).mean()*100:.1f}%)")

    # 7. Model Serialization & Export
    print("\n[7/8] Exporting Trained Model and Configuration Artifacts...")
    onnx_path = MODELS_DIR / "model.onnx"
    export_model_to_onnx(meta_trainer.fitted_model_, n_features=X.shape[1], output_path=onnx_path)

    # Compute feature stats for normalization in cBot
    feat_stats = {
        col: {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()) if X[col].std() > 0 else 1.0,
            "min": float(X[col].min()),
            "max": float(X[col].max()),
        }
        for col in X.columns
    }

    json_path = MODELS_DIR / "model_config.json"
    export_config_to_json(
        feature_names=list(X.columns),
        step_size=0.05,
        execution_threshold=0.5,
        feature_stats=feat_stats,
        model_params={
            "avg_u": avg_u_val,
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "n_samples": len(X),
        },
        output_path=json_path,
    )

    # Also copy exported model artifacts directly into cBot directory if it exists
    if CBOT_PROJECT_DIR.exists():
        import shutil
        cbot_json = CBOT_PROJECT_DIR / "model_config.json"
        cbot_onnx = CBOT_PROJECT_DIR / "model.onnx"
        shutil.copyfile(json_path, cbot_json)
        if onnx_path.exists():
            shutil.copyfile(onnx_path, cbot_onnx)
        print(f"      Copied model artifacts to cBot folder: {CBOT_PROJECT_DIR}")

    print("\n[8/8] Pipeline Complete! All artifacts successfully generated.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
