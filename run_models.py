"""
5-model × 5-feature-set ablation grid for CirCor 2022 murmur detection.

Reads features.csv (output of extract_features.py). Runs:
  - 5-fold stratified patient-level cross-validation
  - 5 models: Lasso-LR, K-NN, SVC-RBF, XGBoost, GPC
  - 5 progressive feature sets:
        (a) MFCCs only
        (b) + spectral
        (c) + wavelet
        (d) + fractal/complexity
        (e) + heart-cycle temporal (all)
  - Metrics: accuracy, F1, AUC-ROC, sensitivity, specificity
  - GPC calibration: Expected Calibration Error (ECE), Brier score

Excludes the 'Unknown' class from training/test. Retains Unknown recordings
for a later UQ-on-Unknown sanity check (run separately).

Outputs:
  - results_ablation.csv  (one row per model×feature-set×fold)
  - results_summary.csv   (aggregated mean ± std)
  - feature_importance.csv (Lasso L1 + XGBoost gain, on full feature set)
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =====================================================================
# Feature-family definitions
# =====================================================================

def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Partition the feature columns into 5 families by naming convention."""
    all_feats = [c for c in df.columns if c not in {
        "patient_id", "location", "murmur", "outcome", "age", "sex"
    }]
    groups = {
        "mfcc":     [c for c in all_feats if c.startswith("mfcc_")],
        "spectral": [c for c in all_feats if c.startswith("spec_") or c in {"zcr", "rms"}],
        "wavelet":  [c for c in all_feats if c.startswith("wavelet_")],
        "fractal":  [c for c in all_feats if c in {"fractal_dfa", "approx_entropy", "sample_entropy", "hurst"}],
        "heart_cycle": [c for c in all_feats if c in {
            "s1_mean_amp", "s2_mean_amp", "s1s2_ratio",
            "systole_dur_mean", "diastole_dur_mean", "sd_ratio"
        }],
        "quality": [c for c in all_feats if c in {"snr_db", "clip_rate"}],
    }
    # sanity report
    assigned = set()
    for v in groups.values():
        assigned.update(v)
    missing = set(all_feats) - assigned
    if missing:
        print(f"[warn] features not assigned to any group: {missing}")
    return groups


def feature_progression(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return the 5 progressive feature sets."""
    # quality features always included (they're for conditioning, not in the ablation story)
    mfcc = groups["mfcc"] + groups["quality"]
    return {
        "(a) MFCC only":     mfcc,
        "(b) +spectral":     mfcc + groups["spectral"],
        "(c) +wavelet":      mfcc + groups["spectral"] + groups["wavelet"],
        "(d) +fractal":      mfcc + groups["spectral"] + groups["wavelet"] + groups["fractal"],
        "(e) +heart-cycle":  mfcc + groups["spectral"] + groups["wavelet"] + groups["fractal"] + groups["heart_cycle"],
    }


# =====================================================================
# Metrics
# =====================================================================

def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE over binned predicted probabilities."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def evaluate(y_true, y_pred, y_prob) -> dict:
    """Accuracy, F1, AUC, sens, spec, ECE, Brier."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "auc":      float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "brier":    float(brier_score_loss(y_true, y_prob)),
        "ece":      expected_calibration_error(np.asarray(y_true), np.asarray(y_prob)),
    }


# =====================================================================
# Models
# =====================================================================

def get_models(class_weight: dict) -> dict:
    """Instantiate the 5 models. Keep hyperparameters fixed (no nested CV here
    to save time; hyperparameter sensitivity noted as limitation).

    Note: KNeighborsClassifier does not accept class_weight in scikit-learn.
    We mitigate imbalance via weights='distance' (closer neighbours vote more)
    so dense regions of the minority class still contribute proportionally.
    """
    return {
        "Lasso-LR": LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, max_iter=1000,
            class_weight=class_weight, random_state=0,
        ),
        "K-NN": KNeighborsClassifier(
            n_neighbors=15, weights="distance", metric="minkowski", p=2,
            n_jobs=-1,
        ),
        "SVC-RBF": SVC(
            kernel="rbf", C=10.0, gamma="scale", probability=True,
            class_weight=class_weight, random_state=0,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=class_weight.get(1, 1.0) / class_weight.get(0, 1.0),
            use_label_encoder=False, eval_metric="logloss",
            random_state=0, n_jobs=-1,
        ),
        "GPC": GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
            random_state=0, n_jobs=-1, max_iter_predict=100,
        ),
    }


# =====================================================================
# Main driver
# =====================================================================

def run(args):
    df = pd.read_csv(args.features_csv)
    print(f"Loaded {len(df)} recordings x {len(df.columns)} cols")

    # Keep only Present/Absent for primary analysis
    df = df[df["murmur"].isin(["Present", "Absent"])].copy()
    df["y"] = (df["murmur"] == "Present").astype(int)
    print(f"After dropping Unknown: {len(df)} recordings")
    print(f"Class balance: Absent={sum(df['y']==0)}, Present={sum(df['y']==1)} (ratio {(df['y']==0).sum() / max(1,(df['y']==1).sum()):.2f}:1)")

    # Drop rows with any NaN in features (heart-cycle features may be NaN if TSV missing)
    feat_cols_all = [c for c in df.columns if c not in {
        "patient_id", "location", "murmur", "outcome", "age", "sex", "y"
    }]
    n_before = len(df)
    df = df.dropna(subset=feat_cols_all).reset_index(drop=True)
    print(f"After dropping NaN rows: {len(df)} ({n_before - len(df)} dropped)")

    groups_map = feature_groups(df)
    feature_sets = feature_progression(groups_map)
    print("\nFeature sets:")
    for name, cols in feature_sets.items():
        print(f"  {name}: {len(cols)} features")

    # Class weighting for imbalance
    n_neg = int((df["y"] == 0).sum())
    n_pos = int((df["y"] == 1).sum())
    class_weight = {0: 1.0, 1: n_neg / n_pos}

    # Patient-level stratified 5-fold CV
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(skf.split(df.index, df["y"], df["patient_id"]))

    rows = []
    for fs_name, fs_cols in feature_sets.items():
        print(f"\n=== Feature set: {fs_name} ({len(fs_cols)} features) ===")
        X = df[fs_cols].values
        y = df["y"].values

        for model_name, model_factory in [
            ("Lasso-LR", lambda: get_models(class_weight)["Lasso-LR"]),
            ("K-NN",     lambda: get_models(class_weight)["K-NN"]),
            ("SVC-RBF",  lambda: get_models(class_weight)["SVC-RBF"]),
            ("XGBoost",  lambda: get_models(class_weight)["XGBoost"]),
            ("GPC",      lambda: get_models(class_weight)["GPC"]),
        ]:
            t0 = time.time()
            fold_metrics = []
            for fold_i, (tr_idx, te_idx) in enumerate(folds):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                # XGBoost and Lasso/SVC/GPC all accept scaled; tree models don't need but benign
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                model = model_factory()
                # XGBoost benefits from unscaled input
                if model_name == "XGBoost":
                    X_tr_m, X_te_m = X_tr, X_te
                else:
                    X_tr_m, X_te_m = X_tr_s, X_te_s
                try:
                    model.fit(X_tr_m, y_tr)
                    y_prob = model.predict_proba(X_te_m)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                    m = evaluate(y_te, y_pred, y_prob)
                except Exception as e:
                    print(f"  [warn] {model_name} fold {fold_i} failed: {e}")
                    m = {"accuracy": np.nan, "f1": np.nan, "auc": np.nan,
                         "sensitivity": np.nan, "specificity": np.nan,
                         "brier": np.nan, "ece": np.nan}
                m.update({"model": model_name, "feature_set": fs_name, "fold": fold_i})
                fold_metrics.append(m)
                rows.append(m)
            elapsed = time.time() - t0
            accs = [m["accuracy"] for m in fold_metrics if not np.isnan(m["accuracy"])]
            f1s = [m["f1"] for m in fold_metrics if not np.isnan(m["f1"])]
            print(f"  {model_name:8s}  acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
                  f"f1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  ({elapsed:.1f}s)")

    # Save full results
    res = pd.DataFrame(rows)
    res.to_csv(args.out_ablation, index=False)
    print(f"\n[ok] Wrote fold-level ablation results to {args.out_ablation}")

    # Summary (mean ± std across folds)
    summary = (
        res.groupby(["feature_set", "model"])
           .agg(["mean", "std"])
           .round(4)
    )
    summary.to_csv(args.out_summary)
    print(f"[ok] Wrote summary to {args.out_summary}")

    # Feature importance on full feature set
    print("\n=== Feature importance (full feature set, trained on all data) ===")
    full_cols = feature_sets[list(feature_sets.keys())[-1]]
    X_full, y_full = df[full_cols].values, df["y"].values
    scaler = StandardScaler()
    X_full_s = scaler.fit_transform(X_full)

    importance_rows = []

    # Lasso L1 sparsity
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                               max_iter=1000, class_weight=class_weight, random_state=0)
    lasso.fit(X_full_s, y_full)
    for c, coef in zip(full_cols, lasso.coef_[0]):
        importance_rows.append({"feature": c, "lasso_coef": float(coef),
                                "xgb_gain": np.nan})

    # XGBoost gain
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                       scale_pos_weight=n_neg / n_pos,
                       use_label_encoder=False, eval_metric="logloss",
                       random_state=0, n_jobs=-1)
    xgb.fit(X_full, y_full)
    gains = xgb.get_booster().get_score(importance_type="gain")
    idx_to_name = {f"f{i}": c for i, c in enumerate(full_cols)}
    for row in importance_rows:
        for fkey, g in gains.items():
            if idx_to_name.get(fkey) == row["feature"]:
                row["xgb_gain"] = float(g)
                break

    imp_df = pd.DataFrame(importance_rows).sort_values(
        by="xgb_gain", ascending=False, na_position="last"
    )
    imp_df.to_csv(args.out_importance, index=False)
    print(f"[ok] Wrote feature importance to {args.out_importance}")

    # Print top 10 by XGBoost gain
    print("\nTop 10 features by XGBoost gain:")
    top10 = imp_df.head(10)
    for _, r in top10.iterrows():
        print(f"  {r['feature']:25s}  gain={r['xgb_gain']:.2f}  lasso_coef={r['lasso_coef']:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, default=Path("./features.csv"))
    parser.add_argument("--out-ablation", type=Path, default=Path("./results_ablation.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("./results_summary.csv"))
    parser.add_argument("--out-importance", type=Path, default=Path("./feature_importance.csv"))
    args = parser.parse_args()
    run(args)
