"""
Sensitivity analysis: SMOTE oversampling vs. class weights.

Re-runs the full feature set (e) on all 5 models with three imbalance strategies:
  A. class_weight (current default)
  B. SMOTE (synthetic oversampling, applied INSIDE each CV fold to avoid leakage)
  C. Random oversampling (RandomOverSampler, no synthesis)

Note: SMOTE is applied AFTER the train/test split per fold, so test data is
NEVER contaminated with synthetic samples. This is the standard correct usage.

Output:
  results_smote.csv
  plots/smote_comparison.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", context="talk")
OUT_DIR = Path("./plots")
OUT_DIR.mkdir(exist_ok=True)


def get_models(class_weight, use_class_weight: bool):
    """Construct models, optionally disabling class_weight (for SMOTE/oversample runs)."""
    cw = class_weight if use_class_weight else None
    spw = (class_weight[1] / class_weight[0]) if use_class_weight else 1.0
    return {
        "Lasso-LR": LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                         max_iter=1000, class_weight=cw, random_state=0),
        "K-NN":     KNeighborsClassifier(n_neighbors=15, weights="distance",
                                          metric="minkowski", p=2, n_jobs=-1),
        "SVC-RBF":  SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                         class_weight=cw, random_state=0),
        "XGBoost":  XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    subsample=0.9, colsample_bytree=0.9,
                                    scale_pos_weight=spw,
                                    use_label_encoder=False, eval_metric="logloss",
                                    random_state=0, n_jobs=-1),
        "GPC":      GaussianProcessClassifier(
                        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                        random_state=0, n_jobs=-1, max_iter_predict=100),
    }


def main():
    df = pd.read_csv("./features.csv")
    df = df[df["murmur"].isin(["Present", "Absent"])].copy().reset_index(drop=True)
    df["y"] = (df["murmur"] == "Present").astype(int)
    feat_cols = [c for c in df.columns if c not in {
        "patient_id", "location", "murmur", "outcome", "age", "sex", "y"
    }]
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    X = df[feat_cols].values
    y = df["y"].values

    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    cw = {0: 1.0, 1: n_neg / n_pos}
    print(f"Data: {len(df)} recordings, ratio {n_neg/n_pos:.2f}:1")

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(skf.split(df.index, y, df["patient_id"]))

    rows = []
    strategies = [
        ("class_weight", lambda Xtr, ytr: (Xtr, ytr), True),
        ("SMOTE",        lambda Xtr, ytr: SMOTE(random_state=0).fit_resample(Xtr, ytr), False),
        ("RandomOverSample", lambda Xtr, ytr: RandomOverSampler(random_state=0).fit_resample(Xtr, ytr), False),
    ]

    for strat_name, resample_fn, use_cw in strategies:
        print(f"\n=== Strategy: {strat_name} ===")
        models = get_models(cw, use_cw)
        for model_name, model in models.items():
            fold_metrics = []
            for fold_i, (tr, te) in enumerate(folds):
                scaler = StandardScaler().fit(X[tr])
                Xtr_s = scaler.transform(X[tr])
                Xte_s = scaler.transform(X[te])
                Xtr_use, ytr_use = resample_fn(Xtr_s, y[tr])
                # Refit a fresh model each fold (sklearn.clone handles nested params)
                m = clone(model)
                Xfit = X[tr] if model_name == "XGBoost" else Xtr_use
                yfit = y[tr] if model_name == "XGBoost" else ytr_use
                if model_name == "XGBoost" and strat_name != "class_weight":
                    # For XGBoost we apply resampling on raw X
                    Xfit, yfit = resample_fn(X[tr], y[tr])
                Xte_use = X[te] if model_name == "XGBoost" else Xte_s
                m.fit(Xfit, yfit)
                p = m.predict_proba(Xte_use)[:, 1]
                pred = (p >= 0.5).astype(int)
                acc = accuracy_score(y[te], pred)
                f1 = f1_score(y[te], pred, zero_division=0)
                auc = roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan
                fold_metrics.append({"acc": acc, "f1": f1, "auc": auc})
            agg = pd.DataFrame(fold_metrics).mean()
            std = pd.DataFrame(fold_metrics).std()
            row = {
                "strategy": strat_name, "model": model_name,
                "accuracy_mean": agg["acc"], "accuracy_std": std["acc"],
                "f1_mean": agg["f1"], "f1_std": std["f1"],
                "auc_mean": agg["auc"], "auc_std": std["auc"],
            }
            rows.append(row)
            print(f"  {model_name:8s}  acc={agg['acc']:.3f}  f1={agg['f1']:.3f}  auc={agg['auc']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv("./results_smote.csv", index=False)

    # Plot AUC and F1 across strategies
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    pivot_auc = res.pivot(index="model", columns="strategy", values="auc_mean")
    pivot_f1 = res.pivot(index="model", columns="strategy", values="f1_mean")
    pivot_auc.plot(kind="bar", ax=axes[0], colormap="viridis", width=0.78, edgecolor="black")
    axes[0].set_title("AUC by imbalance strategy")
    axes[0].set_ylabel("AUC")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0.6, 0.85)
    axes[0].legend(title="strategy", fontsize=10)
    plt.setp(axes[0].get_xticklabels(), rotation=15, ha="right")

    pivot_f1.plot(kind="bar", ax=axes[1], colormap="viridis", width=0.78, edgecolor="black")
    axes[1].set_title("F1 (Murmur-Present) by imbalance strategy")
    axes[1].set_ylabel("F1")
    axes[1].set_xlabel("")
    axes[1].legend(title="strategy", fontsize=10)
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smote_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] results_smote.csv + smote_comparison.png saved")


if __name__ == "__main__":
    main()
