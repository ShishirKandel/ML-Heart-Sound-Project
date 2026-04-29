"""
Uncertainty-on-Unknown validation (validates Hypothesis H4 second half).

H4 second-half claim: GPC's predictive variance/entropy on the held-out
'Unknown' subset should be significantly higher than on Present/Absent
recordings - i.e. the model knows when it doesn't know.

Method
------
1. Load features.csv
2. Train GPC on Present+Absent training data only (uses 5-fold patient-level CV
   so we have out-of-fold predictions for Present/Absent recordings)
3. Then train GPC on ALL Present+Absent data and predict on Unknown
4. Compare predictive entropy distributions across the three subsets:
        H(p) = -p log p - (1-p) log(1-p)
5. Wilcoxon rank-sum tests:
        Unknown vs Absent
        Unknown vs Present
6. Plot histograms

Output:
  uq_results.csv         per-recording entropy + class
  plots/uq_hist.png      entropy histogram by class
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="talk")

OUT_DIR = Path("./plots")
OUT_DIR.mkdir(exist_ok=True)


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    df = pd.read_csv("./features.csv")
    feat_cols = [c for c in df.columns if c not in {
        "patient_id", "location", "murmur", "outcome", "age", "sex"
    }]
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    print(f"Loaded {len(df)} recordings")

    train_df = df[df["murmur"].isin(["Present", "Absent"])].copy().reset_index(drop=True)
    train_df["y"] = (train_df["murmur"] == "Present").astype(int)
    unknown_df = df[df["murmur"] == "Unknown"].copy().reset_index(drop=True)
    print(f"Train (Present+Absent): {len(train_df)}; Unknown: {len(unknown_df)}")

    X_train = train_df[feat_cols].values
    y_train = train_df["y"].values
    X_unknown = unknown_df[feat_cols].values

    # Out-of-fold predictions for Present+Absent (so the predictions are not on training data)
    print("\nGetting GPC out-of-fold predictions on Present/Absent...")
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof_prob = np.zeros(len(train_df))
    for fold_i, (tr, te) in enumerate(skf.split(train_df.index, train_df["y"], train_df["patient_id"])):
        scaler = StandardScaler().fit(X_train[tr])
        Xtr = scaler.transform(X_train[tr])
        Xte = scaler.transform(X_train[te])
        gpc = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
            random_state=0, n_jobs=-1, max_iter_predict=100,
        )
        gpc.fit(Xtr, y_train[tr])
        oof_prob[te] = gpc.predict_proba(Xte)[:, 1]
        print(f"  fold {fold_i+1}/5 done")

    train_df["entropy"] = entropy(oof_prob)
    train_df["prob_present"] = oof_prob

    # Train GPC on full Present+Absent data, predict on Unknown
    print("\nTraining GPC on all Present+Absent, predicting Unknown...")
    scaler_full = StandardScaler().fit(X_train)
    gpc_full = GaussianProcessClassifier(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
        random_state=0, n_jobs=-1, max_iter_predict=100,
    )
    gpc_full.fit(scaler_full.transform(X_train), y_train)
    unk_prob = gpc_full.predict_proba(scaler_full.transform(X_unknown))[:, 1]
    unknown_df["entropy"] = entropy(unk_prob)
    unknown_df["prob_present"] = unk_prob

    # Combine for output
    out = pd.concat([
        train_df[["patient_id", "location", "murmur", "prob_present", "entropy"]],
        unknown_df[["patient_id", "location", "murmur", "prob_present", "entropy"]],
    ], ignore_index=True)
    out.to_csv("./uq_results.csv", index=False)

    # Wilcoxon (Mann-Whitney U) tests
    ent_abs = train_df.loc[train_df["y"] == 0, "entropy"].values
    ent_pres = train_df.loc[train_df["y"] == 1, "entropy"].values
    ent_unk = unknown_df["entropy"].values

    u1, p1 = mannwhitneyu(ent_unk, ent_abs, alternative="greater")
    u2, p2 = mannwhitneyu(ent_unk, ent_pres, alternative="greater")
    print(f"\nEntropy means: Absent={ent_abs.mean():.4f}, Present={ent_pres.mean():.4f}, Unknown={ent_unk.mean():.4f}")
    print(f"Mann-Whitney U (Unknown > Absent):  U={u1:.0f}, p={p1:.4g}")
    print(f"Mann-Whitney U (Unknown > Present): U={u2:.0f}, p={p2:.4g}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(ent_abs,  color="C0", alpha=0.6, label=f"Absent (n={len(ent_abs)}, mean={ent_abs.mean():.3f})", stat="density", ax=ax, bins=30)
    sns.histplot(ent_pres, color="C2", alpha=0.6, label=f"Present (n={len(ent_pres)}, mean={ent_pres.mean():.3f})", stat="density", ax=ax, bins=30)
    sns.histplot(ent_unk,  color="C3", alpha=0.6, label=f"Unknown (n={len(ent_unk)}, mean={ent_unk.mean():.3f})", stat="density", ax=ax, bins=30)
    ax.set_title(f"GPC predictive entropy by murmur class\n(Unknown vs Absent: p={p1:.3g}; Unknown vs Present: p={p2:.3g})")
    ax.set_xlabel("Predictive entropy H(p)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "uq_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] Saved uq_results.csv and {OUT_DIR}/uq_hist.png")


if __name__ == "__main__":
    main()
