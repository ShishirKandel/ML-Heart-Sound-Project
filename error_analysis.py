"""
Error analysis: identify Present recordings that all 5 models miss
(the hard-negative / hard-positive cases).

Reads features.csv. Refits all 5 classifiers via 5-fold patient-level CV with
out-of-fold predictions, then asks:
  - How many Present recordings does each model miss?
  - How many Present recordings do ALL 5 models miss?
  - Are these missed cases shorter, lower SNR, or from specific locations?

Output:
  error_analysis.csv  (one row per Present recording with predictions per model)
  plots/error_analysis_locations.png
  plots/error_analysis_quality.png
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", context="talk")

OUT_DIR = Path("./plots")
OUT_DIR.mkdir(exist_ok=True)


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

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    # OOF predictions per model
    oof = pd.DataFrame(index=df.index)
    oof["y"] = y
    oof["patient_id"] = df["patient_id"]
    oof["location"] = df["location"]

    for name, mk in [
        ("lasso_lr", lambda: LogisticRegression(penalty="l1", solver="liblinear",
                                                  C=1.0, max_iter=1000,
                                                  class_weight=cw, random_state=0)),
        ("knn", lambda: KNeighborsClassifier(n_neighbors=15, weights="distance",
                                               metric="minkowski", p=2, n_jobs=-1)),
        ("svc_rbf", lambda: SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                                  class_weight=cw, random_state=0)),
        ("xgboost", lambda: XGBClassifier(n_estimators=300, max_depth=5,
                                            learning_rate=0.05, subsample=0.9,
                                            colsample_bytree=0.9,
                                            scale_pos_weight=n_neg/n_pos,
                                            use_label_encoder=False,
                                            eval_metric="logloss",
                                            random_state=0, n_jobs=-1)),
        ("gpc", lambda: GaussianProcessClassifier(
                        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                        random_state=0, n_jobs=-1, max_iter_predict=100)),
    ]:
        print(f"  OOF predictions: {name}", flush=True)
        oof[f"prob_{name}"] = 0.0
        for fold_i, (tr, te) in enumerate(skf.split(df.index, y, df["patient_id"])):
            scaler = StandardScaler().fit(X[tr])
            Xtr_s = scaler.transform(X[tr])
            Xte_s = scaler.transform(X[te])
            model = mk()
            Xtr = X[tr] if name == "xgboost" else Xtr_s
            Xte = X[te] if name == "xgboost" else Xte_s
            model.fit(Xtr, y[tr])
            oof.loc[te, f"prob_{name}"] = model.predict_proba(Xte)[:, 1]
        oof[f"pred_{name}"] = (oof[f"prob_{name}"] >= 0.5).astype(int)
        oof[f"correct_{name}"] = (oof[f"pred_{name}"] == oof["y"]).astype(int)

    # Add quality features (SNR, clip rate, duration proxy)
    if "snr_db" in df.columns:
        oof["snr_db"] = df["snr_db"]
    if "clip_rate" in df.columns:
        oof["clip_rate"] = df["clip_rate"]

    # All-models-correct vs all-models-wrong on Present
    pres = oof[oof["y"] == 1].copy()
    pres["n_models_correct"] = pres[[c for c in pres.columns if c.startswith("correct_")]].sum(axis=1)
    print(f"\nPresent recordings: {len(pres)}")
    print("How many models correctly identified each Present recording:")
    print(pres["n_models_correct"].value_counts().sort_index().to_string())

    pres["all_miss"] = (pres["n_models_correct"] == 0).astype(int)
    pres["all_correct"] = (pres["n_models_correct"] == 5).astype(int)
    n_miss = int(pres["all_miss"].sum())
    n_corr = int(pres["all_correct"].sum())
    print(f"\nPresent recordings ALL 5 models missed:    {n_miss}")
    print(f"Present recordings ALL 5 models got right: {n_corr}")

    pres.to_csv("./error_analysis.csv", index=False)

    # Compare quality of all-miss vs all-correct
    if "snr_db" in pres.columns:
        miss = pres[pres["all_miss"] == 1]["snr_db"]
        corr = pres[pres["all_correct"] == 1]["snr_db"]
        print(f"\nSNR (dB) of all-miss Present:    median={miss.median():.2f}, n={len(miss)}")
        print(f"SNR (dB) of all-correct Present: median={corr.median():.2f}, n={len(corr)}")

    # Plot: location breakdown of all-miss vs all-correct
    if n_miss > 0 and n_corr > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        miss_loc = pres[pres["all_miss"] == 1]["location"].value_counts()
        corr_loc = pres[pres["all_correct"] == 1]["location"].value_counts()
        all_loc = sorted(set(miss_loc.index) | set(corr_loc.index))
        x = np.arange(len(all_loc))
        w = 0.35
        ax.bar(x - w/2, [miss_loc.get(l, 0) for l in all_loc], w, label=f"All-miss (n={n_miss})", color="C3")
        ax.bar(x + w/2, [corr_loc.get(l, 0) for l in all_loc], w, label=f"All-correct (n={n_corr})", color="C2")
        ax.set_xticks(x)
        ax.set_xticklabels(all_loc)
        ax.set_xlabel("Auscultation location")
        ax.set_ylabel("Recordings")
        ax.set_title("Where Present recordings are missed vs correctly classified by all 5 models")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "error_analysis_locations.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Plot: SNR / clip_rate distributions for all-miss vs all-correct
    if n_miss > 0 and n_corr > 0 and "snr_db" in pres.columns:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, col, title in zip(axes, ["snr_db", "clip_rate"],
                                   ["SNR estimate (dB)", "Clipping rate"]):
            data = pd.DataFrame({
                "value": list(pres.loc[pres["all_miss"] == 1, col].values) +
                         list(pres.loc[pres["all_correct"] == 1, col].values),
                "group": ["all-miss"] * n_miss + ["all-correct"] * n_corr,
            })
            sns.boxplot(data=data, x="group", y="value", ax=ax,
                        palette={"all-miss": "C3", "all-correct": "C2"})
            ax.set_title(title)
            ax.set_xlabel("")
        fig.suptitle("Quality features: missed vs correctly-classified Present recordings")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "error_analysis_quality.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\n[ok] error_analysis.csv + plots saved")


if __name__ == "__main__":
    main()
