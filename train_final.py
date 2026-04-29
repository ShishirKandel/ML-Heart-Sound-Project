"""
Train and save final production models on the full feature set.

Reads features.csv, fits each of the 5 classifiers on ALL Present/Absent
recordings, and saves them as .pkl files for use by predict.py.

Output:
  models/lasso_lr.pkl
  models/knn.pkl
  models/svc_rbf.pkl
  models/xgboost.pkl
  models/gpc.pkl
  models/scaler.pkl
  models/feature_columns.json   (column order required for prediction)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

OUT = Path("./models")
OUT.mkdir(exist_ok=True)


def main():
    df = pd.read_csv("./features.csv")
    df = df[df["murmur"].isin(["Present", "Absent"])].copy()
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
    print(f"Training on {len(df)} recordings ({n_neg} Absent, {n_pos} Present, ratio {n_neg/n_pos:.2f}:1)")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    models = {
        "lasso_lr": LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                        max_iter=1000, class_weight=cw, random_state=0),
        "knn":      KNeighborsClassifier(n_neighbors=15, weights="distance",
                                          metric="minkowski", p=2, n_jobs=-1),
        "svc_rbf":  SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                        class_weight=cw, random_state=0),
        "xgboost":  XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.9, colsample_bytree=0.9,
                                   scale_pos_weight=n_neg/n_pos,
                                   use_label_encoder=False, eval_metric="logloss",
                                   random_state=0, n_jobs=-1),
        "gpc":      GaussianProcessClassifier(
                        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                        random_state=0, n_jobs=-1, max_iter_predict=100),
    }

    for name, model in models.items():
        print(f"  training {name}...", flush=True)
        # XGBoost gets unscaled (scale-invariant); others get scaled
        Xfit = X if name == "xgboost" else Xs
        model.fit(Xfit, y)
        joblib.dump(model, OUT / f"{name}.pkl")
        print(f"    saved {OUT / (name + '.pkl')}")

    joblib.dump(scaler, OUT / "scaler.pkl")
    with open(OUT / "feature_columns.json", "w") as f:
        json.dump({"feature_columns": feat_cols, "n_neg": n_neg, "n_pos": n_pos}, f, indent=2)

    print(f"\n[ok] All 5 models, scaler, and feature_columns saved in {OUT.resolve()}")


if __name__ == "__main__":
    main()
