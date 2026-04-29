"""
Predict heart-murmur Present/Absent on a new WAV recording.

Usage
-----
    python predict.py path/to/recording.wav
    python predict.py path/to/recording.wav --tsv path/to/segmentation.tsv
    python predict.py path/to/recording.wav --model gpc

If --tsv is omitted, heart-cycle temporal features are set to median of
training distribution (graceful fallback). For best accuracy provide the
TSV segmentation file (in CirCor v1.0.3 .tsv format).

Outputs predicted class and per-model probability for Murmur Present.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

# Reuse the feature-extraction functions from extract_features.py
sys.path.insert(0, str(Path(__file__).parent))
from extract_features import (
    preprocess,
    mfcc_features,
    spectral_features,
    wavelet_features,
    fractal_features,
    heart_cycle_features,
    quality_features,
)

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent / "models"
ALL_MODELS = ["lasso_lr", "knn", "svc_rbf", "xgboost", "gpc"]


def extract_one(wav_path: Path, tsv_path: Path | None) -> pd.Series:
    x_raw, sr = sf.read(str(wav_path), dtype="float32")
    if x_raw.ndim > 1:
        x_raw = x_raw.mean(axis=1)
    x_clean, sr_out = preprocess(x_raw, sr)
    feat = {}
    feat.update(mfcc_features(x_clean, sr_out))
    feat.update(spectral_features(x_clean, sr_out))
    feat.update(wavelet_features(x_clean))
    feat.update(fractal_features(x_clean))
    feat.update(heart_cycle_features(tsv_path))
    feat.update(quality_features(x_raw, x_clean))
    return pd.Series(feat)


def main():
    parser = argparse.ArgumentParser(description="Predict heart-murmur on a WAV file")
    parser.add_argument("wav_path", type=Path, help="Path to WAV file")
    parser.add_argument("--tsv", type=Path, default=None, help="Optional .tsv S1/S2 segmentation file")
    parser.add_argument("--model", type=str, default="all",
                        help=f"One of {ALL_MODELS} or 'all' (default)")
    args = parser.parse_args()

    if not args.wav_path.exists():
        sys.exit(f"[error] WAV file not found: {args.wav_path}")
    if not MODELS_DIR.exists():
        sys.exit(f"[error] Models dir missing. Run `python train_final.py` first.")

    # Load feature column order
    with open(MODELS_DIR / "feature_columns.json") as f:
        meta = json.load(f)
    feat_cols = meta["feature_columns"]

    # Load scaler
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    # Extract features
    print(f"[info] Extracting features from {args.wav_path.name}...")
    feat_series = extract_one(args.wav_path, args.tsv)

    # Fill missing heart-cycle features with 0 (median proxy) if no TSV
    for col in feat_cols:
        if col not in feat_series.index or pd.isna(feat_series[col]):
            feat_series[col] = 0.0
    X = feat_series.reindex(feat_cols).values.reshape(1, -1).astype(np.float32)
    Xs = scaler.transform(X)

    # Pick models
    chosen = ALL_MODELS if args.model == "all" else [args.model]
    print(f"\n=== Predictions ({args.wav_path.name}) ===")
    print(f"{'Model':10s}  {'P(Present)':>12s}  {'Prediction':>12s}")
    print("-" * 40)
    for name in chosen:
        path = MODELS_DIR / f"{name}.pkl"
        if not path.exists():
            print(f"{name:10s}  [model file missing]")
            continue
        m = joblib.load(path)
        # XGBoost gets unscaled; others scaled
        Xuse = X if name == "xgboost" else Xs
        try:
            p = float(m.predict_proba(Xuse)[0, 1])
            label = "Present" if p >= 0.5 else "Absent "
            print(f"{name:10s}  {p:>12.4f}  {label:>12s}")
        except Exception as e:
            print(f"{name:10s}  [error: {e}]")


if __name__ == "__main__":
    main()
