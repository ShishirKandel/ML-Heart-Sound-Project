"""
Generate publication-quality plots for the Results section.

Reads results_ablation.csv and feature_importance.csv (outputs of run_models.py).
Produces:
  plots/ablation_heatmap_acc.png
  plots/ablation_heatmap_f1.png
  plots/ablation_heatmap_auc.png
  plots/calibration_full.png
  plots/feature_importance_top15.png
  plots/metric_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

OUT_DIR = Path("./plots")
OUT_DIR.mkdir(exist_ok=True)


def heatmap(df: pd.DataFrame, metric: str, title: str, fname: str, fmt: str = ".3f", cmap: str = "viridis") -> None:
    pivot = df.pivot_table(index="model", columns="feature_set", values=metric, aggfunc="mean")
    # Order columns by alphabetical (a)..(e)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar_kws={"label": metric}, vmin=pivot.values.min(), vmax=pivot.values.max())
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metric_comparison(df: pd.DataFrame) -> None:
    """Bar chart: 5 metrics × 5 models, on the full feature set."""
    full = df[df["feature_set"] == "(e) +heart-cycle"].copy()
    metrics = ["accuracy", "f1", "auc", "sensitivity", "specificity"]
    agg = full.groupby("model")[metrics].mean().round(4)
    # Order: by AUC descending
    agg = agg.sort_values("auc", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    agg.plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
    ax.set_title("Model performance on full 48-feature set (mean over 5 patient-level folds)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.legend(loc="upper right", ncol=5, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibration_chart(df: pd.DataFrame) -> None:
    """Show ECE and Brier per model on full feature set."""
    full = df[df["feature_set"] == "(e) +heart-cycle"].copy()
    agg = full.groupby("model")[["ece", "brier"]].mean().sort_values("ece")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    agg["ece"].plot(kind="bar", ax=axes[0], color="C3", edgecolor="black")
    axes[0].set_title("Expected Calibration Error (lower is better)")
    axes[0].set_ylabel("ECE")
    axes[0].axhline(0, color="black", lw=0.5)
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%.3f", fontsize=10, padding=2)

    agg["brier"].plot(kind="bar", ax=axes[1], color="C0", edgecolor="black")
    axes[1].set_title("Brier Score (lower is better)")
    axes[1].set_ylabel("Brier")
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")
    for c in axes[1].containers:
        axes[1].bar_label(c, fmt="%.3f", fontsize=10, padding=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_full.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def feature_importance_chart(imp_csv: Path) -> None:
    if not imp_csv.exists():
        print(f"[warn] no {imp_csv}, skipping feature-importance plot")
        return
    imp = pd.read_csv(imp_csv)
    top = imp.dropna(subset=["xgb_gain"]).nlargest(15, "xgb_gain").iloc[::-1]
    # Family annotation by name
    def family(f: str) -> str:
        if f.startswith("mfcc"):       return "MFCC"
        if f.startswith("spec_") or f in ("zcr", "rms"): return "Spectral"
        if f.startswith("wavelet"):    return "Wavelet"
        if f in ("fractal_dfa", "approx_entropy", "sample_entropy", "hurst"): return "Fractal"
        if f in ("s1_mean_amp", "s2_mean_amp", "s1s2_ratio",
                 "systole_dur_mean", "diastole_dur_mean", "sd_ratio"):        return "Heart-cycle"
        if f in ("snr_db", "clip_rate"): return "Quality"
        return "Other"
    palette = {
        "MFCC": "#4C72B0", "Spectral": "#DD8452", "Wavelet": "#55A467",
        "Fractal": "#C44E52", "Heart-cycle": "#8172B3", "Quality": "#937860", "Other": "#888888",
    }
    families = top["feature"].map(family)
    colors = [palette[f] for f in families]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["xgb_gain"], color=colors, edgecolor="black")
    ax.set_title("Top 15 features by XGBoost gain (full 48-feature set)")
    ax.set_xlabel("XGBoost gain")
    # legend
    handles = [plt.Rectangle((0,0),1,1, color=palette[f]) for f in palette]
    ax.legend(handles, list(palette.keys()), loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_top15.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    res = pd.read_csv("./results_ablation.csv")

    heatmap(res, "accuracy",    "Accuracy across 5 models × 5 feature sets",
            "ablation_heatmap_acc.png")
    heatmap(res, "f1",          "F1 (Murmur-Present) across 5 models × 5 feature sets",
            "ablation_heatmap_f1.png", cmap="rocket_r")
    heatmap(res, "auc",         "AUC-ROC across 5 models × 5 feature sets",
            "ablation_heatmap_auc.png", cmap="mako")
    metric_comparison(res)
    calibration_chart(res)
    feature_importance_chart(Path("./feature_importance.csv"))
    print(f"[ok] All plots in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
