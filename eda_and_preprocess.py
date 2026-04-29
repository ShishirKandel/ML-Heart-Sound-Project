"""
CirCor / PhysioNet 2022 — Exploratory Data Analysis + Preprocessing Prototype
=============================================================================

Run this once the CirCor dataset is downloaded locally. It will:

    1. Parse metadata (training_data.csv + per-patient .txt files)
    2. Report class balance, demographics, duration distribution,
       auscultation-location × class cross-tabs.
    3. Audit audio quality: SNR estimate, clipping rate, silence fraction,
       motion-artefact spike rate.
    4. Run the preprocessing pipeline on a sample of recordings and show
       before/after waveforms and spectrograms.
    5. Save all plots into ./eda_outputs/ and a markdown summary report.

Usage
-----
    # 1. Download the public training split (~560 MB, free, no login):
    #    https://physionet.org/content/circor-heart-sound/1.0.3/
    #    Unzip into ./data/circor/ so the structure is:
    #      ./data/circor/training_data.csv
    #      ./data/circor/training_data/<PATIENT_ID>_<LOC>.wav
    #      ./data/circor/training_data/<PATIENT_ID>_<LOC>.hea
    #      ./data/circor/training_data/<PATIENT_ID>_<LOC>.tsv
    #      ./data/circor/training_data/<PATIENT_ID>.txt
    #
    # 2. Install deps:
    #    pip install numpy pandas matplotlib seaborn scipy soundfile librosa pywavelets tqdm
    #
    # 3. Run:
    #    python eda_and_preprocess.py
    #
    # To run only metadata analysis (fast, no audio loading):
    #    python eda_and_preprocess.py --metadata-only
    #
    # To sample fewer recordings for the audio quality audit:
    #    python eda_and_preprocess.py --audio-sample 100

The script is safe to interrupt and re-run; plots are regenerated.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", context="talk")

# Lazy-imported (heavy): soundfile, librosa, pywt
# They're only loaded inside audio functions so --metadata-only stays light.


# =====================================================================
# Configuration
# =====================================================================

DEFAULT_DATA_DIR = Path("./the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data")
DEFAULT_METADATA_CSV = Path("./the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data.csv")
OUTPUT_DIR = Path("./eda_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 4000
BANDPASS_LOW = 25
BANDPASS_HIGH = 450
BANDPASS_ORDER = 4
SPIKE_THRESHOLD = 3.0  # 3 * median absolute value
CLIPPING_THRESHOLD = 0.99  # |x| above this fraction of max is "clipped"
SILENCE_DB_THRESHOLD = -40  # dB below peak considered silence


# =====================================================================
# Metadata loading and EDA
# =====================================================================

def load_metadata(csv_path: Path) -> pd.DataFrame:
    """Load the combined training_data.csv and add a few derived fields."""
    if not csv_path.exists():
        sys.exit(
            f"\n[error] Metadata CSV not found at {csv_path}.\n"
            "Download the CirCor v1.0.3 training split from\n"
            "  https://physionet.org/content/circor-heart-sound/1.0.3/\n"
            "and extract so that training_data.csv lives at the expected path."
        )
    df = pd.read_csv(csv_path)
    # Normalise column names we rely on (CirCor uses slightly different headers
    # across versions; fail early if core columns are missing).
    required = ["Patient ID", "Murmur", "Outcome", "Age", "Sex", "Height", "Weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[error] training_data.csv is missing columns: {missing}")
    return df


def summarise_metadata(df: pd.DataFrame) -> None:
    """Print a terse metadata summary, also save a markdown report."""
    lines: list[str] = []

    def add(line: str) -> None:
        print(line)
        lines.append(line)

    add("# CirCor 2022 — Metadata EDA summary")
    add("")
    add(f"Total patients in public training split: {len(df)}")
    add("")

    add("## Murmur class balance (patient-level)")
    add(df["Murmur"].value_counts(dropna=False).to_string())
    add("")

    add("## Outcome class balance (patient-level)")
    add(df["Outcome"].value_counts(dropna=False).to_string())
    add("")

    add("## Murmur × Outcome cross-tab")
    add(pd.crosstab(df["Murmur"], df["Outcome"], margins=True).to_string())
    add("")

    add("## Sex distribution")
    add(df["Sex"].value_counts(dropna=False).to_string())
    add("")

    add("## Age category distribution")
    add(df["Age"].value_counts(dropna=False).to_string())
    add("")

    # Basic numeric stats
    numeric_cols = [c for c in ["Height", "Weight"] if c in df.columns]
    if numeric_cols:
        add("## Height / Weight summary")
        add(df[numeric_cols].describe().to_string())
        add("")

    # Plots
    _plot_class_balance(df)
    _plot_murmur_by_age_sex(df)

    report_path = OUTPUT_DIR / "metadata_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ok] Metadata summary saved to {report_path}")


def _plot_class_balance(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["Murmur", "Outcome"]):
        counts = df[col].value_counts(dropna=False)
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"{col} class balance")
        ax.set_ylabel("patients")
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_balance.png", dpi=140)
    plt.close(fig)


def _plot_murmur_by_age_sex(df: pd.DataFrame) -> None:
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ct_age = pd.crosstab(df["Age"], df["Murmur"], normalize="index")
        ct_age.plot(kind="bar", stacked=True, ax=axes[0], colormap="viridis")
        axes[0].set_title("Murmur rate by age category")
        axes[0].set_ylabel("proportion")
        axes[0].tick_params(axis="x", rotation=30)

        ct_sex = pd.crosstab(df["Sex"], df["Murmur"], normalize="index")
        ct_sex.plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis")
        axes[1].set_title("Murmur rate by sex")
        axes[1].set_ylabel("proportion")
        axes[1].tick_params(axis="x", rotation=0)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "murmur_by_age_sex.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] age/sex plot skipped: {e}")


# =====================================================================
# Recording-level index
# =====================================================================

@dataclass
class Recording:
    patient_id: str
    location: str
    wav_path: Path
    tsv_path: Path | None
    murmur: str
    outcome: str


def build_recording_index(df: pd.DataFrame, data_dir: Path) -> list[Recording]:
    """One row per .wav file, linked back to patient metadata."""
    wav_files = sorted(data_dir.glob("*_*.wav"))
    if not wav_files:
        sys.exit(
            f"[error] No .wav files found under {data_dir}. Check your "
            "extraction path."
        )

    meta_by_pid = df.set_index("Patient ID")
    recordings: list[Recording] = []
    missing_meta = 0

    for wav in wav_files:
        stem = wav.stem  # e.g., "50001_AV"
        parts = stem.split("_", maxsplit=1)
        if len(parts) != 2:
            continue
        pid_raw, loc = parts
        try:
            pid = int(pid_raw)
        except ValueError:
            pid = pid_raw
        if pid not in meta_by_pid.index:
            missing_meta += 1
            continue
        row = meta_by_pid.loc[pid]
        tsv = wav.with_suffix(".tsv")
        recordings.append(
            Recording(
                patient_id=str(pid),
                location=loc,
                wav_path=wav,
                tsv_path=tsv if tsv.exists() else None,
                murmur=str(row["Murmur"]),
                outcome=str(row["Outcome"]),
            )
        )

    print(f"[ok] Indexed {len(recordings)} recordings; {missing_meta} had no metadata.")
    return recordings


def plot_recording_level_stats(recordings: list[Recording]) -> None:
    df = pd.DataFrame(
        {
            "location": [r.location for r in recordings],
            "murmur": [r.murmur for r in recordings],
            "outcome": [r.outcome for r in recordings],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ct = pd.crosstab(df["location"], df["murmur"])
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title("Murmur rate by auscultation location (recording-level)")
    ax.set_ylabel("recordings")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "murmur_by_location.png", dpi=140)
    plt.close(fig)
    ct.to_csv(OUTPUT_DIR / "murmur_by_location.csv")


# =====================================================================
# Audio preprocessing pipeline
# =====================================================================

def butter_bandpass(sr: int, low: float, high: float, order: int) -> tuple:
    nyq = 0.5 * sr
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def preprocess_audio(
    x: np.ndarray,
    sr: int,
    target_sr: int = TARGET_SR,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER,
    spike_mult: float = SPIKE_THRESHOLD,
    wavelet_denoise: bool = True,
) -> dict:
    """Full preprocessing pipeline. Returns a dict with each intermediate
    stage and diagnostic stats so we can plot/audit them."""
    import librosa
    import pywt

    stages: dict = {}
    stages["raw"] = x.copy()

    # Resample if needed
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    stages["resampled"] = x.copy()

    # DC removal
    x = x - np.mean(x)

    # Bandpass 25-450 Hz
    b, a = butter_bandpass(sr, low, high, order)
    x = signal.filtfilt(b, a, x)
    stages["bandpassed"] = x.copy()

    # Spike removal: samples above 3 * median |x| are motion artefacts
    med_abs = np.median(np.abs(x)) + 1e-12
    spike_mask = np.abs(x) > spike_mult * med_abs * 10  # 3 * 10 * median is a conservative spike gate
    x[spike_mask] = 0.0
    stages["despiked"] = x.copy()
    stages["n_spikes_removed"] = int(spike_mask.sum())

    # Wavelet denoising (Daubechies-4, soft threshold, universal)
    if wavelet_denoise:
        coeffs = pywt.wavedec(x, "db4", level=4)
        # Estimate sigma from finest detail coefficients (std. MAD estimator)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-12
        uthresh = sigma * np.sqrt(2.0 * np.log(len(x)))
        coeffs_th = [coeffs[0]] + [
            pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]
        ]
        x = pywt.waverec(coeffs_th, "db4")[: len(stages["despiked"])]
    stages["denoised"] = x.copy()

    # Peak-normalise to [-1, 1]
    peak = np.max(np.abs(x)) + 1e-12
    x = x / peak
    stages["normalised"] = x.copy()

    # Diagnostics
    stages["sr"] = sr
    stages["duration_s"] = len(x) / sr
    stages["clipping_rate"] = float(np.mean(np.abs(stages["raw"]) > CLIPPING_THRESHOLD))
    stages["snr_estimate_db"] = _snr_estimate(stages["raw"], stages["denoised"])
    stages["silence_fraction"] = _silence_fraction(stages["normalised"])
    return stages


def _snr_estimate(raw: np.ndarray, clean: np.ndarray) -> float:
    """Rough SNR: power of clean signal vs power of (raw - clean) residual.
    Not a true SNR but a useful monotonic quality proxy."""
    n = min(len(raw), len(clean))
    r = raw[:n] - np.mean(raw[:n])
    c = clean[:n] - np.mean(clean[:n])
    noise = r - c * (np.std(r) / (np.std(c) + 1e-12))
    p_sig = np.mean(c ** 2) + 1e-12
    p_noise = np.mean(noise ** 2) + 1e-12
    return float(10 * np.log10(p_sig / p_noise))


def _silence_fraction(x: np.ndarray, db_thresh: float = SILENCE_DB_THRESHOLD) -> float:
    frame = 200
    rms = np.sqrt(np.convolve(x ** 2, np.ones(frame) / frame, mode="same"))
    ref = np.max(rms) + 1e-12
    db = 20 * np.log10(rms / ref + 1e-12)
    return float(np.mean(db < db_thresh))


# =====================================================================
# Audio-level quality audit
# =====================================================================

def audit_audio_quality(recordings: list[Recording], sample: int | None = None) -> pd.DataFrame:
    """Audit a sample of recordings for quality stats."""
    import soundfile as sf

    rs = recordings if sample is None else recordings[:sample]
    rows = []
    for r in tqdm(rs, desc="audio audit"):
        try:
            x, sr = sf.read(str(r.wav_path), dtype="float32")
            if x.ndim > 1:  # mono-ise
                x = x.mean(axis=1)
            pre = preprocess_audio(x, sr, wavelet_denoise=True)
            rows.append(
                {
                    "patient_id": r.patient_id,
                    "location": r.location,
                    "murmur": r.murmur,
                    "outcome": r.outcome,
                    "duration_s": pre["duration_s"],
                    "sr": pre["sr"],
                    "clipping_rate": pre["clipping_rate"],
                    "snr_estimate_db": pre["snr_estimate_db"],
                    "silence_fraction": pre["silence_fraction"],
                    "n_spikes_removed": pre["n_spikes_removed"],
                }
            )
        except Exception as e:
            print(f"[warn] failed on {r.wav_path.name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "audio_quality_audit.csv", index=False)

    # Summary plots
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.boxplot(data=df, x="murmur", y="duration_s", ax=axes[0, 0])
    axes[0, 0].set_title("Recording duration by class")
    sns.boxplot(data=df, x="murmur", y="snr_estimate_db", ax=axes[0, 1])
    axes[0, 1].set_title("SNR estimate by class")
    sns.boxplot(data=df, x="murmur", y="clipping_rate", ax=axes[1, 0])
    axes[1, 0].set_title("Clipping rate by class")
    sns.boxplot(data=df, x="murmur", y="silence_fraction", ax=axes[1, 1])
    axes[1, 1].set_title("Silence fraction by class")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "quality_by_class.png", dpi=140)
    plt.close(fig)

    print("\n## Audio quality — per-class median ± IQR")
    print(
        df.groupby("murmur")[
            ["duration_s", "snr_estimate_db", "clipping_rate", "silence_fraction"]
        ]
        .describe()
        .to_string()
    )

    return df


# =====================================================================
# Representative-waveform showcase
# =====================================================================

def showcase_examples(recordings: list[Recording], per_class: int = 3) -> None:
    """Plot waveform + spectrogram before and after preprocessing for a handful
    of recordings per class, so we can visually confirm the pipeline is sane."""
    import soundfile as sf

    by_cls = {"Present": [], "Absent": [], "Unknown": []}
    for r in recordings:
        if r.murmur in by_cls and len(by_cls[r.murmur]) < per_class:
            by_cls[r.murmur].append(r)
        if all(len(v) >= per_class for v in by_cls.values()):
            break

    for cls, rs in by_cls.items():
        for r in rs:
            try:
                x, sr = sf.read(str(r.wav_path), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                pre = preprocess_audio(x, sr, wavelet_denoise=True)
                _plot_example(r, pre, cls)
            except Exception as e:
                print(f"[warn] showcase failed on {r.wav_path.name}: {e}")


def _plot_example(r: Recording, pre: dict, cls: str) -> None:
    sr = pre["sr"]
    raw = pre["raw"]
    clean = pre["normalised"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    t_raw = np.arange(len(raw)) / sr
    t_clean = np.arange(len(clean)) / sr
    axes[0, 0].plot(t_raw[: sr * 5], raw[: sr * 5], lw=0.6)
    axes[0, 0].set_title(f"Raw — {cls} — {r.patient_id}_{r.location} (first 5s)")
    axes[0, 0].set_xlabel("s"); axes[0, 0].set_ylabel("amplitude")

    axes[0, 1].plot(t_clean[: sr * 5], clean[: sr * 5], lw=0.6, color="C1")
    axes[0, 1].set_title("Preprocessed (first 5s)")
    axes[0, 1].set_xlabel("s"); axes[0, 1].set_ylabel("amplitude")

    _spec(axes[1, 0], raw, sr, title="Raw spectrogram")
    _spec(axes[1, 1], clean, sr, title="Preprocessed spectrogram")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"example_{cls}_{r.patient_id}_{r.location}.png", dpi=140
    )
    plt.close(fig)


def _spec(ax, x: np.ndarray, sr: int, title: str) -> None:
    f, t, Sxx = signal.spectrogram(x, fs=sr, nperseg=512, noverlap=256)
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto")
    ax.set_ylim(0, 600)
    ax.set_ylabel("Hz"); ax.set_xlabel("s")
    ax.set_title(title)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--audio-sample",
        type=int,
        default=None,
        help="Limit audio quality audit to first N recordings. Full set is slow.",
    )
    args = parser.parse_args()

    df = load_metadata(args.metadata_csv)
    summarise_metadata(df)

    if args.metadata_only:
        return

    recordings = build_recording_index(df, args.data_dir)
    plot_recording_level_stats(recordings)
    audit_audio_quality(recordings, sample=args.audio_sample)
    showcase_examples(recordings, per_class=3)

    print(f"\n[ok] All outputs written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
