# Heart Murmur Detection on the CirCor 2022 Dataset

**Coursework:** STW7072CEM Machine Learning
**Author:** Shishir Kandel
**Headline result:** Gaussian Process Classification reaches **84.0% accuracy / AUC 0.752 / ECE 0.046** on CirCor 2022 binary murmur detection — a 7.4-point improvement over the published pure-classical SOTA of 76.61% (Vimalajeewa et al., Nature Sci Reports 2025).

---

## Project structure

```
ML Heart Sound Project/
├── README.md                              ← this file
├── report.tex                             ← final coursework report (LaTeX)
├── references_heart.bib                   ← full bib for the report
├── heart_murmur_pipeline.ipynb            ← guided Jupyter notebook walkthrough
├── proposal/                              ← earlier proposal version (kept for record)
│   ├── main.tex
│   ├── references_proposal.bib
│   └── ML_Proposal_Heart.pdf
│
├── extract_features.py                    ← feature extraction (48 features × 5 domains)
├── eda_and_preprocess.py                  ← exploratory data analysis + preprocessing demo
├── run_models.py                          ← 5-model × 5-feature-set ablation grid
├── train_final.py                         ← train + save final production models
├── predict.py                             ← predict on a new WAV file
├── make_plots.py                          ← generate all figures
├── uq_unknown.py                          ← uncertainty validation on Unknown class
├── error_analysis.py                      ← which Present recordings do all models miss?
├── run_smote.py                           ← SMOTE vs class-weight sensitivity analysis
│
├── features.csv                           ← extracted features (3,163 recordings × 48 cols)
├── results_ablation.csv                   ← fold-level ablation results
├── results_summary.csv                    ← aggregated metrics
├── feature_importance.csv                 ← Lasso L1 + XGBoost gain importance
├── uq_results.csv                         ← per-recording GPC entropy
├── error_analysis.csv                     ← per-Present-recording per-model predictions
├── results_smote.csv                      ← SMOTE vs class-weight comparison
│
├── models/                                ← trained .pkl files for predict.py
│   ├── lasso_lr.pkl
│   ├── knn.pkl
│   ├── svc_rbf.pkl
│   ├── xgboost.pkl
│   ├── gpc.pkl
│   ├── scaler.pkl
│   └── feature_columns.json
│
├── plots/                                 ← all figures
│   ├── ablation_heatmap_acc.png
│   ├── ablation_heatmap_f1.png
│   ├── ablation_heatmap_auc.png
│   ├── metric_comparison.png
│   ├── calibration_full.png
│   ├── feature_importance_top15.png
│   ├── uq_hist.png
│   ├── error_analysis_locations.png
│   ├── error_analysis_quality.png
│   └── smote_comparison.png
│
├── eda_outputs/                           ← EDA tables and class-balance plots
│
└── the-circor-digiscope-phonocardiogram-dataset-1.0.3/   ← dataset (download from PhysioNet)
    ├── training_data.csv
    └── training_data/
        ├── 13918_AV.wav
        ├── 13918_AV.tsv
        ├── 13918.txt
        └── ...
```

## Quick start

### Setup

```bash
pip install numpy pandas matplotlib seaborn scipy soundfile librosa pywavelets \
            xgboost nolds tqdm scikit-learn imbalanced-learn joblib jupyter
```

### Download dataset

```bash
# Free, no login: https://physionet.org/content/circor-heart-sound/1.0.3/
# Unzip into ./the-circor-digiscope-phonocardiogram-dataset-1.0.3/
```

### Reproduce results from scratch

```bash
# 1. EDA on metadata + audio quality (a few minutes)
python eda_and_preprocess.py --audio-sample 200

# 2. Extract 48 features per recording (~50 minutes)
python extract_features.py

# 3. Run the 5-model × 5-feature-set ablation grid (~30 minutes)
python run_models.py

# 4. Generate all plots (seconds)
python make_plots.py

# 5. Train + save final production models on all data (~10 minutes)
python train_final.py

# 6. Sensitivity / robustness analyses (~30 minutes total)
python uq_unknown.py
python error_analysis.py
python run_smote.py
```

### Predict on a new recording

```bash
python predict.py path/to/recording.wav --tsv path/to/segmentation.tsv
```

Example with bundled data:

```bash
python predict.py the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/13918_AV.wav \
                  --tsv the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/13918_AV.tsv
```

### Walk through everything in Jupyter

```bash
jupyter notebook heart_murmur_pipeline.ipynb
```

The notebook loads pre-computed `features.csv` and `results_ablation.csv` so it runs end-to-end in seconds.

## Build the LaTeX documents

```bash
# Final report (main deliverable)
pdflatex report && bibtex report && pdflatex report && pdflatex report

# Proposal (older version, kept for record)
cd proposal && pdflatex main && bibtex main && pdflatex main && pdflatex main && cd ..
```

## Pipeline summary

1. **Preprocessing** — resample 4 kHz → DC removal → 25–450 Hz Butterworth bandpass → 3×median spike removal → Daubechies-4 wavelet soft-thresholding → peak normalisation
2. **48 features per recording** — 26 MFCC, 6 spectral, 4 wavelet, 4 fractal/complexity, 6 heart-cycle temporal, 2 quality
3. **5 classical models** — Lasso-LR, K-NN (k=15, distance), SVC-RBF, XGBoost, GPC
4. **5 progressive feature sets** for ablation: MFCC → +spectral → +wavelet → +fractal → +heart-cycle
5. **5-fold patient-level stratified CV** (`StratifiedGroupKFold` on `patient_id`)
6. **Class imbalance** handled via class weights (3.88:1 ratio); SMOTE / random oversampling tested as sensitivity analysis
7. **Calibration** measured by ECE and Brier; **uncertainty** validated on held-out Unknown class

## Key findings

- **GPC + 48 features = 84.0% accuracy, AUC 0.752, ECE 0.046** — beats published SOTA by 7.4 pts
- **Fractal features unlock GPC** — F1 jumps from 0.000 to 0.404 when fractal is added (step (d))
- **All 5 feature families appear in top-15 importance** — multi-domain hypothesis validated
- **Hypothesis H3 refuted honestly** — we predicted XGBoost would dominate; GPC won
- **34.6% of Present recordings missed by all 5 models** — likely innocent murmurs mixed into CirCor's ``Present'' label
- **Unknown entropy > Absent entropy** (Mann–Whitney p = 0.010) — partial confirmation of UQ hypothesis

## Clinical relevance

For Nepal-style RHD school screening (107,340 children/year, RHD prevalence 2.22 per 1000):
- **Lasso-LR** (sens 61%, spec 74%) → inclusive screening gate, refer broadly
- **GPC** (sens 31%, spec 98%, calibrated probabilities) → high-confidence referral overlay

## Reproducibility

- All random seeds fixed at `random_state=0`
- All hyperparameters fixed (no nested CV — noted as limitation)
- All code released alongside this submission
- Python 3.12.10, scikit-learn 1.x, xgboost 3.2.0, librosa 0.11.0
