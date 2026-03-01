# lightprop

**QSAR conformal prediction** — Mordred descriptors + RDKit fingerprints · LightGBM · MAPIE · Optuna

Builds regression or classification models from SMILES with **statistically valid coverage guarantees** via conformal prediction (jackknife+ / score method). Supports nested Bayesian hyperparameter optimisation so the conformal guarantee is never inflated by tuning.

---

## Features

| | |
|---|---|
| **Featurisation** | Mordred 2D descriptors (~1,600), RDKit fingerprints (Morgan / RDKit / MACCS / atom-pair), or any combination |
| **Model** | LightGBM — fast, robust on high-dimensional tabular data, no GPU required |
| **Uncertainty** | MAPIE conformal prediction — guaranteed marginal coverage at your chosen confidence level |
| **Tasks** | Continuous regression (pIC50, logP, …) and binary classification |
| **HPO** | Optuna TPE — nested architecture keeps the test set fully isolated from tuning |
| **Diagnostics** | Separate in-sample train metrics and held-out test metrics, side-by-side in `metrics.csv` |
| **Outputs** | Predictions, intervals/sets, feature importances, scatter plot, timestamped model card + checkpoint |

---

## Installation

```bash
conda create -n lightprop python=3.11
conda activate lightprop

pip install lightgbm scikit-learn mapie optuna pandas numpy tqdm matplotlib
pip install rdkit
pip install mordred          # or: pip install mordredcommunity
```

> **MAPIE ≥ 1.0 required.** lightprop uses the `CrossConformalRegressor` / `SplitConformalRegressor` API introduced in MAPIE 1.x.

---

## Input format

A CSV with at minimum three columns:

```
ID,SMILES,pIC50
COMP001,CC(=O)Oc1ccccc1C(=O)O,6.3
COMP002,CC(C)Cc1ccc(cc1)C(C)C(=O)O,7.1
```

Column names are configurable with `--smiles_col`, `--id_col`, `--activity_col`.

---

## Quick start

```bash
# Regression — 90% conformal prediction intervals
python lightprop.py train -i data.csv --activity_col pIC50 --confidence_level 0.9

# Fast run: Morgan FPs only, 3 folds  (~3–5 min on 300 compounds)
python lightprop.py train -i data.csv --activity_col pIC50 \
    --features rdkit --fp_types morgan --cv_folds 3 --confidence_level 0.9

# With Bayesian HPO (50 Optuna trials, 3-fold inner CV)
python lightprop.py train -i data.csv --activity_col pIC50 \
    --hpo_trials 50 --confidence_level 0.9

# Binary classification (binarise activity ≥ 7.0)
python lightprop.py train -i data.csv --task classification --threshold 7.0 \
    --confidence_level 0.9 --hpo_trials 50 --save_model -o results/

# Prefit mode (faster for large datasets — uses a separate calibration set)
python lightprop.py train -i data.csv --cv_folds 0 --confidence_level 0.9

# Inference on new compounds using a saved checkpoint
python lightprop.py predict -i new_cpds.csv --load_model results/model_latest.pkl
```

---

## Architecture

```
Input CSV
  │
  ├─ Feature generation
  │     Mordred 2D descriptors (~1,600 features after cleaning)
  │     RDKit fingerprints (Morgan / RDKit / MACCS / atom-pair)
  │     Features cleaned: NaN-heavy and constant columns removed
  │
  ├─ Outer split ── trainval (85%) / test (15%)
  │                 Test set is never touched during HPO or calibration
  │
  ├─ [Optional] Nested HPO
  │     Optuna TPE, N trials, inner k-fold CV on trainval only
  │     Optimises: n_estimators, learning_rate, num_leaves,
  │                min_child_samples, subsample, colsample_bytree,
  │                reg_alpha, reg_lambda
  │     Minimises: CV-RMSE (regression) / CV-log-loss (classification)
  │
  ├─ Conformal model
  │     Cross-conformal (default): k-fold, jackknife+ intervals
  │     Prefit: train on train split, calibrate on holdout set
  │     Base estimator: LightGBM with best HPO params (or defaults)
  │
  ├─ Train metrics  ──  fitted_base.predict(X_trainval)   [in-sample]
  ├─ Test metrics   ──  mapie.predict_interval(X_test)    [held-out]
  │
  └─ Outputs ── metrics.csv  predictions_test.csv  predictions_all.csv
                scatter_pred_vs_exp.png  feature_importance.csv
                [model_<date>.pkl + model_<date>.json  if --save_model]
```

### Conformal modes

| Mode | Flag | Notes |
|------|------|-------|
| Cross-conformal | `--cv_folds 5` (default) | Every compound contributes to both training and calibration. Jackknife+ gives a finite-sample valid marginal coverage bound. |
| Prefit | `--cv_folds 0` | Separate calibration set. Simpler and faster for large datasets, but wastes some training data. |

---

## CLI reference

### `train`

```
python lightprop.py train -i FILE [options]
```

**Data**

| Argument | Default | Description |
|----------|---------|-------------|
| `-i / --input` | *(required)* | Input CSV |
| `--activity_col` | `pIC50` | Target column |
| `--smiles_col` | `SMILES` | SMILES column |
| `--id_col` | `ID` | ID column |
| `--task` | `regression` | `regression` or `classification` |
| `--threshold` | `None` | Binarise activity ≥ threshold (classification) |

**Features**

| Argument | Default | Description |
|----------|---------|-------------|
| `--features` | `mordred rdkit` | `mordred`, `rdkit`, or both |
| `--fp_types` | `morgan` | `morgan`, `rdkit`, `maccs`, `atompair` — pass multiple |
| `--fp_radius` | `2` | Morgan radius |
| `--fp_bits` | `2048` | Fingerprint bit length |

**Conformal prediction**

| Argument | Default | Description |
|----------|---------|-------------|
| `--confidence_level` | *(prompted)* | Target coverage, e.g. `0.9` for 90% |
| `--cv_folds` | `5` | Cross-conformal folds; `0` = prefit mode |
| `--test_size` | `0.15` | Held-out test fraction |
| `--cal_size` | `0.15` | Calibration fraction (prefit mode only) |

**HPO (Optuna)**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hpo_trials` | `0` | Optuna trials (0 = disabled). Recommended: 50–100 |
| `--hpo_cv_folds` | `3` | Inner CV folds for HPO |

**Output**

| Argument | Default | Description |
|----------|---------|-------------|
| `-o / --output` | `lightprop_results` | Output directory |
| `--save_model` | off | Pickle model + write JSON model card |
| `--model_name` | `model` | Base name for checkpoint files |
| `--seed` | `42` | Random seed |

---

### `predict`

```
python lightprop.py predict -i FILE --load_model CHECKPOINT [options]
```

| Argument | Description |
|----------|-------------|
| `-i / --input` | New compounds CSV (must have SMILES and ID columns) |
| `--load_model` | Path to `.pkl` checkpoint (use `model_latest.pkl` for convenience) |
| `--smiles_col` | SMILES column (must match training) |
| `--id_col` | ID column |
| `--features`, `--fp_types`, … | Must match flags used during training |
| `-o / --output` | Output directory for `predictions.csv` |

---

## Output files

| File | Description |
|------|-------------|
| `metrics.csv` | Train and test metrics side by side (see below) |
| `predictions_test.csv` | Per-compound held-out test predictions with conformal intervals or prediction sets |
| `predictions_all.csv` | Predictions for all compounds (train + test) with a `split` column |
| `scatter_pred_vs_exp.png` | Predicted vs experimental; test points have conformal error bars; annotation shows R² train and R² test |
| `feature_importance.csv` | Top-50 LightGBM feature importances (split-based) |
| `model_<yyMMdd>.pkl` | Full checkpoint: MAPIE wrapper + imputer + base model + feature names |
| `model_<yyMMdd>.json` | Human-readable model card: hyperparameters, dataset stats, all metrics |
| `model_latest.pkl` | Alias to the most recent checkpoint |
| `hpo_trials.csv` | Full Optuna trial history (only with `--hpo_trials > 0`) |
| `hpo_param_importance.csv` | HPO hyperparameter importances (only with `--hpo_trials > 0`) |

### `metrics.csv` columns

**Regression**

| Column | Description |
|--------|-------------|
| `Train_R2` | In-sample R² (fitted_base on X_trainval) |
| `Train_RMSE` | In-sample RMSE |
| `Train_MAE` | In-sample MAE |
| `Test_R2` | Held-out test R² |
| `Test_RMSE` | Held-out test RMSE |
| `Test_MAE` | Held-out test MAE |
| `Test_Coverage@N%` | Fraction of test compounds whose true value falls inside the conformal interval |
| `Test_Avg_Interval_Width` | Mean width of conformal prediction intervals |
| `Test_Target_Coverage` | Requested confidence level |

**Classification**

| Column | Description |
|--------|-------------|
| `Train_Accuracy` | In-sample accuracy |
| `Train_Balanced_Accuracy` | In-sample balanced accuracy |
| `Train_MCC` | In-sample Matthews correlation coefficient |
| `Train_ROC_AUC` | In-sample ROC-AUC |
| `Test_Accuracy` | Held-out accuracy |
| `Test_Balanced_Accuracy` | Held-out balanced accuracy |
| `Test_MCC` | Held-out MCC |
| `Test_ROC_AUC` | Held-out ROC-AUC |
| `Test_Coverage@N%` | Fraction of test compounds whose true class is in the conformal prediction set |
| `Test_Avg_Prediction_Set_Size` | Mean prediction set size (1 = certain, 2 = uncertain) |

### `predictions_test.csv` columns

**Regression:** `ID`, `y_true`, `y_pred`, `lower_PI`, `upper_PI`, `PI_width`

**Classification:** `ID`, `y_true`, `y_pred`, `prediction_set_size`, `prob_class0`, `prob_class1`

---

## Example output

Run on SMUG1i dataset (335 compounds, Mordred + Morgan, 5-fold cross-conformal, default LightGBM params):

```
=== Train Metrics (in-sample) ===
  Train_R2:   0.9999    ← LightGBM memorises training data with default params
  Train_RMSE: 0.0070
  Train_MAE:  0.0034

=== Test Metrics (held-out) ===
  RMSE:               0.4565
  MAE:                0.3541
  R2:                 0.7251    ← gap → run --hpo_trials 50 to regularise
  Coverage@90%:       0.9412    ✓ above target of 0.90
  Avg_Interval_Width: 1.79
```

The train/test R² gap is a strong signal to run HPO:

```bash
python lightprop.py train -i SMUG1i_pIC50.csv --activity_col pIC50 \
    --confidence_level 0.9 --hpo_trials 50
```

---

## Conformal prediction

Conformal prediction provides **distribution-free, finite-sample coverage guarantees**:

- With `--confidence_level 0.9`, the conformal intervals/sets will contain the true value **≥ 90% of the time** on exchangeable test data — this is a statistical guarantee, not a heuristic
- Cross-conformal with jackknife+ (regression) gives a valid marginal bound even with a finite calibration set
- **Regression**: interval width reflects prediction uncertainty — wider = less confident
- **Classification**: prediction sets of size 2 flag compounds where the model is uncertain between classes; these are the most valuable candidates for experimental follow-up

The guarantee is **marginal** (holds in expectation over random splits) and assumes exchangeability between training and test distributions. It does not hold under strong distribution shift.

---

## Nested CV and data leakage

When `--hpo_trials > 0`:

```
┌─ Outer split: trainval / test ──────────────────────────────────────┐
│  Test set is fixed before HPO begins and never accessed             │
│  ┌─ Inner HPO CV (--hpo_cv_folds, default 3) ──────────────────┐   │
│  │  Optuna minimises CV-RMSE / CV-log-loss over N trials        │   │
│  └────────────────────────────────────────────────────────────── ┘  │
│  Best params → final conformal model on full trainval               │
│  Coverage guarantee evaluated on outer test set                     │
└─────────────────────────────────────────────────────────────────────┘
```

This ensures the conformal coverage estimate on the test set is unbiased by hyperparameter selection.

---

## Tips

- **Large train/test R² gap?** Run `--hpo_trials 50` — regularisation params (`reg_alpha`, `reg_lambda`, `min_child_samples`) are the key levers
- **Fast first run:** `--features rdkit --fp_types morgan --cv_folds 3` — ~3–5 min on 300 compounds
- **Full run:** `--features mordred rdkit` — Mordred adds physicochemical context at the cost of ~14 s descriptor computation
- **Classification ambiguity:** `prediction_set_size == 2` compounds are borderline — prioritise these for experimental testing
- **WSL2 users:** MAPIE folds run sequentially (`n_jobs=1`) while LightGBM uses all cores internally. Setting `n_jobs=-1` on the MAPIE wrapper causes nested parallelism deadlocks.

---

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.10 |
| lightgbm | ≥ 4.0 |
| mapie | ≥ 1.0 |
| scikit-learn | ≥ 1.3 |
| rdkit | ≥ 2023.3 |
| mordred / mordredcommunity | any |
| optuna | ≥ 3.0 (optional, for HPO) |
| matplotlib | any (optional, for scatter plot) |
| pandas, numpy, tqdm | any |
