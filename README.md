# lightprop 🧪

**A lightweight chemprop/fastprop-style QSAR tool** for building conformal prediction models from SMILES using Mordred descriptors and/or RDKit fingerprints with LightGBM.

---

## Features

| Feature | Details |
|---|---|
| **Featurization** | Mordred 2D descriptors (~1,800), RDKit fingerprints (Morgan, RDKit, MACCS, AtomPair) |
| **ML Method** | LightGBM (fast gradient boosting, handles high-dimensional descriptors well) |
| **Uncertainty** | Conformal prediction via MAPIE — guaranteed coverage intervals/sets |
| **Tasks** | Regression (e.g. pIC50, logP) and binary/multi-class classification |
| **No foundation model needed** | Works well on typical QSAR dataset sizes (100–100k compounds) |

---

## Install

```bash
pip install lightgbm mordred rdkit scikit-learn pandas numpy mapie tqdm

# If mordred gives issues, try the community fork:
pip install mordredcommunity
```

---

## Quick Start

### 1. Prepare your data (CSV format)
```
ID,SMILES,pIC50
COMP_0001,CC(=O)Oc1ccccc1C(=O)O,6.3
COMP_0002,CC(C)Cc1ccc(cc1)C(C)C(=O)O,7.1
...
```

### 2. Train a regression model

```bash
python lightprop.py train \
    --input your_data.csv \
    --smiles_col SMILES \
    --id_col ID \
    --activity_col pIC50 \
    --task regression \
    --features mordred rdkit \
    --fp_types morgan \
    --alpha 0.1 \
    --output results/ \
    --save_model \
    --predict_all
```

### 3. Train a classification model

```bash
python lightprop.py train \
    --input your_data.csv \
    --smiles_col SMILES \
    --id_col ID \
    --activity_col pIC50 \
    --task classification \
    --threshold 7.0 \
    --features mordred rdkit \
    --alpha 0.1 \
    --output results_cls/
```

### 4. Predict on new compounds

```bash
python lightprop.py predict \
    --input new_compounds.csv \
    --smiles_col SMILES \
    --id_col ID \
    --load_model results/model.pkl \
    --features mordred rdkit \
    --output predictions/
```

### 5. Generate example data and test

```bash
python make_example_data.py
python lightprop.py train -i example_data.csv --save_model --predict_all -o test_out/
```

---

## Arguments

### `train`

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Input CSV file |
| `--smiles_col` | `SMILES` | SMILES column name |
| `--id_col` | `ID` | Compound ID column name |
| `--activity_col` | `pIC50` | Activity column name |
| `--task` | `regression` | `regression` or `classification` |
| `--threshold` | None | Binarize activity ≥ threshold for classification |
| `--features` | `mordred rdkit` | Feature types: `mordred`, `rdkit`, or both |
| `--fp_types` | `morgan` | `morgan`, `rdkit`, `maccs`, `atompair` (can pass multiple) |
| `--fp_radius` | `2` | Morgan fingerprint radius |
| `--fp_bits` | `2048` | Fingerprint size (bits) |
| `--alpha` | `0.1` | Conformal error rate → 90% coverage |
| `--test_size` | `0.15` | Fraction held out as test set |
| `--cal_size` | `0.15` | Fraction of train+cal used for calibration |
| `--output` | `lightprop_results` | Output directory |
| `--save_model` | flag | Save model.pkl for later inference |
| `--predict_all` | flag | Predict on full dataset (not just test) |
| `--seed` | `42` | Random seed |

---

## Outputs

```
results/
├── predictions_test.csv      # Test set predictions with intervals/sets
├── predictions_full.csv      # Full dataset (if --predict_all)
├── metrics.csv               # RMSE, R², coverage, interval width, etc.
├── feature_importance.csv    # Top-50 LightGBM feature importances
└── model.pkl                 # Saved model (if --save_model)
```

### Regression output columns
| Column | Description |
|---|---|
| `y_pred` | Point prediction |
| `lower_PI` / `upper_PI` | Conformal prediction interval bounds |

### Classification output columns
| Column | Description |
|---|---|
| `y_pred` | Hard prediction |
| `prob_class0` / `prob_class1` | LightGBM class probabilities |
| `prediction_set_size` | Number of classes in conformal prediction set (1 = certain, 2 = uncertain) |

---

## Conformal Prediction Explained

Conformal prediction provides **distribution-free coverage guarantees**:
- With `--alpha 0.1`, prediction intervals/sets will contain the true value **≥ 90% of the time** on new data
- This is a **statistical guarantee** (not just a calibration)
- **Regression**: wider intervals = more uncertain predictions
- **Classification**: prediction sets of size 2 indicate the model is uncertain between classes

---

## When to use a foundation model?

| Scenario | Recommendation |
|---|---|
| > 1,000 compounds, typical ADMET/potency | **lightprop is sufficient** |
| < 100 compounds | Consider transfer learning or add pre-computed 3D features |
| Scaffold-hopping generalization | Foundation model (e.g. MolBERT, ChemBERTa) may help |
| Novel chemical space | Consider combining lightprop features with pre-trained embeddings |

For most drug discovery QSAR tasks, Mordred + Morgan + LightGBM is competitive with GNN-based models.

---

## Tips

- **Use both `--features mordred rdkit`** for best coverage; RDKit fps are fast, Mordred adds physicochemical context
- **Start with `--features rdkit --fp_types morgan maccs`** for a quick first run
- **For small datasets** (< 500 cpds): use `--cal_size 0.2` and consider cross-validation
- **For classification**: always check the `prediction_set_size` column — size=2 flags ambiguous compounds for experimental follow-up
- **Feature importance**: use `feature_importance.csv` to understand which descriptors drive predictions

---

## Dependencies

- `lightgbm` — gradient boosting
- `mapie` — conformal prediction wrappers
- `mordred` or `mordredcommunity` — 2D molecular descriptors
- `rdkit` — fingerprints and SMILES parsing
- `scikit-learn` — preprocessing and splitting
- `pandas`, `numpy`, `tqdm` — data handling
