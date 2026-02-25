# Installation

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- Git (optional, for cloning)

---

## 1. Create the conda environment

```bash
conda env create -f environment.yml
```

This installs Python 3.11, RDKit, LightGBM, scikit-learn, and all dependencies.
The environment is named `lightprop` — change the `name:` field in `environment.yml` if you prefer something else.

## 2. Activate

```bash
conda activate lightprop
```

## 3. Verify

```bash
python -c "
from rdkit import Chem
import lightgbm, mapie, optuna, mordred
print('rdkit   :', Chem.__version__ if hasattr(Chem, '__version__') else 'OK')
print('lightgbm:', lightgbm.__version__)
print('mapie   :', mapie.__version__)
print('optuna  :', optuna.__version__)
print('mordred : OK')
"
```

---

## Troubleshooting

### mordred fails to install or import on Python 3.11+

`mordred` is unmaintained and may fail on newer Python versions. Use the community fork instead:

```bash
pip uninstall mordred -y
pip install mordredcommunity
```

`lightprop.py` tries `mordred` first and falls back to `mordredcommunity` automatically.

### RDKit import error

RDKit must be installed via conda-forge — the pip version is unofficial and often broken.
If you installed it via pip by mistake:

```bash
pip uninstall rdkit rdkit-pypi -y
conda install -c conda-forge rdkit
```

### MAPIE version conflicts

MAPIE's API changed between 0.7 and 0.8. If you see errors about `MapieRegressor` arguments, ensure you have ≥ 0.8.0:

```bash
pip install --upgrade "mapie>=0.8.0"
```

### Apple Silicon (M1/M2/M3)

The `environment.yml` works on Apple Silicon via conda-forge's native ARM builds.
If you hit issues with LightGBM, install it explicitly:

```bash
conda install -c conda-forge lightgbm
```

---

## Updating the environment

To add or upgrade packages without rebuilding from scratch:

```bash
conda activate lightprop
pip install --upgrade mapie optuna
```

To fully rebuild after editing `environment.yml`:

```bash
conda env remove -n lightprop
conda env create -f environment.yml
```

---

## Quick start

```bash
conda activate lightprop

# Generate example data
python make_example_data.py

# Train (cross-conformal, default settings)
python lightprop.py train \
    --input example_data.csv \
    --smiles_col SMILES \
    --id_col ID \
    --activity_col pIC50 \
    --model_name my_model \
    --save_model \
    --output results/

# Predict on new compounds
python lightprop.py predict \
    --input new_compounds.csv \
    --load_model results/my_model_latest.pkl \
    --output predictions/
```
