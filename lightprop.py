"""
lightprop.py - QSAR Conformal Prediction Tool
============================================
A chemprop/fastprop-style tool for building predictive models from SMILES
using Mordred descriptors and/or RDKit fingerprints with LightGBM.
Supports conformal regression and classification with optional Bayesian HPO.

─── Modes ────────────────────────────────────────────────────────────────────

Cross-conformal (default, --cv_folds k):
  MAPIE trains k LightGBM models; out-of-fold predictions calibrate conformal
  scores. Every compound contributes to both training and calibration.
  Method: jackknife+ (regression) / score (classification).

Prefit (--cv_folds 0):
  Dedicated holdout calibration set. Simpler, faster for large datasets.

─── Hyperparameter Optimisation ──────────────────────────────────────────────

Add --hpo_trials N (e.g. 50) to enable Optuna Bayesian search.

Architecture: NESTED cross-validation
  ┌─ Outer split: trainval vs. test (held out, never touched during HPO) ──┐
  │  ┌─ HPO inner CV (--hpo_cv_folds, default 3) ────────────────────────┐ │
  │  │  Optuna minimises CV-RMSE / CV-log-loss over N trials             │ │
  │  │  Searches: lr, num_leaves, min_child_samples, subsample,          │ │
  │  │            colsample_bytree, reg_alpha, reg_lambda, n_estimators  │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │  Best params → final cross-conformal model on full trainval            │
  │  Coverage guarantee evaluated on outer test set (never seen by Optuna) │
  └────────────────────────────────────────────────────────────────────────┘

This separation ensures the conformal coverage guarantee is not inflated by
hyperparameter tuning — the test set is a true held-out evaluation set.

─── Usage ────────────────────────────────────────────────────────────────────

  # Basic regression
  python lightprop.py train -i data.csv --activity_col pIC50

  # With HPO (50 Optuna trials, 3-fold inner CV)
  python lightprop.py train -i data.csv --hpo_trials 50

  # Classification + HPO + save model
  python lightprop.py train -i data.csv --task classification --threshold 7.0 \\
      --hpo_trials 50 --save_model -o results/

  # Prefit mode (faster, large datasets)
  python lightprop.py train -i data.csv --cv_folds 0

  # Predict with saved model
  python lightprop.py predict -i new_cpds.csv --load_model results/model.pkl

─── Requirements ─────────────────────────────────────────────────────────────

  pip install lightgbm mordred rdkit scikit-learn pandas numpy mapie tqdm optuna
  # mordred alt: pip install mordredcommunity
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Feature Generation ────────────────────────────────────────────────────────

def smiles_to_mol(smiles_list):
    """Convert SMILES to RDKit mol objects; return mols and valid indices."""
    from rdkit import Chem
    mols, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            mols.append(mol)
            valid_idx.append(i)
        else:
            logger.warning(f"Invalid SMILES at index {i}: {smi}")
    return mols, valid_idx


def compute_rdkit_fingerprints(mols, fp_type="morgan", radius=2, nbits=2048):
    """Compute RDKit fingerprints. fp_type: morgan | rdkit | maccs | atompair"""
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.rdFingerprintGenerator import (
        GetMorganGenerator, GetRDKitFPGenerator, GetAtomPairGenerator,
    )
    logger.info(f"Computing {fp_type} fingerprints (nbits={nbits})...")
    fps = []
    for mol in tqdm(mols, desc=f"RDKit {fp_type}"):
        if fp_type == "morgan":
            fp = GetMorganGenerator(radius=radius, fpSize=nbits).GetFingerprintAsNumPy(mol)
        elif fp_type == "rdkit":
            fp = GetRDKitFPGenerator(fpSize=nbits).GetFingerprintAsNumPy(mol)
        elif fp_type == "maccs":
            fp = np.array(MACCSkeys.GenMACCSKeys(mol))
        elif fp_type == "atompair":
            fp = GetAtomPairGenerator(fpSize=nbits).GetFingerprintAsNumPy(mol)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        fps.append(fp)
    fps = np.array(fps)
    return pd.DataFrame(fps, columns=[f"{fp_type}_fp_{i}" for i in range(fps.shape[1])])


def compute_mordred_descriptors(mols, ignore_3d=True):
    """Compute Mordred 2D descriptors (~1800 features)."""
    try:
        from mordred import Calculator, descriptors
    except ImportError:
        try:
            from mordredcommunity import Calculator, descriptors
        except ImportError:
            raise ImportError(
                "Install mordred: pip install mordred\n"
                "Or community fork: pip install mordredcommunity"
            )
    logger.info("Computing Mordred descriptors...")
    calc = Calculator(descriptors, ignore_3D=ignore_3d)
    results = []
    for mol in tqdm(mols, desc="Mordred"):
        try:
            results.append(calc(mol).fill_missing(0))
        except Exception as e:
            logger.warning(f"Mordred failed for a mol: {e}")
            results.append({str(k): 0 for k in calc.descriptors})
    df = pd.DataFrame([dict(r) for r in results])
    return df.apply(pd.to_numeric, errors="coerce")


def build_feature_matrix(mols, feature_types, fp_config=None):
    """Build combined feature matrix from selected feature types."""
    if fp_config is None:
        fp_config = {}
    dfs = []
    if "mordred" in feature_types:
        df = compute_mordred_descriptors(mols)
        dfs.append(df)
        logger.info(f"Mordred: {df.shape[1]} raw descriptors")
    if "rdkit" in feature_types:
        for fp_type in fp_config.get("fp_types", ["morgan"]):
            df = compute_rdkit_fingerprints(
                mols, fp_type=fp_type,
                radius=fp_config.get("radius", 2),
                nbits=fp_config.get("nbits", 2048),
            )
            dfs.append(df)
            logger.info(f"RDKit {fp_type}: {df.shape[1]} bits")
    if not dfs:
        raise ValueError("No features computed. Choose from: mordred, rdkit")
    return pd.concat(dfs, axis=1)


def clean_features(X):
    """Remove constant, NaN-heavy, and infinite columns."""
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, thresh=int(0.8 * len(X)))
    X = X.loc[:, X.nunique() > 1]
    logger.info(f"Features after cleaning: {X.shape[1]}")
    return X


# ─── Hyperparameter Optimisation ──────────────────────────────────────────────

def _lgbm_search_space(trial):
    """Optuna search space for LightGBM hyperparameters."""
    return {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples":  trial.suggest_int("min_child_samples", 5, 100),
        "subsample":          trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def run_hpo(X_trainval, y_trainval, task, n_trials=50, cv_folds=3,
            seed=42, n_jobs=-1, show_progress=True):
    """
    Run Optuna Bayesian HPO on the trainval set using inner k-fold CV.

    The test set is NEVER seen here — this is the inner loop of the
    nested CV architecture. The returned best_params are then used
    to build the final cross-conformal model on all of trainval.

    Parameters
    ----------
    X_trainval : np.ndarray  — imputed feature matrix
    y_trainval : np.ndarray  — labels / activity values
    task       : str         — 'regression' or 'classification'
    n_trials   : int         — number of Optuna trials
    cv_folds   : int         — inner CV folds for HPO scoring
    seed       : int         — random seed

    Returns
    -------
    best_params : dict   — best LightGBM hyperparameters found
    study       : optuna.Study
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Install Optuna: pip install optuna")

    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(
        f"Starting Optuna HPO: {n_trials} trials, {cv_folds}-fold inner CV "
        f"[task={task}]"
    )

    if task == "regression":
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scoring = "neg_root_mean_squared_error"
        direction = "minimize"
        ModelClass = lgb.LGBMRegressor
        fixed = dict(n_jobs=n_jobs, random_state=seed, verbose=-1)

        def objective(trial):
            params = {**_lgbm_search_space(trial), **fixed}
            model = ModelClass(**params)
            scores = cross_val_score(
                model, X_trainval, y_trainval,
                cv=splitter, scoring=scoring, n_jobs=1,
            )
            return -scores.mean()   # minimize RMSE

    else:  # classification
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scoring = "neg_log_loss"
        direction = "minimize"
        ModelClass = lgb.LGBMClassifier
        fixed = dict(class_weight="balanced", n_jobs=n_jobs, random_state=seed, verbose=-1)

        def objective(trial):
            params = {**_lgbm_search_space(trial), **fixed}
            model = ModelClass(**params)
            scores = cross_val_score(
                model, X_trainval, y_trainval,
                cv=splitter, scoring=scoring, n_jobs=1,
            )
            return -scores.mean()   # minimize log-loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)

    callbacks = []
    if show_progress:
        try:
            from tqdm.auto import tqdm as tqdm_optuna
            pbar = tqdm_optuna(total=n_trials, desc="Optuna HPO", unit="trial")

            def _pbar_callback(study, trial):
                pbar.update(1)
                pbar.set_postfix({"best": f"{study.best_value:.4f}"})

            callbacks.append(_pbar_callback)
        except Exception:
            pass

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks,
                   show_progress_bar=False)

    if show_progress:
        try:
            pbar.close()
        except Exception:
            pass

    best_params = study.best_params
    logger.info(f"HPO complete. Best inner-CV score: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    return best_params, study


def save_hpo_report(study, output_dir, task):
    """Save HPO trial history and parameter importances to CSV."""
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "hpo_trials.csv", index=False)
    logger.info(f"HPO trial history saved to {output_dir / 'hpo_trials.csv'}")

    try:
        import optuna
        importances = optuna.importance.get_param_importances(study)
        imp_df = pd.DataFrame(
            importances.items(), columns=["hyperparameter", "importance"]
        ).sort_values("importance", ascending=False)
        imp_df.to_csv(output_dir / "hpo_param_importance.csv", index=False)
        logger.info(f"HPO param importances saved to {output_dir / 'hpo_param_importance.csv'}")
    except Exception as e:
        logger.warning(f"Could not compute HPO param importances: {e}")


# ─── Model Training ────────────────────────────────────────────────────────────

def get_lgbm_model(task, params=None):
    """Return a configured LightGBM estimator."""
    import lightgbm as lgb

    defaults = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    if params:
        defaults.update(params)

    if task == "regression":
        return lgb.LGBMRegressor(**defaults)
    elif task == "classification":
        defaults["class_weight"] = "balanced"
        return lgb.LGBMClassifier(**defaults)
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")


def train_conformal_model(X_trainval, y_trainval, task,
                          cv_folds=5, lgbm_params=None, alpha=0.1,
                          X_cal=None, y_cal=None):
    """
    Train a conformal prediction model using MAPIE.

    cv_folds > 0  →  Cross-conformal (recommended):
        k-fold CV; out-of-fold scores calibrate conformal thresholds.
        jackknife+ (regression) gives valid marginal coverage guarantees.

    cv_folds == 0  →  Prefit mode:
        Trains on X_trainval, calibrates on the provided X_cal / y_cal.

    Returns: mapie, imputer, fitted_base
    """
    from sklearn.impute import SimpleImputer

    try:
        from mapie.regression import CrossConformalRegressor, SplitConformalRegressor
        from mapie.classification import CrossConformalClassifier, SplitConformalClassifier
    except ImportError:
        raise ImportError("Install MAPIE: pip install mapie")

    imputer = SimpleImputer(strategy="median")
    confidence_level = 1.0 - alpha

    if cv_folds > 0:
        logger.info(
            f"Cross-conformal training: cv={cv_folds} folds, "
            f"method={'plus' if task == 'regression' else 'score'}..."
        )
        imputer.fit(X_trainval)
        X_imp = imputer.transform(X_trainval)
        base_model = get_lgbm_model(task, lgbm_params)

        if task == "regression":
            mapie = CrossConformalRegressor(
                estimator=base_model, confidence_level=confidence_level,
                cv=cv_folds, n_jobs=1,  # LightGBM handles threading; nested joblib hangs WSL2
            )
            mapie.fit_conformalize(X_imp, y_trainval)
            fitted_base = mapie._mapie_regressor.estimator_.single_estimator_
        else:
            mapie = CrossConformalClassifier(
                estimator=base_model, confidence_level=confidence_level,
                cv=cv_folds, n_jobs=1,
            )
            mapie.fit_conformalize(X_imp, y_trainval)
            fitted_base = mapie._mapie_classifier.estimator_.single_estimator_

    else:
        if X_cal is None or y_cal is None:
            raise ValueError("Prefit mode (cv_folds=0) requires X_cal and y_cal.")
        logger.info("Prefit conformal: training on train set, calibrating on held-out cal set...")

        imputer.fit(np.vstack([X_trainval, X_cal]))
        X_tr_imp = imputer.transform(X_trainval)
        X_cal_imp = imputer.transform(X_cal)

        fitted_base = get_lgbm_model(task, lgbm_params)
        fitted_base.fit(X_tr_imp, y_trainval)

        if task == "regression":
            mapie = SplitConformalRegressor(
                estimator=fitted_base, confidence_level=confidence_level, prefit=True,
            )
        else:
            mapie = SplitConformalClassifier(
                estimator=fitted_base, confidence_level=confidence_level, prefit=True,
            )
        mapie.conformalize(X_cal_imp, y_cal)

    return mapie, imputer, fitted_base


def predict_conformal(mapie, imputer, X_test, task, alpha=0.1):
    """Generate conformal predictions."""
    X_imp = imputer.transform(X_test)
    if task == "regression":
        y_pred, y_pis = mapie.predict_interval(X_imp)
        return y_pred, y_pis   # y_pis: (n, 2, 1)
    else:
        y_pred, y_sets = mapie.predict_set(X_imp)
        return y_pred, y_sets


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_regression(y_true, y_pred, y_pis, alpha):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    lower, upper = y_pis[:, 0, 0], y_pis[:, 1, 0]
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    return {
        "RMSE":                       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":                        float(mean_absolute_error(y_true, y_pred)),
        "R2":                         float(r2_score(y_true, y_pred)),
        f"Coverage@{int((1-alpha)*100)}%": coverage,
        "Avg_Interval_Width":         float(np.mean(upper - lower)),
        "Target_Coverage":            1 - alpha,
    }


def evaluate_classification(y_true, y_pred, y_sets, alpha):
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
    )
    coverage = float(np.mean([
        y_true[i] in np.where(y_sets[i, :, 0])[0] for i in range(len(y_true))
    ]))
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except Exception:
        auc = float("nan")
    return {
        "Accuracy":                        float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy":               float(balanced_accuracy_score(y_true, y_pred)),
        "MCC":                             float(matthews_corrcoef(y_true, y_pred)),
        "ROC_AUC":                         auc,
        f"Coverage@{int((1-alpha)*100)}%": coverage,
        "Avg_Prediction_Set_Size":         float(np.mean(y_sets[:, :, 0].sum(axis=1))),
        "Target_Coverage":                 1 - alpha,
    }


def evaluate_regression_train(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return {
        "Train_R2":   float(r2_score(y_true, y_pred)),
        "Train_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Train_MAE":  float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_classification_train(y_true, y_pred, y_proba=None):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    )
    try:
        prob_pos = y_proba[:, 1] if y_proba is not None else y_pred
        auc = float(roc_auc_score(y_true, prob_pos))
    except Exception:
        auc = float("nan")
    return {
        "Train_Accuracy":          float(accuracy_score(y_true, y_pred)),
        "Train_Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Train_MCC":               float(matthews_corrcoef(y_true, y_pred)),
        "Train_ROC_AUC":           auc,
    }


def get_feature_importance(fitted_base, feature_names, top_n=50):
    vals = fitted_base.feature_importances_
    idx = np.argsort(vals)[::-1][:top_n]
    return pd.DataFrame({
        "feature": [feature_names[i] for i in idx],
        "importance": vals[idx],
    })


def plot_regression_results(y_true, y_pred, y_pis, split_labels, metrics,
                            output_dir, confidence_level, activity_col="activity",
                            r2_train=None):
    """Save a scatter plot of predicted vs experimental with conformal intervals."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not found — skipping scatter plot.  pip install matplotlib")
        return

    split_arr = np.array(split_labels)
    trainval_mask = split_arr == "trainval"
    test_mask = split_arr == "test"
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Train/val: plain scatter
    ax.scatter(
        y_true[trainval_mask], y_pred[trainval_mask],
        alpha=0.45, s=28, color="#4C9BE8", label="Train/Val", zorder=2,
    )

    # Test: scatter with conformal interval error bars
    if test_mask.any():
        y_lower = y_pis[test_mask, 0, 0]
        y_upper = y_pis[test_mask, 1, 0]
        y_err = np.vstack([y_pred[test_mask] - y_lower,
                           y_upper - y_pred[test_mask]])
        ax.errorbar(
            y_true[test_mask], y_pred[test_mask],
            yerr=y_err, fmt="o",
            color="#E8604C", alpha=0.75, markersize=5,
            elinewidth=0.8, capsize=2, capthick=0.8,
            label=f"Test ({int(confidence_level * 100)}% PI)", zorder=3,
        )

    # Diagonal line (perfect predictions)
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Metric annotation (test set, with optional train R² for gap visibility)
    cov_key = next((k for k in metrics if "Coverage" in k), None)
    lines = [f"RMSE     = {metrics['RMSE']:.3f}"]
    if r2_train is not None:
        lines.append(f"R² train = {r2_train:.3f}")
    lines.append(f"R² test  = {metrics['R2']:.3f}")
    if cov_key:
        lines.append(f"{cov_key} = {metrics[cov_key]:.3f}")
    ax.text(
        0.04, 0.96, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  alpha=0.8, edgecolor="lightgrey"),
    )

    ax.set_xlabel(f"Experimental {activity_col}", fontsize=12)
    ax.set_ylabel(f"Predicted {activity_col}", fontsize=12)
    ax.set_title(
        f"Predicted vs Experimental  ·  "
        f"{int(confidence_level * 100)}% conformal intervals (test set)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    out_path = output_dir / "scatter_pred_vs_exp.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Scatter plot saved to {out_path}")


# ─── Checkpoint I/O ───────────────────────────────────────────────────────────

def save_checkpoint(output_dir, mapie, imputer, fitted_base, feature_names,
                    args, metrics, lgbm_params, dataset_info):
    """
    Save a timestamped model checkpoint (.pkl) and a human-readable
    model card (.json) to output_dir.

    Filenames: <model_name>_<yyMMdd>.pkl / .json
    The 'latest' alias is also updated to <model_name>_latest.pkl.

    Returns the paths to both files.
    """
    import pickle
    import json
    import platform
    from datetime import datetime, timezone

    model_name = getattr(args, "model_name", None) or "model"
    timestamp  = datetime.now(timezone.utc).strftime("%y%m%d")   # yyMMdd
    stem       = f"{model_name}_{timestamp}"

    # ── Build model card ──────────────────────────────────────────────────────
    def _clean(v):
        """Make a value JSON-serialisable."""
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    card = {
        "lightprop_version": "1.0",
        "model_name":      model_name,
        "timestamp_utc":   timestamp,
        "environment": {
            "python":    platform.python_version(),
            "platform":  platform.platform(),
        },
        "training": {
            "input_file":    str(args.input),
            "smiles_col":    args.smiles_col,
            "id_col":        args.id_col,
            "activity_col":  args.activity_col,
            "task":          args.task,
            "threshold":     args.threshold,
            "seed":          args.seed,
        },
        "dataset": {
            "n_compounds_total":    dataset_info["n_total"],
            "n_valid_smiles":       dataset_info["n_valid"],
            "n_trainval":           dataset_info["n_trainval"],
            "n_test":               dataset_info["n_test"],
            "activity_mean":        _clean(dataset_info.get("activity_mean")),
            "activity_std":         _clean(dataset_info.get("activity_std")),
            "activity_min":         _clean(dataset_info.get("activity_min")),
            "activity_max":         _clean(dataset_info.get("activity_max")),
            **({"class_counts": dataset_info["class_counts"]}
               if "class_counts" in dataset_info else {}),
        },
        "features": {
            "types":         args.features,
            "fp_types":      args.fp_types,
            "fp_radius":     args.fp_radius,
            "fp_bits":       args.fp_bits,
            "n_features":    len(feature_names),
        },
        "conformal": {
            "alpha":          args.alpha,
            "target_coverage": 1 - args.alpha,
            "cv_folds":       args.cv_folds,
            "mode":           "cross-conformal" if args.cv_folds > 0 else "prefit",
        },
        "hpo": {
            "enabled":     args.hpo_trials > 0,
            "n_trials":    args.hpo_trials,
            "cv_folds":    args.hpo_cv_folds,
            "best_params": {k: _clean(v) for k, v in (lgbm_params or {}).items()},
        },
        "evaluation_metrics": {k: _clean(v) for k, v in metrics.items()},
    }

    # ── Write JSON card ───────────────────────────────────────────────────────
    card_path = output_dir / f"{stem}.json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)

    # ── Write pickle checkpoint ────────────────────────────────────────────────
    ckpt_path = output_dir / f"{stem}.pkl"
    payload = {
        "mapie":         mapie,
        "imputer":       imputer,
        "fitted_base":   fitted_base,
        "feature_names": feature_names,
        "task":          args.task,
        "alpha":         args.alpha,
        "cv_folds":      args.cv_folds,
        "lgbm_params":   lgbm_params,
        "timestamp_utc": timestamp,
        "card":          card,
    }
    with open(ckpt_path, "wb") as f:
        pickle.dump(payload, f)

    # Always-current alias — useful for scripting without knowing the timestamp
    latest_path = output_dir / f"{model_name}_latest.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"Checkpoint saved : {ckpt_path}")
    logger.info(f"Model card saved : {card_path}")
    logger.info(f"Latest alias     : {latest_path}")

    return ckpt_path, card_path


def load_checkpoint(path):
    """
    Load a lightprop checkpoint (.pkl) and print a summary of its model card.

    Returns the saved dict with keys:
        mapie, imputer, fitted_base, feature_names, task, alpha,
        cv_folds, lgbm_params, timestamp_utc, card
    """
    import pickle
    import json

    with open(path, "rb") as f:
        saved = pickle.load(f)

    card = saved.get("card", {})
    ts   = saved.get("timestamp_utc", "unknown")

    logger.info(f"\n{'='*55}")
    logger.info(f"  Loaded checkpoint: {Path(path).name}")
    logger.info(f"  Trained (UTC)    : {ts}")
    if card:
        tr = card.get("training", {})
        ft = card.get("features", {})
        cf = card.get("conformal", {})
        hp = card.get("hpo", {})
        logger.info(f"  Task             : {tr.get('task')}  |  "
                    f"Activity: {tr.get('activity_col')}")
        logger.info(f"  Features         : {ft.get('types')}  →  "
                    f"{ft.get('n_features')} features")
        logger.info(f"  Conformal mode   : {cf.get('mode')}  "
                    f"(cv={cf.get('cv_folds')}, α={cf.get('alpha')})")
        logger.info(f"  HPO              : {'yes — ' + str(hp.get('n_trials')) + ' trials' if hp.get('enabled') else 'no'}")
        em = card.get("evaluation_metrics", {})
        if em:
            logger.info("  Test metrics     : " +
                        "  ".join(f"{k}={v:.4f}" for k, v in em.items()
                                  if isinstance(v, float) and v is not None))
    logger.info(f"{'='*55}\n")

    return saved


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Confidence level (prompt if not supplied via CLI) ──────────────────────
    if args.confidence_level is None:
        while True:
            try:
                raw = input(
                    "\nConfidence level for conformal predictions "
                    "(e.g. 0.9 for 90%) [0.90]: "
                ).strip()
            except EOFError:
                raw = ""
            if not raw:
                args.confidence_level = 0.90
                break
            try:
                cl = float(raw)
                if 0.0 < cl < 1.0:
                    args.confidence_level = cl
                    break
                print("  Please enter a value strictly between 0 and 1 (e.g. 0.90).")
            except ValueError:
                print("  Invalid input — enter a number like 0.90.")
    args.alpha = 1.0 - args.confidence_level
    logger.info(f"Confidence level: {args.confidence_level:.1%}  (α = {args.alpha:.4g})")

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    for col in [args.smiles_col, args.id_col, args.activity_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    smiles = df[args.smiles_col].tolist()
    ids = df[args.id_col].tolist()
    y_raw = df[args.activity_col].values

    # ── Parse SMILES ───────────────────────────────────────────────────────────
    logger.info("Parsing SMILES...")
    mols, valid_idx = smiles_to_mol(smiles)
    if len(mols) < len(smiles):
        logger.warning(f"{len(smiles) - len(mols)} invalid SMILES skipped.")

    df_valid = df.iloc[valid_idx].copy().reset_index(drop=True)
    y = y_raw[valid_idx]
    ids_valid = [ids[i] for i in valid_idx]

    # ── Classification labelling ───────────────────────────────────────────────
    if args.task == "classification":
        if args.threshold is not None:
            y = (y >= args.threshold).astype(int)
            logger.info(
                f"Binarized at threshold {args.threshold}: "
                f"{y.sum()} active / {(1-y).sum()} inactive"
            )
        else:
            y = y.astype(int)
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Build features ─────────────────────────────────────────────────────────
    fp_config = {
        "fp_types": args.fp_types,
        "radius": args.fp_radius,
        "nbits": args.fp_bits,
    }
    X_raw = build_feature_matrix(mols, args.features, fp_config)
    X_clean = clean_features(X_raw)
    feature_names = list(X_clean.columns)
    X = X_clean.values
    logger.info(f"Final feature matrix: {X.shape}")

    # ── Outer split: trainval vs test ──────────────────────────────────────────
    # The test set is completely isolated — never seen during HPO or calibration.
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer

    stratify = y if args.task == "classification" else None
    idx_all = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    X_trainval, X_test = X[idx_trainval], X[idx_test]
    y_trainval, y_test = y[idx_trainval], y[idx_test]
    ids_test = [ids_valid[i] for i in idx_test]

    # Prefit mode needs a calibration split from trainval
    X_cal = y_cal = None
    if args.cv_folds == 0:
        strat_tv = y_trainval if args.task == "classification" else None
        idx_tv = np.arange(len(X_trainval))
        idx_tr, idx_cal_local = train_test_split(
            idx_tv, test_size=args.cal_size, random_state=args.seed, stratify=strat_tv
        )
        X_cal, y_cal = X_trainval[idx_cal_local], y_trainval[idx_cal_local]
        X_trainval, y_trainval = X_trainval[idx_tr], y_trainval[idx_tr]
        logger.info(
            f"Split (prefit): {len(X_trainval)} train | "
            f"{len(X_cal)} calibration | {len(X_test)} test"
        )
    else:
        logger.info(
            f"Split (cross-conformal cv={args.cv_folds}): "
            f"{len(X_trainval)} train+cal | {len(X_test)} test"
        )

    # ── HPO (inner loop — trainval only) ───────────────────────────────────────
    lgbm_params = None
    hpo_study = None

    if args.hpo_trials > 0:
        logger.info(
            f"\n{'='*60}\n"
            f"  Nested HPO: {args.hpo_trials} Optuna trials, "
            f"{args.hpo_cv_folds}-fold inner CV\n"
            f"  Outer test set is ISOLATED from HPO\n"
            f"{'='*60}"
        )

        # Impute first (HPO operates on the same imputed space as final model)
        hpo_imputer = SimpleImputer(strategy="median")
        hpo_imputer.fit(X_trainval)
        X_trainval_imp = hpo_imputer.transform(X_trainval)

        lgbm_params, hpo_study = run_hpo(
            X_trainval_imp, y_trainval,
            task=args.task,
            n_trials=args.hpo_trials,
            cv_folds=args.hpo_cv_folds,
            seed=args.seed,
        )

        save_hpo_report(hpo_study, output_dir, args.task)

        logger.info(
            f"\nHPO finished. Best hyperparameters:\n"
            + "\n".join(f"  {k}: {v}" for k, v in lgbm_params.items())
        )
    else:
        logger.info("HPO disabled (--hpo_trials 0). Using default LightGBM parameters.")

    # ── Train final conformal model on full trainval ────────────────────────────
    # HPO params carry over; conformal calibration is separate from HPO folds.
    logger.info("\nTraining final conformal model...")
    mapie, imputer, fitted_base = train_conformal_model(
        X_trainval, y_trainval,
        task=args.task,
        cv_folds=args.cv_folds,
        lgbm_params=lgbm_params,
        alpha=args.alpha,
        X_cal=X_cal,
        y_cal=y_cal,
    )

    # ── Train metrics (in-sample, no conformal wrapping) ───────────────────────
    # Coverage would be meaningless in-sample; only point-prediction metrics used.
    logger.info("Computing in-sample train metrics...")
    X_trainval_imp = imputer.transform(X_trainval)
    y_train_pred = fitted_base.predict(X_trainval_imp)
    if args.task == "regression":
        train_metrics = evaluate_regression_train(y_trainval, y_train_pred)
    else:
        y_train_proba = fitted_base.predict_proba(X_trainval_imp)
        train_metrics = evaluate_classification_train(y_trainval, y_train_pred, y_train_proba)

    # ── Evaluate on held-out test set ──────────────────────────────────────────
    # This is the outer evaluation — valid regardless of HPO or not.
    logger.info("Evaluating on held-out test set...")
    if args.task == "regression":
        y_pred, y_pis = predict_conformal(mapie, imputer, X_test, args.task, args.alpha)
        metrics = evaluate_regression(y_test, y_pred, y_pis, args.alpha)
        results_df = pd.DataFrame({
            "ID":       ids_test,
            "y_true":   y_test,
            "y_pred":   y_pred,
            "lower_PI": y_pis[:, 0, 0],
            "upper_PI": y_pis[:, 1, 0],
            "PI_width": y_pis[:, 1, 0] - y_pis[:, 0, 0],
        })
    else:
        y_pred, y_sets = predict_conformal(mapie, imputer, X_test, args.task, args.alpha)
        metrics = evaluate_classification(y_test, y_pred, y_sets, args.alpha)
        classes = fitted_base.classes_
        X_test_imp = imputer.transform(X_test)
        proba = fitted_base.predict_proba(X_test_imp)
        results_df = pd.DataFrame({
            "ID":                    ids_test,
            "y_true":                y_test,
            "y_pred":                y_pred,
            "prediction_set_size":   y_sets[:, :, 0].sum(axis=1),
        })
        for i, cls in enumerate(classes):
            results_df[f"prob_class{cls}"] = proba[:, i]

    # ── Full dataset predictions, table, and scatter plot ─────────────────────
    logger.info("Generating full dataset predictions...")
    test_set = set(idx_test.tolist())
    split_labels = ["test" if i in test_set else "trainval" for i in range(len(X))]
    X_all_imp = imputer.transform(X)

    if args.task == "regression":
        y_pred_all, y_pis_all = mapie.predict_interval(X_all_imp)
        pred_all_df = pd.DataFrame({
            "ID":              ids_valid,
            "SMILES":          df_valid[args.smiles_col].values,
            args.activity_col: y,
            "y_pred":          y_pred_all,
            "lower_PI":        y_pis_all[:, 0, 0],
            "upper_PI":        y_pis_all[:, 1, 0],
            "PI_width":        y_pis_all[:, 1, 0] - y_pis_all[:, 0, 0],
            "split":           split_labels,
        })
        plot_regression_results(
            y_true=y,
            y_pred=y_pred_all,
            y_pis=y_pis_all,
            split_labels=split_labels,
            metrics=metrics,
            output_dir=output_dir,
            confidence_level=args.confidence_level,
            activity_col=args.activity_col,
            r2_train=train_metrics.get("Train_R2"),
        )
    else:
        y_pred_all, y_sets_all = mapie.predict_set(X_all_imp)
        proba_all = fitted_base.predict_proba(X_all_imp)
        pred_all_df = pd.DataFrame({
            "ID":                  ids_valid,
            "SMILES":              df_valid[args.smiles_col].values,
            args.activity_col:     y,
            "y_pred":              y_pred_all,
            "prediction_set_size": y_sets_all[:, :, 0].sum(axis=1),
            "split":               split_labels,
        })
        for i, cls in enumerate(fitted_base.classes_):
            pred_all_df[f"prob_class{cls}"] = proba_all[:, i]

    pred_all_df.to_csv(output_dir / "predictions_all.csv", index=False)
    logger.info(f"Full predictions saved to {output_dir / 'predictions_all.csv'}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    results_df.to_csv(output_dir / "predictions_test.csv", index=False)
    combined_metrics = {
        **train_metrics,
        **{f"Test_{k}": v for k, v in metrics.items()},
    }
    pd.DataFrame([combined_metrics]).to_csv(output_dir / "metrics.csv", index=False)

    logger.info("\n=== Train Metrics (in-sample) ===")
    for k, v in train_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info("\n=== Test Metrics (held-out) ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    fi_df = get_feature_importance(fitted_base, feature_names, top_n=50)
    fi_df.to_csv(output_dir / "feature_importance.csv", index=False)

    if args.save_model:
        dataset_info = {
            "n_total":    len(smiles),
            "n_valid":    len(mols),
            "n_trainval": len(X_trainval),
            "n_test":     len(X_test),
        }
        if args.task == "regression":
            dataset_info.update({
                "activity_mean": float(np.mean(y)),
                "activity_std":  float(np.std(y)),
                "activity_min":  float(np.min(y)),
                "activity_max":  float(np.max(y)),
            })
        else:
            dataset_info["class_counts"] = {
                int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))
            }

        save_checkpoint(
            output_dir=output_dir,
            mapie=mapie,
            imputer=imputer,
            fitted_base=fitted_base,
            feature_names=feature_names,
            args=args,
            metrics=combined_metrics,
            lgbm_params=lgbm_params,
            dataset_info=dataset_info,
        )

    logger.info(f"\nAll outputs written to: {output_dir}")
    return metrics, results_df


# ─── Predict from saved model ──────────────────────────────────────────────────

def run_predict(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load_checkpoint prints the model card summary automatically
    saved = load_checkpoint(args.load_model)

    mapie         = saved["mapie"]
    imputer       = saved["imputer"]
    fitted_base   = saved.get("fitted_base") or saved.get("base_model")
    feature_names = saved["feature_names"]
    task          = saved["task"]
    alpha         = saved.get("alpha", 0.1)

    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    smiles = df[args.smiles_col].tolist()
    ids    = df[args.id_col].tolist()

    mols, valid_idx = smiles_to_mol(smiles)
    fp_config = {
        "fp_types": args.fp_types,
        "radius":   args.fp_radius,
        "nbits":    args.fp_bits,
    }
    X_raw   = build_feature_matrix(mols, args.features, fp_config)
    X_clean = clean_features(X_raw)

    # Align to training feature space
    for col in set(feature_names) - set(X_clean.columns):
        X_clean[col] = 0
    X     = X_clean[feature_names].values
    X_imp = imputer.transform(X)

    if task == "regression":
        y_pred, y_pis = mapie.predict_interval(X_imp)
        pred_df = pd.DataFrame({
            "ID":       [ids[i] for i in valid_idx],
            "SMILES":   [smiles[i] for i in valid_idx],
            "y_pred":   y_pred,
            "lower_PI": y_pis[:, 0, 0],
            "upper_PI": y_pis[:, 1, 0],
            "PI_width": y_pis[:, 1, 0] - y_pis[:, 0, 0],
        })
    else:
        y_pred, y_sets = mapie.predict_set(X_imp)
        proba = fitted_base.predict_proba(X_imp)
        pred_df = pd.DataFrame({
            "ID":                  [ids[i] for i in valid_idx],
            "SMILES":              [smiles[i] for i in valid_idx],
            "y_pred":              y_pred,
            "prediction_set_size": y_sets[:, :, 0].sum(axis=1),
        })
        for i, cls in enumerate(fitted_base.classes_):
            pred_df[f"prob_class{cls}"] = proba[:, i]

    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    logger.info(f"Predictions saved to {output_dir / 'predictions.csv'}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def _add_feature_args(p):
    p.add_argument("--features", nargs="+", choices=["mordred", "rdkit"],
                   default=["mordred", "rdkit"])
    p.add_argument("--fp_types", nargs="+",
                   choices=["morgan", "rdkit", "maccs", "atompair"],
                   default=["morgan"])
    p.add_argument("--fp_radius", type=int, default=2)
    p.add_argument("--fp_bits",   type=int, default=2048)


def parse_args():
    parser = argparse.ArgumentParser(
        description="lightprop: Conformal QSAR + nested Optuna HPO with LightGBM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────────
    train_p = subparsers.add_parser("train", help="Train a conformal model")

    train_p.add_argument("--input", "-i", required=True,
                         help="Input CSV with SMILES, IDs, and activity values")
    train_p.add_argument("--smiles_col",   default="SMILES")
    train_p.add_argument("--id_col",       default="ID")
    train_p.add_argument("--activity_col", default="pIC50")
    train_p.add_argument("--task", choices=["regression", "classification"],
                         default="regression")
    train_p.add_argument("--threshold", type=float, default=None,
                         help="Binarize activity >= threshold (classification only)")
    _add_feature_args(train_p)

    # Conformal
    train_p.add_argument("--confidence_level", type=float, default=None,
                         help="Conformal confidence level, e.g. 0.9 for 90%% coverage. "
                              "Prompted interactively if not specified.")
    train_p.add_argument("--cv_folds", type=int, default=5,
                         help="Cross-conformal folds (default 5). Use 0 for prefit mode.")
    train_p.add_argument("--test_size",  type=float, default=0.15,
                         help="Fraction held out as final test set")
    train_p.add_argument("--cal_size",   type=float, default=0.15,
                         help="Calibration set fraction (only used when --cv_folds 0)")

    # HPO
    hpo_group = train_p.add_argument_group("Hyperparameter Optimisation (Optuna)")
    hpo_group.add_argument("--hpo_trials",    type=int, default=0,
                           help="Number of Optuna trials (default 0 = HPO disabled). "
                                "Recommended: 50-100.")
    hpo_group.add_argument("--hpo_cv_folds",  type=int, default=3,
                           help="Inner CV folds for HPO scoring (default 3). "
                                "Independent from --cv_folds.")

    # Output
    train_p.add_argument("--output", "-o", default="lightprop_results")
    train_p.add_argument("--model_name", default="model",
                         help="Base name for checkpoint files, e.g. 'hERG_pIC50' "
                              "→ hERG_pIC50_250225.pkl  (default: 'model')")
    train_p.add_argument("--save_model",  action="store_true",
                         help="Pickle model for later inference")
    train_p.add_argument("--seed", type=int, default=42)

    # ── predict ────────────────────────────────────────────────────────────────
    pred_p = subparsers.add_parser("predict", help="Predict using a saved model")
    pred_p.add_argument("--input",      "-i", required=True)
    pred_p.add_argument("--load_model",       required=True)
    pred_p.add_argument("--smiles_col",  default="SMILES")
    pred_p.add_argument("--id_col",      default="ID")
    _add_feature_args(pred_p)
    pred_p.add_argument("--output", "-o", default="lightprop_predictions")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        run_pipeline(args)
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
