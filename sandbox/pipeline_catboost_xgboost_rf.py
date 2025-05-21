"""
pipeline_catboost_xgboost_rf.py

Modular script to reduce MAE in a tabular regression problem.
Sections are ordered from MOST to LEAST impactful according to prior analysis:

 1. Hyperparameter tuning with Optuna (CatBoost ⟶ XGBoost ⟶ RandomForest)
 2. Feature engineering: target encoding, historical aggregates, interactions, temporal decomposition
 3. Robust validation (Stratified K‑Fold / Repeated CV / Time-based CV)
 4. Ensembling & stacking (weighted blending of best models)
 5. Local error analysis with SHAP
 6. Outlier handling with IsolationForest
 7. Optional state-of-the-art models (TabNet / FT‑Transformer)

Requirements:
    pip install pandas numpy scikit-learn optuna catboost xgboost category_encoders shap pytorch-tabnet
Date: 2025‑05‑20
"""

# ============================
# 0. BASIC IMPORTS
# ============================
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb
import optuna
import shap
from category_encoders import TargetEncoder

import warnings

warnings.filterwarnings("ignore")

# ============================
# 1. GLOBAL CONFIGURATION
# ============================
RANDOM_STATE = 42
N_FOLDS = 12
N_TRIALS_CATBOOST = 25
N_TRIALS_XGBOOST = 25
ID_COLS: List[str] = []
DATE_COLS: List[str] = []
CATEGORICAL_COLS: List[str] = []
NUM_COLS: List[str] = []
TARGET = "simulated_loan_amount"

# ============================
# 2. UTILITIES
# ============================


def train_test_split_df(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test = df.sample(frac=test_size, random_state=RANDOM_STATE)
    train = df.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cats = [
        c
        for c in df.columns
        if df[c].dtype == "object" or df[c].dtype.name == "category"
    ]
    nums = [c for c in df.columns if c not in cats + [TARGET] + ID_COLS]
    return cats, nums


def is_gpu_available():
    try:
        from catboost.utils import get_gpu_device_count

        return get_gpu_device_count() > 0
    except ImportError:
        return False


# ============================
# 3. FEATURE ENGINEERING
# ============================


def make_date_features(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    for col in date_cols:
        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_dow"] = dt.dt.dayofweek
        df[f"{col}_weekofyear"] = dt.dt.isocalendar().week.astype("int")
    return df


def add_interaction_terms(
    df: pd.DataFrame, num_cols: List[str], max_pairs: int = 20
) -> pd.DataFrame:
    corrs = df[num_cols].corrwith(df[TARGET]).abs().sort_values(ascending=False)
    top = corrs.index.tolist()[:max_pairs]
    for i, col1 in enumerate(top):
        for col2 in top[i + 1 :]:
            name = f"{col1}_x_{col2}"
            df[name] = df[col1] * df[col2]
    return df


def add_group_aggregates_leak_free(
    train_df: pd.DataFrame, test_df: pd.DataFrame, group_cols: List[str], target: str
):
    for col in group_cols:
        grouped = train_df.groupby(col)[target].agg(["mean", "median", "std"])
        grouped.columns = [f"{col}_{stat}" for stat in grouped.columns]
        train_df = train_df.merge(grouped, how="left", left_on=col, right_index=True)
        test_df = test_df.merge(grouped, how="left", left_on=col, right_index=True)
    return train_df, test_df


# ============================
# 4. TARGET / CATBOOST ENCODING
# ============================


def build_preprocessor(cat_cols, num_cols):
    cat_enc = TargetEncoder(cols=cat_cols, smoothing=0.3)
    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer([("cat", cat_enc, cat_cols), ("num", num_pipe, num_cols)])


# ============================
# 5. OPTUNA HPO FOR CATBOOST
# ============================
def objective_catboost(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    task_type: str,
) -> float:
    common = {
        "iterations": trial.suggest_int("iterations", 500, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
    }

    gpu_fix = {
        "task_type": "GPU",
        "devices": "0",
        "border_count": 254,
        "bootstrap_type": "MVS",
        "subsample": 0.8,
        "metric_period": 50,
    }

    params = (
        {**common, **gpu_fix}
        if task_type.upper() == "GPU"
        else {**common, "task_type": "CPU"}
    )

    cat_names = [
        X.columns[c] if isinstance(c, (int, np.integer)) else c for c in categorical
    ]
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mae_scores = []

    for tr_idx, va_idx in cv.split(X):
        model = cb.CatBoostRegressor(**params, cat_features=cat_names)

        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        X_tr[cat_names] = X_tr[cat_names].astype("string")
        X_va[cat_names] = X_va[cat_names].astype("string")

        model.fit(X_tr, y.iloc[tr_idx], eval_set=(X_va, y.iloc[va_idx]), verbose=False)
        preds = model.predict(X_va)
        mae_scores.append(mean_absolute_error(y.iloc[va_idx], preds))

    return float(np.mean(mae_scores))


def tune_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    categorical: List,
    use_gpu: bool | None = None,
):
    cat_names = [
        X.columns[c] if isinstance(c, (int, np.integer)) else c for c in categorical
    ]

    if use_gpu is None:
        task_type = "GPU" if is_gpu_available() else "CPU"
    else:
        task_type = "GPU" if use_gpu else "CPU"

    def _objective(trial: optuna.Trial):
        return objective_catboost(trial, X, y, cat_names, task_type)

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=N_TRIALS_CATBOOST, show_progress_bar=True)

    print("CatBoost best MAE:", study.best_value)
    print("Best params:", study.best_params)

    best_model = cb.CatBoostRegressor(
        **study.best_params,
        loss_function="MAE",
        random_seed=RANDOM_STATE,
        cat_features=cat_names,
        task_type=task_type,
        verbose=False,
    )

    X_cat = X.copy()
    X_cat[cat_names] = X_cat[cat_names].astype("string").fillna("NA")

    best_model.fit(X_cat, y)
    return best_model


# ============================
# 6. OPTUNA HPO FOR XGBOOST
# ============================
def objective_xgboost(
    trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, xgb_params: dict
):
    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        **xgb_params,
    }

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    maes = []

    for train_idx, val_idx in cv.split(X):
        model = xgb.XGBRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50,
        )
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        preds = model.predict(X.iloc[val_idx])
        maes.append(mean_absolute_error(y.iloc[val_idx], preds))

    return np.mean(maes)


def tune_xgboost(X: pd.DataFrame, y: pd.Series, xgb_params: dict):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective_xgboost(t, X, y, xgb_params),
        n_trials=N_TRIALS_XGBOOST,
        show_progress_bar=True,
    )
    print("XGBoost best MAE:", study.best_value)
    print("Best params:", study.best_params)
    best_model = xgb.XGBRegressor(
        **study.best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        **xgb_params,
    )
    best_model.fit(X, y, eval_set=[(X, y)], verbose=False)
    return best_model


# ============================
# 7. BASIC RANDOM FOREST
# ============================


def train_random_forest(X: pd.DataFrame, y: pd.Series):
    model = RandomForestRegressor(
        n_estimators=1500,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model


# ============================
# 8. ENSEMBLE BLENDING
# ============================


def blend_predictions(
    models: List, X: pd.DataFrame, weights: List[float] | None = None
):
    if weights is None:
        weights = [1 / len(models)] * len(models)
    preds = np.zeros(len(X))
    for m, w in zip(models, weights):
        preds += w * m.predict(X)
    return preds


# ============================
# 9. SHAP ANALYSIS
# ============================


def explain_with_shap(
    model, X: pd.DataFrame, out_file: Path = Path("../reports/figures/shap_summary.png")
):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False, plot_size=(10, 6))
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(out_file)
    print(f"SHAP summary saved to {out_file}")


# ============================
# 10. OUTLIER DETECTION
# ============================


def remove_outliers_iforest(df: pd.DataFrame, contamination: float = 0.01):
    clf = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    preds = clf.fit_predict(df[NUM_COLS])
    mask_inliers = preds == 1
    removed = len(df) - mask_inliers.sum()
    print(f"Outliers detected and removed: {removed}")
    return df.loc[mask_inliers].reset_index(drop=True)


# ============================
# 11. ADVANCED TABULAR MODEL (OPTIONAL)
# ============================

try:
    from pytorch_tabnet.tab_model import TabNetRegressor

    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False


def train_tabnet(
    X: pd.DataFrame,
    y: pd.Series,
    use_gpu: bool | None = None,
    max_epochs: int = 120,
    patience: int = 15,
):
    if not HAS_TABNET:
        raise ImportError("pytorch-tabnet no instalado")

    import torch

    device = (
        "cuda"
        if ((use_gpu or use_gpu is None) and torch.cuda.is_available())
        else "cpu"
    )
    print(f"🚀 TabNet entrenará en: {device}")

    model = TabNetRegressor(
        n_d=48,
        n_a=48,
        n_steps=8,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=1e-3, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=50, gamma=0.75),
        seed=RANDOM_STATE,
        device_name=device,
        verbose=10,
    )

    y_arr = y.to_numpy().reshape(-1, 1)

    model.fit(
        X.values,
        y_arr,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
    )
    return model


# ============================
# 12. MAIN PIPELINE
# ============================


def run_pipeline(DATA_PATH: str):
    print("📥 Loading data from file...")
    df = pd.read_csv(DATA_PATH)

    use_gpu = is_gpu_available()
    if use_gpu:
        print("🚀 GPU disponible: se utilizará para acelerar el entrenamiento.")
        xgb_params = {
            "tree_method": "gpu_hist",
            "device": "cuda",
            "predictor": "gpu_predictor",
        }
    else:
        print("⚠️ No se detectó GPU: el entrenamiento se realizará en CPU.")
        xgb_params = {
            "tree_method": "hist",
            "device": "cpu",
            "predictor": "cpu_predictor",
        }

    print("🔍 Detecting categorical and numerical columns...")
    global CATEGORICAL_COLS, NUM_COLS
    if not CATEGORICAL_COLS or not NUM_COLS:
        CATEGORICAL_COLS, NUM_COLS = get_feature_lists(df)

    print("🧠 Performing feature engineering...")
    df = make_date_features(df, DATE_COLS)
    df = add_interaction_terms(df, NUM_COLS)

    print("🧹 Removing outliers (optional)...")
    df = remove_outliers_iforest(df)

    print("✂️ Splitting data into train and test sets...")
    train_df, test_df = train_test_split_df(df)

    cat_cols_for_agg = [c for c in CATEGORICAL_COLS if c not in ID_COLS]
    train_df, test_df = add_group_aggregates_leak_free(
        train_df, test_df, cat_cols_for_agg, TARGET
    )

    cat_cols_valid = [c for c in CATEGORICAL_COLS if c in train_df.columns]
    train_df[cat_cols_valid] = train_df[cat_cols_valid].astype("string").fillna("NA")
    test_df[cat_cols_valid] = test_df[cat_cols_valid].astype("string").fillna("NA")

    y_train = train_df.pop(TARGET)
    y_test = test_df.pop(TARGET)

    print("⚙️ Building preprocessor...")
    num_cols_valid = [c for c in NUM_COLS if c in train_df.columns]
    [train_df.columns.get_loc(c) for c in cat_cols_valid]
    preprocessor = build_preprocessor(cat_cols_valid, num_cols_valid)
    X_train = preprocessor.fit_transform(train_df, y_train)
    X_test = preprocessor.transform(test_df)

    print("🐱 Tuning CatBoost...")
    cat_model = tune_catboost(train_df, y_train, cat_cols_valid, use_gpu=False)

    print("📦 Tuning XGBoost...")
    xgb_model = tune_xgboost(pd.DataFrame(X_train), y_train, xgb_params)

    print("🌲 Training Random Forest...")
    rf_model = train_random_forest(pd.DataFrame(X_train), y_train)
    rf_preds_eval = rf_model.predict(pd.DataFrame(X_test))
    rf_mae = mean_absolute_error(y_test, rf_preds_eval)
    print(f"📊 MAE Random Forest: {rf_mae:0.4f}")

    print("🔗 Performing model ensemble...")
    cat_preds = cat_model.predict(test_df)
    xgb_preds = xgb_model.predict(pd.DataFrame(X_test))
    rf_preds = rf_model.predict(pd.DataFrame(X_test))

    weights = [0.5, 0.3, 0.2]
    ensemble_preds = (
        weights[0] * cat_preds + weights[1] * xgb_preds + weights[2] * rf_preds
    )

    print("📏 Calculating MAE for the ensemble...")
    mae = mean_absolute_error(y_test, ensemble_preds)
    print(f"\n📊 MAE of the Ensemble: {mae:0.4f}\n")

    print("🧠 Running SHAP interpretability analysis...")
    train_df_cat = train_df.copy()
    explain_with_shap(
        cat_model,
        train_df_cat.drop(columns=[TARGET], errors="ignore"),
        out_file=Path("../reports/figures/shap_summary.png"),
    )

    if HAS_TABNET and use_gpu:
        print("🚀 Training TabNet with GPU...")
        tabnet_model = train_tabnet(pd.DataFrame(X_train), y_train, use_gpu=True)
        tabnet_preds = tabnet_model.predict(X_test)
        mae_tabnet = mean_absolute_error(y_test, tabnet_preds)
        print(f"📊 MAE TabNet: {mae_tabnet:0.4f}")
    else:
        print("ℹ️  TabNet no se entrenará porque no hay GPU disponible.")

    print("✅ Pipeline completed 🏁")


if __name__ == "__main__":
    DATA_PATH = "../data/silver/df_model.csv"
    run_pipeline(DATA_PATH)
