#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cudf
import cupy as cp
from cuml.ensemble import RandomForestRegressor
from cuml.decomposition import PCA as cuPCA

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import optuna

# -------------------------
# Config
# -------------------------
OUT_DIR = "rf_results_lag/"
DATE_COL = "Timestamp"
TARGET = "N2O_Flux_ln"
RANDOM_STATE = 42
N_TRIALS = 20   # <-- adjust if you want more tuning depth

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

# -------------------------
# Load Datasets
# -------------------------
print("Loading datasets...")

chamau_lag = pd.read_csv("../datasets/Chamau_2014-2024_clean.csv")
chamau_daily = pd.read_csv("../datasets/Chamau_Daily_2014-2024.csv")
chamau_A = chamau_lag[chamau_lag["Parcel"] == "A"].copy()
chamau_B = chamau_lag[chamau_lag["Parcel"] == "B"].copy()
chamau_daily_A = chamau_daily[chamau_daily["Parcel"] == "A"]
chamau_daily_B = chamau_daily[chamau_daily["Parcel"] == "B"]

aeschi_lag = pd.read_csv("../datasets/Aeschi_2019-20_clean.csv")
oensingen_lag_1 = pd.read_csv("../datasets/Oensingen_2018-19_clean.csv")
oensingen_lag_2 = pd.read_csv("../datasets/Oensingen_2021-23_clean.csv")

datasets = {
    "Chamau": chamau_lag,
    "Chamau A": chamau_A,
    "Chamau B": chamau_B,
    "Aeschi": aeschi_lag,
    "Oensingen 1": oensingen_lag_1,
    "Oensingen 2": oensingen_lag_2,
}

# -------------------------
# Predictor List (KEEP EXACTLY AS BEFORE)
# -------------------------
predictors = [
    # paste your predictor list here unchanged
]

# -------------------------
# Splitting Function
# -------------------------
def split_train_val_test(df, predictors, target, date_col):
    df = df.dropna(subset=predictors + [target]).sort_values(date_col)
    n = len(df)
    n_test, n_val = int(n * 0.15), int(n * 0.15)
    train = df.iloc[: n - n_val - n_test]
    val   = df.iloc[n - n_val - n_test : n - n_test]
    test  = df.iloc[n - n_test :]
    return train, val, test

# -------------------------
# Train GPU Model
# -------------------------
def train_gpu_model(X_gpu, y_gpu, use_pca, params):
    if use_pca:
        pca = cuPCA(n_components=0.95)
        X_gpu = pca.fit_transform(X_gpu)
    else:
        pca = None

    rf = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        max_features=params["max_features"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=RANDOM_STATE,
        n_streams=8  # more GPU parallelism
    )
    rf.fit(X_gpu, y_gpu)

    return rf, pca

# -------------------------
# Main Loop
# -------------------------
def main():
    results = []

    for name, df in datasets.items():
        print(f"\n=== Training on {name} ===")

        available = [p for p in predictors if p in df.columns]
        if len(available) < 8:
            print(f"Skipping {name}: only {len(available)} predictors exist.")
            continue

        train, val, test = split_train_val_test(df, available, TARGET, DATE_COL)

        # Convert to NumPy first (easy), then GPU (cuDF)
        X_train = train[available].to_numpy()
        y_train = train[TARGET].to_numpy()
        X_val   = val[available].to_numpy()
        y_val   = val[TARGET].to_numpy()
        X_test  = test[available].to_numpy()
        y_test  = test[TARGET].to_numpy()

        X_tune = np.vstack([X_train, X_val])
        y_tune = np.concatenate([y_train, y_val])

        X_tune_gpu = cudf.DataFrame(X_tune)
        y_tune_gpu = cudf.Series(y_tune)
        X_val_gpu  = cudf.DataFrame(X_val)
        X_test_gpu = cudf.DataFrame(X_test)

        use_pca = len(available) > 100

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 900),
                "max_depth": trial.suggest_categorical("max_depth", [None] + list(range(6, 30))),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
            }
            rf, pca = train_gpu_model(X_tune_gpu, y_tune_gpu, use_pca, params)
            X_eval = pca.transform(X_val_gpu) if pca else X_val_gpu
            y_val_pred = cp.asnumpy(rf.predict(X_eval))
            return r2_score(y_val, y_val_pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS)

        best_params = study.best_params
        rf, pca = train_gpu_model(X_tune_gpu, y_tune_gpu, use_pca, best_params)

        X_eval_test = pca.transform(X_test_gpu) if pca else X_test_gpu
        y_pred = cp.asnumpy(rf.predict(X_eval_test))

        r2 = r2_score(y_test, y_pred)
        r, _ = pearsonr(y_test, y_pred)

        print(f"✅ {name}: R²={r2:.3f}, r={r:.3f}")

        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title(f"{name} — R²={r2:.3f}, r={r:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "plots", f"{name}_scatter.png"))
        plt.close()

        results.append({
            "Dataset": name,
            "R²": round(r2, 3),
            "Pearson_r": round(r, 3),
            "n_predictors": len(available),
            "use_pca": use_pca,
            "best_params": best_params
        })

    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "rf_summary.csv"), index=False)
    print("\n✅ Finished. Summary saved.")

if __name__ == "__main__":
    main()
