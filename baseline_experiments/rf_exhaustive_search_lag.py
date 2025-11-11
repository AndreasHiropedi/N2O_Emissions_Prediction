#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cudf
from cuml.ensemble import RandomForestRegressor  # GPU RF
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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
N_ITER = 20

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

# -------------------------
# Dataset loading
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
# Predictor list
# -------------------------
predictors = [ ... same list as before ... ]  # <-- keep your predictor list exactly

# -------------------------
# Train/val/test split
# -------------------------
def split_train_val_test(df, predictors, target, date_col):
    df = df.dropna(subset=predictors + [target]).sort_values(date_col)
    n = len(df)
    n_test, n_val = int(n * 0.15), int(n * 0.15)
    train = df.iloc[: n - n_val - n_test]
    val   = df.iloc[n - n_val - n_test : n - n_test]
    test  = df.iloc[n - n_test :]
    return train, val, test

def make_pipeline(use_pca):
    steps = []
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95)))
    steps.append(("rf", RandomForestRegressor(random_state=RANDOM_STATE)))
    return Pipeline(steps)

# -------------------------
# Main
# -------------------------
def main():
    results = []

    for name, df in datasets.items():
        print(f"\n=== Training on {name} ===")

        available = [p for p in predictors if p in df.columns]
        if len(available) < 10:
            print(f"Skipping {name}: only {len(available)} predictors exist.")
            continue

        train, val, test = split_train_val_test(df, available, TARGET, DATE_COL)

        X_train = train[available].values
        y_train = train[TARGET].values
        X_val = val[available].values
        y_val = val[TARGET].values
        X_test = test[available].values
        y_test = test[TARGET].values

        # Combine train+val for tuning
        X_tune = np.vstack([X_train, X_val])
        y_tune = np.concatenate([y_train, y_val])

        # Convert to GPU data formats
        X_tune_gpu = cudf.DataFrame(X_tune)
        y_tune_gpu = cudf.Series(y_tune)
        X_val_gpu = cudf.DataFrame(X_val)
        X_test_gpu = cudf.DataFrame(X_test)

        use_pca = len(available) > 100
        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial):
            params = {
                "rf__n_estimators": trial.suggest_int("rf__n_estimators", 200, 900),
                "rf__max_depth": trial.suggest_categorical("rf__max_depth", [None] + list(range(6, 30))),
                "rf__max_features": trial.suggest_categorical("rf__max_features",
                                                               ["sqrt", "log2"] + list(np.linspace(0.3, 0.8, 4))),
                "rf__min_samples_split": trial.suggest_int("rf__min_samples_split", 2, 15),
                "rf__min_samples_leaf": trial.suggest_int("rf__min_samples_leaf", 1, 12),
            }

            model = make_pipeline(use_pca)
            model.set_params(**params)
            model.fit(X_tune_gpu, y_tune_gpu)

            y_val_pred = np.asarray(model.predict(X_val_gpu))
            return r2_score(y_val, y_val_pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_ITER)
        best_params = study.best_params

        best_model = make_pipeline(use_pca)
        best_model.set_params(**best_params)
        best_model.fit(X_tune_gpu, y_tune_gpu)

        y_pred = np.asarray(best_model.predict(X_test_gpu))

        r2 = r2_score(y_test, y_pred)
        r, _ = pearsonr(y_test, y_pred)

        print(f"✅ {name}: R²={r2:.3f}, r={r:.3f}")

        # Plots (unchanged)
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
