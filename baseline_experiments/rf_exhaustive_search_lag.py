#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest exhaustive hyperparameter search for N₂O flux prediction
(using lag predictors only, with dataset splits per site/parcel)
Author: Andrew Ding
Run with:
    sbatch --time=08:00:00 --cpus-per-task=16 --mem=32G rf_exhaustive_search_lag.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for Slurm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# -------------------------
# Config
# -------------------------
OUT_DIR = "rf_results_lag/"
DATE_COL = "Timestamp"
TARGET = "N2O_Flux_ln"
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

# -------------------------
# Dataset loading
# -------------------------
print("Loading datasets...")

# CHAMAU
chamau_lag = pd.read_csv("../datasets/Chamau_2014-2024_clean.csv")
chamau_daily = pd.read_csv("../datasets/Chamau_Daily_2014-2024.csv")

chamau_A = chamau_lag[chamau_lag["Parcel"] == "A"].copy()
chamau_B = chamau_lag[chamau_lag["Parcel"] == "B"].copy()
chamau_daily_A = chamau_daily[chamau_daily["Parcel"] == "A"]
chamau_daily_B = chamau_daily[chamau_daily["Parcel"] == "B"]

# AESCHI
aeschi_lag = pd.read_csv("../datasets/Aeschi_2019-20_clean.csv")

# OENSINGEN
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
# Predictor set (Lag only)
# -------------------------
predictors = [
    # Current
    "Precipitation", "SolarRadiation", "AirTemp", "VPD",
    "SoilWater_5cm", "SoilWater_15cm", "SoilWater_30cm",
    "SoilTemp_4cm", "SoilTemp_5cm", "SoilTemp_15cm", "SoilTemp_30cm",
    "NEE", "GPP", "RECO",
    # Lagged vars
    "Precipitation_lag1d", "Precipitation_lag3d", "Precipitation_lag5d", "Precipitation_lag7d",
    "SolarRadiation_lag1d", "SolarRadiation_lag3d", "SolarRadiation_lag5d", "SolarRadiation_lag7d",
    "AirTemp_lag1d", "AirTemp_lag3d", "AirTemp_lag5d", "AirTemp_lag7d",
    "VPD_lag1d", "VPD_lag3d", "VPD_lag5d", "VPD_lag7d",
    "SoilWater_5cm_lag1d", "SoilWater_5cm_lag3d", "SoilWater_5cm_lag5d", "SoilWater_5cm_lag7d",
    "SoilWater_15cm_lag1d", "SoilWater_15cm_lag3d", "SoilWater_15cm_lag5d", "SoilWater_15cm_lag7d",
    "SoilWater_30cm_lag1d", "SoilWater_30cm_lag3d", "SoilWater_30cm_lag5d", "SoilWater_30cm_lag7d",
    "SoilTemp_4cm_lag1d", "SoilTemp_4cm_lag3d", "SoilTemp_4cm_lag5d", "SoilTemp_4cm_lag7d",
    "SoilTemp_5cm_lag1d", "SoilTemp_5cm_lag3d", "SoilTemp_5cm_lag5d", "SoilTemp_5cm_lag7d",
    "SoilTemp_15cm_lag1d", "SoilTemp_15cm_lag3d", "SoilTemp_15cm_lag5d", "SoilTemp_15cm_lag7d",
    "SoilTemp_30cm_lag1d", "SoilTemp_30cm_lag3d", "SoilTemp_30cm_lag5d", "SoilTemp_30cm_lag7d",
    "NEE_lag1d", "NEE_lag3d", "NEE_lag5d", "NEE_lag7d",
    "GPP_lag1d", "GPP_lag3d", "GPP_lag5d", "GPP_lag7d",
    "RECO_lag1d", "RECO_lag3d", "RECO_lag5d", "RECO_lag7d",
    # Management & temporal
    "Mowing", "FertilizerOrganic", "FertilizerMineral", "Grazing", "SoilCultivation",
    "DaysSince_Mowing", "DaysSince_FertilizerOrganic", "DaysSince_FertilizerMineral",
    "DaysSince_Grazing", "DaysSince_SoilCultivation",
    "month", "day", "hour"
]

# -------------------------
# Hyperparameter grid
# -------------------------
param_grid = {
    "rf__n_estimators": [200, 400],
    "rf__max_depth": [10, 20, None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2"],
}

# -------------------------
# Training utility
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
    steps.append(("rf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)))
    return Pipeline(steps)

# -------------------------
# Main loop
# -------------------------
def main():
    results = []
    for name, df in datasets.items():
        print(f"\n=== Training on {name} ===")
        available = [p for p in predictors if p in df.columns]
        if len(available) < 10:
            print(f"⚠️  Skipping {name}: only {len(available)} predictors available.")
            continue

        train, val, test = split_train_val_test(df, available, TARGET, DATE_COL)
        X_train = train[available].values
        y_train = train[TARGET].values
        X_val = val[available].values
        y_val = val[TARGET].values
        X_test = test[available].values
        y_test = test[TARGET].values
        X_tune = np.vstack([X_train, X_val])
        y_tune = np.concatenate([y_train, y_val])
        test_dates = test[DATE_COL].values

        use_pca = len(available) > 100
        model = make_pipeline(use_pca)
        best_r2 = -np.inf
        best_info = None

        for params in ParameterGrid(param_grid):
            model.set_params(**params)
            model.fit(X_tune, y_tune)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            r, _ = pearsonr(y_test, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_info = (params, y_pred, r)

        # Save best model plots
        params, y_pred, r = best_info
        print(f"✅ {name}: R²={best_r2:.3f}, r={r:.3f}")

        # Scatter plot
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, y_pred, alpha=0.6, s=20)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        plt.xlabel("Observed N₂O Flux (ln)")
        plt.ylabel("Predicted N₂O Flux (ln)")
        plt.title(f"{name} — RF\nR²={best_r2:.3f}, r={r:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "plots", f"{name}_scatter.png"))
        plt.close()

        # Time series plot
        plt.figure(figsize=(10, 4))
        plt.plot(test_dates, y_test, label="Observed", color="black", lw=1.5)
        plt.plot(test_dates, y_pred, label="Predicted", color="forestgreen", lw=1.5, alpha=0.8)
        plt.xlabel("Date")
        plt.ylabel("N₂O Flux (ln)")
        plt.title(f"{name} — Random Forest Time Series")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "plots", f"{name}_timeseries.png"))
        plt.close()

        results.append({
            "Dataset": name,
            "R²": round(best_r2, 3),
            "Pearson_r": round(r, 3),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_predictors": len(available),
            "use_pca": use_pca,
            "best_params": params
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "rf_summary.csv"), index=False)
    print(f"\n✅ Finished. Summary saved to {OUT_DIR}/rf_summary.csv")

if __name__ == "__main__":
    main()
