#!/bin/bash
#SBATCH --account=dslab_jobs
#SBATCH --qos=anding-dslab_jobs
#SBATCH --partition=jobs
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=rf_lag_search.out
#SBATCH --error=rf_lag_search.err

echo "=== Job started on $(hostname) at $(date) ==="
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate n2o-gpu

python ~/N2O_Emissions_Prediction/baseline_experiments/rf_gpu_timeseries_optuna.py

echo "=== Job finished ==="
