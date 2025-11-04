#!/bin/bash
#SBATCH --account=dslab
#SBATCH --partition=jobs          # use the GPU job queue (7 days limit)
#SBATCH --time=08:00:00           # walltime
#SBATCH --output=rf_lag_search.out
#SBATCH --error=rf_lag_search.err

# Optional but helpful diagnostics
echo "=== Job started on $(hostname) at $(date) ==="
echo "Current working directory: $(pwd)"
echo "GPU Info:"
nvidia-smi || echo "No GPU detected (CPU job?)"

# --- Activate Conda environment ---
source ~/miniconda3/bin/activate /home/anding/miniconda3/envs/dsl
echo "Activated Conda env: $CONDA_DEFAULT_ENV"
which python
python --version

# --- Run your script ---
python ~/N2O_Emissions_Prediction/baseline_experiments/rf_exhaustive_search_lag.py

echo "=== Job finished at $(date) ==="
