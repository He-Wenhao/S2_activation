#!/bin/bash
#SBATCH -A m3706_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -J expF_eqv2
#SBATCH -o logs/expF_%A_%a.out
#SBATCH -e logs/expF_%A_%a.err

# ── Experiment F: EquiformerV2 on QM9 ──────────────────────────────────────
#
# Submit as a SLURM array job:
#   # Generate config list
#   python experiments/expF_equiformerv2_qm9.py sweep > scripts/expF_configs.txt
#   N=$(wc -l < scripts/expF_configs.txt)
#   sbatch --array=0-$((N-1)) scripts/run_expF.sh
#
# Or run a single config:
#   sbatch scripts/run_expF.sh --act SiLU --grid default --target U0 --seed 42
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

module load pytorch/2.6.0-1

cd /pscratch/sd/w/whe1/S2_activation
mkdir -p logs

# If running as array job, read config from file
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    CONFIG_FILE="scripts/expF_configs.txt"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: $CONFIG_FILE not found. Run: python experiments/expF_equiformerv2_qm9.py sweep > $CONFIG_FILE"
        exit 1
    fi
    # Read the config line for this array task
    ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE" | cut -f2-)
    echo "Array task ${SLURM_ARRAY_TASK_ID}: ${ARGS}"
else
    # Use command-line arguments (pass after sbatch script name)
    ARGS="${@:---act SiLU --grid default --target U0 --seed 42}"
    echo "Single run: ${ARGS}"
fi

echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# Check if already completed
ACT=$(echo "$ARGS" | grep -oP '(?<=--act )\S+')
GRID=$(echo "$ARGS" | grep -oP '(?<=--grid )\S+')
TARGET=$(echo "$ARGS" | grep -oP '(?<=--target )\S+')
SEED=$(echo "$ARGS" | grep -oP '(?<=--seed )\S+')
RESULT_DIR="results/expF/runs/${ACT}_${GRID}_${TARGET}_seed${SEED}"

if [ -f "${RESULT_DIR}/results.json" ]; then
    echo "Already completed: ${RESULT_DIR}/results.json exists. Skipping."
    exit 0
fi

srun python experiments/expF_equiformerv2_qm9.py train ${ARGS}

echo "Done: $(date)"
