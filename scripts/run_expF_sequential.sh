#!/bin/bash
# Run a range of expF configs sequentially on one node.
# Supports checkpoint resume — safe to re-run if wall-timed out.
# Usage: bash scripts/run_expF_sequential.sh <START_IDX> <END_IDX>
set -euo pipefail

START=$1
END=$2
CONFIG_FILE="/pscratch/sd/w/whe1/S2_activation/scripts/expF_configs.txt"

cd /pscratch/sd/w/whe1/S2_activation
module load pytorch/2.6.0-1

for i in $(seq $START $END); do
    ARGS=$(sed -n "$((i + 1))p" "$CONFIG_FILE" | cut -f2-)
    echo "=== Config $i: $ARGS ==="

    # Parse args to build result dir name and check if already done
    ACT=$(echo "$ARGS" | grep -oP '(?<=--act )\S+')
    GRID=$(echo "$ARGS" | grep -oP '(?<=--grid )\S+')
    TARGET=$(echo "$ARGS" | grep -oP '(?<=--target )\S+')
    SEED=$(echo "$ARGS" | grep -oP '(?<=--seed )\S+')
    RESULT_DIR="results/expF/runs/${ACT}_${GRID}_${TARGET}_seed${SEED}"

    if [ -f "${RESULT_DIR}/results.json" ]; then
        echo "  -> Already completed, skipping."
        continue
    fi

    echo "  -> Starting at $(date)"
    python experiments/expF_equiformerv2_qm9.py train $ARGS
    echo "  -> Finished at $(date)"
    echo ""
done

echo "All configs $START-$END done!"
