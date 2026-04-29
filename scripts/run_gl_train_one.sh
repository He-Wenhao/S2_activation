#!/bin/bash
# Launch one GL match-DH training run on a single GPU.
# Use as the command of a salloc -N 1 --gpus=1 -t 04:00:00 ... allocation.
# Pass seed as argv[1].

exec > >(tee -a /pscratch/sd/w/whe1/S2_activation/logs/gl_train_one.log) 2>&1

SEED=${1:-42}
LOG=/pscratch/sd/w/whe1/S2_activation/logs/gl_train_seed${SEED}.log

echo "==== one-seed launcher start at $(date), seed=${SEED} ===="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-(not set)}  NODE=${SLURM_NODELIST:-(not set)}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-(not set)}"

cd /pscratch/sd/w/whe1/S2_activation || exit 1
mkdir -p logs

source /etc/profile
module load pytorch/2.6.0-1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Launching seed=${SEED} → ${LOG}"
python3 -u experiments/expF_equiformerv2_qm9.py train \
    --act SiLU --grid default --target U0 --seed ${SEED} \
    --quadrature gl --n_beta 10 --n_alpha 9 \
    --epochs 50 \
    > "${LOG}" 2>&1
echo "$(date) | Seed ${SEED} done."
