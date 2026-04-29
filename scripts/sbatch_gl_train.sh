#!/bin/bash
#SBATCH -A m3706_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH -J expG_GL_train
#SBATCH -o /pscratch/sd/w/whe1/S2_activation/logs/sbatch_gl_train_%j.out
#SBATCH -e /pscratch/sd/w/whe1/S2_activation/logs/sbatch_gl_train_%j.err

set -u
cd /pscratch/sd/w/whe1/S2_activation
mkdir -p logs

source /etc/profile
module load pytorch/2.6.0-1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==== sbatch_gl_train start at $(date) ===="
echo "JOB_ID=$SLURM_JOB_ID  NODE=$SLURM_NODELIST"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-(unset)}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-(unset)}"
echo "Initial CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)}"

# Make all 4 GPUs visible by clearing the restrictive mask SLURM may
# inherit from a single-task alloc.
unset CUDA_VISIBLE_DEVICES
echo "After unset, nvidia-smi -L:"
nvidia-smi -L 2>&1 | head -10

SEEDS=(123 456 789 1024)

for i in 0 1 2 3; do
    seed=${SEEDS[$i]}
    gpu=$i
    log_file="logs/gl_train_seed${seed}.log"
    echo "$(date) | Launching seed=${seed} on GPU ${gpu} -> ${log_file}"

    CUDA_VISIBLE_DEVICES=$gpu \
        python3 -u experiments/expF_equiformerv2_qm9.py train \
            --act SiLU --grid default --target U0 --seed $seed \
            --quadrature gl --n_beta 10 --n_alpha 9 \
            --epochs 50 \
        > "$log_file" 2>&1 &
done

echo "$(date) | All 4 launched. Waiting..."
wait
echo "$(date) | All 4 done."
