#!/bin/bash
# Launch 4 parallel GL training runs using srun to spawn each task
# on its own GPU within a 4-GPU/4-task interactive allocation.

exec > >(tee -a /pscratch/sd/w/whe1/S2_activation/logs/gl_train_launcher.log) 2>&1

echo "==== launcher start at $(date) ===="
echo "PWD=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-(not set)}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-(not set)}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-(not set)}"

cd /pscratch/sd/w/whe1/S2_activation || exit 1
mkdir -p logs

source /etc/profile
module load pytorch/2.6.0-1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=(123 456 789 1024)

for i in 0 1 2 3; do
    seed=${SEEDS[$i]}
    log_file="logs/gl_train_seed${seed}.log"
    echo "$(date) | srun -n 1 task ${i} → seed=${seed}, log=${log_file}"

    srun -n 1 --gpus=1 --gpu-bind=closest --exclusive \
         --output="${log_file}" --error="${log_file}" \
        python3 -u experiments/expF_equiformerv2_qm9.py train \
            --act SiLU --grid default --target U0 --seed $seed \
            --quadrature gl --n_beta 10 --n_alpha 9 \
            --epochs 50 \
        &
done

echo "$(date) | All 4 srun-spawned. Waiting..."
wait
echo "$(date) | All 4 done."
