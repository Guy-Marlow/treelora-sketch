#!/bin/bash

#SBATCH --job-name=vit_comparison_seeds
#SBATCH --time=36:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/treelora/vit_cl_logs/comparison/slurm_output.log
#SBATCH --error=/data/gmar762/research/continuous_learning/treelora/vit_cl_logs/comparison/slurm_error.log
#SBATCH --gres=gpu:2

cd /data/gmar762/research/continuous_learning/treelora/training

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora

python3 ../scripts/setup.py \
    --ptm_dir  ../PTM/vit-base-patch16-224-in21k \
    --data_root ../data

# Comparison of Sketched LoRA (constant adapter cost) vs O-LoRA (linearly growing cost).
#
# Both methods use lora_r=8, lora_depth=-1 (all 24 Q+V linears across 12 blocks).
# lora_depth=-1 maximises the observable FLOP/memory divergence between methods.
#
# Three seeds are run sequentially.  All log/result paths are rooted under
# seed_${SEED}/ so the runs are fully isolated within a single monolithic job log.

for SEED in 1 2 3; do

    RUN=seed_${SEED}
    echo "========================================================"
    echo "  Starting seed ${SEED} / 3   (run dir: ${RUN})"
    echo "========================================================"

    # -----------------------------------------------------------------------
    # Block 1: Sketched LoRA
    # -----------------------------------------------------------------------

    python3 vit_cl_train.py \
        --model_path    ../PTM/vit-base-patch16-224-in21k \
        --dataset       cifar100 \
        --data_root     ../data \
        --method        sketched_lora \
        --lora_r        8 \
        --lora_depth    5 \
        --svd_rank      8 \
        --svd_oversample 4 \
        --svd_period     2 \
        --seed          $SEED \
        --output_dir    ../runs/comparison/${RUN}/sketched_lora/split-cifar-100 \
        --log_dir       ../vit_cl_logs/comparison/${RUN}/sketched_lora

    python3 vit_cl_train.py \
        --model_path       ../PTM/vit-base-patch16-224-in21k \
        --dataset          imagenet_r \
        --imagenet_r_tasks 10 \
        --data_root        ../data \
        --method           sketched_lora \
        --lora_r           8 \
        --lora_depth       5 \
        --svd_rank         8 \
        --svd_oversample   4 \
        --svd_period       2 \
        --seed             $SEED \
        --output_dir       ../runs/comparison/${RUN}/sketched_lora/split-imagenet-r-10t \
        --log_dir          ../vit_cl_logs/comparison/${RUN}/sketched_lora/imagenet_r_10t

    python3 vit_cl_train.py \
        --model_path       ../PTM/vit-base-patch16-224-in21k \
        --dataset          imagenet_r \
        --imagenet_r_tasks 20 \
        --data_root        ../data \
        --method           sketched_lora \
        --lora_r           8 \
        --lora_depth       5 \
        --svd_rank         8 \
        --svd_oversample   4 \
        --svd_period       2 \
        --seed             $SEED \
        --output_dir       ../runs/comparison/${RUN}/sketched_lora/split-imagenet-r-20t \
        --log_dir          ../vit_cl_logs/comparison/${RUN}/sketched_lora/imagenet_r_20t

    python3 vit_cl_train.py \
        --model_path    ../PTM/vit-base-patch16-224-in21k \
        --dataset       cub200 \
        --data_root     ../data \
        --method        sketched_lora \
        --lora_r        8 \
        --lora_depth    5 \
        --svd_rank      8 \
        --svd_oversample 4 \
        --svd_period     2 \
        --seed          $SEED \
        --output_dir    ../runs/comparison/${RUN}/sketched_lora/split-cub-200 \
        --log_dir       ../vit_cl_logs/comparison/${RUN}/sketched_lora

    # -----------------------------------------------------------------------
    # Block 2: O-LoRA
    # -----------------------------------------------------------------------

    python3 vit_cl_train.py \
        --model_path    ../PTM/vit-base-patch16-224-in21k \
        --dataset       cifar100 \
        --data_root     ../data \
        --method        olora \
        --lora_r        8 \
        --lora_depth    5 \
        --olora_lambda  0.5 \
        --seed          $SEED \
        --output_dir    ../runs/comparison/${RUN}/olora/split-cifar-100 \
        --log_dir       ../vit_cl_logs/comparison/${RUN}/olora

    python3 vit_cl_train.py \
        --model_path       ../PTM/vit-base-patch16-224-in21k \
        --dataset          imagenet_r \
        --imagenet_r_tasks 10 \
        --data_root        ../data \
        --method           olora \
        --lora_r           8 \
        --lora_depth       5 \
        --olora_lambda     0.5 \
        --seed             $SEED \
        --output_dir       ../runs/comparison/${RUN}/olora/split-imagenet-r-10t \
        --log_dir          ../vit_cl_logs/comparison/${RUN}/olora/imagenet_r_10t

    python3 vit_cl_train.py \
        --model_path       ../PTM/vit-base-patch16-224-in21k \
        --dataset          imagenet_r \
        --imagenet_r_tasks 20 \
        --data_root        ../data \
        --method           olora \
        --lora_r           8 \
        --lora_depth       5 \
        --olora_lambda     0.5 \
        --seed             $SEED \
        --output_dir       ../runs/comparison/${RUN}/olora/split-imagenet-r-20t \
        --log_dir          ../vit_cl_logs/comparison/${RUN}/olora/imagenet_r_20t

    python3 vit_cl_train.py \
        --model_path    ../PTM/vit-base-patch16-224-in21k \
        --dataset       cub200 \
        --data_root     ../data \
        --method        olora \
        --lora_r        8 \
        --lora_depth    5 \
        --olora_lambda  0.5 \
        --seed          $SEED \
        --output_dir    ../runs/comparison/${RUN}/olora/split-cub-200 \
        --log_dir       ../vit_cl_logs/comparison/${RUN}/olora

    echo "  Seed ${SEED} complete."

done
