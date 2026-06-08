#!/bin/bash

#SBATCH --job-name=vit_method_comparison
#SBATCH --time=24:00:00
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
# Sketched LoRA: every 2 tasks the per-task adapters are RandSVD-compressed into
#   a single r_hat-rank summary (svd_rank=8, svd_oversample=4, svd_period=2).
#   Forward-pass cost stays constant at r_hat+r = 16 rows/layer after the first
#   compression (end of task 2), with the bank holding at most 2 snapshots between
#   compression events.
#
# O-LoRA: frozen bank grows by r rows per task.  Forward-pass cost scales as k*r+r
#   at task k, so 16 → 24 → 32 … rows/layer.
#
# Note: the three ImageNet-R variants share the benchmark slug (split-imagenet-r)
# and would overwrite each other's log files if placed in the same directory.
# Each variant therefore gets its own --log_dir subdirectory.

# ---------------------------------------------------------------------------
# Block 1: Sketched LoRA
# ---------------------------------------------------------------------------

python3 vit_cl_train.py \
    --model_path    ../PTM/vit-base-patch16-224-in21k \
    --dataset       cifar100 \
    --data_root     ../data \
    --method        sketched_lora \
    --lora_r        8 \
    --lora_depth    -1 \
    --svd_rank      8 \
    --svd_oversample 4 \
    --svd_period     2 \
    --output_dir    ../runs/comparison/sketched_lora/split-cifar-100 \
    --log_dir       ../vit_cl_logs/comparison/sketched_lora

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 10 \
    --data_root        ../data \
    --method           sketched_lora \
    --lora_r           8 \
    --lora_depth       -1 \
    --svd_rank         8 \
    --svd_oversample   4 \
    --svd_period       2 \
    --output_dir       ../runs/comparison/sketched_lora/split-imagenet-r-10t \
    --log_dir          ../vit_cl_logs/comparison/sketched_lora/imagenet_r_10t

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 20 \
    --data_root        ../data \
    --method           sketched_lora \
    --lora_r           8 \
    --lora_depth       -1 \
    --svd_rank         8 \
    --svd_oversample   4 \
    --svd_period       2 \
    --output_dir       ../runs/comparison/sketched_lora/split-imagenet-r-20t \
    --log_dir          ../vit_cl_logs/comparison/sketched_lora/imagenet_r_20t

python3 vit_cl_train.py \
    --model_path    ../PTM/vit-base-patch16-224-in21k \
    --dataset       cub200 \
    --data_root     ../data \
    --method        sketched_lora \
    --lora_r        8 \
    --lora_depth    -1 \
    --svd_rank      8 \
    --svd_oversample 4 \
    --svd_period     2 \
    --output_dir    ../runs/comparison/sketched_lora/split-cub-200 \
    --log_dir       ../vit_cl_logs/comparison/sketched_lora

# ---------------------------------------------------------------------------
# Block 2: O-LoRA
# ---------------------------------------------------------------------------

python3 vit_cl_train.py \
    --model_path    ../PTM/vit-base-patch16-224-in21k \
    --dataset       cifar100 \
    --data_root     ../data \
    --method        olora \
    --lora_r        8 \
    --lora_depth    -1 \
    --olora_lambda  0.5 \
    --output_dir    ../runs/comparison/olora/split-cifar-100 \
    --log_dir       ../vit_cl_logs/comparison/olora

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 10 \
    --data_root        ../data \
    --method           olora \
    --lora_r           8 \
    --lora_depth       -1 \
    --olora_lambda     0.5 \
    --output_dir       ../runs/comparison/olora/split-imagenet-r-10t \
    --log_dir          ../vit_cl_logs/comparison/olora/imagenet_r_10t

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 20 \
    --data_root        ../data \
    --method           olora \
    --lora_r           8 \
    --lora_depth       -1 \
    --olora_lambda     0.5 \
    --output_dir       ../runs/comparison/olora/split-imagenet-r-20t \
    --log_dir          ../vit_cl_logs/comparison/olora/imagenet_r_20t

python3 vit_cl_train.py \
    --model_path    ../PTM/vit-base-patch16-224-in21k \
    --dataset       cub200 \
    --data_root     ../data \
    --method        olora \
    --lora_r        8 \
    --lora_depth    -1 \
    --olora_lambda  0.5 \
    --output_dir    ../runs/comparison/olora/split-cub-200 \
    --log_dir       ../vit_cl_logs/comparison/olora