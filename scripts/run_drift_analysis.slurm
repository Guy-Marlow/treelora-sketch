#!/bin/bash

#SBATCH --job-name=treelora_drift
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/treelora/vit_cl_logs/drift_noreg/slurm_output.log
#SBATCH --error=/data/gmar762/research/continuous_learning/treelora/vit_cl_logs/drift_noreg/slurm_error.log
#SBATCH --gres=gpu:2

cd /data/gmar762/research/continuous_learning/treelora/training

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora

python3 ../scripts/setup.py \
    --ptm_dir  ../PTM/vit-base-patch16-224-in21k \
    --data_root ../data

# Per-epoch drift analysis across all 5 benchmarks.
# --reg 0: no regularisation — tree is not constructed, adapter fine-tunes freely.
# --drift_analysis: at the end of each epoch, evaluates all prior tasks and records
#   per (task, epoch, prior_task):
#     forgetting                — canonical_acc - current_acc on prior task
#     cms_ip / cms_l1diff       — normalised A+B CMS metrics
#     c_ip  / c_l1diff          — normalised A+B CS metrics
#     a_cms_ip / a_cms_l1diff   — unnormalised A-only CMS (paper's scale)
#     a_cs_ip  / a_cs_l1diff    — unnormalised A-only CS  (paper's scale)
#     a_l1_concat               — exact L1 of concatenated loranew_A difference
#     a_l1_layer_{0..N}         — exact per-layer loranew_A L1 norms
#   Produces absolute and first-differences Kendall τ / Spearman ρ at run end.
#
# Note: the three ImageNet-R variants write to separate log subdirectories because
# all three share the same benchmark slug (split-imagenet-r) and would otherwise
# overwrite each other's log files.

python3 vit_cl_train.py \
    --model_path ../PTM/vit-base-patch16-224-in21k \
    --dataset    cifar100 \
    --data_root  ../data \
    --reg        0 \
    --drift_analysis \
    --output_dir ../runs/drift-noreg/split-cifar-100 \
    --log_dir    ../vit_cl_logs/drift_noreg

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 5 \
    --data_root        ../data \
    --reg              0 \
    --drift_analysis \
    --output_dir       ../runs/drift-noreg/split-imagenet-r-5t \
    --log_dir          ../vit_cl_logs/drift_noreg/imagenet_r_5t

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 10 \
    --data_root        ../data \
    --reg              0 \
    --drift_analysis \
    --output_dir       ../runs/drift-noreg/split-imagenet-r-10t \
    --log_dir          ../vit_cl_logs/drift_noreg/imagenet_r_10t

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 20 \
    --data_root        ../data \
    --reg              0 \
    --drift_analysis \
    --output_dir       ../runs/drift-noreg/split-imagenet-r-20t \
    --log_dir          ../vit_cl_logs/drift_noreg/imagenet_r_20t

python3 vit_cl_train.py \
    --model_path ../PTM/vit-base-patch16-224-in21k \
    --dataset    cub200 \
    --data_root  ../data \
    --reg        0 \
    --drift_analysis \
    --output_dir ../runs/drift-noreg/split-cub-200 \
    --log_dir    ../vit_cl_logs/drift_noreg
