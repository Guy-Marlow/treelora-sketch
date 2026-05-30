#!/bin/bash

#SBATCH --job-name=treelora_drift
#SBATCH --time=20:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/treelora/vit_cl_logs/drift_noreg/slurm_output.log
#SBATCH --error=/data/gmar762/research/continuous_learning/treelora/vit_cl_logs/drift_noreg/slurm_error.log
#SBATCH --gres=gpu:2

cd /data/gmar762/research/continuous_learning/treelora/training

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora

# Per-epoch drift analysis with no regularisation (reg=0 disables the tree entirely).
# At the end of each training epoch, the model is evaluated on all prior tasks and
# the forgetting is correlated against the ip and l1diff of the current adapter
# sketch vs. each prior task's canonical sketch.

python3 vit_cl_train.py \
    --model_path    ../PTM/vit-base-patch16-224-in21k \
    --dataset       cifar100 \
    --data_root     ../data \
    --reg           0 \
    --drift_analysis \
    --output_dir    ../runs/drift-noreg/split-cifar-100 \
    --log_dir       ../vit_cl_logs/drift_noreg

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 10 \
    --data_root        ../data \
    --reg              0 \
    --drift_analysis \
    --output_dir       ../runs/drift-noreg/split-imagenet-r-10t \
    --log_dir          ../vit_cl_logs/drift_noreg

python3 vit_cl_train.py \
    --model_path    ../PTM/vit-base-patch16-224-in21k \
    --dataset       cub200 \
    --data_root     ../data \
    --reg           0 \
    --drift_analysis \
    --output_dir    ../runs/drift-noreg/split-cub-200 \
    --log_dir       ../vit_cl_logs/drift_noreg
