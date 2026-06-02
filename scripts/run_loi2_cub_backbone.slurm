#!/bin/bash

#SBATCH --job-name=treelora_loi2_cub
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/treelora/vit_cl_logs/loi2_cub_backbone/slurm_output.log
#SBATCH --error=/data/gmar762/research/continuous_learning/treelora/vit_cl_logs/loi2_cub_backbone/slurm_error.log
#SBATCH --gres=gpu:2

cd /data/gmar762/research/continuous_learning/treelora

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora

mkdir -p vit_cl_logs/loi2_cub_backbone/imagenet_r_5t
mkdir -p vit_cl_logs/loi2_cub_backbone/imagenet_r_10t
mkdir -p vit_cl_logs/loi2_cub_backbone/imagenet_r_20t

# Per-epoch drift analysis with both sketch (LOI 1) and direct A-matrix (LOI 2)
# comparisons, using the CUB-200 fine-tuned backbone.
#
# The CUB-200 backbone is weaker on CIFAR-100 / ImageNet-R than the original
# iBOT in21k checkpoint, producing more forgetting signal for correlation testing.
#
# LOI 1 (sketch): cms_ip, cms_l1diff, cs_ip, cs_l1diff vs. forgetting
# LOI 2 (direct): A_global__ip and A_global__l1 (concatenated A matrices, exact),
#                 plus per-layer breakdown in the final correlation summary.
#
# --reg 0: no regularisation — pure sequential LoRA fine-tuning (SeqLoRA).
# --drift_analysis: per-epoch evaluation of all prior tasks + both comparison sets.

python3 training/vit_cl_train.py \
    --model_path PTM/vit-base-patch16-224-cub200 \
    --dataset    cifar100 \
    --data_root  data \
    --reg        0 \
    --drift_analysis \
    --output_dir runs/loi2-cub-backbone/split-cifar-100 \
    --log_dir    vit_cl_logs/loi2_cub_backbone

python3 training/vit_cl_train.py \
    --model_path       PTM/vit-base-patch16-224-cub200 \
    --dataset          imagenet_r \
    --imagenet_r_tasks 5 \
    --data_root        data \
    --reg              0 \
    --drift_analysis \
    --output_dir       runs/loi2-cub-backbone/split-imagenet-r-5t \
    --log_dir          vit_cl_logs/loi2_cub_backbone/imagenet_r_5t

python3 training/vit_cl_train.py \
    --model_path       PTM/vit-base-patch16-224-cub200 \
    --dataset          imagenet_r \
    --imagenet_r_tasks 10 \
    --data_root        data \
    --reg              0 \
    --drift_analysis \
    --output_dir       runs/loi2-cub-backbone/split-imagenet-r-10t \
    --log_dir          vit_cl_logs/loi2_cub_backbone/imagenet_r_10t

python3 training/vit_cl_train.py \
    --model_path       PTM/vit-base-patch16-224-cub200 \
    --dataset          imagenet_r \
    --imagenet_r_tasks 20 \
    --data_root        data \
    --reg              0 \
    --drift_analysis \
    --output_dir       runs/loi2-cub-backbone/split-imagenet-r-20t \
    --log_dir          vit_cl_logs/loi2_cub_backbone/imagenet_r_20t
