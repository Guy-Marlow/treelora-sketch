#!/bin/bash

#SBATCH --job-name=vit_cub200_pretrain
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/treelora/vit_cl_logs/pretrain/slurm_output.log
#SBATCH --error=/data/gmar762/research/continuous_learning/treelora/vit_cl_logs/pretrain/slurm_error.log
#SBATCH --gres=gpu:1

cd /data/gmar762/research/continuous_learning/treelora

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora

mkdir -p vit_cl_logs/pretrain

python3 scripts/setup.py \
    --ptm_dir  PTM/vit-base-patch16-224-in21k \
    --data_root data

# Fine-tune the ViT-B/16 (iBOT ImageNet-21K) backbone end-to-end on all 200
# CUB-200 classes.  The resulting checkpoint is saved in HuggingFace format
# and can be passed directly to vit_cl_train.py via --model_path.
#
# Motivation: the raw iBOT backbone is strong enough that LoRA barely needs
# to move between tasks, giving BWT ≈ 0%.  A backbone pre-adapted to the
# CUB-200 image distribution (but NOT to individual class splits) forces the
# adapter to do more task-specific work, producing a forgetting signal that
# the sketch metrics can correlate against.
#
# Output: PTM/vit-base-patch16-224-cub200/   (gitignored — safe across resets)

python3 training/finetune_vit_cub200.py \
    --model_path PTM/vit-base-patch16-224-in21k \
    --data_root  data \
    --output_dir PTM/vit-base-patch16-224-cub200 \
    --epochs     30 \
    --lr         2e-5
