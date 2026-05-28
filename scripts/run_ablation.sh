#!/bin/bash

cd /home/gmar762/research/continuous_learning/treelora/training

# ── CMS  width=0.1N  d=6 ─────────────────────────────────────────────────────

python3 vit_cl_train.py \
    --model_path    ../PTM/vit-base-patch16-224-in21k \
    --dataset       cub200 \
    --data_root     ../data \
    --output_dir    ../runs/ablation-cms-cub200 \
    --sketch_type   cms \
    --sketch_w_frac 0.1 \
    --sketch_d      6

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 10 \
    --data_root        ../data \
    --output_dir       ../runs/ablation-cms-imagenet-r-10t \
    --sketch_type   cms \
    --sketch_w_frac 0.1 \
    --sketch_d      6

# ── CS  width=0.1N  d=6  signed ──────────────────────────────────────────────

python3 vit_cl_train.py \
    --model_path    ../PTM/vit-base-patch16-224-in21k \
    --dataset       cub200 \
    --data_root     ../data \
    --output_dir    ../runs/ablation-cs-cub200 \
    --sketch_type   cs \
    --sketch_w_frac 0.1 \
    --sketch_d      6

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 10 \
    --data_root        ../data \
    --output_dir       ../runs/ablation-cs-imagenet-r-10t \
    --sketch_type   cs \
    --sketch_w_frac 0.1 \
    --sketch_d      6
