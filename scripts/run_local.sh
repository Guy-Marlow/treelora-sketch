#!/bin/bash

cd /home/gmar762/research/continuous_learning/treelora/training

python3 ../scripts/setup.py \
    --ptm_dir  ../PTM/vit-base-patch16-224-in21k \
    --data_root ../data

python3 vit_cl_train.py \
    --model_path ../PTM/vit-base-patch16-224-in21k \
    --dataset    cifar100 \
    --data_root  ../data \
    --output_dir ../runs/split-cifar-100

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 5 \
    --data_root        ../data \
    --output_dir       ../runs/split-imagenet-r-5t

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 10 \
    --data_root        ../data \
    --output_dir       ../runs/split-imagenet-r-10t

python3 vit_cl_train.py \
    --model_path       ../PTM/vit-base-patch16-224-in21k \
    --dataset          imagenet_r \
    --imagenet_r_tasks 20 \
    --data_root        ../data \
    --output_dir       ../runs/split-imagenet-r-20t

python3 vit_cl_train.py \
    --model_path ../PTM/vit-base-patch16-224-in21k \
    --dataset    cub200 \
    --data_root  ../data \
    --output_dir ../runs/split-cub-200
