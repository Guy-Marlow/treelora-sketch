"""
2-task CIFAR-100 smoke test for sketch collection and correlation analysis.

Runs 3 epochs per task (fast) to verify:
  - All 6 sketches accumulate non-trivial mass during training
  - inner_product and l1_sketch_diff return sensible values
  - analyze_sketch_correlations builds the table and prints correctly
  - acc_matrix is correctly populated for the forgetting calculation

Usage:
    conda run -n treelora python training/test_sketches.py \
        --model_path PTM/vit-base-patch16-224-in21k \
        --data_root data
"""

import argparse
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vit_lora import build_treelora_vit
from training.vit_cl_train import ViTCLTrainer
from utils.data.vision_cl_datasets import make_split_cifar100


def parse_args():
    p = argparse.ArgumentParser(description='2-task sketch smoke test')
    p.add_argument('--model_path', required=True)
    p.add_argument('--data_root',  required=True)
    p.add_argument('--epochs',     type=int, default=3,
                   help='Epochs per task (default 3 — fast debug run)')
    p.add_argument('--lora_depth', type=int, default=5)
    p.add_argument('--lora_r',     type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--batch_size', type=int, default=192)
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--reg',        type=float, default=0.0,
                   help='Tree regularisation (0 = disabled, keeps test simple)')
    return p.parse_args()


def make_trainer_args(cli):
    """Build the args namespace that ViTCLTrainer expects."""
    a = types.SimpleNamespace()
    a.lr             = 0.005
    a.reg            = cli.reg
    a.output_dir     = ''
    a.epochs_per_task = cli.epochs
    # KD_LoRA_Tree fields (only used when reg > 0; set safe defaults)
    a.lora_depth     = cli.lora_depth
    a.lora_r         = cli.lora_r
    a.global_rank    = 0
    a.num_tasks      = 2
    return a


def print_sketch_bank(trainer):
    """Print the total accumulated mass in each sketch for each task."""
    sketch_names = [
        'cm_grad_abs', 'cm_grad_squared', 'cm_taylor',
        'cm_weight_diff', 'cm_weight_diff_squared', 'cm_state',
    ]
    print('\n── Sketch bank contents ─────────────────────────────────────')
    for t, sketches in enumerate(trainer.sketch_bank):
        print(f'\n  Task {t}:')
        for name in sketch_names:
            sk = sketches[name]
            mass    = sk.cms.sum().item()
            nonzero = (sk.cms != 0).sum().item()
            total   = sk.cms.numel()
            print(f'    {name:<30}  mass={mass:>14.2f}  '
                  f'non-zero={nonzero}/{total} '
                  f'({100*nonzero/total:.1f}%)')

    print('\n── Pairwise sketch comparisons (task 1 vs task 0) ──────────')
    if len(trainer.sketch_bank) >= 2:
        s0, s1 = trainer.sketch_bank[0], trainer.sketch_bank[1]
        for name in sketch_names:
            ip  = s1[name].inner_product(s0[name])
            l1d = s1[name].l1_sketch_diff(s0[name])
            print(f'    {name:<30}  inner_product={float(ip):>14.4f}  '
                  f'l1_diff={l1d:>14.4f}')
    print()


def main():
    cli  = parse_args()
    args = make_trainer_args(cli)

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    print('─' * 60)
    print('  Sketch smoke test: 2-task CIFAR-100')
    print(f'  {cli.epochs} epochs/task  ·  reg={cli.reg}')
    print('─' * 60)

    print('\nLoading dataset …')
    task_info = make_split_cifar100(
        data_root=cli.data_root,
        num_tasks=2,
        batch_size=cli.batch_size,
        seed=cli.seed,
        num_workers=cli.num_workers,
    )
    print(f'  2 tasks · {task_info[0]["num_classes"]} classes/task')

    print(f'\nLoading ViT from {cli.model_path} …')
    model = build_treelora_vit(
        checkpoint_path=cli.model_path,
        r=cli.lora_r,
        lora_alpha=cli.lora_alpha,
        lora_depth=cli.lora_depth,
    )

    trainer = ViTCLTrainer(model, task_info, args)
    print(f'\nAdapter params : {trainer._adapter_total_params:,}')
    print(f'Sketch width w : {trainer._sketch_w}  (5% of adapter params)')
    print(f'Sketch depth d : {trainer._sketch_d}')

    # Run training — this populates sketch_bank and acc_matrix,
    # then calls analyze_sketch_correlations automatically.
    trainer.train_continual()

    # Extra debug: print raw sketch masses and pairwise comparisons
    print_sketch_bank(trainer)

    print('\nSmoke test complete.')


if __name__ == '__main__':
    main()
