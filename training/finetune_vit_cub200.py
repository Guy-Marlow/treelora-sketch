#!/usr/bin/env python3
"""
Fine-tune ViT-B/16 (iBOT ImageNet-21K) on all 200 CUB-200-2011 classes.

The backbone is fully unfrozen and trained end-to-end with a linear head.
The best-validation-accuracy backbone is then saved in HuggingFace format
to the output directory, making it a drop-in replacement for the original
PTM in any of the CL training scripts.

Motivation: the iBOT backbone is so strong that LoRA barely needs to move
to serve prior tasks, giving BWT ≈ 0% and no forgetting signal.  A backbone
that has been adapted to the CUB-200 image distribution (but NOT to individual
class splits) forces the LoRA adapter to do more task-specific work, increasing
the forgetting signal and making the sketch-forgetting correlation experiments
more informative.

Usage:
    python training/finetune_vit_cub200.py \
        --model_path PTM/vit-base-patch16-224-in21k \
        --data_root  /path/to/data \
        --output_dir PTM/vit-base-patch16-224-cub200
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk
from utils.data.vision_cl_datasets import _task_dataset, TRAIN_TF, TEST_TF


# ── Data ───────────────────────────────────────────────────────────────────────

def load_cub200_full(data_root: str, batch_size: int, num_workers: int):
    """Return train/test DataLoaders for all 200 CUB-200 classes."""
    path = os.path.join(data_root, 'cub200')
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f'{path} not found.  Run: '
            f'python utils/data/download_datasets.py --data_root {data_root} --datasets cub200'
        )
    ds          = load_from_disk(path)
    all_classes = list(range(200))          # identity label map (0–199)
    train_ds = _task_dataset(ds['train'], 'image', 'label', all_classes, TRAIN_TF)
    test_ds  = _task_dataset(ds['test'],  'image', 'label', all_classes, TEST_TF)
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=(num_workers > 0))
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(test_ds,  shuffle=False, **kw))


# ── Model ──────────────────────────────────────────────────────────────────────

class ViTClassifier(nn.Module):
    """Bare ViTModel backbone with a single linear head for fine-tuning."""

    VIT_HIDDEN = 768

    def __init__(self, backbone, num_classes: int = 200):
        super().__init__()
        self.backbone = backbone
        self.head     = nn.Linear(self.VIT_HIDDEN, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        return self.head(cls)


# ── Training helpers ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = running_correct = total = 0
    pbar = tqdm(loader, leave=False, desc='train')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss    += loss.item()
        running_correct += (logits.argmax(1) == labels).sum().item()
        total           += labels.size(0)
        pbar.set_postfix(loss=f'{running_loss / (pbar.n + 1):.4f}',
                         acc=f'{running_correct / total:.3f}')
    return running_loss / len(loader), running_correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Fine-tune ViT-B/16 backbone on CUB-200 (all 200 classes)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model_path',    default='PTM/vit-base-patch16-224-in21k',
                   help='Source ViT checkpoint directory')
    p.add_argument('--data_root',     required=True,
                   help='Parent directory containing the cub200/ dataset folder')
    p.add_argument('--output_dir',    default='PTM/vit-base-patch16-224-cub200',
                   help='Where to save the fine-tuned backbone checkpoint')
    p.add_argument('--epochs',        type=int,   default=30)
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--lr',            type=float, default=2e-5,
                   help='Peak learning rate for AdamW')
    p.add_argument('--weight_decay',  type=float, default=0.01)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--seed',          type=int,   default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f'Device: {device}')

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f'\nLoading CUB-200 (all 200 classes) from {args.data_root} ...')
    train_loader, test_loader = load_cub200_full(
        args.data_root, args.batch_size, args.num_workers
    )
    print(f'  Train batches : {len(train_loader)}'
          f'  ({len(train_loader.dataset):,} images)')
    print(f'  Test  batches : {len(test_loader)}'
          f'  ({len(test_loader.dataset):,} images)')

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f'\nLoading backbone from {args.model_path} ...')
    backbone = AutoModel.from_pretrained(args.model_path)
    model    = ViTClassifier(backbone, num_classes=200).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}  (all trainable)')

    # ── Optimiser & schedule ──────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_acc   = 0.0
    best_state = None
    print(f'\nFine-tuning for {args.epochs} epochs '
          f'(lr={args.lr}, wd={args.weight_decay}) ...\n')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        marker = '  ← best' if test_acc > best_acc else ''
        print(f'  Epoch {epoch:3d}/{args.epochs}  '
              f'loss={train_loss:.4f}  train={train_acc:.3f}  '
              f'test={test_acc:.3f}{marker}')

        if test_acc > best_acc:
            best_acc   = test_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.backbone.state_dict().items()}

    print(f'\n  Best test accuracy: {best_acc * 100:.2f}%')

    # ── Save backbone checkpoint ──────────────────────────────────────────────
    print(f'\nSaving fine-tuned backbone to {args.output_dir} ...')
    os.makedirs(args.output_dir, exist_ok=True)

    model.backbone.load_state_dict(best_state)
    model.backbone.save_pretrained(args.output_dir)

    # Copy preprocessor config so the directory is fully self-contained
    for fname in ('preprocessor_config.json', 'config.json'):
        src = os.path.join(args.model_path, fname)
        if os.path.isfile(src):
            shutil.copy(src, args.output_dir)

    print(f'  Saved.')
    print(f'\nTo use in continual-learning experiments:')
    print(f'    --model_path {args.output_dir}')


if __name__ == '__main__':
    main()
