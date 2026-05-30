"""
Continual learning training loop for ViT-B/16 + TreeLoRA on vision benchmarks.

Implements the hyperparameters from TreeLoRA paper §A.3:
    Optimizer : Adam  β1=0.9  β2=0.999
    LR        : 0.005  (constant — no decay)
    Batch size: 192
    Epochs    : 20 on Split CIFAR-100,  50 on Split ImageNet-R and Split CUB-200
    Input     : 224×224,  normalised to [0, 1]  (ToTensor only)
    Backbone  : ViT-B/16 iBOT-21K

Benchmark task definitions (paper §5.1):
    Split CIFAR-100  : 100 classes → 10 tasks × 10 classes
    Split ImageNet-R : 200 classes → 5 / 10 / 20 tasks × 40 / 20 / 10 classes
    Split CUB-200    : 200 classes → 10 tasks × 20 classes

Accuracy tracking:
    acc_matrix[i, j] = accuracy on task j after training on tasks 0..i.
    OP  = mean of the final row (overall performance after all tasks).
    BWT = mean drop per task: acc_matrix[N-1, j] − acc_matrix[j, j]  (j < N-1).
    After each task, we print per-task and average accuracy on all seen tasks.

Datasets must be downloaded first:
    python utils/data/download_datasets.py --data_root /path/to/data

Usage example:
    python training/vit_cl_train.py \\
        --model_path PTM/iBOT-ViT-B-16 \\
        --dataset cifar100 \\
        --data_root /path/to/data \\
        --output_dir runs/treelora_cifar100

    # ImageNet-R with 10 tasks:
        --dataset imagenet_r --imagenet_r_tasks 10

    # Disable TreeLoRA regularisation (plain sequential LoRA):
        --reg 0

    # Per-epoch drift analysis (unconstrained forgetting baseline):
        --drift_analysis --reg 0

    # Per-epoch drift analysis with partial regularisation:
        --drift_analysis --reg 0.05
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Logging tee ───────────────────────────────────────────────────────────────

class _Tee:
    """Duplicate stdout writes to a file while keeping terminal output."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file   = open(path, 'w', buffering=1)   # line-buffered
        self._stdout = sys.__stdout__

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Proxy everything else to the real stdout
    def __getattr__(self, name):
        return getattr(self._stdout, name)


def _setup_logging(log_path: str):
    tee = _Tee(log_path)
    sys.stdout = tee
    return tee

from model.vit_lora import build_treelora_vit
from utils.dyadic_cms import CountMinSketch, CountSketch
from utils.kd_lora_tree import KD_LoRA_Tree
from utils.data.vision_cl_datasets import (
    make_split_cifar100,
    make_split_imagenet_r,
    make_split_cub200,
)

# Epochs per dataset (paper §A.3)
_DEFAULT_EPOCHS = {
    'cifar100':   20,
    'imagenet_r': 50,
    'cub200':     50,
}


# ── Trainer ───────────────────────────────────────────────────────────────────

class ViTCLTrainer:
    """
    Continual learning trainer for ViT + TreeLoRA on class-incremental benchmarks.

    One linear head per task (Linear(768, num_classes_per_task)) is trained
    alongside the LoRA adapter.  Per-task heads keep accuracy evaluation
    unambiguous — task j's test set always passes through head j.

    The KD-LoRA Tree regularisation is optional (reg=0 disables it).

    Parameters
    ----------
    model     : PeftModel from build_treelora_vit
    task_info : list of dicts from make_split_* (each has train/test loaders)
    args      : argparse.Namespace — see parse_args()
    """

    VIT_HIDDEN = 768   # ViT-B/16 CLS token dimension

    def __init__(self, model, task_info: list, args):
        self.model     = model
        self.task_info = task_info
        self.args      = args
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        # Per-task linear classification heads
        self.task_heads = nn.ModuleList([
            nn.Linear(self.VIT_HIDDEN, info['num_classes'])
            for info in task_info
        ]).to(self.device)

        # KD-LoRA Tree (created only when regularisation is requested)
        self.tree = None
        if args.reg > 0:
            args.num_tasks   = len(task_info)
            args.global_rank = 0   # single-GPU; suppresses rank-0 guards inside tree
            self.tree = KD_LoRA_Tree(args)

        num_tasks = len(task_info)
        # acc_matrix[i, j] = accuracy on task j after training through task i
        self.acc_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

        # Drift analysis: per-epoch sketch vs. forgetting tracking
        self.canonical_sketches: list = []  # one {'cms': CMS, 'cs': CS} dict per completed task
        self.canonical_accs: list     = []  # acc_matrix[t, t] for each completed task
        self.drift_records: list      = []  # dicts: task/epoch/prior_task/forgetting/ip/l1diff/c_ip/c_l1diff
        _adapter_total_params = sum(
            p.numel() for n, p in self.model.named_parameters()
            if 'loranew_A' in n or 'loranew_B' in n
        )
        self._sketch_d = getattr(args, 'sketch_d', 8)
        self._sketch_w = math.ceil(getattr(args, 'sketch_w_frac', 0.05) * _adapter_total_params)

    # ── Feature extraction ────────────────────────────────────────────────────

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """CLS token from ViT last_hidden_state; shape (B, 768)."""
        return self.model(pixel_values=pixel_values).last_hidden_state[:, 0, :]

    # ── LoRA parameter collection for KD tree ─────────────────────────────────

    def _lora_grad_tensor(self) -> torch.Tensor | None:
        """
        Stack all loranew_A parameter tensors (flattened) into shape
        (lora_depth, r * d_in) — matches KD_LoRA_Tree's expected format.
        """
        params = [
            p.reshape(-1)
            for n, p in self.model.named_parameters()
            if 'loranew_A' in n
        ]
        return torch.stack(params, dim=0) if params else None

    # ── Adapter sketch helpers ────────────────────────────────────────────────

    def _flat_adapter_vec(self) -> torch.Tensor | None:
        """All loranew_A and loranew_B parameter values concatenated and flattened."""
        parts = []
        for name, p in self.model.named_parameters():
            if 'loranew_A' in name or 'loranew_B' in name:
                parts.append(p.data.detach().reshape(-1))
        return torch.cat(parts) if parts else None

    def _build_state_sketches(self) -> dict:
        """
        Build a paired CMS + CS from the L2-normalised current adapter.

        Both sketches are built from the same unit-norm vector. CMS receives
        abs(vec) (non-negativity required); CS receives the signed vec directly,
        so its inner product approximates true cosine similarity rather than
        the absolute-value overlap that CMS measures.
        """
        cms = CountMinSketch(self._sketch_d, self._sketch_w, device=self.device, dtype=torch.float32)
        cs  = CountSketch(   self._sketch_d, self._sketch_w, device=self.device, dtype=torch.float32)
        state_flat = self._flat_adapter_vec()
        if state_flat is not None:
            norm = state_flat.norm(p=2)
            if norm > 0:
                state_flat = state_flat / norm
            cms.insert_vec(state_flat.abs())
            cs.insert_vec(state_flat)
        return {'cms': cms, 'cs': cs}

    # ── Single-task training ──────────────────────────────────────────────────

    def train_one_task(self, task_id: int, epochs: int):
        info         = self.task_info[task_id]
        train_loader = info['train']
        head         = self.task_heads[task_id]
        criterion    = nn.CrossEntropyLoss()

        # Adam with paper's β values and constant LR (no scheduler)
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer   = Adam(
            lora_params + list(head.parameters()),
            lr=self.args.lr,
            betas=(0.9, 0.999),
        )

        for epoch in range(epochs):
            self.model.train()
            head.train()

            if self.tree is not None:
                self.tree.new_epoch_init(len(train_loader))

            running_loss = running_correct = running_total = 0
            pbar = tqdm(
                train_loader,
                desc=f'Task {task_id} | Epoch {epoch + 1}/{epochs}',
                leave=False,
            )

            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.tree is not None:
                    self.tree.step()

                features = self._features(images)
                logits   = head(features)
                loss     = criterion(logits, labels)

                if self.tree is not None:
                    grad_tensor = self._lora_grad_tensor()
                    if grad_tensor is not None:
                        self.tree.insert_grad(grad_tensor)
                        if task_id > 0:
                            prev_ids = self.tree.tree_search(task_id, self.device)
                            reg_loss = self.tree.get_loss(
                                grad_tensor, loss, task_id, prev_ids
                            )
                            loss = loss - reg_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    lora_params + list(head.parameters()), max_norm=1.0
                )

                optimizer.step()

                with torch.no_grad():
                    preds           = logits.argmax(dim=1)
                    running_correct += (preds == labels).sum().item()
                    running_total   += labels.size(0)
                    running_loss    += loss.item()

                pbar.set_postfix(
                    loss=f'{running_loss / (pbar.n + 1):.4f}',
                    acc=f'{running_correct / running_total:.3f}',
                )

            epoch_acc  = running_correct / running_total
            epoch_loss = running_loss / len(train_loader)
            print(
                f'  Task {task_id} | Epoch {epoch + 1}/{epochs} | '
                f'loss={epoch_loss:.4f}  train_acc={epoch_acc:.3f}'
            )

            # ── Per-epoch drift analysis ──────────────────────────────────────
            if getattr(self.args, 'drift_analysis', False):
                with torch.no_grad():
                    raw_vec = self._flat_adapter_vec()
                    if raw_vec is not None:
                        print(f'    [drift] adapter L2 norm: {raw_vec.norm(p=2).item():.4f}')
                    scratch = self._build_state_sketches()
                    if task_id > 0:
                        for prior_t in range(task_id):
                            prior_acc   = self.canonical_accs[prior_t]
                            current_acc = self.evaluate_task(prior_t)
                            forgetting  = prior_acc - current_acc
                            canon       = self.canonical_sketches[prior_t]
                            ip     = float(scratch['cms'].inner_product(canon['cms']))
                            l1d    = float(scratch['cms'].l1_sketch_diff(canon['cms']))
                            c_ip   = float(scratch['cs'].inner_product(canon['cs']))
                            c_l1d  = float(scratch['cs'].l1_sketch_diff(canon['cs']))
                            self.drift_records.append({
                                'task': task_id, 'epoch': epoch, 'prior_task': prior_t,
                                'forgetting': forgetting,
                                'ip': ip, 'l1diff': l1d,
                                'c_ip': c_ip, 'c_l1diff': c_l1d,
                            })
                            print(
                                f'    [drift] vs T{prior_t}: '
                                f'forgetting={forgetting:+.4f}  '
                                f'cms_ip={ip:.4f}  cms_l1diff={l1d:.4f}  '
                                f'cs_ip={c_ip:.4f}  cs_l1diff={c_l1d:.4f}'
                            )
                    # At the last epoch: store canonical sketches and accuracy for this task
                    if epoch == epochs - 1:
                        canon_acc = self.evaluate_task(task_id)
                        self.canonical_sketches.append(scratch)
                        self.canonical_accs.append(canon_acc)
                        print(f'    [drift] T{task_id} canonical stored  acc={canon_acc:.4f}')
            # ── End per-epoch drift analysis ──────────────────────────────────

        if self.tree is not None:
            self.tree.end_task(task_id=task_id)

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_task(self, task_id: int) -> float:
        """Top-1 accuracy on task task_id's test set using its own head."""
        info = self.task_info[task_id]
        head = self.task_heads[task_id]
        self.model.eval()
        head.eval()

        correct = total = 0
        for images, labels in info['test']:
            images   = images.to(self.device, non_blocking=True)
            labels   = labels.to(self.device, non_blocking=True)
            features = self._features(images)
            preds    = head(features).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        return correct / total if total > 0 else 0.0

    def evaluate_all_seen(self, up_to_task: int) -> dict:
        """
        Evaluate tasks 0 .. up_to_task and return {task_id: accuracy}.
        This is the 'average accuracy on all prior tasks' reported after each task.
        """
        return {t: self.evaluate_task(t) for t in range(up_to_task + 1)}

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, task_id: int):
        if not getattr(self.args, 'save_checkpoints', False):
            return
        if not self.args.output_dir:
            return
        out = os.path.join(self.args.output_dir, f'task_{task_id}')
        os.makedirs(out, exist_ok=True)
        self.model.save_pretrained(out)
        torch.save(self.task_heads.state_dict(), os.path.join(out, 'task_heads.pt'))
        print(f'  Checkpoint saved → {out}')

    # ── Main continual-learning loop ──────────────────────────────────────────

    def train_continual(self) -> dict:
        """
        Train sequentially over all tasks.  After each task, evaluate all seen
        tasks and print average accuracy.  Populate acc_matrix throughout.
        Returns the final metrics dict.
        """
        num_tasks = len(self.task_info)
        epochs    = self.args.epochs_per_task

        for task_id in range(num_tasks):
            print(f'\n{"=" * 65}')
            print(f'  Task {task_id + 1}/{num_tasks}  '
                  f'({self.task_info[task_id]["num_classes"]} classes)')
            print(f'{"=" * 65}')

            self.train_one_task(task_id, epochs=epochs)

            # Evaluate all tasks seen so far
            print(f'\n  Test accuracy after task {task_id}:')
            results = self.evaluate_all_seen(task_id)
            for t, acc in results.items():
                self.acc_matrix[task_id, t] = acc
                print(f'    Task {t}: {acc * 100:.2f}%')

            avg_seen = np.mean([results[t] for t in range(task_id + 1)])
            print(f'    Average (tasks 0–{task_id}): {avg_seen * 100:.2f}%')

            self.save_checkpoint(task_id)

        metrics = self._compute_metrics()
        if getattr(self.args, 'drift_analysis', False):
            drift_results = self.analyze_drift_correlations()
            metrics['drift_analysis'] = drift_results
        return metrics

    # ── Final metrics ─────────────────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        N = len(self.task_info)

        # Overall Performance: average accuracy over all tasks after the last task
        op = float(np.mean(self.acc_matrix[N - 1, :N]))

        # Backward Transfer: mean drop vs. per-task accuracy right after it was learned
        if N > 1:
            bwt = float(np.mean([
                self.acc_matrix[N - 1, j] - self.acc_matrix[j, j]
                for j in range(N - 1)
            ]))
        else:
            bwt = 0.0

        # Pretty-print accuracy matrix
        print(f'\n{"=" * 65}')
        print('  Accuracy matrix  (row i = after task i, col j = task j)')
        print(f'{"=" * 65}')
        header = '        ' + ''.join(f'  T{j:<3}' for j in range(N))
        print(header)
        for i in range(N):
            row = ''.join(
                f'  {self.acc_matrix[i, j] * 100:4.1f}'
                if j <= i else '     -'
                for j in range(N)
            )
            print(f'  T{i:<4} {row}')

        print(f'\n  OP  (Overall Performance):  {op  * 100:.2f}%')
        print(f'  BWT (Backward Transfer):    {bwt * 100:.2f}%')
        print(f'{"=" * 65}')

        metrics = {
            'op':             op,
            'bwt':            bwt,
            'acc_matrix':     self.acc_matrix.tolist(),
            'per_task_final': {str(j): float(self.acc_matrix[N - 1, j]) for j in range(N)},
        }

        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, 'results.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f'\n  Results saved → {self.args.output_dir}/results.json')

        return metrics

    # ── Drift correlation analysis ────────────────────────────────────────────

    def analyze_drift_correlations(self) -> dict:
        """
        Rank-correlation analysis over all per-epoch drift records.

        Each record is one (task, epoch, prior_task) triple containing:
            forgetting = canonical_acc[prior_task] - current_acc_on_prior_task
            ip         = inner_product(current_sketch, canonical_sketch[prior_task])
            l1diff     = l1_sketch_diff(current_sketch, canonical_sketch[prior_task])

        Reports global Kendall τ / Spearman ρ for ip and l1diff vs. forgetting,
        then a per-prior-task breakdown so per-task drift trajectories are visible.
        """
        from scipy.stats import kendalltau, spearmanr

        if not self.drift_records:
            print('\n  [drift analysis] No records — was --drift_analysis set?')
            return {}

        forgetting = np.array([r['forgetting'] for r in self.drift_records])
        ip         = np.array([r['ip']         for r in self.drift_records])
        l1diff     = np.array([r['l1diff']     for r in self.drift_records])
        c_ip       = np.array([r['c_ip']       for r in self.drift_records])
        c_l1diff   = np.array([r['c_l1diff']   for r in self.drift_records])
        n          = len(forgetting)

        print(f'\n{"=" * 70}')
        print('  Drift Analysis — Sketch–Forgetting Correlation (per-epoch)')
        print(f'  {n} data points  (tasks × epochs × prior tasks)')
        print(f'{"=" * 70}')

        results = {}
        print(f'\n  {"Metric":<25}  {"τ":>8}  {"p(τ)":>8}  {"ρ":>8}  {"p(ρ)":>8}')
        print(f'  {"-" * 25}  {"-" * 8}  {"-" * 8}  {"-" * 8}  {"-" * 8}')
        for label, metric in [
            ('cm_state__ip',    ip),
            ('cm_state__l1diff', l1diff),
            ('c_state__ip',     c_ip),
            ('c_state__l1diff', c_l1diff),
        ]:
            tau, p_tau = kendalltau(forgetting, metric)
            rho, p_rho = spearmanr(forgetting,  metric)
            sig_tau = '*' if p_tau < 0.05 else (' .' if p_tau < 0.10 else '  ')
            sig_rho = '*' if p_rho < 0.05 else (' .' if p_rho < 0.10 else '  ')
            print(
                f'  {label:<25}  {tau:>+8.4f}  {p_tau:>8.4f}{sig_tau} '
                f'{rho:>+8.4f}  {p_rho:>8.4f}{sig_rho}'
            )
            results[label] = {
                'tau': float(tau), 'p_tau': float(p_tau),
                'rho': float(rho), 'p_rho': float(p_rho),
            }

        # Per-prior-task breakdown for both ip metrics
        prior_tasks = sorted(set(r['prior_task'] for r in self.drift_records))
        if len(prior_tasks) > 1:
            for ip_label, ip_key in [('cm_state__ip', 'ip'), ('c_state__ip', 'c_ip')]:
                print(f'\n  Per-prior-task ({ip_label}):')
                print(f'  {"Prior":>6}  {"n":>4}  {"τ":>8}  {"p(τ)":>8}')
                print(f'  {"-" * 6}  {"-" * 4}  {"-" * 8}  {"-" * 8}')
                for pt in prior_tasks:
                    recs   = [r for r in self.drift_records if r['prior_task'] == pt]
                    f_pt   = np.array([r['forgetting'] for r in recs])
                    ip_pt  = np.array([r[ip_key]       for r in recs])
                    if len(f_pt) >= 3:
                        tau, p_tau = kendalltau(f_pt, ip_pt)
                        sig = '*' if p_tau < 0.05 else (' .' if p_tau < 0.10 else '  ')
                        print(f'  T{pt:<5d}  {len(recs):>4d}  {tau:>+8.4f}  {p_tau:>8.4f}{sig}')

        print(f'{"=" * 70}')

        return {
            'global':  results,
            'records': self.drift_records,
        }


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='ViT-B/16 + TreeLoRA continual learning — vision benchmarks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument('--model_path', required=True,
                   help='Local path to iBOT ViT-B/16 checkpoint directory')
    p.add_argument('--lora_depth', type=int, default=5,
                   help='Max LoRA replacements (5 = paper default for ViT; -1 = all 24)')
    p.add_argument('--lora_r', type=int, default=8,
                   help='LoRA rank r')
    p.add_argument('--lora_alpha', type=int, default=32,
                   help='LoRA alpha scaling constant')

    # Dataset
    p.add_argument('--dataset', choices=['cifar100', 'imagenet_r', 'cub200'],
                   default='cifar100')
    p.add_argument('--data_root', required=True,
                   help='Parent directory where download_datasets.py saved the data')
    p.add_argument('--imagenet_r_tasks', type=int, choices=[5, 10, 20], default=10,
                   help='Number of tasks for ImageNet-R (paper evaluates 5, 10, 20)')
    p.add_argument('--seed', type=int, default=42,
                   help='Seed for class-order shuffle (controls task composition)')

    # Training — defaults match paper §A.3
    p.add_argument('--epochs_per_task', type=int, default=-1,
                   help='Training epochs per task (-1 = auto: 20 for CIFAR-100, 50 for others)')
    p.add_argument('--batch_size', type=int, default=192,
                   help='Batch size (paper: 192)')
    p.add_argument('--lr', type=float, default=0.005,
                   help='Learning rate — constant, no decay (paper: 0.005)')
    p.add_argument('--num_workers', type=int, default=4)

    # Sketch dimensions
    p.add_argument('--sketch_w_frac', type=float, default=0.05,
                   help='Sketch width as a fraction of total adapter params')
    p.add_argument('--sketch_d', type=int, default=8,
                   help='Sketch depth (number of independent hash rows)')

    # Tree regularisation
    p.add_argument('--reg', type=float, default=0.1,
                   help='Regularisation coefficient λ (0 = disable tree reg)')

    # Drift analysis
    p.add_argument('--drift_analysis', action='store_true', default=False,
                   help='Per-epoch sketch–forgetting drift tracking: evaluates all prior tasks '
                        'at the end of each epoch and records ip/l1diff vs. forgetting. '
                        'Use --reg 0 for unconstrained forgetting or reduce --reg for partial drift.')

    # Output
    p.add_argument('--output_dir', default='',
                   help='Directory for results.json (and checkpoints if --save_checkpoints)')
    p.add_argument('--save_checkpoints', action='store_true', default=False,
                   help='Save LoRA adapter + heads after each task (off by default)')
    p.add_argument('--log_dir', default='',
                   help='Directory for the run log file (default: <script_dir>/../vit_cl_logs)')

    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Logging setup ─────────────────────────────────────────────────────────
    _benchmark_slug = {
        'cifar100':   'split-cifar-100',
        'imagenet_r': 'split-imagenet-r',
        'cub200':     'split-cub-200',
    }[args.dataset]
    _log_dir = args.log_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'vit_cl_logs',
    )
    _log_path = os.path.join(_log_dir, f'vitb-16-21k-{_benchmark_slug}.log')
    _tee = _setup_logging(_log_path)
    print(f'Logging to {_log_path}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Auto-select epochs if not overridden
    if args.epochs_per_task < 0:
        args.epochs_per_task = _DEFAULT_EPOCHS[args.dataset]
        print(f'epochs_per_task auto-set to {args.epochs_per_task} '
              f'(paper default for {args.dataset})')

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f'\nLoading {args.dataset} …')
    if args.dataset == 'cifar100':
        task_info = make_split_cifar100(
            data_root=args.data_root,
            num_tasks=10,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    elif args.dataset == 'imagenet_r':
        task_info = make_split_imagenet_r(
            data_root=args.data_root,
            num_tasks=args.imagenet_r_tasks,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    else:  # cub200
        task_info = make_split_cub200(
            data_root=args.data_root,
            num_tasks=10,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )

    num_tasks = len(task_info)
    print(f'  {num_tasks} tasks, {task_info[0]["num_classes"]} classes/task')
    print(f'  Train batches (task 0): {len(task_info[0]["train"])}')
    print(f'  Test  batches (task 0): {len(task_info[0]["test"])}')

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f'\nLoading ViT from {args.model_path} …')
    model = build_treelora_vit(
        checkpoint_path=args.model_path,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_depth=args.lora_depth,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'  Trainable: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.3f}%)')

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = ViTCLTrainer(model, task_info, args)
    metrics = trainer.train_continual()

    print('\nDone.')
    _tee.close()
    sys.stdout = sys.__stdout__
    return metrics


if __name__ == '__main__':
    main()
