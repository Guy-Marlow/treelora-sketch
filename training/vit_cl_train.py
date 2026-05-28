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

        # Sketch bank: one dict of 7 CountMinSketches per task, built in train_one_task
        self.sketch_bank: list[dict] = []
        self._adapter_total_params = sum(
            p.numel() for n, p in self.model.named_parameters()
            if 'loranew_A' in n or 'loranew_B' in n
        )
        self._sketch_w = math.ceil(0.05 * self._adapter_total_params)
        self._sketch_d = 8
        # Separate width for the AB-product sketch (d_out×d_in per layer >> r×d per layer)
        _A_shapes = [p.shape for n, p in self.model.named_parameters() if 'loranew_A' in n]
        _B_shapes = [p.shape for n, p in self.model.named_parameters() if 'loranew_B' in n]
        self._ab_total_params = sum(b[0] * a[1] for a, b in zip(_A_shapes, _B_shapes))
        self._sketch_w_ab = math.ceil(0.05 * self._ab_total_params)

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

    def _flat_adapter_vec(self, use_grad: bool = False) -> torch.Tensor | None:
        """
        Return a 1-D detached tensor: all loranew_A params/grads (in layer order)
        concatenated with all loranew_B params/grads.  Returns None if any entry
        is missing (e.g. grads not yet computed).
        """
        parts_A, parts_B = [], []
        for name, p in self.model.named_parameters():
            src = p.grad if use_grad else p.data
            if 'loranew_A' in name:
                if src is None:
                    return None
                parts_A.append(src.detach().reshape(-1))
            elif 'loranew_B' in name:
                if src is None:
                    return None
                parts_B.append(src.detach().reshape(-1))
        if not parts_A and not parts_B:
            return None
        return torch.cat(parts_A + parts_B)

    def _flat_ab_product_vec(self) -> torch.Tensor | None:
        """
        For each LoRA layer (in parameter iteration order), compute
        (loranew_B.weight @ loranew_A.weight) and concatenate the flattened
        products.  Returns None if any A/B pair is missing or counts differ.
        """
        A_params, B_params = [], []
        for name, p in self.model.named_parameters():
            if 'loranew_A' in name:
                A_params.append(p.detach())
            elif 'loranew_B' in name:
                B_params.append(p.detach())
        if not A_params or len(A_params) != len(B_params):
            return None
        return torch.cat([(B @ A).reshape(-1) for A, B in zip(A_params, B_params)])

    def _init_task_sketches(self) -> dict:
        """Create 7 fresh sketches for one task (CMS or CountSketch per --sketch_type)."""
        use_cs = getattr(self.args, 'sketch_type', 'cms') == 'cs'
        if use_cs:
            def make():
                return CountSketch(
                    self._sketch_d, self._sketch_w,
                    device=self.device, dtype=torch.float32,
                )
            return {
                'cm_grad_abs':            make(),
                'cm_grad_squared':        make(),
                'cm_taylor':              make(),
                'cm_weight_diff':         make(),
                'cm_weight_diff_squared': make(),
                'cm_state':               make(),
                'cm_state_ab':            CountSketch(
                    self._sketch_d, self._sketch_w_ab,
                    device=self.device, dtype=torch.float32,
                ),
            }
        else:
            def make():
                return CountMinSketch(
                    self._sketch_d, self._sketch_w,
                    device=self.device, dtype=torch.float32,
                )
            return {
                'cm_grad_abs':            make(),
                'cm_grad_squared':        make(),
                'cm_taylor':              make(),
                'cm_weight_diff':         make(),
                'cm_weight_diff_squared': make(),
                'cm_state':               make(),
                'cm_state_ab':            CountMinSketch(
                    self._sketch_d, self._sketch_w_ab,
                    device=self.device, dtype=torch.float32,
                ),
            }

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

        # Fresh set of 7 sketches for this task
        sketches  = self._init_task_sketches()
        use_cs    = getattr(self.args, 'sketch_type', 'cms') == 'cs'

        # CountSketch: accumulate signed running sums; insert once at task end.
        # CMS:         insert abs values per batch throughout training.
        cs_grad_acc        = None
        cs_taylor_acc      = None
        cs_weight_diff_acc = None
        cs_batch_count     = 0

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

                # ── Per-batch sketch updates (read-only; no graph impact) ────
                pre_step = None
                with torch.no_grad():
                    grad_flat  = self._flat_adapter_vec(use_grad=True)
                    param_flat = self._flat_adapter_vec(use_grad=False)
                    if grad_flat is not None and param_flat is not None:
                        if use_cs:
                            # Accumulate signed quantities; insert at task end.
                            cs_grad_acc   = grad_flat        if cs_grad_acc   is None else cs_grad_acc   + grad_flat
                            cs_taylor_acc = grad_flat * param_flat if cs_taylor_acc is None else cs_taylor_acc + grad_flat * param_flat
                        else:
                            sketches['cm_grad_abs'].insert_vec(grad_flat.abs())
                            sketches['cm_taylor'].insert_vec((grad_flat * param_flat).abs())
                        # Squared quantities are always positive — insert per-batch for both paths.
                        sketches['cm_grad_squared'].insert_vec(grad_flat ** 2)
                        pre_step = param_flat

                optimizer.step()

                with torch.no_grad():
                    if pre_step is not None:
                        post_flat = self._flat_adapter_vec(use_grad=False)
                        diff = post_flat - pre_step
                        if use_cs:
                            cs_weight_diff_acc = diff if cs_weight_diff_acc is None else cs_weight_diff_acc + diff
                        else:
                            sketches['cm_weight_diff'].insert_vec(diff.abs())
                        sketches['cm_weight_diff_squared'].insert_vec(diff ** 2)

                if use_cs and grad_flat is not None:
                    cs_batch_count += 1
                # ── End per-batch sketch updates ─────────────────────────────

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

        # ── Task boundary: record final adapter state, store sketches ─────────
        with torch.no_grad():
            # Signed accumulated quantities — insert mean for CountSketch path.
            if use_cs and cs_batch_count > 0:
                sketches['cm_grad_abs'].insert_vec(cs_grad_acc / cs_batch_count)
                sketches['cm_taylor'].insert_vec(cs_taylor_acc / cs_batch_count)
                sketches['cm_weight_diff'].insert_vec(cs_weight_diff_acc / cs_batch_count)

            # State snapshots use abs() for both paths: direction is not meaningful
            # for a final-position comparison, only magnitude is.
            state_flat = self._flat_adapter_vec(use_grad=False)
            if state_flat is not None:
                sketches['cm_state'].insert_vec(state_flat.abs())
            ab_flat = self._flat_ab_product_vec()
            if ab_flat is not None:
                sketches['cm_state_ab'].insert_vec(ab_flat.abs())

        self.sketch_bank.append(sketches)

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
        sketch_results = self.analyze_sketch_correlations()
        metrics['sketch_analysis'] = sketch_results
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

    # ── Sketch–forgetting correlation analysis ────────────────────────────────

    def analyze_sketch_correlations(self) -> dict:
        """
        For all (t, t') pairs where t > t', compare the sketches built during
        training on task t against those from task t', and measure how well
        each comparison metric predicts the forgetting of task t' by model M_t.

        All pairs are included — including backward-transfer cases (forgetting ≤ 0).
        The hypothesis is monotonic: increasing forgetting should correlate with
        decreasing sketch similarity (lower inner_product, higher l1_sketch_diff).
        Backward-transfer pairs sit at the low end of the forgetting spectrum and
        provide additional support for the same relationship.

        Table layout  (15 rows × n_pairs columns):
            row  0          : forgetting = acc_matrix[t',t'] - acc_matrix[t,t']
            rows 1,3,5,...  : inner_product(sketch_t, sketch_t')   per sketch
            rows 2,4,6,...  : l1_sketch_diff(sketch_t, sketch_t')  per sketch

        Sketch order: cm_grad_abs, cm_grad_squared, cm_taylor,
                      cm_weight_diff, cm_weight_diff_squared, cm_state,
                      cm_state_ab

        Kendall's tau is computed between the forgetting row and every metric
        row when at least 3 pairs are available.
        """
        from scipy.stats import kendalltau

        num_tasks = len(self.task_info)
        sketch_names = [
            'cm_grad_abs', 'cm_grad_squared', 'cm_taylor',
            'cm_weight_diff', 'cm_weight_diff_squared', 'cm_state',
            'cm_state_ab',
        ]

        # All (t, t') pairs with t > t', ordered outer-by-t, inner-by-t'
        pairs   = [(t, tp) for t in range(num_tasks) for tp in range(t)]
        n_pairs = len(pairs)

        if n_pairs == 0:
            print('\n  [sketch analysis] Need ≥ 2 tasks; nothing to compare.')
            return {}

        row_labels = ['forgetting']
        for sk in sketch_names:
            row_labels += [f'{sk}__ip', f'{sk}__l1diff']
        n_rows = len(row_labels)   # 1 + 7*2 = 15

        table = np.zeros((n_rows, n_pairs), dtype=np.float64)

        for col, (t, tp) in enumerate(pairs):
            # Forgetting: peak accuracy on task tp minus current model's accuracy on tp
            table[0, col] = float(self.acc_matrix[tp, tp] - self.acc_matrix[t, tp])

            for sk_idx, sk_name in enumerate(sketch_names):
                sk_t  = self.sketch_bank[t][sk_name]
                sk_tp = self.sketch_bank[tp][sk_name]

                ip  = sk_t.inner_product(sk_tp)
                l1d = sk_t.l1_sketch_diff(sk_tp)

                table[1 + sk_idx * 2,     col] = float(ip)
                table[1 + sk_idx * 2 + 1, col] = float(l1d)

        # ── Print raw table ───────────────────────────────────────────────────
        pair_labels = [f'(M{t}←T{tp})' for (t, tp) in pairs]
        col_w = max(10, max(len(l) for l in pair_labels) + 1)

        print(f'\n{"=" * 70}')
        print('  Sketch–Forgetting Correlation Analysis')
        print(f'  {n_pairs} task pair(s)  ·  {n_rows - 1} sketch metrics')
        print(f'{"=" * 70}')

        # Header row
        header = f'  {"Metric":<42}' + ''.join(f'{l:>{col_w}}' for l in pair_labels)
        print(header)
        print(f'  {"-" * 42}' + '-' * (col_w * n_pairs))

        for row_idx, label in enumerate(row_labels):
            vals = ''.join(f'{table[row_idx, col]:>{col_w}.4f}' for col in range(n_pairs))
            print(f'  {label:<42}{vals}')

        # ── Kendall's tau ─────────────────────────────────────────────────────
        tau_results = {}
        if n_pairs >= 3:
            forgetting = table[0, :]
            print(f'\n  {"Metric":<42}  {"τ":>8}  {"p-val":>8}')
            print(f'  {"-" * 42}  {"-" * 8}  {"-" * 8}')
            for row_idx in range(1, n_rows):
                label = row_labels[row_idx]
                tau, pval = kendalltau(forgetting, table[row_idx, :])
                tau_results[label] = {'tau': float(tau), 'pval': float(pval)}
                sig = '*' if pval < 0.05 else (' .' if pval < 0.10 else '  ')
                print(f'  {label:<42}  {tau:>8.4f}  {pval:>8.4f} {sig}')
        else:
            print(f'\n  (Kendall τ requires ≥ 3 pairs; have {n_pairs} — raw values above.)')

        print(f'{"=" * 70}')

        return {
            'table':      table.tolist(),
            'row_labels': row_labels,
            'pairs':      [(int(t), int(tp)) for (t, tp) in pairs],
            'tau':        tau_results,
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

    # Sketch algorithm
    p.add_argument('--sketch_type', choices=['cms', 'cs'], default='cms',
                   help='cms = CountMinSketch (per-batch abs inserts); '
                        'cs  = CountSketch (JL projection, signed mean gradient)')

    # Tree regularisation
    p.add_argument('--reg', type=float, default=0.1,
                   help='Regularisation coefficient λ (0 = disable tree reg)')

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
