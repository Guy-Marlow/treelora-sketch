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
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.randsvd import rand_svd


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

        # KD-LoRA Tree (treelora method only)
        self.tree = None
        _method = getattr(args, 'method', 'treelora')
        if _method == 'olora' and args.reg > 0:
            print('  [O-LoRA] --reg ignored (tree regularisation not used with O-LoRA)')
        if _method == 'sketched_lora' and args.reg > 0:
            print('  [SketchedLoRA] --reg ignored (no tree regularisation with Sketched LoRA)')
        if _method == 'treelora' and args.reg > 0:
            args.num_tasks   = len(task_info)
            args.global_rank = 0   # single-GPU; suppresses rank-0 guards inside tree
            self.tree = KD_LoRA_Tree(args)

        # O-LoRA: number of tasks whose adapters have been consolidated into lora_A/B.
        # After consolidating task k, lora_A[adapter] has exactly k*r rows — the
        # modules are replaced with newly-allocated Linear(in, k*r) each task.
        self._olora_filled_tasks: int = 0

        # Sketched LoRA bank (Algorithm 1): per-task lists of per-layer (B, A) snapshots (CPU)
        self.svd_bank_B: list[list[torch.Tensor]] = []
        self.svd_bank_A: list[list[torch.Tensor]] = []


        num_tasks = len(task_info)
        # acc_matrix[i, j] = accuracy on task j after training through task i
        self.acc_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

        # Drift analysis: per-epoch sketch vs. forgetting tracking
        self.canonical_sketches: list   = []  # one {'cms': CMS, 'cs': CS} dict per completed task (A+B, normalised)
        self.canonical_A_mats:   list   = []  # one list[Tensor] of loranew_A params (CPU) per completed task
        self.canonical_accs: list       = []  # acc_matrix[t, t] for each completed task
        self.drift_records: list        = []  # dicts with forgetting + sketch metrics per epoch

        # Accumulator for batch-mean of loranew_A during the final training epoch.
        # Matches the paper's all_accumulate_grads[task_id] = mean of loranew_A
        # values across batches within the last epoch (reset each epoch via new_epoch_init).
        self._A_epoch_sum:   list | None = None
        self._A_epoch_steps: int         = 0

        _w_frac = getattr(args, 'sketch_w_frac', 0.05)
        self._sketch_d = getattr(args, 'sketch_d', 8)
        _adapter_total_params = sum(
            p.numel() for n, p in self.model.named_parameters()
            if 'loranew_A' in n or 'loranew_B' in n
        )
        self._sketch_w = math.ceil(_w_frac * _adapter_total_params)

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

    def _get_current_A_mats(self) -> list[torch.Tensor]:
        """Per-layer loranew_A tensors, detached and cloned to CPU."""
        return [
            p.data.detach().cpu().clone()
            for n, p in self.model.named_parameters()
            if 'loranew_A' in n
        ]

    def _get_epoch_mean_A_mats(self) -> list[torch.Tensor]:
        """
        Mean of loranew_A values accumulated across batches in the current epoch.
        Matches the paper's all_accumulate_grads[task_id] = mean(loranew_A over
        batches in the final epoch). Falls back to the current snapshot if the
        accumulator was never populated (e.g. drift_analysis was off).
        """
        if self._A_epoch_sum is None or self._A_epoch_steps == 0:
            return self._get_current_A_mats()
        return [s / self._A_epoch_steps for s in self._A_epoch_sum]

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

    # ── O-LoRA helpers ────────────────────────────────────────────────────────

    def _olora_loss(self) -> torch.Tensor:
        """
        Orthogonal regularisation loss between the current task's loranew_A and
        all prior tasks' A-matrices stored in each layer's frozen lora_A.

        With dynamic growth, lora_A.weight has exactly k*r rows — no slicing
        needed. Gradients flow only through loranew_A; lora_A is detached.
        """
        loss = torch.tensor(0.0, device=self.device)
        if self._olora_filled_tasks == 0:
            return loss
        lora_A_params    = [p for n, p in self.model.named_parameters()
                            if 'lora_A' in n and 'loranew' not in n]
        loranew_A_params = [p for n, p in self.model.named_parameters()
                            if 'loranew_A' in n]
        for frozen_A, cur_A in zip(lora_A_params, loranew_A_params):
            loss = loss + torch.abs(torch.mm(frozen_A.detach(), cur_A.T)).sum()
        return loss

    def _olora_consolidate_task(self):
        """
        Called at the end of each task (O-LoRA only).

        For each adapted Linear, concatenates the just-trained loranew_A/B
        weights onto the end of the frozen lora_A/B by replacing those
        nn.ModuleDict entries with newly-allocated, larger nn.Linear modules.
        After this call lora_A[adapter].weight has exactly (k+1)*r rows —
        no zero padding — so forward-pass FLOPs grow by exactly 2*seq*r*(d_in+d_out)
        per task, matching O-LoRA's theoretical cost curve.

        Then re-initialises loranew_A/B so the next task trains into a fresh adapter.
        """
        with torch.no_grad():
            for _, module in self.model.named_modules():
                if not (hasattr(module, 'lora_A') and hasattr(module, 'loranew_A')):
                    continue
                adapter_name = next(iter(module.lora_A))  # 'default'

                old_A  = module.lora_A[adapter_name]     # nn.Linear(d_in, k*r)
                old_B  = module.lora_B[adapter_name]     # nn.Linear(k*r, d_out)
                new_A  = module.loranew_A[adapter_name]  # nn.Linear(d_in, r)
                new_B  = module.loranew_B[adapter_name]  # nn.Linear(r, d_out)

                # weight shapes: lora_A → [k*r, d_in],  loranew_A → [r, d_in]
                grown_A_w = torch.cat([old_A.weight.data, new_A.weight.data], dim=0)
                # weight shapes: lora_B → [d_out, k*r], loranew_B → [d_out, r]
                grown_B_w = torch.cat([old_B.weight.data, new_B.weight.data], dim=1)

                new_r_sum = grown_A_w.shape[0]  # (k+1)*r

                grown_A = nn.Linear(module.in_features, new_r_sum, bias=False)
                grown_A.weight.data.copy_(grown_A_w)
                for p in grown_A.parameters():
                    p.requires_grad_(False)
                module.lora_A[adapter_name] = grown_A.to(self.device)

                grown_B = nn.Linear(new_r_sum, module.out_features, bias=False)
                grown_B.weight.data.copy_(grown_B_w)
                for p in grown_B.parameters():
                    p.requires_grad_(False)
                module.lora_B[adapter_name] = grown_B.to(self.device)

                # Re-initialise loranew for the next task
                nn.init.kaiming_uniform_(new_A.weight.data, a=math.sqrt(5))
                nn.init.zeros_(new_B.weight.data)

        self._olora_filled_tasks += 1
        r = self.args.lora_r
        print(f'  [O-LoRA] task consolidated: bank now {self._olora_filled_tasks} task(s), '
              f'{self._olora_filled_tasks * r} rows/layer in forward pass')

    # ── Sketched LoRA helpers (Algorithm 1) ──────────────────────────────────

    def _sketched_lora_snapshot(self):
        """
        Algorithm 1, Step 9: K ← K ∪ {(B^(n)_ℓ, A^(n)_ℓ)^L_{ℓ=1}}

        Appends CPU copies of the just-trained loranew_B and loranew_A to the
        adapter bank.  Order matches named_parameters() iteration, which is
        consistent across calls and matches _sketched_lora_compress().
        """
        snap_B = [
            p.data.detach().cpu().clone()
            for n, p in self.model.named_parameters() if 'loranew_B' in n
        ]
        snap_A = [
            p.data.detach().cpu().clone()
            for n, p in self.model.named_parameters() if 'loranew_A' in n
        ]
        self.svd_bank_B.append(snap_B)
        self.svd_bank_A.append(snap_A)
        print(f'  [SketchedLoRA] bank holds {len(self.svd_bank_A)} adapter(s)')

    def _sketched_lora_compress(self):
        """
        Algorithm 1, Steps 11-15: form ΔW_ℓ = B̂_ℓÂ_ℓ + Σ_k B_k A_k per layer,
        compress to rank r̂ via RandSVD, write back into lora_A/B, clear bank.

        On the first compression lora_A/B have 0 rows (r_sum started at 0), so we
        replace the nn.Linear modules with r̂-row ones rather than copying in-place.
        All subsequent compressions overwrite existing r̂-row modules in-place.
        """
        r_hat    = self.args.svd_rank
        oversamp = self.args.svd_oversample

        lora_modules = [
            (name, module) for name, module in self.model.named_modules()
            if hasattr(module, 'lora_A') and hasattr(module, 'loranew_A')
        ]
        num_layers   = len(lora_modules)
        adapter_name = next(iter(lora_modules[0][1].lora_A))

        # Detect first compression: lora_A currently has 0 rows
        first_compress = (
            lora_modules[0][1].lora_A[adapter_name].weight.shape[0] == 0
        )

        for layer_idx, (_, module) in enumerate(lora_modules):
            A_hat_w = module.lora_A[adapter_name].weight  # [0 or r̂, d_in]
            B_hat_w = module.lora_B[adapter_name].weight  # [d_out, 0 or r̂]

            # ΔW_ℓ = B̂_ℓ Â_ℓ  (zeros on first compress since weights are 0-row)
            # + Σ_{k∈K} B_k A_k
            delta_W = B_hat_w.data @ A_hat_w.data
            for snap_B, snap_A in zip(self.svd_bank_B, self.svd_bank_A):
                delta_W = delta_W + snap_B[layer_idx].to(self.device) @ snap_A[layer_idx].to(self.device)

            new_B, new_A = rand_svd(delta_W, r_hat, oversamp)  # [d_out, r̂], [r̂, d_in]

            if first_compress:
                grown_A = nn.Linear(module.in_features, r_hat, bias=False)
                grown_A.weight.data.copy_(new_A)
                for prm in grown_A.parameters(): prm.requires_grad_(False)
                module.lora_A[adapter_name] = grown_A.to(self.device)

                grown_B = nn.Linear(r_hat, module.out_features, bias=False)
                grown_B.weight.data.copy_(new_B)
                for prm in grown_B.parameters(): prm.requires_grad_(False)
                module.lora_B[adapter_name] = grown_B.to(self.device)
            else:
                A_hat_w.data.copy_(new_A)
                B_hat_w.data.copy_(new_B)

        self.svd_bank_B.clear()
        self.svd_bank_A.clear()
        print(f'  [SketchedLoRA] compressed {num_layers} layers → rank {r_hat}; bank cleared')

    def _sketched_lora_reinit(self):
        """
        Algorithm 1, Step 4: B^(n)_ℓ ← 0^{d_ℓ×r},  A^(n)_ℓ ∼ N(0, σ²)^{r×d_ℓ}  ∀ ℓ

        Re-initialises loranew_B/A for the next task.  Called after evaluation
        and (when applicable) compression, matching the algorithm's semantics:
        Step 4 logically precedes training on task n, but is implemented here as
        a post-task step so that evaluation can use the just-trained adapter.
        """
        for name, param in self.model.named_parameters():
            if 'loranew_A' in name:
                # A^(n)_ℓ ∼ N(0, σ²),  σ = 1/√d_in  (standard LoRA scale)
                nn.init.normal_(param.data, std=1.0 / math.sqrt(param.shape[1]))
            elif 'loranew_B' in name:
                nn.init.zeros_(param.data)

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

            # Reset batch-mean accumulator each epoch (only last epoch's mean is kept)
            if getattr(self.args, 'drift_analysis', False):
                self._A_epoch_sum   = None
                self._A_epoch_steps = 0

            running_loss = running_correct = running_total = 0
            pbar = tqdm(
                train_loader,
                desc=f'Task {task_id} | Epoch {epoch + 1}/{epochs}',
                leave=False,
            )

            for step, (images, labels) in enumerate(pbar):
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

                # O-LoRA orthogonal regularisation
                if getattr(self.args, 'method', 'treelora') == 'olora' and self._olora_filled_tasks > 0:
                    orth_loss = self._olora_loss()
                    loss = loss + self.args.olora_lambda * orth_loss
                    if step % 30 == 0:
                        print(f'    [O-LoRA] step={step}  orth_loss={orth_loss.item():.4f}  '
                              f'λ={self.args.olora_lambda}')

                # Accumulate loranew_A values BEFORE the gradient update, matching
                # the paper's insert_grad call site (before zero_grad/backward/step).
                # The epoch-start reset ensures only the current epoch's mean accumulates,
                # giving the intra-epoch running mean the paper uses for task similarity.
                # No epoch guard — we accumulate every epoch so the drift block can use
                # the epoch mean as the current-side representation each time it runs.
                if getattr(self.args, 'drift_analysis', False):
                    with torch.no_grad():
                        a_step = self._get_current_A_mats()
                        if self._A_epoch_sum is None:
                            self._A_epoch_sum = a_step
                        else:
                            self._A_epoch_sum = [s + a for s, a in zip(self._A_epoch_sum, a_step)]
                        self._A_epoch_steps += 1

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

                    scratch    = self._build_state_sketches()
                    cur_A_mats = self._get_epoch_mean_A_mats()

                    if task_id > 0:
                        for prior_t in range(task_id):
                            prior_acc   = self.canonical_accs[prior_t]
                            current_acc = self.evaluate_task(prior_t)
                            forgetting  = prior_acc - current_acc

                            # Sketch-based comparisons (LOI 1)
                            canon = self.canonical_sketches[prior_t]
                            ip    = float(scratch['cms'].inner_product(canon['cms']))
                            l1d   = float(scratch['cms'].l1_sketch_diff(canon['cms']))
                            c_ip  = float(scratch['cs'].inner_product(canon['cs']))
                            c_l1d = float(scratch['cs'].l1_sketch_diff(canon['cs']))

                            # Direct A-matrix comparisons (LOI 2, exact — no sketch)
                            can_A_mats = self.canonical_A_mats[prior_t]
                            cur_A_flat = torch.cat([m.reshape(-1) for m in cur_A_mats])
                            can_A_flat = torch.cat([m.reshape(-1) for m in can_A_mats])
                            a_l1 = float((cur_A_flat - can_A_flat).abs().sum())
                            a_ip = float(torch.dot(cur_A_flat, can_A_flat))
                            a_l1_layers = [
                                float((c.reshape(-1) - k.reshape(-1)).abs().sum())
                                for c, k in zip(cur_A_mats, can_A_mats)
                            ]
                            a_ip_layers = [
                                float(torch.dot(c.reshape(-1), k.reshape(-1)))
                                for c, k in zip(cur_A_mats, can_A_mats)
                            ]

                            record = {
                                'task': task_id, 'epoch': epoch, 'prior_task': prior_t,
                                'forgetting': forgetting,
                                'ip':   ip,   'l1diff':   l1d,
                                'c_ip': c_ip, 'c_l1diff': c_l1d,
                                'a_l1': a_l1, 'a_ip': a_ip,
                                'a_l1_layers': a_l1_layers,
                                'a_ip_layers': a_ip_layers,
                            }
                            self.drift_records.append(record)
                            print(
                                f'    [drift] vs T{prior_t}: '
                                f'forgetting={forgetting:+.4f}  '
                                f'cms_ip={ip:.4f}  cms_l1diff={l1d:.4f}  '
                                f'cs_ip={c_ip:.4f}  cs_l1diff={c_l1d:.4f}  '
                                f'A_ip={a_ip:.4f}  A_l1={a_l1:.4f}'
                            )
                    # At the last epoch: store canonicals for this task.
                    # canonical_A_mats uses the batch-mean of loranew_A across the final
                    # epoch, matching the paper's all_accumulate_grads[task_id].
                    if epoch == epochs - 1:
                        canon_acc = self.evaluate_task(task_id)
                        self.canonical_sketches.append(scratch)
                        self.canonical_A_mats.append(self._get_epoch_mean_A_mats())
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

    # ── Inference FLOP and memory profiler ───────────────────────────────────

    def _flop_and_memory_profile(self, task_id: int):
        """
        Report inference FLOPs and adapter memory after training task task_id.

        Three sections:
          [1] Actual forward-pass FLOPs via torch.utils.flop_counter.FlopCounterMode
              on a single dummy image — what the model currently executes.
          [2] Analytical adapter overhead per method, showing how it scales with
              task count.  O-LoRA grows as k*r rows per layer; Sketched LoRA is
              bounded at (r̂+r) rows.  This uses the correct per-token formula
              (seq_len=197 for 224×224 ViT-B/16) so numbers are exact.
          [3] Adapter memory footprint broken down by component and location.
        """
        from torch.utils.flop_counter import FlopCounterMode

        method      = getattr(self.args, 'method', 'treelora')
        r           = self.args.lora_r
        n_trained   = task_id + 1

        # ── Gather per-layer geometry ─────────────────────────────────────────
        loranew_A = [(n, p) for n, p in self.model.named_parameters() if 'loranew_A' in n]
        loranew_B = [(n, p) for n, p in self.model.named_parameters() if 'loranew_B' in n]
        frozen_A  = [(n, p) for n, p in self.model.named_parameters()
                     if 'lora_A' in n and 'loranew' not in n]
        frozen_B  = [(n, p) for n, p in self.model.named_parameters()
                     if 'lora_B' in n and 'loranew' not in n]

        num_layers = len(loranew_A)
        d_in  = loranew_A[0][1].shape[1] if loranew_A else 768   # [r, d_in]
        d_out = loranew_B[0][1].shape[0] if loranew_B else 768   # [d_out, r]
        # For Sketched LoRA: r̂ (compressed rank).
        # For O-LoRA: exactly k*r rows after k consolidated tasks (no zero padding).
        # For TreeLoRA: 0 (no frozen adapter).
        r_hat = frozen_A[0][1].shape[0] if frozen_A else 0

        # ViT-B/16 at 224×224: (224//16)² patches + 1 CLS token
        seq_len = (224 // 16) ** 2 + 1   # 197

        def adapter_flops(n_rows: int) -> int:
            """FLOPs for one LoRA pass (batch=1) across all adapted layers.

            Each LoRA pass does two matmuls per layer:
              x @ A.T  : [seq, d_in] × [d_in, r] → 2 * seq * d_in * r
              mid @ B.T: [seq, r]    × [r, d_out] → 2 * seq * r * d_out
            Combined: 2 * seq * r * (d_in + d_out) per layer.
            """
            return 2 * seq_len * n_rows * (d_in + d_out) * num_layers

        # ── [1] Actual FLOPs via FlopCounterMode ──────────────────────────────
        self.model.eval()
        dummy = torch.zeros(1, 3, 224, 224, device=self.device)
        with FlopCounterMode(display=False) as fcm:
            with torch.no_grad():
                _ = self.model(pixel_values=dummy)
        total_actual = fcm.get_total_flops()

        # Rows currently active in forward pass per layer:
        #   O-LoRA:        r_hat (= k*r, dynamically grown) + r (current loranew)
        #   Sketched LoRA: r̂ + r
        #   TreeLoRA:      0 + r
        actual_rows       = r_hat + r
        actual_adpt_flops = adapter_flops(actual_rows)
        # Backbone = everything that isn't the LoRA adapter matmuls
        backbone_flops    = total_actual - actual_adpt_flops

        # ── [2] Memory footprint ─────────────────────────────────────────────
        def _mb(tensors) -> float:
            return sum(t.numel() * t.element_size() for t in tensors) / (1024 ** 2)

        loranew_mb = _mb(p for _, p in loranew_A) + _mb(p for _, p in loranew_B)
        # O-LoRA: frozen_mb = exactly k*r rows (dynamically grown, no zero padding).
        # Sketched LoRA: frozen_mb = r̂ compressed rows.
        frozen_mb  = _mb(p for _, p in frozen_A) + _mb(p for _, p in frozen_B)
        svd_cpu_mb = (_mb(t for snap in self.svd_bank_B for t in snap) +
                      _mb(t for snap in self.svd_bank_A for t in snap))

        # Inference footprint: frozen adapter + loranew = total GPU adapter memory.
        inf_gpu_mb = loranew_mb + frozen_mb

        # ── Print ─────────────────────────────────────────────────────────────
        W = 70
        print(f'\n{"═"*W}')
        print(f'  Inference Profile — After Task {task_id}  '
              f'({n_trained} task(s) trained,  method={method})')
        print(f'{"═"*W}')

        # --- [1] Actual FLOPs ---
        print(f'\n  [1] Actual forward-pass FLOPs  '
              f'(FlopCounterMode, batch=1, seq={seq_len})')
        print(f'      {"Total model":30s}  {total_actual:>18,}  FLOPs')
        print(f'      {"Backbone (est.)":30s}  {backbone_flops:>18,}  FLOPs  (constant)')
        print(f'      {"Adapter (est.)":30s}  {actual_adpt_flops:>18,}  FLOPs  '
              f'({actual_rows} rows × {num_layers} layers)')

        # --- [2] Memory ---
        print(f'\n  [2] Adapter memory footprint')
        print(f'      {"Component":<46}  {"GPU (MB)":>9}  {"CPU (MB)":>9}')
        print(f'      {"─"*46}  {"─"*9}  {"─"*9}')
        _D = '—'
        print(f'      {"loranew_A/B  (current-task, always trainable)":<46}  {loranew_mb:>9.3f}  {_D:>9}')
        if method == 'olora':
            k_label = f'lora_A/B  (O-LoRA bank, {self._olora_filled_tasks} tasks × r rows)'
            print(f'      {k_label:<46}  {frozen_mb:>9.3f}  {_D:>9}')
        elif method == 'sketched_lora':
            print(f'      {"lora_A/B  (compressed history, r̂ rows)":<46}  {frozen_mb:>9.3f}  {_D:>9}')
            print(f'      {"SketchedLoRA pre-compress bank  (B+A, CPU)":<46}  {_D:>9}  {svd_cpu_mb:>9.3f}')
        else:
            print(f'      {"lora_A/B  (historical, r_sum rows)":<46}  {frozen_mb:>9.3f}  {_D:>9}')
        print(f'      {"─"*46}  {"─"*9}  {"─"*9}')
        print(f'      {"Total GPU at inference":<46}  {inf_gpu_mb:>9.3f}  {_D:>9}')
        print(f'\n{"═"*W}\n')

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

            # Profile FLOPs and memory after evaluation but before bank operations,
            # so we see the post-training state with the current adapter still active.
            self._flop_and_memory_profile(task_id)

            if getattr(self.args, 'method', 'treelora') == 'olora':
                self._olora_consolidate_task()

            if getattr(self.args, 'method', 'treelora') == 'sketched_lora':
                # Step 9: K ← K ∪ {(B^(n)_ℓ, A^(n)_ℓ)}
                self._sketched_lora_snapshot()
                # Steps 10-15: compress if at a period boundary or the final task
                if (task_id + 1) % self.args.svd_period == 0 or task_id == num_tasks - 1:
                    self._sketched_lora_compress()
                # Step 4 (deferred to post-eval): re-initialise for the next task
                self._sketched_lora_reinit()

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
        a_l1       = np.array([r['a_l1']       for r in self.drift_records]) if 'a_l1' in self.drift_records[0] else None
        a_ip       = np.array([r['a_ip']       for r in self.drift_records]) if 'a_ip' in self.drift_records[0] else None
        n          = len(forgetting)

        print(f'\n{"=" * 70}')
        print('  Drift Analysis — Sketch–Forgetting Correlation (per-epoch)')
        print(f'  {n} data points  (tasks × epochs × prior tasks)')
        print(f'{"=" * 70}')

        results = {}
        print(f'\n  {"Metric":<30}  {"τ":>8}  {"p(τ)":>8}  {"ρ":>8}  {"p(ρ)":>8}')
        print(f'  {"-" * 30}  {"-" * 8}  {"-" * 8}  {"-" * 8}  {"-" * 8}')

        all_metrics = [
            ('cm_state__ip',     ip),
            ('cm_state__l1diff', l1diff),
            ('c_state__ip',      c_ip),
            ('c_state__l1diff',  c_l1diff),
        ]
        if a_ip is not None:
            all_metrics += [
                ('A_global__ip',  a_ip),
                ('A_global__l1',  a_l1),
            ]

        for label, metric in all_metrics:
            tau, p_tau = kendalltau(forgetting, metric)
            rho, p_rho = spearmanr(forgetting,  metric)
            sig_tau = '*' if p_tau < 0.05 else (' .' if p_tau < 0.10 else '  ')
            sig_rho = '*' if p_rho < 0.05 else (' .' if p_rho < 0.10 else '  ')
            print(
                f'  {label:<30}  {tau:>+8.4f}  {p_tau:>8.4f}{sig_tau} '
                f'{rho:>+8.4f}  {p_rho:>8.4f}{sig_rho}'
            )
            results[label] = {
                'tau': float(tau), 'p_tau': float(p_tau),
                'rho': float(rho), 'p_rho': float(p_rho),
            }

        # Per-prior-task breakdown for the key ip metrics
        prior_tasks = sorted(set(r['prior_task'] for r in self.drift_records))
        _ip_keys = [('cm_state__ip', 'ip'), ('c_state__ip', 'c_ip')]
        if a_ip is not None:
            _ip_keys.append(('A_global__ip', 'a_ip'))
        if len(prior_tasks) > 1:
            for ip_label, ip_key in _ip_keys:
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

        # ── Per-layer A-matrix breakdown ──────────────────────────────────────
        if a_ip is not None and 'a_ip_layers' in self.drift_records[0]:
            num_layers = len(self.drift_records[0]['a_ip_layers'])
            print(f'\n  Per-layer A-matrix ip (A_global__ip split by layer):')
            print(f'  {"Layer":>6}  {"n":>4}  {"τ":>8}  {"p(τ)":>8}')
            print(f'  {"-"*6}  {"-"*4}  {"-"*8}  {"-"*8}')
            for j in range(num_layers):
                ip_j   = np.array([r['a_ip_layers'][j] for r in self.drift_records])
                tau_j, p_tau_j = kendalltau(forgetting, ip_j)
                sig = '*' if p_tau_j < 0.05 else (' .' if p_tau_j < 0.10 else '  ')
                print(f'  L{j:<5d}  {n:>4d}  {tau_j:>+8.4f}  {p_tau_j:>8.4f}{sig}')
            print()

        # ── First-differences analysis ────────────────────────────────────────
        # Correlate epoch-to-epoch CHANGES in forgetting against epoch-to-epoch
        # CHANGES in sketch metrics.  This removes any monotonic trend shared by
        # both series (e.g. l1diff always increases while forgetting drifts up
        # overall) and tests whether the rate of sketch movement genuinely tracks
        # the rate of forgetting change — a stricter and more informative test.
        _has_a = a_ip is not None
        d = {k: [] for k in ['forgetting', 'ip', 'l1diff', 'c_ip', 'c_l1diff', 'a_l1', 'a_ip']}

        sorted_recs = sorted(
            self.drift_records,
            key=lambda r: (r['task'], r['prior_task'], r['epoch'])
        )
        prev = None
        for r in sorted_recs:
            if (prev is not None
                    and r['task'] == prev['task']
                    and r['prior_task'] == prev['prior_task']):
                d['forgetting'].append(r['forgetting'] - prev['forgetting'])
                d['ip'].append(        r['ip']         - prev['ip'])
                d['l1diff'].append(    r['l1diff']     - prev['l1diff'])
                d['c_ip'].append(      r['c_ip']       - prev['c_ip'])
                d['c_l1diff'].append(  r['c_l1diff']   - prev['c_l1diff'])
                if _has_a:
                    d['a_l1'].append(  r['a_l1']       - prev['a_l1'])
                    d['a_ip'].append(  r['a_ip']       - prev['a_ip'])
            prev = r

        nd = len(d['forgetting'])
        if nd >= 3:
            d = {k: np.array(v) for k, v in d.items()}

            print(f'\n{"=" * 70}')
            print('  First-Differences Analysis (epoch-to-epoch Δ)')
            print(f'  {nd} consecutive-epoch pairs')
            print(f'{"=" * 70}')
            print(f'\n  {"Metric":<30}  {"τ":>8}  {"p(τ)":>8}  {"ρ":>8}  {"p(ρ)":>8}')
            print(f'  {"-" * 30}  {"-" * 8}  {"-" * 8}  {"-" * 8}  {"-" * 8}')

            d_metrics = [
                ('Δcm_state__ip',     d['ip']),
                ('Δcm_state__l1diff', d['l1diff']),
                ('Δc_state__ip',      d['c_ip']),
                ('Δc_state__l1diff',  d['c_l1diff']),
            ]
            if _has_a and d['a_ip']:
                d_metrics += [
                    ('ΔA_global__ip',  np.array(d['a_ip'])),
                    ('ΔA_global__l1',  np.array(d['a_l1'])),
                ]

            for label, metric in d_metrics:
                tau, p_tau = kendalltau(d['forgetting'], metric)
                rho, p_rho = spearmanr(d['forgetting'],  metric)
                sig_tau = '*' if p_tau < 0.05 else (' .' if p_tau < 0.10 else '  ')
                sig_rho = '*' if p_rho < 0.05 else (' .' if p_rho < 0.10 else '  ')
                print(
                    f'  {label:<30}  {tau:>+8.4f}  {p_tau:>8.4f}{sig_tau} '
                    f'{rho:>+8.4f}  {p_rho:>8.4f}{sig_rho}'
                )
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

    # Method selection
    p.add_argument('--method', choices=['treelora', 'olora', 'sketched_lora'],
                   default='treelora',
                   help='Continual learning method: treelora (KD-tree regularisation), '
                        'olora (orthogonal subspace regularisation), or '
                        'sketched_lora (randomised SVD adapter bank compression)')
    p.add_argument('--olora_lambda', type=float, default=0.5,
                   help='O-LoRA orthogonality regularisation coefficient λ₁ '
                        '(only used when --method olora)')

    # Sketched LoRA (Algorithm 1) hyperparameters
    p.add_argument('--svd_rank', type=int, default=-1,
                   help='Target compressed rank r̂ (-1 = same as --lora_r). '
                        'Only used with --method sketched_lora.')
    p.add_argument('--svd_period', type=int, default=1,
                   help='Compression period P: run RandSVD every P tasks. '
                        'Only used with --method sketched_lora.')
    p.add_argument('--svd_oversample', type=int, default=2,
                   help='Oversampling parameter p passed to RandSVD. '
                        'Only used with --method sketched_lora.')

    # Tree regularisation
    p.add_argument('--reg', type=float, default=0.1,
                   help='Regularisation coefficient λ (0 = disable tree reg; '
                        'ignored when --method olora)')

    # Drift analysis
    p.add_argument('--drift_analysis', action='store_true', default=False,
                   help='Per-epoch sketch–forgetting drift tracking: evaluates all prior tasks '
                        'at the end of each epoch and records ip/l1diff vs. forgetting. '
                        'Use --reg 0 for unconstrained forgetting or reduce --reg for partial drift.')

    # Task count limit (for quick trials)
    p.add_argument('--max_tasks', type=int, default=-1,
                   help='Stop after this many tasks (-1 = run all tasks in the dataset)')

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
    _method_slug = f'-{args.method}' if args.method != 'treelora' else ''
    _log_path = os.path.join(_log_dir, f'vitb-16-21k-{_benchmark_slug}{_method_slug}.log')
    _tee = _setup_logging(_log_path)
    print(f'Logging to {_log_path}')
    if args.method == 'sketched_lora' and args.svd_rank < 0:
        args.svd_rank = args.lora_r

    print(f'Method: {args.method}'
          + (f'  λ={args.olora_lambda}' if args.method == 'olora' else
             f'  r̂={args.svd_rank}  P={args.svd_period}  p={args.svd_oversample}'
             if args.method == 'sketched_lora' else
             f'  reg={args.reg}' if args.reg > 0 else '  reg=disabled'))

    random.seed(args.seed)
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

    if args.max_tasks > 0:
        task_info = task_info[:args.max_tasks]
        print(f'  (limited to {args.max_tasks} tasks by --max_tasks)')
    num_tasks = len(task_info)
    print(f'  {num_tasks} tasks, {task_info[0]["num_classes"]} classes/task')
    print(f'  Train batches (task 0): {len(task_info[0]["train"])}')
    print(f'  Test  batches (task 0): {len(task_info[0]["test"])}')

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f'\nLoading ViT from {args.model_path} …')
    # All methods start with r_sum=0 (empty frozen adapter). O-LoRA grows lora_A/B
    # by r rows per task; Sketched LoRA jumps to r̂ rows at first compression and
    # stays there. Neither pre-allocates unused capacity.
    r_sum = 0
    model = build_treelora_vit(
        checkpoint_path=args.model_path,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_depth=args.lora_depth,
        r_sum=r_sum,
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
