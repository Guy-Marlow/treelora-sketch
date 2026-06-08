#!/usr/bin/env python3
"""
Plot Δforgetting and Δcms_ip across training epochs for the final task
of three H200 benchmarks: CIFAR-100 10t, ImageNet-R 10t, ImageNet-R 20t.

Layout: 1 row × 3 columns — one panel per prior task (T0, T1, T2).
Each panel overlays all three benchmarks with dual y-axes:
  - Left  (solid)  : Δforgetting
  - Right (dashed) : Δcms_ip
Both series are smoothed with a Gaussian filter.

Usage:
    python3 scripts/plot_h200_deltas.py
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# ── Config ─────────────────────────────────────────────────────────────────────

BASE = os.path.join(os.path.dirname(__file__), '..', 'vit_cl_logs', 'H200_logs_sketching')

BENCHMARKS = [
    ('CIFAR-100 10t',    os.path.join(BASE, 'cifar_10t',    'log.txt')),
    ('ImageNet-R 10t',   os.path.join(BASE, 'imagenet_10t', 'log.txt')),
    ('ImageNet-R 20t',   os.path.join(BASE, 'imagenet_20t', 'log.txt')),
]

PRIOR_TASKS  = [0, 1, 2]          # T0, T1, T2  (tasks 1, 2, 3 in 1-indexed)
SMOOTH_SIGMA = 2.0                 # Gaussian σ in epoch units
OUT_PATH     = os.path.join(BASE, 'h200_delta_comparison.png')

# ── Regex ──────────────────────────────────────────────────────────────────────

_TASK_EPOCH_RE = re.compile(r'Task (\d+) \| Epoch (\d+)/(\d+)')
_DRIFT_RE      = re.compile(
    r'\[drift\] vs T(\d+): forgetting=([+-]?\d+\.\d+)\s+cms_ip=([+-]?\d+\.\d+)'
)

# ── Parse ──────────────────────────────────────────────────────────────────────

def parse_last_task(path: str) -> dict[int, dict]:
    """
    Return {prior_task: {'epochs': [...], 'forgetting': [...], 'cms_ip': [...]}}
    for the final task only.
    """
    records = []
    current_task = current_epoch = None
    last_task_id = -1

    with open(path) as f:
        for line in f:
            m = _TASK_EPOCH_RE.search(line)
            if m:
                current_task  = int(m.group(1))
                current_epoch = int(m.group(2))
                last_task_id  = max(last_task_id, current_task)
                continue
            m = _DRIFT_RE.search(line)
            if m:
                records.append({
                    'task':       current_task,
                    'epoch':      current_epoch,
                    'prior_task': int(m.group(1)),
                    'forgetting': float(m.group(2)),
                    'cms_ip':     float(m.group(3)),
                })

    last = [r for r in records if r['task'] == last_task_id]
    result = {}
    for pt in PRIOR_TASKS:
        recs = sorted([r for r in last if r['prior_task'] == pt],
                      key=lambda r: r['epoch'])
        if not recs:
            continue
        result[pt] = {
            'epochs':     [r['epoch']      for r in recs],
            'forgetting': [r['forgetting'] for r in recs],
            'cms_ip':     [r['cms_ip']     for r in recs],
        }
    return result


def first_diff(series: list[float]) -> np.ndarray:
    a = np.array(series)
    return a[1:] - a[:-1]


def smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(arr, sigma=max(sigma, 0.5), mode='nearest')


# ── Load all data ──────────────────────────────────────────────────────────────

all_data = {}
for label, path in BENCHMARKS:
    all_data[label] = parse_last_task(path)

# ── Plot ───────────────────────────────────────────────────────────────────────

sns.set_theme(style='whitegrid', font_scale=1.05)
palette = sns.color_palette('tab10')
bm_colors = {label: palette[i] for i, (label, _) in enumerate(BENCHMARKS)}

fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.suptitle(
    'Forgetting graphed against cms inner product — final task vs prior tasks T1, T2, T3\n'
    'H200 runs  (Gaussian smoothed, z-score normalised)',
    fontsize=13, y=1.01,
)

def zscore(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    return (arr - arr.mean()) / std if std > 0 else arr - arr.mean()

for row, (label, _) in enumerate(BENCHMARKS):
    color = bm_colors[label]
    for col, prior_t in enumerate(PRIOR_TASKS):
        ax = axes[row, col]

        data = all_data[label].get(prior_t)
        if data is None:
            continue

        forgetting = zscore(smooth(np.array(data['forgetting']), SMOOTH_SIGMA))
        cms_ip     = zscore(smooth(np.array(data['cms_ip']),     SMOOTH_SIGMA))
        n          = len(forgetting)
        x          = np.linspace(0, 1, n)

        ln1, = ax.plot(x, forgetting, color=color,   linewidth=1.8, linestyle='-',
                       label='forgetting')
        ln2, = ax.plot(x, cms_ip,     color='black', linewidth=1.4, linestyle='--',
                       label='cms_ip', alpha=0.7)

        ax.axhline(0, color='grey', linewidth=0.6, linestyle=':', alpha=0.7)

        if row == 0:
            ax.set_title(f'vs prior Task {prior_t + 1}  (T{prior_t})', fontsize=12)
        if col == 0:
            ax.set_ylabel(f'{label}\nz-score', fontsize=9)
        else:
            ax.set_ylabel('z-score', fontsize=9)
        if row == 2:
            ax.set_xlabel('Training progress (fraction of final task)', fontsize=9)

        ax.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()],
                  loc='best', fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved → {OUT_PATH}')
