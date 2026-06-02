#!/usr/bin/env python3
"""
Plot forgetting vs cms_ip for task 9 vs prior tasks 2, 5, 8 (CUB-200, no reg).

Row 1: epoch-to-epoch Δ for both metrics (y-axis fixed to [-0.02, 0.02]).
Row 2: absolute values at each epoch, with consistent y-axis scale across panels.

Usage:
    python3 scripts/plot_drift_deltas.py
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

LOG_PATH    = os.path.join(os.path.dirname(__file__), '..', 'vit_cl_logs',
                           'drift_noreg', 'vitb-16-21k-split-cub-200.log')
TARGET_TASK = 9
PRIOR_TASKS = [2, 5, 8]
OUT_DIR     = os.path.join(os.path.dirname(__file__), '..', 'vit_cl_logs', 'drift_noreg')
DELTA_YLIM  = (-0.02, 0.02)

_TASK_EPOCH_RE = re.compile(r'Task (\d+) \| Epoch (\d+)/\d+')
_DRIFT_RE = re.compile(
    r'\[drift\] vs T(\d+): '
    r'forgetting=([+-]?\d+\.\d+)\s+'
    r'cms_ip=([+-]?\d+\.\d+)'
)

# ── Parse ──────────────────────────────────────────────────────────────────────

records = []
current_task = current_epoch = None

with open(LOG_PATH) as f:
    for line in f:
        m = _TASK_EPOCH_RE.search(line)
        if m:
            current_task  = int(m.group(1))
            current_epoch = int(m.group(2)) - 1
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

task9 = [r for r in records if r['task'] == TARGET_TASK]

# Pre-compute series for all prior tasks so we can find global y-limits for row 2
series = {}
for prior_t in PRIOR_TASKS:
    recs = sorted([r for r in task9 if r['prior_task'] == prior_t],
                  key=lambda r: r['epoch'])
    epochs     = [r['epoch']      for r in recs]
    forgetting = [r['forgetting'] for r in recs]
    cms_ip     = [r['cms_ip']     for r in recs]
    d_epoch  = epochs[1:]
    d_forget = [forgetting[i] - forgetting[i - 1] for i in range(1, len(forgetting))]
    d_ip     = [cms_ip[i]     - cms_ip[i - 1]     for i in range(1, len(cms_ip))]
    series[prior_t] = dict(epochs=epochs, forgetting=forgetting, cms_ip=cms_ip,
                           d_epoch=d_epoch, d_forget=d_forget, d_ip=d_ip)

# Global y-limits for row 2 (consistent across panels per metric)
all_forget = [v for pt in PRIOR_TASKS for v in series[pt]['forgetting']]
all_ip     = [v for pt in PRIOR_TASKS for v in series[pt]['cms_ip']]

pad = 0.05
f_lo = min(all_forget) - pad * (max(all_forget) - min(all_forget))
f_hi = max(all_forget) + pad * (max(all_forget) - min(all_forget))
ip_lo = min(all_ip) - pad * (max(all_ip) - min(all_ip))
ip_hi = max(all_ip) + pad * (max(all_ip) - min(all_ip))

# ── Plot ───────────────────────────────────────────────────────────────────────

sns.set_theme(style='whitegrid', font_scale=1.05)
palette    = sns.color_palette('tab10')
col_forget = palette[0]
col_ip     = palette[1]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Task 9 vs prior Tasks 2, 5, 8  (CUB-200, no reg)',
             fontsize=14, y=1.01)

for col, prior_t in enumerate(PRIOR_TASKS):
    s = series[prior_t]

    # ── Row 0: first differences ─────────────────────────────────────────────
    ax  = axes[0, col]
    ax2 = ax.twinx()

    ln1, = ax.plot(s['d_epoch'], s['d_forget'], color=col_forget, linewidth=1.4,
                   label='Δ forgetting')
    ln2, = ax2.plot(s['d_epoch'], s['d_ip'],    color=col_ip,     linewidth=1.4,
                    label='Δ cms_ip', linestyle='--')

    ax.axhline(0,  color=col_forget, linewidth=0.5, linestyle=':', alpha=0.5)
    ax2.axhline(0, color=col_ip,     linewidth=0.5, linestyle=':', alpha=0.5)

    ax.set_ylim(DELTA_YLIM)
    ax2.set_ylim(DELTA_YLIM)
    ax.set_xlim(s['d_epoch'][0], s['d_epoch'][-1])

    ax.set_title(f'vs prior Task {prior_t}', fontsize=12)
    ax.set_ylabel('Δ forgetting', color=col_forget, fontsize=10)
    ax2.set_ylabel('Δ cms_ip',    color=col_ip,     fontsize=10)
    ax.tick_params(axis='y', labelcolor=col_forget)
    ax2.tick_params(axis='y', labelcolor=col_ip)

    lines  = [ln1, ln2]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=8)

    # ── Row 1: absolute values ────────────────────────────────────────────────
    ax  = axes[1, col]
    ax2 = ax.twinx()

    ln1, = ax.plot(s['epochs'], s['forgetting'], color=col_forget, linewidth=1.4,
                   label='forgetting')
    ln2, = ax2.plot(s['epochs'], s['cms_ip'],    color=col_ip,     linewidth=1.4,
                    label='cms_ip', linestyle='--')

    ax.set_ylim(f_lo,  f_hi)
    ax2.set_ylim(ip_lo, ip_hi)
    ax.set_xlim(s['epochs'][0], s['epochs'][-1])

    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('forgetting', color=col_forget, fontsize=10)
    ax2.set_ylabel('cms_ip',    color=col_ip,     fontsize=10)
    ax.tick_params(axis='y', labelcolor=col_forget)
    ax2.tick_params(axis='y', labelcolor=col_ip)

    lines  = [ln1, ln2]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=8)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'task9_drift_deltas.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
