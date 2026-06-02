#!/usr/bin/env python3
"""
Parse a vit_cl_train drift-analysis log and run linear regression between
sketch metrics and forgetting.

Handles both log formats:
  New (with CS):  cms_ip=...  cms_l1diff=...  cs_ip=...  cs_l1diff=...
  Old (CMS only): ip=...  l1diff=...

Usage:
    python3 scripts/analyze_drift_log.py <logfile> [<logfile2> ...]

If multiple log files are given, results are printed separately for each.
"""

import re
import sys
import numpy as np
from scipy import stats


# ── Regex patterns ─────────────────────────────────────────────────────────────

_TASK_EPOCH_RE = re.compile(r'Task (\d+) \| Epoch (\d+)/\d+')

_DRIFT_NEW_RE = re.compile(
    r'\[drift\] vs T(\d+): '
    r'forgetting=([+-]?\d+\.\d+)\s+'
    r'cms_ip=([+-]?\d+\.\d+)\s+'
    r'cms_l1diff=([+-]?\d+\.\d+)\s+'
    r'cs_ip=([+-]?\d+\.\d+)\s+'
    r'cs_l1diff=([+-]?\d+\.\d+)'
)

_DRIFT_OLD_RE = re.compile(
    r'\[drift\] vs T(\d+): '
    r'forgetting=([+-]?\d+\.\d+)\s+'
    r'ip=([+-]?\d+\.\d+)\s+'
    r'l1diff=([+-]?\d+\.\d+)'
)


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_log(path: str) -> list[dict]:
    records = []
    current_task = None
    current_epoch = None

    with open(path) as f:
        for line in f:
            m = _TASK_EPOCH_RE.search(line)
            if m:
                current_task  = int(m.group(1))      # already 0-indexed in log
                current_epoch = int(m.group(2)) - 1  # log is 1-indexed
                continue

            m = _DRIFT_NEW_RE.search(line)
            if m:
                records.append({
                    'task':       current_task,
                    'epoch':      current_epoch,
                    'prior_task': int(m.group(1)),
                    'forgetting': float(m.group(2)),
                    'cms_ip':     float(m.group(3)),
                    'cms_l1diff': float(m.group(4)),
                    'cs_ip':      float(m.group(5)),
                    'cs_l1diff':  float(m.group(6)),
                })
                continue

            m = _DRIFT_OLD_RE.search(line)
            if m:
                records.append({
                    'task':       current_task,
                    'epoch':      current_epoch,
                    'prior_task': int(m.group(1)),
                    'forgetting': float(m.group(2)),
                    'cms_ip':     float(m.group(3)),
                    'cms_l1diff': float(m.group(4)),
                    'cs_ip':      None,
                    'cs_l1diff':  None,
                })

    return records


# ── Regression helpers ─────────────────────────────────────────────────────────

def _row(label: str, x: np.ndarray, y: np.ndarray) -> dict:
    slope, intercept, r, p, se = stats.linregress(x, y)
    sig = '*' if p < 0.05 else (' .' if p < 0.10 else '  ')
    print(
        f'  {label:<22}  {slope:>+10.6f}  {r**2:>6.4f}  {p:>8.4f}{sig}'
    )
    return {'label': label, 'slope': slope, 'intercept': intercept,
            'r2': r**2, 'p': p, 'n': len(x)}


def _section(title: str, n: int, records_or_arrays: dict, has_cs: bool):
    print(f'\n{"=" * 65}')
    print(f'  {title}  ({n} data points)')
    print(f'{"=" * 65}')
    print(f'  {"Metric":<22}  {"slope":>10}  {"R²":>6}  {"p":>8}')
    print(f'  {"-" * 22}  {"-" * 10}  {"-" * 6}  {"-" * 8}')

    forgetting = records_or_arrays['forgetting']
    _row('cms_ip'    if 'Δ' not in title else 'Δcms_ip',     records_or_arrays['cms_ip'],     forgetting)
    _row('cms_l1diff' if 'Δ' not in title else 'Δcms_l1diff', records_or_arrays['cms_l1diff'], forgetting)
    if has_cs:
        _row('cs_ip'    if 'Δ' not in title else 'Δcs_ip',     records_or_arrays['cs_ip'],     forgetting)
        _row('cs_l1diff' if 'Δ' not in title else 'Δcs_l1diff', records_or_arrays['cs_l1diff'], forgetting)


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(records: list[dict], path: str):
    if not records:
        print(f'  No drift records found in {path}')
        return

    has_cs = any(r['cs_ip'] is not None for r in records)
    n      = len(records)

    forgetting = np.array([r['forgetting'] for r in records])
    cms_ip     = np.array([r['cms_ip']     for r in records])
    cms_l1diff = np.array([r['cms_l1diff'] for r in records])

    print(f'\n  forgetting: mean={forgetting.mean():+.4f}  std={forgetting.std():.4f}'
          f'  range=[{forgetting.min():+.4f}, {forgetting.max():+.4f}]')

    abs_arrays = {
        'forgetting': forgetting,
        'cms_ip':     cms_ip,
        'cms_l1diff': cms_l1diff,
    }
    if has_cs:
        abs_arrays['cs_ip']     = np.array([r['cs_ip']     for r in records])
        abs_arrays['cs_l1diff'] = np.array([r['cs_l1diff'] for r in records])

    _section('Absolute values', n, abs_arrays, has_cs)

    # ── First differences ─────────────────────────────────────────────────────
    d = {k: [] for k in ['forgetting', 'cms_ip', 'cms_l1diff', 'cs_ip', 'cs_l1diff']}

    sorted_recs = sorted(records, key=lambda r: (r['task'], r['prior_task'], r['epoch']))
    prev = None
    for r in sorted_recs:
        if (prev is not None
                and r['task'] == prev['task']
                and r['prior_task'] == prev['prior_task']):
            d['forgetting'].append(r['forgetting'] - prev['forgetting'])
            d['cms_ip'].append(    r['cms_ip']     - prev['cms_ip'])
            d['cms_l1diff'].append(r['cms_l1diff'] - prev['cms_l1diff'])
            if has_cs and r['cs_ip'] is not None and prev['cs_ip'] is not None:
                d['cs_ip'].append(    r['cs_ip']     - prev['cs_ip'])
                d['cs_l1diff'].append(r['cs_l1diff'] - prev['cs_l1diff'])
        prev = r

    nd = len(d['forgetting'])
    if nd < 3:
        return

    diff_arrays = {k: np.array(v) for k, v in d.items() if v}
    _section('Δ First differences', nd, diff_arrays, has_cs and bool(d['cs_ip']))


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    paths = sys.argv[1:]
    if not paths:
        print(f'Usage: {sys.argv[0]} <logfile> [<logfile2> ...]')
        sys.exit(1)

    for path in paths:
        print(f'\n{"#" * 65}')
        print(f'  {path}')
        print(f'{"#" * 65}')
        records = parse_log(path)
        print(f'  Parsed {len(records)} drift records')
        analyze(records, path)


if __name__ == '__main__':
    main()
