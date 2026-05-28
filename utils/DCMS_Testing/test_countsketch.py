#!/usr/bin/env python3
"""
Empirical comparison of CountSketch (CS) vs CountMinSketch (CMS) on
dense vectors matching the ViT/LoRA adapter parameter scale.

Tests
-----
1. Point-query accuracy           — bias and RMSE per element
2. Inner-product estimation       — bias, std dev, theory vs observed
3. L1 pairwise-distance ranking   — Kendall τ: l1_sketch_diff vs ||a-b||_1 and ||a-b||_2
4. IP pairwise-similarity ranking — Kendall τ: inner_product vs true <a,b>
5. Width sensitivity              — τ as a function of sketch width w

All tests use CPU (no GPU required).  Vectors are dense Gaussian, matching
the expected distribution of LoRA adapter gradients (dense backprop signal
through 768-dim attention layers with no sparsity-inducing nonlinearity).
"""

import math
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.dyadic_cms import CountMinSketch, CountSketch

# ── Parameters matching the real ViT/LoRA use case ────────────────────────────
N         = 61_440   # adapter vector length (5 layers × r=8 × d=768 A+B concat)
D         = 8        # sketch depth
W_DEFAULT = 3_072    # 5 % of N — current default
N_TASKS   = 25       # task vectors (→ 300 pairs)
SEED      = 42


# ── Vector generation ─────────────────────────────────────────────────────────

def make_task_vectors(n_tasks, n, rng):
    """
    Simulate realistic adapter gradient vectors.
    Each vector is dense Gaussian noise plus a random linear combination of
    5 shared 'topic' directions, giving controlled inter-task similarity.
    """
    basis = rng.standard_normal((5, n)).astype(np.float32)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    vecs = []
    for _ in range(n_tasks):
        coefs = rng.standard_normal(5).astype(np.float32)
        noise = rng.standard_normal(n).astype(np.float32)
        vecs.append(torch.from_numpy(noise + (basis * coefs[:, None]).sum(0)))
    return vecs


# ── Sketch factories ──────────────────────────────────────────────────────────

def make_cms(vec, d, w):
    s = CountMinSketch(d, w, device='cpu', dtype=torch.float32)
    s.insert_vec(vec.abs())
    return s


def make_cs(vec, d, w):
    s = CountSketch(d, w, device='cpu', dtype=torch.float32)
    s.insert_vec(vec)   # signed
    return s


# ── CS point-query helper (sign × bucket, median across rows) ─────────────────

def cs_point_query(cs_sketch, indices: torch.Tensor) -> np.ndarray:
    """Return median{g_i(j) * C[i, h_i(j)]} for each j in indices."""
    row_ests = []
    for row in range(cs_sketch.d):
        h = cs_sketch.hash_funcs[row](indices)
        g = cs_sketch.sign_funcs[row](indices).float()
        row_ests.append((g * cs_sketch.cs[row][h]).numpy())
    return np.median(np.stack(row_ests, axis=0), axis=0)


def all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# ── Test 1: Point-query accuracy ─────────────────────────────────────────────

def test_point_query(vecs, d, w, n_sample=500):
    sep()
    print(f'  Test 1: Point-query accuracy  (d={d}, w={w}, n={N:,}, sample={n_sample})')
    sep()

    rng_idx = np.random.default_rng(0)
    sample  = torch.from_numpy(rng_idx.choice(N, n_sample, replace=False).astype(np.int64))

    cms_biases, cms_rmses = [], []
    cs_biases,  cs_rmses  = [], []

    for v in vecs[:8]:
        true_abs    = v.abs().numpy()[sample.numpy()]
        true_signed = v.numpy()[sample.numpy()]

        # CMS: query = min across rows
        cms = make_cms(v, d, w)
        cms_est = np.array([cms.query(int(i)) for i in sample])
        cms_biases.append((cms_est - true_abs).mean())
        cms_rmses.append(np.sqrt(((cms_est - true_abs) ** 2).mean()))

        # CS: point query = median{ g_i(j) * C[i, h_i(j)] }
        cs    = make_cs(v, d, w)
        cs_est = cs_point_query(cs, sample)
        cs_biases.append((cs_est - true_signed).mean())
        cs_rmses.append(np.sqrt(((cs_est - true_signed) ** 2).mean()))

    # Theoretical CMS overestimation: E[overest_j] = ||a||_1 / w  (Theorem 1, Cormode)
    theory_cms_bias = float(vecs[0].abs().sum()) / w
    # Theoretical CS RMSE per element: ||a||_2 / sqrt(w) (single-row std dev)
    theory_cs_rmse  = float(vecs[0].norm()) / math.sqrt(w)

    print(f'  {"Metric":<40} {"CMS":>12}  {"CS":>12}')
    print(f'  {"─"*40}  {"─"*12}  {"─"*12}')
    print(f'  {"Mean bias (observed)":40} {np.mean(cms_biases):>+12.4f}  {np.mean(cs_biases):>+12.6f}')
    print(f'  {"RMSE per element (observed)":40} {np.mean(cms_rmses):>12.4f}  {np.mean(cs_rmses):>12.4f}')
    print(f'  {"CMS theory  E[overest] = ||a||₁/w":40} {theory_cms_bias:>12.4f}  {"—":>12}')
    print(f'  {"CS  theory  RMSE ≈ ||a||₂/√w":40} {"—":>12}  {theory_cs_rmse:>12.4f}')
    print()
    print('  CMS is always non-negative biased (overestimates ||a_j|);')
    print('  CS is unbiased (E[estimate] = a_j, signed).')


# ── Test 2: Inner-product estimation ─────────────────────────────────────────

def test_inner_product(vecs, d, w):
    sep()
    print(f'  Test 2: Inner-product estimation  (d={d}, w={w})')
    sep()

    pairs = all_pairs(len(vecs))
    cs_sketches = [make_cs(v, d, w) for v in vecs]

    true_ip, cs_ip = [], []
    for i, j in pairs:
        true_ip.append(float(torch.dot(vecs[i], vecs[j])))
        cs_ip.append(float(cs_sketches[i].inner_product(cs_sketches[j])))

    true_ip = np.array(true_ip)
    cs_ip   = np.array(cs_ip)
    err     = cs_ip - true_ip

    mean_l2     = float(np.mean([v.norm().item() for v in vecs]))
    theory_std  = mean_l2 ** 2 / math.sqrt(d * w)   # single-pair std dev

    tau, p = kendalltau(true_ip, cs_ip)

    print(f'  {"Metric":<45} {"Value":>12}')
    print(f'  {"─"*45}  {"─"*12}')
    print(f'  {"Bias (mean error)":45} {err.mean():>+12.4f}')
    print(f'  {"Std dev of error (observed)":45} {err.std():>12.4f}')
    print(f'  {"Std dev of error (theory  ||a||₂²/√(dw))":45} {theory_std:>12.4f}')
    print(f'  {"Max |error|":45} {np.abs(err).max():>12.4f}')
    print(f'  {"Kendall τ (CS-ip vs true-ip)":45} {tau:>+12.4f}')
    print(f'  {"p-value":45} {p:>12.2e}')
    print()
    print('  Theory (lec_8 §1): E[Z_ℓ] = f_i (unbiased);  Var[Z_ℓ] = ||f||₂²/w.')
    print('  Inner-product estimator is the per-row analogue: unbiased, var ∝ 1/(dw).')


# ── Test 3: L1 pairwise-distance rank preservation ───────────────────────────

def test_l1_rank(vecs, d, w):
    sep()
    print(f'  Test 3: L1 pairwise-distance rank preservation  (d={d}, w={w})')
    sep()

    pairs        = all_pairs(len(vecs))
    cms_sketches = [make_cms(v, d, w) for v in vecs]
    cs_sketches  = [make_cs(v, d, w)  for v in vecs]

    true_l1    = []   # ||a - b||_1  (signed difference)
    true_l1abs = []   # ||a| - |b||_1  (abs-value difference; what CMS approximates)
    true_l2    = []   # ||a - b||_2
    cms_diff   = []
    cs_diff    = []

    for i, j in pairs:
        diff = vecs[i] - vecs[j]
        true_l1.append(float(diff.abs().sum()))
        true_l1abs.append(float((vecs[i].abs() - vecs[j].abs()).abs().sum()))
        true_l2.append(float(diff.norm()))
        cms_diff.append(cms_sketches[i].l1_sketch_diff(cms_sketches[j]))
        cs_diff.append(cs_sketches[i].l1_sketch_diff(cs_sketches[j]))

    true_l1    = np.array(true_l1)
    true_l1abs = np.array(true_l1abs)
    true_l2    = np.array(true_l2)
    cms_diff   = np.array(cms_diff)
    cs_diff    = np.array(cs_diff)

    # CS l1_diff ≈ d * √w * ||a-b||₂  (L1 of bucket sums ≈ √bucket_size * ||diff||₂)
    # so theoretically tracks L2 better than L1
    tau_cs_vs_l1,     p1 = kendalltau(true_l1,    cs_diff)
    tau_cs_vs_l2,     p2 = kendalltau(true_l2,    cs_diff)
    tau_cms_vs_l1abs, p3 = kendalltau(true_l1abs, cms_diff)
    tau_cms_vs_l1,    p4 = kendalltau(true_l1,    cms_diff)

    # Empirical scale: sketch / true
    scale_cs  = np.mean(cs_diff  / (true_l2  + 1e-9))
    scale_cms = np.mean(cms_diff / (true_l1abs + 1e-9))
    theory_cs_scale = D * math.sqrt(w)

    print(f'  {"Comparison":<48} {"τ":>8}  {"p-val":>8}')
    print(f'  {"─"*48}  {"─"*8}  {"─"*8}')
    print(f'  {"CS  l1_diff  vs  ||a−b||₁":48} {tau_cs_vs_l1:>+8.4f}  {p1:>8.2e}')
    print(f'  {"CS  l1_diff  vs  ||a−b||₂  (theory: ≈ d√w·||a-b||₂)":48} {tau_cs_vs_l2:>+8.4f}  {p2:>8.2e}')
    print(f'  {"CMS l1_diff  vs  ||a|−|b||₁":48} {tau_cms_vs_l1abs:>+8.4f}  {p3:>8.2e}')
    print(f'  {"CMS l1_diff  vs  ||a−b||₁":48} {tau_cms_vs_l1:>+8.4f}  {p4:>8.2e}')
    print()
    print(f'  CS  scale observed: {scale_cs:.2f}   theory d√w = {theory_cs_scale:.2f}')
    print(f'  CMS scale observed: {scale_cms:.2f}')
    print()
    print('  Note: CS l1_diff is the L1 norm of the signed sketch difference,')
    print('  which theoretically tracks ||a−b||₂ (not L1).  Both L1 and L2')
    print('  are correlated for Gaussian vectors, so both τ values may be high.')
    print('  CMS stores |a|, so its l1_diff approximates ||a|−|b||_1.')


# ── Test 4: Inner-product similarity rank preservation ───────────────────────

def test_ip_rank(vecs, d, w):
    sep()
    print(f'  Test 4: Inner-product similarity rank preservation  (d={d}, w={w})')
    sep()

    pairs        = all_pairs(len(vecs))
    cs_sketches  = [make_cs(v, d, w)  for v in vecs]
    cms_sketches = [make_cms(v, d, w) for v in vecs]

    true_ip_signed = []   # <a, b>          (what CS estimates)
    true_ip_abs    = []   # <|a|, |b|>      (what CMS estimates)
    cs_ip          = []
    cms_ip         = []

    for i, j in pairs:
        true_ip_signed.append(float(torch.dot(vecs[i], vecs[j])))
        true_ip_abs.append(float(torch.dot(vecs[i].abs(), vecs[j].abs())))
        cs_ip.append(float(cs_sketches[i].inner_product(cs_sketches[j])))
        cms_ip.append(float(cms_sketches[i].inner_product(cms_sketches[j])))

    true_ip_signed = np.array(true_ip_signed)
    true_ip_abs    = np.array(true_ip_abs)
    cs_ip          = np.array(cs_ip)
    cms_ip         = np.array(cms_ip)

    tau_cs,        p1 = kendalltau(true_ip_signed, cs_ip)
    tau_cs_abs,    p2 = kendalltau(true_ip_abs,    cs_ip)
    tau_cms,       p3 = kendalltau(true_ip_abs,    cms_ip)
    tau_cms_signed,p4 = kendalltau(true_ip_signed, cms_ip)

    print(f'  {"Comparison":<48} {"τ":>8}  {"p-val":>8}')
    print(f'  {"─"*48}  {"─"*8}  {"─"*8}')
    print(f'  {"CS  ip  vs  <a, b>  (signed)":48} {tau_cs:>+8.4f}  {p1:>8.2e}')
    print(f'  {"CS  ip  vs  <|a|, |b|>":48} {tau_cs_abs:>+8.4f}  {p2:>8.2e}')
    print(f'  {"CMS ip  vs  <|a|, |b|>":48} {tau_cms:>+8.4f}  {p3:>8.2e}')
    print(f'  {"CMS ip  vs  <a, b>  (signed)":48} {tau_cms_signed:>+8.4f}  {p4:>8.2e}')
    print()
    print('  CS inner product is an unbiased estimator of <a,b> (signed).')
    print('  CMS inner product approximates <|a|,|b|> but with additive')
    print('  error ε||a||₁||b||₁/e, which for dense vectors is very large.')


# ── Test 5: Width sensitivity ─────────────────────────────────────────────────

def test_width_sensitivity(vecs, d):
    sep()
    print(f'  Test 5: Width sensitivity — Kendall τ vs w  (d={d}, n={N:,})')
    sep()

    pairs          = all_pairs(len(vecs))
    true_l1        = np.array([float((vecs[i]-vecs[j]).abs().sum())        for i,j in pairs])
    true_l2        = np.array([float((vecs[i]-vecs[j]).norm())             for i,j in pairs])
    true_ip_signed = np.array([float(torch.dot(vecs[i], vecs[j]))         for i,j in pairs])
    true_ip_abs    = np.array([float(torch.dot(vecs[i].abs(), vecs[j].abs())) for i,j in pairs])

    widths = [32, 64, 128, 256, 512, 1024, 2048, 3072, 6144, N // 2, N]
    print(f'  {"w":>6}  {"n/w":>6}  {"CS τ(l1)":>9}  {"CS τ(l2)":>9}  {"CS τ(ip)":>9}  '
          f'{"CMS τ(l1)":>10}  {"CMS τ(ip)":>10}')
    print(f'  {"─"*6}  {"─"*6}  {"─"*9}  {"─"*9}  {"─"*9}  {"─"*10}  {"─"*10}')

    for w in widths:
        cs_sk  = [make_cs(v, d, w)  for v in vecs]
        cms_sk = [make_cms(v, d, w) for v in vecs]

        cs_l1d  = np.array([cs_sk[i].l1_sketch_diff(cs_sk[j])   for i,j in pairs])
        cms_l1d = np.array([cms_sk[i].l1_sketch_diff(cms_sk[j]) for i,j in pairs])
        cs_ipd  = np.array([float(cs_sk[i].inner_product(cs_sk[j]))   for i,j in pairs])
        cms_ipd = np.array([float(cms_sk[i].inner_product(cms_sk[j])) for i,j in pairs])

        t_cs_l1,  _ = kendalltau(true_l1,        cs_l1d)
        t_cs_l2,  _ = kendalltau(true_l2,        cs_l1d)
        t_cs_ip,  _ = kendalltau(true_ip_signed, cs_ipd)
        t_cms_l1, _ = kendalltau(true_l1,        cms_l1d)
        t_cms_ip, _ = kendalltau(true_ip_abs,    cms_ipd)

        print(f'  {w:>6}  {N/w:>6.1f}  {t_cs_l1:>+9.4f}  {t_cs_l2:>+9.4f}  '
              f'{t_cs_ip:>+9.4f}  {t_cms_l1:>+10.4f}  {t_cms_ip:>+10.4f}')


# ── Helpers ───────────────────────────────────────────────────────────────────

def sep():
    print(f'\n{"─"*70}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(SEED)

    print(f'\n{"═"*70}')
    print(f'  CountSketch vs CountMinSketch — Dense Vector Approximation Tests')
    print(f'  n={N:,}  d={D}  w={W_DEFAULT}  n_tasks={N_TASKS}  → {N_TASKS*(N_TASKS-1)//2} pairs')
    print(f'{"═"*70}')

    t0   = time.time()
    vecs = make_task_vectors(N_TASKS, N, rng)

    norms_l1 = [v.abs().sum().item() for v in vecs]
    norms_l2 = [v.norm().item()      for v in vecs]
    print(f'  Vectors generated  ({time.time()-t0:.2f}s)')
    print(f'  Mean ||v||₁ = {np.mean(norms_l1):,.0f}   '
          f'Mean ||v||₂ = {np.mean(norms_l2):.1f}')
    print(f'  ||v||₁/||v||₂ = {np.mean(norms_l1)/np.mean(norms_l2):.1f}  '
          f'(≈ √n = {math.sqrt(N):.1f} for standard Gaussian — dense)')

    test_point_query(vecs, D, W_DEFAULT)
    test_inner_product(vecs, D, W_DEFAULT)
    test_l1_rank(vecs, D, W_DEFAULT)
    test_ip_rank(vecs, D, W_DEFAULT)
    test_width_sensitivity(vecs, D)

    print(f'\n{"═"*70}')
    print(f'  Total: {time.time()-t0:.1f}s')
    print(f'{"═"*70}\n')
