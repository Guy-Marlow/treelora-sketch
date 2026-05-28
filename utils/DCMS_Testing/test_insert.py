"""
Tests for the new insert approach: replace scatter_add_(scaled_range) with
pad → reshape → sum for grouping elements into dyadic blocks.

Old approach (exact tensor level):
    inds = arange(n) // k            # length-n index tensor
    target.scatter_add_(0, inds, vec)

New approach (exact tensor level):
    pad vec to length N (= 2^ceil(log2(n)), divisible by k)
    target = padded.reshape(N//k, k).sum(dim=1)

Old approach (CMS level):
    inds = arange(n) // k
    for each row: cms[row].scatter_add_(0, hash(inds), vec)

New approach (CMS level):
    block_sums = padded.reshape(num_blocks, k).sum(dim=1)
    block_inds = arange(num_blocks)
    for each row: cms[row].scatter_add_(0, hash(block_inds), block_sums)

Mathematical equivalence: for each cms bucket b,
    sum_{j: hash(j//k)==b} vec[j]  ==  sum_{p: hash(p)==b} block_sums[p]
because block_sums[p] = sum_{j: j//k == p} vec[j].

Additionally tests the "same-block same-bucket" invariant:
    elements whose index maps to the same dyadic block must always hash
    to the same CMS bucket.
"""

import math
import random
import time
import unittest

import torch


# ── Reference helpers ─────────────────────────────────────────────────────────
BIG_MERSENNE = 2 ** 31 - 1


def scaled_range_old(n, k, device='cpu'):
    """Original: float division then cast — maps each of n indices to its block."""
    return torch.floor(torch.arange(n, device=device) / k).type(torch.int64)


def make_hash(w, a=None, b=None, seed=None):
    """Pairwise-independent hash (ax+b) mod p mod w, matching dyadic_cms/utils.py."""
    if seed is not None:
        random.seed(seed)
        a = random.randint(1, BIG_MERSENNE)
        b = random.randint(0, BIG_MERSENNE)
    width = w
    def h(val):
        return ((a * val + b) % BIG_MERSENNE) % width
    return h


def N_for(n):
    """Smallest power of 2 >= n."""
    return 1 << math.ceil(math.log2(max(n, 1)))


# ── New block-sum helper ──────────────────────────────────────────────────────
def block_sums_new(vec, k, N):
    """
    Aggregate vec (length n) into N//k blocks of size k.
    Pads vec to length N with zeros, then reshape + sum.
    N must be >= len(vec) and divisible by k.
    """
    n = len(vec)
    pad = N - n
    padded = torch.cat([vec, vec.new_zeros(pad)]) if pad > 0 else vec
    num_blocks = N // k
    return padded.reshape(num_blocks, k).sum(dim=1)


# ── Tests: exact tensor level ─────────────────────────────────────────────────
class TestExactTensorEquivalence(unittest.TestCase):
    """
    scatter_add_(0, scaled_range(n,k), vec)
    must equal
    block_sums_new(vec, k, N)
    for all valid (n, k).
    """

    def _check(self, n, k, device='cpu', dtype=torch.float32):
        N = N_for(n)
        # ensure N is divisible by k (true when k is a power of 2 and N is a power of 2)
        while N % k != 0:
            N *= 2
        num_blocks = N // k

        torch.manual_seed(7)
        vec = torch.randn(n, device=device, dtype=dtype)

        # Old
        inds = scaled_range_old(n, k, device)
        target_old = torch.zeros(num_blocks, device=device, dtype=dtype)
        target_old.scatter_add_(0, inds, vec)

        # New
        target_new = block_sums_new(vec, k, N)

        self.assertEqual(target_old.shape, target_new.shape,
                         f"Shape mismatch n={n} k={k}: old={target_old.shape} new={target_new.shape}")
        self.assertTrue(
            torch.allclose(target_old, target_new, atol=1e-5),
            f"Value mismatch n={n} k={k}\nold={target_old}\nnew={target_new}"
        )

    def test_aligned_n(self):
        for exp_n in range(0, 9):
            for exp_k in range(0, exp_n + 1):
                self._check(1 << exp_n, 1 << exp_k)

    def test_unaligned_n(self):
        """n is not a multiple of k — padding must handle the partial last block."""
        for n in [1, 3, 5, 7, 9, 100, 127, 200, 255, 500, 1000]:
            for k in [1, 2, 4, 8, 16, 32]:
                self._check(n, k)

    def test_k_equals_1(self):
        """Block size 1: each element is its own block."""
        self._check(256, 1)
        self._check(100, 1)

    def test_k_equals_n(self):
        """Block size = n: all elements collapse into one block."""
        for n in [1, 4, 8, 16, 64]:
            self._check(n, n)

    def test_single_element(self):
        self._check(1, 1)

    def test_large(self):
        self._check(100_000, 64)
        self._check(100_000, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        self._check(50_000, 64, device='cuda')


# ── Tests: CMS level ─────────────────────────────────────────────────────────
class TestCMSInsertEquivalence(unittest.TestCase):
    """
    Old per-element CMS insert must produce the same counter array as
    new block-aggregated CMS insert.
    """

    def _check(self, n, k, d=4, w=64, device='cpu'):
        N = N_for(n)
        while N % k != 0:
            N *= 2
        num_blocks = N // k

        hash_funcs = [make_hash(w, seed=i) for i in range(d)]

        torch.manual_seed(3)
        vec = torch.abs(torch.randn(n, device=device, dtype=torch.float32))

        # Old: scatter per-element values using scaled block indices
        cms_old = torch.zeros((d, w), device=device, dtype=torch.float32)
        inds = scaled_range_old(n, k, device)
        for row in range(d):
            hashed = hash_funcs[row](inds)
            cms_old[row].scatter_add_(0, hashed, vec)

        # New: aggregate blocks first, scatter block sums
        cms_new = torch.zeros((d, w), device=device, dtype=torch.float32)
        sums = block_sums_new(vec, k, N)
        block_inds = torch.arange(num_blocks, device=device, dtype=torch.int64)
        for row in range(d):
            hashed = hash_funcs[row](block_inds)
            cms_new[row].scatter_add_(0, hashed, sums)

        self.assertTrue(
            torch.allclose(cms_old, cms_new, atol=1e-4),
            f"CMS mismatch n={n} k={k} d={d} w={w}"
        )

    def test_various(self):
        for n in [32, 63, 64, 127, 256, 500, 1000]:
            for k in [1, 2, 4, 8]:
                self._check(n, k)

    def test_k_equals_1(self):
        """Level 0: each element is its own block — direct CMS insert."""
        self._check(256, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        self._check(10_000, 32, device='cuda')


# ── Tests: same-block same-bucket invariant ───────────────────────────────────
class TestSameBlockSameBucket(unittest.TestCase):
    """
    Every element whose index falls in the same dyadic block (same i // k)
    must hash to the same CMS bucket in every row.

    This is the core correctness property that makes per-element indexing
    and block-aggregated indexing produce identical CMS counters.
    """

    def _check(self, n, k, d, w):
        hash_funcs = [make_hash(w, seed=row) for row in range(d)]
        block_indices = scaled_range_old(n, k)  # shape [n], values in [0, n//k)

        for row in range(d):
            hashed_per_element = hash_funcs[row](block_indices)  # shape [n]

            num_blocks = int(block_indices.max().item()) + 1
            for block in range(num_blocks):
                mask = (block_indices == block)
                buckets = hashed_per_element[mask]
                if len(buckets) == 0:
                    continue
                # All elements in this block must land in the same bucket
                self.assertTrue(
                    (buckets == buckets[0]).all(),
                    f"n={n} k={k} row={row} block={block}: "
                    f"elements map to different buckets {buckets.tolist()}"
                )

    def test_small(self):
        for k in [1, 2, 4, 8]:
            self._check(n=64, k=k, d=3, w=16)

    def test_unaligned(self):
        for n in [30, 60, 100]:
            for k in [4, 8]:
                self._check(n=n, k=k, d=2, w=32)

    def test_k_equals_1(self):
        """Every element is its own block — trivially satisfied."""
        self._check(n=128, k=1, d=3, w=32)


# ── Timing comparison ─────────────────────────────────────────────────────────
class TestInsertTiming(unittest.TestCase):
    """
    Empirical speed comparison: old scatter_add(scaled_range) vs new reshape+sum.
    The test always passes; it prints timing for informational purposes.
    """

    def _bench(self, n, k, reps=300, device='cpu'):
        N = N_for(n)
        while N % k != 0:
            N *= 2
        num_blocks = N // k

        torch.manual_seed(0)
        vec = torch.randn(n, device=device, dtype=torch.float32)
        if device != 'cpu':
            torch.cuda.synchronize()

        def sync():
            if device != 'cpu':
                torch.cuda.synchronize()

        # Old
        t0 = time.perf_counter()
        for _ in range(reps):
            inds = scaled_range_old(n, k, device)
            tgt = torch.zeros(num_blocks, device=device, dtype=torch.float32)
            tgt.scatter_add_(0, inds, vec)
        sync()
        t_old = (time.perf_counter() - t0) / reps * 1e3

        # New
        t0 = time.perf_counter()
        for _ in range(reps):
            tgt = block_sums_new(vec, k, N)
        sync()
        t_new = (time.perf_counter() - t0) / reps * 1e3

        print(f"\n  [Timing] n={n:>8,}  k={k:>5}  device={device}  "
              f"old={t_old:.4f}ms  new={t_new:.4f}ms  speedup={t_old/t_new:.2f}x")

        return t_old, t_new

    def test_timing_cpu_small_k(self):
        self._bench(500_000, 1)

    def test_timing_cpu_medium_k(self):
        self._bench(500_000, 64)

    def test_timing_cpu_large_k(self):
        self._bench(500_000, 1024)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_timing_cuda(self):
        self._bench(1_000_000, 64, device='cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)
