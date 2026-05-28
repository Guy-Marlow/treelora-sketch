"""
Tests for dyadic_cover: old float-based algorithm vs new bit-manipulation algorithm.

Format note
-----------
Old algorithm returns (level, actual_start_address), where actual_start is
always a multiple of 2^level and the covered interval is
  [actual_start,  actual_start + 2^level - 1]  (both endpoints INCLUSIVE).

New algorithm returns (level, scaled_block_index), where
  scaled_block_index = actual_start >> level
and the covered interval is
  [scaled_idx * 2^level,  (scaled_idx + 1) * 2^level - 1]  (both INCLUSIVE).

dyadic_cover(l, r) accepts INCLUSIVE [l, r] and must cover exactly that set.
"""

import math
import random
import unittest


# ── Reference (old) implementation — copied verbatim from dyadic_cms/utils.py ─
def dyadic_cover_old(l, r):
    if r - l < 0:
        return None          # bug: should be [] but we preserve original behaviour
    elif r - l == 0:
        return [(0, l)]

    k = math.floor(math.log2((r - l) + 1))
    ranges = []
    cur_w = 2 ** k
    l_cur = math.ceil(l / cur_w) * cur_w
    r_cur = math.floor((r + 1) / cur_w) * cur_w

    if l_cur != r_cur:
        ranges.append((k, l_cur))

    cur_k = k - 1
    while (l_cur >= l or r_cur <= r) and cur_k >= 0:
        cur_w = 2 ** cur_k
        if l_cur - cur_w >= l:
            ranges.append((cur_k, l_cur - cur_w))
            l_cur -= cur_w
        if r_cur + cur_w - 1 <= r:
            ranges.append((cur_k, r_cur))
            r_cur += cur_w
        cur_k -= 1

    return ranges


# ── New implementation (bit-manipulation) ────────────────────────────────────
def dyadic_cover_new(l, r):
    """
    Returns list of (level, scaled_block_index) whose union is [l, r] inclusive.

    Treats the range as half-open [l, r+1) and peels misaligned ends one bit
    at a time.  All arithmetic is integer / bitwise — no floats.

    Endpoint semantics: both l and r are INCLUSIVE.
    """
    if r < l:
        return []
    results = []
    level = 0
    hi = r + 1          # convert inclusive right end to exclusive
    while l < hi:
        if l & 1:           # l not aligned — peel off singleton at current level
            results.append((level, l))
            l += 1
        if hi & 1:          # hi not aligned — peel off singleton just below hi
            hi -= 1
            results.append((level, hi))
        l >>= 1
        hi >>= 1
        level += 1
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────
def old_to_scaled_set(covers):
    """Convert old (level, actual_start) list → frozenset of (level, scaled_idx)."""
    return frozenset((level, start >> level) for level, start in covers)


def expand_new(covers):
    """Expand new-format (level, scaled_idx) list → frozenset of individual indices."""
    indices = set()
    for level, scaled_idx in covers:
        start = scaled_idx << level
        for j in range(1 << level):
            indices.add(start + j)
    return frozenset(indices)


def expand_old(covers):
    """Expand old-format (level, actual_start) list → frozenset of individual indices."""
    indices = set()
    for level, start in covers:
        for j in range(1 << level):
            indices.add(start + j)
    return frozenset(indices)


# ── Correctness tests for the new algorithm ──────────────────────────────────
class TestNewAlgorithmCorrectness(unittest.TestCase):

    def _check(self, l, r):
        covers = dyadic_cover_new(l, r)

        if r < l:
            self.assertEqual(covers, [],
                             f"Empty range [{l},{r}] must return []")
            return

        expected = frozenset(range(l, r + 1))
        got = expand_new(covers)

        # Correct coverage
        self.assertEqual(got, expected,
                         f"[{l},{r}]: got indices {sorted(got)}, expected {sorted(expected)}")

        # No overlaps: sum of block sizes == r - l + 1
        total = sum(1 << lv for lv, _ in covers)
        self.assertEqual(total, r - l + 1,
                         f"[{l},{r}]: overlapping intervals (total sizes={total}, range={r-l+1})")

    def test_empty_range(self):
        self.assertEqual(dyadic_cover_new(5, 3), [])
        self.assertEqual(dyadic_cover_new(0, -1), [])
        self.assertEqual(dyadic_cover_new(10, 9), [])

    def test_singleton(self):
        for i in [0, 1, 3, 7, 8, 15, 16, 63, 64, 100, 1023, 1024]:
            self._check(i, i)

    def test_aligned_blocks(self):
        """Ranges that are already a single aligned dyadic block."""
        for exp in range(0, 10):
            size = 1 << exp
            for start in [0, size, 2 * size, 4 * size]:
                self._check(start, start + size - 1)

    def test_small_exhaustive(self):
        """Check every (l, r) pair with l, r in [0, 63]."""
        for l in range(64):
            for r in range(l, 64):
                self._check(l, r)

    def test_random_medium(self):
        random.seed(42)
        for _ in range(2000):
            l = random.randint(0, 100_000)
            r = random.randint(l, l + random.randint(0, 100_000))
            self._check(l, r)

    def test_inclusive_right_endpoint(self):
        """r must appear in the cover — catches off-by-one in hi = r + 1."""
        for r in [1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256]:
            covers = dyadic_cover_new(0, r)
            got = expand_new(covers)
            self.assertIn(r, got,
                          f"Right endpoint {r} absent from dyadic_cover_new(0, {r})")

    def test_inclusive_left_endpoint(self):
        """l must appear in the cover."""
        for l in [1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33]:
            covers = dyadic_cover_new(l, l + 15)
            got = expand_new(covers)
            self.assertIn(l, got,
                          f"Left endpoint {l} absent from dyadic_cover_new({l}, {l+15})")

    def test_both_endpoints_singleton(self):
        """Single-element range [x, x] must return exactly [(0, x)]."""
        for x in [0, 1, 4, 7, 8, 100]:
            covers = dyadic_cover_new(x, x)
            self.assertEqual(len(covers), 1)
            self.assertEqual(covers[0], (0, x),
                             f"Singleton [{x},{x}] returned {covers}")


# ── Equivalence: new must produce the same intervals as old ──────────────────
class TestEquivalenceWithOld(unittest.TestCase):

    def _compare(self, l, r):
        old = dyadic_cover_old(l, r)
        new = dyadic_cover_new(l, r)

        # Normalize: both to frozenset of (level, scaled_idx)
        if old is None:
            # old returns None for empty ranges (its own bug); new returns []
            self.assertEqual(new, [],
                             f"[{l},{r}]: old→None but new→{new}")
            return

        old_norm = old_to_scaled_set(old)
        new_norm = frozenset(new)
        self.assertEqual(old_norm, new_norm,
                         f"[{l},{r}]:\n  old={sorted(old_norm)}\n  new={sorted(new_norm)}")

    def test_small_exhaustive(self):
        for l in range(128):
            for r in range(l, 128):
                self._compare(l, r)

    def test_random(self):
        random.seed(99)
        for _ in range(3000):
            l = random.randint(0, 1_000_000)
            r = random.randint(l, l + random.randint(0, 1_000_000))
            self._compare(l, r)


if __name__ == '__main__':
    unittest.main(verbosity=2)
