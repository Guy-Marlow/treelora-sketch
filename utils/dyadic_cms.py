import random
import math
import torch
from copy import deepcopy
import heapq

BIG_MERSENNE = 2**31 - 1

def dyadic_cover(l, r):
    """
    Return a list of (level, scaled_block_index) whose union is [l, r] inclusive.

    A block at (level, idx) covers [idx * 2^level, (idx+1) * 2^level - 1].

    Algorithm: treat [l, r] as half-open [l, r+1) and peel misaligned ends
    one bit at a time — no floating-point arithmetic.
    """
    if r < l:
        return []
    results = []
    level = 0
    hi = r + 1
    while l < hi:
        if l & 1:
            results.append((level, l))
            l += 1
        if hi & 1:
            hi -= 1
            results.append((level, hi))
        l >>= 1
        hi >>= 1
        level += 1
    return results

def generate_hash_function(w):
    a = random.randint(1, BIG_MERSENNE)
    b = random.randint(0, BIG_MERSENNE)
    width = w
    def hash(val):
        return ((a * val + b) % BIG_MERSENNE) % width
    return hash

def generate_sign_function():
    a = random.randint(1, BIG_MERSENNE)
    b = random.randint(0, BIG_MERSENNE)
    def sign(val):
        return (((a * val + b) % BIG_MERSENNE) & 1).float() * 2 - 1
    return sign


class CountSketch:
    """
    CountSketch (Charikar, Chen, Farach-Colton 2002).

    Unlike CountMinSketch, this is a Johnson-Lindenstrauss projection: each
    element is mapped to a bucket and multiplied by a random ±1 sign before
    accumulation.  The inner product estimator is unbiased and its error scales
    as ||a||₂ × ||b||₂ / √(d×w) — roughly √n times smaller than CMS for dense
    vectors, where n is the vector length.

    Handles signed vectors natively; no abs() required before insertion.

    Parameters
    ----------
    d     : number of independent rows (depth)
    w     : number of buckets per row (width)
    device: torch device
    dtype : torch dtype for the sketch matrix
    seed  : RNG seed — all sketches with the same (d, w, seed) share hash/sign
            functions and are therefore comparable under inner_product /
            l1_sketch_diff.
    """

    def __init__(self, d, w, device, dtype, seed=0):
        self.d      = d
        self.w      = w
        self.device = device
        self.dtype  = dtype
        self.seed   = seed

        random.seed(seed)
        self.hash_funcs = [generate_hash_function(w) for _ in range(d)]
        self.sign_funcs = [generate_sign_function()  for _ in range(d)]

        self.cs = torch.zeros((d, w), device=device, dtype=dtype)
        # Cached index tensor — reused across insert_vec calls of the same length
        # to avoid repeated GPU allocations.  Lazily populated on first use.
        self._idx_cache: torch.Tensor | None = None

    def _get_indices(self, n: int) -> torch.Tensor:
        if self._idx_cache is None or len(self._idx_cache) != n:
            self._idx_cache = torch.arange(n, device=self.device, dtype=torch.int64)
        return self._idx_cache

    def insert_vec(self, val_tensor):
        """Insert a 1-D tensor, applying per-element ±1 signs before hashing."""
        if not isinstance(val_tensor, torch.Tensor):
            return
        indices = self._get_indices(len(val_tensor))
        for i in range(self.d):
            hashed_inds = self.hash_funcs[i](indices)
            signs       = self.sign_funcs[i](indices).to(self.dtype)
            self.cs[i].scatter_add_(0, hashed_inds, signs * val_tensor)

    def inner_product(self, other):
        """
        Unbiased estimator of <a, b>: mean of per-row dot products.

        The mean is unbiased (E[row_dot_i] = <a,b>) because the cross-term
        collisions carry opposite signs that cancel in expectation (pairwise
        independence of the sign functions).  Variance per row is
        ||a||₂²||b||₂²/w; averaging d rows reduces it by d.
        Error std dev ≈ ||a||₂ × ||b||₂ / √(d×w).

        Note: Charikar et al. use the median for point-query estimation because
        the median is robust to outlier rows; the mean is standard for inner
        product estimation (AMS-sketch style) and adequate at d=8.
        """
        if not (isinstance(other, CountSketch)
                and self.d == other.d and self.w == other.w):
            print('Sketches not compatible; returning 0.')
            return 0
        # Row-wise dot products (one fused GPU op), then mean across rows.
        # Mean is unbiased for inner products: per-row noise is zero-mean, so
        # averaging d rows gives 1/d variance reduction (Var = ||a||₂²||b||₂²/dw).
        # Median is NOT used here: for inner products the per-row noise is
        # asymmetric (heavy positive tail from same-sign collision products),
        # which makes the median negatively biased.  Median is correct for
        # Charikar's point-query use case where noise IS symmetric.
        return (self.cs * other.cs).sum(dim=1).mean()

    def l1_sketch_diff(self, other):
        """L1 norm of the element-wise sketch difference (same semantics as CMS version)."""
        if not (isinstance(other, CountSketch)
                and self.d == other.d and self.w == other.w):
            print('Sketches not compatible; returning 0.')
            return 0
        return torch.sum(torch.abs(self.cs - other.cs)).item()

class CountMinSketch:
    def __init__(self, d, w, device, dtype, hash_funcs=None, seed=0):
        self.d = d
        self.w = w
        self.device = device
        self.dtype = dtype
        self.seed = seed

        if hash_funcs is None:
            random.seed(self.seed)
            self.hash_funcs = [generate_hash_function(w) for _ in range(d)]
        else:
            self.hash_funcs = hash_funcs

        self.cms = torch.zeros((d, w), device=device, dtype=dtype)

    def insert_vec(self, val_tensor, indices=None):
        if isinstance(val_tensor, torch.Tensor):
            if indices is None:
                indices = torch.arange(len(val_tensor), device=self.device, dtype=torch.int64)
            for i in range(self.d):
                hashed_inds = self.hash_funcs[i](indices)
                self.cms[i].scatter_add_(0, hashed_inds, val_tensor)

    def insert(self, val, ind):
        for i in range(self.d):
            hashed_ind = self.hash_funcs[i](ind)
            self.cms[i][hashed_ind] += val

    def query(self, index):
        candidates = []
        for i in range(self.d):
            candidates.append(self.cms[i][self.hash_funcs[i](index)])
        return min(candidates)
    
    def inner_product(self, other):
        if isinstance(other, CountMinSketch) and self.d == other.d and self.w == other.w:
            candidates = []
            for i in range(self.d):
                candidates.append(torch.sum(self.cms[i]*other.cms[i]))
            return min(candidates)
        else:
            print('Sketches not compatible; returning 0.')
            return 0

    def l1_sketch_diff(self, other):
        """
        L1 norm of the element-wise difference of the two count matrices:
            sum_{j,k} |cms_a[j,k] - cms_b[j,k]|

        This is the L1 distance between the sketch representations, not an
        estimate of ||a - b||_1 for the underlying vectors.  It is well-defined
        because subtraction is legal under the turnstile model (CMS is a linear
        sketch): (S_a - S_b)[j,k] = sum_{i: h_j(i)=k} (a_i - b_i).

        Useful property: for any single row j, sum_k count[j,k] = ||a||_1 exactly
        (each element hashes to exactly one bucket per row), so the per-row absolute
        sum is bounded in [| ||a||_1 - ||b||_1 |,  ||a||_1 + ||b||_1].
        """
        if isinstance(other, CountMinSketch) and self.d == other.d and self.w == other.w:
            return torch.sum(torch.abs(self.cms - other.cms)).item()
        else:
            print('Sketches not compatible; returning 0.')
            return 0

class DyadicRangeCMS:
    def __init__(self, w_frac: float, num_indices: int, device, dtype,
                 phi: float = 0.001, delta: float = 0.075):
        """
        w_frac:      fraction of num_indices to use as sketch width
        num_indices: size of the universe (number of parameters)
        phi:         heavy hitter threshold fraction (used to derive d)
        delta:       desired overall false-positive failure probability (used to derive d)

        d is derived from the CM paper section 5.2 formula for the heavy hitter
        guarantee. Across all queries made during a heavy hitter search, the total
        probability of any false positive is <= delta:

            d = ceil( log( 2 * log2(n) / (delta * phi) ) )
        """
        self.w = math.ceil(w_frac * num_indices)
        self.phi = phi
        self.delta = delta
        self.device = device
        self.dtype = dtype

        self.num_indices = num_indices
        self.num_sketches = math.ceil(math.log2(num_indices))
        self.N = 2 ** self.num_sketches

        self.d = math.ceil(math.log2(2 * self.num_sketches / (delta * phi)))

        self.sketches: list[CountMinSketch | torch.Tensor] = []
        for i in range(self.num_sketches + 1):
            num_ranges = 2 ** (self.num_sketches - i)
            if num_ranges > (self.w*self.d):
                self.sketches.append(CountMinSketch(
                    self.d, self.w,
                    device=self.device, dtype=self.dtype, seed=i))
            else:
                self.sketches.append(
                    torch.zeros(num_ranges, device=self.device, dtype=self.dtype))

    def insert(self, vec):
        pad = self.N - self.num_indices
        vec_padded = torch.cat([vec, vec.new_zeros(pad)]) if pad > 0 else vec

        for i in range(self.num_sketches + 1):
            k = 1 << i
            num_blocks = self.N >> i
            block_sums = vec_padded.reshape(num_blocks, k).sum(dim=1)

            if isinstance(self.sketches[i], CountMinSketch):
                block_inds = torch.arange(num_blocks, device=self.device, dtype=torch.int64)
                self.sketches[i].insert_vec(block_sums, block_inds)
            else:
                self.sketches[i] += block_sums

    def range_sum_query(self, l, r):
        total = 0
        for level, block_idx in dyadic_cover(l, r):
            structure = self.sketches[level]
            if isinstance(structure, CountMinSketch):
                total += structure.query(block_idx)
            else:
                total += structure[block_idx]
        return total

    def _subtract_dcms(self, other):
        """In-place subtraction: self -= other."""
        if self.num_sketches == other.num_sketches and self.w == other.w:
            for i in range(self.num_sketches + 1):
                if isinstance(self.sketches[i], CountMinSketch):
                    self.sketches[i].cms -= other.sketches[i].cms
                else:
                    self.sketches[i] -= other.sketches[i]

    def subtract_dcms(self, other):
        """Out-of-place subtraction: returns new sketch = self - other."""
        if self.num_sketches == other.num_sketches and self.w == other.w:
            out = deepcopy(self)
            for i in range(out.num_sketches + 1):
                if isinstance(out.sketches[i], CountMinSketch):
                    out.sketches[i].cms -= other.sketches[i].cms
                else:
                    out.sketches[i] -= other.sketches[i]
            return out

    def __str__(self):
        total_size = 0
        out = (f'DyadicRangeCMS: {self.num_indices} indices, w={self.w} d={self.d} N={self.N}\n'
               f'  phi={self.phi}, delta={self.delta} -> d={self.d}\n'
               f'  Device: {self.device}\n'
               f'  Levels ({self.num_sketches + 1} total):')
        for i in range(len(self.sketches)):
            num_ranges = 2 ** (self.num_sketches - i)
            block_size = 2 ** i
            if isinstance(self.sketches[i], CountMinSketch):
                out += f'\n    [{i}] CMS:    {num_ranges:>8} ranges of size {block_size}'
                total_size += self.w * self.d
            else:
                out += f'\n    [{i}] Tensor: {num_ranges:>8} ranges of size {block_size}'
                total_size += num_ranges
        out += f'\n  Total entries: {total_size}, {total_size * self.dtype.itemsize} bytes'
        return out

    def _query_block(self, level, block_idx):
        """Direct O(1) lookup of a single dyadic block by level and block index."""
        structure = self.sketches[level]
        if isinstance(structure, CountMinSketch):
            return structure.query(block_idx)
        else:
            return structure[block_idx]

    def _batch_query_level(self, level, block_indices):
        """
        Query all block_indices at a given level simultaneously.
        For exact tensor levels this is a single indexed tensor read.
        For CMS levels this is d hash evaluations over the full candidate tensor,
        taking the elementwise minimum across rows.
        """
        structure = self.sketches[level]
        if isinstance(structure, torch.Tensor):
            return structure[block_indices]
        else:
            estimates = torch.full((len(block_indices),), float('inf'),
                                device=self.device, dtype=self.dtype)
            for row in range(structure.d):
                hashed = structure.hash_funcs[row](block_indices)
                row_vals = structure.cms[row][hashed]
                estimates = torch.minimum(estimates, row_vals)
            return estimates

    def ranges_to_indices(self, ranges):
        indices = []
        for (l, r, _) in ranges:
            indices.extend(range(l, r + 1))
        return indices
    
    def top_k_hh(self, k, min_block_size=1):

        min_level = math.ceil(math.log2(max(min_block_size, 1)))
        root_sum = self.sketches[self.num_sketches][0].item() \
                if isinstance(self.sketches[self.num_sketches], torch.Tensor) \
                else self._query_block(self.num_sketches, 0)

        heap = [(-root_sum, 0, self.num_sketches)]
        results = []

        while heap and len(results) < k:
            mass, block_idx, level = heapq.heappop(heap)

            block_mass = -mass

            if level <= min_level:
                # At target granularity, collect as result
                block_size = 2 ** level
                l = block_idx * block_size
                r = min(l + block_size - 1, self.num_indices - 1)
                if l < self.num_indices:
                    results.append((l, r, block_mass))
            else:
                # Split into two children and batch-query both at once
                child_level = level - 1
                child_indices = torch.tensor(
                    [block_idx * 2, block_idx * 2 + 1],
                    device=self.device, dtype=torch.int64)
                estimates = self._batch_query_level(child_level, child_indices)
                left_est, right_est = estimates[0].item(), estimates[1].item()

                heapq.heappush(heap, (-left_est,  block_idx * 2,     child_level))
                heapq.heappush(heap, (-right_est, block_idx * 2 + 1, child_level))

        return results

    def bottom_k_hh(self, k, min_block_size=1):

        min_level = math.ceil(math.log2(max(min_block_size, 1)))
        root_sum = self.sketches[self.num_sketches][0].item() \
                if isinstance(self.sketches[self.num_sketches], torch.Tensor) \
                else self._query_block(self.num_sketches, 0)

        heap = [(root_sum, 0, self.num_sketches)]
        results = []

        while heap and len(results) < k:
            mass, block_idx, level = heapq.heappop(heap)
            block_mass = mass

            if level <= min_level:
                # At target granularity, collect as result
                block_size = 2 ** level
                l = block_idx * block_size
                r = min(l + block_size - 1, self.num_indices - 1)
                if l < self.num_indices:
                    results.append((l, r, block_mass))
            else:
                # Split into two children and batch-query both at once
                child_level = level - 1
                child_indices = torch.tensor(
                    [block_idx * 2, block_idx * 2 + 1],
                    device=self.device, dtype=torch.int64)
                estimates = self._batch_query_level(child_level, child_indices)
                left_est, right_est = estimates[0].item(), estimates[1].item()

                heapq.heappush(heap, (left_est,  block_idx * 2,     child_level))
                heapq.heappush(heap, (right_est, block_idx * 2 + 1, child_level))

        return results
    
    def cosine_sim(self, other):
        if isinstance(other, DyadicRangeCMS) and self.d == other.d:
            minimum = None
            sketch = self.sketches[0]
            other_sketch = other.sketches[0]

            for i in range(self.d):
                inner_prod = torch.sum(sketch.cms[i]*other_sketch.cms[i])
                if(minimum is None or inner_prod < minimum[0]):
                    minimum = (inner_prod, i)
            
            if(minimum is None):
                return 0

            denominator = torch.linalg.norm(sketch.cms[minimum[1]])*torch.linalg.norm(other_sketch.cms[minimum[1]])
            return minimum[0]/denominator
        else:
            print('Sketches not compatible; returning 0.')
            return 0

    def l1_norm(self):
        total = 0
        for i in range(self.num_sketches + 1):
            structure = self.sketches[i]
            if isinstance(structure, CountMinSketch):
                total += torch.sum(structure.cms)
            else:
                total += torch.sum(structure)
        return total