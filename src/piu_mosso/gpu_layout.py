"""
The UMAP layout on a GPU: umap-learn's Euclidean layout with move_other, as one CUDA kernel (CuPy), repeatable.

piu_mosso: fast single-cell utilities
Copyright (C) 2026 Ning Wang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

# Kernel launches per epoch. Each launch covers a sixteenth of the edges (in a seeded random order) and computes
# every step from the positions at its start, so updates are synchronous within a launch and sequential between
# launches. With one launch per epoch a cell with many edges sums many clipped steps at once and the layout
# overshoots; 16 launches keep the layout's quality while costing little time.
CHUNKS = 16

# A cell's summed move per launch is clipped to this many times the learning rate per coordinate, umap-learn's clip of
# one step. Sequential updates would see a cell move after each step; summed at once, a hub cell's thousands of
# clipped steps would fling it far out of the layout. Cells with few edges per launch are rarely affected.
MAX_MOVE = 4.0

# Steps are summed in 64-bit fixed point with this resolution (2^-32), so the sums, and the layout, do not depend on
# the order in which threads add them. One step is at most 4 alpha per coordinate (alpha the learning rate, 1 by
# default), so a cell's sum within a launch stays far below 2^63 / 2^32.
FIXED_POINT_SCALE = 2.0**32

# The kernel keeps a cell's coordinates in registers (its `float current[8]`); scanpy's layouts have 2 dimensions
MAX_COMPONENTS = 8

# Device memory: per edge two int32 cell indices and three float64 schedule values (epochs per sample, next sample,
# next negative sample); per coordinate a float32 position and an int64 fixed-point sum; plus a margin for CuPy
BYTES_PER_EDGE = 2 * 4 + 3 * 8
BYTES_PER_COORDINATE = 4 + 8
MEMORY_MARGIN = 256 * 2**20

THREADS_PER_BLOCK = 256

_SOURCE = r"""
// umap-learn's clip of each coordinate's step
__device__ __forceinline__ float clip4(float v) { return fminf(fmaxf(v, -4.0f), 4.0f); }

// splitmix64's finaliser: a well-mixed 64-bit hash of a counter
__device__ __forceinline__ unsigned long long mix(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Add a step to a coordinate's fixed-point sum (integer adds give the same sum in any order)
__device__ __forceinline__ void add(long long* sums, long long index, float step, double scale) {
    atomicAdd((unsigned long long*)(sums + index), (unsigned long long)llrint((double)step * scale));
}

// One launch: every due edge of [first, end) as umap-learn's _optimize_layout_euclidean_single_epoch, reading
// positions, writing steps to sums
extern "C" __global__ void layout_edges(
    const float* positions, long long* sums, const int* head, const int* tail, const double* epochs_per_sample,
    double* next_sample, double* next_negative, long long first, long long end, int n_cells, int dim, float a,
    float b, float gamma, float alpha, int epoch, double negative_sample_rate, unsigned long long seed, double scale)
{
    long long i = first + (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end || next_sample[i] > (double)epoch) return;
    long long j = head[i], k = tail[i];
    float current[8];  // MAX_COMPONENTS
    float d2 = 0.0f;
    for (int d = 0; d < dim; d++) {
        current[d] = positions[j * dim + d];
        float diff = current[d] - positions[k * dim + d];
        d2 += diff * diff;
    }

    // Attraction: the head moves by the step and the tail by its negative (move_other)
    float coeff = 0.0f;
    if (d2 > 0.0f) coeff = -2.0f * a * b * powf(d2, b - 1.0f) / (a * powf(d2, b) + 1.0f);
    for (int d = 0; d < dim; d++) {
        float step = clip4(coeff * (current[d] - positions[k * dim + d])) * alpha;
        add(sums, j * dim + d, step, scale);
        add(sums, k * dim + d, -step, scale);
        current[d] += step;
    }
    next_sample[i] += epochs_per_sample[i];
    if (negative_sample_rate <= 0.0) return;

    // Negative samples owed since the last ones (truncated toward zero, as umap-learn's int()), from the head's
    // position after its attraction step; tails drawn uniformly over all cells by a hash of (seed, epoch, edge, p)
    double epochs_per_negative = epochs_per_sample[i] / negative_sample_rate;
    int n_negative = (int)(((double)epoch - next_negative[i]) / epochs_per_negative);
    unsigned long long base = mix(seed ^ mix(((unsigned long long)epoch << 40) ^ (unsigned long long)i));
    for (int p = 0; p < n_negative; p++) {
        long long t = (long long)(mix(base + (unsigned long long)p) % (unsigned long long)n_cells);
        float nd2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = current[d] - positions[t * dim + d];
            nd2 += diff * diff;
        }
        // The head itself (umap-learn skips it), or another cell at the same place (a zero gradient): no step
        if (t == j || nd2 <= 0.0f) continue;
        float repulsion = 2.0f * gamma * b / ((0.001f + nd2) * (a * powf(nd2, b) + 1.0f));
        for (int d = 0; d < dim; d++) {
            float step = clip4(repulsion * (current[d] - positions[t * dim + d])) * alpha;
            add(sums, j * dim + d, step, scale);
            current[d] += step;
        }
    }
    next_negative[i] += n_negative * epochs_per_negative;
}

// After a launch: add the fixed-point sums to the positions, each coordinate's sum clipped to max_step, and clear them
extern "C" __global__ void apply_sums(float* positions, long long* sums, long long n_values, double scale,
                                      double max_step) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_values) return;
    positions[i] += (float)fmin(fmax((double)sums[i] / scale, -max_step), max_step);
    sums[i] = 0;
}
"""

_module = None


def check_layout_arguments(n_cells, n_components, negative_sample_rate):
    """
    Raise ValueError for a layout the GPU kernel cannot run: 2^31 cells or more, more than 8 dimensions, or a
    negative negative_sample_rate.
    """
    if n_cells >= 2**31:
        raise ValueError(f"The GPU layout takes fewer than 2^31 cells, but the graph has {n_cells}: use use_gpu=False")
    if n_components > MAX_COMPONENTS:
        raise ValueError(
            f"The GPU layout takes up to {MAX_COMPONENTS} dimensions, but n_components is {n_components}: use "
            f"use_gpu=False"
        )
    if negative_sample_rate < 0:
        raise ValueError(f"negative_sample_rate must be 0 or more on the GPU, but is {negative_sample_rate}")


def compile_kernel():
    """Compile the CUDA kernels (once per process), so a CuPy without a working CUDA compiler fails before any work."""
    _kernels()


def check_memory(n_cells, n_edges, n_components):
    """Raise RuntimeError if the layout's device memory (estimated from the sizes) exceeds the GPU's free memory."""
    import cupy

    needed = BYTES_PER_EDGE * n_edges + BYTES_PER_COORDINATE * n_cells * n_components + MEMORY_MARGIN
    # Free device memory plus what CuPy's pool holds but does not use
    free = cupy.cuda.runtime.memGetInfo()[0] + cupy.get_default_memory_pool().free_bytes()
    if needed > free:
        raise RuntimeError(
            f"The GPU layout of {n_cells} cells and {n_edges} edges needs about {needed / 2**30:.1f} GiB of GPU "
            f"memory, but {free / 2**30:.1f} GiB are free: use use_gpu=False"
        )


def optimize_layout_gpu(
    embedding, head, tail, n_epochs, epochs_per_sample, a, b, rng_state, gamma=1.0, initial_alpha=1.0,
    negative_sample_rate=5.0,
):
    """
    The layout umap-learn's optimize_layout_euclidean(move_other=True) optimizes, computed on the GPU; a float32 numpy
    array of embedding's shape (embedding itself, the start, is not changed).

    Per edge, the steps of umap-learn's _optimize_layout_euclidean_single_epoch: attraction moving both cells, then
    the negative samples owed (umap-learn's schedule, float64 counters, counts truncated toward zero), with the same
    gradients and the per-coordinate clip at 4 before the learning rate, which falls linearly from initial_alpha as
    in umap-learn. negative_sample_rate=0 means no negative samples. It differs from umap-learn in three ways, so its
    results never equal umap-learn's or the CPU path's:
    - updates are synchronous within each of CHUNKS launches per epoch (all steps of a launch use the positions at
      its start), and sequential between launches, over the edges in a seeded random order; a cell's summed move per
      launch is clipped to MAX_MOVE times the learning rate per coordinate (the bound of one step in umap-learn);
    - negative-sample cells come from a hash of (seed, epoch, edge, sample), not umap-learn's per-cell generator;
    - steps are summed in fixed point (resolution 2^-32), so the result does not depend on thread order: the same
      inputs and rng_state give the same layout on the same GPU model with the same CUDA and CuPy versions.
    A negative gamma is passed through (umap-learn's layout gives such repulsion a zero step instead).

    rng_state: umap-learn's three int64 layout seeds. The hash seed is s0 + 2^32 s1 (each modulo 2^32), and the edge
    order is a permutation by numpy's default_rng seeded with the three seeds modulo 2^32.
    """
    n_cells, dim = embedding.shape
    n_edges = len(head)
    check_layout_arguments(n_cells, dim, negative_sample_rate)
    # A mismatch would make the kernel read out of bounds (a CUDA fault that cannot be caught)
    if not len(tail) == len(epochs_per_sample) == n_edges:
        raise ValueError(
            f"head, tail and epochs_per_sample must have one entry per edge, but have lengths {n_edges}, {len(tail)} "
            f"and {len(epochs_per_sample)}"
        )
    # CuPy is imported here, not at module level: piu_mosso imports without it
    import cupy

    layout_edges, apply_sums = _kernels()

    # Edges in a seeded random order (on the host, so the device holds no permutation), then on the device
    seeds = [int(seed) % 2**32 for seed in rng_state]
    order = np.random.default_rng(seeds).permutation(n_edges)
    eps = np.asarray(epochs_per_sample, dtype=np.float64)[order]
    positions = cupy.asarray(np.ascontiguousarray(embedding, dtype=np.float32))
    head = cupy.asarray(np.asarray(head, dtype=np.int32)[order])
    tail = cupy.asarray(np.asarray(tail, dtype=np.int32)[order])
    epochs_per_sample = cupy.asarray(eps)
    next_sample = epochs_per_sample.copy()
    if negative_sample_rate > 0:
        next_negative = epochs_per_sample / negative_sample_rate
    else:
        # With no negative samples the counter is never read; infinity avoids dividing by zero
        next_negative = cupy.full_like(epochs_per_sample, np.inf)
    sums = cupy.zeros(positions.size, dtype=cupy.int64)
    hash_seed = np.uint64(seeds[0] + 2**32 * seeds[1])
    bounds = np.linspace(0, n_edges, CHUNKS + 1).astype(np.int64)
    apply_blocks = (positions.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    alpha = initial_alpha
    for epoch in range(n_epochs):
        for first, end in zip(bounds[:-1], bounds[1:]):
            blocks = int(end - first + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            if blocks == 0:
                continue
            layout_edges(
                (blocks,), (THREADS_PER_BLOCK,),
                (positions, sums, head, tail, epochs_per_sample, next_sample, next_negative, np.int64(first),
                 np.int64(end), np.int32(n_cells), np.int32(dim), np.float32(a), np.float32(b), np.float32(gamma),
                 np.float32(alpha), np.int32(epoch), np.float64(negative_sample_rate), hash_seed,
                 np.float64(FIXED_POINT_SCALE)),
            )
            apply_sums((apply_blocks,), (THREADS_PER_BLOCK,),
                       (positions, sums, np.int64(positions.size), np.float64(FIXED_POINT_SCALE),
                        np.float64(MAX_MOVE * alpha)))
        # umap-learn sets the next epoch's learning rate after each epoch (so epochs 0 and 1 both use initial_alpha)
        alpha = initial_alpha * (1.0 - epoch / n_epochs)
    return cupy.asnumpy(positions)


def _kernels():
    """The compiled layout_edges and apply_sums kernels (compiled at the first call)."""
    import cupy

    global _module
    if _module is None:
        _module = cupy.RawModule(code=_SOURCE)
    return _module.get_function("layout_edges"), _module.get_function("apply_sums")
