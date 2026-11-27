"""
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

import numbers
import warnings

import faiss
import numpy as np
import scipy.sparse

# Default largest number of cells searched exactly (IndexFlatL2, n^2 d work); above it, the search is approximate
# (IVF). Chosen from a benchmark of exact against IVF search time on each device: exact search on a GPU stays faster
# than IVF up to much larger sizes than on the CPU, so each device has its own default
CPU_EXACT_MAX_CELLS = 2**16
GPU_EXACT_MAX_CELLS = 2**21

# nprobe is doubled until the IVF search finds this fraction of the exact neighbours of a random sample of cells
TARGET_RECALL = 0.99
RECALL_SAMPLE_SIZE = 2_000

# FAISS's guidelines: nlist between 4 sqrt(n) and 16 sqrt(n) lists, and at least 39 training cells per list
NLIST_PER_SQRT_CELL = 4
TRAINING_CELLS_PER_LIST = 64

# Cells per add or search call, which bounds FAISS's temporary memory (tighter on a GPU)
CPU_BATCH_SIZE = 2**20
GPU_BATCH_SIZE = 2**16

# FAISS's GPU indexes return at most 2048 neighbours per query and probe at most 2048 lists
GPU_MAX_K = 2048
GPU_MAX_NPROBE = 2048

# Values per chunk when recomputing distances: a chunk of cells holds cells x k x dims float64 differences
# (2^25 values = 256 MiB)
DISTANCE_CHUNK_VALUES = 2**25


def knn(X, k, *, exact_max_cells=None, use_gpu=False, random_state=0):
    """
    The k nearest neighbours of every cell (row of X) other than the cell itself, by Euclidean distance.

    Up to exact_max_cells cells (by default 2**16 on the CPU, 2**21 on a GPU) the search is exact (FAISS
    IndexFlatL2). Above, it is approximate (IndexIVFFlat with 4 sqrt(n) lists): the number of lists searched (nprobe)
    is doubled from 1 until the neighbours of a random sample of 2,000 cells include 99% of their exact neighbours.
    Cells left with fewer than k neighbours by the approximate search (FAISS marks the gaps with -1) are searched
    again exactly.

    The distances of the pairs found are then recomputed exactly (in float64), since FAISS computes them as
    |x|^2 + |y|^2 - 2 x.y, which loses precision for close pairs, and each row is sorted by them.

    With use_gpu, the index is built and searched on the first visible GPU; the exact searches of the recall sample
    and of incomplete cells run on the CPU (faiss.knn).

    Parameters
    ----------
    X
        Dense array of shape (n_cells, n_dims), e.g. a PCA or latent space. Converted to float32 (a copy unless X
        already is a C-contiguous float32 array).
    k
        Number of neighbours per cell, not counting the cell itself (scanpy's n_neighbors counts it: k = n_neighbors
        - 1). Duplicates of a cell can be among its neighbours, at distance 0.
    exact_max_cells
        Largest number of cells searched exactly. None (the default): CPU_EXACT_MAX_CELLS (2**16) on the CPU,
        GPU_EXACT_MAX_CELLS (2**21) with use_gpu.
    use_gpu
        Search on the first visible GPU. Needs the faiss-gpu-cu12 build of faiss in place of faiss-cpu.
    random_state
        Seed for the IVF training sample, its k-means and the recall sample.

    Returns
    -------
    indices
        int64 array of shape (n_cells, k): the neighbours of each cell, nearest first.
    distances
        float32 array of shape (n_cells, k): their Euclidean distances.
    """
    # Cheap checks first, before X is converted
    _check_dense(X)
    _check_shape(np.shape(X))
    _check_k(k, np.shape(X)[0])
    if use_gpu:
        _check_gpu_available()
        _check_k_on_gpu(k)
    X = _as_float32_matrix(X)
    if exact_max_cells is None:
        exact_max_cells = GPU_EXACT_MAX_CELLS if use_gpu else CPU_EXACT_MAX_CELLS
    rng = np.random.default_rng(random_state)
    # The GPU indexes need their resources for as long as they are used
    gpu_resources = faiss.StandardGpuResources() if use_gpu else None

    if len(X) <= exact_max_cells:
        index = _build_index(faiss.IndexFlatL2(X.shape[1]), X, gpu_resources)
    else:
        index = _build_ivf_index(X, gpu_resources, rng, random_state)
        index.nprobe = _tuned_nprobe(index, X, k, use_gpu, rng)
    # Search k + 1 neighbours, since the cell itself is usually among them
    candidates = _search(index, X, k + 1, use_gpu)
    indices = _drop_self(candidates, np.arange(len(X)), k)

    # Cells whose IVF lists held fewer than k other cells are searched again exactly
    incomplete = np.flatnonzero(np.any(indices < 0, axis=1))
    if len(incomplete) > 0:
        indices[incomplete] = _exact_neighbors(X, incomplete, k)

    return _exact_distances(X, np.arange(len(X)), indices)


def _as_float32_matrix(X):
    """X as a C-contiguous float32 array, after checking its values are finite."""
    # Values too large for float32 become infinite, which the check below reports
    with np.errstate(over="ignore"):
        X = np.ascontiguousarray(X, dtype=np.float32)
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X has NaN or infinite values (or values too large for float32): remove or replace them before "
            "searching neighbours"
        )
    return X


def _check_dense(X):
    """Raise if X is a sparse matrix, which FAISS cannot search."""
    if scipy.sparse.issparse(X):
        raise ValueError(
            "X is a sparse matrix, but FAISS searches dense arrays: pass a dense matrix such as a PCA of the data"
        )


def _check_shape(shape):
    """Raise if X's shape is not (n_cells, n_dims) with at least one dimension."""
    if len(shape) != 2 or shape[1] < 1:
        raise ValueError(f"X must be a 2-D array of shape (n_cells, n_dims) with n_dims >= 1, but has shape {shape}")


def _check_k(k, n_cells):
    """Raise if k is not an integer between 1 and n_cells - 1 (each cell needs k other cells)."""
    if not isinstance(k, numbers.Integral) or isinstance(k, bool):
        raise ValueError(f"k must be an integer, but is {k!r}")
    if k < 1:
        raise ValueError(f"k must be at least 1, but is {k}")
    if k >= n_cells:
        raise ValueError(f"k must be fewer than the {n_cells} cells, since a cell is not its own neighbour, but is {k}")


def _check_gpu_available():
    """Raise if this faiss build cannot search on a GPU here."""
    if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() == 0:
        raise RuntimeError(
            "use_gpu=True, but this faiss has no GPU support or sees no GPU: install faiss-gpu-cu12 in place of "
            "faiss-cpu (see the README) on a machine with a CUDA GPU, or use use_gpu=False"
        )


def _check_k_on_gpu(k):
    """Raise if k + 1 neighbours (with the cell itself) are more than a FAISS GPU index returns."""
    if k + 1 > GPU_MAX_K:
        raise ValueError(
            f"On a GPU, k can be at most {GPU_MAX_K - 1} (FAISS's limit), but is {k}: lower k or use use_gpu=False"
        )


def _build_index(index, X, gpu_resources):
    """A trained, empty index filled with X, moved to the GPU first if gpu_resources are given."""
    index = _on_device(index, gpu_resources)
    _add_in_batches(index, X, gpu_resources is not None)
    return index


def _build_ivf_index(X, gpu_resources, rng, random_state):
    """An IVF index of X, its lists trained by k-means (seeded by random_state) on a random sample of cells."""
    n_cells, n_dims = X.shape
    # k-means needs at least one training cell per list, which 4 sqrt(n) lists exceed below 16 cells
    nlist = min(n_cells, max(1, int(NLIST_PER_SQRT_CELL * np.sqrt(n_cells))))
    index = _on_device(faiss.IndexIVFFlat(faiss.IndexFlatL2(n_dims), n_dims, nlist, faiss.METRIC_L2), gpu_resources)
    index.cp.seed = random_state

    # Sorted, so the training sample is read from X in memory order
    sample = np.sort(rng.choice(n_cells, size=min(n_cells, TRAINING_CELLS_PER_LIST * nlist), replace=False))
    index.train(X[sample])
    _add_in_batches(index, X, gpu_resources is not None)
    return index


def _on_device(index, gpu_resources):
    """The index itself, or a copy on the first visible GPU using gpu_resources if they are given."""
    if gpu_resources is None:
        return index
    return faiss.index_cpu_to_gpu(gpu_resources, 0, index)


def _add_in_batches(index, X, use_gpu):
    """Add the cells of X to the index, a batch at a time."""
    batch_size = GPU_BATCH_SIZE if use_gpu else CPU_BATCH_SIZE
    for start in range(0, len(X), batch_size):
        index.add(X[start:start + batch_size])


def _search(index, X, n_results, use_gpu):
    """The indices of the n_results nearest cells in the index for every cell of X (-1 where FAISS found fewer)."""
    batch_size = GPU_BATCH_SIZE if use_gpu else CPU_BATCH_SIZE
    results = np.empty((len(X), n_results), dtype=np.int64)
    for start in range(0, len(X), batch_size):
        _, results[start:start + batch_size] = index.search(X[start:start + batch_size], n_results)
    return results


def _drop_self(candidates, cells, k):
    """
    The first k candidates of each row other than its own cell. Duplicates of a cell tie with it at distance 0, so
    the cell can be anywhere among them, or missing; if it is missing, the last candidate is dropped instead.
    """
    # A stable sort on "is the cell itself" moves the cell to the end and keeps the other candidates in order
    order = np.argsort(candidates == cells[:, None], axis=1, kind="stable")[:, :k]
    return np.take_along_axis(candidates, order, axis=1)


def _exact_neighbors(X, cells, k):
    """The exact k nearest neighbours of the given cells (indices into X) other than themselves, by brute force."""
    _, candidates = faiss.knn(X[cells], X, k + 1)
    return _drop_self(candidates, cells, k)


def _tuned_nprobe(index, X, k, use_gpu, rng):
    """
    The smallest power of 2 nprobe (lists searched per query) at which a random sample of cells finds TARGET_RECALL
    of its exact neighbours, or the most lists the index can probe. The search sets index.nprobe as it goes; the
    caller sets the returned value. Warns if even the most lists miss the target, which happens only on a GPU
    (FAISS's GPU limit of 2,048 lists); on the CPU the most is every list, an exhaustive search.
    """
    sample = rng.choice(len(X), size=min(len(X), RECALL_SAMPLE_SIZE), replace=False)
    _, exact_distances = _exact_distances(X, sample, _exact_neighbors(X, sample, k))
    max_nprobe = min(index.nlist, GPU_MAX_NPROBE) if use_gpu else index.nlist

    nprobe = 1
    while True:
        index.nprobe = nprobe
        found = _drop_self(_search(index, X[sample], k + 1, use_gpu), sample, k)
        recall = _recall(X, sample, found, exact_distances)
        if recall >= TARGET_RECALL or nprobe >= max_nprobe:
            break
        nprobe = min(2 * nprobe, max_nprobe)
    if recall < TARGET_RECALL:
        warnings.warn(
            f"The approximate neighbour search reaches a recall of {recall:.3f}, below the target {TARGET_RECALL}, "
            f"at the most lists FAISS searches on a GPU ({nprobe}): use use_gpu=False for an exhaustive search"
        )
    return nprobe


def _recall(X, cells, found, exact_distances):
    """
    Fraction of the k exact neighbours of the given cells that the search found. A found neighbour counts as a hit
    when it is no farther than the k-th exact neighbour, so a tie (e.g. a duplicate) replacing an exact neighbour of
    the same distance is not a miss. Missing neighbours (-1) are never hits.
    """
    valid = found >= 0
    _, found_distances = _exact_distances(X, cells, np.where(valid, found, cells[:, None]))
    hits = valid & (found_distances <= exact_distances[:, -1:])
    return hits.mean()


def _exact_distances(X, cells, neighbors):
    """
    The neighbours of the given cells (indices into X; one row of neighbours per cell) sorted by their exact
    Euclidean distance, with those distances (float32). Computed from the differences x - y in float64, a chunk of
    cells at a time; ties (e.g. duplicates at distance 0) keep their order.
    """
    sorted_neighbors = np.empty_like(neighbors)
    distances = np.empty(neighbors.shape, dtype=np.float32)
    chunk_size = max(1, DISTANCE_CHUNK_VALUES // (neighbors.shape[1] * X.shape[1]))
    for start in range(0, len(cells), chunk_size):
        chunk = slice(start, start + chunk_size)
        differences = X[neighbors[chunk]].astype(np.float64) - X[cells[chunk], None, :]
        chunk_distances = np.sqrt(np.einsum("ijk,ijk->ij", differences, differences))
        order = np.argsort(chunk_distances, axis=1, kind="stable")
        sorted_neighbors[chunk] = np.take_along_axis(neighbors[chunk], order, axis=1)
        distances[chunk] = np.take_along_axis(chunk_distances, order, axis=1)
    return sorted_neighbors, distances
