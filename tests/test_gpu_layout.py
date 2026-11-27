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

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from umap.layouts import optimize_layout_euclidean
from umap.umap_ import find_ab_params, make_epochs_per_sample

import piu_mosso.gpu_layout
from piu_mosso.gpu_layout import check_layout_arguments, optimize_layout_gpu


def cupy_gpu_available():
    """Whether CuPy imports and sees a CUDA GPU (GPU tests are skipped otherwise)."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no CuPy, or no usable CUDA
        return False


needs_gpu = pytest.mark.skipif(not cupy_gpu_available(), reason="needs CuPy and a CUDA GPU")
A, B = find_ab_params(1.0, 0.5)
SEEDS = np.array([11, -22, 33], dtype=np.int64)


def disjoint_edges(n_pairs, seed=0):
    """Cells 2i and 2i + 1 joined by one edge (stored once) with mixed weights, and a random start in [0, 10]."""
    rng = np.random.default_rng(seed)
    head = np.arange(0, 2 * n_pairs, 2, dtype=np.int32)
    weights = rng.choice([1.0, 0.5, 0.3, 0.1], size=n_pairs)
    start = (rng.random((2 * n_pairs, 2)) * 10).astype(np.float32)
    return head, head + 1, weights, start


def clustered_edges(n_clusters, cells_per_cluster, degree, seed=0):
    """Each cell joined to `degree` random cells of its own cluster (both directions stored), weights in (0, 1]."""
    rng = np.random.default_rng(seed)
    n = n_clusters * cells_per_cluster
    labels = np.repeat(np.arange(n_clusters), cells_per_cluster)
    head = np.repeat(np.arange(n), degree)
    tail = labels[head] * cells_per_cluster + rng.integers(0, cells_per_cluster, head.size)
    keep = head != tail
    head, tail = head[keep], tail[keep]
    weights = rng.random(head.size) * 0.9 + 0.1
    return (np.concatenate([head, tail]).astype(np.int32), np.concatenate([tail, head]).astype(np.int32),
            np.concatenate([weights, weights]), labels)


def cluster_agreement(layout, labels, k=15):
    """Mean fraction of each cell's k nearest layout neighbours that share its cluster."""
    _, neighbours = NearestNeighbors(n_neighbors=k + 1).fit(layout).kneighbors(layout)
    return (labels[neighbours[:, 1:]] == labels[:, None]).mean()


def cpu_serial_layout(start, head, tail, n_epochs, eps, gamma=1.0):
    """umap-learn's serial layout with move_other, as umap_embedding calls it with parallel=False."""
    embedding = start.copy()
    return optimize_layout_euclidean(embedding, embedding, head, tail, n_epochs, start.shape[0], eps, A, B, SEEDS,
                                     gamma=gamma, parallel=False, move_other=True)


def test_too_many_cells_raise():
    with pytest.raises(ValueError, match="2\\^31"):
        check_layout_arguments(2**31, 2, 5)


def test_too_many_dimensions_raise():
    with pytest.raises(ValueError, match="use_gpu=False"):
        check_layout_arguments(100, 9, 5)


def test_negative_negative_sample_rate_raises():
    with pytest.raises(ValueError, match="negative_sample_rate"):
        check_layout_arguments(100, 2, -1)


@needs_gpu
def test_attraction_and_schedule_equal_umap_learn_on_disjoint_edges():
    # Each cell has one edge, so updating a launch's edges together equals updating them one by one; with gamma=0 the
    # negative samples do not move anything, so the GPU layout must equal umap-learn's serial layout: the attraction
    # step, the epochs at which each weight's edge is sampled and the learning rate. Tolerance: float32 rounding.
    head, tail, weights, start = disjoint_edges(2000)
    eps = make_epochs_per_sample(weights, 50)
    expected = cpu_serial_layout(start, head, tail, 50, eps, gamma=0.0)
    result = optimize_layout_gpu(start, head, tail, 50, eps, A, B, SEEDS, gamma=0.0)
    np.testing.assert_allclose(result, expected, atol=1e-4)


@needs_gpu
def test_no_negative_samples_at_rate_zero():
    # A negative_sample_rate of 0 draws no negative samples: the same layout as gamma=0, where they move nothing
    head, tail, weights, start = disjoint_edges(500)
    eps = make_epochs_per_sample(weights, 30)
    without = optimize_layout_gpu(start, head, tail, 30, eps, A, B, SEEDS, negative_sample_rate=0)
    no_repulsion = optimize_layout_gpu(start, head, tail, 30, eps, A, B, SEEDS, gamma=0.0)
    np.testing.assert_array_equal(without, no_repulsion)


@needs_gpu
def test_same_seeds_give_the_same_layout():
    head, tail, weights, _ = clustered_edges(10, 300, 8)
    start = (np.random.default_rng(1).random((3000, 2)) * 10).astype(np.float32)
    eps = make_epochs_per_sample(weights, 50)
    first = optimize_layout_gpu(start, head, tail, 50, eps, A, B, SEEDS)
    np.testing.assert_array_equal(optimize_layout_gpu(start, head, tail, 50, eps, A, B, SEEDS), first)
    assert not np.array_equal(optimize_layout_gpu(start, head, tail, 50, eps, A, B, SEEDS + 1), first)


@needs_gpu
def test_layout_keeps_clusters_as_umap_learn_does():
    # 20 clusters from a random start: umap-learn's serial layout gathers each cluster; the GPU layout must do as
    # well. 0.02 is about the run-to-run spread of the agreement between umap-learn seeds on this graph.
    head, tail, weights, labels = clustered_edges(20, 200, 10, seed=3)
    start = (np.random.default_rng(4).random((4000, 2)) * 10).astype(np.float32)
    eps = make_epochs_per_sample(weights, 200)
    expected = cluster_agreement(cpu_serial_layout(start, head, tail, 200, eps), labels)
    result = optimize_layout_gpu(start, head, tail, 200, eps, A, B, SEEDS)
    assert np.isfinite(result).all()
    assert cluster_agreement(result, labels) >= expected - 0.02


@needs_gpu
def test_a_cell_with_many_edges_keeps_the_layout_bounded(monkeypatch):
    # A hub joined to 5,000 cells sums thousands of clipped steps per launch; with 2 launches per epoch the layout
    # must still stay finite and of the start's size
    monkeypatch.setattr(piu_mosso.gpu_layout, "CHUNKS", 2)
    n = 5001
    head = np.concatenate([np.zeros(n - 1), np.arange(1, n)]).astype(np.int32)
    tail = np.concatenate([np.arange(1, n), np.zeros(n - 1)]).astype(np.int32)
    start = (np.random.default_rng(5).random((n, 2)) * 10).astype(np.float32)
    eps = make_epochs_per_sample(np.ones(head.size), 50)
    result = optimize_layout_gpu(start, head, tail, 50, eps, A, B, SEEDS)
    assert np.isfinite(result).all()
    assert np.ptp(result, axis=0).max() < 100


@needs_gpu
def test_too_little_gpu_memory_raises(monkeypatch):
    import cupy

    monkeypatch.setattr(cupy.cuda.runtime, "memGetInfo", lambda: (2**20, 2**30))
    with pytest.raises(RuntimeError, match="use_gpu=False"):
        piu_mosso.gpu_layout.check_memory(10_000_000, 200_000_000, 2)


def test_mismatched_edge_arrays_raise():
    head, tail, weights, start = disjoint_edges(10)
    with pytest.raises(ValueError, match="one entry per edge"):
        optimize_layout_gpu(start, head, tail[:-1], 5, make_epochs_per_sample(weights, 5), A, B, SEEDS)
