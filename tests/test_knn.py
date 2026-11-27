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

import faiss
import numpy as np
import pytest
import scipy.sparse
from sklearn.neighbors import NearestNeighbors

import piu_mosso.faiss_knn
from piu_mosso import knn

# Whether this faiss build can use a GPU here; GPU tests are skipped otherwise
HAS_GPU = hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0

# An exact_max_cells above the cell count of every test here, so the search is exact
EXACT = 10**6


def clustered_data(n_clusters, cells_per_cluster, n_dims=8, spread=0.1, seed=0):
    """Cells in well-separated Gaussian clusters, float32."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=10.0, size=(n_clusters, n_dims))
    noise = rng.normal(scale=spread, size=(n_clusters * cells_per_cluster, n_dims))
    cells = centers.repeat(cells_per_cluster, axis=0) + noise
    return cells.astype(np.float32)


def exact_neighbors(X, k):
    """The k nearest neighbours of each cell other than itself, from scikit-learn's brute-force search."""
    distances, indices = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(X).kneighbors(X)
    return indices[:, 1:], distances[:, 1:]


def recall(found, expected):
    """Fraction of the expected neighbours that were found, over all cells."""
    return np.mean([len(np.intersect1d(f, e)) / len(e) for f, e in zip(found, expected)])


def test_exact_search_equals_brute_force():
    X = np.random.default_rng(0).normal(size=(500, 10)).astype(np.float32)
    indices, distances = knn(X, 10)
    expected_indices, expected_distances = exact_neighbors(X, 10)
    assert indices.shape == distances.shape == (500, 10)
    assert indices.dtype == np.int64 and distances.dtype == np.float32
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(distances, expected_distances, rtol=1e-5, atol=1e-6)


def test_float64_input_gives_the_same_neighbors():
    X = np.random.default_rng(1).normal(size=(300, 6))
    indices64, distances64 = knn(X, 5)
    indices32, distances32 = knn(X.astype(np.float32), 5)
    np.testing.assert_array_equal(indices64, indices32)
    np.testing.assert_allclose(distances64, distances32, rtol=1e-5)


@pytest.mark.parametrize("exact_max_cells", [EXACT, 0], ids=["exact", "ivf"])
def test_more_copies_than_neighbors_never_return_self(exact_max_cells):
    # Each cell appears six times and k + 1 = 3, so FAISS can return three other copies and leave the cell itself out
    X = np.repeat(np.random.default_rng(3).normal(size=(200, 5)).astype(np.float32), 6, axis=0)
    indices, distances = knn(X, 2, exact_max_cells=exact_max_cells)
    assert not np.any(indices == np.arange(len(X))[:, None])
    # Both neighbours are copies of the cell
    np.testing.assert_array_equal(indices // 6, np.repeat(np.arange(len(X))[:, None] // 6, 2, axis=1))
    np.testing.assert_array_equal(distances, 0)


@pytest.mark.parametrize("exact_max_cells", [EXACT, 0], ids=["exact", "ivf"])
def test_duplicate_cells_are_neighbors_but_never_self(exact_max_cells):
    # Each cell appears three times, so two copies tie with the cell itself at distance 0 and FAISS may list
    # them before it
    X = np.repeat(np.random.default_rng(2).normal(size=(400, 5)).astype(np.float32), 3, axis=0)
    indices, distances = knn(X, 4, exact_max_cells=exact_max_cells)
    cells = np.arange(len(X))
    assert not np.any(indices == cells[:, None])
    # The two other copies of each cell come first, at distance 0
    copies = np.stack([cells - cells % 3 + offset for offset in range(3)], axis=1)
    twins = copies[copies != cells[:, None]].reshape(-1, 2)
    np.testing.assert_array_equal(np.sort(indices[:, :2], axis=1), twins)
    np.testing.assert_array_equal(distances[:, :2], 0)


def test_ivf_search_reaches_the_target_recall_with_exact_distances():
    X = clustered_data(n_clusters=50, cells_per_cluster=100, spread=1.0)
    indices, distances = knn(X, 15, exact_max_cells=0)
    expected_indices, _ = exact_neighbors(X, 15)
    # The 0.99 target is reached on a 2,000-cell sample, so the recall of all cells can be a little lower
    assert recall(indices, expected_indices) >= 0.98
    # Distances are the exact Euclidean distances of the pairs found, sorted within each row
    exact = np.linalg.norm(X[indices].astype(np.float64) - X[:, None, :], axis=2)
    np.testing.assert_allclose(distances, exact, rtol=1e-6, atol=1e-6)
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_ivf_search_is_reproducible():
    X = clustered_data(n_clusters=50, cells_per_cluster=100, spread=1.0)
    first = knn(X, 15, exact_max_cells=0, random_state=3)
    second = knn(X, 15, exact_max_cells=0, random_state=3)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


class _IVFChosen(Exception):
    """Raised in place of building an IVF index, to stop knn once it has chosen the approximate search."""


def ivf_chosen(X, monkeypatch, **kwargs):
    """
    Whether knn(X, 1, **kwargs) chooses the approximate (IVF) search rather than the exact one. Tests an internal
    rule: no output tells the two apart, since the IVF search often finds the exact neighbours. Only the IVF search
    builds an IndexIVFFlat; the exact search runs to the end and its output shape is checked.
    """
    def stop(*_args, **_kwargs):
        raise _IVFChosen

    monkeypatch.setattr(faiss, "IndexIVFFlat", stop)
    try:
        indices, distances = knn(X, 1, **kwargs)
    except _IVFChosen:
        return True
    assert indices.shape == distances.shape == (len(X), 1)
    return False


def test_default_is_exact_up_to_2_16_cells_on_the_cpu(monkeypatch):
    X = np.random.default_rng(6).normal(size=(2**16 + 1, 2)).astype(np.float32)
    assert not ivf_chosen(X[:2**16], monkeypatch)
    assert ivf_chosen(X, monkeypatch)


def test_exact_max_cells_overrides_the_default(monkeypatch):
    X = np.random.default_rng(6).normal(size=(2**16 + 1, 2)).astype(np.float32)
    assert not ivf_chosen(X, monkeypatch, exact_max_cells=2**16 + 1)
    assert ivf_chosen(X[:100], monkeypatch, exact_max_cells=0)


@pytest.mark.skipif(not HAS_GPU, reason="needs faiss-gpu and a GPU")
def test_default_is_exact_up_to_2_21_cells_on_a_gpu(monkeypatch):
    X = np.random.default_rng(7).normal(size=(2**21 + 1, 2)).astype(np.float32)
    assert not ivf_chosen(X[:2**21], monkeypatch, use_gpu=True)
    assert ivf_chosen(X, monkeypatch, use_gpu=True)


def test_tiny_data_in_ivf_mode():
    # Fewer cells than the 4 sqrt(n) lists FAISS would otherwise be asked to train
    X = np.random.default_rng(4).normal(size=(10, 3)).astype(np.float32)
    indices, _ = knn(X, 3, exact_max_cells=0)
    np.testing.assert_array_equal(indices, exact_neighbors(X, 3)[0])


def test_non_contiguous_input():
    X = np.random.default_rng(5).normal(size=(300, 12))
    indices, _ = knn(X[:, ::2], 5)
    np.testing.assert_array_equal(indices, exact_neighbors(np.ascontiguousarray(X[:, ::2]), 5)[0])


def test_cells_with_invalid_ivf_neighbors_are_searched_exactly(monkeypatch):
    # Tests an internal rule (TARGET_RECALL is set to 0); the behaviour test of the IVF search is the one above.
    # No recall target, so nprobe stays 1, and k + 1 is more than any IVF list holds (1,200 cells in tight clusters
    # of 4, spread over 4 sqrt(1200) = 138 lists): FAISS fills every row up with -1, and every cell must be searched
    # again exactly
    monkeypatch.setattr(piu_mosso.faiss_knn, "TARGET_RECALL", 0.0)
    X = clustered_data(n_clusters=300, cells_per_cluster=4, spread=0.01)
    indices, distances = knn(X, 40, exact_max_cells=0)
    expected_indices, expected_distances = exact_neighbors(X, 40)
    assert np.all(indices >= 0)
    assert recall(indices, expected_indices) == 1.0
    np.testing.assert_allclose(distances, expected_distances, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("X", "k", "message"),
    [
        (np.zeros(10, dtype=np.float32), 3, "2-D"),
        (np.float32(3.0), 1, "2-D"),
        (np.zeros((5, 0)), 1, "n_dims >= 1"),
        (np.array([[0.0, 1.0], [np.nan, 0.0], [1.0, 1.0]]), 1, "NaN or infinite"),
        (np.zeros((5, 2)), 0, "at least 1"),
        (np.zeros((5, 2)), 5, "fewer than the 5 cells"),
        (scipy.sparse.random(20, 5, density=0.5, format="csr"), 3, "sparse matrix"),
        (np.zeros((5, 2)), 2.0, "integer"),
        (np.array([[1e39, 0.0], [0.0, 0.0], [1.0, 1.0]]), 1, "too large for float32"),
    ],
    ids=["one_dimensional", "scalar", "no_columns", "nan", "k_zero", "k_too_large", "sparse", "k_float", "overflow"],
)
def test_bad_input_raises(X, k, message):
    with pytest.raises(ValueError, match=message):
        knn(X, k)


@pytest.mark.skipif(HAS_GPU, reason="tests the error without a GPU")
def test_use_gpu_without_a_gpu_raises():
    with pytest.raises(RuntimeError, match="faiss-gpu"):
        knn(np.zeros((10, 2), dtype=np.float32), 3, use_gpu=True)


@pytest.mark.skipif(not HAS_GPU, reason="needs faiss-gpu and a GPU")
def test_k_above_the_gpu_limit_raises():
    with pytest.raises(ValueError, match="use_gpu=False"):
        knn(np.zeros((3000, 2), dtype=np.float32), 2048, use_gpu=True)


@pytest.mark.skipif(not HAS_GPU, reason="needs faiss-gpu and a GPU")
@pytest.mark.parametrize("exact_max_cells", [EXACT, 0], ids=["exact", "ivf"])
def test_gpu_search_equals_cpu_search(exact_max_cells):
    X = clustered_data(n_clusters=50, cells_per_cluster=100, spread=1.0)
    cpu_indices, cpu_distances = knn(X, 15, exact_max_cells=exact_max_cells)
    gpu_indices, gpu_distances = knn(X, 15, exact_max_cells=exact_max_cells, use_gpu=True)
    expected_indices, _ = exact_neighbors(X, 15)
    # Both reach the recall target. The exact search finds the same neighbours on both devices, except where the
    # k-th and (k + 1)-th neighbours nearly tie and the devices' float32 distances rank them differently
    assert recall(gpu_indices, expected_indices) >= 0.98
    if exact_max_cells > 0:
        assert recall(gpu_indices, cpu_indices) >= 0.999
        np.testing.assert_allclose(gpu_distances, cpu_distances, rtol=1e-4)
