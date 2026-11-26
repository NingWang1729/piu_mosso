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

import anndata as ad
import numpy as np
import pytest
import scanpy as sc

from piu_mosso import knn, knn_distance_matrix, umap_connectivities


# Tolerances against scanpy. Distances: float32 rounding. Weights: umap-learn's sigma search (smooth_knn_dist) stops
# when a row's weight sum is within 1e-5 of its target, so two runs on slightly different distances can differ by
# 2e-5 in each directed weight (all weights of a row move together with sigma), and by up to 4e-5 after the fuzzy
# union w + w^T - w * w^T
DISTANCE_ATOL = 1e-5
CONNECTIVITY_ATOL = 5e-5


def pca_adata(n_cells=1000, n_dims=20, seed=0):
    """An AnnData with a clustered representation in obsm['X_pca'] (below scanpy's 8,192-cell exact-search limit)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(10, n_dims))
    X_pca = (centers[rng.integers(10, size=n_cells)] + rng.normal(size=(n_cells, n_dims))).astype(np.float32)
    adata = ad.AnnData(np.zeros((n_cells, 1), dtype=np.float32))
    adata.obsm["X_pca"] = X_pca
    return adata


@pytest.mark.parametrize("n_neighbors", [5, 15, 30])
def test_exact_graph_equals_scanpy(n_neighbors):
    adata = pca_adata()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    # scanpy's n_neighbors counts the cell itself
    indices, distances = knn(adata.obsm["X_pca"], n_neighbors - 1)

    connectivities = umap_connectivities(indices, distances)
    distance_matrix = knn_distance_matrix(indices, distances)
    # scanpy's distance matrix holds n_neighbors - 1 entries per row, like piu_mosso's
    assert distance_matrix.nnz == adata.obsp["distances"].nnz == adata.n_obs * (n_neighbors - 1)
    for result, expected, atol in [
        (distance_matrix, adata.obsp["distances"], DISTANCE_ATOL),
        (connectivities, adata.obsp["connectivities"], CONNECTIVITY_ATOL),
    ]:
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result.toarray() != 0, expected.toarray() != 0)
        np.testing.assert_allclose(result.toarray(), expected.toarray(), atol=atol)


def test_distance_matrix_stores_duplicates_at_distance_zero():
    # Cells 2i and 2i + 1 are copies; the copy's distance 0 is stored, so every row keeps k entries
    X = np.repeat(np.random.default_rng(1).normal(size=(50, 3)), 2, axis=0)
    distance_matrix = knn_distance_matrix(*knn(X, 4))
    np.testing.assert_array_equal(np.diff(distance_matrix.indptr), 4)
    first_row = dict(zip(distance_matrix[0].indices, distance_matrix[0].data))
    assert first_row[1] == 0


@pytest.mark.parametrize("function", [umap_connectivities, knn_distance_matrix])
@pytest.mark.parametrize(
    ("indices", "distances", "message"),
    [
        (np.array([[1], [-1]]), np.ones((2, 1)), "outside that range"),
        (np.array([[1], [2]]), np.ones((2, 1)), "outside that range"),
        (np.array([[1], [0]]), np.ones((2, 2)), "same shape"),
        (np.zeros((2, 0), dtype=int), np.zeros((2, 0)), "k >= 1"),
        (np.array([[1.0], [0.0]]), np.ones((2, 1)), "integer"),
        (np.array([[0, 1], [1, 0]]), np.ones((2, 2)), "own neighbour"),
    ],
    ids=["missing_neighbor", "too_large", "shapes_differ", "no_neighbors", "float_indices", "self_column"],
)
def test_bad_neighbors_raise(function, indices, distances, message):
    with pytest.raises(ValueError, match=message):
        function(indices, distances)
