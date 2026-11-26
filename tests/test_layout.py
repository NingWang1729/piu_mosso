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
import scipy.sparse
from sklearn.utils import check_random_state
from umap.umap_ import find_ab_params, simplicial_set_embedding

from piu_mosso import knn, umap_connectivities, umap_embedding
from piu_mosso.layout import default_n_epochs


def clustered_graph(n_clusters=5, cells_per_cluster=100, separation=5.0, seed=0):
    """Data in Gaussian clusters and its UMAP graph (15 neighbours); far-apart clusters give separate components."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=separation, size=(n_clusters, 10))
    noise = rng.normal(size=(n_clusters * cells_per_cluster, 10))
    X = (centers.repeat(cells_per_cluster, axis=0) + noise).astype(np.float32)
    return X, umap_connectivities(*knn(X, 14))


def umap_learn_layout(X, graph, init, n_epochs=None, min_dist=0.5, seed=0):
    """umap-learn's serial layout (as sc.tl.umap calls it) with scanpy's defaults."""
    a, b = find_ab_params(1.0, min_dist)
    embedding, _ = simplicial_set_embedding(
        X, graph, 2, 1.0, a, b, 1.0, 5, n_epochs, init, check_random_state(seed), "euclidean", {},
        densmap=False, densmap_kwds={}, output_dens=False, parallel=False,
    )
    return embedding


@pytest.mark.parametrize(
    ("graph_kwargs", "init"),
    [
        ({}, "spectral"),
        ({}, "pca"),
        ({"n_clusters": 12, "separation": 100.0}, "spectral"),  # many connected components
    ],
    ids=["spectral", "pca", "spectral_many_components"],
)
def test_serial_layout_equals_umap_learn(graph_kwargs, init):
    X, graph = clustered_graph(**graph_kwargs)
    result = umap_embedding(X, graph, init=init, n_epochs=50, parallel=False)
    np.testing.assert_array_equal(result, umap_learn_layout(X, graph, init, n_epochs=50))


@pytest.mark.parametrize("duplicates", [False, True], ids=["distinct", "duplicate_rows"])
def test_serial_layout_from_an_array_equals_umap_learn(duplicates):
    X, graph = clustered_graph()
    start = np.random.default_rng(1).normal(size=(len(X), 2))
    if duplicates:
        start[1] = start[0]
    result = umap_embedding(X, graph, init=start, n_epochs=50, parallel=False)
    np.testing.assert_array_equal(result, umap_learn_layout(X, graph, start, n_epochs=50))


def test_few_epochs_prune_as_umap_learn():
    # With 10 or fewer epochs umap-learn prunes weak edges by the default number of epochs instead
    X, graph = clustered_graph()
    result = umap_embedding(X, graph, n_epochs=5, parallel=False)
    np.testing.assert_array_equal(result, umap_learn_layout(X, graph, "spectral", n_epochs=5))


def test_sparse_data_pca_start_equals_umap_learn():
    X, graph = clustered_graph()
    X = scipy.sparse.csr_matrix(np.maximum(X, 0))
    result = umap_embedding(X, graph, init="pca", n_epochs=20, parallel=False)
    np.testing.assert_array_equal(result, umap_learn_layout(X, graph, "pca", n_epochs=20))


def test_default_epochs_follow_umap_learn():
    assert default_n_epochs(10_000) == 500
    assert default_n_epochs(10_001) == 200
    X, graph = clustered_graph()
    np.testing.assert_array_equal(
        umap_embedding(X, graph, parallel=False), umap_embedding(X, graph, n_epochs=500, parallel=False)
    )


def test_parallel_layout_keeps_clusters_apart():
    # The default parallel layout: right shape and dtype, finite, and each well-separated cluster stays compact
    X, graph = clustered_graph(separation=20.0)
    embedding = umap_embedding(X, graph)
    assert embedding.shape == (len(X), 2) and embedding.dtype == np.float32
    assert np.all(np.isfinite(embedding))
    clusters = embedding.reshape(5, 100, 2)
    centers = clusters.mean(axis=1)
    within = np.linalg.norm(clusters - centers[:, None], axis=2).max()
    between = min(np.linalg.norm(centers[i] - centers[j]) for i in range(5) for j in range(i + 1, 5))
    assert within < between


def test_layout_does_not_change_the_graph():
    X, graph = clustered_graph()
    before = graph.copy()
    umap_embedding(X, graph, n_epochs=20)
    assert (graph != before).nnz == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"init": "random"}, "init must be one of"),
        ({"init": np.zeros((3, 2))}, "shape"),
        ({"n_epochs": 0}, "n_epochs"),
        ({"n_epochs": True}, "n_epochs"),
    ],
    ids=["unknown_init", "init_shape", "zero_epochs", "bool_epochs"],
)
def test_bad_options_raise(kwargs, message):
    X, graph = clustered_graph()
    with pytest.raises(ValueError, match=message):
        umap_embedding(X, graph, **kwargs)


def test_mismatched_shapes_raise():
    X, graph = clustered_graph()
    with pytest.raises(ValueError, match="one row per cell"):
        umap_embedding(X[1:], graph, init="pca")
    with pytest.raises(ValueError, match="one row per cell"):
        umap_embedding(X, graph[1:], init="pca")


def test_init_names_ignore_case():
    X, graph = clustered_graph()
    np.testing.assert_array_equal(
        umap_embedding(X, graph, init="PCA", n_epochs=20, parallel=False),
        umap_embedding(X, graph, init="pca", n_epochs=20, parallel=False),
    )
