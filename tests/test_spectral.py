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
import umap.spectral
from sklearn.utils import check_random_state

import piu_mosso.spectral
from piu_mosso import knn, umap_connectivities
from piu_mosso.spectral import choose_eigensolver, spectral_layout


def clustered_graph(n_clusters, cells_per_cluster, separation, seed=0):
    """Data in Gaussian clusters and its UMAP graph (15 neighbours); far-apart clusters give separate components."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=separation, size=(n_clusters, 10))
    noise = rng.normal(size=(n_clusters * cells_per_cluster, 10))
    X = (centers.repeat(cells_per_cluster, axis=0) + noise).astype(np.float32)
    return X, umap_connectivities(*knn(X, 14)).tocoo()


# Tests of an internal rule: the size at which the solver switches, checked without building graphs of millions
# of cells. The behaviour (lobpcg gives the same layout) is tested below.
@pytest.mark.parametrize(
    ("n_vertices", "expected"),
    [
        (1_000, "eigsh"),
        (1_600_000, "eigsh"),
        (1_700_000, "lobpcg"),  # eigsh's Lanczos basis would pass 2^31 - 1 entries
        (2_500_000, "lobpcg"),  # umap-learn's own switch, from 2,000,000
    ],
)
def test_eigensolver_switches_where_arpack_would_overflow(n_vertices, expected):
    assert choose_eigensolver(n_vertices, 2) == expected


def test_eigensolver_switches_at_the_first_overflowing_size():
    # int(sqrt(n)) = 1290 here: 1,664,716 x 1290 = 2,147,483,640 <= 2^31 - 1 = 2,147,483,647 < 1,664,717 x 1290
    assert choose_eigensolver(1_664_716, 2) == "eigsh"
    assert choose_eigensolver(1_664_717, 2) == "lobpcg"


@pytest.mark.parametrize(
    ("n_clusters", "separation"), [(1, 1.0), (12, 100.0)], ids=["one_component", "many_components"]
)
def test_spectral_layout_equals_umap_learn(n_clusters, separation):
    X, graph = clustered_graph(n_clusters, 1200 // n_clusters, separation)
    expected = umap.spectral.spectral_layout(X, graph, 2, check_random_state(0))
    result = spectral_layout(X, graph, 2, check_random_state(0))
    np.testing.assert_array_equal(result, expected)


def test_lobpcg_layout_spans_the_same_space_as_eigsh(monkeypatch):
    # Large graphs use lobpcg; forced here on a small one, its eigenvectors span eigsh's (up to sign and rotation)
    X, graph = clustered_graph(1, 1000, 1.0)
    eigsh_layout = spectral_layout(X, graph, 2, check_random_state(0))
    monkeypatch.setattr(piu_mosso.spectral, "choose_eigensolver", lambda n_vertices, dim: "lobpcg")
    lobpcg_layout = spectral_layout(X, graph, 2, check_random_state(0))
    # Projecting one basis onto the other keeps its length when the spaces are the same
    q_eigsh, _ = np.linalg.qr(eigsh_layout)
    q_lobpcg, _ = np.linalg.qr(lobpcg_layout)
    np.testing.assert_allclose(np.linalg.svd(q_eigsh.T @ q_lobpcg, compute_uv=False), 1.0, atol=1e-3)


def test_tiny_components_are_placed_at_random_as_umap_learn():
    # Components too small for an eigensolver (fewer than 2 dim cells) are placed at random near their centroid's
    # position, as umap-learn does, not treated as a failure
    sizes = [300, 5, 4, 3, 1, 2, 6, 200]
    rng = np.random.default_rng(2)
    blocks = [scipy.sparse.random(n, n, density=0.5, random_state=rng) for n in sizes]
    graph = scipy.sparse.block_diag([block + block.T for block in blocks])
    X = rng.normal(size=(graph.shape[0], 5))
    expected = umap.spectral.spectral_layout(X, graph.tocoo(), 2, check_random_state(0))
    np.testing.assert_array_equal(spectral_layout(X, graph.tocoo(), 2, check_random_state(0)), expected)


@pytest.mark.parametrize("solver", ["eigsh", "lobpcg"])
def test_failed_eigensolver_raises_naming_pca(monkeypatch, solver):
    # One iteration is too few for either solver to converge; umap-learn would fall back to a random start
    X, graph = clustered_graph(1, 1000, 1.0)
    monkeypatch.setattr(piu_mosso.spectral, "choose_eigensolver", lambda n_vertices, dim: solver)
    with pytest.raises(RuntimeError, match="init='pca'"):
        spectral_layout(X, graph, 2, check_random_state(0), maxiter=1)
