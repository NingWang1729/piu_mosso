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
import matplotlib
import numpy as np
import pytest
import scanpy as sc
from sklearn.utils import check_random_state
from umap.umap_ import find_ab_params, simplicial_set_embedding

import piu_mosso

# Plots in the end-to-end test are drawn without a display
matplotlib.use("Agg")


def cupy_gpu_available():
    """Whether CuPy imports and sees a CUDA GPU (GPU tests are skipped otherwise)."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no CuPy, or no usable CUDA
        return False


def clustered_adata(n_cells=600, n_genes=100, seed=0):
    """An AnnData with clustered expression-like data, its PCA and a scanpy neighbour graph."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(6, n_genes))
    X = (centers[rng.integers(6, size=n_cells)] + rng.normal(size=(n_cells, n_genes))).astype(np.float32)
    adata = ad.AnnData(X)
    sc.pp.pca(adata, n_comps=20)
    return adata


def test_serial_umap_equals_scanpy():
    adata = clustered_adata()
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=10)
    expected = adata.copy()
    sc.tl.umap(expected, maxiter=50)

    piu_mosso.tl.umap(adata, maxiter=50, parallel=False)
    np.testing.assert_array_equal(adata.obsm["X_umap"], expected.obsm["X_umap"])
    assert adata.uns["umap"] == expected.uns["umap"]


@pytest.mark.parametrize("n_pcs", [None, 0], ids=["few_genes", "n_pcs_0"])
def test_serial_umap_on_x_equals_scanpy(n_pcs):
    # scanpy builds the graph on .X when it has at most 50 genes, or when n_pcs is 0
    adata = clustered_adata(n_genes=30)
    sc.pp.neighbors(adata, n_pcs=n_pcs)
    expected = adata.copy()
    sc.tl.umap(expected, maxiter=50)

    piu_mosso.tl.umap(adata, maxiter=50, parallel=False)
    np.testing.assert_array_equal(adata.obsm["X_umap"], expected.obsm["X_umap"])


def test_serial_umap_from_pca_equals_umap_learn():
    # "pca" is not one of sc.tl.umap's documented starts, so the reference is umap-learn's own layout of the same
    # graph from the representation scanpy would choose (the first n_pcs = 10 PCs)
    adata = clustered_adata()
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=10)
    a, b = find_ab_params(1.0, 0.5)
    expected, _ = simplicial_set_embedding(
        adata.obsm["X_pca"][:, :10], adata.obsp["connectivities"].copy(), 2, 1.0, a, b, 1.0, 5, 50, "pca",
        check_random_state(0), "euclidean", {}, densmap=False, densmap_kwds={}, output_dens=False,
    )

    piu_mosso.tl.umap(adata, maxiter=50, init_pos="pca", parallel=False)
    np.testing.assert_array_equal(adata.obsm["X_umap"], expected)


def test_serial_umap_from_an_array_equals_scanpy():
    adata = clustered_adata()
    sc.pp.neighbors(adata)
    start = np.random.default_rng(1).normal(size=(adata.n_obs, 2))
    expected = adata.copy()
    sc.tl.umap(expected, maxiter=50, init_pos=start, random_state=3)

    piu_mosso.tl.umap(adata, maxiter=50, init_pos=start, random_state=3, parallel=False)
    np.testing.assert_array_equal(adata.obsm["X_umap"], expected.obsm["X_umap"])
    assert adata.uns["umap"] == expected.uns["umap"]


def test_neighbors_and_umap_end_to_end():
    # A cheap end-to-end run as a user runs it: neighbours, the default parallel UMAP, and scanpy's plot
    adata = clustered_adata(n_cells=3000)
    piu_mosso.pp.neighbors(adata, use_rep="X_pca")
    piu_mosso.tl.umap(adata)
    assert adata.obsm["X_umap"].shape == (3000, 2)
    assert np.all(np.isfinite(adata.obsm["X_umap"]))
    figure = sc.pl.umap(adata, show=False, return_fig=True)
    assert figure is not None


def test_non_euclidean_graph_warns():
    adata = clustered_adata()
    sc.pp.neighbors(adata, metric="cosine")
    with pytest.warns(UserWarning, match="metric='cosine'"):
        piu_mosso.tl.umap(adata, maxiter=5)


@pytest.mark.parametrize("init_pos", ["paga", "X_pca"])
def test_unsupported_init_pos_raises(init_pos):
    adata = clustered_adata()
    piu_mosso.pp.neighbors(adata)
    with pytest.raises(ValueError, match="init_pos must be one of"):
        piu_mosso.tl.umap(adata, init_pos=init_pos)


def test_umap_without_neighbors_raises():
    with pytest.raises(ValueError, match="pp.neighbors"):
        piu_mosso.tl.umap(clustered_adata())


def test_umap_of_a_gauss_graph_raises():
    adata = clustered_adata()
    sc.pp.neighbors(adata, method="gauss")
    with pytest.raises(ValueError, match="method='umap'"):
        piu_mosso.tl.umap(adata)


def test_umap_without_the_graphs_representation_raises():
    adata = clustered_adata()
    piu_mosso.pp.neighbors(adata, use_rep="X_pca")
    del adata.obsm["X_pca"]
    with pytest.raises(ValueError, match="no longer has"):
        piu_mosso.tl.umap(adata)


def test_umap_with_too_few_stored_pcs_raises():
    adata = clustered_adata()
    sc.pp.neighbors(adata, n_pcs=10)
    adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :5]
    with pytest.raises(ValueError, match="n_pcs=10"):
        piu_mosso.tl.umap(adata)


def test_umap_passes_use_gpu_and_records_it(monkeypatch):
    # Without a GPU: the layout is replaced, to check what tl.umap passes and writes
    calls = []
    monkeypatch.setattr(
        piu_mosso.tl, "umap_embedding", lambda X, graph, **kwargs: calls.append(kwargs) or np.zeros((len(X), 2))
    )
    adata = clustered_adata()
    piu_mosso.pp.neighbors(adata, use_rep="X_pca")
    piu_mosso.tl.umap(adata, use_gpu=True)
    assert calls[-1]["use_gpu"] is True
    assert adata.uns["umap"]["params"]["use_gpu"] is True
    piu_mosso.tl.umap(adata)
    assert calls[-1]["use_gpu"] is False
    assert "use_gpu" not in adata.uns["umap"]["params"]


@pytest.mark.skipif(not cupy_gpu_available(), reason="needs CuPy and a CUDA GPU")
def test_gpu_umap_end_to_end_repeats():
    # A PCA start: the spectral start of this graph's separate clusters may itself vary between runs
    adata = clustered_adata(n_cells=3000)
    piu_mosso.pp.neighbors(adata, use_rep="X_pca")
    piu_mosso.tl.umap(adata, init_pos="pca", use_gpu=True)
    first = adata.obsm["X_umap"].copy()
    assert first.shape == (3000, 2) and np.all(np.isfinite(first))
    piu_mosso.tl.umap(adata, init_pos="pca", use_gpu=True)
    np.testing.assert_array_equal(adata.obsm["X_umap"], first)
