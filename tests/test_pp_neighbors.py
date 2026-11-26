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
import faiss
import numpy as np
import pytest
import scanpy as sc
import scipy.sparse

import piu_mosso


def pca_adata(n_cells=1000, n_dims=20, seed=0):
    """An AnnData with a clustered representation in obsm['X_pca'] (below scanpy's 8,192-cell exact-search limit)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(10, n_dims))
    X_pca = (centers[rng.integers(10, size=n_cells)] + rng.normal(size=(n_cells, n_dims))).astype(np.float32)
    adata = ad.AnnData(np.zeros((n_cells, 1), dtype=np.float32))
    adata.obsm["X_pca"] = X_pca
    return adata


@pytest.mark.parametrize("use_rep", ["X_pca", "X"])
def test_neighbors_equals_scanpy(use_rep):
    adata = pca_adata()
    if use_rep == "X":
        adata = ad.AnnData(adata.obsm["X_pca"])
    expected = adata.copy()
    sc.pp.neighbors(expected, n_neighbors=10, use_rep=use_rep)

    piu_mosso.pp.neighbors(adata, n_neighbors=10, use_rep=use_rep)
    # Weights within a few 1e-5: umap-learn's sigma search stops within 1e-5 of its target (see test_graph.py)
    for key, atol in [("distances", 1e-5), ("connectivities", 5e-5)]:
        np.testing.assert_allclose(adata.obsp[key].toarray(), expected.obsp[key].toarray(), atol=atol, err_msg=key)
    neighbors = adata.uns["neighbors"]
    assert neighbors["connectivities_key"] == "connectivities" and neighbors["distances_key"] == "distances"
    assert neighbors["params"] == {
        "n_neighbors": 10, "method": "umap", "metric": "euclidean", "random_state": 0, "use_rep": use_rep,
    }


def test_scanpy_umap_runs_on_the_graph():
    # End-to-end check that scanpy's tools find the graph where piu_mosso writes it
    adata = pca_adata(n_cells=300)
    piu_mosso.pp.neighbors(adata)
    sc.tl.umap(adata, maxiter=5)
    assert adata.obsm["X_umap"].shape == (300, 2)


class _IVFChosen(Exception):
    """Raised in place of building an IVF index, to stop knn once it has chosen the approximate search."""


def test_default_exact_search_size_is_knn_device_default(monkeypatch):
    # exact_max_cells=None leaves the choice to knn, which reads its CPU default when called: lowering that default
    # below the number of cells makes the search approximate (IVF), stopped here when it builds its index
    def stop(*_args, **_kwargs):
        raise _IVFChosen

    monkeypatch.setattr(piu_mosso.faiss_knn, "CPU_EXACT_MAX_CELLS", 100)
    monkeypatch.setattr(faiss, "IndexIVFFlat", stop)
    with pytest.raises(_IVFChosen):
        piu_mosso.pp.neighbors(pca_adata(n_cells=300))


@pytest.mark.parametrize(
    ("adata", "kwargs", "message"),
    [
        (ad.AnnData(np.zeros((20, 3), dtype=np.float32)), {}, "not a key of .obsm"),
        (
            ad.AnnData(scipy.sparse.random(20, 30, density=0.3, format="csr", dtype=np.float32)),
            {"use_rep": "X"},
            "sparse",
        ),
        (pca_adata(n_cells=20), {"n_neighbors": 1}, "from 2 to the 20 cells"),
        (pca_adata(n_cells=20), {"n_neighbors": 21}, "from 2 to the 20 cells"),
        (pca_adata(n_cells=20), {"n_neighbors": 9.0}, "n_neighbors must be an integer"),
        (pca_adata(n_cells=20), {"random_state": np.random.RandomState(0)}, "random_state must be an integer"),
    ],
    ids=["missing_rep", "sparse_X", "one_neighbor", "too_many_neighbors", "float_neighbors", "random_state_object"],
)
def test_bad_input_raises(adata, kwargs, message):
    with pytest.raises(ValueError, match=message):
        piu_mosso.pp.neighbors(adata, **kwargs)
