"""
Preprocessing functions that take an AnnData, in the style of scanpy's ``sc.pp``.

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

import scipy.sparse

from piu_mosso.faiss_knn import knn
from piu_mosso.graph import knn_distance_matrix, umap_connectivities


def neighbors(adata, n_neighbors=15, *, use_rep="X_pca", exact_max_cells=None, use_gpu=False, random_state=0):
    """
    The UMAP neighbour graph of the cells, written where sc.pp.neighbors writes it, so scanpy's tools (sc.tl.umap,
    sc.tl.leiden, ...) use it unchanged. With exact neighbours (up to exact_max_cells cells) the
    graph equals the one scanpy builds from the same exact neighbours (sc.pp.neighbors is exact below 8,192 cells):
    distances within float32 rounding, weights within 5e-5 (the tolerance of umap-learn's search for each cell's
    sigma).

    Parameters
    ----------
    adata
        The AnnData; changed in place.
    n_neighbors
        Neighbours per cell counting the cell itself, as in scanpy (so n_neighbors - 1 other cells). An integer from
        2 to the number of cells.
    use_rep
        The representation to search: a key of .obsm (default "X_pca", from sc.pp.pca), or "X" for a dense .X.
    exact_max_cells, use_gpu
        Passed to piu_mosso.knn. exact_max_cells=None (the default) is knn's default for the device: exact search up
        to 2**16 cells on the CPU and 2**21 with use_gpu.
    random_state
        Integer seed, passed to piu_mosso.knn and stored in .uns['neighbors'].

    Writes
    ------
    .obsp['distances']
        Each cell's n_neighbors - 1 neighbour distances (CSR, float32).
    .obsp['connectivities']
        The UMAP fuzzy graph (CSR, float32).
    .uns['neighbors']
        Keys and parameters, as scanpy writes them: {'connectivities_key', 'distances_key', 'params': {'n_neighbors',
        'method': 'umap', 'metric': 'euclidean', 'random_state', 'use_rep'}}.
    """
    _check_n_neighbors(n_neighbors, adata.n_obs)
    if not isinstance(random_state, numbers.Integral):
        raise ValueError(f"random_state must be an integer (it is stored in .uns), but is {random_state!r}")
    X = _representation(adata, use_rep)

    indices, distances = knn(
        X, n_neighbors - 1, exact_max_cells=exact_max_cells, use_gpu=use_gpu, random_state=random_state
    )
    adata.obsp["distances"] = knn_distance_matrix(indices, distances)
    adata.obsp["connectivities"] = umap_connectivities(indices, distances)
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {
            "n_neighbors": n_neighbors,
            "method": "umap",
            "metric": "euclidean",
            "random_state": random_state,
            "use_rep": use_rep,
        },
    }


def _check_n_neighbors(n_neighbors, n_cells):
    """Raise if n_neighbors (counting the cell itself) is not an integer from 2 to n_cells."""
    if not isinstance(n_neighbors, numbers.Integral) or isinstance(n_neighbors, bool):
        raise ValueError(f"n_neighbors must be an integer, but is {n_neighbors!r}")
    if not 2 <= n_neighbors <= n_cells:
        raise ValueError(
            f"n_neighbors counts the cell itself and must be from 2 to the {n_cells} cells, but is {n_neighbors}"
        )


def _representation(adata, use_rep):
    """The dense matrix to search: adata.X for use_rep "X", else adata.obsm[use_rep]."""
    if use_rep == "X":
        if scipy.sparse.issparse(adata.X):
            raise ValueError(
                "use_rep='X', but .X is sparse (cells x genes): search a PCA instead, e.g. sc.pp.pca(adata) and "
                "use_rep='X_pca'"
            )
        return adata.X
    if use_rep not in adata.obsm:
        raise ValueError(
            f"use_rep={use_rep!r} is not a key of .obsm (keys: {list(adata.obsm)}): compute it first, e.g. "
            f"sc.pp.pca(adata) for 'X_pca', or give another key"
        )
    return adata.obsm[use_rep]
