"""
The UMAP neighbour graph from k nearest neighbours, as scanpy and umap-learn build it.

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
import scipy.sparse
from umap.umap_ import fuzzy_simplicial_set


def umap_connectivities(knn_indices, knn_distances):
    """
    The UMAP fuzzy graph of k nearest neighbours, as sc.pp.neighbors(method="umap") computes it.

    umap-learn's fuzzy_simplicial_set, with scanpy's settings (set_op_mix_ratio 1, local_connectivity 1): each cell's
    neighbours get weights exp(-(d - rho) / sigma), where rho is the distance to the nearest neighbour and sigma is
    set so the weights sum to log2(k + 1); the graph is then made symmetric by the fuzzy union w + w^T - w * w^T.
    umap-learn and scanpy count the cell itself as its first neighbour (at distance 0, weight 0), so it is added here
    and n_neighbors is k + 1, as in scanpy.

    Parameters
    ----------
    knn_indices, knn_distances
        Arrays of shape (n_cells, k): each cell's k nearest neighbours other than itself, nearest first, and their
        distances, e.g. from piu_mosso.knn.

    Returns
    -------
    float32 CSR matrix of shape (n_cells, n_cells).
    """
    _check_knn(knn_indices, knn_distances)
    n_cells, k = knn_indices.shape
    cells = np.arange(n_cells)[:, None]
    with_self_indices = np.hstack([cells, knn_indices])
    with_self_distances = np.hstack([np.zeros((n_cells, 1), dtype=knn_distances.dtype), knn_distances])

    # Only the shape of X is used when the neighbours are given (scanpy passes the same empty matrix)
    connectivities, _, _ = fuzzy_simplicial_set(
        scipy.sparse.coo_matrix((n_cells, 1)),
        k + 1,
        None,
        None,
        knn_indices=with_self_indices,
        knn_dists=with_self_distances,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )
    return connectivities.tocsr().astype(np.float32)


def knn_distance_matrix(knn_indices, knn_distances):
    """
    The k nearest neighbour distances as a sparse matrix, scanpy's obsp['distances']: row i holds the distances to
    cell i's k neighbours (not itself). Distances of 0 (duplicate cells) are stored explicitly, as in scanpy.

    Parameters
    ----------
    knn_indices, knn_distances
        Arrays of shape (n_cells, k): each cell's k nearest neighbours other than itself, nearest first, and their
        distances, e.g. from piu_mosso.knn.

    Returns
    -------
    float32 CSR matrix of shape (n_cells, n_cells) with k entries per row.
    """
    _check_knn(knn_indices, knn_distances)
    n_cells, k = knn_indices.shape
    row_starts = np.arange(0, n_cells * k + 1, k)
    return scipy.sparse.csr_matrix(
        (knn_distances.astype(np.float32).ravel(), knn_indices.ravel(), row_starts), shape=(n_cells, n_cells)
    )


def _check_knn(knn_indices, knn_distances):
    """
    Raise if the neighbour arrays differ in shape, hold no neighbours or non-integer indices, name a cell outside
    [0, n_cells), or list a cell as its own neighbour (the self column of scanpy-style arrays, which this module
    adds itself).
    """
    if knn_indices.ndim != 2 or knn_indices.shape != knn_distances.shape or knn_indices.shape[1] < 1:
        raise ValueError(
            f"knn_indices and knn_distances must be arrays of the same shape (n_cells, k) with k >= 1, but have shapes "
            f"{knn_indices.shape} and {knn_distances.shape}"
        )
    if not np.issubdtype(knn_indices.dtype, np.integer):
        raise ValueError(f"knn_indices must hold integer cell indices, but has dtype {knn_indices.dtype}")
    n_cells = knn_indices.shape[0]
    if np.any((knn_indices < 0) | (knn_indices >= n_cells)):
        raise ValueError(
            f"knn_indices must name cells 0 to {n_cells - 1}, but has values outside that range (e.g. -1 for a "
            f"missing neighbour): compute the neighbours with piu_mosso.knn"
        )
    if np.any(knn_indices == np.arange(n_cells)[:, None]):
        raise ValueError(
            "knn_indices lists a cell as its own neighbour, but the neighbours must exclude the cell itself (it is "
            "added here): drop the self column, e.g. knn_indices[:, 1:] and knn_distances[:, 1:]"
        )
