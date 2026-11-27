"""
The eigensolver of the spectral start on a GPU: CuPy's LOBPCG on the graph's normalized Laplacian.

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


def lobpcg_eigenpairs(graph, sqrt_deg, X, tol, maxiter):
    """
    The eigenpairs LOBPCG finds for the smallest eigenvalues of L = I - D^-1/2 W D^-1/2 (W the graph, D its degrees)
    from the start vectors X, computed on the GPU in float64; returns host arrays (eigenvalues, eigenvectors,
    residuals), the residual of each pair being ||L v - lambda v|| / ||v||.

    As scipy's lobpcg on the CPU, except that L is applied as I - S with S = D^-1/2 W D^-1/2 stored once (W's values
    scaled in place; no sparse matrix products), and that CuPy's lobpcg does not warn when it stops short of tol, so
    the residuals are returned for the caller to check. Running out of GPU memory raises RuntimeError.
    """
    # CuPy is imported here, not at module level: piu_mosso imports without it (the caller has run check_cupy)
    import cupy
    from cupyx.scipy.sparse.linalg import LinearOperator, lobpcg

    try:
        S = _scaled_adjacency(graph, sqrt_deg)
        n = graph.shape[0]
        L = LinearOperator((n, n), matvec=lambda v: v - S @ v, matmat=lambda V: V - S @ V, dtype=cupy.float64)
        values, vectors = lobpcg(L, cupy.asarray(X, dtype=cupy.float64), largest=False, tol=tol, maxiter=maxiter)
        residuals = cupy.linalg.norm(L.matmat(vectors) - vectors * values, axis=0) / cupy.linalg.norm(vectors, axis=0)
    except cupy.cuda.memory.OutOfMemoryError as error:
        raise RuntimeError(
            f"The spectral start ran out of GPU memory on a connected component of {graph.shape[0]} cells: use "
            f"use_gpu=False, or start from a PCA with init='pca' (init_pos='pca' in piu_mosso.tl.umap)"
        ) from error
    return cupy.asnumpy(values), cupy.asnumpy(vectors), cupy.asnumpy(residuals)


def _scaled_adjacency(graph, sqrt_deg):
    """S = D^-1/2 W D^-1/2 on the GPU (float64 CSR): W's value at (i, j) divided by sqrt(deg_i deg_j)."""
    import cupy
    from cupyx.scipy.sparse import csr_matrix

    W = csr_matrix(graph.tocsr().astype(np.float64))
    inverse = cupy.asarray(1.0 / sqrt_deg, dtype=cupy.float64)
    # The row of every stored value: row r repeated once per value it stores (from the CSR row pointers)
    rows = cupy.repeat(cupy.arange(W.shape[0]), cupy.diff(W.indptr))
    W.data *= inverse[rows] * inverse[W.indices]
    return W
