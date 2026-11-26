"""
The spectral start of the UMAP layout: umap-learn's spectral layout with two changes.

1. The eigensolver is lobpcg wherever ARPACK (scipy's eigsh) would overflow its 32-bit indices, not only from
   2,000,000 vertices on (see choose_eigensolver).
2. A failed eigensolver raises RuntimeError, naming init="pca" as the fix, instead of warning and falling back to a
   random start.

Everything else is umap-learn's, so for the same graph and random state the result equals umap-learn's wherever
umap-learn uses the same eigensolver and does not fail.

multi_component_layout, spectral_layout and _spectral_layout are copied from umap-learn 0.5.12 (umap/spectral.py),
changed only where marked "piu_mosso:". umap-learn is distributed under the BSD 3-Clause License:

Copyright (c) 2017, Leland McInnes
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The changes and the rest of this file are part of piu_mosso: fast single-cell utilities.
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

import warnings

import numpy as np

import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from umap.spectral import component_layout

# ARPACK indexes its work arrays with 32-bit signed integers
ARPACK_MAX_ENTRIES = 2**31 - 1
# umap-learn's own switch from eigsh to lobpcg
UMAP_LOBPCG_MIN_VERTICES = 2_000_000


def choose_eigensolver(n_vertices, dim):
    """
    The eigensolver for the spectral layout of a connected graph of n_vertices in dim dimensions: "lobpcg" from
    umap-learn's 2,000,000 vertices on, or wherever eigsh's Lanczos basis would overflow ARPACK; else "eigsh".

    eigsh keeps ncv = max(2 (dim + 1) + 1, int(sqrt(n_vertices))) Lanczos vectors of n_vertices values (the ncv
    umap-learn passes), one work array indexed with 32-bit integers. n_vertices x ncv passes 2^31 - 1 from about
    1,660,000 vertices on, where ARPACK crashes the process (a segmentation fault, not a Python error).
    """
    n_lanczos_vectors = max(2 * (dim + 1) + 1, int(np.sqrt(n_vertices)))
    if n_vertices >= UMAP_LOBPCG_MIN_VERTICES or n_vertices * n_lanczos_vectors > ARPACK_MAX_ENTRIES:
        return "lobpcg"
    return "eigsh"


def multi_component_layout(
    data,
    graph,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    init="random",
    tol=0.0,
    maxiter=0,
):
    """Specialised layout algorithm for dealing with graphs with many connected components.
    This will first find relative positions for the components by spectrally embedding
    their centroids, then spectrally embed each individual connected component positioning
    them according to the centroid embeddings. This provides a decent embedding of each
    component while placing the components in good relative positions to one another.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data -- required so we can generate centroids for each
        connected component of the graph.

    graph: sparse matrix
        The adjacency matrix of the graph to be embedded.

    n_components: int
        The number of distinct components to be layed out.

    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.

    dim: int
        The chosen embedding dimension.

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.

    init: string, either "random" or "tsvd"
        Indicates to initialize the eigensolver. Use "random" (the default) to
        use uniformly distributed random initialization; use "tsvd" to warm-start the
        eigensolver with singular vectors of the Laplacian associated to the largest
        singular values. This latter option also forces usage of the LOBPCG eigensolver;
        with the former, ARPACK's solver ``eigsh`` will be used for smaller Laplacians.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_samples, dim)
        The initial embedding of ``graph``.
    """

    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            data,
            n_components,
            component_labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim or component_graph.shape[0] <= dim + 1:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
        else:
            component_embedding = _spectral_layout(
                data=None,
                graph=component_graph,
                dim=dim,
                random_state=random_state,
                metric=metric,
                metric_kwds=metric_kwds,
                init=init,
                tol=tol,
                maxiter=maxiter,
            )
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )

    return result


def spectral_layout(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    tol=0.0,
    maxiter=0,
):
    """
    Given a graph compute the spectral embedding of the graph. This is
    simply the eigenvectors of the laplacian of the graph. Here we use the
    normalized laplacian.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data

    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.

    dim: int
        The dimension of the space into which to embed.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.

    piu_mosso: Raises RuntimeError if the eigensolver fails (umap-learn warns and starts at random instead).
    """
    return _spectral_layout(
        data=data,
        graph=graph,
        dim=dim,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        init="random",
        tol=tol,
        maxiter=maxiter,
    )


def _spectral_layout(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    init="random",
    method=None,
    tol=0.0,
    maxiter=0,
):
    """General implementation of the spectral embedding of the graph, derived as
    a subset of the eigenvectors of the normalized Laplacian of the graph. The numerical
    method for computing the eigendecomposition is chosen through heuristics.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data

    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.

    dim: int
        The dimension of the space into which to embed.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
        Used only if the multiple connected components are found in the
        graph.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'average'.
        Used only if the multiple connected components are found in the
        graph.

    init: string, either "random" or "tsvd"
        Indicates to initialize the eigensolver. Use "random" (the default) to
        use uniformly distributed random initialization; use "tsvd" to warm-start the
        eigensolver with singular vectors of the Laplacian associated to the largest
        singular values. This latter option also forces usage of the LOBPCG eigensolver;
        with the former, ARPACK's solver ``eigsh`` will be used for smaller Laplacians.

    method: string -- either "eigsh" or "lobpcg" -- or None
        Name of the eigenvalue computation method to use to compute the spectral
        embedding. If left to None (or empty string), as by default, the method is
        chosen from the number of vectors in play: larger vector collections are
        handled with lobpcg, smaller collections with eigsh. Method names correspond
        to SciPy routines in scipy.sparse.linalg.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.

    piu_mosso: Raises RuntimeError if the eigensolver fails (umap-learn warns and starts at random instead).
    """
    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        return multi_component_layout(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )

    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, graph.shape[0], graph.shape[0])
    L = I - D * graph * D
    if not scipy.sparse.issparse(L):
        L = np.asarray(L)

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    gen = (
        random_state
        if isinstance(random_state, (np.random.Generator, np.random.RandomState))
        else np.random.default_rng(seed=random_state)
    )
    if not method:
        # piu_mosso: lobpcg also wherever eigsh would overflow ARPACK (umap-learn: from 2,000,000 vertices on)
        method = choose_eigensolver(L.shape[0], dim)

    try:
        if init == "random":
            X = gen.normal(size=(L.shape[0], k))
        elif init == "tsvd":
            X = TruncatedSVD(
                n_components=k,
                random_state=random_state,
                # algorithm="arpack"
            ).fit_transform(L)
        else:
            raise ValueError(
                "The init parameter must be either 'random' or 'tsvd': "
                f"{init} is invalid."
            )
        # For such a normalized Laplacian, the first eigenvector is always
        # proportional to sqrt(degrees). We thus replace the first t-SVD guess
        # with the exact value.
        X[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)

        if method == "eigsh":
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=tol or 1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=maxiter or graph.shape[0] * 5,
            )
        elif method == "lobpcg":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    category=UserWarning,
                    message=r"(?ms).*not reaching the requested tolerance",
                    action="error",
                )
                eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                    L,
                    np.asarray(X),
                    largest=False,
                    tol=tol or 1e-4,
                    maxiter=maxiter or 5 * graph.shape[0],
                )
        else:
            raise ValueError("Method should either be None, 'eigsh' or 'lobpcg'")

        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except (scipy.sparse.linalg.ArpackError, UserWarning) as error:
        # piu_mosso: raise instead of warning and falling back to a random start (umap-learn)
        raise RuntimeError(
            f"The spectral start failed: the eigensolver ({method}) did not converge on a connected component of "
            f"{graph.shape[0]} cells, likely because of too small an eigengap. Start from a PCA instead with "
            f"init='pca' (init_pos='pca' in piu_mosso.tl.umap), or pass start coordinates"
        ) from error
