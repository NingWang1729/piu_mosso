"""
The UMAP layout of a fuzzy neighbour graph, as umap-learn's simplicial_set_embedding computes it, in parallel.

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

import warnings

import numpy as np
import scipy.sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state
from umap.layouts import optimize_layout_euclidean
from umap.umap_ import INT32_MAX, INT32_MIN, find_ab_params, make_epochs_per_sample, noisy_scale_coords

from piu_mosso import gpu_layout
from piu_mosso._cupy import check_cupy
from piu_mosso.spectral import spectral_layout

# umap-learn's epochs: 500 for graphs of up to 10,000 cells, 200 above
SMALL_GRAPH_MAX_CELLS = 10_000
SMALL_GRAPH_EPOCHS = 500
LARGE_GRAPH_EPOCHS = 200

# umap-learn's pruning: with more than this many epochs, edges are dropped by n_epochs, else by the default
MIN_EPOCHS_FOR_PRUNING = 10

# The spectral and PCA starts are scaled to [-10, 10] with Gaussian noise of this standard deviation (umap-learn's)
START_MAX_COORD = 10.0
START_NOISE = 0.0001

# Every start is rescaled to [0, 10] in each dimension before the layout (umap-learn's)
LAYOUT_SCALE = 10.0

# An array start with duplicate rows gets noise of this fraction of the mean nearest-neighbour distance
DUPLICATE_JITTER = 0.001

INITS = ("spectral", "pca")


def umap_embedding(
    X,
    connectivities,
    *,
    n_components=2,
    min_dist=0.5,
    spread=1.0,
    n_epochs=None,
    init="spectral",
    negative_sample_rate=5,
    initial_alpha=1.0,
    gamma=1.0,
    parallel=True,
    random_state=0,
    use_gpu=False,
):
    """
    The UMAP layout of a fuzzy neighbour graph (e.g. umap_connectivities), with scanpy's and umap-learn's defaults.

    This is umap-learn's simplicial_set_embedding (as sc.tl.umap calls it) with two differences. The layout's
    stochastic gradient descent runs in parallel by default (umap-learn's optimize_layout_euclidean with
    parallel=True), and a failed spectral start raises instead of falling back to a random start. As in
    simplicial_set_embedding, each sampled edge moves both of its cells (move_other=True). The low-level default
    (False) halves the attraction and splits clusters into islands. With parallel=False, the result equals
    simplicial_set_embedding's for the same graph, start and random_state, with metric="euclidean" (umap-learn places
    separate connected components by that metric), except on connected components of about 1.66M cells or more,
    where this spectral start uses lobpcg: there umap-learn's would crash (ARPACK, below 2M cells) or stop at a looser
    tolerance (piu_mosso.spectral.LOBPCG_TOL). In parallel, threads race on shared coordinates, so results vary from
    run to run.

    Steps, as in umap-learn:
    1. Edges weaker than max weight / n_epochs are dropped (they would never be sampled); with 10 or fewer epochs,
       max weight / the default number of epochs.
    2. The start: the spectral layout of the graph (piu_mosso.spectral), or a PCA of X (TruncatedSVD if X is
       sparse), each scaled to [-10, 10] plus noise; or the given array, with small noise only if rows are
       duplicated.
    3. The start is rescaled to [0, 10] in each dimension, then optimized for n_epochs.

    Parameters
    ----------
    X
        The data the graph was built from, shape (n_cells, n_features). Used only to place the graph's connected
        components relative to each other in the spectral start, and for init="pca".
    connectivities
        Sparse (n_cells, n_cells) fuzzy graph; not changed.
    n_components
        Dimensions of the layout.
    min_dist, spread
        How tightly cells pack: the layout's attraction curve 1 / (1 + a d^(2b)) is fitted to these.
    n_epochs
        Epochs of the layout; default 500 for up to 10,000 cells, else 200. At least 1.
    init
        "spectral", "pca" (any case), or an array of shape (n_cells, n_components) with the start coordinates.
    negative_sample_rate
        Non-neighbours sampled (pushed away) per neighbour pulled.
    initial_alpha
        Learning rate at the first epoch; it falls linearly to 0.
    gamma
        Weight of repulsion relative to attraction.
    parallel
        Run the layout's epochs on numba's current number of threads (numba.get_num_threads()); not reproducible.
        No effect with use_gpu (a warning if False).
    random_state
        Seed (or numpy RandomState) for the start, its noise and the layout's sampling.
    use_gpu
        Run the spectral start (connected components of 10,000 cells or more) and the layout on the first visible
        CUDA GPU with CuPy (install the cuda12 or cuda13 extra). The start draws the same random numbers as on the
        CPU, and the GPU layout (piu_mosso.gpu_layout) follows umap-learn's per-edge updates and schedule, but its
        results never equal the CPU path's: updates are synchronous within each of 16 launches per epoch, a cell's
        summed move per launch is clipped to 4 times the learning rate, negative samples come from a counter hash, and
        steps are summed in fixed point. The layout is repeatable: from the same
        start, the same random_state gives the same layout on the same GPU model with the same CUDA and CuPy versions
        (a spectral start can itself vary where the graph has several connected components, as on the CPU). Missing
        CuPy raises ImportError and no usable GPU RuntimeError, 2^31 cells or more, more than 8 n_components or a
        negative negative_sample_rate ValueError, all before any work; too little GPU memory raises RuntimeError
        after pruning, before the start.

    Returns
    -------
    float32 array of shape (n_cells, n_components).
    """
    _check_shapes(X, connectivities)
    init = _canonicalize_init(init, connectivities.shape[0], n_components)
    _check_n_epochs(n_epochs)
    if use_gpu:
        _check_gpu(connectivities.shape[0], n_components, negative_sample_rate, parallel)
    # One random state for every draw, in umap-learn's order: the start, its noise, then the layout's seeds
    random_state = check_random_state(random_state)
    a, b = find_ab_params(spread, min_dist)

    graph, n_epochs = _pruned_graph(connectivities, n_epochs)
    if use_gpu:
        gpu_layout.check_memory(graph.shape[0], graph.nnz, n_components)
    start = _start(X, graph, init, n_components, random_state, use_gpu)
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
    layout_seeds = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    embedding = _rescale(start)
    if use_gpu:
        return gpu_layout.optimize_layout_gpu(
            embedding,
            graph.row,
            graph.col,
            n_epochs,
            epochs_per_sample,
            a,
            b,
            layout_seeds,
            gamma=gamma,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
        )
    return optimize_layout_euclidean(
        embedding,
        embedding,
        graph.row,
        graph.col,
        n_epochs,
        graph.shape[1],
        epochs_per_sample,
        a,
        b,
        layout_seeds,
        gamma=gamma,
        initial_alpha=initial_alpha,
        negative_sample_rate=negative_sample_rate,
        parallel=parallel,
        move_other=True,
    )


def default_n_epochs(n_cells):
    """umap-learn's number of epochs for a graph of n_cells: 500 up to 10,000 cells, else 200."""
    return SMALL_GRAPH_EPOCHS if n_cells <= SMALL_GRAPH_MAX_CELLS else LARGE_GRAPH_EPOCHS


def _check_shapes(X, connectivities):
    """
    Raise unless the graph is square with one row per cell of X. The layout indexes the start by the graph's cells in
    compiled code without bounds checks, so a mismatch would corrupt memory rather than raise there.
    """
    n_rows, n_cols = connectivities.shape
    if n_rows != n_cols or X.shape[0] != n_rows:
        raise ValueError(
            f"connectivities must be square with one row per cell of X, but has shape {connectivities.shape} while X "
            f"has {X.shape[0]} cells"
        )


def _canonicalize_init(init, n_cells, n_components):
    """
    init as "spectral" or "pca", or as an array of shape (n_cells, n_components); raise otherwise. The array keeps
    its dtype, as in umap-learn (the start is cast to float32 when rescaled).
    """
    if isinstance(init, str):
        if init.lower() not in INITS:
            raise ValueError(f"init must be one of {INITS} or an array of start coordinates, but is {init!r}")
        return init.lower()
    init = np.asarray(init)
    if init.shape != (n_cells, n_components):
        raise ValueError(
            f"An init array must have shape (n_cells, n_components) = {(n_cells, n_components)}, but has shape "
            f"{init.shape}"
        )
    return init


def _check_n_epochs(n_epochs):
    """Raise if n_epochs is given and is not a positive integer."""
    if n_epochs is None:
        return
    if not isinstance(n_epochs, (int, np.integer)) or isinstance(n_epochs, bool) or n_epochs < 1:
        raise ValueError(f"n_epochs must be a positive integer (or None for the default), but is {n_epochs!r}")


def _check_gpu(n_cells, n_components, negative_sample_rate, parallel):
    """
    The GPU path's checks, before any work: the layout's arguments (ValueError), CuPy and a GPU (ImportError,
    RuntimeError), and the kernel's compilation; warn that parallel=False has no effect there.
    """
    gpu_layout.check_layout_arguments(n_cells, n_components, negative_sample_rate)
    check_cupy()
    gpu_layout.compile_kernel()
    if not parallel:
        warnings.warn(
            "parallel=False has no effect with use_gpu=True: the GPU layout is always parallel (and repeatable from a "
            "given start), and it does not equal umap-learn's serial layout",
            stacklevel=3,
        )


def _pruned_graph(connectivities, n_epochs):
    """
    A COO copy of the graph without edges below max weight / n_epochs, and the number of epochs (the default if
    n_epochs is None). make_epochs_per_sample samples an edge every max / weight epochs, so a weaker edge would
    never be sampled. With 10 or fewer epochs, umap-learn prunes by the default number of epochs instead.
    """
    graph = connectivities.tocoo(copy=True)
    graph.sum_duplicates()
    default_epochs = default_n_epochs(graph.shape[0])
    n_epochs = default_epochs if n_epochs is None else n_epochs
    pruning_epochs = n_epochs if n_epochs > MIN_EPOCHS_FOR_PRUNING else default_epochs
    graph.data[graph.data < graph.data.max() / float(pruning_epochs)] = 0.0
    graph.eliminate_zeros()
    return graph, n_epochs


def _start(X, graph, init, n_components, random_state, use_gpu=False):
    """
    The start coordinates, before rescaling: spectral (on the GPU with use_gpu) or PCA (scaled to [-10, 10] plus
    noise), or the array.
    """
    if isinstance(init, np.ndarray):
        return _jitter_duplicates(init, random_state)
    if init == "spectral":
        start = spectral_layout(X, graph, n_components, random_state, use_gpu=use_gpu)
    else:
        pca = TruncatedSVD if scipy.sparse.issparse(X) else PCA
        start = pca(n_components=n_components, random_state=random_state).fit_transform(X).astype(np.float32)
    return noisy_scale_coords(start, random_state, max_coord=START_MAX_COORD, noise=START_NOISE)


def _jitter_duplicates(start, random_state):
    """
    The start array itself, or, if two cells share coordinates, the array plus Gaussian noise with a standard
    deviation of 0.001 x the mean distance to the nearest other cell (umap-learn's), so they do not stay together.
    """
    if np.unique(start, axis=0).shape[0] == start.shape[0]:
        return start
    distances, _ = KDTree(start).query(start, k=2)
    scale = DUPLICATE_JITTER * np.mean(distances[:, 1])
    return start + random_state.normal(scale=scale, size=start.shape).astype(np.float32)


def _rescale(start):
    """The start shifted and scaled to [0, 10] in each dimension, as a C-contiguous float32 array."""
    low, high = np.min(start, axis=0), np.max(start, axis=0)
    return (LAYOUT_SCALE * (start - low) / (high - low)).astype(np.float32, order="C")
