"""
Tools that take an AnnData, in the style of scanpy's ``sc.tl``.

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
from umap.umap_ import find_ab_params

from piu_mosso.layout import umap_embedding

# scanpy uses .X directly as the representation when it has at most this many genes (scanpy's settings.N_PCS)
SCANPY_N_PCS = 50

INIT_POS = ("spectral", "pca")


def umap(
    adata,
    *,
    min_dist=0.5,
    spread=1.0,
    n_components=2,
    maxiter=None,
    alpha=1.0,
    gamma=1.0,
    negative_sample_rate=5,
    init_pos="spectral",
    random_state=0,
    parallel=True,
    use_gpu=False,
):
    """
    The UMAP of the neighbour graph in .uns['neighbors'] (from piu_mosso.pp.neighbors or sc.pp.neighbors), written
    where sc.tl.umap writes it, so sc.pl.umap shows it. The arguments are a subset of sc.tl.umap's (no a, b, copy,
    neighbors_key, key_added or method; init_pos without obsm keys or "paga"), plus parallel and use_gpu.

    The layout is piu_mosso.umap_embedding: umap-learn's, run in parallel by default, and a failed spectral start
    raises an error naming init_pos="pca" instead of silently starting at random. With parallel=False the result
    equals sc.tl.umap's for the same graph, start and random_state, for graphs built with the Euclidean metric
    (piu_mosso places a graph's separate connected components by Euclidean distance).

    Parameters
    ----------
    adata
        The AnnData; changed in place.
    min_dist, spread, n_components, negative_sample_rate, gamma
        As in piu_mosso.umap_embedding and sc.tl.umap.
    maxiter
        Epochs of the layout (umap_embedding's n_epochs); default 500 for up to 10,000 cells, else 200.
    alpha
        Learning rate at the first epoch (umap_embedding's initial_alpha).
    init_pos
        "spectral", "pca", or an array of shape (n_cells, n_components), cast to float32 as sc.tl.umap does.
    random_state
        Seed (or numpy RandomState) for the start and the layout.
    parallel
        Run the layout on numba's current number of threads (numba.get_num_threads(); not reproducible), or
        serially. No effect with use_gpu (a warning if False).
    use_gpu
        Run the spectral start and the layout on a CUDA GPU (see piu_mosso.umap_embedding; needs the cuda12 or
        cuda13 extra). Repeatable for a random_state from the same start (see piu_mosso.umap_embedding), but not
        equal to the CPU result.

    Writes
    ------
    .obsm['X_umap']
        The layout, float32 (n_cells, n_components).
    .uns['umap']
        {'params': {'a', 'b'}} (plus 'random_state' if it is not 0), as scanpy writes it, plus 'use_gpu': True for
        a GPU layout.
    """
    init_pos = _canonicalize_init_pos(init_pos)
    neighbors = _get_neighbors(adata)
    params = neighbors["params"]
    _warn_non_euclidean_metric(params)
    X = _representation(adata, params.get("use_rep"), params.get("n_pcs"))

    adata.obsm["X_umap"] = umap_embedding(
        X,
        adata.obsp[neighbors.get("connectivities_key", "connectivities")],
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        n_epochs=maxiter,
        init=init_pos,
        negative_sample_rate=negative_sample_rate,
        initial_alpha=alpha,
        gamma=gamma,
        parallel=parallel,
        random_state=random_state,
        use_gpu=use_gpu,
    )
    a, b = find_ab_params(spread, min_dist)
    adata.uns["umap"] = {"params": {"a": a, "b": b}}
    if random_state != 0:
        adata.uns["umap"]["params"]["random_state"] = random_state
    # Recorded only for GPU layouts, so CPU results keep scanpy's keys
    if use_gpu:
        adata.uns["umap"]["params"]["use_gpu"] = True


def _canonicalize_init_pos(init_pos):
    """init_pos as "spectral" or "pca", or as a float32 array (cast as sc.tl.umap casts it); raise otherwise."""
    if isinstance(init_pos, str):
        if init_pos.lower() not in INIT_POS:
            raise ValueError(
                f"init_pos must be one of {INIT_POS} or an array of start coordinates, but is {init_pos!r} "
                f"(piu_mosso.tl.umap takes no .obsm keys or 'paga': pass adata.obsm[key] as an array)"
            )
        return init_pos.lower()
    return np.asarray(init_pos, dtype=np.float32)


def _get_neighbors(adata):
    """adata.uns['neighbors'], after checking it holds a UMAP graph."""
    if "neighbors" not in adata.uns or "params" not in adata.uns["neighbors"]:
        raise ValueError(
            "adata has no .uns['neighbors'] with its 'params': run piu_mosso.pp.neighbors (or sc.pp.neighbors) first"
        )
    neighbors = adata.uns["neighbors"]
    method = neighbors["params"].get("method", "umap")
    if method != "umap":
        raise ValueError(
            f"The neighbour graph was built with method={method!r}, but UMAP needs method='umap': rerun "
            f"piu_mosso.pp.neighbors (or sc.pp.neighbors with method='umap')"
        )
    return neighbors


def _warn_non_euclidean_metric(params):
    """
    Warn if the graph used another metric: the spectral start places separate connected components by Euclidean
    distance, so on a graph with several components the layout differs from sc.tl.umap's.
    """
    metric = params.get("metric", "euclidean")
    if metric != "euclidean":
        warnings.warn(
            f"The neighbour graph used metric={metric!r}, but piu_mosso places its separate connected components "
            f"by Euclidean distance, so if it has several, the layout differs from sc.tl.umap's",
            stacklevel=3,
        )


def _representation(adata, use_rep, n_pcs):
    """
    The representation the graph was built on, chosen as scanpy's _choose_representation does. It is used only to
    place the graph's connected components and for init_pos="pca".
    """
    if use_rep is None and n_pcs == 0:
        use_rep = "X"
    if use_rep is None:
        # scanpy's default: .X when it has few genes, else the PCA
        if adata.n_vars <= SCANPY_N_PCS:
            return adata.X
        use_rep = "X_pca"
    if use_rep == "X":
        return adata.X
    if use_rep not in adata.obsm:
        raise ValueError(
            f"The neighbour graph was built on .obsm[{use_rep!r}], which adata no longer has: recompute it "
            f"(e.g. sc.pp.pca) or rerun the neighbours"
        )
    representation = adata.obsm[use_rep]
    if n_pcs is not None and n_pcs > representation.shape[1]:
        raise ValueError(
            f"The neighbour graph used n_pcs={n_pcs} dimensions of .obsm[{use_rep!r}], which now has only "
            f"{representation.shape[1]}: recompute it or rerun the neighbours"
        )
    return representation if n_pcs is None else representation[:, :n_pcs]
