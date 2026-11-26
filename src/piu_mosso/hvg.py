"""
Seurat v3 highly variable genes over one or several AnnDatas, read in chunks of cells.

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
import numpy as np
import pandas as pd
import scipy.sparse
from skmisc.loess import loess

# skmisc's LOESS crashes the process (segmentation fault) instead of raising when given fewer points than this
MIN_LOESS_GENES = 4

# Error messages name this many example genes, enough to spot a naming mismatch without a long message
N_EXAMPLE_GENES = 5


def highly_variable_genes(adatas, *, n_top_genes=2000, batch_key=None, span=0.3, chunk_size=10_000, genes=None):
    """
    Seurat v3 highly variable genes of the cells of one or several AnnDatas taken together, as
    ``sc.pp.highly_variable_genes(flavor="seurat_v3", inplace=False)`` gives for their concatenation, without
    concatenating them or loading them into memory.

    Every AnnData, in memory or backed (``read_h5ad(path, backed="r")``), is read in contiguous chunks of chunk_size
    cells, twice: the first pass sums the counts and their squares per batch, the second the counts clipped as in
    Seurat v3. Results differ from scanpy's only by floating-point rounding (another summation order), except that
    non-integer or negative counts raise an error (scanpy warns), and so does a batch with fewer than 4 varying
    genes (where skmisc's LOESS crashes the process).

    Seurat v3, per batch (all cells form one batch without batch_key): fit LOESS (degree 2, span) of log10 variance
    on log10 mean over the genes with nonzero variance; clip each count at mean + sqrt(n_cells) x the fitted standard
    deviation; the normalized variance of a gene is the variance of its clipped counts around its mean, divided by
    the fitted variance. Genes are ranked by normalized variance within each batch; a gene's rank is its median rank
    over the batches where it is among the top n_top_genes, and genes are selected by that rank, ties broken by the
    number of such batches.

    Parameters
    ----------
    adatas
        An AnnData or a list of them, with raw counts (non-negative integers, of any dtype) in .X and the same
        var_names in the same order.
    n_top_genes
        Number of highly variable genes to select.
    batch_key
        Column of .obs with each cell's batch. The same label in two AnnDatas is one batch.
    span
        Fraction of the genes used for each local LOESS fit, in (0, 1].
    chunk_size
        Cells read at a time. A dense chunk takes up to 2 x chunk_size x n_genes float64 values at a time. Backed
        sparse data should be CSR: rows of a backed CSC matrix are slow to read.
    genes
        Rank only these genes (e.g. a gene set): the statistics, trend and ranks use only their counts, as if each
        AnnData held only them. Ranking within a set can select different genes than ranking all genes and keeping
        those in the set. Every gene must be in var_names, once.

    Returns
    -------
    A DataFrame indexed by gene (var_names order; with genes, only those), with scanpy's columns: means, variances,
    gene_name, highly_variable_rank (NaN outside the top genes), variances_norm (averaged over batches),
    highly_variable and, with batch_key, highly_variable_nbatches. The AnnDatas are not changed.
    """
    adatas = _as_adata_list(adatas)
    _check_options(n_top_genes, span, chunk_size)
    _check_same_var_names(adatas)
    columns = _gene_columns(adatas[0].var_names, genes)
    batches, labels = _batch_codes(adatas, batch_key)
    n_cells = np.bincount(np.concatenate(batches), minlength=len(labels))
    _check_batch_sizes(n_cells, labels, batch_key)

    # Pass 1: mean and variance of each gene in each batch, and the LOESS fit of the variance on the mean
    sums, squares = _batch_sums(adatas, batches, len(labels), columns, chunk_size, clip=None)
    means, variances = _mean_var(n_cells, sums, squares)
    fitted_std = np.stack([
        _fitted_std(means[b], variances[b], span, "all cells" if batch_key is None else f"batch {label!r}")
        for b, label in enumerate(labels)
    ])

    # Pass 2: Seurat v3 clips each count at mean + sqrt(n_cells) x fitted std, then takes the variance of the
    # clipped counts c around the (unclipped) mean m, in units of the fitted variance s^2:
    #   sum_i (c_i - m)^2 / ((n - 1) s^2) = (n m^2 + sum c^2 - 2 m sum c) / ((n - 1) s^2)
    clip = means + np.sqrt(n_cells)[:, None] * fitted_std
    clipped_sums, clipped_squares = _batch_sums(adatas, batches, len(labels), columns, chunk_size, clip=clip)
    n = n_cells[:, None]
    norm_variances = (n * means**2 + clipped_squares - 2 * means * clipped_sums) / ((n - 1) * fitted_std**2)

    # The whole data's mean and variance, from the batches' sums
    total_means, total_variances = _mean_var(
        n_cells.sum(keepdims=True), sums.sum(axis=0, keepdims=True), squares.sum(axis=0, keepdims=True)
    )
    var_names = adatas[0].var_names if columns is None else adatas[0].var_names[columns]
    result = _rank_genes(var_names, total_means[0], total_variances[0], norm_variances, n_top_genes)
    if batch_key is None:
        result = result.drop(columns="highly_variable_nbatches")
    return result


def _as_adata_list(adatas):
    """adatas as a list of AnnDatas (one AnnData becomes a list of one)."""
    if isinstance(adatas, ad.AnnData):
        return [adatas]
    adatas = list(adatas)
    if len(adatas) == 0:
        raise ValueError("adatas must hold at least one AnnData, but is empty")
    for index, adata in enumerate(adatas):
        if not isinstance(adata, ad.AnnData):
            raise ValueError(f"adatas must be AnnDatas, but item {index} is a {type(adata).__name__}")
    return adatas


def _check_options(n_top_genes, span, chunk_size):
    """Raise if n_top_genes or chunk_size is not a positive integer, or span is not in (0, 1]."""
    for name, value in [("n_top_genes", n_top_genes), ("chunk_size", chunk_size)]:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, but is {value!r}")
    if not 0 < span <= 1:
        raise ValueError(f"span is the fraction of genes in each LOESS fit and must be in (0, 1], but is {span!r}")


def _check_same_var_names(adatas):
    """Raise if the AnnDatas do not have the same genes in the same order, since their sums are combined by position."""
    for index, adata in enumerate(adatas[1:], start=1):
        if not adata.var_names.equals(adatas[0].var_names):
            raise ValueError(
                f"adata {index} has different var_names (or a different gene order) than adata 0: subset each to the "
                f"same genes in the same order, e.g. adata[:, genes]"
            )


def _gene_columns(var_names, genes):
    """The positions in var_names of the genes to rank, in var_names order, or None for all genes."""
    if genes is None:
        return None
    genes = pd.Index(genes)
    repeated = genes[genes.duplicated()]
    if len(repeated) > 0:
        raise ValueError(f"genes lists {list(repeated.unique()[:N_EXAMPLE_GENES])} more than once: give each gene once")
    unknown = genes[~genes.isin(var_names)]
    if len(unknown) > 0:
        raise ValueError(
            f"{len(unknown)} of the {len(genes)} genes are not in the data's var_names, e.g. "
            f"{list(unknown[:N_EXAMPLE_GENES])}: give only genes of var_names"
        )
    return np.flatnonzero(var_names.isin(genes))


def _batch_codes(adatas, batch_key):
    """
    Each AnnData's cells as integer batch codes (one array per AnnData), and the batch labels in sorted order (as
    np.unique sorts them, like scanpy). Without batch_key all cells are batch 0.
    """
    if batch_key is None:
        return [np.zeros(adata.n_obs, dtype=np.int64) for adata in adatas], np.array([0])
    for index, adata in enumerate(adatas):
        if batch_key not in adata.obs:
            raise ValueError(f"batch_key {batch_key!r} is not a column of adata {index}.obs")
        n_missing = adata.obs[batch_key].isna().sum()
        if n_missing > 0:
            raise ValueError(
                f"adata {index}.obs[{batch_key!r}] has {n_missing} missing batch labels: fill them or remove those "
                f"cells"
            )
    try:
        all_labels = np.concatenate([adata.obs[batch_key].to_numpy() for adata in adatas])
        labels, codes = np.unique(all_labels, return_inverse=True)
    except TypeError as error:
        raise ValueError(
            f"The batch labels in .obs[{batch_key!r}] mix types that cannot be sorted together (e.g. numbers and "
            f"strings): use one type, e.g. adata.obs[{batch_key!r}] = adata.obs[{batch_key!r}].astype(str)"
        ) from error
    ends = np.cumsum([adata.n_obs for adata in adatas])[:-1]
    return np.split(codes, ends), labels


def _check_batch_sizes(n_cells, labels, batch_key):
    """Raise if a batch has fewer than 2 cells, since its sample variance is undefined."""
    too_small = np.flatnonzero(n_cells < 2)
    if len(too_small) > 0:
        where = "The data" if batch_key is None else f"Batch {labels[too_small[0]]!r} of batch_key {batch_key!r}"
        raise ValueError(
            f"{where} has only {n_cells[too_small[0]]} cell, but a variance needs at least 2: remove or merge it"
        )


def _batch_sums(adatas, batches, n_batches, columns, chunk_size, clip):
    """
    Per batch and gene, the sum of the counts and the sum of their squares (float64 arrays of shape (n_batches,
    n_genes)), over all AnnDatas, read chunk_size cells at a time. With clip (shape (n_batches, n_genes)), each count
    is first clipped at its batch's value for its gene; without it, the counts are checked to be raw counts.
    """
    n_genes = adatas[0].n_vars if columns is None else len(columns)
    sums = np.zeros((n_batches, n_genes))
    squares = np.zeros((n_batches, n_genes))
    for index, adata in enumerate(adatas):
        for start in range(0, adata.n_obs, chunk_size):
            counts = _read_chunk(adata, start, chunk_size, columns)
            codes = batches[index][start:start + chunk_size]
            if clip is None:
                _check_raw_counts(counts, index)
            else:
                counts = _clip(counts, clip, codes)

            # A (batches x cells) indicator matrix sums the chunk's cells into their batches in one product
            indicator = scipy.sparse.csr_matrix(
                (np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(n_batches, len(codes))
            )
            sums += _dense(indicator @ counts)
            # Squared in place: the chunk is already a float64 copy, and it is not used again
            if scipy.sparse.issparse(counts):
                np.square(counts.data, out=counts.data)
            else:
                np.square(counts, out=counts)
            squares += _dense(indicator @ counts)
    return sums, squares


def _read_chunk(adata, start, chunk_size, columns):
    """
    A float64 copy of the counts of cells start to start + chunk_size (CSR if sparse), only the given gene columns
    (None for all). The rows are read contiguously, which is fast on disk for a backed AnnData, and the genes selected
    in memory: a backed view (adata[:, genes]) would load the whole matrix.
    """
    counts = adata.X[start:start + chunk_size]
    # Always a copy, since the chunk is squared and clipped in place (a slice of float64 data can be a view)
    if scipy.sparse.issparse(counts):
        counts = scipy.sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    else:
        counts = np.array(counts, dtype=np.float64, copy=True)
    return counts if columns is None else counts[:, columns]


def _check_raw_counts(counts, index):
    """Raise if a chunk of adata index has negative or non-integer values: Seurat v3 needs raw counts."""
    values = counts.data if scipy.sparse.issparse(counts) else counts
    if np.any(values < 0):
        raise ValueError(
            f"adata {index}.X has negative values, but Seurat v3 needs raw counts: put the raw counts in .X"
        )
    if np.any(values != np.round(values)):
        raise ValueError(
            f"adata {index}.X has non-integer values, but Seurat v3 needs raw counts (not normalized or log values): "
            f"put the raw counts in .X"
        )


def _clip(counts, clip, codes):
    """The chunk's counts, each clipped at clip[its cell's batch, its gene]. Zeros stay zero (clip is positive)."""
    # Clipped in place: the chunk is a float64 copy of the data
    if not scipy.sparse.issparse(counts):
        return np.minimum(counts, clip[codes], out=counts)
    # The cell (row) of each stored value, from the CSR row pointers
    rows = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    np.minimum(counts.data, clip[codes[rows], counts.indices], out=counts.data)
    return counts


def _dense(matrix):
    """A sparse or dense product as a dense array."""
    return matrix.toarray() if scipy.sparse.issparse(matrix) else np.asarray(matrix)


def _mean_var(n_cells, sums, squares):
    """Mean and sample variance (ddof 1) per row and gene from the number of cells, sums and sums of squares."""
    n = n_cells[:, None]
    means = sums / n
    # sum (x - m)^2 = sum x^2 - n m^2; the same cancellation as scanpy's mean/variance, kept small by float64 sums
    variances = (squares - n * means**2) / (n - 1)
    return means, variances


def _fitted_std(means, variances, span, cells):
    """
    The standard deviation of each gene predicted by Seurat v3's trend: LOESS of log10 variance on log10 mean over
    the genes with nonzero variance; 1 (log10 variance 0) for the others, as scanpy sets it. cells names the batch.
    """
    varying = variances > 0
    n_varying = np.count_nonzero(varying)
    if n_varying < MIN_LOESS_GENES:
        raise ValueError(
            f"Only {n_varying} genes have nonzero variance in {cells}, but the mean-variance trend needs at least "
            f"{MIN_LOESS_GENES}: remove or merge that batch, or rank more genes"
        )
    model = loess(np.log10(means[varying]), np.log10(variances[varying]), span=span, degree=2)
    model.fit()
    log_variances = np.zeros_like(means)
    log_variances[varying] = model.outputs.fitted_values
    return np.sqrt(10**log_variances)


# _rank_genes follows scanpy's seurat_v3 ranking (scanpy/preprocessing/_highly_variable_genes.py), distributed under
# the BSD 3-Clause License:
#
# Copyright (c) 2025 scverse
# Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def _rank_genes(var_names, means, variances, norm_variances, n_top_genes):
    """
    The result table: each gene's rank by normalized variance within each batch (0 = most variable), its median
    rank over the batches where it is in the top n_top_genes, how many such batches, and the selection (seurat_v3's
    order: median rank ascending, then number of batches descending), all as scanpy computes them.

    Follows scanpy's seurat_v3 ranking (BSD-3; notice above).
    """
    # Argsort twice gives ranks
    ranks = np.argsort(np.argsort(-norm_variances, axis=1), axis=1).astype(np.float32)
    n_batches_top = np.sum(ranks < n_top_genes, axis=0)
    ranks[ranks >= n_top_genes] = np.nan
    median_ranks = np.ma.median(np.ma.masked_invalid(ranks), axis=0).filled(np.nan)

    result = pd.DataFrame(index=var_names)
    result["means"] = means
    result["variances"] = variances
    result["gene_name"] = result.index
    result["highly_variable_nbatches"] = n_batches_top
    result["highly_variable_rank"] = median_ranks
    result["variances_norm"] = norm_variances.mean(axis=0)
    order = result.sort_values(
        ["highly_variable_rank", "highly_variable_nbatches"], ascending=[True, False], na_position="last"
    ).index
    result["highly_variable"] = False
    result.loc[order[:n_top_genes], "highly_variable"] = True
    return result
