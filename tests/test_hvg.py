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
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse

from piu_mosso import highly_variable_genes


def counts_adata(n_cells=600, n_genes=300, n_batches=3, sparse=True, seed=0):
    """Negative binomial counts with gene-specific means and batch-specific scales, and a 'batch' column."""
    rng = np.random.default_rng(seed)
    gene_means = rng.lognormal(mean=0.0, sigma=1.5, size=n_genes)
    batches = rng.integers(n_batches, size=n_cells)
    batch_scales = rng.uniform(0.5, 2.0, size=n_batches)
    means = gene_means[None, :] * batch_scales[batches, None]
    counts = rng.negative_binomial(n=2, p=2 / (2 + means)).astype(np.float32)
    adata = ad.AnnData(
        X=scipy.sparse.csr_matrix(counts) if sparse else counts,
        obs=pd.DataFrame(
            {"batch": pd.Categorical([f"b{b}" for b in batches])}, index=[f"c{i}" for i in range(n_cells)]
        ),
        var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]),
    )
    return adata


def scanpy_hvg(adata, **kwargs):
    return sc.pp.highly_variable_genes(adata, flavor="seurat_v3", inplace=False, **kwargs)


def assert_same_hvgs(result, expected):
    """Same genes, columns (order and dtypes) and selection; floats within rounding of another summation order."""
    pd.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-6)


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_equals_scanpy_seurat_v3(sparse, batch_key):
    adata = counts_adata(sparse=sparse)
    result = highly_variable_genes(adata, n_top_genes=50, batch_key=batch_key)
    assert_same_hvgs(result, scanpy_hvg(adata, n_top_genes=50, batch_key=batch_key))


def test_clipped_outliers_equal_scanpy():
    # One huge count per gene in the first cells, far above mean + sqrt(n) x std, so the clipping changes the result
    adata = counts_adata(sparse=False)
    adata.X[np.arange(20), np.arange(20)] = 10_000
    result = highly_variable_genes(adata, n_top_genes=50, batch_key="batch")
    assert_same_hvgs(result, scanpy_hvg(adata, n_top_genes=50, batch_key="batch"))


@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_list_of_adatas_equals_their_concatenation(batch_key):
    adata = counts_adata()
    # Uneven parts; every batch label appears in several parts and counts as one batch
    parts = [adata[:100].copy(), adata[100:350].copy(), adata[350:].copy()]
    result = highly_variable_genes(parts, n_top_genes=50, batch_key=batch_key, chunk_size=64)
    assert_same_hvgs(result, scanpy_hvg(adata, n_top_genes=50, batch_key=batch_key))


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_backed_adatas_equal_in_memory(tmp_path, sparse):
    # Files go to pytest's tmp_path; the chunk size does not divide the cell numbers, so chunks cross part ends
    adata = counts_adata(sparse=sparse)
    paths = [tmp_path / "part0.h5ad", tmp_path / "part1.h5ad"]
    adata[:250].copy().write_h5ad(paths[0])
    adata[250:].copy().write_h5ad(paths[1])
    backed = [ad.read_h5ad(path, backed="r") for path in paths]
    result = highly_variable_genes(backed, n_top_genes=50, batch_key="batch", chunk_size=97)
    in_memory = highly_variable_genes(adata, n_top_genes=50, batch_key="batch")
    pd.testing.assert_frame_equal(result, in_memory, rtol=1e-9)


def test_genes_ranks_within_the_gene_set():
    adata = counts_adata()
    # Given out of order; the result follows var_names
    genes = [f"g{j}" for j in range(250, 0, -2)]
    result = highly_variable_genes(adata, n_top_genes=30, batch_key="batch", genes=genes)
    expected = scanpy_hvg(adata[:, sorted(genes, key=lambda g: int(g[1:]))].copy(), n_top_genes=30, batch_key="batch")
    assert_same_hvgs(result, expected)


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_result_does_not_change_the_adatas(sparse):
    # float64 counts in one chunk, where reading could hand back the data itself rather than a copy
    adata = counts_adata(sparse=sparse)
    adata.X = adata.X.astype(np.float64)
    before = adata.copy()
    highly_variable_genes(adata, n_top_genes=50, chunk_size=10_000)
    assert adata.var.columns.empty and "hvg" not in adata.uns
    np.testing.assert_array_equal(adata.X.toarray() if sparse else adata.X, before.X.toarray() if sparse else before.X)


def adata_with(change):
    """counts_adata with one change applied (a function of the AnnData returning the AnnData or a list)."""
    return change(counts_adata())


def set_value(value):
    def change(adata):
        adata.X = adata.X.toarray()
        adata.X[0, 0] = value
        return adata
    return change


def rename_genes(adata):
    other = adata.copy()
    other.var_names = [f"x{j}" for j in range(adata.n_vars)]
    return [adata, other]


def missing_batch_label(adata):
    adata.obs["batch"] = adata.obs["batch"].astype(object)
    adata.obs.iloc[0, 0] = None
    return adata


def single_cell_batch(adata):
    adata.obs["batch"] = adata.obs["batch"].astype(object)
    adata.obs.iloc[0, 0] = "alone"
    return adata


def mixed_label_types(adata):
    other = adata.copy()
    other.obs["batch"] = 1
    return [adata, other]


@pytest.mark.parametrize(
    ("change", "kwargs", "message"),
    [
        (set_value(0.5), {}, "non-integer"),
        (set_value(-1.0), {}, "negative"),
        (rename_genes, {}, "different var_names"),
        (lambda adata: adata, {"batch_key": "sample"}, "not a column"),
        (missing_batch_label, {"batch_key": "batch"}, "missing"),
        (single_cell_batch, {"batch_key": "batch"}, "only 1 cell"),
        (lambda adata: adata, {"genes": ["g1", "nope"]}, "not in the data"),
        (lambda adata: adata, {"genes": ["g1", "g1", "g2"]}, "more than once"),
        (lambda adata: adata, {"n_top_genes": 0}, "n_top_genes"),
        (lambda adata: adata, {"n_top_genes": True}, "n_top_genes"),
        (lambda adata: adata, {"chunk_size": 0}, "chunk_size"),
        (lambda adata: adata, {"span": 0.0}, "span"),
        (lambda adata: [], {}, "at least one AnnData"),
        (mixed_label_types, {"batch_key": "batch"}, "mix types"),
    ],
    ids=[
        "non_integer", "negative", "different_genes", "unknown_batch_key", "missing_batch_label", "one_cell_batch",
        "unknown_genes", "duplicate_genes", "n_top_genes", "n_top_genes_bool", "chunk_size", "span", "empty_list",
        "mixed_label_types",
    ],
)
def test_bad_input_raises(change, kwargs, message):
    with pytest.raises(ValueError, match=message):
        highly_variable_genes(adata_with(change), **kwargs)


def test_too_few_varying_genes_in_a_batch_raises():
    # skmisc's LOESS crashes the process below 4 points, so this must be an error instead
    adata = counts_adata(n_genes=3)
    with pytest.raises(ValueError, match="nonzero variance"):
        highly_variable_genes(adata, n_top_genes=2)
