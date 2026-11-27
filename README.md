# piu_mosso

*più mosso*: "more motion", faster. Fast single-cell utilities: FAISS nearest neighbours, a fast UMAP and Seurat v3
highly variable gene selection across several AnnDatas.

piu_mosso is free software, licensed under the GNU Lesser General Public License, version 3 or (at your option) any
later version (LGPL-3.0-or-later). The LGPL builds on the GNU General Public License, version 3;
[LICENSE](https://github.com/NingWang1729/piu_mosso/blob/master/LICENSE) holds both texts.

## Install

```bash
pip install piu-mosso
```

piu_mosso needs Python 3.11 or later.

For development, an editable install with the test dependencies (scanpy, pytest):

```bash
pip install -e ".[test]"
```

### GPU nearest neighbours

piu_mosso depends on `faiss-cpu`. FAISS's GPU build, `faiss-gpu-cu12`, installs the same `faiss` module, so the two
cannot share an environment (importing faiss fails when both are installed). For GPU search, swap them:

```bash
pip uninstall faiss-cpu
pip install faiss-gpu-cu12
```

Then pass `use_gpu=True` to `piu_mosso.knn` or `piu_mosso.pp.neighbors`. The UMAP layout runs on the CPU; a GPU
layout is planned.

## Usage

piu_mosso works on scanpy's AnnData objects and writes its results where scanpy does, so scanpy's other tools and
plots use them unchanged.

```python
import piu_mosso
import scanpy as sc

piu_mosso.pp.neighbors(adata, use_rep="X_pca")   # FAISS neighbours and the UMAP graph, where scanpy keeps them
piu_mosso.tl.umap(adata)                         # spectral start, parallel layout; writes .obsm["X_umap"]
sc.pl.umap(adata, color="cell_type")
```

Seurat v3 highly variable genes over several datasets together, from raw counts, without concatenating them or
loading them into memory:

```python
import anndata as ad

adatas = [ad.read_h5ad(path, backed="r") for path in ["part1.h5ad", "part2.h5ad"]]
hvg = piu_mosso.highly_variable_genes(adatas, n_top_genes=2000, batch_key="sample")
genes = hvg.index[hvg["highly_variable"]]
```

The lower-level functions are `piu_mosso.knn` (nearest neighbours of an array), `piu_mosso.umap_connectivities` and
`piu_mosso.knn_distance_matrix` (scanpy's graph and distances from them) and `piu_mosso.umap_embedding` (the layout of
a graph). Their docstrings give the details.

## Changes and citation

[CHANGELOG.md](https://github.com/NingWang1729/piu_mosso/blob/master/CHANGELOG.md) lists the changes in each release. To
cite piu_mosso, see [CITATION.cff](https://github.com/NingWang1729/piu_mosso/blob/master/CITATION.cff) (GitHub's "Cite
this repository" button reads it). Releases are archived on Zenodo under the concept DOI
[10.5281/zenodo.24576962](https://doi.org/10.5281/zenodo.24576962), which always resolves to the latest version.

## Tests

```bash
bash tests/run_tests.sh [pytest arguments]   # in the active environment
```

## Third-party code

src/piu_mosso/spectral.py holds a copy of umap-learn's spectral layout (umap/spectral.py, version 0.5.12), changed where
marked "piu_mosso:". umap-learn is Copyright (c) 2017, Leland McInnes, under the BSD 3-Clause License; the full notice
is at the top of that file.

src/piu_mosso/layout.py follows umap-learn's simplicial_set_embedding (umap/umap_.py) step by step, under the same
licence.

src/piu_mosso/tl.py chooses the representation of a neighbour graph as scanpy's _choose_representation does
(scanpy/tools/_utils.py), Copyright (c) 2025 scverse and Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab,
under the BSD 3-Clause License.

piu_mosso.highly_variable_genes follows scanpy's Seurat v3 selection (scanpy/preprocessing/_highly_variable_genes.py),
Copyright (c) 2025 scverse and Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab, under the BSD 3-Clause
License; the full notice is in piu_mosso/hvg.py.
