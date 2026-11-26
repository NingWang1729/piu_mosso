# piu_mosso

*più mosso*: "more motion", faster. Fast single-cell utilities: FAISS nearest neighbours, a fast UMAP and Seurat v3
highly variable gene selection across several AnnDatas.

piu_mosso is free software, licensed under the GNU Lesser General Public License, version 3 or (at your option) any
later version (LGPL-3.0-or-later). The LGPL builds on the GNU General Public License, version 3; [LICENSE](LICENSE)
holds both texts.

## Install

```bash
pip install piu_mosso   # once released on PyPI
```

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

## Usage

```python
import piu_mosso
import scanpy as sc

piu_mosso.pp.neighbors(adata, use_rep="X_pca")   # FAISS neighbours and the UMAP graph, where scanpy keeps them
piu_mosso.tl.umap(adata)                         # spectral start, parallel layout; writes .obsm["X_umap"]
sc.pl.umap(adata, color="cell_type")
```

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
