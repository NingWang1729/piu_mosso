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

## Tests

```bash
bash tests/run_tests.sh [pytest arguments]   # in the active environment
```

## Third-party code

src/piu_mosso/spectral.py holds a copy of umap-learn's spectral layout (umap/spectral.py, version 0.5.12), changed where
marked "piu_mosso:". umap-learn is Copyright (c) 2017, Leland McInnes, under the BSD 3-Clause License; the full notice
is at the top of that file.
