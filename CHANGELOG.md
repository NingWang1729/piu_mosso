# Changelog

All notable changes to piu_mosso are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-26

First release.

### Added

- `piu_mosso.knn`: the k nearest neighbours of every cell with FAISS. The search is exact up to a size that depends
  on the device (2\*\*16 cells on the CPU, 2\*\*21 on a GPU) and approximate above it (an IVF index whose number of
  probed lists is raised until a random sample of cells finds 99% of its exact neighbours). Distances are recomputed
  exactly. GPU search is opt-in (`use_gpu=True`, with `faiss-gpu-cu12` installed instead of `faiss-cpu`).
- `piu_mosso.umap_connectivities` and `piu_mosso.knn_distance_matrix`: scanpy's UMAP neighbour graph and distance
  matrix from k nearest neighbours.
- `piu_mosso.pp.neighbors`: FAISS neighbours and the UMAP graph, written where `sc.pp.neighbors` writes them, so
  scanpy's tools use them unchanged.
- The spectral start of the UMAP layout (`piu_mosso.spectral`): umap-learn's, but it uses lobpcg wherever ARPACK
  would overflow and crash on large graphs, and a failed eigensolver raises an error instead of falling back to a
  random start.
- `piu_mosso.umap_embedding`: the UMAP layout of a neighbour graph with scanpy's defaults, parallel by default, moving
  both cells of each sampled edge (`move_other=True`), from a spectral, PCA or given start.
- `piu_mosso.tl.umap`: the UMAP of the neighbour graph, written where `sc.tl.umap` writes it, so `sc.pl.umap` shows
  it.
- `piu_mosso.highly_variable_genes`: Seurat v3 highly variable genes over one or several AnnDatas, backed or in
  memory, equal to scanpy's for their concatenation up to floating-point rounding, without concatenating them or
  loading them into memory. Non-integer or negative counts raise an error where scanpy warns.

### Known issues

- With scanpy 1.13 and pandas 3 (the current releases; seen on Python 3.13), seven tests that compare piu_mosso with
  scanpy fail ([#7](https://github.com/NingWang1729/piu_mosso/issues/7)). scanpy is only a test dependency of
  piu_mosso; the failures are differences from scanpy's output:
  - `highly_variable_genes` with `batch_key` can rank tied genes half a rank differently from scanpy's `seurat_v3`
    (`highly_variable_rank`).
  - scanpy 1.13's `sc.tl.umap` records extra entries in `uns['umap']['params']` (`compatibility_layout`, and
    `random_state` even when 0) that `piu_mosso.tl.umap` does not write.

  With scanpy 1.11 and pandas 2, all tests pass. To be resolved in a later release.

[Unreleased]: https://github.com/NingWang1729/piu_mosso/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NingWang1729/piu_mosso/releases/tag/v0.1.0
