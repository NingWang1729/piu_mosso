# Changelog

All notable changes to piu_mosso are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- The spectral start's LOBPCG (used on the CPU for connected components of about 1.66M cells or more) stops at a
  tolerance of 1e-5 by default instead of umap-learn's 1e-4, as on the GPU. On large graphs, where eigenvalues lie
  close together, 1e-4 could return a mix of two eigenvectors or the wrong one, and the layout's clusters came out
  less coherent. Starts of such components change and take longer to compute (`spectral_layout(..., tol=1e-4)` gives
  the old behaviour). The constant is now `piu_mosso.spectral.LOBPCG_TOL`; `GPU_SPECTRAL_TOL` still works but warns
  (DeprecationWarning).

## [0.2.0] - 2026-11-27

### Added

- GPU UMAP: `use_gpu=True` in `piu_mosso.tl.umap` and `piu_mosso.umap_embedding` runs the spectral start and the
  layout on a CUDA GPU with CuPy (extras `cuda12` and `cuda13`; install one). The layout (`piu_mosso.gpu_layout`)
  follows umap-learn's per-edge updates and schedule with both cells of each edge moving; from the same start it is
  repeatable for a `random_state` on the same GPU and software versions, and it differs from the CPU layout
  (synchronous updates within each of 16 launches per epoch, a cell's summed move per launch clipped to 4 times the
  learning rate, negative samples from a counter hash, steps summed in fixed point). Missing CuPy, no usable GPU or
  unsupported arguments raise before any work, too little GPU memory before the start. `tl.umap` records `use_gpu`
  in `.uns['umap']['params']` for GPU layouts.
- `use_gpu` in `piu_mosso.spectral.spectral_layout`: connected components of 10,000 cells or more are solved with
  CuPy's LOBPCG on the GPU, at a default tolerance of 1e-5 (the CPU's is 1e-4): the same eigenvectors as on the CPU
  up to sign, within the solvers' tolerances.

### Known issues

- As in 0.1.0, seven tests that compare piu_mosso with scanpy fail with scanpy 1.13 and pandas 3
  ([#7](https://github.com/NingWang1729/piu_mosso/issues/7)); with scanpy 1.11 and pandas 2, all tests pass.

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

[Unreleased]: https://github.com/NingWang1729/piu_mosso/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/NingWang1729/piu_mosso/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NingWang1729/piu_mosso/releases/tag/v0.1.0
