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

import sys

import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg
import umap.spectral
from sklearn.utils import check_random_state

import piu_mosso.gpu_spectral
import piu_mosso.spectral
from piu_mosso import knn, umap_connectivities
from piu_mosso.spectral import choose_eigensolver, spectral_layout


def cupy_gpu_available():
    """Whether CuPy imports and sees a CUDA GPU (GPU tests are skipped otherwise)."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no CuPy, or no usable CUDA
        return False


HAS_CUPY_GPU = cupy_gpu_available()
needs_gpu = pytest.mark.skipif(not HAS_CUPY_GPU, reason="needs CuPy and a CUDA GPU")


def clustered_graph(n_clusters, cells_per_cluster, separation, seed=0):
    """Data in Gaussian clusters and its UMAP graph (15 neighbours); far-apart clusters give separate components."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=separation, size=(n_clusters, 10))
    noise = rng.normal(size=(n_clusters * cells_per_cluster, 10))
    X = (centers.repeat(cells_per_cluster, axis=0) + noise).astype(np.float32)
    return X, umap_connectivities(*knn(X, 14)).tocoo()


# Tests of an internal rule: the size at which the solver switches, checked without building graphs of millions
# of cells. The behaviour (lobpcg gives the same layout) is tested below.
@pytest.mark.parametrize(
    ("n_vertices", "expected"),
    [
        (1_000, "eigsh"),
        (1_600_000, "eigsh"),
        (1_700_000, "lobpcg"),  # eigsh's Lanczos basis would pass 2^31 - 1 entries
        (2_500_000, "lobpcg"),  # umap-learn's own switch, from 2,000,000
    ],
)
def test_eigensolver_switches_where_arpack_would_overflow(n_vertices, expected):
    assert choose_eigensolver(n_vertices, 2) == expected


def test_eigensolver_switches_at_the_first_overflowing_size():
    # int(sqrt(n)) = 1290 here: 1,664,716 x 1290 = 2,147,483,640 <= 2^31 - 1 = 2,147,483,647 < 1,664,717 x 1290
    assert choose_eigensolver(1_664_716, 2) == "eigsh"
    assert choose_eigensolver(1_664_717, 2) == "lobpcg"


@pytest.mark.parametrize(
    ("n_clusters", "separation"), [(1, 1.0), (12, 100.0)], ids=["one_component", "many_components"]
)
def test_spectral_layout_equals_umap_learn(n_clusters, separation):
    X, graph = clustered_graph(n_clusters, 1200 // n_clusters, separation)
    expected = umap.spectral.spectral_layout(X, graph, 2, check_random_state(0))
    result = spectral_layout(X, graph, 2, check_random_state(0))
    np.testing.assert_array_equal(result, expected)


def test_lobpcg_layout_spans_the_same_space_as_eigsh(monkeypatch):
    # Large graphs use lobpcg; forced here on a small one, its eigenvectors span eigsh's (up to sign and rotation)
    X, graph = clustered_graph(1, 1000, 1.0)
    eigsh_layout = spectral_layout(X, graph, 2, check_random_state(0))
    monkeypatch.setattr(piu_mosso.spectral, "choose_eigensolver", lambda n_vertices, dim: "lobpcg")
    lobpcg_layout = spectral_layout(X, graph, 2, check_random_state(0))
    # Projecting one basis onto the other keeps its length when the spaces are the same
    q_eigsh, _ = np.linalg.qr(eigsh_layout)
    q_lobpcg, _ = np.linalg.qr(lobpcg_layout)
    np.testing.assert_allclose(np.linalg.svd(q_eigsh.T @ q_lobpcg, compute_uv=False), 1.0, atol=1e-3)


def test_tiny_components_are_placed_at_random_as_umap_learn():
    # Components too small for an eigensolver (fewer than 2 dim cells) are placed at random near their centroid's
    # position, as umap-learn does, not treated as a failure
    sizes = [300, 5, 4, 3, 1, 2, 6, 200]
    rng = np.random.default_rng(2)
    blocks = [scipy.sparse.random(n, n, density=0.5, random_state=rng) for n in sizes]
    graph = scipy.sparse.block_diag([block + block.T for block in blocks])
    X = rng.normal(size=(graph.shape[0], 5))
    expected = umap.spectral.spectral_layout(X, graph.tocoo(), 2, check_random_state(0))
    np.testing.assert_array_equal(spectral_layout(X, graph.tocoo(), 2, check_random_state(0)), expected)


@pytest.mark.parametrize("solver", ["eigsh", "lobpcg"])
def test_failed_eigensolver_raises_naming_pca(monkeypatch, solver):
    # One iteration is too few for either solver to converge; umap-learn would fall back to a random start
    X, graph = clustered_graph(1, 1000, 1.0)
    monkeypatch.setattr(piu_mosso.spectral, "choose_eigensolver", lambda n_vertices, dim: solver)
    with pytest.raises(RuntimeError, match="init='pca'.*init_pos='pca'"):
        spectral_layout(X, graph, 2, check_random_state(0), maxiter=1)


@pytest.fixture
def cpu_solver_tols(monkeypatch):
    """(solver, tol) of each call to scipy's eigsh or lobpcg during a test; the real solvers still run."""
    calls = []
    for name in ("eigsh", "lobpcg"):
        solve = getattr(scipy.sparse.linalg, name)

        def recording(*args, _name=name, _solve=solve, **kwargs):
            calls.append((_name, kwargs["tol"]))
            return _solve(*args, **kwargs)

        monkeypatch.setattr(scipy.sparse.linalg, name, recording)
    return calls


def test_lobpcg_default_tolerance_is_1e_5(monkeypatch, cpu_solver_tols):
    # Large graphs use lobpcg (forced here on a small one); at umap-learn's 1e-4 it can stop before telling apart
    # eigenvectors whose eigenvalues are closer than that, as on graphs of millions of cells
    monkeypatch.setattr(piu_mosso.spectral, "choose_eigensolver", lambda n_vertices, dim: "lobpcg")
    X, graph = clustered_graph(1, 1000, 1.0)
    spectral_layout(X, graph, 2, check_random_state(0))
    assert cpu_solver_tols == [("lobpcg", 1e-5)]


def test_eigsh_default_tolerance_stays_1e_4(cpu_solver_tols):
    # Guard (passes before the change too): below ARPACK's limit the start keeps umap-learn's eigsh at 1e-4
    X, graph = clustered_graph(1, 1000, 1.0)
    spectral_layout(X, graph, 2, check_random_state(0))
    assert cpu_solver_tols == [("eigsh", 1e-4)]


def test_a_given_tolerance_overrides_the_cpu_default(monkeypatch, cpu_solver_tols):
    # Guard (passes before the change too): a tol passed to spectral_layout reaches lobpcg
    monkeypatch.setattr(piu_mosso.spectral, "choose_eigensolver", lambda n_vertices, dim: "lobpcg")
    X, graph = clustered_graph(1, 1000, 1.0)
    spectral_layout(X, graph, 2, check_random_state(0), tol=1e-6)
    assert cpu_solver_tols == [("lobpcg", 1e-6)]


def test_old_tolerance_name_warns_and_gives_the_new_constant():
    # GPU_SPECTRAL_TOL was the GPU's tolerance in 0.2.0; it now names LOBPCG_TOL, which both paths use
    with pytest.warns(DeprecationWarning, match="LOBPCG_TOL"):
        assert piu_mosso.spectral.GPU_SPECTRAL_TOL == piu_mosso.spectral.LOBPCG_TOL


def test_use_gpu_without_cupy_raises_naming_the_extra(monkeypatch):
    # sys.modules[name] = None makes `import cupy` fail, whether or not CuPy is installed
    monkeypatch.setitem(sys.modules, "cupy", None)
    X, graph = clustered_graph(1, 300, 1.0)
    with pytest.raises(ImportError, match=r"piu-mosso\[cuda12\]"):
        spectral_layout(X, graph, 2, check_random_state(0), use_gpu=True)


def subspace_cosines(layout, reference):
    """Cosines of the principal angles between the column spaces of two layouts (1 = the same space; eigenvectors
    are defined up to sign, and up to rotation where eigenvalues are close)."""
    q_layout, _ = np.linalg.qr(layout)
    q_reference, _ = np.linalg.qr(reference)
    return np.linalg.svd(q_layout.T @ q_reference, compute_uv=False)


@pytest.fixture
def gpu_solves(monkeypatch):
    """(size, tol) of each component the GPU eigensolver is called on during a test."""
    calls = []
    solve = piu_mosso.gpu_spectral.lobpcg_eigenpairs

    def counting(graph, *args, **kwargs):
        calls.append((graph.shape[0], kwargs["tol"]))
        return solve(graph, *args, **kwargs)

    monkeypatch.setattr(piu_mosso.gpu_spectral, "lobpcg_eigenpairs", counting)
    return calls


@needs_gpu
def test_gpu_layout_has_the_cpu_eigenvectors(gpu_solves):
    # 12,000 cells: above GPU_SPECTRAL_MIN_CELLS, so solved on the GPU; its eigenvectors span the CPU's space within
    # the solvers' tolerances (eigsh on the CPU at 1e-4, lobpcg on the GPU at 1e-5)
    X, graph = clustered_graph(1, 12_000, 1.0)
    cpu = spectral_layout(X, graph, 2, check_random_state(0))
    gpu = spectral_layout(X, graph, 2, check_random_state(0), use_gpu=True)
    # Solved once, on the GPU, at the GPU's default tolerance
    assert gpu_solves == [(12_000, 1e-5)]
    assert gpu.shape == cpu.shape
    assert subspace_cosines(gpu, cpu).min() > 0.999


def test_components_below_the_gpu_size_keep_the_cpu_result(monkeypatch, gpu_solves):
    # Runs without a GPU: the CuPy check is skipped, and a 1,000-cell graph never reaches the GPU solver
    monkeypatch.setattr(piu_mosso.spectral, "check_cupy", lambda: None)
    X, graph = clustered_graph(1, 1000, 1.0)
    cpu = spectral_layout(X, graph, 2, check_random_state(0))
    np.testing.assert_array_equal(spectral_layout(X, graph, 2, check_random_state(0), use_gpu=True), cpu)
    assert gpu_solves == []


@needs_gpu
def test_gpu_layout_of_separate_components_is_finite(monkeypatch, gpu_solves):
    monkeypatch.setattr(piu_mosso.spectral, "GPU_SPECTRAL_MIN_CELLS", 500)
    X, graph = clustered_graph(2, 1000, 100.0)
    cpu = spectral_layout(X, graph, 2, check_random_state(0))
    gpu = spectral_layout(X, graph, 2, check_random_state(0), use_gpu=True)
    assert [size for size, _ in gpu_solves] == [1000, 1000]
    assert gpu.shape == cpu.shape
    assert np.isfinite(gpu).all()


@needs_gpu
def test_failed_gpu_eigensolver_raises_naming_pca(monkeypatch, gpu_solves):
    # One iteration is too few for lobpcg to converge
    monkeypatch.setattr(piu_mosso.spectral, "GPU_SPECTRAL_MIN_CELLS", 500)
    X, graph = clustered_graph(1, 1000, 1.0)
    with pytest.raises(RuntimeError, match="init='pca'.*init_pos='pca'"):
        spectral_layout(X, graph, 2, check_random_state(0), maxiter=1, use_gpu=True)
    assert [size for size, _ in gpu_solves] == [1000]


@needs_gpu
def test_a_given_tolerance_overrides_the_gpu_default(monkeypatch, gpu_solves):
    monkeypatch.setattr(piu_mosso.spectral, "GPU_SPECTRAL_MIN_CELLS", 500)
    X, graph = clustered_graph(1, 1000, 1.0)
    spectral_layout(X, graph, 2, check_random_state(0), tol=1e-4, use_gpu=True)
    assert gpu_solves == [(1000, 1e-4)]


@needs_gpu
def test_running_out_of_gpu_memory_raises_naming_the_alternatives(monkeypatch):
    import cupy
    import cupyx.scipy.sparse.linalg

    def out_of_memory(*args, **kwargs):
        raise cupy.cuda.memory.OutOfMemoryError(2**40, 0)

    monkeypatch.setattr(cupyx.scipy.sparse.linalg, "lobpcg", out_of_memory)
    monkeypatch.setattr(piu_mosso.spectral, "GPU_SPECTRAL_MIN_CELLS", 500)
    X, graph = clustered_graph(1, 1000, 1.0)
    with pytest.raises(RuntimeError, match="use_gpu=False.*init='pca'"):
        spectral_layout(X, graph, 2, check_random_state(0), use_gpu=True)
