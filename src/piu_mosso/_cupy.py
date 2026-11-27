"""
The check that CuPy can run on a GPU, made at the call with use_gpu=True, before any work, so a missing extra or GPU
fails early. A missing CuPy raises ImportError (Python's convention for a missing optional dependency).

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

import importlib

INSTALL_HINT = (
    "pip install 'piu-mosso[cuda12]' for CUDA 12 drivers or 'piu-mosso[cuda13]' for CUDA 13 (only one of the two), "
    "or use use_gpu=False"
)


def check_cupy():
    """Raise ImportError if CuPy cannot be imported, RuntimeError if it cannot use a CUDA GPU."""
    try:
        cupy = importlib.import_module("cupy")
    except ImportError as error:
        raise ImportError(f"use_gpu=True needs CuPy: {INSTALL_HINT}") from error
    try:
        n_devices = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        raise RuntimeError(f"use_gpu=True, but CuPy cannot use a CUDA GPU here ({error}): use use_gpu=False") from \
            error
    if n_devices == 0:
        raise RuntimeError("use_gpu=True, but CuPy sees no CUDA GPU: use use_gpu=False")
