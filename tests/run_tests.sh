#!/bin/bash
# piu_mosso: fast single-cell utilities. Copyright (C) 2026 Ning Wang. Licensed under the GNU Lesser General Public
# License, version 3 or (at your option) any later version; see LICENSE.
#
# Run the unit tests in the active Python environment (activate the piu_mosso environment first).
# GPU-only tests are skipped automatically when faiss finds no GPU.
#
# Usage: bash tests/run_tests.sh [extra pytest arguments]
#   e.g. bash tests/run_tests.sh -k knn
set -euo pipefail

# Fail early with install instructions if the active environment has no pytest
if ! python -c "import pytest" 2>/dev/null; then
    echo "Error: pytest is not installed for $(command -v python). Install the test extra: pip install -e '.[test]'" >&2
    exit 1
fi

# No core dumps from crashes in native code (FAISS, numba); they can fill a disk
ulimit -c 0

# Run from the repository root
cd "$(dirname "$(realpath "$0")")/.."
python -m pytest tests/ "$@"
