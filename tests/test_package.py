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

import importlib.metadata
import pathlib
import re

import pytest

import piu_mosso


def test_version_is_the_installed_version():
    assert piu_mosso.__version__ == importlib.metadata.version("piu_mosso")


def test_scanpy_style_namespaces_exist():
    # pp and tl hold the AnnData wrappers, as sc.pp and sc.tl do in scanpy
    assert piu_mosso.pp.__name__ == "piu_mosso.pp"
    assert piu_mosso.tl.__name__ == "piu_mosso.tl"


# A release version is only numbers (e.g. 0.1.0); a development version has a suffix (e.g. 0.2.0.dev0)
RELEASE_VERSION = re.compile(r"\d+\.\d+\.\d+")
REPOSITORY = pathlib.Path(__file__).resolve().parent.parent


def test_release_is_dated_in_changelog_and_citation():
    # A release's version must match its dated CHANGELOG entry and CITATION.cff (which Zenodo reads at each release)
    version = piu_mosso.__version__
    if not RELEASE_VERSION.fullmatch(version):
        pytest.skip(f"development version {version}")
    changelog = (REPOSITORY / "CHANGELOG.md").read_text()
    entry = re.search(rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", changelog, re.MULTILINE)
    assert entry, f"CHANGELOG.md has no dated entry for {version}"
    citation = (REPOSITORY / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(version)}$", citation, re.MULTILINE)
    assert re.search(rf'^date-released: "{entry[1]}"$', citation, re.MULTILINE)
