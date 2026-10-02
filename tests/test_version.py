# SPDX-License-Identifier: MIT
"""The package version lives in two places: pyproject.toml (what the build
publishes) and fituna.__version__ (what the release workflow compares the
tag against, and what users see). They must never drift apart."""

import tomllib
from pathlib import Path

import fituna

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_pyproject_version_matches_package_version():
    with PYPROJECT.open("rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert declared == fituna.__version__
