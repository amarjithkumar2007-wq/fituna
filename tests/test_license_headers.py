# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FiTuna contributors
"""Every Python source file carries the project's license and copyright header.

Line 1 is the SPDX license tag and line 2 is the copyright notice matching the
root ``LICENSE``. License scanners read these per-file headers, so a file
without them shows up as "unknown license" or "no copyright" even though the
repository as a whole is MIT.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPDX_LINE = "# SPDX-License-Identifier: MIT"
COPYRIGHT_LINE = "# Copyright (c) 2026 FiTuna contributors"

PY_FILES = sorted(
    p for d in ("fituna", "tests") for p in (ROOT / d).rglob("*.py")
)


def test_python_files_found():
    assert PY_FILES, "no Python files collected; ROOT is wrong"


@pytest.mark.parametrize(
    "path", PY_FILES, ids=lambda p: p.relative_to(ROOT).as_posix()
)
def test_license_and_copyright_header(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [SPDX_LINE, COPYRIGHT_LINE]
