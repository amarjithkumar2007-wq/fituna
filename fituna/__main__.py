# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FiTuna contributors
"""``python -m fituna`` entry point."""

import sys

from fituna.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
