# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FiTuna contributors
"""fituna.errors
================

Re-exports the :class:`FiTunaError` hierarchy from :mod:`fituna.config`.

The exceptions are defined once, in ``config.py``, alongside the rest of the
interface contract (single source of truth for cross-module types). This
module exists only so ``from fituna.errors import ...`` reads naturally from
call sites that only care about error handling, not the data model.

# this file is a pure re-export shim, not a second definition site.
# Every other module in this package imports the exceptions straight from
# fituna.config (the contract), so nothing here can drift out of sync with
# it -- there is exactly one class body for each exception, in config.py.
# Duplicating the class definitions here would violate the "single source
# of truth" rule stated in the interface contract and risk two copies that
# compare unequal via isinstance/except across modules.
"""

from fituna.config import (
    BenchTimeoutError,
    BinaryNotFoundError,
    FiTunaError,
    ModelConversionError,
    NoFeasibleConfigError,
)

__all__ = [
    "FiTunaError",
    "BenchTimeoutError",
    "BinaryNotFoundError",
    "ModelConversionError",
    "NoFeasibleConfigError",
]


def _self_check() -> None:
    """Minimal assert-based guard for this module's one job: re-exporting.

    Not a full test suite -- just a runnable check that the names imported
    from ``fituna.errors`` are *identical* (not merely equal-looking copies)
    to the ones in ``fituna.config``, that the hierarchy is intact so
    ``except FiTunaError`` catches every subtype, and that
    ``NoFeasibleConfigError``'s ``closest`` payload survives the re-export.
    """
    from fituna import config as _config

    # 1. Identity, not just equality: re-export must be the *same* class
    #    object, so `isinstance`/`except` behave identically regardless of
    #    which module a caller imported the exception from.
    assert FiTunaError is _config.FiTunaError
    assert BinaryNotFoundError is _config.BinaryNotFoundError
    assert ModelConversionError is _config.ModelConversionError
    assert NoFeasibleConfigError is _config.NoFeasibleConfigError

    # 2. Hierarchy: every concrete error is a FiTunaError, so a single
    #    `except FiTunaError` at the CLI boundary (cli.py) catches all of
    #    them, and each is also a plain Exception as a safety net.
    for cls in (BinaryNotFoundError, ModelConversionError, NoFeasibleConfigError):
        assert issubclass(cls, FiTunaError)
        assert issubclass(cls, Exception)

    # 3. Raising/catching works through the re-exported names exactly like
    #    it does through fituna.config's names.
    try:
        raise BinaryNotFoundError("llama-quantize not found on PATH")
    except FiTunaError as exc:
        assert isinstance(exc, BinaryNotFoundError)
        assert str(exc) == "llama-quantize not found on PATH"
    else:
        raise AssertionError("BinaryNotFoundError must be catchable as FiTunaError")

    # 4. NoFeasibleConfigError's extra `closest` field is not lost by the
    #    re-export (search.py relies on this to report a best-effort result).
    err = NoFeasibleConfigError("no quant met the target speed", closest=None)
    assert err.closest is None
    assert str(err) == "no quant met the target speed"

    sentinel = object()
    err_with_closest = NoFeasibleConfigError("no quant met the target", closest=sentinel)  # type: ignore[arg-type]
    assert err_with_closest.closest is sentinel


if __name__ == "__main__":
    _self_check()
    print("fituna.errors self-check OK")
