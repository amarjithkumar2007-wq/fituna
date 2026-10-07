# SPDX-License-Identifier: MIT
"""FiTuna: hardware-aware auto-tuner for llama.cpp GGUF quantization + runtime configs."""

__version__ = "0.3.6"

__all__ = ["__version__"]


def _self_check() -> None:
    """Guard against __version__ drifting from pyproject.toml's [project.version]."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        return  # installed outside the source tree (e.g. site-packages) — nothing to compare against
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    pyproject_version = data["project"]["version"]
    assert __version__ == pyproject_version, (
        f"__version__ ({__version__}) drifted from pyproject.toml ({pyproject_version})"
    )
    assert isinstance(__version__, str) and __version__.count(".") == 2, (
        f"__version__ ({__version__!r}) is not a MAJOR.MINOR.PATCH string"
    )


if __name__ == "__main__":
    _self_check()
    print("fituna.__init__ self-check passed:", __version__)
