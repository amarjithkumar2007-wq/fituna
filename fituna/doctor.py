# SPDX-License-Identifier: MIT
"""fituna.doctor
================

Environment self-diagnosis for ``fituna doctor``. Runs a fixed battery of
checks -- Python version, llama.cpp binaries (found *and* able to start),
whether the build has a GPU backend, llama.cpp version, hardware detection,
output directory writability, and free disk space -- and reports
each as PASS/WARN/FAIL with a one-line remedy for anything short of PASS.

Why this exists: today ``fituna run`` only reveals what is missing once it
fails partway through a search, with a message scoped to whatever step broke.
A third party running FiTuna for the first time (e.g. a competition judging
agency working from the docs alone) needs a single command that tells them,
up front and precisely, what is missing and how to fix it.

Never raises: every individual check is guarded (see ``_safe``) so a bug in
one check can only ever turn into a FAIL row for that check -- it can never
abort the rest of the diagnosis or crash the process. A diagnostic tool that
crashes mid-diagnosis is worse than useless for a user with no other context.

Binary discovery: the three required binaries (llama-quantize/llama-bench/
llama-perplexity), plus llama-cli (which ``fituna.binaries`` does not model
at all; see its own module docstring), are each looked up individually via
``fituna.binaries.find_exe()`` -- the same single-binary lookup
``fituna.binaries.locate_binaries()`` itself uses internally, reused rather
than re-wrapped. Checking one binary at a time (rather than delegating to
``locate_binaries()``, which is all-or-nothing by contract) is what lets a
FAIL row say *which* binary is missing. Version detection -- the one piece
of real, non-trivial logic in this area, with its regex-based probing of
--version/--help across binaries and llama.cpp release eras -- is reused via
``fituna.binaries.get_llama_cpp_version()`` unconditionally, never
duplicated.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Callable, Optional

from fituna import binaries, hardware
from fituna.config import BinaryPaths, DoctorCheck, GPUVendor

_MIN_PYTHON = (3, 11)
_MIN_FREE_DISK_GB = 20.0
_REQUIRED_BINARIES = binaries.REQUIRED_BINARIES  # single source of truth: fituna.binaries (public)

_NAME_WIDTH = 18  # fixed-width name column in the human report
_WRAP_WIDTH = 76  # remedy line-wrap width (indent included); matches the task brief's example exactly

_BINARY_REMEDY = (
    "Install llama.cpp (brew install llama.cpp on macOS/Linux via Homebrew) "
    "or build it from source "
    "(https://github.com/ggml-org/llama.cpp#building-the-project), then add "
    "its build/bin directory to PATH or pass --llama-bin-dir."
)


# ---------------------------------------------------------------------------
# individual checks -- each one is a pure function returning a DoctorCheck;
# no check here catches its own exceptions -- that is _safe()'s job, applied
# at each check's run_checks() call site, so each check's happy-path logic
# stays readable. _check_required_binaries is the one exception: it produces
# three rows instead of one, so it calls _safe itself, once per binary,
# rather than being wrapped externally like everything else.
# ---------------------------------------------------------------------------


def _check_python(version_info: tuple = sys.version_info) -> DoctorCheck:
    major, minor, micro = version_info[0], version_info[1], version_info[2]
    version = f"{major}.{minor}.{micro}"
    if (major, minor) >= _MIN_PYTHON:
        return DoctorCheck("python", "PASS", version, None)
    return DoctorCheck(
        "python",
        "FAIL",
        f"{version} (requires 3.11+)",
        "Install Python 3.11 or newer and re-run fituna with that interpreter.",
    )


def _launch_error(path: Path) -> Optional[str]:
    """None if the binary starts at all, else why it doesn't. The exit code
    is deliberately ignored (llama-quantize --version exits 1 on a healthy
    install); only a failed exec or a signal death counts -- that is what a
    missing dylib (dyld -> SIGABRT) or a macOS quarantine kill looks like,
    and both used to pass as "found"."""
    try:
        proc = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    # ponytail: POSIX signals only; a Windows missing-DLL exit (0xC0000135) still passes
    if proc.returncode < 0:
        # dyld puts the useful part ("Library not loaded: @rpath/...") on line 1
        lines = (proc.stderr or "").strip().splitlines()
        return f"killed by signal {-proc.returncode}" + (f": {lines[0][:160]}" if lines else "")
    return None


def _check_binary(name: str, bin_dir: Optional[Path], *, required: bool) -> DoctorCheck:
    # Reuses fituna.binaries.find_exe -- the same single-binary lookup
    # locate_binaries() itself calls internally -- rather than duplicating
    # its shutil.which(...) wrapping here (see module docstring).
    path = binaries.find_exe(name, bin_dir)
    if path is None:
        if required:
            return DoctorCheck(name, "FAIL", "not found", _BINARY_REMEDY)
        return DoctorCheck(
            name,
            "WARN",
            "not found",
            f"{name} is only used to run the final chosen config, not by "
            "fituna itself; " + _BINARY_REMEDY,
        )
    error = _launch_error(path)
    if error is None:
        return DoctorCheck(name, "PASS", str(path), None)
    return DoctorCheck(
        name,
        "FAIL" if required else "WARN",
        f"{path} found but does not run: {error}",
        "Reinstall llama.cpp. A build downloaded on macOS may need "
        "`xattr -dr com.apple.quarantine <bin dir>`; a 'Library not loaded' "
        "error means the bin directory was copied without its shared libraries.",
    )


def _check_required_binaries(
    bin_dir: Optional[Path],
) -> tuple[list[DoctorCheck], Optional[BinaryPaths]]:
    """PASS/FAIL rows for the three required llama.cpp binaries, plus the
    resolved BinaryPaths for the version check to reuse (None if any are
    missing, since BinaryPaths' required fields would then have nothing
    real to hold).

    Guards each row with _safe individually, so this function itself never
    raises -- no bespoke "guard the whole batch" wrapper needed (see the
    checks-section banner above). `paths` is read back from the rows' own
    already-resolved detail strings (== str(path) for a PASS row -- see
    _check_binary) rather than by calling find_exe a second time, so it
    stays a pure, can't-fail step even once every row is a verified PASS.

    The final `by_name[...]` lookups below assume `_REQUIRED_BINARIES`
    still spells out exactly these three names; guarded by a KeyError
    catch (rather than a second _safe-style wrapper) so a future rename or
    reordering of `_REQUIRED_BINARIES` degrades to paths=None instead of
    raising out of this function.
    """
    rows = [
        _safe(name, lambda name=name: _check_binary(name, bin_dir, required=True))
        for name in _REQUIRED_BINARIES
    ]
    if any(row.status != "PASS" for row in rows):
        return rows, None

    by_name = {row.name: row.detail for row in rows}
    try:
        paths = BinaryPaths(
            llama_quantize=Path(by_name["llama-quantize"]),
            llama_bench=Path(by_name["llama-bench"]),
            llama_perplexity=Path(by_name["llama-perplexity"]),
        )
    except KeyError:
        return rows, None
    return rows, paths


def _probe_paths_for_version(
    bin_dir: Optional[Path], paths: Optional[BinaryPaths]
) -> Optional[BinaryPaths]:
    """BinaryPaths to feed get_llama_cpp_version(), or None if there is
    nothing real to probe. Reuses `paths` as-is when the required-binaries
    check already resolved it; otherwise probes llama-bench/llama-perplexity
    individually and, if only one exists, reuses that same real path for
    both fields (get_llama_cpp_version only reads these two) rather than
    fabricate a path that might accidentally resolve to an unrelated binary
    on PATH."""
    if paths is not None:
        return paths
    bench_path = binaries.find_exe("llama-bench", bin_dir)
    ppl_path = binaries.find_exe("llama-perplexity", bin_dir)
    if bench_path is None and ppl_path is None:
        return None
    return BinaryPaths(
        llama_quantize=bench_path or ppl_path,  # unused by get_llama_cpp_version
        llama_bench=bench_path or ppl_path,
        llama_perplexity=ppl_path or bench_path,
    )


def _check_llama_version(
    bin_dir: Optional[Path], paths: Optional[BinaryPaths]
) -> DoctorCheck:
    probe = _probe_paths_for_version(bin_dir, paths)
    version = binaries.get_llama_cpp_version(probe) if probe is not None else None
    if version is not None:
        return DoctorCheck("llama.cpp version", "PASS", version, None)
    return DoctorCheck(
        "llama.cpp version",
        "WARN",
        "could not be detected",
        'bench cache falls back to "unknown"; results from different builds '
        "may be reused. Upgrade llama.cpp or pass --llama-bin-dir.",
    )


# Backends llama-bench lists that are not a GPU offload target.
_CPU_SIDE_BACKENDS = ("BLAS", "CPU", "RPC")


def _list_devices(bench: Path) -> Optional[str]:
    """`llama-bench --list-devices` stdout, or None if the build can't list."""
    try:
        proc = subprocess.run(
            [str(bench), "--list-devices"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or "Available devices" not in proc.stdout:
        return None
    return proc.stdout


def _parse_gpu_devices(listing: str) -> list[str]:
    """Device lines after "Available devices:" that are real GPU backends
    (MTL0, CUDA0, Vulkan0, ROCm0, ...)."""
    _, _, devices = listing.partition("Available devices:")
    return [
        line.strip()
        for line in devices.splitlines()
        if ":" in line and not line.strip().startswith(_CPU_SIDE_BACKENDS)
    ]


def _check_gpu_backend(paths: Optional[BinaryPaths]) -> DoctorCheck:
    """WARN when the machine has a GPU but this llama.cpp build can't use it:
    -ngl then does nothing and every measured speed is CPU speed."""
    if paths is None:
        return DoctorCheck(
            "gpu-backend",
            "WARN",
            "skipped: llama-bench not available",
            "Fix the llama-bench row above, then re-run fituna doctor.",
        )
    # ponytail: second detect_hardware() call (also in _check_hardware); <1 s, keeps each row independent
    hw = hardware.detect_hardware()
    listing = _list_devices(paths.llama_bench)
    if listing is None:
        return DoctorCheck(
            "gpu-backend",
            "WARN",
            "llama-bench --list-devices failed (llama.cpp build too old?)",
            "Upgrade llama.cpp so FiTuna can confirm the build can use your GPU.",
        )
    gpus = _parse_gpu_devices(listing)
    if gpus:
        return DoctorCheck("gpu-backend", "PASS", "; ".join(gpus), None)
    if hw.gpu_vendor == GPUVendor.NONE:
        return DoctorCheck("gpu-backend", "PASS", "CPU-only build (no GPU detected either)", None)
    return DoctorCheck(
        "gpu-backend",
        "WARN",
        f"{hw.gpu_vendor.value} GPU detected, but this llama.cpp build has no GPU backend",
        "-ngl will have no effect and every measured speed will be CPU speed. "
        "Install a GPU-enabled llama.cpp (Homebrew's includes Metal; for NVIDIA "
        "build with -DGGML_CUDA=ON) and point --llama-bin-dir at it.",
    )


def _check_hardware() -> DoctorCheck:
    hw = hardware.detect_hardware()
    detail = (
        f"gpu={hw.gpu_vendor.value}"
        + (f" ({hw.gpu_name})" if hw.gpu_name else "")
        + (f", vram={hw.vram_mb}MB" if hw.vram_mb is not None else "")
        + f", cpu={hw.cpu_cores} cores, ram={hw.ram_mb}MB, os={hw.os_name}"
    )
    if hw.gpu_vendor != GPUVendor.NONE:
        return DoctorCheck("hardware", "PASS", detail, None)
    return DoctorCheck(
        "hardware",
        "WARN",
        detail,
        "No GPU detected; CPU-only inference still works but is slower. If "
        "this is a misdetection, override it with --gpu and --vram-mb.",
    )


def _nearest_existing_ancestor(path: Path) -> Path:
    """Walk up from ``path`` to the nearest ancestor that actually exists,
    without creating anything -- used to check "could this be created" and
    "how much free space is available" for a --out directory that may not
    exist yet. Returns ``path`` itself when it already exists -- which may
    be a file, not a directory (hence "ancestor", not "dir", in the name);
    callers that care about that distinction check it themselves (see
    _check_out_dir)."""
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:  # reached the filesystem root
            return current
        current = parent
    return current


def _check_out_dir(out_dir: Path) -> DoctorCheck:
    if out_dir.exists() and not out_dir.is_dir():
        return DoctorCheck(
            "out-dir",
            "FAIL",
            f"{out_dir} exists and is not a directory",
            "Pass a different --out path (the existing one is a file).",
        )

    existing = _nearest_existing_ancestor(out_dir)
    if not existing.is_dir():
        # os.access(<regular file>, W_OK) is True, so without this the check
        # would PASS on a path like `somefile.txt/out` that mkdir cannot
        # create (ENOTDIR) -- a false PASS on the one check whose job is to
        # catch this before a multi-hour run.
        return DoctorCheck(
            "out-dir",
            "FAIL",
            f"{out_dir} cannot be created: {existing} exists and is not a directory",
            "Pass a different --out path (an ancestor of this one is a file).",
        )

    writable = os.access(existing, os.W_OK)
    if out_dir.exists():
        detail = f"{out_dir} ({'writable' if writable else 'not writable'})"
    elif writable:
        detail = f"{out_dir} does not exist yet; can be created under {existing}"
    else:
        detail = f"{out_dir} does not exist and cannot be created ({existing} is not writable)"

    if not writable:
        return DoctorCheck(
            "out-dir", "FAIL", detail, f"Make {existing} writable, or pass a different --out path."
        )
    return DoctorCheck("out-dir", "PASS", detail, None)


def _check_disk_space(out_dir: Path) -> DoctorCheck:
    existing = _nearest_existing_ancestor(out_dir)
    usage = shutil.disk_usage(existing)
    free_gb = usage.free / (1024**3)
    detail = f"{free_gb:.1f} GB free at {existing}"
    if free_gb < _MIN_FREE_DISK_GB:
        return DoctorCheck(
            "disk-space",
            "WARN",
            detail,
            "Quantizing 4 candidate configs of a 4B model needs ~12 GB of "
            "free space; free up space or point --out at a volume with more room.",
        )
    return DoctorCheck("disk-space", "PASS", detail, None)


# ---------------------------------------------------------------------------
# orchestration: run every check, never let one crash take down the rest
# ---------------------------------------------------------------------------


def _safe(name: str, fn: Callable[[], DoctorCheck]) -> DoctorCheck:
    """Run one check, converting any exception into a FAIL row instead of
    letting it propagate (see module docstring: doctor must never crash).
    The name-reconciliation step below is inside the same try -- a check
    that breaks its own contract (returns something other than a
    DoctorCheck) fails ``dataclasses.replace`` with a TypeError, which is
    just another exception this same guard catches, so it becomes a FAIL
    row instead of escaping.

    ``name`` is forced onto the returned row either way -- the single
    source of truth for a row's name, so it can never silently drift from
    whatever a check function happens to hardcode internally between its
    normal and crashed forms.
    """
    try:
        row = fn()
        row = dataclasses.replace(row, name=name)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see above
        return DoctorCheck(
            name,
            "FAIL",
            f"check crashed unexpectedly: {exc}",
            "This looks like a fituna bug; please file an issue with this output.",
        )
    return row


def run_checks(bin_dir: Optional[Path], out_dir: Path) -> list[DoctorCheck]:
    """Run the full diagnostic battery and return one DoctorCheck per row,
    in report order. Never raises."""
    checks: list[DoctorCheck] = [_safe("python", _check_python)]

    binary_rows, paths = _check_required_binaries(bin_dir)
    checks.extend(binary_rows)
    for optional in ("llama-cli", "llama-server"):
        checks.append(
            _safe(optional, lambda n=optional: _check_binary(n, bin_dir, required=False))
        )
    checks.append(_safe("gpu-backend", lambda: _check_gpu_backend(paths)))
    checks.append(_safe("llama.cpp version", lambda: _check_llama_version(bin_dir, paths)))
    checks.append(_safe("hardware", _check_hardware))
    checks.append(_safe("out-dir", lambda: _check_out_dir(out_dir)))
    checks.append(_safe("disk-space", lambda: _check_disk_space(out_dir)))
    return checks


def summarize(checks: list[DoctorCheck]) -> tuple[int, int, int]:
    """(passed, warned, failed) counts across all checks."""
    passed = sum(1 for c in checks if c.status == "PASS")
    warned = sum(1 for c in checks if c.status == "WARN")
    failed = sum(1 for c in checks if c.status == "FAIL")
    return passed, warned, failed


def exit_code(checks: list[DoctorCheck]) -> int:
    """0 if nothing failed; 2 if any of the three required llama.cpp
    binaries failed (matches the existing BinaryNotFoundError convention --
    see fituna.cli.main); 1 for any other failure; WARN-only is still 0."""
    if any(c.status == "FAIL" for c in checks if c.name in _REQUIRED_BINARIES):
        return 2
    if any(c.status == "FAIL" for c in checks):
        return 1
    return 0


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def _format_row(check: DoctorCheck) -> list[str]:
    lines = [f"  [{check.status}] {check.name:<{_NAME_WIDTH}}{check.detail}"]
    if check.remedy:
        lines.extend(
            textwrap.wrap(
                check.remedy,
                width=_WRAP_WIDTH,
                initial_indent=" " * 9 + "-> ",
                subsequent_indent=" " * 12,
                break_on_hyphens=False,  # keep --flags and URLs from splitting mid-word
            )
        )
    return lines


def to_human(checks: list[DoctorCheck]) -> str:
    """Render the fixed-width human report: one line per check (plus a
    wrapped remedy line for anything short of PASS), then a summary line."""
    lines = ["FiTuna doctor"]
    for check in checks:
        lines.extend(_format_row(check))

    passed, warned, failed = summarize(checks)
    check_word = "check" if passed == 1 else "checks"
    warning_word = "warning" if warned == 1 else "warnings"
    lines.append("")
    lines.append(f"{passed} {check_word} passed, {warned} {warning_word}, {failed} failed.")
    first_fail = next((c.name for c in checks if c.status == "FAIL"), None)
    if first_fail:
        lines.append(f"Next: fix {first_fail} first (see its -> line), then re-run `fituna doctor`.")
    else:
        lines.append(
            "Next: run `fituna quickstart` for a guided first run, or see `fituna run -h`."
        )
    return "\n".join(lines)


def to_json(checks: list[DoctorCheck]) -> str:
    """Serialize checks to the documented schema: {checks: [...], summary:
    {...}}, ready to print to stdout for automated collection."""
    passed, warned, failed = summarize(checks)
    payload = {
        "checks": [dataclasses.asdict(c) for c in checks],
        "summary": {"passed": passed, "warned": warned, "failed": failed},
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# self-check (run: python -m fituna.doctor, or python -m fituna.doctor
# --selfcheck -- any argv is ignored, this module has no CLI of its own;
# real usage is `fituna doctor`, registered in fituna/cli.py)
# ---------------------------------------------------------------------------


def _selfcheck() -> None:
    # 1. Python version check: both branches, via dependency-injected
    #    version_info rather than monkeypatching sys itself.
    ok = _check_python(version_info=(3, 13, 1, "final", 0))
    assert ok == DoctorCheck("python", "PASS", "3.13.1", None), ok

    old = _check_python(version_info=(3, 9, 6, "final", 0))
    assert old.status == "FAIL" and old.remedy is not None, old

    # 2. never-raises guarantee: a check that blows up becomes a FAIL row,
    #    not a propagated exception (the hard requirement this module exists
    #    to satisfy -- see tests/test_doctor.py for the fuller version,
    #    including run_checks()-level coverage).
    def _boom() -> DoctorCheck:
        raise RuntimeError("simulated bug")

    crashed = _safe("some-check", _boom)
    assert crashed.status == "FAIL" and "simulated bug" in crashed.detail, crashed

    # 3. summarize/exit_code, including the binary-FAIL-takes-precedence rule.
    p = DoctorCheck("x", "PASS", "ok", None)
    w = DoctorCheck("y", "WARN", "meh", "fix meh")
    f_other = DoctorCheck("z", "FAIL", "bad", "fix bad")
    f_binary = DoctorCheck("llama-quantize", "FAIL", "not found", "install it")

    assert summarize([p, p, w]) == (2, 1, 0)
    assert exit_code([p, w]) == 0
    assert exit_code([p, f_other]) == 1
    assert exit_code([p, f_binary]) == 2

    # 4. to_human: fixed-width alignment, wrapped remedy, pluralized summary.
    human = to_human([p, w])
    assert "[PASS] x" in human, human
    assert "-> fix meh" in human, human
    assert "1 check passed, 1 warning, 0 failed." in human, human
    assert human.splitlines()[-1].startswith("Next: run `fituna quickstart`"), human

    # 5. to_json: documented schema, round-trips through json.loads.
    payload = json.loads(to_json([p, w, f_other]))
    assert payload["summary"] == {"passed": 1, "warned": 1, "failed": 1}, payload
    assert payload["checks"][0] == {
        "name": "x",
        "status": "PASS",
        "detail": "ok",
        "remedy": None,
    }, payload

    # 6. run_checks() on the real machine: must never raise, must always
    #    produce one row per required binary plus the fixed non-binary rows,
    #    regardless of what's actually installed here (mirrors the same
    #    "never raises, whatever the machine has" smoke test in
    #    fituna.hardware._selfcheck / fituna.binaries._selfcheck).
    checks = run_checks(None, Path("./out"))
    assert len(checks) == 11, len(checks)
    assert {c.status for c in checks} <= {"PASS", "WARN", "FAIL"}
    assert exit_code(checks) in (0, 1, 2)

    print("fituna.doctor self-check OK")


if __name__ == "__main__":
    _selfcheck()
