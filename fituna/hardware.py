# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FiTuna contributors
"""fituna.hardware
==================

Detects the local :class:`~fituna.config.HardwareProfile` by shelling out to
vendor CLIs (``nvidia-smi``, ``rocm-smi``, ``system_profiler``) with a
``platform``-module CPU-only fallback, and merges it with any user-supplied
overrides from the CLI.

Design notes:
- Every vendor probe is split into "run the CLI" (``_run``) and "parse its
  output" (``_parse_*``) so the parsing logic is unit-testable without a
  real GPU or subprocess mocking gymnastics -- tests can feed sample text
  straight to the ``_parse_*`` functions, or monkeypatch ``subprocess.run``
  for the ``_run``-based integration path (see tests/test_hardware.py).
- Detection is best-effort everywhere: a missing binary, non-zero exit code,
  or unparsable output all fall through to the next vendor / CPU-only,
  never raise.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Optional

from fituna.config import GPUVendor, HardwareProfile


# ---------------------------------------------------------------------------
# subprocess plumbing
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 5.0) -> Optional[str]:
    """Run ``cmd``, returning stdout on success or None on any failure
    (binary missing, non-zero exit, timeout). Never raises."""
    try:
        # encoding/errors explicit: text=True alone decodes with
        # locale.getpreferredencoding(), which on non-English Windows can be
        # a non-UTF-8 codepage (cp949/cp1252) and raise UnicodeDecodeError on
        # any non-ASCII byte in a tool's output.
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------


def _parse_nvidia_csv(out: str) -> Optional[tuple[str, Optional[int]]]:
    """Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`.
    Sample line: 'NVIDIA GeForce RTX 4090, 24564'"""
    line = next((ln for ln in out.splitlines() if ln.strip()), "")
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    name = parts[0] if parts and parts[0] else "NVIDIA GPU"
    vram_mb: Optional[int] = None
    if len(parts) > 1:
        m = re.search(r"\d+", parts[1])
        if m:
            vram_mb = int(m.group())
    return (name, vram_mb)


def _detect_nvidia() -> Optional[tuple[str, Optional[int]]]:
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if out is None:
        return None
    return _parse_nvidia_csv(out)


# ---------------------------------------------------------------------------
# AMD (ROCm)
# ---------------------------------------------------------------------------


def _parse_rocm_smi(out: str) -> Optional[tuple[str, Optional[int]]]:
    """Parse `rocm-smi --showproductname --showmeminfo vram` output. Format
    varies across rocm-smi versions; look for 'Card series/model' and
    'VRAM Total Memory (B)' lines rather than fixed column positions."""
    name: Optional[str] = None
    vram_mb: Optional[int] = None
    for line in out.splitlines():
        if name is None:
            m = re.search(r"Card (?:series|model)\s*:\s*(.+)", line)
            if m:
                name = m.group(1).strip()
        m = re.search(r"VRAM Total Memory\s*\(B\)\s*:\s*(\d+)", line)
        if m:
            vram_mb = int(m.group(1)) // (1024 * 1024)
    if name is None and vram_mb is None:
        return None
    return (name or "AMD GPU", vram_mb)


def _detect_amd() -> Optional[tuple[str, Optional[int]]]:
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])
    if out is None:
        return None
    return _parse_rocm_smi(out)


# ---------------------------------------------------------------------------
# Apple (system_profiler)
# ---------------------------------------------------------------------------


def _parse_system_profiler(out: str) -> Optional[tuple[GPUVendor, str, Optional[int]]]:
    """Parse `system_profiler SPDisplaysDataType` output. Apple Silicon
    entries report a 'Chipset Model: Apple M-series' line and (on newer
    macOS) no separate VRAM line at all since GPU memory is unified with
    system RAM -- the caller fills vram_mb from ram_mb in that case."""
    name: Optional[str] = None
    vram_mb: Optional[int] = None
    for raw in out.splitlines():
        line = raw.strip()
        m = re.match(r"Chipset Model:\s*(.+)", line)
        if m and name is None:
            name = m.group(1).strip()
        m = re.match(r"VRAM \((?:Total|Dynamic, Max)\):\s*([\d.]+)\s*(MB|GB)", line, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            vram_mb = int(value * 1024) if m.group(2).upper() == "GB" else int(value)
    if name is None:
        return None
    upper = name.upper()
    if name.startswith("Apple"):
        vendor = GPUVendor.APPLE
    elif "AMD" in upper or "RADEON" in upper:
        vendor = GPUVendor.AMD
    elif "NVIDIA" in upper:
        vendor = GPUVendor.NVIDIA
    else:
        # Intel integrated graphics or unrecognized chipset: no GPUVendor
        # maps to it, treat as CPU-only.
        return None
    return (vendor, name, vram_mb)


def _detect_apple() -> Optional[tuple[GPUVendor, str, Optional[int]]]:
    out = _run(["system_profiler", "SPDisplaysDataType"])
    if out is None:
        return None
    return _parse_system_profiler(out)


# ---------------------------------------------------------------------------
# CPU / RAM (platform fallback, always available)
# ---------------------------------------------------------------------------


def _parse_meminfo_kb(text: str) -> Optional[int]:
    """Parse the 'MemTotal:  16384000 kB' line from /proc/meminfo."""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            m = re.search(r"(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def _detect_ram_mb() -> int:
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                kb = _parse_meminfo_kb(f.read())
            if kb is not None:
                return kb // 1024
        except OSError:
            pass
    elif system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                return int(out.strip()) // (1024 * 1024)
            except ValueError:
                pass
    elif system == "Windows":
        # stdlib-only Windows RAM query via ctypes; if this ever
        # needs more precision (e.g. per-NUMA-node), swap in `psutil`.
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys) // (1024 * 1024)
        except Exception:
            pass
    return 0  # detection failed / unknown platform, caller treats 0 as "unknown"


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def detect_hardware() -> HardwareProfile:
    """Auto-detect GPU vendor/name/VRAM (nvidia-smi -> rocm-smi ->
    system_profiler, in that priority order) and CPU core count / RAM
    (platform/os, per-OS), falling back to CPU-only when nothing is found."""
    os_name = platform.system().lower()
    cpu_cores = os.cpu_count() or 1
    ram_mb = _detect_ram_mb()

    gpu_vendor = GPUVendor.NONE
    gpu_name: Optional[str] = None
    vram_mb: Optional[int] = None

    nvidia = _detect_nvidia()
    if nvidia is not None:
        gpu_vendor = GPUVendor.NVIDIA
        gpu_name, vram_mb = nvidia
    else:
        amd = _detect_amd()
        if amd is not None:
            gpu_vendor = GPUVendor.AMD
            gpu_name, vram_mb = amd
        elif os_name == "darwin":
            apple = _detect_apple()
            if apple is not None:
                gpu_vendor, gpu_name, vram_mb = apple
                if gpu_vendor == GPUVendor.APPLE and vram_mb is None:
                    # Apple Silicon unified memory has no separate
                    # VRAM figure; approximate with total system RAM. Upgrade
                    # if a precise unified-memory-GPU-share API shows up.
                    vram_mb = ram_mb

    return HardwareProfile(
        gpu_vendor=gpu_vendor,
        gpu_name=gpu_name,
        vram_mb=vram_mb,
        cpu_cores=cpu_cores,
        ram_mb=ram_mb,
        os_name=os_name,
    )


def parse_manual_hardware(
    gpu_vendor: Optional[str], vram_mb: Optional[int]
) -> HardwareProfile:
    """Build a HardwareProfile from user-supplied --gpu/--vram-mb, merged
    with detect_hardware() results. gpu_vendor=None -> pure auto-detect;
    otherwise the manual value(s) override the corresponding detected
    field(s). gpu_name is kept from detection only if the manual vendor
    matches the detected vendor (otherwise we have no name for it)."""
    base = detect_hardware()
    if gpu_vendor is None and vram_mb is None:
        return base

    vendor = base.gpu_vendor
    if gpu_vendor is not None:
        try:
            vendor = GPUVendor(gpu_vendor.strip().lower())
        except ValueError as exc:
            valid = ", ".join(v.value for v in GPUVendor)
            raise ValueError(
                f"Unknown --gpu value {gpu_vendor!r}; expected one of: {valid}"
            ) from exc

    gpu_name = base.gpu_name if vendor == base.gpu_vendor else None
    final_vram = vram_mb if vram_mb is not None else base.vram_mb

    if vendor == GPUVendor.NONE:
        gpu_name = None
        final_vram = None

    return HardwareProfile(
        gpu_vendor=vendor,
        gpu_name=gpu_name,
        vram_mb=final_vram,
        cpu_cores=base.cpu_cores,
        ram_mb=base.ram_mb,
        os_name=base.os_name,
    )


# ---------------------------------------------------------------------------
# self-check (run: python -m fituna.hardware)
# ---------------------------------------------------------------------------


def _selfcheck() -> None:
    # nvidia-smi csv parsing
    assert _parse_nvidia_csv("NVIDIA GeForce RTX 4090, 24564\n") == (
        "NVIDIA GeForce RTX 4090",
        24564,
    )
    assert _parse_nvidia_csv("") is None

    # rocm-smi parsing (real-world-ish multi-line layout)
    rocm_out = (
        "GPU[0]\t\t: Card series:\t\tRadeon RX 7900 XTX\n"
        "GPU[0]\t\t: VRAM Total Memory (B): 25757220864\n"
    )
    name, vram = _parse_rocm_smi(rocm_out)
    assert name == "Radeon RX 7900 XTX"
    assert vram == 25757220864 // (1024 * 1024)
    assert _parse_rocm_smi("no useful data here\n") is None

    # system_profiler parsing: Apple Silicon (no explicit VRAM line)
    apple_out = "Graphics/Displays:\n    Apple M2 Pro:\n      Chipset Model: Apple M2 Pro\n      Type: GPU\n"
    parsed = _parse_system_profiler(apple_out)
    assert parsed is not None
    assert parsed[0] == GPUVendor.APPLE
    assert parsed[1] == "Apple M2 Pro"
    assert parsed[2] is None  # caller fills from ram_mb

    # system_profiler parsing: discrete AMD card with explicit VRAM
    amd_mac_out = (
        "Graphics/Displays:\n"
        "    AMD Radeon Pro 5500M:\n"
        "      Chipset Model: AMD Radeon Pro 5500M\n"
        "      VRAM (Total): 8 GB\n"
    )
    parsed_amd = _parse_system_profiler(amd_mac_out)
    assert parsed_amd == (GPUVendor.AMD, "AMD Radeon Pro 5500M", 8192)

    # Intel integrated graphics -> unsupported vendor -> None
    intel_out = "Graphics/Displays:\n    Intel Iris Plus:\n      Chipset Model: Intel Iris Plus Graphics 655\n"
    assert _parse_system_profiler(intel_out) is None

    # /proc/meminfo parsing
    assert _parse_meminfo_kb("MemTotal:       16384000 kB\nMemFree: 1000 kB\n") == 16384000
    assert _parse_meminfo_kb("garbage\n") is None

    # manual override merge logic (no real hardware access needed since
    # both fields are supplied explicitly)
    hw = parse_manual_hardware(gpu_vendor="nvidia", vram_mb=12000)
    assert hw.gpu_vendor == GPUVendor.NVIDIA
    assert hw.vram_mb == 12000

    hw_none = parse_manual_hardware(gpu_vendor="none", vram_mb=None)
    assert hw_none.gpu_vendor == GPUVendor.NONE
    assert hw_none.gpu_name is None
    assert hw_none.vram_mb is None

    try:
        parse_manual_hardware(gpu_vendor="bogus", vram_mb=None)
        raise AssertionError("expected ValueError for unknown --gpu value")
    except ValueError:
        pass

    # detect_hardware must always return a usable profile on this machine,
    # never raise, regardless of what's actually installed.
    real = detect_hardware()
    assert real.os_name in {"linux", "darwin", "windows"}
    assert real.cpu_cores >= 1
    assert real.ram_mb >= 0

    print("fituna.hardware self-check: OK")


if __name__ == "__main__":
    _selfcheck()
