# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FiTuna contributors
"""Hardware detection tests.

Exercises fituna.hardware by monkeypatching subprocess.run so no real
nvidia-smi/rocm-smi/system_profiler binary (or GPU) is required. Covers:
- the pure parsing helpers (_parse_nvidia_csv/_parse_rocm_smi/
  _parse_system_profiler/_parse_meminfo_kb) with realistic sample output,
- detect_hardware()'s vendor-priority fallthrough (nvidia -> amd -> apple ->
  cpu-only) driven entirely through a faked subprocess.run,
- parse_manual_hardware()'s override-merge semantics against a faked
  detect_hardware() baseline.
"""

from __future__ import annotations

import subprocess

import pytest

from fituna.config import GPUVendor
from fituna import hardware
from fituna.hardware import (
    _parse_meminfo_kb,
    _parse_nvidia_csv,
    _parse_rocm_smi,
    _parse_system_profiler,
    detect_hardware,
    parse_manual_hardware,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def _fake_run_factory(responses: dict[str, str]):
    """Build a subprocess.run stand-in. `responses` maps the command's
    argv[0] (the binary name) to stdout it should return; any binary not
    present in the mapping raises FileNotFoundError, matching real
    subprocess.run behaviour for a missing executable."""

    def _fake_run(cmd, capture_output=True, text=True, timeout=5.0, check=False, **kwargs):
        binary = cmd[0]
        if binary not in responses:
            raise FileNotFoundError(f"no such file: {binary}")
        return _FakeCompleted(stdout=responses[binary], returncode=0)

    return _fake_run


# ---------------------------------------------------------------------------
# pure parsing helpers
# ---------------------------------------------------------------------------


def test_parse_nvidia_csv_happy_path():
    assert _parse_nvidia_csv("NVIDIA GeForce RTX 4090, 24564\n") == (
        "NVIDIA GeForce RTX 4090",
        24564,
    )


def test_parse_nvidia_csv_empty_is_none():
    assert _parse_nvidia_csv("") is None
    assert _parse_nvidia_csv("\n\n") is None


def test_parse_rocm_smi_multiline():
    out = (
        "GPU[0]\t\t: Card series:\t\tRadeon RX 7900 XTX\n"
        "GPU[0]\t\t: VRAM Total Memory (B): 25757220864\n"
    )
    name, vram = _parse_rocm_smi(out)
    assert name == "Radeon RX 7900 XTX"
    assert vram == 25757220864 // (1024 * 1024)


def test_parse_rocm_smi_no_useful_data():
    assert _parse_rocm_smi("no useful data here\n") is None


def test_parse_system_profiler_apple_silicon_no_vram_line():
    out = (
        "Graphics/Displays:\n"
        "    Apple M2 Pro:\n"
        "      Chipset Model: Apple M2 Pro\n"
        "      Type: GPU\n"
    )
    parsed = _parse_system_profiler(out)
    assert parsed == (GPUVendor.APPLE, "Apple M2 Pro", None)


def test_parse_system_profiler_amd_discrete_with_vram():
    out = (
        "Graphics/Displays:\n"
        "    AMD Radeon Pro 5500M:\n"
        "      Chipset Model: AMD Radeon Pro 5500M\n"
        "      VRAM (Total): 8 GB\n"
    )
    assert _parse_system_profiler(out) == (GPUVendor.AMD, "AMD Radeon Pro 5500M", 8192)


def test_parse_system_profiler_intel_unsupported_vendor():
    out = (
        "Graphics/Displays:\n"
        "    Intel Iris Plus:\n"
        "      Chipset Model: Intel Iris Plus Graphics 655\n"
    )
    assert _parse_system_profiler(out) is None


def test_parse_meminfo_kb():
    assert _parse_meminfo_kb("MemTotal:       16384000 kB\nMemFree: 1000 kB\n") == 16384000
    assert _parse_meminfo_kb("garbage\n") is None


# ---------------------------------------------------------------------------
# detect_hardware(): vendor-priority fallthrough via monkeypatched subprocess
# ---------------------------------------------------------------------------


def test_detect_hardware_nvidia_path(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run_factory(
            {"nvidia-smi": "NVIDIA GeForce RTX 4090, 24564\n"}
        ),
    )
    # Linux RAM path reads /proc/meminfo directly; avoid touching the real
    # filesystem file by monkeypatching the parser's input via builtins.open.
    monkeypatch.setattr(
        hardware, "_detect_ram_mb", lambda: 32000
    )

    hw = detect_hardware()
    assert hw.gpu_vendor == GPUVendor.NVIDIA
    assert hw.gpu_name == "NVIDIA GeForce RTX 4090"
    assert hw.vram_mb == 24564
    assert hw.cpu_cores == 16
    assert hw.ram_mb == 32000
    assert hw.os_name == "linux"


def test_detect_hardware_falls_through_to_amd_when_no_nvidia(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(hardware, "_detect_ram_mb", lambda: 16000)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run_factory(
            {
                "rocm-smi": (
                    "GPU[0]\t\t: Card series:\t\tRadeon RX 7900 XTX\n"
                    "GPU[0]\t\t: VRAM Total Memory (B): 25757220864\n"
                )
            }
        ),
    )

    hw = detect_hardware()
    assert hw.gpu_vendor == GPUVendor.AMD
    assert hw.gpu_name == "Radeon RX 7900 XTX"
    assert hw.vram_mb == 25757220864 // (1024 * 1024)


def test_detect_hardware_apple_silicon_uses_unified_memory_for_vram(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(hardware, "_detect_ram_mb", lambda: 32768)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run_factory(
            {
                "system_profiler": (
                    "Graphics/Displays:\n"
                    "    Apple M2 Pro:\n"
                    "      Chipset Model: Apple M2 Pro\n"
                    "      Type: GPU\n"
                )
            }
        ),
    )

    hw = detect_hardware()
    assert hw.gpu_vendor == GPUVendor.APPLE
    assert hw.gpu_name == "Apple M2 Pro"
    # No explicit VRAM line -> falls back to total system RAM (unified memory).
    assert hw.vram_mb == 32768
    assert hw.os_name == "darwin"


def test_detect_hardware_cpu_only_when_nothing_found(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(hardware, "_detect_ram_mb", lambda: 8192)

    def _always_missing(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", _always_missing)

    hw = detect_hardware()
    assert hw.gpu_vendor == GPUVendor.NONE
    assert hw.gpu_name is None
    assert hw.vram_mb is None
    assert hw.cpu_cores == 4
    assert hw.ram_mb == 8192


def test_detect_hardware_non_gpu_return_code_falls_back(monkeypatch):
    """A binary that exists but exits non-zero (e.g. no GPU present) must be
    treated the same as a missing binary, never raise."""
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(hardware, "_detect_ram_mb", lambda: 4096)

    def _nonzero_exit(cmd, **kwargs):
        return _FakeCompleted(stdout="", returncode=1)

    monkeypatch.setattr(subprocess, "run", _nonzero_exit)

    hw = detect_hardware()
    assert hw.gpu_vendor == GPUVendor.NONE


def test_detect_hardware_never_raises_on_real_machine():
    """Smoke test with the real subprocess plumbing (whatever is actually
    installed on this machine): must always return a usable profile."""
    hw = detect_hardware()
    assert hw.os_name in {"linux", "darwin", "windows"}
    assert hw.cpu_cores >= 1
    assert hw.ram_mb >= 0


# ---------------------------------------------------------------------------
# parse_manual_hardware(): override-merge semantics
# ---------------------------------------------------------------------------


def test_manual_gpu_overrides_detection(monkeypatch):
    from fituna.config import HardwareProfile

    monkeypatch.setattr(
        hardware,
        "detect_hardware",
        lambda: HardwareProfile(
            gpu_vendor=GPUVendor.NONE,
            gpu_name=None,
            vram_mb=None,
            cpu_cores=8,
            ram_mb=16000,
            os_name="linux",
        ),
    )

    hw = parse_manual_hardware(gpu_vendor="nvidia", vram_mb=12000)
    assert hw.gpu_vendor == GPUVendor.NVIDIA
    assert hw.vram_mb == 12000
    # detected vendor (none) != manual vendor (nvidia) -> no name available.
    assert hw.gpu_name is None
    assert hw.cpu_cores == 8
    assert hw.ram_mb == 16000


def test_manual_none_clears_gpu_fields(monkeypatch):
    from fituna.config import HardwareProfile

    monkeypatch.setattr(
        hardware,
        "detect_hardware",
        lambda: HardwareProfile(
            gpu_vendor=GPUVendor.NVIDIA,
            gpu_name="NVIDIA GeForce RTX 4090",
            vram_mb=24564,
            cpu_cores=16,
            ram_mb=32000,
            os_name="linux",
        ),
    )

    hw = parse_manual_hardware(gpu_vendor="none", vram_mb=None)
    assert hw.gpu_vendor == GPUVendor.NONE
    assert hw.gpu_name is None
    assert hw.vram_mb is None


def test_manual_no_overrides_returns_detection_verbatim(monkeypatch):
    from fituna.config import HardwareProfile

    detected = HardwareProfile(
        gpu_vendor=GPUVendor.AMD,
        gpu_name="Radeon RX 7900 XTX",
        vram_mb=24560,
        cpu_cores=12,
        ram_mb=65536,
        os_name="linux",
    )
    monkeypatch.setattr(hardware, "detect_hardware", lambda: detected)

    hw = parse_manual_hardware(gpu_vendor=None, vram_mb=None)
    assert hw == detected


def test_manual_vram_only_override_keeps_detected_vendor_and_name(monkeypatch):
    from fituna.config import HardwareProfile

    monkeypatch.setattr(
        hardware,
        "detect_hardware",
        lambda: HardwareProfile(
            gpu_vendor=GPUVendor.NVIDIA,
            gpu_name="NVIDIA GeForce RTX 4090",
            vram_mb=24564,
            cpu_cores=16,
            ram_mb=32000,
            os_name="linux",
        ),
    )

    hw = parse_manual_hardware(gpu_vendor=None, vram_mb=20000)
    assert hw.gpu_vendor == GPUVendor.NVIDIA
    assert hw.gpu_name == "NVIDIA GeForce RTX 4090"
    assert hw.vram_mb == 20000


def test_manual_unknown_gpu_value_raises_value_error():
    with pytest.raises(ValueError, match="Unknown --gpu value"):
        parse_manual_hardware(gpu_vendor="bogus", vram_mb=None)


def test_manual_matching_vendor_keeps_detected_name(monkeypatch):
    """When the manual --gpu vendor matches what was actually detected, the
    detected gpu_name should be preserved (not dropped)."""
    from fituna.config import HardwareProfile

    monkeypatch.setattr(
        hardware,
        "detect_hardware",
        lambda: HardwareProfile(
            gpu_vendor=GPUVendor.AMD,
            gpu_name="Radeon RX 7900 XTX",
            vram_mb=24560,
            cpu_cores=12,
            ram_mb=65536,
            os_name="linux",
        ),
    )

    hw = parse_manual_hardware(gpu_vendor="amd", vram_mb=None)
    assert hw.gpu_vendor == GPUVendor.AMD
    assert hw.gpu_name == "Radeon RX 7900 XTX"
    assert hw.vram_mb == 24560


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
