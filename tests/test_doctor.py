# SPDX-License-Identifier: MIT
"""Tests for fituna.doctor (the `fituna doctor` subcommand).

Every underlying dependency (subprocess-backed hardware/binaries detection,
filesystem checks) is monkeypatched -- CI has no llama.cpp installed, and
disk-space/permission behavior must not depend on the machine running the
suite. See fituna/hardware.py and tests/test_hardware.py for the same
convention this file follows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fituna import cli, doctor
from fituna.config import BinaryPaths, DoctorCheck, GPUVendor, HardwareProfile


@pytest.fixture(autouse=True)
def _binaries_launch_fine(monkeypatch):
    # Fake paths from find_exe stubs don't exist; tests that care about the
    # launch probe call monkeypatch.undo() first.
    monkeypatch.setattr(doctor, "_launch_error", lambda path: None)


# ---------------------------------------------------------------------------
# the hard requirement: doctor must never raise, even if every underlying
# check is broken (a diagnostic tool that crashes mid-diagnosis defeats its
# own purpose -- this is the property the task brief explicitly calls out).
# ---------------------------------------------------------------------------


def test_run_checks_isolates_a_crash_to_its_own_row(monkeypatch, tmp_path):
    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(doctor.hardware, "detect_hardware", _boom)
    monkeypatch.setattr(doctor.binaries, "find_exe", _boom)
    monkeypatch.setattr(doctor.shutil, "disk_usage", _boom)
    monkeypatch.setattr(doctor.os, "access", _boom)

    checks = doctor.run_checks(None, tmp_path / "out")  # must not raise

    assert len(checks) == 11
    by_name = {c.name: c for c in checks}
    # "python" touches none of the broken dependencies -> still a clean PASS.
    assert by_name["python"].status == "PASS"
    for name in (
        "llama-quantize",
        "llama-bench",
        "llama-perplexity",
        "llama-cli",
        "llama-server",
        "llama.cpp version",
        "hardware",
        "out-dir",
        "disk-space",
    ):
        row = by_name[name]
        assert row.status == "FAIL", (name, row)
        assert "boom" in row.detail
        assert row.remedy is not None

    # no llama-bench path to probe -> skipped, not crashed
    assert by_name["gpu-backend"].status == "WARN"

    # a FAIL on a required binary must still win exit-code precedence (2).
    assert doctor.exit_code(checks) == 2


def test_safe_converts_exception_to_fail_row_not_a_raise():
    def _boom():
        raise ValueError("simulated bug")

    row = doctor._safe("some-check", _boom)
    assert row.name == "some-check"
    assert row.status == "FAIL"
    assert row.detail == "check crashed unexpectedly: simulated bug"
    assert row.remedy is not None


def test_safe_forces_its_name_onto_the_returned_row_even_if_fn_disagrees():
    # the row name has exactly one source of truth: _safe's own argument --
    # not whatever DoctorCheck a check function independently constructs.
    def _mismatched_name():
        return DoctorCheck("wrong-name", "PASS", "ok", None)

    row = doctor._safe("right-name", _mismatched_name)
    assert row.name == "right-name"


def test_safe_converts_non_doctorcheck_return_to_fail_row_not_a_raise():
    # a check function that breaks its own contract (returns something that
    # isn't a DoctorCheck) must still come back as a FAIL row, not raise --
    # the name-reconciliation step (dataclasses.replace) is guarded too, not
    # just fn()'s own body.
    row = doctor._safe("some-check", lambda: None)
    assert row.name == "some-check"
    assert row.status == "FAIL"
    assert "check crashed unexpectedly" in row.detail
    assert row.remedy is not None


# ---------------------------------------------------------------------------
# _check_python
# ---------------------------------------------------------------------------


def test_check_python_pass_when_recent():
    row = doctor._check_python(version_info=(3, 13, 1, "final", 0))
    assert row == DoctorCheck("python", "PASS", "3.13.1", None)


def test_check_python_pass_at_exact_minimum():
    row = doctor._check_python(version_info=(3, 11, 0, "final", 0))
    assert row.status == "PASS"


def test_check_python_fail_when_too_old():
    row = doctor._check_python(version_info=(3, 9, 6, "final", 0))
    assert row.status == "FAIL"
    assert "3.9.6" in row.detail
    assert row.remedy is not None


# ---------------------------------------------------------------------------
# _check_binary / _check_required_binaries
# ---------------------------------------------------------------------------


def test_check_binary_required_pass(monkeypatch):
    monkeypatch.setattr(
        doctor.binaries, "find_exe", lambda name, bin_dir: Path(f"/opt/bin/{name}")
    )
    row = doctor._check_binary("llama-quantize", None, required=True)
    assert row == DoctorCheck(
        "llama-quantize", "PASS", str(Path("/opt/bin/llama-quantize")), None
    )


def test_check_binary_required_fail_mentions_brew_and_bin_dir_flag(monkeypatch):
    monkeypatch.setattr(doctor.binaries, "find_exe", lambda name, bin_dir: None)
    row = doctor._check_binary("llama-bench", None, required=True)
    assert row.status == "FAIL"
    assert "brew install llama.cpp" in row.remedy
    assert "--llama-bin-dir" in row.remedy


def test_check_binary_optional_llama_cli_warns_not_fails(monkeypatch):
    monkeypatch.setattr(doctor.binaries, "find_exe", lambda name, bin_dir: None)
    row = doctor._check_binary("llama-cli", None, required=False)
    assert row.status == "WARN"  # missing llama-cli must never be FAIL
    assert "llama-cli" in row.remedy


def test_check_required_binaries_all_present_returns_rows_and_paths(monkeypatch):
    monkeypatch.setattr(
        doctor.binaries, "find_exe", lambda name, bin_dir: Path(f"/x/{name}")
    )

    rows, paths = doctor._check_required_binaries(None)

    assert rows == [
        DoctorCheck("llama-quantize", "PASS", str(Path("/x/llama-quantize")), None),
        DoctorCheck("llama-bench", "PASS", str(Path("/x/llama-bench")), None),
        DoctorCheck("llama-perplexity", "PASS", str(Path("/x/llama-perplexity")), None),
    ]
    assert paths == BinaryPaths(
        llama_quantize=Path("/x/llama-quantize"),
        llama_bench=Path("/x/llama-bench"),
        llama_perplexity=Path("/x/llama-perplexity"),
    )


def test_check_required_binaries_mixed_pass_and_fail(monkeypatch):
    # only llama-quantize actually resolves; bench/perplexity are missing.
    monkeypatch.setattr(
        doctor.binaries,
        "find_exe",
        lambda name, bin_dir: Path("/x/llama-quantize") if name == "llama-quantize" else None,
    )

    rows, paths = doctor._check_required_binaries(None)

    assert paths is None
    by_name = {r.name: r for r in rows}
    assert by_name["llama-quantize"].status == "PASS"
    assert by_name["llama-quantize"].detail == str(Path("/x/llama-quantize"))
    assert by_name["llama-bench"].status == "FAIL"
    assert by_name["llama-perplexity"].status == "FAIL"


def test_check_required_binaries_survives_a_renamed_required_tuple(monkeypatch):
    # Regression: _check_required_binaries's final BinaryPaths build indexes
    # by_name with the literal names "llama-quantize"/"llama-bench"/
    # "llama-perplexity". If _REQUIRED_BINARIES (doctor.py's alias for
    # fituna.binaries.REQUIRED_BINARIES) is ever renamed/reordered/extended,
    # those rows all still PASS but by_name won't have those exact keys --
    # this must degrade to paths=None, not KeyError out of run_checks.
    monkeypatch.setattr(
        doctor.binaries, "find_exe", lambda name, bin_dir: Path(f"/x/{name}")
    )
    monkeypatch.setattr(
        doctor,
        "_REQUIRED_BINARIES",
        ("llama-quantize-v2", "llama-bench", "llama-perplexity"),
    )

    rows, paths = doctor._check_required_binaries(None)  # must not raise

    assert paths is None
    assert all(row.status == "PASS" for row in rows)


# ---------------------------------------------------------------------------
# _check_llama_version
# ---------------------------------------------------------------------------


def test_llama_version_pass_reuses_supplied_paths_without_extra_lookups(monkeypatch):
    fake_paths = BinaryPaths(
        llama_quantize=Path("/x/llama-quantize"),
        llama_bench=Path("/x/llama-bench"),
        llama_perplexity=Path("/x/llama-perplexity"),
    )

    def _must_not_be_called(*_a, **_k):
        raise AssertionError("must not re-probe when paths were already resolved")

    monkeypatch.setattr(doctor.binaries, "find_exe", _must_not_be_called)
    monkeypatch.setattr(
        doctor.binaries,
        "get_llama_cpp_version",
        lambda paths: "9960 (a935fbffe)" if paths is fake_paths else None,
    )

    row = doctor._check_llama_version(None, fake_paths)
    assert row == DoctorCheck("llama.cpp version", "PASS", "9960 (a935fbffe)", None)


def test_llama_version_warn_when_none_uses_brief_wording_verbatim(monkeypatch):
    fake_paths = BinaryPaths(
        llama_quantize=Path("/x/q"), llama_bench=Path("/x/b"), llama_perplexity=Path("/x/p")
    )
    monkeypatch.setattr(doctor.binaries, "get_llama_cpp_version", lambda paths: None)

    row = doctor._check_llama_version(None, fake_paths)
    assert row.status == "WARN"
    assert row.detail == "could not be detected"
    assert row.remedy == (
        'bench cache falls back to "unknown"; results from different builds '
        "may be reused. Upgrade llama.cpp or pass --llama-bin-dir."
    )


def test_llama_version_falls_back_to_individual_probe_when_paths_missing(monkeypatch):
    # required-binaries check failed (paths=None), but llama-perplexity
    # individually still resolves -- version check should still work,
    # reusing that one real path for both required BinaryPaths fields.
    monkeypatch.setattr(
        doctor.binaries,
        "find_exe",
        lambda name, bin_dir: Path("/x/llama-perplexity") if name == "llama-perplexity" else None,
    )

    seen = {}

    def _fake_get_version(paths):
        seen["paths"] = paths
        return "1234 (deadbee)"

    monkeypatch.setattr(doctor.binaries, "get_llama_cpp_version", _fake_get_version)

    row = doctor._check_llama_version(None, None)
    assert row.status == "PASS"
    assert row.detail == "1234 (deadbee)"
    # the missing llama-bench field was filled with the one real path found,
    # not a fabricated/nonexistent one.
    assert str(seen["paths"].llama_bench) == str(Path("/x/llama-perplexity"))
    assert str(seen["paths"].llama_perplexity) == str(Path("/x/llama-perplexity"))


def test_llama_version_warn_when_nothing_resolvable_and_paths_none(monkeypatch):
    monkeypatch.setattr(doctor.binaries, "find_exe", lambda name, bin_dir: None)

    def _must_not_be_called(_paths):
        raise AssertionError("get_llama_cpp_version must not be called with nothing to probe")

    monkeypatch.setattr(doctor.binaries, "get_llama_cpp_version", _must_not_be_called)

    row = doctor._check_llama_version(None, None)
    assert row.status == "WARN"
    assert row.detail == "could not be detected"


# ---------------------------------------------------------------------------
# _check_hardware
# ---------------------------------------------------------------------------


def test_hardware_pass_when_gpu_detected(monkeypatch):
    monkeypatch.setattr(
        doctor.hardware,
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
    row = doctor._check_hardware()
    assert row.status == "PASS"
    assert "gpu=nvidia" in row.detail
    assert "NVIDIA GeForce RTX 4090" in row.detail
    assert "cpu=16 cores" in row.detail


def test_hardware_warn_when_no_gpu_mentions_manual_override_flags(monkeypatch):
    monkeypatch.setattr(
        doctor.hardware,
        "detect_hardware",
        lambda: HardwareProfile(
            gpu_vendor=GPUVendor.NONE,
            gpu_name=None,
            vram_mb=None,
            cpu_cores=4,
            ram_mb=8192,
            os_name="linux",
        ),
    )
    row = doctor._check_hardware()
    assert row.status == "WARN"
    assert "gpu=none" in row.detail
    assert "--gpu" in row.remedy and "--vram-mb" in row.remedy


# ---------------------------------------------------------------------------
# _check_out_dir
# ---------------------------------------------------------------------------


def test_out_dir_pass_when_exists_and_writable(tmp_path):
    row = doctor._check_out_dir(tmp_path)
    assert row.status == "PASS"


def test_out_dir_fail_when_not_writable(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.os, "access", lambda *_a, **_k: False)
    row = doctor._check_out_dir(tmp_path)
    assert row.status == "FAIL"
    assert row.remedy is not None


def test_out_dir_pass_when_missing_but_parent_writable(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    row = doctor._check_out_dir(target)
    assert row.status == "PASS"
    assert "does not exist yet" in row.detail
    # brief's hard requirement: checking must never actually create it.
    assert not target.exists()
    assert not (tmp_path / "does").exists()


def test_out_dir_never_creates_the_directory_even_on_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.os, "access", lambda *_a, **_k: False)
    target = tmp_path / "missing"
    doctor._check_out_dir(target)
    assert not target.exists()


def test_out_dir_fail_when_path_is_a_file_not_a_directory(tmp_path):
    conflicting_file = tmp_path / "out"
    conflicting_file.write_text("not a directory")
    row = doctor._check_out_dir(conflicting_file)
    assert row.status == "FAIL"
    assert "not a directory" in row.detail


def test_out_dir_fail_when_an_ancestor_is_a_file(tmp_path):
    # os.access(<file>, W_OK) is True, so the nearest existing ancestor being
    # a regular file used to report PASS; creating it would fail with ENOTDIR.
    blocking_file = tmp_path / "somefile.txt"
    blocking_file.write_text("not a directory")
    row = doctor._check_out_dir(blocking_file / "out")
    assert row.status == "FAIL"
    assert row.remedy is not None
    assert "not a directory" in row.detail
    assert not (blocking_file / "out").exists()


# ---------------------------------------------------------------------------
# _check_disk_space
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, free_bytes: int):
        self.free = free_bytes
        self.total = free_bytes * 2
        self.used = free_bytes


def test_disk_space_pass_when_plenty_free(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: _FakeUsage(100 * 1024**3))
    row = doctor._check_disk_space(tmp_path)
    assert row.status == "PASS"
    assert "100.0 GB free" in row.detail


def test_disk_space_warn_when_below_threshold_mentions_rationale(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: _FakeUsage(5 * 1024**3))
    row = doctor._check_disk_space(tmp_path)
    assert row.status == "WARN"
    assert "12 GB" in row.remedy
    assert "4B model" in row.remedy or "4 candidate" in row.remedy


def test_disk_space_boundary_exactly_20gb_is_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: _FakeUsage(20 * 1024**3))
    row = doctor._check_disk_space(tmp_path)
    assert row.status == "PASS"


def test_disk_space_just_under_threshold_warns(monkeypatch, tmp_path):
    monkeypatch.setattr(
        doctor.shutil, "disk_usage", lambda p: _FakeUsage(int(19.9 * 1024**3))
    )
    row = doctor._check_disk_space(tmp_path)
    assert row.status == "WARN"


# ---------------------------------------------------------------------------
# summarize / exit_code
# ---------------------------------------------------------------------------


def _mk(name, status, remedy=None):
    return DoctorCheck(name, status, "detail", remedy)


def test_summarize_counts_each_status():
    checks = [_mk("a", "PASS"), _mk("b", "PASS"), _mk("c", "WARN"), _mk("d", "FAIL")]
    assert doctor.summarize(checks) == (2, 1, 1)


def test_exit_code_zero_when_no_fail():
    assert doctor.exit_code([_mk("a", "PASS")]) == 0
    assert doctor.exit_code([_mk("a", "PASS"), _mk("hardware", "WARN")]) == 0


def test_exit_code_one_for_non_binary_fail():
    # a FAIL row whose name isn't one of the three required binaries maps to
    # 1 -- including a stray "llama-cli" FAIL: llama-cli is WARN-only in
    # normal operation, but exit_code's binary check is scoped to the
    # required three regardless of what else might show up as FAIL.
    assert doctor.exit_code([_mk("python", "FAIL"), _mk("a", "PASS")]) == 1
    assert doctor.exit_code([_mk("llama-cli", "FAIL")]) == 1


def test_exit_code_two_for_binary_fail_takes_precedence():
    assert doctor.exit_code([_mk("llama-bench", "FAIL")]) == 2
    # a required-binary FAIL wins even alongside another, unrelated FAIL.
    assert doctor.exit_code([_mk("python", "FAIL"), _mk("llama-quantize", "FAIL")]) == 2


# ---------------------------------------------------------------------------
# to_human / to_json formatting
# ---------------------------------------------------------------------------


def test_to_human_matches_brief_example_exactly():
    checks = [
        DoctorCheck("python", "PASS", "3.13.1", None),
        DoctorCheck("llama-quantize", "PASS", "/opt/homebrew/bin/llama-quantize", None),
        DoctorCheck(
            "llama.cpp version",
            "WARN",
            "could not be detected",
            'bench cache falls back to "unknown"; results from different builds '
            "may be reused. Upgrade llama.cpp or pass --llama-bin-dir.",
        ),
    ]
    human = doctor.to_human(checks)
    expected_rows = (
        "FiTuna doctor\n"
        "  [PASS] python            3.13.1\n"
        "  [PASS] llama-quantize    /opt/homebrew/bin/llama-quantize\n"
        "  [WARN] llama.cpp version could not be detected\n"
        '         -> bench cache falls back to "unknown"; results from different\n'
        "            builds may be reused. Upgrade llama.cpp or pass --llama-bin-dir.\n"
        "\n"
        "2 checks passed, 1 warning, 0 failed.\n"
        "Next: run `fituna quickstart` for a guided first run, or see `fituna run -h`."
    )
    assert human == expected_rows


def test_to_human_pluralizes_checks_and_warnings():
    checks = [_mk("a", "PASS"), _mk("b", "PASS"), _mk("c", "WARN"), _mk("d", "WARN")]
    human = doctor.to_human(checks)
    assert "2 checks passed, 2 warnings, 0 failed." in human


def test_to_json_schema_and_null_remedy_for_pass():
    checks = [_mk("a", "PASS"), _mk("b", "WARN", remedy="fix b"), _mk("c", "FAIL", remedy="fix c")]
    payload = json.loads(doctor.to_json(checks))

    assert payload["summary"] == {"passed": 1, "warned": 1, "failed": 1}
    assert payload["checks"] == [
        {"name": "a", "status": "PASS", "detail": "detail", "remedy": None},
        {"name": "b", "status": "WARN", "detail": "detail", "remedy": "fix b"},
        {"name": "c", "status": "FAIL", "detail": "detail", "remedy": "fix c"},
    ]


# ---------------------------------------------------------------------------
# CLI wiring (fituna/cli.py's `doctor` subcommand)
# ---------------------------------------------------------------------------


def test_cli_doctor_exit_code_matches_doctor_module(monkeypatch):
    fixed = [DoctorCheck("llama-quantize", "FAIL", "not found", "install it")]
    monkeypatch.setattr(cli.doctor, "run_checks", lambda bin_dir, out_dir: fixed)
    assert cli.main(["doctor"]) == 2


def test_cli_doctor_json_flag_selects_json_output(monkeypatch, capsys):
    fixed = [DoctorCheck("a", "PASS", "ok", None)]
    monkeypatch.setattr(cli.doctor, "run_checks", lambda bin_dir, out_dir: fixed)
    code = cli.main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["summary"] == {"passed": 1, "warned": 0, "failed": 0}


def test_cli_doctor_passes_llama_bin_dir_and_out(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_checks(bin_dir, out_dir):
        captured["bin_dir"] = bin_dir
        captured["out_dir"] = out_dir
        return [DoctorCheck("x", "PASS", "ok", None)]

    monkeypatch.setattr(cli.doctor, "run_checks", _fake_run_checks)
    cli.main(
        ["doctor", "--llama-bin-dir", str(tmp_path), "--out", str(tmp_path / "o")]
    )
    assert captured["bin_dir"] == tmp_path
    assert captured["out_dir"] == tmp_path / "o"


def test_cli_doctor_defaults_llama_bin_dir_none_and_out_to_dot_out(monkeypatch):
    captured = {}

    def _fake_run_checks(bin_dir, out_dir):
        captured["bin_dir"] = bin_dir
        captured["out_dir"] = out_dir
        return [DoctorCheck("x", "PASS", "ok", None)]

    monkeypatch.setattr(cli.doctor, "run_checks", _fake_run_checks)
    cli.main(["doctor"])
    assert captured["bin_dir"] is None
    assert captured["out_dir"] == Path("./out")


def test_cli_doctor_smoke_never_raises_on_real_environment(capsys):
    """Whatever this machine actually has installed, `fituna doctor` must
    never raise and must return one of doctor's documented exit codes --
    mirrors test_hardware.py's own "never raises regardless of what's
    installed" real-environment smoke test."""
    code = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert code in (0, 1, 2)
    assert out.startswith("FiTuna doctor")
    assert "passed" in out and "failed" in out


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))


# ---------------------------------------------------------------------------
# launch probe: a binary on PATH that cannot actually start (missing dylib,
# macOS quarantine kill) used to show up as PASS
# ---------------------------------------------------------------------------


def test_required_binary_that_cannot_start_is_a_fail(monkeypatch):
    monkeypatch.setattr(doctor.binaries, "find_exe", lambda name, bin_dir: Path("/x/llama-bench"))
    monkeypatch.setattr(
        doctor, "_launch_error", lambda path: "killed by signal 6: dyld: Library not loaded"
    )

    row = doctor._check_binary("llama-bench", None, required=True)

    assert row.status == "FAIL"
    assert "does not run" in row.detail and "dyld" in row.detail
    assert "quarantine" in row.remedy


def test_optional_binary_that_cannot_start_is_only_a_warn(monkeypatch):
    monkeypatch.setattr(doctor.binaries, "find_exe", lambda name, bin_dir: Path("/x/llama-server"))
    monkeypatch.setattr(doctor, "_launch_error", lambda path: "boom")
    assert doctor._check_binary("llama-server", None, required=False).status == "WARN"


def _script(tmp_path: Path, body: str) -> Path:
    exe = tmp_path / "fake"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)
    return exe


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell scripts")
def test_launch_error_ignores_a_nonzero_exit_code(tmp_path, monkeypatch):
    monkeypatch.undo()  # real probe
    # llama-quantize --version exits 1 on a healthy install.
    assert doctor._launch_error(_script(tmp_path, "exit 1\n")) is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_launch_error_reports_a_signal_kill(tmp_path, monkeypatch):
    monkeypatch.undo()
    exe = _script(tmp_path, "echo 'dyld: Library not loaded' >&2\necho '  Reason: tried: x' >&2\nkill -ABRT $$\n")
    err = doctor._launch_error(exe)
    assert err is not None and "signal" in err and "dyld" in err


def test_launch_error_reports_a_file_that_cannot_execute(tmp_path, monkeypatch):
    monkeypatch.undo()
    exe = tmp_path / "fake"
    exe.write_text("not a program")  # no exec bit
    assert doctor._launch_error(exe) is not None


# ---------------------------------------------------------------------------
# gpu-backend: a GPU on the machine but a CPU-only llama.cpp build means -ngl
# does nothing and every measured speed is CPU speed
# ---------------------------------------------------------------------------

_METAL_DEVICES = """\
ggml_metal_device_init: use shared buffers    = true
Available devices:
  MTL0: Apple M3 Pro (13639 MiB, 13639 MiB free)
  BLAS: Accelerate (0 MiB, 0 MiB free)
"""
_CPU_ONLY_DEVICES = "Available devices:\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n"


def test_parse_gpu_devices_skips_cpu_side_backends():
    assert doctor._parse_gpu_devices(_METAL_DEVICES) == [
        "MTL0: Apple M3 Pro (13639 MiB, 13639 MiB free)"
    ]
    assert doctor._parse_gpu_devices(_CPU_ONLY_DEVICES) == []
    assert doctor._parse_gpu_devices("Available devices:\n") == []


def _hw(vendor: GPUVendor) -> HardwareProfile:
    return HardwareProfile(
        gpu_vendor=vendor, gpu_name=None, vram_mb=None, cpu_cores=8, ram_mb=16384, os_name="darwin"
    )


def _paths() -> BinaryPaths:
    return BinaryPaths(Path("/x/q"), Path("/x/llama-bench"), Path("/x/p"))


@pytest.mark.parametrize(
    "vendor, listing, status",
    [
        (GPUVendor.APPLE, _METAL_DEVICES, "PASS"),
        (GPUVendor.APPLE, _CPU_ONLY_DEVICES, "WARN"),
        (GPUVendor.NONE, _CPU_ONLY_DEVICES, "PASS"),
        (GPUVendor.NVIDIA, None, "WARN"),  # --list-devices unsupported or failed
    ],
)
def test_gpu_backend_status(monkeypatch, vendor, listing, status):
    monkeypatch.setattr(doctor.hardware, "detect_hardware", lambda: _hw(vendor))
    monkeypatch.setattr(doctor, "_list_devices", lambda bench: listing)

    row = doctor._check_gpu_backend(_paths())

    assert row.status == status, row
    if status == "WARN":
        assert row.remedy


def test_gpu_backend_without_llama_bench_is_a_warn_not_a_crash(monkeypatch):
    monkeypatch.setattr(doctor.hardware, "detect_hardware", lambda: _hw(GPUVendor.APPLE))
    row = doctor._check_gpu_backend(None)
    assert row.status == "WARN" and "llama-bench" in row.detail


# ---------------------------------------------------------------------------
# next-step line
# ---------------------------------------------------------------------------


def test_to_human_next_step_names_the_first_fail():
    human = doctor.to_human(
        [_mk("a", "PASS"), _mk("llama-bench", "FAIL", remedy="x"), _mk("z", "FAIL", remedy="y")]
    )
    assert human.splitlines()[-1].startswith("Next: fix llama-bench first")
