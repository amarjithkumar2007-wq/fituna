# SPDX-License-Identifier: MIT
"""Tests for fituna.quickstart (the `fituna quickstart` interactive wizard).

Nothing here touches a real TTY, the network, a subprocess, or the
filesystem outside pytest's tmp_path: `input()` is scripted per test,
`sys.stdin` is a fake with a controllable `isatty()`, `doctor.run_checks` /
`hardware.detect_hardware` / `corpus.fetch_corpus` are stubbed, and
`cli._cmd_run` (the real search entry point the wizard calls in-process) is
replaced with a recorder so the assembled argv can be asserted without
running llama.cpp.

The HF search parser is checked against `tests/fixtures/hf_models_search.json`
-- a *real* response captured from the live HuggingFace API on 2026-08-02
(see fituna/quickstart.py's module docstring for the request), not a guess.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from fituna import cli, quickstart
from fituna.config import (
    BenchResult,
    CandidateConfig,
    FiTunaError,
    GPUVendor,
    HardwareProfile,
    NoFeasibleConfigError,
    QualityResult,
    SearchResult,
)

FIXTURE = Path(__file__).parent / "fixtures" / "hf_models_search.json"

_HW = HardwareProfile(
    gpu_vendor=GPUVendor.APPLE,
    gpu_name="Apple M3 Pro",
    vram_mb=18432,
    cpu_cores=11,
    ram_mb=18432,
    os_name="darwin",
)


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class _FakeStdin:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _args(out: Path, llama_bin_dir=None):
    return cli._build_parser().parse_args(
        ["quickstart", "--out", str(out)] + (["--llama-bin-dir", llama_bin_dir] if llama_bin_dir else [])
    )


def _passing_checks():
    from fituna.config import DoctorCheck

    return [DoctorCheck("python", "PASS", "3.13.1", None)]


@pytest.fixture
def wizard(monkeypatch, tmp_path):
    """Drive run_wizard with scripted answers; returns (exit_code, recorded).

    `recorded["argv"]` is the argv the wizard printed AND executed (they are
    the same list by construction -- see quickstart._step_run).
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quickstart.hardware, "detect_hardware", lambda: _HW)
    monkeypatch.setattr(quickstart.doctor, "run_checks", lambda b, o: _passing_checks())

    recorded: dict = {}

    def run(answers, *, stdin_tty=True, cmd_run=None, checks=None):
        it = iter(answers)
        monkeypatch.setattr("sys.stdin", _FakeStdin(stdin_tty))
        monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
        if checks is not None:
            monkeypatch.setattr(quickstart.doctor, "run_checks", lambda b, o: checks)

        def _default_cmd_run(ns):
            recorded["ns"] = ns
            return 0

        monkeypatch.setattr(cli, "_cmd_run", cmd_run or _default_cmd_run)

        # capture the argv the wizard builds, at the single place it is built
        real_build = quickstart.build_run_argv

        def spy_build(**kwargs):
            argv = real_build(**kwargs)
            recorded["argv"] = argv
            return argv

        monkeypatch.setattr(quickstart, "build_run_argv", spy_build)
        return quickstart.run_wizard(_args(tmp_path)), recorded

    return run


# answer scripts: [2/6] preset, [3/6] license, [4/6] model, [5/6] corpus,
# corpus path, Modelfile?, run now?
def _answers(preset="1", license_="1", model="1", corpus="3", corpus_path="c.txt",
             modelfile="n", run="y", extra=()):
    return [preset, license_, model, *extra, corpus, corpus_path, modelfile, run]


# ---------------------------------------------------------------------------
# [1/6] TTY guard + doctor
# ---------------------------------------------------------------------------


def test_no_tty_exits_1_and_points_at_fituna_run(wizard, capsys):
    code, _ = wizard([], stdin_tty=False)
    assert code == 1
    err = capsys.readouterr().err
    assert "TTY" in err
    assert "fituna run" in err


def test_doctor_fail_aborts_with_doctors_own_exit_code(wizard, capsys):
    from fituna.config import DoctorCheck

    missing_binary = [DoctorCheck("llama-quantize", "FAIL", "not found", "Install llama.cpp")]
    code, recorded = wizard([], checks=missing_binary)
    assert code == 2  # doctor's binary-FAIL mapping, not a generic 1
    assert "argv" not in recorded  # nothing past step 1 ran
    assert "Install llama.cpp" in capsys.readouterr().out


def test_doctor_warn_only_continues(wizard, tmp_path):
    from fituna.config import DoctorCheck

    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    warn = [DoctorCheck("hardware", "WARN", "no gpu", "override with --gpu")]
    code, recorded = wizard(_answers(), checks=warn)
    assert code == 0
    assert recorded["argv"][0] == "run"


# ---------------------------------------------------------------------------
# [2/6] targets: presets, manual entry, garbage re-prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "choice,expected",
    [("1", (20.0, 5.0, 4096)), ("2", (30.0, 3.0, 8192)), ("3", (15.0, 5.0, 16384))],
)
def test_each_preset_lands_in_the_run_argv(wizard, tmp_path, choice, expected):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    code, recorded = wizard(_answers(preset=choice))
    assert code == 0
    tps, loss, ctx = expected
    argv = recorded["argv"]
    assert argv[argv.index("--target-tps") + 1] == f"{tps:g}"
    assert argv[argv.index("--max-quality-loss") + 1] == f"{loss:g}"
    assert argv[argv.index("--ctx") + 1] == str(ctx)


def test_manual_entry_with_enter_for_defaults(wizard, tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    # preset "4" = 직접 입력; three empty answers = accept every default
    code, recorded = wizard(_answers(preset="4", extra=()) [:1] + ["", "", ""] + _answers()[1:])
    assert code == 0
    argv = recorded["argv"]
    assert argv[argv.index("--target-tps") + 1] == "20"
    assert argv[argv.index("--max-quality-loss") + 1] == "5"
    assert argv[argv.index("--ctx") + 1] == "4096"


def test_manual_entry_reprompts_on_garbage_and_never_crashes(wizard, tmp_path, capsys):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    answers = (
        ["4"]
        + ["abc", "-5", "0", "25"]  # non-numeric, negative, zero, then valid
        + ["3.5"]
        + ["4096.5", "8192"]  # ctx must be an integer
        + _answers()[1:]
    )
    code, recorded = wizard(answers)
    assert code == 0
    argv = recorded["argv"]
    assert argv[argv.index("--target-tps") + 1] == "25"
    assert argv[argv.index("--max-quality-loss") + 1] == "3.5"
    assert argv[argv.index("--ctx") + 1] == "8192"
    out = capsys.readouterr().out
    assert "숫자를 입력해 주세요" in out
    assert "정수를 입력해 주세요" in out


def test_menu_reprompts_on_out_of_range_and_garbage(wizard, tmp_path, capsys):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    code, _ = wizard(["nope", "99", "1"] + _answers()[1:])
    assert code == 0
    assert "사이의 번호를 입력해 주세요" in capsys.readouterr().out


def test_targets_copy_refuses_speed_prediction(wizard, tmp_path, capsys):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    wizard(_answers())
    out = capsys.readouterr().out
    assert "속도는 예측하지 않습니다" in out
    assert "실측값이 아닙니다" in out  # presets labelled as conventional starting points


# ---------------------------------------------------------------------------
# [3/6] license filter -- the pure predicate, all three needs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", ["cc-by-nc-4.0", "cc-by-nc-sa-4.0", "exaone", "other-research", None])
def test_personal_filters_nothing(slug):
    assert quickstart.license_allows(slug, "personal") is True


@pytest.mark.parametrize("slug", ["cc-by-nc-4.0", "cc-by-nc-sa-4.0", "CC-BY-NC-ND-4.0", "exaone", "research-only"])
def test_commercial_excludes_noncommercial_and_research_only(slug):
    assert quickstart.license_allows(slug, "commercial") is False


@pytest.mark.parametrize("slug", ["mit", "apache-2.0", "bsd-3-clause", "gemma", "llama3.2", None])
def test_commercial_keeps_everything_not_known_to_be_noncommercial(slug):
    # A deny-list on purpose: unknown licenses stay, carrying the
    # uploader-metadata caveat -- the wizard classifies, it does not advise.
    assert quickstart.license_allows(slug, "commercial") is True


@pytest.mark.parametrize("slug", ["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"])
def test_redistribution_allows_only_permissive(slug):
    assert quickstart.license_allows(slug, "redistribution") is True


@pytest.mark.parametrize("slug", ["gemma", "llama3.2", "cc-by-sa-4.0", "exaone", "other", None])
def test_redistribution_excludes_everything_else(slug):
    assert quickstart.license_allows(slug, "redistribution") is False


def test_unknown_need_is_a_programming_error():
    with pytest.raises(ValueError):
        quickstart.license_allows("mit", "whatever")


def test_curated_shortlist_survives_the_strictest_filter_with_its_badge(wizard, tmp_path, capsys):
    # All three curated models are MIT/Apache-2.0, so all three survive even
    # the redistribution filter -- what matters is that each carries the
    # license-evidence badge it actually earned and its measured anchor. The
    # badge is split on purpose: Qwen3 and Midm ship a real LICENSE file that
    # was fetched and compared, SmolLM2 ships none, so only two may claim
    # "원문 확인됨" (see quickstart._LICENSE_TEXT_VERIFIED).
    elsewhere = tmp_path / "models"
    elsewhere.mkdir()
    (elsewhere / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    # no local *.gguf in cwd/out -> menu is 1..3 curated, 4 search, 5 manual
    code, _ = wizard(_answers(license_="3", model="5", extra=(str(elsewhere / "m.gguf"),)))
    assert code == 0
    out = capsys.readouterr().out
    text_verified = [m for m in quickstart.CURATED if m.license_evidence == quickstart._LICENSE_TEXT_VERIFIED]
    metadata_only = [m for m in quickstart.CURATED if m.license_evidence == quickstart._LICENSE_METADATA_ONLY]
    assert text_verified and metadata_only  # the split is real, not vacuous
    assert out.count("라이선스 원문 확인됨") == len(text_verified)
    assert out.count("라이선스 메타데이터만 확인 — 원문 파일 없음") == len(metadata_only)
    assert "기록이지 이 컴퓨터의 예측이 아닙니다" in out
    assert "docs/RESULTS.md Run 5" in out  # measured anchor, labelled as a record


def test_curated_model_over_detected_memory_stays_selectable_and_is_flagged(
    wizard, monkeypatch, tmp_path, capsys
):
    # ~1 GiB of RAM: the 135M SmolLM (~258 MiB) fits, the 4B Qwen3 (~7.5 GiB)
    # and the 2.3B Midm (~4.3 GiB) do not. They are still listed and still
    # selectable: the F16 size is what is being compared, but the file that
    # actually runs is the quantized one, and search.py raises ngl from 0, so
    # "F16 doesn't fit" is not "FiTuna can't do this". Hiding them hid this
    # project's own Run-2 flagship on an 8 GB machine.
    tiny = HardwareProfile(GPUVendor.NONE, None, None, 4, 1024, "linux")
    monkeypatch.setattr(quickstart.hardware, "detect_hardware", lambda: tiny)
    monkeypatch.setattr(quickstart, "_download", lambda url, dest: dest)
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    # no local *.gguf -> menu is 1..3 curated, 4 search, 5 manual. Pick the
    # 4B Qwen3 (option 2) -- the one the old memory filter used to hide.
    code, recorded = wizard(_answers(model="2", extra=("y",)))
    assert code == 0
    out = capsys.readouterr().out
    assert "SmolLM2-135M" in out and "Qwen3-4B" in out and "Midm-2.0" in out
    assert "부족합니다" in out  # the verdict line still flags them
    assert "감지된 메모리보다 큰 모델 2개도 그대로 고를 수 있습니다" in out
    # selecting one prints what the F16 stage actually costs, accurately
    assert "디스크·다운로드 비용은 F16 원본 크기 그대로입니다" in out
    assert "양자화 단계는 F16을 읽으므로" in out
    assert "ngl을 0부터" in out
    argv = recorded["argv"]
    assert argv[argv.index("--model") + 1] == str(tmp_path / "Qwen3-4B-Instruct-2507-F16.gguf")


def test_no_curated_model_fits_detected_memory_all_still_listed(
    wizard, monkeypatch, tmp_path, capsys
):
    # ~1 MiB of RAM: nothing in the curated shortlist fits the arithmetic.
    # The section is not emptied -- all three stay, each flagged 부족합니다.
    starved = HardwareProfile(GPUVendor.NONE, None, None, 4, 1, "linux")
    monkeypatch.setattr(quickstart.hardware, "detect_hardware", lambda: starved)
    elsewhere = tmp_path / "models"
    elsewhere.mkdir()
    (elsewhere / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    # menu is 1..3 curated, 4) search, 5) manual
    code, recorded = wizard(_answers(model="5", extra=(str(elsewhere / "m.gguf"),)))
    assert code == 0
    out = capsys.readouterr().out
    assert "SmolLM2" in out and "Qwen3-4B" in out and "Midm-2.0" in out
    assert out.count("부족합니다") == len(quickstart.CURATED)
    assert f"감지된 메모리보다 큰 모델 {len(quickstart.CURATED)}개도 그대로 고를 수 있습니다" in out
    assert "제외했습니다" not in out
    argv = recorded["argv"]
    assert argv[argv.index("--model") + 1] == str(elsewhere / "m.gguf")


# ---------------------------------------------------------------------------
# [4/6] memory fit, local scan, HF parser
# ---------------------------------------------------------------------------


def test_memory_fit_arithmetic_fits_doesnt_and_unknown():
    gpu = HardwareProfile(GPUVendor.NVIDIA, "RTX 4090", 24564, 16, 65536, "linux")
    assert quickstart.memory_fit(8_051_285_344, gpu) is True
    assert quickstart.memory_fit(30_000_000_000, gpu) is False
    # exactly at the margin boundary: <= passes
    budget = int(24564 * 1024 * 1024 * quickstart.MEMORY_MARGIN)
    assert quickstart.memory_fit(budget, gpu) is True
    assert quickstart.memory_fit(budget + 1, gpu) is False

    blind = HardwareProfile(GPUVendor.NONE, None, None, 4, 0, "linux")
    assert quickstart.memory_fit(1, blind) is None
    assert "판정하지 않습니다" in quickstart._memory_fit_line(1, blind)

    cpu_only = HardwareProfile(GPUVendor.NONE, None, None, 4, 8192, "linux")
    assert quickstart.available_memory_mb(cpu_only) == (8192, "RAM")
    assert quickstart.memory_fit(1_000_000_000, cpu_only) is True


def test_memory_fit_line_states_its_own_assumption():
    gpu = HardwareProfile(GPUVendor.NVIDIA, "RTX 4090", 24564, 16, 65536, "linux")
    line = quickstart._memory_fit_line(4_617_053_184, gpu)
    assert "들어갑니다" in line
    assert "18432 MB" not in line  # this hw has 24564 MB
    # the margin is labelled an assumption, once per menu rather than per model
    assert "실측이 아니며" in quickstart._MEMORY_CAVEAT


def test_local_gguf_scan_selects_a_file_on_disk(wizard, tmp_path):
    (tmp_path / "alpha.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    code, recorded = wizard(_answers(model="1"))
    assert code == 0
    argv = recorded["argv"]
    assert argv[argv.index("--model") + 1] == str(tmp_path / "alpha.gguf")


def test_local_scan_deduplicates_cwd_and_out_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.gguf").write_bytes(b"x")
    found = quickstart._local_ggufs(tmp_path)  # cwd == out dir here
    assert [str(p) for p in found] == [str(tmp_path / "a.gguf")]


def test_local_scan_puts_base_precision_files_first(tmp_path, monkeypatch):
    """--out fills with FiTuna's own quantized artifacts; alphabetical order
    would offer `...-Q2_K.gguf` as the base model, which llama-quantize
    refuses to requantize (hit on a real run)."""
    monkeypatch.chdir(tmp_path)
    for name in ("a-Q2_K.gguf", "z-f16.gguf", "b-Q8_0.gguf"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in quickstart._local_ggufs(tmp_path)] == [
        "z-f16.gguf",
        "a-Q2_K.gguf",
        "b-Q8_0.gguf",
    ]


def test_local_menu_labels_already_quantized_files(wizard, tmp_path, capsys):
    (tmp_path / "m-Q4_K_M.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    code, _ = wizard(_answers())
    assert code == 0
    assert "이미 양자화됨 — run이 경고합니다" in capsys.readouterr().out


def test_manual_path_entry_rejects_a_missing_file_then_accepts(wizard, tmp_path, capsys):
    real = tmp_path / "real.gguf"
    real.write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    # no *.gguf found by the scan? there is one, so the menu is:
    # 1) real.gguf  2) 3) 4) curated  5) search  6) manual
    code, recorded = wizard(
        _answers(model="6", extra=(str(tmp_path / "nope.gguf"), "6", str(real)))
    )
    assert code == 0
    assert "찾지 못했습니다" in capsys.readouterr().out
    argv = recorded["argv"]
    assert argv[argv.index("--model") + 1] == str(real)


def test_hf_parser_against_the_captured_real_response():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    candidates = quickstart.parse_hf_search(payload)
    assert len(candidates) == len(payload)

    by_id = {c.model_id: c for c in candidates}
    exaone = by_id["LGAI-EXAONE/EXAONE-4.0-32B-GGUF"]
    # cardData.license is the placeholder "other"; license_name carries the
    # real one -- reading only `license` would file this under "other".
    assert exaone.license == "exaone"
    assert exaone.license_link == "LICENSE"
    assert exaone.gated is False
    assert exaone.downloads > 0
    assert all(f.endswith(".gguf") for f in exaone.gguf_files)
    assert not any("-of-" in f for f in exaone.gguf_files)  # shards dropped
    assert "EXAONE-4.0-32B-Q4_K_M.gguf" in exaone.gguf_files

    # a repo whose cardData carries no license key at all is a real case in
    # this capture (cardData itself is present -- base_model, tags -- just no
    # license/license_name entry)
    no_card = by_id["mradermacher/EXAONE-4.0-1.2B-abliterated-i1-GGUF"]
    assert no_card.license is None
    assert no_card.base_model == "addansee2/EXAONE-4.0-1.2B-abliterated"

    # ...and the commercial filter rejects every EXAONE-derived row, even
    # this one: its own license metadata is absent, but its model_id and
    # cardData.base_model both name EXAONE -- the disqualifying evidence was
    # in hand, just filed under a different key than `license`.
    commercial = [
        c
        for c in candidates
        if quickstart.license_allows(c.license, "commercial", model_id=c.model_id, base_model=c.base_model)
    ]
    assert commercial == []


def test_relative_license_link_resolves_against_the_repo():
    # The live API returns a bare, repo-relative "LICENSE" for cardData.license_link
    # on many models (EXAONE among them, in this real capture) -- printed as-is it
    # reads as a stray word, not the link the caveat tells the user to open.
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    candidates = {c.model_id: c for c in quickstart.parse_hf_search(payload)}
    exaone = candidates["LGAI-EXAONE/EXAONE-4.0-32B-GGUF"]
    assert exaone.license_link == "LICENSE"
    assert quickstart._resolved_license_link(exaone) == (
        "https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B-GGUF/blob/main/LICENSE"
    )

    # already-absolute links pass through untouched
    absolute = quickstart.HFCandidate("a/b", "mit", "https://example.com/LICENSE", False, 1, ())
    assert quickstart._resolved_license_link(absolute) == "https://example.com/LICENSE"

    # no link at all falls back to the repo's own HF page
    no_link = quickstart.HFCandidate("a/b", "mit", None, False, 1, ())
    assert quickstart._resolved_license_link(no_link) == "https://huggingface.co/a/b"


def test_hf_parser_tolerates_a_shape_change_and_marks_gated():
    assert quickstart.parse_hf_search({"error": "boom"}) == []
    assert quickstart.parse_hf_search([{"no_id": 1}]) == []
    # `gated` is false or a string ("manual"/"auto"), verified on
    # meta-llama/Llama-2-7b-chat-hf -- truthiness, not `is True`.
    [c] = quickstart.parse_hf_search([{"id": "a/b", "gated": "manual", "siblings": []}])
    assert c.gated is True


def test_hf_search_flow_skips_gated_and_fileless_repos(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    answers = iter(["qwen", "1", "1"])
    monkeypatch.setattr(
        quickstart,
        "_hf_search",
        lambda q: [
            quickstart.HFCandidate("gated/repo", "mit", None, True, 10, ("a-f16.gguf",)),
            quickstart.HFCandidate("empty/repo", "mit", None, False, 10, ()),
            quickstart.HFCandidate("ok/repo", "mit", "http://x", False, 99, ("b-Q4.gguf", "b-f16.gguf")),
        ],
    )
    monkeypatch.setattr(quickstart, "_download", lambda url, dest: dest)

    got = quickstart._hf_search_flow("commercial", tmp_path)
    assert str(got) == str(tmp_path / "b-f16.gguf")  # F16 offered first
    out = capsys.readouterr().out
    assert "gated 저장소" in out
    assert "단일 .gguf 파일이 없습니다" in out
    assert "업로더가 모델 카드에 적어 넣은 메타데이터" in out


def test_hf_search_flow_reports_license_excluded_candidates(monkeypatch, tmp_path, capsys):
    # License-filtered hits used to be dropped before the 제외 loop, so a
    # search whose results were all license-ineligible printed nothing about
    # licenses -- the exclusion happened but was never reported.
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    answers = iter(["qwen", "1", "1"])
    monkeypatch.setattr(
        quickstart,
        "_hf_search",
        lambda q: [
            quickstart.HFCandidate("nc/repo", "cc-by-nc-4.0", None, False, 10, ("a-f16.gguf",)),
            quickstart.HFCandidate("ok/repo", "mit", "http://x", False, 99, ("b-f16.gguf",)),
        ],
    )
    monkeypatch.setattr(quickstart, "_download", lambda url, dest: dest)

    got = quickstart._hf_search_flow("commercial", tmp_path)
    assert str(got) == str(tmp_path / "b-f16.gguf")
    out = capsys.readouterr().out
    assert "(제외) nc/repo — 선택하신 라이선스 조건에 맞지 않습니다" in out
    assert "cc-by-nc-4.0" in out


def test_curated_shortlist_matches_ai_model_usage_doc():
    doc = (Path(__file__).parents[1] / "docs" / "AI_MODEL_USAGE.md").read_text(encoding="utf-8")
    for model in quickstart.CURATED:
        assert model.base_model in doc, model.base_model
        assert model.gguf_repo in doc, model.gguf_repo


# ---------------------------------------------------------------------------
# [5/6] corpus
# ---------------------------------------------------------------------------


def test_corpus_preset_calls_fetch_corpus_once(wizard, monkeypatch, tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"x")
    calls: list = []

    def fake_fetch(out, lang="en", progress_cb=None, **kw):
        calls.append((str(out), lang))
        Path(out).write_text("text", encoding="utf-8")
        return 7

    monkeypatch.setattr(quickstart.corpus, "fetch_corpus", fake_fetch)
    code, recorded = wizard(_answers(corpus="2", corpus_path=None)[:3] + ["2", "n", "y"])
    assert code == 0
    assert calls == [("kowiki-corpus.txt", "ko")]
    argv = recorded["argv"]
    assert argv[argv.index("--quality-corpus") + 1] == "kowiki-corpus.txt"


def test_a_failed_corpus_fetch_returns_to_the_menu_instead_of_aborting(
    wizard, monkeypatch, tmp_path, capsys
):
    from fituna.config import FiTunaError

    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    calls: list = []

    def flaky(out, lang="en", progress_cb=None, **kw):
        calls.append(lang)
        raise FiTunaError("could not reach the API")

    monkeypatch.setattr(quickstart.corpus, "fetch_corpus", flaky)
    # try en (fails), try ko (fails), fall back to an existing file
    code, recorded = wizard(_answers()[:3] + ["1", "2", "3", "c.txt", "n", "y"])
    assert code == 0
    assert calls == ["en", "ko"]
    assert "could not reach the API" in capsys.readouterr().out
    argv = recorded["argv"]
    assert argv[argv.index("--quality-corpus") + 1] == str(Path("c.txt"))


def test_a_failed_model_download_returns_to_the_model_menu(wizard, monkeypatch, tmp_path, capsys):
    from fituna.config import FiTunaError

    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    real = tmp_path / "models" / "real.gguf"
    real.parent.mkdir()
    real.write_bytes(b"x")

    def boom(url, dest):
        raise FiTunaError("다운로드에 실패했습니다: connection reset")

    monkeypatch.setattr(quickstart, "_download", boom)
    # no local *.gguf -> 1..3 curated, 4 search, 5 manual.
    # pick curated 1 -> confirm download -> it fails -> menu again -> manual
    code, recorded = wizard(_answers(model="1", extra=("y", "5", str(real))))
    assert code == 0
    assert "connection reset" in capsys.readouterr().out
    argv = recorded["argv"]
    assert argv[argv.index("--model") + 1] == str(real)


def test_corpus_is_not_refetched_when_the_file_already_exists(wizard, monkeypatch, tmp_path, capsys):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "wikitext-2-raw-test.txt").write_text("cached", encoding="utf-8")

    def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("fetch_corpus must not run when the file exists")

    monkeypatch.setattr(quickstart.corpus, "fetch_corpus", boom)
    code, _ = wizard(_answers()[:3] + ["1", "n", "y"])
    assert code == 0
    assert "이미 있습니다" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# [6/6] the assembled command, in-process execution, exit codes
# ---------------------------------------------------------------------------


def test_assembled_argv_parses_cleanly_through_the_real_cli_parser(wizard, tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    code, recorded = wizard(_answers(modelfile="y"))
    assert code == 0

    ns = recorded["ns"]  # the Namespace _cmd_run actually received
    parsed = cli._build_parser().parse_args(recorded["argv"])
    assert vars(parsed) == vars(ns)  # printed argv == executed argv, exactly

    assert ns.command == "run"
    assert ns.model == str(tmp_path / "m.gguf")
    assert ns.wikitext == str(Path("c.txt"))  # entered relative, kept relative
    assert ns.out == str(tmp_path)
    assert ns.resume is True
    assert ns.export_ollama is True


def test_the_command_is_printed_before_it_runs(wizard, tmp_path, capsys):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    _code, recorded = wizard(_answers())
    out = capsys.readouterr().out
    # The wizard prints shlex.join(argv), which equals " ".join on clean
    # POSIX paths but not on Windows, where backslashes in tmp_path trigger
    # quoting -- asserting the plain join broke both windows-latest CI legs.
    assert "fituna " + shlex.join(recorded["argv"]) in out
    assert "다음부터는 이 명령을 직접 쓰시면 됩니다" in out


def test_declining_the_run_still_leaves_the_user_the_command(wizard, tmp_path, capsys):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    code, recorded = wizard(_answers(run="n"))
    assert code == 0
    assert "ns" not in recorded  # nothing was executed
    assert "복사해" in capsys.readouterr().out


def test_run_exit_code_passes_through(wizard, tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    code, _ = wizard(_answers(), cmd_run=lambda ns: 1)
    assert code == 1


def _no_feasible(tmp_path) -> NoFeasibleConfigError:
    cfg = CandidateConfig(quant="Q4_K_M", ngl=33, ctx=4096)
    closest = SearchResult(
        config=cfg,
        bench=BenchResult(cfg, 120.0, 24.53, 2048, "{}"),
        quality=QualityResult("Q4_K_M", 9.02, 8.87, 1.73),
        gguf_path=tmp_path / "m-Q4_K_M.gguf",
        run_command=["llama-cli"],
        meets_target=False,
    )
    return NoFeasibleConfigError("nothing met the target", closest=closest)


def test_exit_3_passthrough_prints_the_measured_lower_target(wizard, tmp_path, capsys):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")

    def raiser(ns):
        raise _no_feasible(tmp_path)

    with pytest.raises(NoFeasibleConfigError):
        wizard(_answers(), cmd_run=raiser)
    out = capsys.readouterr().out
    assert "24.53 tok/s" in out
    assert "Q4_K_M" in out and "ngl=33" in out
    assert "실제로 측정한 값" in out  # measured, not predicted


def test_exit_3_suppresses_a_timed_out_bench_as_the_lower_target(wizard, tmp_path, capsys):
    # search.py records a bench that timed out as gen_tok_per_sec == 0.0 -- a
    # sentinel meaning "never finished", not a measurement. If every
    # candidate times out, `closest` carries that 0.0, and the wizard must
    # not print "목표를 0.00 tok/s 로 낮추면" as though 0.00 had been measured.
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    cfg = CandidateConfig(quant="Q4_K_M", ngl=0, ctx=4096)
    timed_out = NoFeasibleConfigError(
        "nothing met the target",
        closest=SearchResult(
            config=cfg,
            bench=BenchResult(cfg, 0.0, 0.0, None, "bench timed out"),
            quality=QualityResult("Q4_K_M", 9.02, 8.87, 1.73),
            gguf_path=tmp_path / "m-Q4_K_M.gguf",
            run_command=["llama-cli"],
            meets_target=False,
        ),
    )

    def raiser(ns):
        raise timed_out

    with pytest.raises(NoFeasibleConfigError):
        wizard(_answers(), cmd_run=raiser)
    out = capsys.readouterr().out
    assert "0.00" not in out  # the timeout sentinel, never printed as a measurement
    assert "타임아웃" in out


def test_exit_3_reaches_cli_main_as_exit_code_3(monkeypatch, tmp_path):
    """The wizard re-raises rather than handling exit 3 itself, so cli.main()
    does the mapping (and the best-effort report + Modelfile export) exactly
    as it does for `fituna run`."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")
    monkeypatch.setattr(quickstart.hardware, "detect_hardware", lambda: _HW)
    monkeypatch.setattr(quickstart.doctor, "run_checks", lambda b, o: _passing_checks())
    monkeypatch.setattr("sys.stdin", _FakeStdin(True))
    answers = iter(_answers(modelfile="y"))
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    exported: list = []

    def raiser(ns):
        raise _no_feasible(tmp_path)

    monkeypatch.setattr(cli, "_cmd_run", raiser)
    monkeypatch.setattr(
        cli.report, "export_ollama_modelfile", lambda g, c: exported.append(g) or Path("Modelfile")
    )

    assert cli.main(["quickstart", "--out", str(tmp_path)]) == 3
    # args.export_ollama was mirrored onto the quickstart Namespace, so
    # main()'s exit-3 branch exported the best-effort Modelfile too.
    assert exported == [tmp_path / "m-Q4_K_M.gguf"]


@pytest.mark.parametrize("boom", [KeyboardInterrupt, EOFError])
def test_ctrl_c_and_eof_are_clean_exits_not_tracebacks(monkeypatch, tmp_path, capsys, boom):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quickstart.hardware, "detect_hardware", lambda: _HW)
    monkeypatch.setattr(quickstart.doctor, "run_checks", lambda b, o: _passing_checks())
    monkeypatch.setattr("sys.stdin", _FakeStdin(True))

    def interrupt(prompt=""):
        raise boom

    monkeypatch.setattr("builtins.input", interrupt)
    assert quickstart.run_wizard(_args(tmp_path)) == 1
    assert "취소했습니다" in capsys.readouterr().out


def test_quickstart_is_a_registered_subcommand_with_run_compatible_flags():
    args = cli._build_parser().parse_args(["quickstart"])
    assert args.command == "quickstart"
    assert args.out == "./out"
    assert args.llama_bin_dir is None
    assert cli._DISPATCH["quickstart"] is quickstart.run_wizard


# ---------------------------------------------------------------------------
# fituna run --hf (non-interactive HF download)
# ---------------------------------------------------------------------------


def test_pick_f16_file_single_match():
    names = ["m-Q4_K_M.gguf", "m-f16.gguf", "README.md"]
    assert quickstart.pick_f16_file(names, "org/repo") == "m-f16.gguf"


def test_pick_f16_file_accepts_bf16_and_is_case_insensitive():
    assert quickstart.pick_f16_file(["Model-BF16.gguf"], "org/repo") == "Model-BF16.gguf"


def test_pick_f16_file_does_not_match_f16_inside_a_word():
    with pytest.raises(quickstart.FiTunaError, match="no F16/BF16"):
        quickstart.pick_f16_file(["mf16x.gguf"], "org/repo")


def test_pick_f16_file_zero_matches_lists_the_ggufs(capsys):
    with pytest.raises(quickstart.FiTunaError) as exc:
        quickstart.pick_f16_file(["m-Q4_K_M.gguf", "m-Q8_0.gguf"], "org/repo")
    assert "m-Q4_K_M.gguf" in str(exc.value)
    assert "org/repo:<filename>" in str(exc.value)


def test_pick_f16_file_multiple_matches_refuses_to_guess():
    with pytest.raises(quickstart.FiTunaError, match="more than one"):
        quickstart.pick_f16_file(["a-f16.gguf", "b-bf16.gguf"], "org/repo")


def test_resolve_hf_model_rejects_empty_repo(tmp_path):
    with pytest.raises(quickstart.FiTunaError, match="repo"):
        quickstart.resolve_hf_model(":file.gguf", tmp_path)


def test_resolve_hf_model_reuses_file_already_on_disk(tmp_path, capsys):
    (tmp_path / "m-f16.gguf").write_bytes(b"gguf")
    got = quickstart.resolve_hf_model("org/repo:m-f16.gguf", tmp_path)
    assert got == tmp_path / "m-f16.gguf"
    assert "reusing" in capsys.readouterr().out


def test_resolve_hf_model_downloads_named_file_without_listing_api(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        dest.write_bytes(b"gguf")
        return dest

    monkeypatch.setattr(quickstart, "_download", fake_download)
    monkeypatch.setattr(
        quickstart,
        "_hf_repo_listing",
        lambda repo: pytest.fail("named file must not hit the listing API"),
    )
    got = quickstart.resolve_hf_model("org/repo:m-f16.gguf", tmp_path)
    assert got == tmp_path / "m-f16.gguf"
    assert calls == ["https://huggingface.co/org/repo/resolve/main/m-f16.gguf"]


def test_resolve_hf_model_bare_repo_picks_f16_and_reports_license(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        quickstart,
        "_hf_repo_listing",
        lambda repo: {
            "siblings": [{"rfilename": "m-f16.gguf"}, {"rfilename": "m-Q4_K_M.gguf"}],
            "cardData": {"license": "apache-2.0"},
        },
    )
    monkeypatch.setattr(
        quickstart, "_download", lambda url, dest: (dest.write_bytes(b"g"), dest)[1]
    )
    got = quickstart.resolve_hf_model("org/repo", tmp_path)
    assert got == tmp_path / "m-f16.gguf"
    assert "apache-2.0" in capsys.readouterr().out


def test_run_parser_model_and_hf_are_mutually_exclusive_and_one_required(capsys):
    parser = cli._build_parser()
    common = ["--target-tps", "20", "--max-quality-loss", "5", "--wikitext", "w.txt"]
    with pytest.raises(SystemExit):
        parser.parse_args(["run", *common])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["run", "--model", "m.gguf", "--hf", "org/repo", *common]
        )
    err = capsys.readouterr().err
    assert "not allowed with" in err
    args = parser.parse_args(["run", "--hf", "org/repo", *common])
    assert args.hf == "org/repo"
    assert args.model is None


def test_hf_listing_turns_a_401_into_a_repo_not_found_hint(monkeypatch):
    # HF's API answers 401 (not 404) for a repo that doesn't exist; the raw
    # "HTTP Error 401: Unauthorized" made users think they needed a token.
    import urllib.error

    def _raise(url, timeout):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(quickstart.urllib.request, "urlopen", _raise)
    with pytest.raises(FiTunaError, match="not found"):
        quickstart._hf_repo_listing("nobody/does-not-exist")
