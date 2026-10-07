# SPDX-License-Identifier: MIT
"""Dataclass immutability / defaults sanity checks for fituna.config."""

import dataclasses
from pathlib import Path

import pytest

from fituna.config import (
    BaseLogitsResult,
    BenchResult,
    BinaryNotFoundError,
    BinaryPaths,
    CandidateConfig,
    FiTunaError,
    GPUVendor,
    HardwareProfile,
    ModelConversionError,
    ModelInfo,
    NoFeasibleConfigError,
    QualityResult,
    SearchResult,
    TargetSpec,
)


def test_base_logits_result_is_frozen():
    result = BaseLogitsResult(Path("base.kld"), 6.0)
    assert result.logits_path == Path("base.kld")
    assert result.perplexity == 6.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.perplexity = 7.0


# ---------------------------------------------------------------------------
# GPUVendor
# ---------------------------------------------------------------------------

def test_gpu_vendor_is_str_enum_for_trivial_json_serialization():
    assert GPUVendor.NVIDIA == "nvidia"
    assert GPUVendor.AMD == "amd"
    assert GPUVendor.APPLE == "apple"
    assert GPUVendor.NONE == "none"
    assert isinstance(GPUVendor.NVIDIA, str)


def test_gpu_vendor_constructs_from_raw_string():
    assert GPUVendor("amd") is GPUVendor.AMD
    with pytest.raises(ValueError):
        GPUVendor("intel")


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------

def test_hardware_profile_is_frozen():
    hw = HardwareProfile(
        gpu_vendor=GPUVendor.NONE,
        gpu_name=None,
        vram_mb=None,
        cpu_cores=8,
        ram_mb=16384,
        os_name="darwin",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        hw.cpu_cores = 4  # type: ignore[misc]


def test_hardware_profile_equality_is_structural():
    a = HardwareProfile(GPUVendor.NVIDIA, "RTX 4090", 24576, 16, 65536, "linux")
    b = HardwareProfile(GPUVendor.NVIDIA, "RTX 4090", 24576, 16, 65536, "linux")
    assert a == b
    assert a is not b


def test_hardware_profile_allows_none_gpu_fields_for_cpu_only():
    hw = HardwareProfile(GPUVendor.NONE, None, None, 4, 8192, "linux")
    assert hw.gpu_name is None
    assert hw.vram_mb is None


# ---------------------------------------------------------------------------
# TargetSpec
# ---------------------------------------------------------------------------

def test_target_spec_defaults():
    spec = TargetSpec(
        model_path=Path("model.gguf"),
        target_tokens_per_sec=20.0,
        max_quality_loss_pct=3.0,
    )
    assert spec.prompt_tokens == 512
    assert spec.gen_tokens == 128
    assert spec.ctx == 4096
    assert spec.ctx_candidates == (4096,)
    assert spec.quant_candidates[0] == "Q8_0"
    assert spec.quant_candidates == (
        "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K",
    )
    assert spec.ngl_max_calls == 6
    assert spec.max_bench_seconds is None


def test_target_spec_is_frozen():
    spec = TargetSpec(Path("model.gguf"), 20.0, 3.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.target_tokens_per_sec = 30.0  # type: ignore[misc]


def test_target_spec_is_hashable_for_cache_keys():
    # all fields (Path, float, tuple[str,...], tuple[int,...], int, Optional[int])
    # are hashable, so a frozen TargetSpec must itself hash without error.
    spec = TargetSpec(Path("model.gguf"), 20.0, 3.0)
    assert hash(spec) is not None


def test_target_spec_accepts_custom_quant_and_ctx_candidates():
    spec = TargetSpec(
        model_path=Path("model.gguf"),
        target_tokens_per_sec=15.0,
        max_quality_loss_pct=5.0,
        ctx_candidates=(2048, 4096, 8192),
        quant_candidates=("Q4_K_M", "Q2_K"),
    )
    assert spec.ctx_candidates == (2048, 4096, 8192)
    assert spec.quant_candidates == ("Q4_K_M", "Q2_K")


# ---------------------------------------------------------------------------
# BinaryPaths
# ---------------------------------------------------------------------------

def test_binary_paths_optional_fields_default_to_none():
    paths = BinaryPaths(
        llama_quantize=Path("/usr/local/bin/llama-quantize"),
        llama_bench=Path("/usr/local/bin/llama-bench"),
        llama_perplexity=Path("/usr/local/bin/llama-perplexity"),
    )
    assert paths.llama_imatrix is None
    assert paths.convert_script is None


def test_binary_paths_is_frozen():
    paths = BinaryPaths(Path("q"), Path("b"), Path("p"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        paths.llama_quantize = Path("other")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------

def test_model_info_is_frozen_and_holds_base_gguf_path():
    info = ModelInfo(architecture="llama", n_layers=32, n_params=7_000_000_000,
                      base_gguf_path=Path("base-f16.gguf"))
    assert info.n_layers == 32
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.n_layers = 40  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CandidateConfig
# ---------------------------------------------------------------------------

def test_candidate_config_is_hashable():
    c = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    assert hash(c) is not None


def test_candidate_config_equality_by_value_not_identity():
    a = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    b = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    c = CandidateConfig(quant="Q4_K_M", ngl=0, ctx=4096)
    assert a == b and a is not b
    assert a != c


def test_candidate_config_is_frozen():
    c = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.ngl = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BenchResult
# ---------------------------------------------------------------------------

def test_bench_result_is_frozen_and_carries_candidate():
    cand = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    bench = BenchResult(candidate=cand, prompt_tok_per_sec=500.0, gen_tok_per_sec=40.0,
                         vram_used_mb=4096, raw_stdout="{}")
    assert bench.candidate == cand
    with pytest.raises(dataclasses.FrozenInstanceError):
        bench.gen_tok_per_sec = 0.0  # type: ignore[misc]


def test_bench_result_allows_none_vram_for_cpu_only():
    cand = CandidateConfig(quant="Q4_K_M", ngl=0, ctx=4096)
    bench = BenchResult(candidate=cand, prompt_tok_per_sec=50.0, gen_tok_per_sec=5.0,
                         vram_used_mb=None, raw_stdout="")
    assert bench.vram_used_mb is None


# ---------------------------------------------------------------------------
# QualityResult
# ---------------------------------------------------------------------------

def test_quality_result_is_frozen_and_quant_only_dependent():
    q = QualityResult(candidate_quant="Q4_K_M", perplexity=6.1,
                       baseline_perplexity=6.0, quality_loss_pct=1.6666666666666667)
    assert q.candidate_quant == "Q4_K_M"
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.perplexity = 7.0  # type: ignore[misc]


def test_quality_loss_pct_formula_matches_contract():
    baseline = 6.0
    perplexity = 6.3
    expected = (perplexity - baseline) / baseline * 100
    q = QualityResult(candidate_quant="Q4_K_M", perplexity=perplexity,
                       baseline_perplexity=baseline, quality_loss_pct=expected)
    assert q.quality_loss_pct == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

def _make_search_result(meets_target: bool = True) -> SearchResult:
    cand = CandidateConfig(quant="Q4_K_M", ngl=20, ctx=4096)
    bench = BenchResult(candidate=cand, prompt_tok_per_sec=100.0, gen_tok_per_sec=30.0,
                         vram_used_mb=2048, raw_stdout="{}")
    quality = QualityResult(candidate_quant="Q4_K_M", perplexity=6.1,
                             baseline_perplexity=6.0, quality_loss_pct=1.67)
    return SearchResult(config=cand, bench=bench, quality=quality,
                         gguf_path=Path("out/model-Q4_K_M.gguf"),
                         run_command=["llama-cli", "-m", "out/model-Q4_K_M.gguf"],
                         meets_target=meets_target)


def test_search_result_is_frozen():
    result = _make_search_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.meets_target = False  # type: ignore[misc]


def test_search_result_run_command_is_a_plain_executable_list():
    result = _make_search_result()
    assert result.run_command[0] == "llama-cli"
    assert "-m" in result.run_command


def test_search_result_meets_target_flag_reflects_best_effort_state():
    ok = _make_search_result(meets_target=True)
    best_effort = _make_search_result(meets_target=False)
    assert ok.meets_target is True
    assert best_effort.meets_target is False


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

def test_fituna_error_hierarchy():
    assert issubclass(FiTunaError, Exception)
    assert issubclass(BinaryNotFoundError, FiTunaError)
    assert issubclass(ModelConversionError, FiTunaError)
    assert issubclass(NoFeasibleConfigError, FiTunaError)


def test_binary_not_found_error_message_is_preserved():
    err = BinaryNotFoundError("llama-quantize not found; install llama.cpp and add it to PATH")
    assert "llama-quantize" in str(err)


def test_no_feasible_config_error_defaults_closest_to_none():
    err = NoFeasibleConfigError("no quant met the target")
    assert err.closest is None
    assert str(err) == "no quant met the target"


def test_no_feasible_config_error_carries_closest_result_payload():
    closest = _make_search_result(meets_target=False)
    err = NoFeasibleConfigError("no quant met the target", closest=closest)
    assert err.closest is closest
    assert err.closest.meets_target is False


def test_no_feasible_config_error_is_catchable_as_base_fituna_error():
    with pytest.raises(FiTunaError):
        raise NoFeasibleConfigError("boom")


def test_is_already_quantized():
    from pathlib import Path

    from fituna.model_info import is_already_quantized

    def info(ft):
        return ModelInfo(
            architecture="llama", n_layers=1, n_params=1,
            base_gguf_path=Path("x.gguf"), file_type=ft,
        )

    assert not is_already_quantized(info(None))   # key absent -> assume fine
    assert not is_already_quantized(info(0))      # F32
    assert not is_already_quantized(info(1))      # F16
    assert not is_already_quantized(info(32))     # BF16
    assert is_already_quantized(info(15))         # Q4_K_M etc.


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_tokens_per_sec": -1.0},
        {"target_tokens_per_sec": float("nan")},
        {"max_quality_loss_pct": 101.0},
        {"max_quality_loss_pct": float("inf")},
        {"ctx": 0, "ctx_candidates": (0,)},
    ],
)
def test_target_spec_rejects_out_of_range_values(kwargs):
    # Library/MCP callers build TargetSpec without going through the CLI,
    # so the range check must live on the dataclass itself.
    base = {"model_path": Path("m.gguf"), "target_tokens_per_sec": 20.0, "max_quality_loss_pct": 5.0}
    with pytest.raises(FiTunaError):
        TargetSpec(**{**base, **kwargs})
