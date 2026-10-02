# SPDX-License-Identifier: MIT
"""fituna.config
================

Shared types for every FiTuna module: :class:`Enum`/``frozen`` ``dataclass``
value objects and the :class:`FiTunaError` exception hierarchy.

This module is the interface contract. No module outside ``config.py``
defines its own cross-module type -- everyone imports from here. Keep every
dataclass ``frozen=True`` (immutable, hashable-friendly) so passing them
between modules can never produce spooky action-at-a-distance bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class GPUVendor(str, Enum):
    NONE = "none"
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"


@dataclass(frozen=True)
class HardwareProfile:
    gpu_vendor: GPUVendor
    gpu_name: Optional[str]  # 없으면 None
    vram_mb: Optional[int]  # 없으면 None
    cpu_cores: int
    ram_mb: int
    os_name: str  # "linux" | "darwin" | "windows"


@dataclass(frozen=True)
class TargetSpec:
    model_path: Path  # .gguf 파일 또는 HF 포맷 디렉토리
    target_tokens_per_sec: float  # generation(tg) 기준 목표 속도
    max_quality_loss_pct: float  # baseline 대비 perplexity 증가 허용 상한(%)
    prompt_tokens: int = 512  # bench용 pp 길이
    gen_tokens: int = 128  # bench용 tg 길이
    ctx: int = 4096  # 필요 컨텍스트 길이(주 탐색 기준값)
    ctx_candidates: tuple[int, ...] = (4096,)  # 다중 지정 시 각각 검증, ctx가 그중 하나로 강제 포함
    quant_candidates: tuple[str, ...] = (
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q4_K_M",
        "Q3_K_M",
        "Q2_K",
    )  # 품질 내림차순 고정 순서
    ngl_max_calls: int = 6  # ngl 이진탐색 상한 호출 수(안전장치)
    max_bench_seconds: Optional[int] = None  # 전체 탐색 시간 예산, 초과 시 best-effort 반환
    # llama-perplexity에 넘길 --chunks 상한. 실측 결과: 3B 모델+wikitext-2 test
    # 전체(제한 없음)로 quant 4개 돌리니 3시간 44분 걸림 -- 품질손실 %는 통계적
    # 추정치일 뿐이라 전체 코퍼스가 필요하지 않음. None(무제한)을 기본값으로
    # 두면 사용자가 아무 것도 모른 채 몇 시간짜리 실행을 돌리게 되므로, 기본을
    # 유한한 값으로 못박는다. 더 엄밀한 평가가 필요하면 명시적으로 None이나
    # 큰 값을 넘기면 된다.
    ppl_chunks: Optional[int] = 32
    quality_metric: str = "ppl"  # "ppl" | "kld"

    def __post_init__(self) -> None:
        # cli.py always builds ctx_candidates with ctx first, but a library
        # caller constructing TargetSpec directly could pass a ctx not in
        # ctx_candidates -- enforce the invariant here rather than only at
        # one call site, so it holds regardless of who constructs this.
        if self.ctx not in self.ctx_candidates:
            raise ValueError(
                f"ctx={self.ctx} must be one of ctx_candidates={self.ctx_candidates}"
            )
        if self.quality_metric not in ("ppl", "kld"):
            raise ValueError(
                f"quality_metric={self.quality_metric!r} must be 'ppl' or 'kld'"
            )


@dataclass(frozen=True)
class BinaryPaths:
    llama_quantize: Path
    llama_bench: Path
    llama_perplexity: Path
    llama_imatrix: Optional[Path] = None
    convert_script: Optional[Path] = None  # HF->GGUF 변환용 (llama.cpp 저장소 내 스크립트)


@dataclass(frozen=True)
class ModelInfo:
    architecture: str
    n_layers: int
    n_params: int
    base_gguf_path: Path  # 양자화 기준 원본(F16/F32) GGUF
    # GGUF 'general.file_type' (llama.cpp LLAMA_FTYPE enum). 0=F32, 1=F16,
    # 32=BF16; anything else means the file is already quantized. None when
    # the writer didn't set the key.
    file_type: Optional[int] = None


@dataclass(frozen=True)
class CandidateConfig:
    quant: str
    ngl: int
    ctx: int


@dataclass(frozen=True)
class BenchResult:
    candidate: CandidateConfig
    prompt_tok_per_sec: float
    gen_tok_per_sec: float
    vram_used_mb: Optional[int]
    raw_stdout: str


@dataclass(frozen=True)
class QualityResult:
    candidate_quant: str  # quality는 quant에만 의존(ngl/ctx 무관)
    perplexity: float
    baseline_perplexity: float
    quality_loss_pct: float  # (perplexity-baseline)/baseline*100
    metric: str = "ppl"
    kld: Optional[float] = None


@dataclass(frozen=True)
class SearchResult:
    config: CandidateConfig
    bench: BenchResult
    quality: QualityResult
    gguf_path: Path
    run_command: list[str]  # 사용자가 그대로 실행 가능한 llama-cli 커맨드
    meets_target: bool  # False면 best-effort(목표 미달) 결과
    # 아래 두 필드는 나중에 추가된 것이라 기본값 None을 갖는다(기존 생성자
    # 호출 호환). 산출물(gguf)을 실제로 쓰는 두 경로:
    llama_server_command: Optional[list[str]] = None  # OpenAI 호환 로컬 API 서버 커맨드
    modelfile_path: Optional[Path] = None  # --export-ollama로 쓴 Modelfile, 미생성 시 None


@dataclass(frozen=True)
class DoctorCheck:
    """One row of ``fituna doctor`` output: a single environment check and
    its outcome.

    ``status`` is a plain str, not an Enum -- always one of "PASS" | "WARN" |
    "FAIL" (matching the words the human formatter prints), but nothing else
    needs to branch on it structurally, so a bare str is enough here (same
    reasoning as HardwareProfile.os_name above).
    """

    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str  # one-line human-readable outcome, e.g. a version or a path
    remedy: Optional[str] = None  # one-line fix instruction; None means "PASS"


@dataclass(frozen=True)
class CorpusPreset:
    """One ``fituna fetch-corpus --lang`` preset: which HuggingFace dataset
    (dataset/config/split/text_field) to pull plain-text rows from, how many
    rows to fetch by default, and the license notice to print on stdout
    after a successful fetch.

    Only the shape lives here -- the concrete EN/KO preset data lives in
    ``fituna.corpus.PRESETS`` (the module that owns corpus-fetching), the
    same "dataclass shape in config.py, concrete data at its owning module"
    split already used for BinaryPaths (config.py) / REQUIRED_BINARIES
    (binaries.py).
    """

    dataset: str
    config: str
    split: str
    text_field: str
    default_rows: int
    license_note: str  # ready-to-print stdout line; verified, not guessed -- see fituna/corpus.py


class FiTunaError(Exception):
    """Base class for all FiTuna errors."""


class BinaryNotFoundError(FiTunaError):
    """Raised when a required llama.cpp binary could not be located."""


class ModelConversionError(FiTunaError):
    """Raised when HF -> GGUF conversion fails."""


class BenchTimeoutError(FiTunaError):
    """Raised when a single llama-bench invocation exceeds its timeout.

    Distinct from FiTunaError so the search loop can treat "this candidate is
    too slow to even finish a bench" as a below-target measurement instead of
    aborting the whole search."""


class NoFeasibleConfigError(FiTunaError):
    """Raised when no candidate configuration could satisfy the target at all."""

    def __init__(self, message: str, closest: Optional["SearchResult"] = None) -> None:
        super().__init__(message)
        self.closest = closest


def _self_check() -> None:
    """Minimal assert-based sanity check for this module's core contract.

    Not a full test suite (see tests/test_config.py for that) -- just a
    runnable guard against the two ways this file could regress silently:
    dataclasses becoming mutable, and NoFeasibleConfigError losing its
    ``closest`` payload.
    """
    import dataclasses

    # 1. GPUVendor is a str Enum: compares equal to plain strings, so it
    #    serializes trivially to JSON via str() / json.dumps(..., default=str).
    assert GPUVendor.NVIDIA == "nvidia"
    assert GPUVendor("amd") is GPUVendor.AMD

    # 2. TargetSpec defaults match the frozen interface contract exactly.
    spec = TargetSpec(model_path=Path("model.gguf"), target_tokens_per_sec=20.0,
                       max_quality_loss_pct=5.0)
    assert spec.prompt_tokens == 512 and spec.gen_tokens == 128
    assert spec.ctx == 4096 and spec.ctx_candidates == (4096,)
    assert spec.quant_candidates == ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")
    assert spec.ngl_max_calls == 6 and spec.max_bench_seconds is None

    # 3. All dataclasses are frozen: mutation must raise, never silently succeed.
    hw = HardwareProfile(gpu_vendor=GPUVendor.NONE, gpu_name=None, vram_mb=None,
                          cpu_cores=4, ram_mb=8192, os_name="linux")
    try:
        hw.cpu_cores = 8  # type: ignore[misc]
        raise AssertionError("HardwareProfile must be frozen")
    except dataclasses.FrozenInstanceError:
        pass

    cand = CandidateConfig(quant="Q4_K_M", ngl=20, ctx=4096)
    bench = BenchResult(candidate=cand, prompt_tok_per_sec=100.0, gen_tok_per_sec=30.0,
                         vram_used_mb=2048, raw_stdout="{}")
    quality = QualityResult(candidate_quant="Q4_K_M", perplexity=6.1,
                             baseline_perplexity=6.0, quality_loss_pct=1.67)
    result = SearchResult(config=cand, bench=bench, quality=quality,
                           gguf_path=Path("out/model-Q4_K_M.gguf"),
                           run_command=["llama-cli", "-m", "out/model-Q4_K_M.gguf"],
                           meets_target=True)
    try:
        result.meets_target = False  # type: ignore[misc]
        raise AssertionError("SearchResult must be frozen")
    except dataclasses.FrozenInstanceError:
        pass

    # 4. NoFeasibleConfigError must carry its closest-result payload through.
    err = NoFeasibleConfigError("no quant met the target", closest=result)
    assert err.closest is result
    assert str(err) == "no quant met the target"
    assert NoFeasibleConfigError("x").closest is None

    # 5. Value equality (dataclasses compare structurally, not by identity).
    assert CandidateConfig(quant="Q4_K_M", ngl=20, ctx=4096) == cand


if __name__ == "__main__":
    _self_check()
    print("fituna.config self-check OK")
