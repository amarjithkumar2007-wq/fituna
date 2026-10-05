# SPDX-License-Identifier: MIT
"""fituna.quickstart
===================

The ``fituna quickstart`` interactive wizard: an ``input()``-based shell over
capabilities that are *all* reachable non-interactively through ``fituna
run``'s public flags. It ends by printing the fully assembled ``fituna run
...`` command and then executing it in-process through the very same code
path ``fituna/cli.py`` dispatches to -- the wizard graduates its users to the
CLI rather than becoming a second, divergent interface.

**Every search parameter maps to a public ``run`` flag.** Every answer that
feeds the search (target speed, quality-loss ceiling, ctx, license filter)
turns into one ``fituna run`` argument (see :func:`build_run_argv`);
``_selfcheck`` and the tests both parse that argv back through
``cli._build_parser()`` and assert it is exactly the argv actually executed,
so a search parameter this wizard could express but ``run`` cannot fails
immediately instead of quietly forking the product. Model download (the
curated shortlist, HuggingFace search) is a wizard convenience with no
``run`` equivalent -- ``run --model`` still expects a ``.gguf`` already on
disk.

The honesty line this module has to embody
------------------------------------------
- **Memory fit is arithmetic and is allowed**: published file size vs the
  VRAM/RAM ``fituna.hardware`` detected (:func:`memory_fit`), with the
  assumed margin spelled out on screen (``_MEMORY_CAVEAT``).
- **Speed is never predicted.** The wizard says so in its own visible copy.
  Numbers from ``docs/RESULTS.md`` appear only as *records of past
  measurements on named hardware*, never as a forecast for the user's box.
- **License provenance is never laundered.** The three curated models carry a
  "라이선스 원문 확인됨" badge because this project checked their license text
  (``docs/AI_MODEL_USAGE.md`` B-1/B-2/B-3). Everything coming out of HF search
  is labelled uploader-supplied metadata, with the uploader's own link.
- **Preset targets are conventional starting points**, labelled as such --
  the search is what judges them.

HuggingFace search API -- verified by hand against the live API on
2026-08-02, not guessed, same practice as ``fituna/corpus.py``:

    GET https://huggingface.co/api/models
        ?search=<q>&filter=gguf&limit=<n>&sort=downloads&direction=-1
        &expand[]=cardData&expand[]=gated&expand[]=downloads
        &expand[]=siblings&expand[]=tags

    -> 200 [ {"id": "LGAI-EXAONE/EXAONE-4.0-32B-GGUF",
              "cardData": {"license": "other", "license_name": "exaone",
                           "license_link": "LICENSE", ...},
              "gated": false, "downloads": 81725,
              "tags": [..., "license:other", ...],
              "siblings": [{"rfilename": "EXAONE-4.0-32B-Q4_K_M.gguf"}, ...]},
            ... ]

Confirmed characteristics that shaped the parser:
- The response is a bare JSON *array*, not an object with a "models" key.
- There is **no top-level ``license`` field**. The license lives in
  ``cardData.license``; when that is the literal string ``"other"`` the real
  name is in ``cardData.license_name`` (EXAONE reports ``license: "other"`` +
  ``license_name: "exaone"``, Kanana ``license_name:
  "kanana-open-license"``) -- reading only ``cardData.license`` would file
  every research-only license under the meaningless label "other". A
  ``license:<slug>`` entry also appears in ``tags`` and is used as a fallback.
- ``cardData`` may be present with no ``license`` key at all (e.g.
  ``mradermacher/EXAONE-4.0-1.2B-abliterated-i1-GGUF`` in the captured
  fixture, whose ``cardData`` carries only ``base_model``/``tags``) --
  "no license metadata" is a real case, not an edge case. ``cardData`` being
  *absent entirely* is covered separately, by the synthetic
  ``someone/Model-GGUF`` row in ``_selfcheck`` (no ``cardData`` key at all,
  license recovered from the ``tags`` fallback instead).
- ``gated`` is ``false`` or a *string* (``"manual"``/``"auto"``, observed on
  ``meta-llama/Llama-2-7b-chat-hf``), so it is truthiness-tested, not
  compared to ``True``.
- ``siblings`` lists every file in the repo as ``{"rfilename": ...}``; the
  ``.gguf`` entries are the only downloadable candidates, and multi-part
  shards appear as ``...-00001-of-00002.gguf``.
- ``expand[]`` and the default listing are both accepted without auth or a
  custom User-Agent; ``full=true`` returns the same fields except
  ``cardData``, which is why ``expand[]`` is used instead.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shlex
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fituna import corpus, doctor, hardware, report
from fituna.config import FiTunaError, GPUVendor, HardwareProfile, NoFeasibleConfigError

HF_SEARCH_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{filename}"
TIMEOUT_SEC = 30
SEARCH_LIMIT = 10

# Arithmetic head-room assumed for KV cache + runtime overhead when deciding
# "does this file fit". An assumption, not a measurement -- every line that
# uses it says so out loud (see _memory_fit_line).
MEMORY_MARGIN = 0.8

# [2/6] presets: conventional starting values, NOT measurements. The search
# is the judge -- an unrealistic target is allowed and simply exits 3 with the
# measured best effort.
PRESETS: tuple[tuple[str, float, float, int], ...] = (
    ("대화형 챗봇 (interactive chat)", 20.0, 5.0, 4096),
    ("코딩 보조 (coding assistant)", 30.0, 3.0, 8192),
    ("문서 처리, 긴 입력 (long-document processing)", 15.0, 5.0, 16384),
)

# [3/6] license needs -> filter predicate (see license_allows).
LICENSE_NEEDS: tuple[tuple[str, str], ...] = (
    ("personal", "개인/연구용 (필터 없음)"),
    # "알려진" is load-bearing: the filter is a deny-list of known
    # non-commercial markers, so a model with no license metadata at all --
    # or one whose license string this list has never seen -- stays in the
    # list. Promising "비상업 라이선스 제외" would claim an exclusion the
    # deny-list cannot make for unlabelled models.
    ("commercial", "상업적 이용 (알려진 비상업 라이선스 제외)"),
    ("redistribution", "재배포/파생물 배포 (MIT/Apache-2.0/BSD 계열만)"),
)

# OSI-approved permissive licenses that allow redistribution and derivative
# works without a copyleft or field-of-use restriction.
_PERMISSIVE = frozenset(
    {"mit", "apache-2.0", "bsd", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear", "isc"}
)

# Substrings marking a license as non-commercial or research-only. A
# deny-list, deliberately: an unknown license is *not* silently reclassified
# as commercial-safe -- it stays in the list carrying the "uploader-supplied
# metadata, verify before commercial use" caveat every HF candidate gets.
# "exaone" is here because this project read that license text and recorded
# the finding (docs/AI_MODEL_USAGE.md, the note under B-3).
_NON_COMMERCIAL_MARKERS = (
    "cc-by-nc",
    "-nc-",
    "noncommercial",
    "non-commercial",
    "research",
    "exaone",
)

_METADATA_CAVEAT = (
    "위 라이선스 값은 업로더가 모델 카드에 적어 넣은 메타데이터입니다 — "
    "FiTuna가 원문을 확인한 것이 아닙니다. 상업적으로 쓰기 전에 링크의 "
    "라이선스 원문을 직접 확인하세요."
)

# Per-model license evidence. These are two *different* claims and the menu
# must not blur them -- "원문 확인됨" is exactly the evidence class
# _METADATA_CAVEAT disclaims the absence of, so it may only be printed where
# a real LICENSE file was fetched and compared. Checked 2026-08-02 against
# https://huggingface.co/api/models/<repo> siblings + the raw file:
# Qwen/Qwen3-4B-Instruct-2507 ships LICENSE (Apache-2.0 text) and
# K-intelligence/Midm-2.0-Mini-Instruct ships LICENSE.txt (MIT text), but
# HuggingFaceTB/SmolLM2-135M-Instruct ships no license file at all -- for it
# only the model-card metadata exists. See docs/AI_MODEL_USAGE.md B-1/B-2/B-3.
_LICENSE_TEXT_VERIFIED = "[라이선스 원문 확인됨(가중치 원본 저장소) — docs/AI_MODEL_USAGE.md]"
_LICENSE_METADATA_ONLY = "[라이선스 메타데이터만 확인 — 원문 파일 없음, docs/AI_MODEL_USAGE.md]"

_MEMORY_CAVEAT = (
    "메모리 판정은 '공개된 파일 크기 vs 감지된 메모리' 산술입니다. 여유 20 %는\n"
    "  KV 캐시·런타임 몫으로 잡은 가정이지 실측이 아니며, 기준 파일은 F16/BF16\n"
    "  원본이라 실제로 돌릴 양자화 파일은 이보다 작습니다."
)

# Printed when a selected curated model's F16 file is larger than the memory
# budget. The verdict line already said so; this says what it actually costs,
# because the naive reading ("too big, won't work") is wrong: the artifact you
# end up running is the quantized file.
_F16_STAGE_WARNING = (
    "  ⚠ 이 모델의 F16 원본은 감지된 메모리 예산보다 큽니다. 그래도 진행할 수\n"
    "    있습니다 — 다만 정확히 무엇이 걸리는지 알고 고르십시오:\n"
    "      · 디스크·다운로드 비용은 F16 원본 크기 그대로입니다.\n"
    "      · 양자화 단계는 F16을 읽으므로 그 동안 메모리 압박이 실제로 있습니다.\n"
    "      · 반면 최종 산출물(예: Q4_K_M)은 훨씬 작고, 탐색은 ngl을 0부터\n"
    "        올려가며 맞추므로 통째로 올라가지 않아도 측정은 진행됩니다."
)

_NO_SPEED_PREDICTION = (
    "속도는 예측하지 않습니다 — 고르시면 측정합니다. "
    "(FiTuna never predicts throughput; it measures it.)"
)


@dataclass(frozen=True)
class CuratedModel:
    """One entry of the project-verified shortlist.

    Every field below is copied from ``docs/AI_MODEL_USAGE.md`` B-1/B-2/B-3
    (model ids, GGUF repo, license) and ``docs/RESULTS.md`` (the measured
    anchor); ``size_bytes`` was read from the HuggingFace API's own
    ``siblings[].size`` for that exact file on 2026-08-02. Nothing here is
    estimated -- see this module's docstring on the honesty line.

    ``anchor`` is a *record of a past measurement on named hardware*, never a
    prediction for the user's machine.
    """

    label: str
    base_model: str  # upstream weights (AI_MODEL_USAGE.md "기반 모델명")
    gguf_repo: str  # repo the bytes actually come from
    filename: str
    # Published size of `filename` in `gguf_repo` as of the snapshot date
    # below -- a recorded figure, not a live HEAD request. Every line that
    # prints it says "게시 시점 크기" for that reason.
    size_bytes: int  # HuggingFace API siblings[].size, snapshot 2026-08-02
    license: str
    license_evidence: str  # one of the _LICENSE_* badges above
    anchor: str


CURATED: tuple[CuratedModel, ...] = (
    CuratedModel(
        label="SmolLM2-135M-Instruct (135M, 영어)",
        base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        gguf_repo="bartowski/SmolLM2-135M-Instruct-GGUF",
        filename="SmolLM2-135M-Instruct-f16.gguf",
        size_bytes=270_885_952,
        license="apache-2.0",
        license_evidence=_LICENSE_METADATA_ONLY,
        anchor="Apple M3 Pro 실측 249.50 tok/s @Q6_K, ngl=30 (docs/RESULTS.md Run 1)",
    ),
    CuratedModel(
        label="Qwen3-4B-Instruct-2507 (4B, 다국어)",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        gguf_repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        filename="Qwen3-4B-Instruct-2507-F16.gguf",
        size_bytes=8_051_285_344,
        license="apache-2.0",
        license_evidence=_LICENSE_TEXT_VERIFIED,
        anchor="Apple M3 Pro 실측 30.81 tok/s @Q4_K_M, ngl=33 (docs/RESULTS.md Run 2)",
    ),
    CuratedModel(
        label="Midm-2.0-Mini-Instruct (2.3B, 한국어)",
        base_model="K-intelligence/Midm-2.0-Mini-Instruct",
        gguf_repo="mykor/Midm-2.0-Mini-Instruct-gguf",
        filename="Midm-2.0-Mini-Instruct-BF16.gguf",
        size_bytes=4_617_053_184,
        license="mit",
        license_evidence=_LICENSE_TEXT_VERIFIED,
        anchor="Apple M3 Pro 실측 44.62 tok/s @Q4_K_M, ngl=48 (docs/RESULTS.md Run 5)",
    ),
)

# [5/6] corpus presets -- filenames match the README's manual commands so the
# printed `fituna run` line stays copy-pasteable afterwards.
CORPUS_CHOICES: tuple[tuple[str, str, str], ...] = (
    ("en", "wikitext-2-raw-test.txt", "영어 (wikitext-2)"),
    ("ko", "kowiki-corpus.txt", "한국어 (한국어 위키백과)"),
)


# ---------------------------------------------------------------------------
# pure logic -- no I/O, unit-testable, shared by _selfcheck and the tests
# ---------------------------------------------------------------------------


def license_allows(
    license_slug: Optional[str],
    need: str,
    *,
    model_id: str = "",
    base_model: Optional[str] = None,
) -> bool:
    """Does a model under ``license_slug`` satisfy the user's ``need``?

    - ``personal``: everything (no filter).
    - ``commercial``: deny-list -- anything whose license slug carries a
      known non-commercial / research-only marker is out. Missing license
      *metadata* does not mean missing evidence: ``model_id`` and
      ``base_model`` (when supplied) are scanned for the same markers too,
      lowercased, because an EXAONE derivative with an absent license field
      but ``exaone`` in its own id or its ``cardData.base_model`` still
      carries the disqualifying evidence -- it is just filed under a
      different key. Nothing left over after that scan passes the filter
      but keeps the metadata caveat: this function classifies, it does not
      give legal advice.
    - ``redistribution``: allow-list -- permissive licenses only, so an
      unknown or missing license is excluded.
    """
    if need == "personal":
        return True
    slug = (license_slug or "").strip().lower()
    if need == "redistribution":
        return slug in _PERMISSIVE
    if need == "commercial":
        haystack = " ".join([slug, model_id.lower(), (base_model or "").lower()])
        return not any(marker in haystack for marker in _NON_COMMERCIAL_MARKERS)
    raise ValueError(f"unknown license need {need!r}")


def available_memory_mb(hw: HardwareProfile) -> tuple[Optional[int], str]:
    """(MB, label) of the memory a model would have to fit into: the GPU's
    VRAM when one was detected, otherwise system RAM. ``(None, "")`` when
    detection produced neither -- the caller must then say "판정 불가"
    instead of inventing a number."""
    if hw.gpu_vendor != GPUVendor.NONE and hw.vram_mb:
        return hw.vram_mb, "VRAM"
    if hw.ram_mb:
        return hw.ram_mb, "RAM"
    return None, ""


def memory_fit(size_bytes: int, hw: HardwareProfile) -> Optional[bool]:
    """``True``/``False`` if the arithmetic can be done, ``None`` when there
    is no detected memory figure to compare against (never guess)."""
    mem_mb, _label = available_memory_mb(hw)
    if mem_mb is None:
        return None
    return size_bytes <= mem_mb * 1024 * 1024 * MEMORY_MARGIN


def _memory_fit_line(size_bytes: int, hw: HardwareProfile) -> str:
    """The arithmetic itself, one line. Its assumptions are stated once per
    menu in _MEMORY_CAVEAT rather than repeated after every model."""
    mem_mb, label = available_memory_mb(hw)
    size = report.human_size(size_bytes)
    if mem_mb is None:
        return f"메모리: {size} — 감지된 메모리 정보가 없어 판정하지 않습니다"
    budget = mem_mb * 1024 * 1024 * MEMORY_MARGIN
    verdict = "들어갑니다" if memory_fit(size_bytes, hw) else "부족합니다"
    return (
        f"메모리: {size} vs 감지된 {label} {report.human_size(mem_mb * 1024 * 1024)}"
        f" ({mem_mb} MB) × {MEMORY_MARGIN * 100:.0f} % = {report.human_size(budget)} → {verdict}"
    )


@dataclass(frozen=True)
class HFCandidate:
    """One parsed HuggingFace search hit (see the module docstring for the
    verified response shape this is read out of)."""

    model_id: str
    license: Optional[str]
    license_link: Optional[str]
    gated: bool
    downloads: int
    gguf_files: tuple[str, ...]
    # cardData.base_model -- kept so the commercial filter can catch a
    # derivative whose *own* license metadata is absent but whose upstream
    # model is a known non-commercial one (see license_allows).
    base_model: Optional[str] = None


def _license_of(item: dict) -> tuple[Optional[str], Optional[str]]:
    """(license slug, uploader's license link) from one search hit.

    ``cardData.license == "other"`` is a placeholder, not a license -- the
    real name is then in ``cardData.license_name`` (verified: EXAONE reports
    exactly that). Falls back to the ``license:<slug>`` entry in ``tags``.
    """
    card = item.get("cardData") or {}
    slug = card.get("license")
    if slug == "other" and card.get("license_name"):
        slug = card["license_name"]
    if not slug:
        for tag in item.get("tags") or []:
            if isinstance(tag, str) and tag.startswith("license:"):
                slug = tag.split(":", 1)[1]
                break
    return (slug or None), card.get("license_link")


def parse_hf_search(payload: object) -> list[HFCandidate]:
    """Parse the verified ``/api/models`` array into candidates.

    Tolerant by construction: anything that isn't a list of dicts yields an
    empty list rather than raising, because a shape change on someone else's
    API must degrade to "no results" in a wizard, not a traceback.
    Multi-part shards (``...-00001-of-00002.gguf``) are dropped -- they
    cannot be passed to ``--model`` as a single file.
    """
    if not isinstance(payload, list):
        return []
    out: list[HFCandidate] = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        slug, link = _license_of(item)
        files = tuple(
            name
            for sib in item.get("siblings") or []
            if isinstance(sib, dict)
            for name in [sib.get("rfilename") or ""]
            if name.endswith(".gguf") and "-of-" not in name
        )
        card = item.get("cardData") or {}
        out.append(
            HFCandidate(
                model_id=str(item["id"]),
                license=slug,
                license_link=link,
                # `gated` is false or a string ("manual"/"auto") -- truthiness,
                # not `is True`.
                gated=bool(item.get("gated")),
                downloads=int(item.get("downloads") or 0),
                gguf_files=files,
                base_model=card.get("base_model"),
            )
        )
    return out


def _prefer_unquantized(names: tuple[str, ...]) -> list[str]:
    """F16/BF16/F32 files first: those are what ``fituna run`` wants as
    ``--model`` (it warns when handed an already-quantized GGUF)."""
    return sorted(names, key=lambda n: (not _is_base_precision(n), n.lower()))


def _is_base_precision(name: str) -> bool:
    low = name.lower()
    return any(tag in low for tag in ("f16", "bf16", "fp16", "f32", "fp32"))


def build_run_argv(
    model: Path,
    target_tps: float,
    max_quality_loss: float,
    ctx: int,
    quality_corpus: Path,
    out_dir: Path,
    export_ollama: bool,
    llama_bin_dir: Optional[str] = None,
) -> list[str]:
    """The exact argv the wizard prints and then executes, minus ``fituna``.

    This is the whole "no wizard-only features" contract in one function:
    every answer the wizard collected has to come out the other side as a
    public ``fituna run`` flag, or it does not exist. ``--resume`` is always
    included -- re-running the same search then costs about a second.
    """
    argv = [
        "run",
        "--model", str(model),
        "--target-tps", f"{target_tps:g}",
        "--max-quality-loss", f"{max_quality_loss:g}",
        "--ctx", str(ctx),
        "--quality-corpus", str(quality_corpus),
        "--out", str(out_dir),
        "--resume",
    ]
    if llama_bin_dir:
        argv += ["--llama-bin-dir", llama_bin_dir]
    if export_ollama:
        argv.append("--export-ollama")
    return argv


# ---------------------------------------------------------------------------
# prompting -- every one of these re-prompts forever rather than crashing on
# garbage; EOF/Ctrl-C is the only way out, and run_wizard turns that into a
# clean exit 1 instead of a traceback.
# ---------------------------------------------------------------------------


def _ask(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{text}{suffix}: ").strip()
        if raw:
            return raw
        if default:
            return default
        print("  값을 입력해 주세요.")


def _ask_number(text: str, default: float, *, as_int: bool = False) -> float:
    """Positive number, Enter = ``default``. Anything else re-prompts."""
    shown = str(int(default)) if as_int else f"{default:g}"
    while True:
        raw = _ask(text, shown)
        try:
            value = float(raw)
        except ValueError:
            print(f"  숫자를 입력해 주세요 (입력하신 값: {raw!r}).")
            continue
        if value <= 0 or value != value or value in (float("inf"), float("-inf")):
            print("  0보다 큰 숫자를 입력해 주세요.")
            continue
        if as_int and value != int(value):
            print("  정수를 입력해 주세요.")
            continue
        return int(value) if as_int else value


def _ask_choice(text: str, count: int, default: int = 1) -> int:
    """1-based menu pick. Out-of-range and non-numeric both re-prompt."""
    while True:
        raw = _ask(text, str(default))
        try:
            picked = int(raw)
        except ValueError:
            print(f"  1~{count} 사이의 번호를 입력해 주세요 (입력하신 값: {raw!r}).")
            continue
        if not 1 <= picked <= count:
            print(f"  1~{count} 사이의 번호를 입력해 주세요.")
            continue
        return picked


def _ask_yes_no(text: str, default: bool = True) -> bool:
    while True:
        raw = _ask(text, "y" if default else "n").lower()
        if raw in ("y", "yes", "예", "네"):
            return True
        if raw in ("n", "no", "아니오", "아니요"):
            return False
        print("  y 또는 n 으로 답해 주세요.")


# ---------------------------------------------------------------------------
# download -- same urllib + temp-file + os.replace pattern as
# fituna.corpus.fetch_corpus (partial download never lands at the target
# path, no temp file left behind)
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    except OSError as exc:
        raise FiTunaError(f"{dest} 을(를) 쓸 준비를 하지 못했습니다: {exc}") from exc

    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SEC) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                step = -1
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        now = int(done * 20 / total)  # every 5 %
                        if now != step:
                            step = now
                            # language-neutral on purpose: this line is shared
                            # by the Korean wizard and the English --hf path
                            print(
                                f"  {done * 100 // total:3d}% "
                                f"({report.human_size(done)} / {report.human_size(total)})"
                            )
        os.replace(tmp, dest)
    except (OSError, TimeoutError, http.client.IncompleteRead) as exc:
        # urllib.error.HTTPError/URLError and a failing os.replace are OSError
        # subclasses, as is TimeoutError itself -- but a truncated multi-GB
        # transfer (Content-Length promised more than the socket delivered)
        # surfaces from read() as http.client.IncompleteRead, which is *not*
        # an OSError, so it needs its own name in the tuple. One catch, one
        # message either way (fituna.corpus convention) -- a dropped
        # connection here must return the user to the model menu, not throw
        # away four completed steps.
        tmp.unlink(missing_ok=True)
        raise FiTunaError(f"{url} 다운로드에 실패했습니다: {exc}") from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


# ---------------------------------------------------------------------------
# non-interactive HF download -- the `fituna run --hf` path. English messages:
# unlike the wizard above (Korean, interactive), this is reached from the
# plain CLI/CI surface whose output language is English.
# ---------------------------------------------------------------------------

_F16_NAME = re.compile(r"(?:^|[-_.])(?:bf16|f16|fp16)(?:[-_.]|$)", re.IGNORECASE)


def pick_f16_file(filenames: list[str], repo: str) -> str:
    """Pick the single F16/BF16 .gguf from a repo file listing.

    Zero or several matches are both errors: guessing a multi-GB download
    is worse than asking the user to name the file (``--hf repo:filename``).
    """
    ggufs = [n for n in filenames if n.lower().endswith(".gguf")]
    matches = [n for n in ggufs if _F16_NAME.search(Path(n).stem)]
    if len(matches) == 1:
        return matches[0]
    listing = "\n".join(f"  {n}" for n in ggufs) or "  (no .gguf files found)"
    kind = "no F16/BF16 .gguf" if not matches else "more than one F16/BF16 .gguf"
    raise FiTunaError(
        f"{repo} has {kind}; pass the exact file as --hf '{repo}:<filename>'. "
        f"Files in the repo:\n{listing}"
    )


def _hf_repo_listing(repo: str) -> dict:
    url = f"https://huggingface.co/api/models/{urllib.parse.quote(repo)}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SEC) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 404):  # HF answers 401, not 404, for a repo that doesn't exist
            raise FiTunaError(
                f"HuggingFace repo {repo!r} not found (or private/gated) -- check the "
                f"spelling, e.g. bartowski/SmolLM2-135M-Instruct-GGUF (HTTP {exc.code})"
            ) from exc
        raise FiTunaError(f"could not query HuggingFace for {repo}: {exc}") from exc
    except OSError as exc:
        raise FiTunaError(f"could not query HuggingFace for {repo}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FiTunaError(f"unexpected HuggingFace API response for {repo}: {exc}") from exc


def resolve_hf_model(spec: str, out_dir: Path) -> Path:
    """Resolve ``--hf repo[:filename]`` to a local .gguf, downloading if needed.

    The license of the downloaded weights is the model publisher's, not
    FiTuna's -- printed when the API reports one, flagged when it doesn't
    (a bare .gguf carries no license metadata).
    """
    repo, _, filename = spec.partition(":")
    repo = repo.strip().strip("/")
    if not repo:
        raise FiTunaError(f"--hf expects 'repo[:filename]', got {spec!r}")
    if not filename:
        info = _hf_repo_listing(repo)
        filename = pick_f16_file([s.get("rfilename", "") for s in info.get("siblings", [])], repo)
        license_id = (info.get("cardData") or {}).get("license")
        print(
            f"license: {license_id} (weights published by {repo}; their terms, not FiTuna's)"
            if license_id
            else f"license: not reported by the HuggingFace API for {repo} -- check the model card"
        )
    dest = out_dir / Path(filename).name
    if dest.exists():
        print(f"model already on disk, reusing: {dest}")
        return dest
    print(f"downloading {HF_RESOLVE.format(repo=repo, filename=filename)}")
    return _download(HF_RESOLVE.format(repo=repo, filename=filename), dest)


def _hf_search(query: str) -> list[HFCandidate]:
    params = [
        ("search", query),
        ("filter", "gguf"),
        ("limit", str(SEARCH_LIMIT)),
        ("sort", "downloads"),
        ("direction", "-1"),
    ]
    params += [("expand[]", f) for f in ("cardData", "gated", "downloads", "siblings", "tags")]
    url = f"{HF_SEARCH_API}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SEC) as resp:
            body = resp.read()
    except OSError as exc:
        raise FiTunaError(f"HuggingFace 검색 API에 접근하지 못했습니다: {exc}") from exc
    try:
        return parse_hf_search(json.loads(body))
    except json.JSONDecodeError as exc:
        raise FiTunaError(f"HuggingFace 검색 API 응답을 해석하지 못했습니다: {exc}") from exc


# ---------------------------------------------------------------------------
# the six steps
# ---------------------------------------------------------------------------


def _step_doctor(bin_dir: Optional[Path], out_dir: Path) -> int:
    print("[1/6] 환경 점검 (fituna doctor)")
    print()
    checks = doctor.run_checks(bin_dir, out_dir)
    print(doctor.to_human(checks))
    print()
    code = doctor.exit_code(checks)
    if code != 0:
        # doctor.to_human already printed a remedy line per failing check --
        # repeating them here would just be a second, drift-prone copy.
        print("환경 점검에서 FAIL이 나왔습니다. 위 '->' 줄의 해결 방법을 먼저 처리한 뒤")
        print("`fituna quickstart` 를 다시 실행해 주세요.")
    return code


def _step_targets() -> tuple[float, float, int]:
    print("[2/6] 목표 설정")
    print(f"  {_NO_SPEED_PREDICTION}")
    print("  아래 프리셋 숫자는 통용되는 시작값이며 실측값이 아닙니다 — 측정이 판정합니다.")
    print("  목표가 과하면 그대로 두셔도 됩니다: 미달 시 실측된 최선값을 알려드립니다.")
    print()
    for i, (label, tps, loss, ctx) in enumerate(PRESETS, start=1):
        print(f"  {i}) {label}: {tps:g} tok/s, 품질손실 {loss:g} % 이내, ctx {ctx}")
    print(f"  {len(PRESETS) + 1}) 직접 입력")
    print()
    picked = _ask_choice("번호를 고르세요", len(PRESETS) + 1, default=1)
    if picked <= len(PRESETS):
        _label, tps, loss, ctx = PRESETS[picked - 1]
        return tps, loss, ctx

    print("  (Enter 만 누르면 괄호 안 기본값이 쓰입니다)")
    tps = _ask_number("  목표 생성 속도 tok/s — 초당 생성 토큰 수", 20.0)
    loss = _ask_number("  허용 품질손실 % — F16 대비 perplexity 증가 상한", 5.0)
    ctx = int(_ask_number("  컨텍스트 길이 ctx — 한 번에 처리할 토큰 수", 4096, as_int=True))
    return tps, loss, ctx


def _step_license() -> str:
    print("[3/6] 라이선스 조건 — 어떤 조건으로 쓰시나요?")
    print()
    for i, (_need, label) in enumerate(LICENSE_NEEDS, start=1):
        print(f"  {i}) {label}")
    print()
    picked = _ask_choice("번호를 고르세요", len(LICENSE_NEEDS), default=1)
    return LICENSE_NEEDS[picked - 1][0]


def _local_ggufs(out_dir: Path) -> list[Path]:
    """``*.gguf`` already on disk in the cwd and in ``--out``, de-duplicated
    by resolved path (the same file reachable from two directories must not
    appear twice) and ordered F16/BF16-first: those are the files ``run``
    actually wants as ``--model``."""
    seen: dict[Path, Path] = {}
    for directory in (Path.cwd(), out_dir):
        try:
            found = sorted(directory.glob("*.gguf"))
        except OSError:  # pragma: no cover - unreadable directory
            continue
        for path in found:
            seen.setdefault(path.resolve(), path)
    return sorted(seen.values(), key=lambda p: (not _is_base_precision(p.name), str(p)))


def _curated_menu_lines(model: CuratedModel, hw: HardwareProfile) -> list[str]:
    return [
        f"     라이선스: {model.license}  {model.license_evidence}",
        f"     가중치: {model.base_model} / GGUF: {model.gguf_repo}",
        f"     {_memory_fit_line(model.size_bytes, hw)}",
        f"     실측 기록: {model.anchor} — 기록이지 이 컴퓨터의 예측이 아닙니다",
    ]


def _step_model(need: str, out_dir: Path, hw: HardwareProfile) -> Path:
    """Returns the path to use as ``--model``. Loops until one is chosen."""
    print("[4/6] 모델 선택")
    while True:
        local = _local_ggufs(out_dir)
        license_ok = [m for m in CURATED if license_allows(m.license, need)]
        # A model whose *F16* file exceeds the memory budget is shown and
        # stays selectable, flagged by its own "부족합니다" verdict line.
        # Excluding it would be wrong: the file that actually gets run is the
        # quantized one (Q4_K_M of Qwen3-4B is 2.3 GB, not 8.1 GB), and
        # search.py binary-searches ngl up from 0, so a model that does not
        # fit whole still runs partly offloaded. Hiding it hid this project's
        # own flagship Run-2 model on an 8 GB machine. The real constraint is
        # narrower -- see _F16_STAGE_WARNING, printed on selection.
        curated = license_ok
        tight = [m for m in curated if memory_fit(m.size_bytes, hw) is False]

        print()
        options: list[tuple[str, object]] = []
        if local:
            print("  이미 가지고 계신 .gguf 파일:")
            for path in local:
                options.append(("local", path))
                # --out fills up with FiTuna's own quantized artifacts, so an
                # unlabelled scan happily offers a Q2_K file as the *base*
                # model -- llama-quantize then refuses to requantize it. Same
                # label and ordering as the HF file picker.
                mark = "원본" if _is_base_precision(path.name) else "이미 양자화됨 — run이 경고합니다"
                print(f"  {len(options)}) {path}  ({mark})")
        if curated:
            print("  이 프로젝트가 직접 검증한 모델:")
            print(f"  ({_MEMORY_CAVEAT})")
            for model in curated:
                options.append(("curated", model))
                print(f"  {len(options)}) {model.label}")
                for line in _curated_menu_lines(model, hw):
                    print(line)
        license_excluded = len(CURATED) - len(license_ok)
        if license_excluded:
            print(f"  (라이선스 조건에 맞지 않아 {license_excluded}개는 목록에서 제외했습니다)")
        if tight:
            print(
                f"  (감지된 메모리보다 큰 모델 {len(tight)}개도 그대로 고를 수 있습니다 — "
                "제외하지 않습니다)"
            )
        options.append(("search", None))
        print(f"  {len(options)}) HuggingFace에서 검색")
        options.append(("manual", None))
        print(f"  {len(options)}) 직접 경로 입력")
        print()

        kind, payload = options[_ask_choice("번호를 고르세요", len(options), default=1) - 1]

        if kind == "local":
            assert isinstance(payload, Path)
            return payload
        if kind == "manual":
            path = Path(_ask("  .gguf 파일 또는 HF 포맷 디렉터리 경로"))
            if not path.exists():
                print(f"  {path} 을(를) 찾지 못했습니다.")
                continue
            return path

        # Network paths only. A failed search or download must drop the user
        # back into this menu, not abort a wizard they are four steps into.
        try:
            if kind == "curated":
                assert isinstance(payload, CuratedModel)
                if memory_fit(payload.size_bytes, hw) is False:
                    print()
                    print(_F16_STAGE_WARNING)
                chosen = _download_curated(payload, out_dir)
            else:
                chosen = _hf_search_flow(need, out_dir)
        except FiTunaError as exc:
            print(f"  {exc}")
            chosen = None
        if chosen is not None:
            return chosen


def _download_curated(model: CuratedModel, out_dir: Path) -> Optional[Path]:
    dest = out_dir / model.filename
    if dest.exists():
        print(f"  이미 있습니다: {dest}")
        return dest
    url = HF_RESOLVE.format(repo=model.gguf_repo, filename=model.filename)
    print(f"  받을 파일: {url}")
    print(f"  게시 시점 크기: {report.human_size(model.size_bytes)} → {dest}")
    print(f"  라이선스: {model.license} (가중치 제공: {model.base_model})")
    print("  이 파일의 라이선스는 FiTuna가 아니라 위 가중치 제공자의 것입니다.")
    if not _ask_yes_no("  내려받을까요?", True):
        return None
    return _download(url, dest)


def _resolved_license_link(candidate: HFCandidate) -> str:
    """The link to print/open for a candidate's license.

    ``cardData.license_link`` is often repo-relative (the live API returns
    the bare string ``"LICENSE"`` for many models, EXAONE among them) --
    printed as-is it reads as a stray word, not a link, even though the
    surrounding caveat tells the user to open it. Anything not already an
    absolute URL is resolved against the model's own repo on
    ``huggingface.co``.
    """
    link = candidate.license_link
    if not link:
        return f"https://huggingface.co/{candidate.model_id}"
    if link.startswith("http"):
        return link
    return f"https://huggingface.co/{candidate.model_id}/blob/main/{link}"


def _hf_search_flow(need: str, out_dir: Path) -> Optional[Path]:
    query = _ask("  검색어 (예: qwen3 4b)")
    # License-filtered candidates used to be dropped before this loop, so the
    # 제외 report silently under-counted: a search where every hit failed the
    # license filter printed nothing about licenses at all. Filter after
    # fetching and report all three exclusion reasons from the same list.
    found = _hf_search(query)
    candidates = [
        c
        for c in found
        if license_allows(c.license, need, model_id=c.model_id, base_model=c.base_model)
    ]
    usable = [c for c in candidates if not c.gated and c.gguf_files]
    for c in found:
        if c not in candidates:
            print(
                f"  (제외) {c.model_id} — 선택하신 라이선스 조건에 맞지 않습니다"
                f" (업로더 기재: {c.license or '표기 없음'})"
            )
        elif c.gated:
            print(f"  (제외) {c.model_id} — gated 저장소라 인증 없이 받을 수 없습니다")
        elif not c.gguf_files:
            print(f"  (제외) {c.model_id} — 단일 .gguf 파일이 없습니다")
    if not usable:
        print("  조건에 맞는 결과가 없습니다. 다른 검색어로 다시 시도해 보세요.")
        return None

    print()
    for i, c in enumerate(usable, start=1):
        print(f"  {i}) {c.model_id}  (다운로드 {c.downloads:,}회)")
        print(f"     라이선스(업로더 기재): {c.license or '표기 없음'} — {_resolved_license_link(c)}")
    print(f"  {len(usable) + 1}) 취소하고 돌아가기")
    print(f"  {_METADATA_CAVEAT}")
    print()

    picked = _ask_choice("번호를 고르세요", len(usable) + 1, default=1)
    if picked == len(usable) + 1:
        return None
    chosen = usable[picked - 1]

    files = _prefer_unquantized(chosen.gguf_files)
    print()
    print("  받을 파일을 고르세요 (F16/BF16 원본이 맨 위 — 이미 양자화된 파일을")
    print("  넘기면 fituna run 이 경고합니다):")
    for i, name in enumerate(files, start=1):
        mark = "  원본" if _is_base_precision(name) else "  이미 양자화됨"
        print(f"  {i}) {name}{mark}")
    print(f"  {len(files) + 1}) 취소하고 돌아가기")
    print()
    picked = _ask_choice("번호를 고르세요", len(files) + 1, default=1)
    if picked == len(files) + 1:
        return None

    filename = files[picked - 1]
    dest = out_dir / filename
    if dest.exists():
        print(f"  이미 있습니다: {dest}")
        return dest
    print(f"  라이선스: {chosen.license or '표기 없음'} (업로더 기재)  — {_METADATA_CAVEAT}")
    return _download(HF_RESOLVE.format(repo=chosen.model_id, filename=filename), dest)


def _step_corpus() -> Path:
    print("[5/6] 품질 평가용 코퍼스")
    print("  품질손실은 평문 코퍼스의 perplexity 증가율로 측정합니다 —")
    print("  실제로 쓸 언어의 텍스트로 재야 의미가 있습니다.")
    print()
    for i, (_lang, filename, label) in enumerate(CORPUS_CHOICES, start=1):
        print(f"  {i}) {label} → {filename}")
    print(f"  {len(CORPUS_CHOICES) + 1}) 이미 가지고 있는 텍스트 파일 사용")
    print()
    while True:
        picked = _ask_choice("번호를 고르세요", len(CORPUS_CHOICES) + 1, default=1)
        if picked == len(CORPUS_CHOICES) + 1:
            path = Path(_ask("  UTF-8 텍스트 파일 경로"))
            if path.is_file():
                return path
            print(f"  {path} 을(를) 찾지 못했습니다.")
            continue

        lang, filename, _label = CORPUS_CHOICES[picked - 1]
        out = Path(filename)
        if out.exists():
            print(f"  이미 있습니다: {out} (다시 받지 않습니다)")
            return out
        try:
            count = corpus.fetch_corpus(out, lang=lang, progress_cb=lambda msg: print(f"  {msg}"))
        except FiTunaError as exc:
            # Same reasoning as the model step: a network failure here must
            # not throw away four completed steps.
            print(f"  {exc}")
            continue
        print(f"  {out} 에 {count}행을 저장했습니다.")
        print(f"  {corpus.PRESETS[lang].license_note}")
        return out


def _lower_target_line(closest) -> Optional[str]:
    """The measured "you'd pass at X" line for the exit-3 path.

    Derived from the best-effort result the search actually measured -- so it
    is a measurement, not a prediction, which is the only reason the wizard
    is allowed to print a number here at all. ``search.py`` records a bench
    that timed out as ``0.0`` tok/s -- a sentinel meaning "never finished",
    not a measurement -- so a non-positive value here must not be printed as
    one; the whole point of this line is that its number was really measured.
    """
    if closest is None:
        return None
    if closest.bench.gen_tok_per_sec <= 0:
        return (
            "best-effort 후보의 벤치마크가 타임아웃되어 유효한 실측값이 없습니다 — "
            "목표를 얼마로 낮추면 통과하는지 알려드릴 수 없습니다."
        )
    return (
        f"목표를 {closest.bench.gen_tok_per_sec:.2f} tok/s 로 낮추면 아래 best-effort "
        f"구성({closest.config.quant}, ngl={closest.config.ngl}, ctx={closest.config.ctx})이 "
        "통과합니다 — 이 숫자는 방금 실제로 측정한 값입니다."
    )


def _step_run(argv: list[str]) -> int:
    print("[6/6] 확인 후 실행")
    print()
    print("  다음부터는 이 명령을 직접 쓰시면 됩니다:")
    print()
    print(f"    fituna {shlex.join(argv)}")
    print()
    if not _ask_yes_no("  지금 실행할까요?", True):
        print("  실행하지 않았습니다. 위 명령을 그대로 복사해 쓰시면 됩니다.")
        return 0

    # In-process, through the very same function cli.py dispatches `run` to,
    # parsed from the very same argv printed above -- so what runs and what
    # was shown cannot diverge. Imported here, not at module top: cli.py
    # imports this module.
    from fituna import cli

    run_args = cli._build_parser().parse_args(argv)
    print()
    try:
        return cli._cmd_run(run_args)
    except NoFeasibleConfigError as exc:
        line = _lower_target_line(exc.closest)
        if line:
            print()
            print(f"  {line}")
        # Re-raised, not handled: cli.main()'s exit-3 branch already logs the
        # best-effort report and (thanks to the export_ollama mirrored onto
        # our own namespace in run_wizard) writes the Modelfile too. Handling
        # it here would be a second copy of that logic.
        raise


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_wizard(args: argparse.Namespace) -> int:
    """``fituna quickstart``. Returns the exit code (``run``'s, once it gets
    that far), or raises for cli.main() to map -- see _step_run."""
    if not sys.stdin.isatty():
        print(
            "fituna quickstart는 대화형 터미널(TTY)이 필요합니다.\n"
            "파이프/CI 환경에서는 `fituna run --model <gguf> --target-tps <n> "
            "--max-quality-loss <n> --quality-corpus <txt>` 를 직접 쓰세요 "
            "(`fituna run -h`).",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out)
    bin_dir = Path(args.llama_bin_dir) if args.llama_bin_dir else None

    print("FiTuna quickstart — 6단계로 실측 구성을 찾습니다.")
    print(f"작업 디렉터리(--out): {out_dir}")
    print()

    try:
        code = _step_doctor(bin_dir, out_dir)
        if code != 0:
            return code
        print()

        target_tps, max_loss, ctx = _step_targets()
        print()
        need = _step_license()
        print()
        hw = hardware.detect_hardware()
        model = _step_model(need, out_dir, hw)
        print()
        quality_corpus = _step_corpus()
        print()
        export_ollama = _ask_yes_no("Ollama용 Modelfile도 만들까요?", False)
        # cli.main()'s exit-3 branch reads args.export_ollama to export from
        # the best-effort result; mirror the answer onto our own namespace so
        # that path behaves exactly like `fituna run --export-ollama`.
        args.export_ollama = export_ollama
        print()

        argv = build_run_argv(
            model=model,
            target_tps=target_tps,
            max_quality_loss=max_loss,
            ctx=ctx,
            quality_corpus=quality_corpus,
            out_dir=out_dir,
            export_ollama=export_ollama,
            llama_bin_dir=args.llama_bin_dir,
        )
        return _step_run(argv)
    except (EOFError, KeyboardInterrupt):
        print()
        print("취소했습니다. 언제든 `fituna quickstart` 로 다시 시작하실 수 있습니다.")
        return 1


# ---------------------------------------------------------------------------
# self-check (run: python -m fituna.quickstart, or --selfcheck -- any argv is
# ignored; this module has no CLI of its own). Pure logic only: no TTY, no
# network, no subprocess, no downloads.
# ---------------------------------------------------------------------------


def _selfcheck() -> None:
    from fituna.config import GPUVendor as _V

    # 1. license filter: the three needs, including the two exclusions the
    #    project actually verified (EXAONE research-only, CC BY-NC).
    assert license_allows("cc-by-nc-4.0", "personal")
    assert license_allows(None, "personal")
    assert not license_allows("cc-by-nc-4.0", "commercial")
    assert not license_allows("exaone", "commercial")
    assert license_allows("apache-2.0", "commercial")
    assert license_allows(None, "commercial")  # unknown: kept, with the caveat
    assert license_allows("mit", "redistribution")
    assert license_allows("apache-2.0", "redistribution")
    assert not license_allows("gemma", "redistribution")
    assert not license_allows(None, "redistribution")

    # 2. memory-fit arithmetic: fits / doesn't / nothing detected.
    gpu = HardwareProfile(_V.NVIDIA, "RTX 4090", 24564, 16, 65536, "linux")
    assert memory_fit(8_051_285_344, gpu) is True
    assert memory_fit(60_000_000_000, gpu) is False
    blind = HardwareProfile(_V.NONE, None, None, 4, 0, "linux")
    assert memory_fit(1, blind) is None
    assert "판정하지 않습니다" in _memory_fit_line(1, blind)
    assert "들어갑니다" in _memory_fit_line(1, gpu)
    assert "실측이 아니며" in _MEMORY_CAVEAT
    cpu_only = HardwareProfile(_V.NONE, None, None, 4, 8192, "linux")
    assert available_memory_mb(cpu_only) == (8192, "RAM")

    # 3. HF parser against the verified response shape (module docstring):
    #    license:"other" resolves through license_name, a missing cardData
    #    falls back to the tags entry, gated is a *string*, shards dropped.
    parsed = parse_hf_search(
        [
            {
                "id": "LGAI-EXAONE/EXAONE-4.0-32B-GGUF",
                "cardData": {"license": "other", "license_name": "exaone", "license_link": "LICENSE"},
                "gated": False,
                "downloads": 81725,
                "tags": ["gguf", "license:other"],
                "siblings": [
                    {"rfilename": "README.md"},
                    {"rfilename": "EXAONE-4.0-32B-BF16-00001-of-00002.gguf"},
                    {"rfilename": "EXAONE-4.0-32B-Q4_K_M.gguf"},
                ],
            },
            {"id": "someone/Model-GGUF", "tags": ["license:mit"], "gated": "manual", "siblings": []},
        ]
    )
    assert [c.model_id for c in parsed] == ["LGAI-EXAONE/EXAONE-4.0-32B-GGUF", "someone/Model-GGUF"]
    assert parsed[0].license == "exaone" and parsed[0].license_link == "LICENSE"
    assert parsed[0].gguf_files == ("EXAONE-4.0-32B-Q4_K_M.gguf",)  # shard dropped
    assert parsed[0].downloads == 81725 and parsed[0].gated is False
    assert parsed[1].license == "mit" and parsed[1].gated is True  # "manual" -> gated
    assert parse_hf_search({"error": "nope"}) == []  # shape change -> no results

    # 4. the no-wizard-only-features contract: the assembled argv must parse
    #    cleanly through the *real* cli parser and land on `run`'s own fields.
    from fituna import cli

    argv = build_run_argv(
        model=Path("m.gguf"),
        target_tps=20.0,
        max_quality_loss=5.0,
        ctx=4096,
        quality_corpus=Path("wiki.txt"),
        out_dir=Path("./out"),
        export_ollama=True,
        llama_bin_dir="/opt/llama",
    )
    parsed_args = cli._build_parser().parse_args(argv)
    assert parsed_args.command == "run"
    assert parsed_args.model == "m.gguf"
    assert parsed_args.target_tps == 20.0
    assert parsed_args.max_quality_loss == 5.0
    assert parsed_args.ctx == "4096"
    assert parsed_args.wikitext == "wiki.txt"
    assert parsed_args.out == str(Path("./out"))
    assert parsed_args.resume is True
    assert parsed_args.export_ollama is True
    assert parsed_args.llama_bin_dir == "/opt/llama"
    assert "--export-ollama" not in build_run_argv(
        Path("m.gguf"), 1.0, 1.0, 512, Path("w.txt"), Path("o"), False
    )

    # 5. curated shortlist matches docs/AI_MODEL_USAGE.md B-1/B-2/B-3 exactly
    #    (model ids, GGUF repos, licenses) -- the badge claims verified text.
    assert [m.base_model for m in CURATED] == [
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "Qwen/Qwen3-4B-Instruct-2507",
        "K-intelligence/Midm-2.0-Mini-Instruct",
    ]
    assert [m.gguf_repo for m in CURATED] == [
        "bartowski/SmolLM2-135M-Instruct-GGUF",
        "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "mykor/Midm-2.0-Mini-Instruct-gguf",
    ]
    assert [m.license for m in CURATED] == ["apache-2.0", "apache-2.0", "mit"]
    assert all(m.filename.endswith(".gguf") and m.size_bytes > 0 for m in CURATED)

    # 6. the wizard's copy must contain the refusal itself, not just imply it.
    assert "속도는 예측하지 않습니다" in _NO_SPEED_PREDICTION

    # 7. TTY guard: no TTY -> exit 1, and nothing else runs.
    class _NoTTY:
        def isatty(self) -> bool:
            return False

    import io

    saved_stdin, saved_stderr = sys.stdin, sys.stderr
    sys.stdin = _NoTTY()  # type: ignore[assignment]
    sys.stderr = io.StringIO()  # the guard's message is expected here, not a failure
    try:
        assert run_wizard(argparse.Namespace(out="./out", llama_bin_dir=None)) == 1
        assert "TTY" in sys.stderr.getvalue()
    finally:
        sys.stdin, sys.stderr = saved_stdin, saved_stderr

    print("fituna.quickstart self-check OK")


if __name__ == "__main__":
    _selfcheck()
