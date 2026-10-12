# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FiTuna contributors
"""fituna.corpus
================

Fetches a plain-text quality-evaluation corpus from HuggingFace's public
"datasets-server" REST API -- no auth, no third-party packages -- and writes
it to a UTF-8 text file, one row's text per line: the same format
``fituna.quality.compute_perplexity`` already expects for ``--quality-corpus``.

Why this exists: the only previously-documented way to get this corpus was
``pip install datasets`` plus a Python snippet, which drags in pyarrow/pandas
(hundreds of MB) and contradicts FiTuna's zero-runtime-dependency claim. This
module replaces it with stdlib ``urllib`` only.

API shape -- verified by hand against the live API on 2026-07-30, not
guessed:

    GET https://datasets-server.huggingface.co/rows
        ?dataset=<dataset>&config=<config>&split=<split>&offset=<n>&length=<n>

    -> 200 {"features": [{"feature_idx": int, "name": str, "type": {...}}, ...],
            "rows": [{"row_idx": int, "row": {<field>: <value>, ...},
                       "truncated_cells": [...]}, ...],
            "num_rows_total": int, "num_rows_per_page": int, "partial": bool}

Confirmed characteristics that shaped this module:
- ``length`` is server-capped at 100 rows/request -- HTTP 422
  ``{"error": "Parameter 'length' must not be greater than 100"}`` above
  that -- so fetching more than 100 rows means paginating with ``offset``.
- An ``offset`` at or past the end of the split is *not* an error: the
  server returns 200 with ``"rows": []`` (or fewer rows than asked for).
  That is this module's "stop paginating" signal, not an exception.
- A non-existent/private/gated dataset returns 401 with
  ``{"error": "..."}``; a malformed request returns 422 the same way.
- The default urllib User-Agent is accepted; no custom headers needed.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from fituna.config import CorpusPreset, FiTunaError

API_BASE = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # server-enforced hard cap on `length`; see module docstring
TIMEOUT_SEC = 30

# Both license facts below were read from each dataset's own HuggingFace
# metadata (https://huggingface.co/api/datasets/<id> -> cardData.license),
# not assumed: both report ["cc-by-sa-3.0", "gfdl"].
PRESETS: dict[str, CorpusPreset] = {
    "en": CorpusPreset(
        dataset="Salesforce/wikitext",
        config="wikitext-2-raw-v1",
        split="test",
        text_field="text",
        default_rows=1000,
        license_note=(
            "Corpus: Salesforce/wikitext (wikitext-2-raw-v1, test split). "
            "License: CC BY-SA 3.0 (also dual-licensed GFDL) -- attribution "
            "and share-alike apply. "
            "Source: https://huggingface.co/datasets/Salesforce/wikitext"
        ),
    ),
    "ko": CorpusPreset(
        dataset="wikimedia/wikipedia",
        config="20231101.ko",
        split="train",
        text_field="text",
        default_rows=500,
        license_note=(
            "Corpus: wikimedia/wikipedia (20231101.ko, train split). "
            "License: CC BY-SA 3.0 (also dual-licensed GFDL) -- attribution "
            "and share-alike apply. "
            "Source: https://huggingface.co/datasets/wikimedia/wikipedia"
        ),
    ),
}

_MANUAL_FALLBACK = (
    "Download it manually from the dataset's HuggingFace page instead, or "
    "point --quality-corpus at any UTF-8 plain-text file you already have -- "
    "FiTuna's quality gate only needs text resembling your workload, not "
    "this specific corpus."
)

# Shared guidance for every way writing to `--out` can fail: its parent
# already existing as a file, its parent directory not being writable, or
# the final rename failing. One wording for all three, reused below, so the
# error looks the same regardless of which step actually failed.
_OUT_WRITE_GUIDANCE = (
    "Check that --out names a file path, not an existing directory, and "
    "that its parent directory is writable (on Windows, also make sure the "
    "destination isn't open in another program), then try again."
)


def _resolve_source(
    lang: str, dataset: Optional[str], config: Optional[str], split: Optional[str]
) -> tuple[str, str, str, str]:
    """(dataset, config, split, text_field) -- either the ``lang`` preset's
    values, or the caller's ``--dataset``/``--config``/``--split`` override.

    The override only takes effect when all three are given together (see
    task brief): a partial override is rejected with a FiTunaError rather
    than silently falling back to the preset, which would otherwise look
    like the override "worked" when it silently didn't.
    """
    if lang not in PRESETS:
        raise FiTunaError(
            f"unknown --lang {lang!r}; expected one of: {', '.join(sorted(PRESETS))}"
        )
    preset = PRESETS[lang]

    given = {"--dataset": dataset, "--config": config, "--split": split}
    n_given = sum(v is not None for v in given.values())
    if n_given == 3:
        assert dataset is not None and config is not None and split is not None
        return dataset, config, split, preset.text_field
    if n_given == 0:
        return preset.dataset, preset.config, preset.split, preset.text_field

    missing = ", ".join(k for k, v in given.items() if v is None)
    raise FiTunaError(
        "--dataset/--config/--split must all be given together to override "
        f"the --lang {lang!r} preset (missing {missing})."
    )


def _fetch_page(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    """One GET against the dataset-viewer rows API (shape verified -- see
    module docstring). Never lets a raw urllib/json exception escape --
    always FiTunaError, with guidance on what to do instead rather than a
    stack trace (this module has no retry logic by design -- see brief)."""
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    url = f"{API_BASE}?{query}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SEC) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise FiTunaError(
            f"HuggingFace dataset-viewer API returned HTTP {exc.code} for "
            f"dataset={dataset!r} config={config!r} split={split!r} "
            f"(offset={offset}, length={length}): {detail}\n{_MANUAL_FALLBACK}"
        ) from exc
    except OSError as exc:
        # urllib.error.URLError (DNS failure, connection refused, connect-time
        # timeout via its .reason) is itself an OSError subclass; a bare
        # TimeoutError/ConnectionResetError can also surface directly from
        # resp.read() on a connection urlopen() already established. One
        # catch covers all of it -- same convention as fituna.hardware._run.
        raise FiTunaError(
            f"could not reach the HuggingFace dataset-viewer API: {exc}. "
            f"{_MANUAL_FALLBACK}"
        ) from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise FiTunaError(
            f"HuggingFace dataset-viewer API returned unparseable JSON for "
            f"dataset={dataset!r}: {exc}. {_MANUAL_FALLBACK}"
        ) from exc


def fetch_corpus(
    out: Path,
    lang: str = "en",
    rows: Optional[int] = None,
    dataset: Optional[str] = None,
    config: Optional[str] = None,
    split: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """Fetch up to ``rows`` rows of plain text from the HuggingFace
    dataset-viewer API and write them, one per line, to ``out`` as UTF-8.

    Write is atomic: content is written to a temp file in ``out``'s own
    directory and only ``os.replace``-d into ``out`` after every requested
    row is fetched successfully. Any failure -- network, HTTP, malformed
    response, preparing the destination (``out``'s parent already existing
    as a plain file, or a parent directory that isn't writable), or the
    final rename itself -- deletes the temp file and re-raises as
    ``FiTunaError`` -- ``out`` is never left holding a partial download and
    no temp file is ever left behind.

    ``rows`` defaults to the ``lang`` preset's ``default_rows`` (1000 for
    en, 500 for ko) when not given -- even when ``dataset``/``config``/
    ``split`` override the preset's source, since ``--lang`` is still the
    only signal for which default row count applies. A resolved row count
    that is not positive (``rows=0`` or negative) raises ``FiTunaError``
    immediately, before any file or network I/O. A *positive* row count
    whose split nonetheless yields zero rows on the very first page (a
    valid HTTP 200 with ``"rows": []``) also raises ``FiTunaError`` --
    otherwise this would write a 0-byte file and print a full license
    notice for a corpus that doesn't exist, the same dishonesty the
    non-positive-``rows`` guard above already exists to prevent.

    Reports progress via ``progress_cb(str)`` if given, once per page
    fetched (e.g. ``"fetched 300/1000 rows"``).

    Returns the number of rows actually written (always positive -- see
    above), which may be less than requested if the split runs out first
    (not an error -- see module docstring's "offset past the end" note).
    """
    resolved_dataset, resolved_config, resolved_split, text_field = _resolve_source(
        lang, dataset, config, split
    )
    total = rows if rows is not None else PRESETS[lang].default_rows
    if total <= 0:
        # Reject before touching the filesystem or network at all -- a
        # 0-byte file with a full CC BY-SA license notice printed over it
        # would otherwise be the (silent, misleading) result of `--rows 0`
        # or a negative --rows.
        raise FiTunaError(f"--rows must be positive, got {total}")
    progress: Callable[[str], None] = progress_cb or (lambda _msg: None)

    try:
        # Two sibling bad-`--out` shapes land here: `out.parent` already
        # exists as a plain file (mkdir raises FileExistsError), or it
        # exists but isn't writable (mkstemp raises PermissionError). Both
        # must become the same clear FiTunaError as a failed os.replace
        # below, not a raw traceback.
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    except OSError as exc:
        raise FiTunaError(
            f"could not prepare {out} for writing: {exc}. {_OUT_WRITE_GUIDANCE}"
        ) from exc
    tmp_path = Path(tmp_name)
    written = 0
    schema_checked = False
    try:
        # newline="\n": pin LF-only output on every OS (including Windows
        # CI) so the same --lang fetch produces byte-identical corpus files
        # regardless of host platform.
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            offset = 0
            while written < total:
                length = min(PAGE_SIZE, total - written)
                page = _fetch_page(resolved_dataset, resolved_config, resolved_split, offset, length)

                if not schema_checked:
                    # Only the two built-in presets are verified to have a
                    # "text" field; a custom --dataset/--config/--split
                    # override could point at a dataset with a differently
                    # named text column, which would otherwise silently
                    # write an all-blank-lines corpus with no error at all.
                    field_names = [feat.get("name") for feat in page.get("features", [])]
                    if text_field not in field_names:
                        raise FiTunaError(
                            f"dataset={resolved_dataset!r} config={resolved_config!r} "
                            f"split={resolved_split!r} has no {text_field!r} field "
                            f"(available fields: {', '.join(n for n in field_names if n) or 'none'}). "
                            f"{_MANUAL_FALLBACK}"
                        )
                    schema_checked = True

                page_rows = page.get("rows", [])
                if not page_rows:
                    break  # split exhausted before reaching `total`; not an error
                for row in page_rows:
                    f.write(row.get("row", {}).get(text_field, ""))
                    f.write("\n")
                written += len(page_rows)
                offset += len(page_rows)
                progress(f"fetched {written}/{total} rows")

        if written == 0:
            # The fetched-count twin of the `total <= 0` guard above: a
            # split that returns zero rows on its very first page (a valid
            # HTTP 200, not an error) would otherwise silently produce a
            # 0-byte corpus file plus a full CC BY-SA license notice for
            # content that doesn't exist.
            raise FiTunaError(
                f"dataset={resolved_dataset!r} config={resolved_config!r} "
                f"split={resolved_split!r} returned no rows at all -- "
                f"nothing to write. {_MANUAL_FALLBACK}"
            )

        # Rename must stay inside this try: a `run`/`doctor` habit of
        # pointing `--out` at a directory (that flag means a directory for
        # those subcommands, a file here) makes os.replace raise
        # IsADirectoryError; a read-only destination directory or -- on
        # Windows -- the destination being open elsewhere raises
        # PermissionError instead. Either way this must land in the
        # `except BaseException` below so the temp file is always cleaned
        # up, and become a FiTunaError so the user gets guidance instead of
        # a raw traceback.
        try:
            os.replace(tmp_path, out)
        except OSError as exc:
            raise FiTunaError(
                f"could not save the fetched corpus to {out}: {exc}. {_OUT_WRITE_GUIDANCE}"
            ) from exc
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return written


# ---------------------------------------------------------------------------
# self-check (run: python -m fituna.corpus -- urllib.request.urlopen is
# faked in-process, no real network access needed or used)
# ---------------------------------------------------------------------------


def _selfcheck() -> None:
    import io
    import tempfile as _tempfile

    # 1. Preset data matches the task brief's exact values, verbatim.
    en = PRESETS["en"]
    assert (en.dataset, en.config, en.split, en.text_field, en.default_rows) == (
        "Salesforce/wikitext", "wikitext-2-raw-v1", "test", "text", 1000
    ), en
    ko = PRESETS["ko"]
    assert (ko.dataset, ko.config, ko.split, ko.text_field, ko.default_rows) == (
        "wikimedia/wikipedia", "20231101.ko", "train", "text", 500
    ), ko
    assert "CC BY-SA" in en.license_note and "CC BY-SA" in ko.license_note

    # 2. preset/override resolution -- pure logic, no network.
    assert _resolve_source("en", None, None, None) == (
        "Salesforce/wikitext", "wikitext-2-raw-v1", "test", "text",
    )
    assert _resolve_source("ko", "x/y", "cfg", "split") == ("x/y", "cfg", "split", "text")
    try:
        _resolve_source("en", "x/y", None, None)
        raise AssertionError("expected FiTunaError for a partial dataset/config/split override")
    except FiTunaError:
        pass
    try:
        _resolve_source("de", None, None, None)
        raise AssertionError("expected FiTunaError for an unknown --lang")
    except FiTunaError:
        pass

    # 3. fetch_corpus with urllib.request.urlopen faked in-process: normal
    #    pagination assembly, progress_cb messages, and the "split has
    #    fewer rows than requested" path (not an error).
    def _fake_page(total: int, offset: int, length: int) -> dict:
        remaining = max(0, total - offset)
        n = min(length, remaining)
        rows = [
            {"row_idx": offset + i, "row": {"text": f"row{offset + i}"}, "truncated_cells": []}
            for i in range(n)
        ]
        return {
            "features": [{"feature_idx": 0, "name": "text", "type": {"dtype": "string", "_type": "Value"}}],
            "rows": rows,
            "num_rows_total": total,
            "num_rows_per_page": PAGE_SIZE,
            "partial": False,
        }

    def _fake_urlopen(url, timeout=None):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        payload = _fake_page(total=7, offset=int(qs["offset"][0]), length=int(qs["length"][0]))
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    orig_urlopen = urllib.request.urlopen
    urllib.request.urlopen = _fake_urlopen
    try:
        with _tempfile.TemporaryDirectory() as td:
            out = Path(td) / "corpus.txt"
            messages: list = []
            n = fetch_corpus(out, lang="en", rows=7, progress_cb=messages.append)
            assert n == 7, n
            assert out.read_text(encoding="utf-8").splitlines() == [f"row{i}" for i in range(7)]
            assert messages[-1] == "fetched 7/7 rows", messages

            # asking for more rows than the (fake) split has: stop early, no error.
            out2 = Path(td) / "corpus2.txt"
            n2 = fetch_corpus(out2, lang="en", rows=1000)
            assert n2 == 7, n2
    finally:
        urllib.request.urlopen = orig_urlopen

    # 4. atomic write: a failure partway through must leave nothing at
    #    `out` and no leftover temp file -- the safety property the brief
    #    calls out explicitly.
    def _fake_urlopen_drops_on_page_2(url, timeout=None):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        offset = int(qs["offset"][0])
        if offset > 0:
            raise urllib.error.URLError("simulated connection drop")
        payload = _fake_page(total=500, offset=0, length=int(qs["length"][0]))
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    urllib.request.urlopen = _fake_urlopen_drops_on_page_2
    try:
        with _tempfile.TemporaryDirectory() as td:
            out3 = Path(td) / "corpus3.txt"
            try:
                fetch_corpus(out3, lang="en", rows=250)  # needs 3 pages of 100
                raise AssertionError("expected FiTunaError from the simulated connection drop")
            except FiTunaError:
                pass
            assert not out3.exists(), "a partial download must not leave a file at the target path"
            assert list(Path(td).iterdir()) == [], "no leftover temp file should remain after a failure"
    finally:
        urllib.request.urlopen = orig_urlopen

    print("fituna.corpus self-check OK")


if __name__ == "__main__":
    _selfcheck()
