# SPDX-License-Identifier: MIT
"""fituna.cache
================

sqlite3-backed cache for bench/quality results, keyed by
(model_fingerprint, hardware_fingerprint, candidate) so re-running the same
search (``--resume``) skips subprocess calls that already have an answer.

Schema::

    bench_cache(model_fp, hw_fp, quant, ngl, ctx,
                prompt_tps, gen_tps, vram_mb, raw_stdout, created_at,
                PRIMARY KEY(model_fp, hw_fp, quant, ngl, ctx))

    quality_cache(model_fp, quant, ppl_chunks,
                  perplexity, baseline_perplexity, loss_pct, created_at,
                  PRIMARY KEY(model_fp, quant, ppl_chunks))

``ppl_chunks`` is part of the quality_cache key, not just an input to the
perplexity computation: a cached result computed over 32 chunks is not the
same measurement as one over the full corpus, and must not be silently
served as a --resume cache hit for a differently-scoped request. ``None``
(no chunk limit) is stored as the sentinel ``-1`` -- SQLite does allow NULL
inside a composite PRIMARY KEY, but NULLs are never considered equal to each
other for uniqueness, so "INSERT OR REPLACE" with a raw NULL key component
would accumulate a new row on every write instead of overwriting the
previous one.

``corpus_fp`` is part of the key for the same reason: perplexity is a
property of (model, quant, corpus). Without it, switching --quality-corpus
from English wikitext to a Korean corpus under --resume would silently
serve the English measurements as "Korean" results -- found designing
exactly that comparison experiment. Old cache files created before this
column existed are dropped and rebuilt (it's a cache; remeasuring is the
correct response to a schema upgrade).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fituna.config import BenchResult, CandidateConfig, FiTunaError, QualityResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bench_cache (
    model_fp TEXT NOT NULL,
    hw_fp TEXT NOT NULL,
    quant TEXT NOT NULL,
    ngl INTEGER NOT NULL,
    ctx INTEGER NOT NULL,
    prompt_tps REAL NOT NULL,
    gen_tps REAL NOT NULL,
    vram_mb INTEGER,
    raw_stdout TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (model_fp, hw_fp, quant, ngl, ctx)
);

CREATE TABLE IF NOT EXISTS quality_cache (
    model_fp TEXT NOT NULL,
    quant TEXT NOT NULL,
    ppl_chunks INTEGER NOT NULL,
    corpus_fp TEXT NOT NULL DEFAULT '',
    perplexity REAL NOT NULL,
    baseline_perplexity REAL NOT NULL,
    loss_pct REAL NOT NULL,
    metric TEXT NOT NULL DEFAULT 'ppl',
    kld REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (model_fp, quant, ppl_chunks, corpus_fp, metric)
);
"""

# Sentinel stored in quality_cache.ppl_chunks for "no chunk limit" (Python
# None) -- see the module docstring for why None itself can't be the key.
_UNLIMITED_CHUNKS = -1


def _chunks_key(chunks: Optional[int]) -> int:
    return _UNLIMITED_CHUNKS if chunks is None else chunks


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResultCache:
    """sqlite3-backed cache. One connection per instance; caller owns
    lifecycle (no context-manager requirement, but callers may close()).
    """

    def __init__(self, db_path: Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # single connection, no pooling -- this is a local CLI tool,
        # not a server; check_same_thread=False costs nothing since search()
        # runs sequentially anyway.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            # Pre-corpus_fp cache files have a quality_cache without the
            # corpus dimension; serving their rows would repeat the exact
            # stale-measurement bug the column exists to prevent. Drop and
            # rebuild -- a cache remeasures, it doesn't migrate.
            cols = {
                row[1]
                for row in self._conn.execute(
                    "PRAGMA table_info(quality_cache)"
                ).fetchall()
            }
            if cols and ("corpus_fp" not in cols or "metric" not in cols):
                self._conn.execute("DROP TABLE quality_cache")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.DatabaseError as exc:
            # sqlite3.connect() never touches the file -- the first real
            # operation does, and that's where "file is not a database"
            # (corrupt file, disk-full mid-write, or some unrelated file
            # sitting at this path) actually surfaces. Reproduced directly:
            # a garbage-bytes file at db_path raises this exact exception
            # here. Without this, --resume would crash with a bare sqlite3
            # traceback instead of telling the user what to do about it.
            self._conn.close()
            raise FiTunaError(
                f"cache file {db_path} is not a valid sqlite3 database ({exc}). "
                "It may be corrupt from an interrupted write, or an unrelated "
                "file happens to sit at this path -- delete it and re-run "
                "(the cache will be rebuilt from scratch)."
            ) from exc

    def close(self) -> None:
        self._conn.close()

    def get_bench(
        self, model_fp: str, hw_fp: str, cand: CandidateConfig
    ) -> Optional[BenchResult]:
        row = self._conn.execute(
            """SELECT prompt_tps, gen_tps, vram_mb, raw_stdout
               FROM bench_cache
               WHERE model_fp=? AND hw_fp=? AND quant=? AND ngl=? AND ctx=?""",
            (model_fp, hw_fp, cand.quant, cand.ngl, cand.ctx),
        ).fetchone()
        if row is None:
            return None
        prompt_tps, gen_tps, vram_mb, raw_stdout = row
        return BenchResult(
            candidate=cand,
            prompt_tok_per_sec=prompt_tps,
            gen_tok_per_sec=gen_tps,
            vram_used_mb=vram_mb,
            raw_stdout=raw_stdout,
        )

    def put_bench(self, model_fp: str, hw_fp: str, result: BenchResult) -> None:
        cand = result.candidate
        self._conn.execute(
            """INSERT OR REPLACE INTO bench_cache
               (model_fp, hw_fp, quant, ngl, ctx, prompt_tps, gen_tps, vram_mb,
                raw_stdout, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_fp,
                hw_fp,
                cand.quant,
                cand.ngl,
                cand.ctx,
                result.prompt_tok_per_sec,
                result.gen_tok_per_sec,
                result.vram_used_mb,
                result.raw_stdout,
                _now(),
            ),
        )
        self._conn.commit()

    def get_quality(
        self,
        model_fp: str,
        quant: str,
        ppl_chunks: Optional[int] = None,
        corpus_fp: str = "",
        metric: str = "ppl",
    ) -> Optional[QualityResult]:
        row = self._conn.execute(
            """SELECT perplexity, baseline_perplexity, loss_pct, metric, kld
               FROM quality_cache
               WHERE model_fp=? AND quant=? AND ppl_chunks=? AND corpus_fp=? AND metric=?""",
            (model_fp, quant, _chunks_key(ppl_chunks), corpus_fp, metric),
        ).fetchone()
        if row is None:
            return None
        perplexity, baseline_perplexity, loss_pct, metric_val, kld = row
        return QualityResult(
            candidate_quant=quant,
            perplexity=perplexity,
            baseline_perplexity=baseline_perplexity,
            quality_loss_pct=loss_pct,
            metric=metric_val,
            kld=kld,
        )

    def put_quality(
        self,
        model_fp: str,
        result: QualityResult,
        ppl_chunks: Optional[int] = None,
        corpus_fp: str = "",
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO quality_cache
               (model_fp, quant, ppl_chunks, corpus_fp, perplexity,
                baseline_perplexity, loss_pct, metric, kld, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_fp,
                result.candidate_quant,
                _chunks_key(ppl_chunks),
                corpus_fp,
                result.perplexity,
                result.baseline_perplexity,
                result.quality_loss_pct,
                result.metric,
                result.kld,
                _now(),
            ),
        )
        self._conn.commit()


def _self_check() -> None:
    """Minimal assert-based sanity check -- roundtrip + miss + resume-reopen.

    Full behavioral coverage lives in tests/test_cache.py; this is just a
    runnable guard against the two ways this file regresses silently: schema
    drift and a cache that doesn't survive being reopened (the whole point of
    ``--resume`` is a fresh ResultCache() over the same db_path).
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "cache.sqlite3"
        cache = ResultCache(db_path)

        cand = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
        bench = BenchResult(
            candidate=cand,
            prompt_tok_per_sec=100.0,
            gen_tok_per_sec=25.0,
            vram_used_mb=6000,
            raw_stdout="{}",
        )

        # 1. Miss before write.
        assert cache.get_bench("model-fp", "hw-fp", cand) is None

        # 2. Roundtrip.
        cache.put_bench("model-fp", "hw-fp", bench)
        fetched = cache.get_bench("model-fp", "hw-fp", cand)
        assert fetched == bench

        # 3. Different hw_fp/candidate is a distinct key (miss).
        assert cache.get_bench("model-fp", "other-hw", cand) is None
        other_cand = CandidateConfig(quant="Q4_K_M", ngl=0, ctx=4096)
        assert cache.get_bench("model-fp", "hw-fp", other_cand) is None

        # 4. INSERT OR REPLACE: put_bench again with new numbers overwrites, no
        #    duplicate-key crash.
        bench2 = BenchResult(
            candidate=cand,
            prompt_tok_per_sec=110.0,
            gen_tok_per_sec=30.0,
            vram_used_mb=6100,
            raw_stdout="{}",
        )
        cache.put_bench("model-fp", "hw-fp", bench2)
        assert cache.get_bench("model-fp", "hw-fp", cand).gen_tok_per_sec == 30.0

        # 5. Quality cache roundtrip + miss.
        quality = QualityResult(
            candidate_quant="Q4_K_M",
            perplexity=6.1,
            baseline_perplexity=6.0,
            quality_loss_pct=1.67,
        )
        assert cache.get_quality("model-fp", "Q4_K_M") is None
        cache.put_quality("model-fp", quality)
        assert cache.get_quality("model-fp", "Q4_K_M") == quality
        assert cache.get_quality("model-fp", "Q8_0") is None

        # 5b. ppl_chunks is part of the key: a result cached at the default
        # (None == unlimited) must NOT be served for a request scoped to a
        # different chunk count -- and vice versa -- even though it uses the
        # exact same put_quality/get_quality call with a different keyword.
        assert cache.get_quality("model-fp", "Q4_K_M", ppl_chunks=32) is None
        quality_32 = QualityResult(
            candidate_quant="Q4_K_M", perplexity=6.2,
            baseline_perplexity=6.0, quality_loss_pct=3.33,
        )
        cache.put_quality("model-fp", quality_32, ppl_chunks=32)
        assert cache.get_quality("model-fp", "Q4_K_M", ppl_chunks=32) == quality_32
        assert cache.get_quality("model-fp", "Q4_K_M") == quality  # unlimited entry untouched
        assert cache.get_quality("model-fp", "Q4_K_M", ppl_chunks=64) is None

        cache.close()

        # 6. --resume scenario: reopening the same db_path must see prior data.
        reopened = ResultCache(db_path)
        assert reopened.get_bench("model-fp", "hw-fp", cand).gen_tok_per_sec == 30.0
        assert reopened.get_quality("model-fp", "Q4_K_M") == quality
        reopened.close()

        # 7. A corrupt/non-sqlite file at db_path must raise a FiTunaError
        #    with recovery guidance, not a bare sqlite3.DatabaseError.
        corrupt_path = Path(tmp) / "corrupt.sqlite3"
        corrupt_path.write_bytes(b"not a real sqlite file, just garbage bytes")
        try:
            ResultCache(corrupt_path)
            raise AssertionError("expected FiTunaError for a corrupt cache file")
        except FiTunaError as exc:
            assert not isinstance(exc, AssertionError)
            assert str(corrupt_path) in str(exc)


if __name__ == "__main__":
    _self_check()
    print("fituna.cache self-check OK")
