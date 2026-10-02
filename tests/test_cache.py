# SPDX-License-Identifier: MIT
"""sqlite3 ResultCache read/write + resume-scenario tests.

fituna.cache.ResultCache is fully implemented; these tests exercise the
public contract directly (real sqlite3 file under tmp_path, no mocking
needed -- the whole point of this cache is to be a thin, honest sqlite3
wrapper).
"""

import re
import sqlite3
from pathlib import Path

import pytest

from fituna.cache import ResultCache
from fituna.config import BenchResult, CandidateConfig, FiTunaError, QualityResult


def _bench(cand: CandidateConfig, gen_tps: float = 25.0) -> BenchResult:
    return BenchResult(
        candidate=cand,
        prompt_tok_per_sec=100.0,
        gen_tok_per_sec=gen_tps,
        vram_used_mb=6000,
        raw_stdout="{}",
    )


def _quality(quant: str = "Q4_K_M") -> QualityResult:
    return QualityResult(
        candidate_quant=quant,
        perplexity=6.1,
        baseline_perplexity=6.0,
        quality_loss_pct=1.67,
    )


@pytest.fixture
def cache(tmp_path: Path) -> ResultCache:
    c = ResultCache(tmp_path / "cache.sqlite3")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# bench_cache
# ---------------------------------------------------------------------------


def test_bench_roundtrip(cache: ResultCache):
    cand = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    result = _bench(cand)
    cache.put_bench("model-fp", "hw-fp", result)
    fetched = cache.get_bench("model-fp", "hw-fp", cand)
    assert fetched == result


def test_get_bench_miss_returns_none(cache: ResultCache):
    cand = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    assert cache.get_bench("no-such-model", "no-such-hw", cand) is None


def test_get_bench_miss_before_any_write(cache: ResultCache):
    # Empty cache: even a freshly-created db must miss cleanly, not error.
    cand = CandidateConfig(quant="Q4_K_M", ngl=0, ctx=4096)
    assert cache.get_bench("m", "h", cand) is None


@pytest.mark.parametrize(
    "other_model_fp,other_hw_fp,other_cand",
    [
        ("other-model", "hw-fp", None),  # different model_fp
        ("model-fp", "other-hw", None),  # different hw_fp
        (None, None, CandidateConfig(quant="Q5_K_M", ngl=32, ctx=4096)),  # diff quant
        (None, None, CandidateConfig(quant="Q4_K_M", ngl=0, ctx=4096)),  # diff ngl
        (None, None, CandidateConfig(quant="Q4_K_M", ngl=32, ctx=2048)),  # diff ctx
    ],
)
def test_bench_key_is_full_tuple(cache, other_model_fp, other_hw_fp, other_cand):
    """The primary key is (model_fp, hw_fp, quant, ngl, ctx) -- changing any
    single component must miss, proving no component is silently ignored."""
    cand = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    cache.put_bench("model-fp", "hw-fp", _bench(cand))

    lookup_model = other_model_fp or "model-fp"
    lookup_hw = other_hw_fp or "hw-fp"
    lookup_cand = other_cand or cand
    if (lookup_model, lookup_hw, lookup_cand) == ("model-fp", "hw-fp", cand):
        pytest.fail("test parametrization must vary at least one key component")
    assert cache.get_bench(lookup_model, lookup_hw, lookup_cand) is None


def test_put_bench_overwrites_on_same_key(cache: ResultCache):
    cand = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    cache.put_bench("model-fp", "hw-fp", _bench(cand, gen_tps=25.0))
    cache.put_bench("model-fp", "hw-fp", _bench(cand, gen_tps=30.0))
    fetched = cache.get_bench("model-fp", "hw-fp", cand)
    assert fetched.gen_tok_per_sec == 30.0
    # No duplicate rows: INSERT OR REPLACE, not INSERT.
    row_count = cache._conn.execute(
        "SELECT COUNT(*) FROM bench_cache WHERE model_fp=? AND hw_fp=? AND quant=? "
        "AND ngl=? AND ctx=?",
        ("model-fp", "hw-fp", cand.quant, cand.ngl, cand.ctx),
    ).fetchone()[0]
    assert row_count == 1


def test_bench_optional_vram_none_roundtrips(cache: ResultCache):
    cand = CandidateConfig(quant="Q4_K_M", ngl=0, ctx=4096)
    result = BenchResult(
        candidate=cand,
        prompt_tok_per_sec=10.0,
        gen_tok_per_sec=5.0,
        vram_used_mb=None,  # CPU-only bench: no VRAM reading available
        raw_stdout="cpu only",
    )
    cache.put_bench("model-fp", "hw-fp", result)
    fetched = cache.get_bench("model-fp", "hw-fp", cand)
    assert fetched == result
    assert fetched.vram_used_mb is None


def test_multiple_candidates_same_model_hw_coexist(cache: ResultCache):
    cands = [
        CandidateConfig(quant="Q4_K_M", ngl=0, ctx=4096),
        CandidateConfig(quant="Q4_K_M", ngl=16, ctx=4096),
        CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096),
        CandidateConfig(quant="Q8_0", ngl=32, ctx=4096),
    ]
    for i, cand in enumerate(cands):
        cache.put_bench("model-fp", "hw-fp", _bench(cand, gen_tps=float(i)))

    for i, cand in enumerate(cands):
        fetched = cache.get_bench("model-fp", "hw-fp", cand)
        assert fetched is not None
        assert fetched.gen_tok_per_sec == float(i)


# ---------------------------------------------------------------------------
# quality_cache
# ---------------------------------------------------------------------------


def test_quality_roundtrip(cache: ResultCache):
    quality = _quality("Q4_K_M")
    assert cache.get_quality("model-fp", "Q4_K_M") is None
    cache.put_quality("model-fp", quality)
    assert cache.get_quality("model-fp", "Q4_K_M") == quality


def test_get_quality_miss_returns_none(cache: ResultCache):
    assert cache.get_quality("no-such-model", "Q8_0") is None


def test_quality_key_ignores_neither_model_nor_quant(cache: ResultCache):
    cache.put_quality("model-fp", _quality("Q4_K_M"))
    assert cache.get_quality("other-model", "Q4_K_M") is None
    assert cache.get_quality("model-fp", "Q8_0") is None


def test_put_quality_overwrites_on_same_key(cache: ResultCache):
    cache.put_quality("model-fp", _quality("Q4_K_M"))
    updated = QualityResult(
        candidate_quant="Q4_K_M",
        perplexity=6.5,
        baseline_perplexity=6.0,
        quality_loss_pct=8.33,
    )
    cache.put_quality("model-fp", updated)
    fetched = cache.get_quality("model-fp", "Q4_K_M")
    assert fetched == updated
    row_count = cache._conn.execute(
        "SELECT COUNT(*) FROM quality_cache WHERE model_fp=? AND quant=?",
        ("model-fp", "Q4_K_M"),
    ).fetchone()[0]
    assert row_count == 1


def test_quality_per_quant_independent_of_ngl_ctx(cache: ResultCache):
    """QualityResult has no ngl/ctx dimension -- put_quality/get_quality key
    only on (model_fp, quant), matching the design's core insight that
    perplexity is independent of ngl/ctx."""
    cache.put_quality("model-fp", _quality("Q4_K_M"))
    cache.put_quality("model-fp", _quality("Q8_0"))
    assert cache.get_quality("model-fp", "Q4_K_M").candidate_quant == "Q4_K_M"
    assert cache.get_quality("model-fp", "Q8_0").candidate_quant == "Q8_0"


# ---------------------------------------------------------------------------
# --resume scenario: cache must survive being closed and reopened over the
# same db_path, since that's exactly what a second `fituna run --resume`
# invocation does.
# ---------------------------------------------------------------------------


def test_resume_reopen_sees_prior_bench_and_quality(tmp_path: Path):
    db_path = tmp_path / "resume.sqlite3"
    cand = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)

    first = ResultCache(db_path)
    first.put_bench("model-fp", "hw-fp", _bench(cand, gen_tps=30.0))
    first.put_quality("model-fp", _quality("Q4_K_M"))
    first.close()

    resumed = ResultCache(db_path)
    try:
        fetched_bench = resumed.get_bench("model-fp", "hw-fp", cand)
        assert fetched_bench is not None
        assert fetched_bench.gen_tok_per_sec == 30.0
        assert resumed.get_quality("model-fp", "Q4_K_M") == _quality("Q4_K_M")
    finally:
        resumed.close()


def test_resume_skips_recompute_for_cached_candidate(tmp_path: Path):
    """Simulates the actual --resume search loop: a candidate already in the
    cache should be servable without calling run_bench again."""
    db_path = tmp_path / "resume.sqlite3"
    cand = CandidateConfig(quant="Q6_K", ngl=20, ctx=4096)

    warm = ResultCache(db_path)
    warm.put_bench("model-fp", "hw-fp", _bench(cand, gen_tps=42.0))
    warm.close()

    calls = {"n": 0}

    def fake_run_bench(candidate: CandidateConfig) -> BenchResult:
        calls["n"] += 1
        return _bench(candidate, gen_tps=999.0)

    resumed = ResultCache(db_path)
    try:
        cached = resumed.get_bench("model-fp", "hw-fp", cand)
        result = cached if cached is not None else fake_run_bench(cand)
        assert result.gen_tok_per_sec == 42.0
        assert calls["n"] == 0  # subprocess bench never invoked -- cache hit
    finally:
        resumed.close()


def test_resume_new_db_path_starts_empty(tmp_path: Path):
    """A different db_path (fresh --out dir, no --resume history) must not
    see another run's cached data -- cache isolation is per file, not global."""
    cand = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)

    warm = ResultCache(tmp_path / "a.sqlite3")
    warm.put_bench("model-fp", "hw-fp", _bench(cand))
    warm.close()

    fresh = ResultCache(tmp_path / "b.sqlite3")
    try:
        assert fresh.get_bench("model-fp", "hw-fp", cand) is None
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# schema / lifecycle
# ---------------------------------------------------------------------------


def test_db_file_created_at_given_path(tmp_path: Path):
    db_path = tmp_path / "nested" / "dir" / "cache.sqlite3"
    assert not db_path.exists()
    c = ResultCache(db_path)
    try:
        assert db_path.exists()  # parent dirs auto-created, per contract
    finally:
        c.close()


def test_schema_has_expected_tables(cache: ResultCache):
    tables = {
        row[0]
        for row in cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"bench_cache", "quality_cache"}.issubset(tables)


def test_corrupt_db_file_raises_fitunaerror(tmp_path: Path):
    """A non-sqlite file at db_path (corrupt from an interrupted write, or
    an unrelated file happening to sit at this path) must surface as a
    FiTunaError with recovery guidance, not a bare sqlite3.DatabaseError."""
    db_path = tmp_path / "corrupt.sqlite3"
    db_path.write_bytes(b"not a real sqlite file, just garbage bytes")
    # re.escape: `match` is a regex, and a Windows path like C:\Users\...
    # contains \U -- an invalid regex escape that fails the whole test.
    with pytest.raises(FiTunaError, match=re.escape(str(db_path))):
        ResultCache(db_path)


def test_close_then_reuse_raises(tmp_path: Path):
    """Once closed, the underlying sqlite3 connection must refuse further
    use rather than silently corrupt state -- the caller is expected to
    treat close() as terminal."""
    c = ResultCache(tmp_path / "cache.sqlite3")
    cand = CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    c.close()
    with pytest.raises(sqlite3.ProgrammingError):
        c.get_bench("model-fp", "hw-fp", cand)


def test_quality_cache_keyed_by_corpus(tmp_path):
    """Perplexity is a property of (model, quant, corpus): a measurement on
    the English corpus must not be served for a request scoped to a Korean
    corpus, and vice versa."""
    cache = ResultCache(tmp_path / "c.sqlite3")
    en = QualityResult(candidate_quant="Q4_K_M", perplexity=6.1,
                       baseline_perplexity=6.0, quality_loss_pct=1.67)
    ko = QualityResult(candidate_quant="Q4_K_M", perplexity=9.5,
                       baseline_perplexity=9.0, quality_loss_pct=5.56)

    cache.put_quality("m", en, ppl_chunks=32, corpus_fp="corpus-en")
    assert cache.get_quality("m", "Q4_K_M", ppl_chunks=32, corpus_fp="corpus-ko") is None
    cache.put_quality("m", ko, ppl_chunks=32, corpus_fp="corpus-ko")
    assert cache.get_quality("m", "Q4_K_M", ppl_chunks=32, corpus_fp="corpus-en") == en
    assert cache.get_quality("m", "Q4_K_M", ppl_chunks=32, corpus_fp="corpus-ko") == ko
    cache.close()


def test_old_schema_quality_cache_dropped_not_served(tmp_path):
    """A cache file created before corpus_fp existed must be rebuilt, not
    have its corpus-ambiguous rows served as hits."""
    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """CREATE TABLE quality_cache (
               model_fp TEXT NOT NULL, quant TEXT NOT NULL,
               ppl_chunks INTEGER NOT NULL, perplexity REAL NOT NULL,
               baseline_perplexity REAL NOT NULL, loss_pct REAL NOT NULL,
               created_at TEXT NOT NULL,
               PRIMARY KEY (model_fp, quant, ppl_chunks));
           INSERT INTO quality_cache VALUES ('m', 'Q8_0', 32, 6.0, 6.0, 0.0, 'x');"""
    )
    conn.commit()
    conn.close()

    cache = ResultCache(db)
    assert cache.get_quality("m", "Q8_0", ppl_chunks=32) is None
    cache.close()


def test_quality_cache_distinguishes_metric(tmp_path):
    """Quality measurements under 'ppl' and 'kld' must be stored independently."""
    cache = ResultCache(tmp_path / "metric.sqlite3")
    ppl_res = QualityResult(
        candidate_quant="Q4_K_M",
        perplexity=6.1,
        baseline_perplexity=6.0,
        quality_loss_pct=1.67,
        metric="ppl",
    )
    kld_res = QualityResult(
        candidate_quant="Q4_K_M",
        perplexity=6.1,
        baseline_perplexity=6.0,
        quality_loss_pct=1.67,
        metric="kld",
        kld=0.00315,
    )

    cache.put_quality("m", ppl_res)
    assert cache.get_quality("m", "Q4_K_M", metric="ppl") == ppl_res
    assert cache.get_quality("m", "Q4_K_M", metric="kld") is None

    cache.put_quality("m", kld_res)
    assert cache.get_quality("m", "Q4_K_M", metric="ppl") == ppl_res
    assert cache.get_quality("m", "Q4_K_M", metric="kld") == kld_res
    cache.close()

