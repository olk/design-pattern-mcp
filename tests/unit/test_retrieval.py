"""Unit tests for the hybrid retrieval index (§2.2, §6): offline implementations only.

All tests run without network: the dense leg uses :class:`OfflineHashEmbedder`
(deterministic hashed bag-of-words), the rerank leg :class:`LexicalReranker`
(token overlap); TEI HTTP behavior is exercised against a monkeypatched
``httpx.Client.post``.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.patterns.loader import PatternLoader, JoinedPattern, LanguageEntry, SchemaEnums
from src.config import (
    DenseCacheConfig,
    EmbedderConfig,
    EmbedderInnerConfig,
    RetrievalConfig,
    RerankerConfig,
    RerankerInnerConfig,
)
from src.errors import ERROR_RETRIEVAL_UNAVAILABLE, DesignPatternError
from src.patterns.embedder import EmbedderIdentity, OfflineHashEmbedder, TEIEmbedder, build_embedder
from src.patterns.dense_cache import PROBE_TEXT
from src.patterns.retriever import PartQuery, RetrievalIndex
from src.patterns.safe_tei_rerank import LexicalReranker, TEIReranker, build_reranker

RRF_K = 60


# --------------------------------------------------------------------------------------
# Fixture: a small hand-made catalog with distinct pattern vocabularies
# --------------------------------------------------------------------------------------


def _record(
    name: str,
    intent: str,
    problem: str,
    context: str,
    use_cases: tuple[str, ...],
    aliases: tuple[str, ...] = (),
) -> JoinedPattern:
    """Minimal joined record carrying only the retrieval-relevant fields."""
    return JoinedPattern(
        name=name,
        category="concurrency-resilience" if name == "timeout" else "construction-composition",
        subgroup="sub",
        intent=intent,
        problem=problem,
        context=context,
        aliases=aliases,
        benefits=(),
        tradeoffs=(),
        quality_attributes={"simplicity": 5.0},
        language_implementations={
            "generic": LanguageEntry(language="generic", support="recommended", implementation_notes="n"),
        },
        use_cases=use_cases,
    )


def _catalog(records: list[JoinedPattern]) -> PatternLoader:
    """PatternLoader over ``records`` with identity order and the records' own alias index."""
    return PatternLoader(
        records={record.name: record for record in records},
        order={record.name: position for position, record in enumerate(records)},
        subgroups={},
        alias_map={alias.lower(): record.name for record in records for alias in record.aliases},
        ambiguous_aliases=(),
        enums=SchemaEnums(categories=(), scopes=(), languages=(), support_levels=(), paradigms=(), provenances=()),
    )


def _records() -> list[JoinedPattern]:
    return [
        _record(
            "singleton",
            "Ensure a class has only one instance and provide a global access point to it",
            "Global state must be shared across the application without passing a reference everywhere",
            "Configuration access from arbitrary modules of one process",
            ("logging", "configuration access"),
            aliases=("Borg",),
        ),
        _record(
            "strategy",
            "Define a family of interchangeable algorithms and select one at runtime",
            "An algorithm variant must be selected at runtime without littering the client with conditionals",
            "Pricing engine with interchangeable discount calculations",
            ("payment processing", "sorting algorithms"),
        ),
        _record(
            "repository",
            "Mediate between the domain and data mapping layers with a collection-like interface",
            "Domain logic must not depend on persistence infrastructure details",
            "A domain model persisted to a relational database",
            ("data access abstraction",),
        ),
        _record(
            "observer",
            "Define a one-to-many dependency so state changes notify all dependents",
            "State changes must propagate to dependent objects without tight coupling",
            "Event notification across components of a dashboard",
            ("event publication",),
        ),
        _record(
            "timeout",
            "Bound the waiting time of a remote call with a deadline",
            "Remote calls may hang indefinitely and exhaust threads and memory",
            "Service-to-service communication over unreliable networks",
            ("resilience",),
        ),
        _record(
            "data-transfer-object",
            "Carry data between processes in a serializable envelope",
            "Serializing domain entities across the wire couples them to the transport",
            "Remote API boundary between two services",
            ("service communication",),
        ),
    ]


def _retrieval_config(
    *,
    dense_top_k: int | None = None,
    bm25_top_k: int | None = None,
) -> RetrievalConfig:
    overrides: dict[str, int] = {}
    if dense_top_k is not None:
        overrides["dense_top_k"] = dense_top_k
    if bm25_top_k is not None:
        overrides["bm25_top_k"] = bm25_top_k
    return RetrievalConfig(dense_weight=0.7, bm25_weight=0.3, rrf_k=RRF_K, min_fusion_score=0.0, **overrides)


_SINGLETON_QUERY = "Share global configuration state through a single instance access point"


async def _build_multi(
    records: list[JoinedPattern] | None = None,
    rerank_top_n: int = 3,
    *,
    dense_cache: DenseCacheConfig | None = None,
    identity: EmbedderIdentity | None = None,
    embedder: OfflineHashEmbedder | None = None,
) -> RetrievalIndex:
    """Index over the hand-made catalog with the deterministic offline implementations."""
    return await RetrievalIndex.build(
        _catalog(records if records is not None else _records()),
        embedder if embedder is not None else OfflineHashEmbedder(),
        LexicalReranker(),
        _retrieval_config(),
        rerank_top_n,
        dense_cache=dense_cache,
        identity=identity,
    )


# --------------------------------------------------------------------------------------
# (a) own-vocabulary query ranks the pattern inside the returned survivors
# --------------------------------------------------------------------------------------


async def test_own_vocabulary_query_ranks_pattern_in_survivors() -> None:
    index = await _build_multi()
    hits = await index.retrieve(
        statement=_SINGLETON_QUERY,
        constraints=(),
        scope="process-lifetime",
    )
    names = [hit.name for hit in hits]
    assert "singleton" in names
    # The exact-name match must not be crowded out by lexically related records.
    assert names.index("singleton") < len(names) - 1


# --------------------------------------------------------------------------------------
# (b) retrieval_score equals RR(fused_rank) + RR(rerank_rank) recomputed per hit
# --------------------------------------------------------------------------------------


async def test_retrieval_score_is_reciprocal_rank_blend() -> None:
    index = await _build_multi()
    hits = await index.retrieve(statement=_SINGLETON_QUERY, constraints=(), scope="process-lifetime")
    assert hits
    for hit in hits:
        expected = 1 / (RRF_K + hit.fused_rank - 1) + 1 / (RRF_K + hit.rerank_rank - 1)
        assert math.isclose(hit.retrieval_score, expected, rel_tol=1e-12), hit


# --------------------------------------------------------------------------------------
# (c) survivor cap: rerank_top_n by default, exact with the limit keyword
# --------------------------------------------------------------------------------------


async def test_result_length_capped_by_rerank_top_n_and_limit() -> None:
    index = await _build_multi(rerank_top_n=3)
    default_hits = await index.retrieve(statement=_SINGLETON_QUERY, constraints=(), scope="process-lifetime")
    assert len(default_hits) == 3
    limited_hits = await index.retrieve(statement=_SINGLETON_QUERY, constraints=(), scope="process-lifetime", limit=2)
    assert len(limited_hits) == 2
    scores = [hit.retrieval_score for hit in limited_hits]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------------------
# (d) sticky reselection exclusions are removed before ranking
# --------------------------------------------------------------------------------------


async def test_exclude_removes_pattern_from_results() -> None:
    index = await _build_multi()
    hits = await index.retrieve(
        statement=_SINGLETON_QUERY,
        constraints=(),
        scope="process-lifetime",
        exclude={"singleton", "Borg"},
    )
    assert all(hit.name != "singleton" for hit in hits)


# --------------------------------------------------------------------------------------
# (e) single-candidate pool skips the reranker; rerank_rank falls back to fused_rank
# --------------------------------------------------------------------------------------


class _RecordingReranker:
    """Reranker seam that records the document pool it was handed."""

    def __init__(self) -> None:
        self.documents: list[str] = []

    async def rerank(self, query: str, documents: Any, top_n: int) -> list[tuple[int, float]]:
        self.documents = list(documents)
        return [(index, 1.0 / (index + 1)) for index in range(len(documents))]


class _ExplodingReranker:
    """Reranker seam that fails the test when invoked (single-candidate pools skip it)."""

    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: str, documents: Any, top_n: int) -> list[tuple[int, float]]:
        self.calls += 1
        raise AssertionError("reranker must not run for a single-candidate pool")


async def test_single_candidate_pool_skips_reranker() -> None:
    records = [
        _record(
            "timeout", "Bound the waiting time of a remote call", "Remote calls may hang", "Networks", ("resilience",)
        )
    ]
    reranker = _ExplodingReranker()
    index = await RetrievalIndex.build(_catalog(records), OfflineHashEmbedder(), reranker, _retrieval_config(), 5)
    hits = await index.retrieve(statement="Remote calls may hang", constraints=(), scope="task-lifetime")
    assert len(hits) == 1
    hit = hits[0]
    assert hit.name == "timeout"
    assert hit.fused_rank == 1
    assert hit.rerank_rank == hit.fused_rank
    assert reranker.calls == 0


async def test_rerank_text_capsule_feeds_compact_documents() -> None:
    """``rerank_text="capsule"`` feeds name+intent+problem+first two use cases to the reranker."""
    seen_documents: list[list[str]] = []

    class _CapturingReranker:
        async def rerank(self, query: str, documents: Any, top_n: int) -> list[tuple[int, float]]:
            seen_documents.append(list(documents))
            return [(index, float(len(documents) - index)) for index in range(len(documents))]

    records = [
        _record(
            "timeout",
            "Bound the waiting time of a remote call",
            "Remote calls may hang indefinitely",
            "Networks",
            ("bound every remote call", "fail fast", "never leaked to callers"),
        ),
        _record(
            "circuit-breaker",
            "Stop calling a failing dependency",
            "Repeated failures waste latency",
            "Resilience",
            ("open the circuit", "half-open probe", "cooldown bookkeeping"),
        ),
    ]

    for mode in ("full", "capsule"):
        seen_documents.clear()
        index = await RetrievalIndex.build(
            _catalog(records),
            OfflineHashEmbedder(),
            _CapturingReranker(),  # type: ignore[arg-type]
            _retrieval_config(),
            5,
            rerank_text=mode,  # type: ignore[arg-type]
        )
        await index.retrieve(statement="Remote calls may hang", constraints=(), scope="task-lifetime")
        assert len(seen_documents) == 1
        documents = seen_documents[0]
        assert len(documents) == 2
        capsule = documents[0]
        if mode == "capsule":
            assert "timeout" in capsule and "Remote calls may hang indefinitely" in capsule
            assert "bound every remote call" in capsule and "fail fast" in capsule
            assert "never leaked to callers" not in capsule, "capsule carries only two use cases"
            assert "Networks" not in capsule, "capsule omits the context section"
        else:
            assert "Networks" in documents[0], "full document text carries the context section"
            assert "never leaked to callers" in documents[0]


# --------------------------------------------------------------------------------------
# (f) TEIReranker maps HTTP failures to the structured retrieval error
# --------------------------------------------------------------------------------------


class _FakeResponse:
    """httpx.Response stand-in with a controllable status and body."""

    def __init__(self, status_code: int, body: Any = None, malformed: bool = False, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self._malformed = malformed
        self.text = text

    def json(self) -> Any:
        if self._malformed:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._body


def _patch_post(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
    def _fake_post(client: httpx.Client, url: str, **kwargs: Any) -> _FakeResponse:  # noqa: ARG001
        return response

    monkeypatch.setattr(httpx.Client, "post", _fake_post)


async def test_tei_reranker_maps_http_500_to_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = TEIReranker(
        RerankerConfig(rerank_top_n=3, config=RerankerInnerConfig(base_url="http://tei.test", timeout=5.0))
    )
    _patch_post(monkeypatch, _FakeResponse(500, text="model load failed"))
    with pytest.raises(DesignPatternError) as excinfo:
        await reranker.rerank("query", ["doc one", "doc two"], 2)
    assert excinfo.value.code == ERROR_RETRIEVAL_UNAVAILABLE
    assert "HTTP 500" in excinfo.value.detail


async def test_tei_reranker_maps_non_list_body_to_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = TEIReranker(
        RerankerConfig(rerank_top_n=3, config=RerankerInnerConfig(base_url="http://tei.test", timeout=5.0))
    )
    _patch_post(monkeypatch, _FakeResponse(200, body={"detail": "unexpected"}))
    with pytest.raises(DesignPatternError) as excinfo:
        await reranker.rerank("query", ["doc one", "doc two"], 2)
    assert excinfo.value.code == ERROR_RETRIEVAL_UNAVAILABLE


async def test_tei_reranker_maps_malformed_json_to_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = TEIReranker(
        RerankerConfig(rerank_top_n=3, config=RerankerInnerConfig(base_url="http://tei.test", timeout=5.0))
    )
    _patch_post(monkeypatch, _FakeResponse(200, malformed=True))
    with pytest.raises(DesignPatternError) as excinfo:
        await reranker.rerank("query", ["doc one", "doc two"], 2)
    assert excinfo.value.code == ERROR_RETRIEVAL_UNAVAILABLE


async def test_tei_reranker_batches_reindexes_sorts_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_batch_size is respected, batch-local indexes are re-based globally, output sorted + capped."""
    reranker = TEIReranker(
        RerankerConfig(
            rerank_top_n=4,
            config=RerankerInnerConfig(base_url="http://tei.test", timeout=5.0, max_batch_size=2),
        )
    )
    seen_batch_sizes: list[int] = []

    def _fake_post(client: httpx.Client, url: str, **kwargs: Any) -> _FakeResponse:  # noqa: ARG001
        texts = kwargs["json"]["texts"]
        seen_batch_sizes.append(len(texts))
        # Batch-local echo scoring: index i within the batch scores 1/(i+1).
        return _FakeResponse(200, body=[{"index": i, "score": 1.0 / (i + 1)} for i in range(len(texts))])

    monkeypatch.setattr(httpx.Client, "post", _fake_post)
    documents = ["doc a", "doc b", "doc c", "doc d", "doc e"]
    pairs = await reranker.rerank("query", documents, 4)
    assert seen_batch_sizes == [2, 2, 1]
    # Global re-indexing (indexes 2 and 4 exist) + score-desc sort + top_n=4 cap;
    # equal scores keep document order (stable sort).
    assert pairs == [(0, 1.0), (2, 1.0), (4, 1.0), (1, 0.5)]


# --------------------------------------------------------------------------------------
# (g) embedder/reranker factories fail fast on empty base_url
# --------------------------------------------------------------------------------------


def test_build_embedder_rejects_empty_base_url_and_builds_otherwise() -> None:
    with pytest.raises(DesignPatternError) as excinfo:
        build_embedder(EmbedderConfig(config=EmbedderInnerConfig(base_url="", model="openai//test-model")))
    assert excinfo.value.code == ERROR_RETRIEVAL_UNAVAILABLE
    built = build_embedder(EmbedderConfig(config=EmbedderInnerConfig(base_url="http://tei.test")))
    assert isinstance(built, TEIEmbedder)


def test_build_reranker_rejects_empty_base_url() -> None:
    with pytest.raises(DesignPatternError) as excinfo:
        build_reranker(RerankerConfig(config=RerankerInnerConfig(base_url="")))
    assert excinfo.value.code == ERROR_RETRIEVAL_UNAVAILABLE
    built = build_reranker(RerankerConfig(config=RerankerInnerConfig(base_url="http://tei.test")))
    assert isinstance(built, TEIReranker)


# --------------------------------------------------------------------------------------
# Offline embedder contract + build idempotency (resource hygiene)
# --------------------------------------------------------------------------------------


async def test_offline_embedder_is_deterministic_and_l2_normalized() -> None:
    embedder = OfflineHashEmbedder()
    texts = ["singleton global instance", "strategy interchangeable algorithms"]
    vectors = await embedder.embed_documents(texts)
    again = await embedder.embed_documents(texts)
    assert vectors == again
    for vector in vectors:
        assert len(vector) == 512
        assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-5)
    query = await embedder.embed_query("singleton global instance")
    assert query == vectors[0]


async def test_embed_queries_matches_n_single_queries() -> None:
    """``embed_queries`` is the exact batched form of N × ``embed_query``."""
    embedder = OfflineHashEmbedder()
    texts = ["singleton global instance", "strategy interchangeable algorithms", "remote call deadline"]
    batched = await embedder.embed_queries(texts)
    single = [await embedder.embed_query(text) for text in texts]
    assert batched == single


class _CountingEmbedder(OfflineHashEmbedder):
    """Offline embedder recording each query path and every embedded document text."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_query_calls = 0
        self.embed_queries_calls = 0
        self.embedded_texts: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return await super().embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls += 1
        return await super().embed_query(text)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_queries_calls += 1
        return await super().embed_queries(texts)


async def test_retrieve_many_matches_sequential_retrieve_with_one_embed_call() -> None:
    """N parts get identical hits from one batched embed round trip (§6, L1)."""
    embedder = _CountingEmbedder()
    index = await RetrievalIndex.build(_catalog(_records()), embedder, LexicalReranker(), _retrieval_config(), 3)
    queries = [
        PartQuery(statement=_SINGLETON_QUERY, constraints=(), scope="process-lifetime"),
        PartQuery(statement="Remote calls may hang", constraints=(), scope="task-lifetime"),
        PartQuery(statement="Interchangeable algorithms selected at runtime", constraints=("hot swap",), scope=""),
    ]
    batched = await index.retrieve_many(queries)
    assert embedder.embed_queries_calls == 1, "N parts must share one embed round trip"
    assert embedder.embed_query_calls == 0
    sequential = [
        await index.retrieve(statement=query.statement, constraints=query.constraints, scope=query.scope)
        for query in queries
    ]
    assert [[(hit.name, hit.fused_rank, hit.rerank_rank, hit.retrieval_score) for hit in hits] for hits in batched] == [
        [(hit.name, hit.fused_rank, hit.rerank_rank, hit.retrieval_score) for hit in hits] for hits in sequential
    ]


async def test_retrieve_many_bounds_concurrent_part_tails() -> None:
    """The fuse → rerank → blend tail never exceeds ``max_concurrent_parts`` inflight (L2)."""
    inflight = 0
    peak = 0
    lock = asyncio.Lock()

    class _TrackingReranker(LexicalReranker):
        async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
            nonlocal inflight, peak
            async with lock:
                inflight += 1
                peak = max(peak, inflight)
            try:
                await asyncio.sleep(0.01)  # suspend so concurrent tails can actually overlap
                return await super().rerank(query, documents, top_n)
            finally:
                async with lock:
                    inflight -= 1

    for cap in (1, 2, 3):
        index = await RetrievalIndex.build(
            _catalog(_records()),
            OfflineHashEmbedder(),
            _TrackingReranker(),
            _retrieval_config(),
            3,
        )
        index._part_semaphore = asyncio.Semaphore(cap)
        queries = [PartQuery(statement=record.intent, constraints=(), scope="") for record in _records()]
        hits = await index.retrieve_many(queries)
        assert len(hits) == len(queries)
        assert peak <= cap, f"cap {cap}: observed {peak} concurrent rerank legs"
        peak = 0


async def test_build_twice_is_independent_and_reproducible() -> None:
    index_one = await _build_multi()
    index_two = await _build_multi()
    hits_one = await index_one.retrieve(statement=_SINGLETON_QUERY, constraints=(), scope="process-lifetime")
    hits_two = await index_two.retrieve(statement=_SINGLETON_QUERY, constraints=(), scope="process-lifetime")
    assert [(hit.name, hit.fused_rank, hit.rerank_rank) for hit in hits_one] == [
        (hit.name, hit.fused_rank, hit.rerank_rank) for hit in hits_two
    ]


# --------------------------------------------------------------------------------------
# Dense-cache integration: cache-hit, fallback path, cross-catalog miss (§2.2)
# --------------------------------------------------------------------------------------


async def test_retrieval_falls_back_to_embedded_when_no_cache_configured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    index = await _build_multi()
    assert index._dense_cache is None
    assert index._dense_identity is None
    with caplog.at_level("INFO", logger="src.patterns.retriever"):
        await index._build_dense(tuple(index._doc_texts))
    sources = [getattr(record, "source", None) for record in caplog.records if record.name == "src.patterns.retriever"]
    assert "embedded" in sources


async def test_retrieval_ignores_cache_when_identity_is_absent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Cache configured but no declared identity → the pair is incomplete, so the
    dense leg embeds directly (a lone ``dense_cache`` may not drive resolution)."""
    config = DenseCacheConfig(path=str(tmp_path), mode="require", write=False)
    with caplog.at_level("INFO", logger="src.patterns.retriever"):
        # mode="require" would raise on a missing manifest if the cache were used;
        # the absent identity keeps the resolver out of the picture entirely.
        index = await _build_multi(dense_cache=config, identity=None)
    assert index._faiss_index is not None
    sources = [
        record.source  # type: ignore[attr-defined]
        for record in caplog.records
        if record.name == "src.patterns.retriever" and record.getMessage() == "Dense vectors resolved"
    ]
    assert sources == ["embedded"]
    assert not (tmp_path / "manifest.json").exists()


async def test_dense_cache_hit_on_second_build(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    identity = EmbedderIdentity(model="offline-hash")

    with caplog.at_level("INFO", logger="src.patterns.retriever"):
        index_first = await _build_multi(dense_cache=config, identity=identity)
    # First build had no manifest → embedded (warm-up write-back).
    sources_first = [
        getattr(record, "source", None) for record in caplog.records if record.name == "src.patterns.retriever"
    ]
    assert "embedded" in sources_first

    with caplog.at_level("INFO", logger="src.patterns.retriever"):
        index_second = await _build_multi(dense_cache=config, identity=identity)
    sources_second = [
        getattr(record, "source", None) for record in caplog.records if record.name == "src.patterns.retriever"
    ]
    assert "cache-hit" in sources_second

    # Same hits for the same query, with scores within 1e-9.
    hits_first = await index_first.retrieve(statement=_SINGLETON_QUERY, constraints=(), scope="process-lifetime")
    hits_second = await index_second.retrieve(statement=_SINGLETON_QUERY, constraints=(), scope="process-lifetime")
    assert [hit.name for hit in hits_first] == [hit.name for hit in hits_second]
    for got, expected in zip(hits_first, hits_second, strict=True):
        assert abs(got.retrieval_score - expected.retrieval_score) < 1e-9


async def test_warm_start_embeds_only_the_probe_text(tmp_path: Path) -> None:
    """A restart with a valid snapshot re-embeds NOTHING but the probe text.

    This is the cache's entire purpose: the corpus vectors come off disk, so the
    embedder sees exactly one text (the probe, which confirms the served model
    still matches the snapshot) and no pattern document.
    """
    config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    identity = EmbedderIdentity(model="offline-hash")

    cold = _CountingEmbedder()
    index_cold = await _build_multi(dense_cache=config, identity=identity, embedder=cold)
    corpus_texts = set(index_cold._doc_texts)
    # Cold start: the whole corpus plus one probe embed for the write-back manifest.
    assert corpus_texts <= set(cold.embedded_texts)
    assert [text for text in cold.embedded_texts if text == PROBE_TEXT] == [PROBE_TEXT]

    warm = _CountingEmbedder()
    index_warm = await _build_multi(dense_cache=config, identity=identity, embedder=warm)
    assert warm.embedded_texts == [PROBE_TEXT]  # zero corpus embed calls

    hits = await index_warm.retrieve(statement=_SINGLETON_QUERY, constraints=(), scope="process-lifetime")
    assert "singleton" in [hit.name for hit in hits]


async def test_dense_cache_misses_for_a_different_catalog(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    identity = EmbedderIdentity(model="offline-hash")

    # Seed the cache with the standard hand-made catalog.
    await _build_multi(dense_cache=config, identity=identity)

    # Build a different catalog with a single-candidate singleton record. The
    # record count + corpus_sha256 differ → cache-miss → rebuild → still serves hits.
    other_records = [
        _record(
            "only",
            "Unique vocab word: zymurgy quokka",
            "Unique vocab word: zymurgy quokka",
            "Unique vocab word: zymurgy quokka",
            ("unique",),
        ),
    ]
    with caplog.at_level("INFO", logger="src.patterns.retriever"):
        index_other = await _build_multi(
            records=other_records,
            rerank_top_n=1,
            dense_cache=config,
            identity=identity,
        )
    sources = [getattr(record, "source", None) for record in caplog.records if record.name == "src.patterns.retriever"]
    assert "embedded" in sources

    hits = await index_other.retrieve(statement="Unique vocab word: zymurgy quokka", constraints=(), scope="")
    assert any(hit.name == "only" for hit in hits)


# --------------------------------------------------------------------------------------
# Per-leg stage-1 recall caps (§6 step 2): the cross-encoder pool is bounded
# --------------------------------------------------------------------------------------


def _wide_records(count: int = 30) -> list[JoinedPattern]:
    return [
        _record(
            f"pattern-{index}",
            f"Intent of pattern {index} covering requirement {index}",
            f"Problem {index} that pattern {index} resolves",
            f"Context {index} for pattern {index}",
            (f"use case {index}",),
        )
        for index in range(count)
    ]


async def test_per_leg_top_k_bounds_the_rerank_pool() -> None:
    """dense_top_k/bm25_top_k cap the cross-encoder pool (the 2026-09-24 outage fix)."""
    reranker = _RecordingReranker()
    index = await RetrievalIndex.build(
        _catalog(_wide_records()),
        OfflineHashEmbedder(),
        reranker,
        _retrieval_config(dense_top_k=1, bm25_top_k=1),
        5,
    )
    await index.retrieve(statement="pattern 7", constraints=(), scope="task-lifetime")
    assert len(reranker.documents) <= 2


async def test_zero_top_k_restores_full_corpus_recall() -> None:
    """0 is the lossless escape hatch: every catalog document reaches the cross-encoder.

    Observed per leg: with one leg at 0 and the other capped to a single document,
    only the uncapped leg can fill the pool, so each assertion pins its leg's
    ``0``-semantics rather than the union of two working legs.
    """
    records = _wide_records()

    # BM25 at 0 (full corpus), dense capped to one document.
    bm25_uncapped = _RecordingReranker()
    index = await RetrievalIndex.build(
        _catalog(records),
        OfflineHashEmbedder(),
        bm25_uncapped,
        _retrieval_config(dense_top_k=1, bm25_top_k=0),
        5,
    )
    await index.retrieve(statement="pattern 7", constraints=(), scope="task-lifetime")
    assert len(bm25_uncapped.documents) == len(records)

    # Dense at 0 (full corpus), BM25 capped to one document.
    dense_uncapped = _RecordingReranker()
    index = await RetrievalIndex.build(
        _catalog(records),
        OfflineHashEmbedder(),
        dense_uncapped,
        _retrieval_config(dense_top_k=0, bm25_top_k=1),
        5,
    )
    await index.retrieve(statement="pattern 7", constraints=(), scope="task-lifetime")
    assert len(dense_uncapped.documents) == len(records)
