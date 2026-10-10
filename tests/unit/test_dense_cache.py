"""Tests for the precomputed dense-embedding cache (:mod:`src.patterns.dense_cache`, §2.2).

The cache builder is exercised through :func:`resolve_dense_vectors` against a tiny
in-memory catalog so the suite stays deterministic and offline. Each test pins one
observable: cache-hit status, cache-miss reason, store behavior, mode-gate, or
data-defect handling.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from src.config import DenseCacheConfig, EmbedderConfig, EmbedderInnerConfig
from src.errors import ERROR_RETRIEVAL_UNAVAILABLE, DesignPatternError
from src.patterns.dense_cache import (
    BUILDER_VERSION,
    CACHE_SCHEMA_VERSION,
    MANIFEST_NAME,
    PROBE_TEXT,
    VECTORS_NAME,
    DenseCacheManifest,
    build_manifest,
    corpus_digest,
    instruction_digest,
    load_manifest,
    load_vectors,
    main,
    names_digest,
    probe_cosine,
    resolve_dense_vectors,
    store_cache,
)
from src.patterns.embedder import EmbedderIdentity, OfflineHashEmbedder


def _identity(
    *,
    model: str = "openai/test-model",
    instruction: str = "",
    model_repo: str = "",
    onnx_variant: str = "",
) -> EmbedderIdentity:
    return EmbedderIdentity(
        model=model,
        text_instruction=instruction,
        model_repo=model_repo,
        onnx_variant=onnx_variant,
    )


def _names_doc_texts() -> tuple[list[str], list[str]]:
    names = ["alpha", "beta", "gamma"]
    doc_texts = [
        "alpha description text for retrieval",
        "beta description text for retrieval",
        "gamma description text for retrieval",
    ]
    return names, doc_texts


# --------------------------------------------------------------------------------------
# Pure-function digests + cosine
# --------------------------------------------------------------------------------------


def test_digests_are_byte_pinned() -> None:
    """The three digests are a cross-version contract: a builder from an older
    release must agree with a runtime from a newer one, so the exact hex values
    (join separators included) are pinned here, not merely self-consistent."""
    names = ["alpha", "beta", "gamma"]
    texts = [f"{name} description text for retrieval" for name in names]
    assert names_digest(names) == "f3220283d05d1ff2ae350cfe9e0e367cb5aef46e10efb203c8a53c678e2218c8"
    assert corpus_digest(names, texts) == (
        "7fa9fdaf2a9e246cbdd988aa12566084be2dc4da9b1f9b894de91ce486cc7163"
    )
    assert instruction_digest("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert instruction_digest("doc: ") == (
        "31125b93db9d507557bfa5f3ba8d50a887fe4fde3142c093c5893fac54311172"
    )


def test_names_digest_matches_joined_order() -> None:
    assert names_digest(["a", "b"]) == names_digest(["a", "b"])
    assert names_digest(["a", "b"]) != names_digest(["b", "a"])


def test_corpus_digest_changes_with_text() -> None:
    names, texts = _names_doc_texts()
    digest_one = corpus_digest(names, texts)
    assert digest_one == corpus_digest(names, list(texts))
    altered = list(texts)
    altered[0] = altered[0] + "!"
    assert corpus_digest(names, altered) != digest_one


def test_instruction_digest_matches_text_instruction() -> None:
    assert instruction_digest("") != instruction_digest("doc: ")


def test_probe_cosine_unit_and_zero() -> None:
    assert probe_cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert probe_cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert probe_cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    # Mismatched lengths must NOT crash the loader path; treat as a no-match (0.0).
    assert probe_cosine([1.0, 0.0], [1.0]) == 0.0


# --------------------------------------------------------------------------------------
# Manifest / vector on-disk I/O
# --------------------------------------------------------------------------------------


def _seed_cache(tmp_path: Path, *, embedder: OfflineHashEmbedder) -> tuple[list[str], list[str]]:
    """Pre-populate a cache directory with one OfflineHashEmbedder-shaped run."""
    names, doc_texts = _names_doc_texts()
    vectors = asyncio.run(embedder.embed_documents(doc_texts))
    normalized = [
        (np.asarray(v, dtype=np.float32) / max(np.linalg.norm(v), 1e-12)).tolist()
        for v in vectors
    ]
    probe = asyncio.run(embedder.embed_documents([PROBE_TEXT]))[0]
    manifest = build_manifest(
        names=names,
        doc_texts=doc_texts,
        identity=_identity(model="openai/test-model"),
        dimension=len(normalized[0]),
        probe_vector=probe,
        tei_info={},
    )
    assert store_cache(tmp_path, manifest, normalized) is None
    return names, doc_texts


def test_manifest_round_trip(tmp_path: Path) -> None:
    _seed_cache(tmp_path, embedder=OfflineHashEmbedder())
    manifest, reason = load_manifest(tmp_path)
    assert reason == "ok"
    assert manifest is not None
    assert manifest.schema_version == CACHE_SCHEMA_VERSION
    assert manifest.builder_version == BUILDER_VERSION
    assert manifest.record_count == 3
    assert manifest.dimension == 512
    assert manifest.probe_text == PROBE_TEXT


def test_vectors_round_trip(tmp_path: Path) -> None:
    _seed_cache(tmp_path, embedder=OfflineHashEmbedder())
    manifest, _ = load_manifest(tmp_path)
    assert manifest is not None
    vectors, reason = load_vectors(tmp_path, manifest)
    assert reason == "ok"
    assert vectors is not None
    assert len(vectors) == 3
    for row in vectors:
        assert math.isclose(math.sqrt(sum(value * value for value in row)), 1.0, rel_tol=1e-5)


def test_manifest_schema_version_mismatch(tmp_path: Path) -> None:
    _seed_cache(tmp_path, embedder=OfflineHashEmbedder())
    path = tmp_path / MANIFEST_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest, reason = load_manifest(tmp_path)
    assert manifest is None
    assert reason == "schema-version"


def test_manifest_unknown_key_rejected(tmp_path: Path) -> None:
    _seed_cache(tmp_path, embedder=OfflineHashEmbedder())
    path = tmp_path / MANIFEST_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["rogue"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest, reason = load_manifest(tmp_path)
    assert manifest is None
    assert reason == "manifest-unreadable"


def test_vectors_unreadable_when_truncated(tmp_path: Path) -> None:
    _seed_cache(tmp_path, embedder=OfflineHashEmbedder())
    path = tmp_path / VECTORS_NAME
    path.write_bytes(b"not a numpy array")
    manifest, _ = load_manifest(tmp_path)
    assert manifest is not None
    vectors, reason = load_vectors(tmp_path, manifest)
    assert vectors is None
    assert reason == "vectors-unreadable"


def test_vectors_shape_mismatch(tmp_path: Path) -> None:
    _seed_cache(tmp_path, embedder=OfflineHashEmbedder())
    manifest, _ = load_manifest(tmp_path)
    assert manifest is not None
    np.save(
        tmp_path / VECTORS_NAME,
        np.zeros((manifest.record_count + 1, manifest.dimension), dtype=np.float32),
        allow_pickle=False,
    )
    vectors, reason = load_vectors(tmp_path, manifest)
    assert vectors is None
    assert reason == "vectors-shape"


def test_store_cache_path_is_a_regular_file(tmp_path: Path) -> None:
    """When ``cache_dir`` is not a directory the store path must still report a clean reason."""
    not_a_dir = tmp_path / "obstacle"
    not_a_dir.write_text("not a directory", encoding="utf-8")
    manifest = DenseCacheManifest(
        schema_version=CACHE_SCHEMA_VERSION,
        created_at="1970-01-01T00:00:00Z",
        record_count=2,
        dimension=4,
        names_sha256="x" * 64,
        corpus_sha256="y" * 64,
        embedder_model="openai/test",
        text_instruction_sha256=instruction_digest(""),
        probe_text=PROBE_TEXT,
        probe_vector=[0.5, 0.5, 0.5, 0.5],
        builder_version=BUILDER_VERSION,
    )
    reason = store_cache(not_a_dir, manifest, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    assert reason is not None
    # We still need the loader to succeed; resolve_dense_vectors re-embeds without crashing.
    config = DenseCacheConfig(path=str(not_a_dir))
    vectors, status = asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=["a", "b"],
            doc_texts=["alpha text", "beta text"],
            identity=_identity(),
        )
    )
    assert status == "embedded"
    assert len(vectors) == 2


# --------------------------------------------------------------------------------------
# resolve_dense_vectors — mode + miss reasons
# --------------------------------------------------------------------------------------


def test_resolve_off_skips_cache_io(tmp_path: Path) -> None:
    config = DenseCacheConfig(path=str(tmp_path), mode="off")
    names, doc_texts = _names_doc_texts()
    vectors, status = asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(),
        )
    )
    assert status == "disabled"
    assert len(vectors) == 3
    assert not (tmp_path / MANIFEST_NAME).exists()
    assert not (tmp_path / VECTORS_NAME).exists()


def test_resolve_require_without_manifest_raises(tmp_path: Path) -> None:
    config = DenseCacheConfig(path=str(tmp_path), mode="require")
    names, doc_texts = _names_doc_texts()
    with pytest.raises(DesignPatternError) as excinfo:
        asyncio.run(
            resolve_dense_vectors(
                config=config,
                embedder=OfflineHashEmbedder(),
                names=names,
                doc_texts=doc_texts,
                identity=_identity(),
            )
        )
    assert excinfo.value.code == ERROR_RETRIEVAL_UNAVAILABLE
    assert "manifest-absent" in excinfo.value.detail


def test_resolve_auto_writes_after_first_embedded_run(tmp_path: Path) -> None:
    config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    names, doc_texts = _names_doc_texts()

    # First call: no cache → re-embed + write.
    first_vectors, first_status = asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(model="openai/test-model"),
        )
    )
    assert first_status == "embedded"
    assert (tmp_path / MANIFEST_NAME).exists()
    assert (tmp_path / VECTORS_NAME).exists()

    # Second call: identical catalog and identity → cache-hit with the same vectors.
    second_vectors, second_status = asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(model="openai/test-model"),
        )
    )
    assert second_status == "cache-hit"
    for got, expected in zip(second_vectors, first_vectors, strict=True):
        for lhs, rhs in zip(got, expected, strict=True):
            assert abs(lhs - rhs) < 1e-6


def test_resolve_misses_on_corpus_text_change(tmp_path: Path) -> None:
    config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    names, doc_texts = _names_doc_texts()
    asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(),
        )
    )
    # Cache now exists. Mutate one document text → corpus_sha256 miss, rewrite.
    mutated = list(doc_texts)
    mutated[1] = mutated[1] + " changed"
    _, status = asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=mutated,
            identity=_identity(),
        )
    )
    assert status == "embedded"
    manifest, _ = load_manifest(tmp_path)
    assert manifest is not None
    assert manifest.corpus_sha256 == corpus_digest(names, mutated)


def test_resolve_reports_the_record_count_reason_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A *shrunk* catalog must report the record-count reason, not fall through to
    the name digest: ``!=`` and ``<`` disagree only when the cached count is the
    larger one, so this direction is the one that pins the check's operator."""
    config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=False)
    names, doc_texts = _names_doc_texts()
    seed = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    asyncio.run(
        resolve_dense_vectors(
            config=seed,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(),
        )
    )
    shrunk_names = [names[0], names[1]]
    shrunk_texts = [doc_texts[0], doc_texts[1]]
    with caplog.at_level("INFO", logger="src.patterns.dense_cache"):
        _, status = asyncio.run(
            resolve_dense_vectors(
                config=config,
                embedder=OfflineHashEmbedder(),
                names=shrunk_names,
                doc_texts=shrunk_texts,
                identity=_identity(),
            )
        )
    assert status == "embedded"
    reasons = [
        record.reason  # type: ignore[attr-defined]
        for record in caplog.records
        if record.name == "src.patterns.dense_cache"
    ]
    assert "cache-miss:record-count-3-vs-2" in reasons


def test_resolve_misses_on_negated_probe_vector(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    names, doc_texts = _names_doc_texts()
    asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(),
        )
    )
    # Negate the probe vector in the manifest → cosine ≈ -1 → cache miss.
    path = tmp_path / MANIFEST_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["probe_vector"] = [-value for value in data["probe_vector"]]
    path.write_text(json.dumps(data), encoding="utf-8")

    config_read = DenseCacheConfig(path=str(tmp_path), mode="auto", write=False)
    with caplog.at_level("INFO", logger="src.patterns.dense_cache"):
        _, status = asyncio.run(
            resolve_dense_vectors(
                config=config_read,
                embedder=OfflineHashEmbedder(),
                names=names,
                doc_texts=doc_texts,
                identity=_identity(),
            )
        )
    assert status == "embedded"
    miss_reasons = [
        record.__dict__.get("reason", "")
        for record in caplog.records
        if record.name == "src.patterns.dense_cache"
    ]
    assert any(reason.startswith("cache-miss:probe-cosine-") for reason in miss_reasons)


def test_resolve_accepts_probe_at_exactly_the_threshold(tmp_path: Path) -> None:
    """The gate is ``cosine < probe_min_cosine`` (strict): a cosine equal to the
    configured minimum is an accept, not a miss."""
    config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    names, doc_texts = _names_doc_texts()
    asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(),
        )
    )
    manifest, _ = load_manifest(tmp_path)
    assert manifest is not None
    # Re-store with probe_min_cosine set exactly to the achievable self-cosine.
    embedder = OfflineHashEmbedder()
    probe = asyncio.run(embedder.embed_documents([manifest.probe_text]))[0]
    exact = probe_cosine(probe, manifest.probe_vector)
    exact_config = DenseCacheConfig(
        path=str(tmp_path), mode="require", write=False, probe_min_cosine=exact
    )
    _, status = asyncio.run(
        resolve_dense_vectors(
            config=exact_config,
            embedder=embedder,
            names=names,
            doc_texts=doc_texts,
            identity=_identity(),
        )
    )
    assert status == "cache-hit"


def test_resolve_require_raises_on_probe_cosine_mismatch(tmp_path: Path) -> None:
    # Seed the cache in auto mode (require would refuse the missing-manifest warm-up).
    seed_config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    names, doc_texts = _names_doc_texts()
    asyncio.run(
        resolve_dense_vectors(
            config=seed_config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(),
        )
    )
    # Tamper with the probe vector to force a cosine mismatch on the next call.
    path = tmp_path / MANIFEST_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["probe_vector"] = [-value for value in data["probe_vector"]]
    path.write_text(json.dumps(data), encoding="utf-8")

    require_config = DenseCacheConfig(path=str(tmp_path), mode="require", write=True)
    with pytest.raises(DesignPatternError) as excinfo:
        asyncio.run(
            resolve_dense_vectors(
                config=require_config,
                embedder=OfflineHashEmbedder(),
                names=names,
                doc_texts=doc_texts,
                identity=_identity(),
            )
        )
    assert excinfo.value.code == ERROR_RETRIEVAL_UNAVAILABLE


def test_resolve_uses_declared_identity_when_provided(tmp_path: Path) -> None:
    config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    names, doc_texts = _names_doc_texts()
    asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(model="openai/test", model_repo="onnx-community/Qwen3-Embedding-0.6B-ONNX", onnx_variant="fp32"),
        )
    )
    manifest, _ = load_manifest(tmp_path)
    assert manifest is not None
    assert manifest.model_repo == "onnx-community/Qwen3-Embedding-0.6B-ONNX"
    assert manifest.onnx_variant == "fp32"


def test_resolve_rejects_model_repo_mismatch(tmp_path: Path) -> None:
    config = DenseCacheConfig(path=str(tmp_path), mode="auto", write=True)
    names, doc_texts = _names_doc_texts()
    asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(model_repo="repo-a", onnx_variant="fp32"),
        )
    )
    _, status = asyncio.run(
        resolve_dense_vectors(
            config=config,
            embedder=OfflineHashEmbedder(),
            names=names,
            doc_texts=doc_texts,
            identity=_identity(model_repo="repo-b", onnx_variant="fp32"),
        )
    )
    assert status == "embedded"


# --------------------------------------------------------------------------------------
# main() entry point — argument parsing, success + failure paths
# --------------------------------------------------------------------------------------


def test_main_build_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main(['build'])`` writes a manifest + vectors when given a stub config.

    The build path uses ``build_embedder`` to construct the embedder; we swap it
    for an offline hash embedder so the test stays network-free. A real run
    invokes TEI through the sidecar and falls under ``make embed-cache``.
    """
    config = EmbedderConfig(
        config=EmbedderInnerConfig(
            base_url="http://stub:0", document_embed_attempts=1, document_embed_backoff_seconds=0.01
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "embedder": {
                    "provider": config.provider,
                    "config": {
                        "base_url": config.config.base_url,
                        "api_key": "",
                        "model": config.config.model,
                        "embed_batch_size": config.config.embed_batch_size,
                        "document_embed_attempts": config.config.document_embed_attempts,
                        "document_embed_backoff_seconds": config.config.document_embed_backoff_seconds,
                    },
                },
                "paths": {
                    "pattern_directory": "pattern",
                    "docs_directory": "docs",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.patterns.dense_cache.build_embedder",
        lambda _cfg: OfflineHashEmbedder(),
    )

    out_dir = tmp_path / "cache"
    start = time.monotonic()
    rc = main(["build", "--out", str(out_dir), "--config", str(config_path)])
    elapsed = time.monotonic() - start

    assert rc == 0
    assert (out_dir / MANIFEST_NAME).exists()
    assert (out_dir / VECTORS_NAME).exists()
    # Offline embedder + tiny retry budget; well under any silent-hang threshold.
    assert elapsed < 30.0


def test_main_build_fails_without_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DESIGN_PATTERN_CONFIG_PATH", raising=False)
    rc = main(["build", "--out", str(tmp_path / "cache"), "--config", str(tmp_path / "missing.json")])
    assert rc == 1
