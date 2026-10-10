"""Tests for the bounded retry loop in :meth:`TEIEmbedder.embed_documents` (§2.2).

A TEI sidecar that is still warming or restarting must not fail the warm-up;
``embed_query`` stays fail-fast because request latency must not absorb an outage.
The unit battery uses an ``OfflineHashEmbedder``-shaped fake (``get_text_embedding_batch``
returns ``list[list[float]]``) plus a monkeypatched ``TEIEmbedder._embedding`` so we
exercise the retry arithmetic and the WARNING log surface directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import EmbedderConfig, EmbedderInnerConfig
from src.errors import ERROR_RETRIEVAL_UNAVAILABLE, DesignPatternError
from src.patterns.embedder import TEIEmbedder, build_embedder


def _make_embedder(attempts: int = 3, backoff: float = 0.01) -> TEIEmbedder:
    """Construct a TEIEmbedder with the llama-index inner client replaced by a mock.

    The retry path only touches ``self._embedding.get_text_embedding_batch``; stubbing
    that attribute on the instance sidesteps the real LiteLLM construction entirely.
    """
    config = EmbedderConfig(
        config=EmbedderInnerConfig(
            base_url="http://stub:0",
            document_embed_attempts=attempts,
            document_embed_backoff_seconds=backoff,
        )
    )
    instance = TEIEmbedder.__new__(TEIEmbedder)
    inner = config.config
    instance._query_instruction = inner.query_instruction
    instance._text_instruction = inner.text_instruction
    instance._doc_attempts = inner.document_embed_attempts
    instance._doc_backoff = inner.document_embed_backoff_seconds
    instance._config = config
    instance._embedding = MagicMock()
    return instance


async def test_embed_documents_retries_then_succeeds(caplog: pytest.LogCaptureFixture) -> None:
    embedder = _make_embedder(attempts=3, backoff=0.001)
    fake = MagicMock(
        side_effect=[
            RuntimeError("connection error"),
            RuntimeError("connection error"),
            [[1.0, 0.0]],
        ]
    )
    embedder._embedding.get_text_embedding_batch = fake  # type: ignore[method-assign]

    with caplog.at_level("WARNING", logger="src.patterns.embedder"):
        vectors = await embedder.embed_documents(["x"])

    assert fake.call_count == 3
    assert len(vectors) == 1
    # Result is unit-normed by l2_normalize_rows.
    assert abs(vectors[0][0] - 1.0) < 1e-6
    assert abs(vectors[0][1]) < 1e-6
    messages = [record.getMessage() for record in caplog.records]
    assert any("Embedder unavailable (attempt 1/3" in msg for msg in messages)
    assert any("Embedder unavailable (attempt 2/3" in msg for msg in messages)
    assert all("retrying in" in msg for msg in messages)


async def test_embed_documents_exhausts_attempts_and_propagates() -> None:
    embedder = _make_embedder(attempts=3, backoff=0.001)
    fake = MagicMock(side_effect=RuntimeError("down"))
    embedder._embedding.get_text_embedding_batch = fake  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="down"):
        await embedder.embed_documents(["x"])
    assert fake.call_count == 3


async def test_embed_query_is_not_retried() -> None:
    """``embed_query`` stays fail-fast — a transient sidecar blip must not delay a request."""
    embedder = _make_embedder(attempts=5, backoff=0.001)
    fake = MagicMock(side_effect=RuntimeError("down"))
    embedder._embedding.get_query_embedding = fake  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="down"):
        await embedder.embed_query("hello")
    assert fake.call_count == 1


def test_build_embedder_rejects_empty_base_url_with_retry_fields() -> None:
    config = EmbedderConfig(
        config=EmbedderInnerConfig(
            base_url="",
            document_embed_attempts=5,
            document_embed_backoff_seconds=0.5,
        )
    )
    with pytest.raises(DesignPatternError) as excinfo:
        build_embedder(config)
    assert excinfo.value.code == ERROR_RETRIEVAL_UNAVAILABLE
