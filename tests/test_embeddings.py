"""Tests for the fake embedder, vector packing and embedding of stored chunks."""

import math
from collections.abc import Sequence

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.llm import UnavailableEmbedder, create_embedder
from app.llm.base import EmbeddingError
from app.llm.fake import FakeEmbedder
from app.llm.gemini import GeminiEmbedder
from app.models import Chunk, ChunkEmbedding
from app.rag.embeddings import (
    EMBED_GROUP_SIZE,
    count_embedded_chunks,
    embed_missing_chunks,
    pack_vector,
    unpack_vector,
)
from app.rag.ingest import ingest_pdf
from tests.pdf_factory import make_pdf


def cosine(first: list[float], second: list[float]) -> float:
    """Return the cosine similarity of two vectors."""
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    norms = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    return dot / norms


class WrongCountEmbedder:
    """A broken provider that returns one vector too few."""

    model_name = "broken"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return fewer vectors than texts."""
        return [[1.0]] * (len(texts) - 1)


def ingest_sample(session: Session, pages: list[str]) -> int:
    """Ingest a generated PDF and return the new document id."""
    document = ingest_pdf(
        session, "doc.pdf", make_pdf(pages), max_bytes=10**6, max_chars=200, overlap=20
    )
    return document.id


def test_pack_and_unpack_round_trip() -> None:
    """Store and read back vector values exactly (they are exact in 32-bit floats)."""
    assert unpack_vector(pack_vector([0.5, -2.0, 0.0])) == [0.5, -2.0, 0.0]


def test_fake_embedder_is_deterministic_and_normalised() -> None:
    """Give identical text identical vectors of length 1."""
    embedder = FakeEmbedder()

    first, second = embedder.embed_documents(["vacation policy", "vacation policy"])

    assert first == second
    assert len(first) == embedder.dimensions
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


def test_fake_embedder_ranks_shared_words_higher() -> None:
    """Make texts with common words more similar than unrelated texts."""
    query, related, unrelated = FakeEmbedder().embed_documents(
        ["vacation days policy", "annual vacation days for staff", "database backup schedule"]
    )

    assert cosine(query, related) > cosine(query, unrelated)


def test_embed_missing_chunks_stores_one_vector_per_chunk(db_session: Session) -> None:
    """Create a vector row for every chunk, recording model and size."""
    document_id = ingest_sample(db_session, ["Vacation policy.", "Expense policy."])

    embedded = embed_missing_chunks(db_session, FakeEmbedder(), document_id)

    assert embedded == 2
    rows = db_session.scalars(select(ChunkEmbedding)).all()
    assert len(rows) == 2
    assert {row.model for row in rows} == {"fake-hash-embedding"}
    assert all(len(unpack_vector(row.vector)) == row.dimensions for row in rows)
    assert count_embedded_chunks(db_session) == {document_id: 2}


def test_embed_missing_chunks_skips_chunks_that_already_have_vectors(db_session: Session) -> None:
    """Make a second run do nothing, so retrying is always safe."""
    document_id = ingest_sample(db_session, ["Only page."])
    embed_missing_chunks(db_session, FakeEmbedder(), document_id)

    assert embed_missing_chunks(db_session, FakeEmbedder(), document_id) == 0
    assert len(db_session.scalars(select(ChunkEmbedding)).all()) == 1


def test_failed_provider_stores_nothing(db_session: Session) -> None:
    """Leave chunks unembedded (and retryable) when the provider fails."""
    document_id = ingest_sample(db_session, ["Some text."])

    with pytest.raises(EmbeddingError):
        embed_missing_chunks(db_session, UnavailableEmbedder("down"), document_id)

    assert db_session.scalars(select(ChunkEmbedding)).all() == []
    assert db_session.scalars(select(Chunk)).all() != []


class FailsAfterFirstGroup:
    """A provider that rate-limits every call after the first one."""

    model_name = "flaky"

    def __init__(self) -> None:
        """Start with no calls made."""
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Succeed once, then fail like a rate limit."""
        self.calls += 1
        if self.calls > 1:
            raise EmbeddingError("rate limited")
        return [[1.0, 0.0]] * len(texts)


def test_progress_is_kept_when_the_provider_fails_midway(db_session: Session) -> None:
    """Keep finished groups, then let a later retry finish only the rest."""
    long_page = " ".join(f"word{number}" for number in range(2000))
    document = ingest_pdf(
        db_session, "big.pdf", make_pdf([long_page]), max_bytes=10**6, max_chars=100, overlap=10
    )
    assert document.chunk_count > EMBED_GROUP_SIZE

    with pytest.raises(EmbeddingError, match="rate limited"):
        embed_missing_chunks(db_session, FailsAfterFirstGroup(), document.id)

    saved = count_embedded_chunks(db_session)[document.id]
    assert saved == EMBED_GROUP_SIZE
    remaining = embed_missing_chunks(db_session, FakeEmbedder(), document.id)
    assert remaining == document.chunk_count - EMBED_GROUP_SIZE
    assert count_embedded_chunks(db_session)[document.id] == document.chunk_count


def test_wrong_vector_count_is_rejected(db_session: Session) -> None:
    """Refuse a provider answer that does not match the chunks, storing nothing."""
    document_id = ingest_sample(db_session, ["Page one.", "Page two."])

    with pytest.raises(EmbeddingError, match="different number"):
        embed_missing_chunks(db_session, WrongCountEmbedder(), document_id)

    assert db_session.scalars(select(ChunkEmbedding)).all() == []


def test_factory_builds_gemini_embedder_from_settings() -> None:
    """Pick Gemini when LLM_PROVIDER is gemini, using the configured model."""
    settings = Settings(llm_provider="gemini", embed_model="some-model", _env_file=None)

    embedder = create_embedder(settings)

    assert isinstance(embedder, GeminiEmbedder)
    assert embedder.model_name == "some-model"


def test_factory_does_not_crash_for_unsupported_provider() -> None:
    """Start fine with ollama selected; only embedding itself fails."""
    embedder = create_embedder(Settings(llm_provider="ollama", _env_file=None))

    with pytest.raises(EmbeddingError, match="gemini"):
        embedder.embed_documents(["x"])
