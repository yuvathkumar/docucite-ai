"""Tests for hybrid search: query building, ranking, fusion and fallback."""

from sqlalchemy.orm import Session

from app.llm import UnavailableEmbedder
from app.llm.fake import FakeEmbedder
from app.rag.embeddings import embed_missing_chunks
from app.rag.ingest import ingest_pdf
from app.rag.search import (
    build_fts_query,
    cosine_similarity,
    hybrid_search,
    keyword_search,
    reciprocal_rank_fusion,
)
from tests.pdf_factory import make_pdf

PAGES = [
    "Vacation policy: employees accrue two vacation days per month of service.",
    "Expense policy: submit expense reports before the fifth of each month.",
    "Security policy: laptops must use full disk encryption at all times.",
]


def load_documents(session: Session, embedded: bool = True) -> int:
    """Ingest the sample pages as one document, optionally embedding them."""
    document = ingest_pdf(
        session, "handbook.pdf", make_pdf(PAGES), max_bytes=10**6, max_chars=300, overlap=30
    )
    if embedded:
        embed_missing_chunks(session, FakeEmbedder(), document.id)
    return document.id


def test_fts_query_quotes_every_word_so_syntax_cannot_leak() -> None:
    """Neutralise FTS5 operators and punctuation typed by a user."""
    assert build_fts_query('vacation AND "days" *') == '"vacation" OR "and" OR "days"'
    assert build_fts_query("???") is None
    assert build_fts_query("") is None


def test_keyword_search_finds_matching_chunk(db_session: Session) -> None:
    """Return the chunk containing the query word."""
    load_documents(db_session)

    ids = keyword_search(db_session, "expense reports")

    assert len(ids) == 1


def test_keyword_search_survives_hostile_input(db_session: Session) -> None:
    """Never raise on characters that are special in FTS5 syntax."""
    load_documents(db_session)

    assert keyword_search(db_session, 'NEAR( " OR ) * : ^ -') == []


def test_cosine_similarity_basics() -> None:
    """Score identical vectors 1, orthogonal 0 and zero vectors 0."""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_rank_fusion_rewards_agreement() -> None:
    """Put an item found by both searches above items found by one."""
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]])

    assert [item for item, _ in fused] == [1, 3, 2]


def test_rank_fusion_of_nothing_is_empty() -> None:
    """Return no results when both searches found nothing."""
    assert reciprocal_rank_fusion([[], []]) == []


def test_hybrid_search_returns_citable_hits(db_session: Session) -> None:
    """Give each hit its filename, page and text, best passage first."""
    load_documents(db_session)

    hits = hybrid_search(db_session, FakeEmbedder(), "how many vacation days per month", 3)

    assert hits[0].filename == "handbook.pdf"
    assert hits[0].page_number == 1
    assert "vacation" in hits[0].text
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_hybrid_search_respects_top_k(db_session: Session) -> None:
    """Return at most ``top_k`` hits."""
    load_documents(db_session)

    assert len(hybrid_search(db_session, FakeEmbedder(), "policy", 2)) == 2


def test_hybrid_search_falls_back_to_keywords_when_embedding_fails(db_session: Session) -> None:
    """Keep answering from keywords when the embedding provider is down."""
    load_documents(db_session)

    hits = hybrid_search(db_session, UnavailableEmbedder("down"), "full disk encryption", 3)

    assert [hit.page_number for hit in hits] == [3]


def test_hybrid_search_with_no_documents_returns_nothing(db_session: Session) -> None:
    """Return an empty list on an empty database."""
    assert hybrid_search(db_session, FakeEmbedder(), "anything", 5) == []
