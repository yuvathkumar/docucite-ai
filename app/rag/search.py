"""Hybrid search: FTS5 keyword ranking and vector similarity, fused into one ranking."""

import logging
import math
import re
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db import FTS_TABLE_NAME
from app.llm.base import Embedder, EmbeddingError
from app.models import Chunk, ChunkEmbedding, Document
from app.rag.embeddings import unpack_vector

logger = logging.getLogger(__name__)

RRF_K = 60
CANDIDATE_LIMIT = 20
MAX_QUERY_TERMS = 32
TERM_PATTERN = re.compile(r"\w+")


@dataclass(frozen=True)
class SearchHit:
    """A passage returned by search, with everything needed to cite it.

    Attributes:
        chunk_id: Id of the chunk.
        document_id: Id of the document it belongs to.
        filename: File name of that document.
        page_number: 1-based page of the passage.
        text: The passage text.
        score: Fused relevance score (higher is better).
    """

    chunk_id: int
    document_id: int
    filename: str
    page_number: int
    text: str
    score: float


def build_fts_query(question: str) -> str | None:
    """Turn free text into a safe FTS5 query that matches any of its words.

    Every word is wrapped in double quotes, so characters that mean something
    in FTS5 syntax (``AND``, ``*``, ``:``, quotes) in a user's question cannot
    change the query or make it fail.

    Args:
        question: The user's question.

    Returns:
        A query like ``"vacation" OR "days"``, or None if the question has no words.

    Example:
        >>> build_fts_query('How many "vacation" days? AND *')
        '"how" OR "many" OR "vacation" OR "days" OR "and"'
    """
    terms = TERM_PATTERN.findall(question.lower())[:MAX_QUERY_TERMS]
    return " OR ".join(f'"{term}"' for term in terms) if terms else None


def keyword_search(session: Session, question: str, limit: int = CANDIDATE_LIMIT) -> list[int]:
    """Rank chunks by BM25 keyword relevance using the FTS5 index.

    Args:
        session: Database session.
        question: The user's question.
        limit: Maximum number of chunk ids to return.

    Returns:
        Chunk ids, best match first.
    """
    query = build_fts_query(question)
    if query is None:
        return []
    rows = session.execute(
        text(
            f"SELECT chunk_id FROM {FTS_TABLE_NAME} WHERE {FTS_TABLE_NAME} MATCH :query "
            f"ORDER BY bm25({FTS_TABLE_NAME}) LIMIT :limit"
        ),
        {"query": query, "limit": limit},
    )
    return [int(row[0]) for row in rows]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Measure how similar the direction of two vectors is.

    Args:
        first: One vector.
        second: Another vector of the same length.

    Returns:
        A value from -1 to 1; higher means more similar meaning. Zero if either
        vector has no length.

    Example:
        >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
        1.0
        >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
    """
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    norm = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    return dot / norm if norm else 0.0


def vector_search(
    session: Session, query_vector: list[float], model: str, limit: int = CANDIDATE_LIMIT
) -> list[int]:
    """Rank chunks by cosine similarity between their vector and the question's.

    Only vectors made by the same model as the question's are compared, because
    vectors from different models live in unrelated spaces.

    Args:
        session: Database session.
        query_vector: Embedding of the question.
        model: Name of the model that produced ``query_vector``.
        limit: Maximum number of chunk ids to return.

    Returns:
        Chunk ids, most similar first.
    """
    rows = session.execute(
        select(ChunkEmbedding.chunk_id, ChunkEmbedding.vector).where(ChunkEmbedding.model == model)
    ).all()
    scored: list[tuple[float, int]] = []
    for chunk_id, vector_bytes in rows:
        vector = unpack_vector(vector_bytes)
        if len(vector) == len(query_vector):
            scored.append((cosine_similarity(query_vector, vector), chunk_id))
    scored.sort(reverse=True)
    return [chunk_id for _, chunk_id in scored[:limit]]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Combine several ranked lists into one.

    Each item scores ``1 / (k + rank)`` in every list it appears in, and the
    scores are added. Using ranks instead of raw scores avoids having to compare
    BM25 values with cosine values, which are on unrelated scales. An item found
    by both searches beats one found by only one.

    Args:
        rankings: Lists of ids, each best first.
        k: Damping constant; larger values flatten the difference between ranks.

    Returns:
        ``(id, score)`` pairs, best first.

    Example:
        >>> [i for i, _ in reciprocal_rank_fusion([[1, 2, 3], [3, 1]])]
        [1, 3, 2]
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(
    session: Session, embedder: Embedder, question: str, top_k: int
) -> list[SearchHit]:
    """Find the passages that best answer a question.

    Runs keyword and vector search and fuses them. If the embedding provider is
    unavailable, keyword search alone still answers, so a provider outage
    degrades quality instead of breaking the app.

    Args:
        session: Database session.
        embedder: Provider used to embed the question.
        question: The user's question.
        top_k: Number of passages to return.

    Returns:
        Up to ``top_k`` hits, best first.
    """
    keyword_ids = keyword_search(session, question)
    vector_ids: list[int] = []
    try:
        query_vector = embedder.embed_query(question)
        vector_ids = vector_search(session, query_vector, embedder.model_name)
    except EmbeddingError as error:
        logger.warning("Vector search skipped, using keywords only: %s", error)

    fused = reciprocal_rank_fusion([keyword_ids, vector_ids])[:top_k]
    if not fused:
        return []
    rows = session.execute(
        select(Chunk, Document)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_([chunk_id for chunk_id, _ in fused]))
    ).all()
    by_id = {chunk.id: (chunk, document) for chunk, document in rows}
    return [
        SearchHit(
            chunk_id=chunk_id,
            document_id=by_id[chunk_id][1].id,
            filename=by_id[chunk_id][1].filename,
            page_number=by_id[chunk_id][0].page_number,
            text=by_id[chunk_id][0].text,
            score=score,
        )
        for chunk_id, score in fused
        if chunk_id in by_id
    ]
