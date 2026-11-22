"""Create embeddings for stored chunks and read them back as vectors."""

import logging
from array import array
from typing import Final

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.llm.base import Embedder, EmbeddingError
from app.models import Chunk, ChunkEmbedding

logger = logging.getLogger(__name__)

# 32-bit float: half the size of Python's float, plenty of precision. ``Final`` lets the type
# checker see the literal "f" and so know the array holds floats.
VECTOR_TYPECODE: Final = "f"
# Small groups keep each API request under free-tier token limits and bound lost work.
EMBED_GROUP_SIZE = 25


def pack_vector(values: list[float]) -> bytes:
    """Pack a vector into bytes for storage in a BLOB column.

    Args:
        values: The vector.

    Returns:
        The values as 32-bit floats (native byte order).

    Example:
        >>> unpack_vector(pack_vector([0.5, -2.0]))
        [0.5, -2.0]
    """
    return array(VECTOR_TYPECODE, values).tobytes()


def unpack_vector(data: bytes) -> list[float]:
    """Read a vector stored with :func:`pack_vector`.

    Args:
        data: The stored bytes.

    Returns:
        The vector as a list of floats.
    """
    values = array(VECTOR_TYPECODE)
    values.frombytes(data)
    return values.tolist()


def embed_missing_chunks(session: Session, embedder: Embedder, document_id: int) -> int:
    """Embed every chunk of a document that does not have a vector yet.

    Safe to call again: chunks that already have a vector are skipped, so a
    failed run (provider down, rate limit) can simply be retried. Progress is
    saved after every group of ``EMBED_GROUP_SIZE`` chunks, so a failure midway
    keeps everything embedded before it.

    Args:
        session: Database session.
        embedder: Provider that creates the vectors.
        document_id: The document whose chunks to embed.

    Returns:
        How many chunks were embedded in this call.

    Raises:
        EmbeddingError: If the provider fails or returns the wrong number of vectors.
    """
    chunks = session.scalars(
        select(Chunk)
        .outerjoin(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .where(Chunk.document_id == document_id, ChunkEmbedding.chunk_id.is_(None))
        .order_by(Chunk.id)
    ).all()
    embedded = 0
    for start in range(0, len(chunks), EMBED_GROUP_SIZE):
        group = chunks[start : start + EMBED_GROUP_SIZE]
        vectors = embedder.embed_documents([chunk.text for chunk in group])
        if len(vectors) != len(group):
            raise EmbeddingError("The provider returned a different number of vectors than chunks.")
        for chunk, vector in zip(group, vectors, strict=True):
            session.add(
                ChunkEmbedding(
                    chunk_id=chunk.id,
                    model=embedder.model_name,
                    dimensions=len(vector),
                    vector=pack_vector(vector),
                )
            )
        # Commit per group: if the provider rate-limits us midway, the groups already
        # done stay saved and a retry continues from where this one stopped.
        session.commit()
        embedded += len(group)
    return embedded


def count_embedded_chunks(session: Session) -> dict[int, int]:
    """Count embedded chunks per document.

    Args:
        session: Database session.

    Returns:
        A mapping from document id to its number of embedded chunks. Documents
        with none are absent.
    """
    rows = session.execute(
        select(Chunk.document_id, func.count(ChunkEmbedding.chunk_id))
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .group_by(Chunk.document_id)
    ).all()
    return {document_id: count for document_id, count in rows}
