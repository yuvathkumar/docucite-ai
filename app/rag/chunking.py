"""Split page text into overlapping chunks for keyword and vector search."""

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 200
WORD_SEPARATOR = " "


def split_into_chunks(
    text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Split a long text into overlapping chunks for embedding.

    Overlap keeps a sentence that crosses a chunk boundary readable in
    at least one chunk, which improves retrieval quality. Chunks end at a
    word boundary when one exists, so words are not cut in half. All
    whitespace (including PDF line breaks) is collapsed to single spaces first.

    Args:
        text: The full text of one document page or section.
        max_chars: Maximum characters per chunk.
        overlap: Characters repeated between neighbouring chunks.

    Returns:
        The chunks in their original order. Returns an empty list for empty text.

    Raises:
        ValueError: If ``max_chars`` is not positive, ``overlap`` is negative,
            or ``overlap`` is not smaller than ``max_chars``.

    Example:
        >>> split_into_chunks("abc", max_chars=2, overlap=1)
        ['ab', 'bc']
        >>> split_into_chunks("one two three four", max_chars=10, overlap=4)
        ['one two', 'two three', 'three four']
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be at least 0 and smaller than max_chars")

    normalized = WORD_SEPARATOR.join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            end = _last_word_boundary(normalized, start, end, overlap)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = _next_start(normalized, start, end, overlap)
    return chunks


def _last_word_boundary(text: str, start: int, end: int, overlap: int) -> int:
    """Return where to cut a chunk so it ends between words.

    The cut must lie beyond ``start + overlap``; otherwise the next chunk would
    start at or before this one and the loop would never advance.

    Args:
        text: The normalised text.
        start: Index where the current chunk starts.
        end: Hard upper limit for the cut (``start + max_chars``).
        overlap: Overlap between chunks, in characters.

    Returns:
        The index of the last separator inside the allowed range, or ``end``
        if the range has none (a very long word is cut hard).
    """
    boundary = text.rfind(WORD_SEPARATOR, start + overlap + 1, end)
    return boundary if boundary != -1 else end


def _next_start(text: str, previous_start: int, end: int, overlap: int) -> int:
    """Return where the next chunk starts, about ``overlap`` characters before ``end``.

    The start is moved to the beginning of a word so chunks do not begin in the
    middle of one: forward if there is a space inside the overlap window,
    otherwise back to the start of the word that straddles it (a slightly larger
    overlap is better than a chunk that starts with half a word).

    Args:
        text: The normalised text.
        previous_start: Where the chunk that just ended started.
        end: Where the previous chunk ended.
        overlap: Overlap between chunks, in characters.

    Returns:
        The start index of the next chunk (always after ``previous_start``).
    """
    if overlap == 0:
        return end
    start = end - overlap
    forward = text.find(WORD_SEPARATOR, start, end)
    if forward != -1:
        return forward + 1
    backward = text.rfind(WORD_SEPARATOR, previous_start, start)
    return backward + 1 if backward != -1 else start
