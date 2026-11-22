"""Tests for the text chunker."""

import pytest

from app.rag.chunking import split_into_chunks

SAMPLE = (
    "Employees accrue vacation days monthly. Unused days carry over for one year. "
    "Managers approve requests within five working days. Sick leave is tracked separately."
)
LONG_SAMPLE = " ".join([SAMPLE] * 5)


def test_empty_and_whitespace_text_gives_no_chunks() -> None:
    """Return an empty list when there is nothing to split."""
    assert split_into_chunks("") == []
    assert split_into_chunks("  \n\t ") == []


def test_short_text_is_a_single_chunk() -> None:
    """Keep text that fits in one chunk as one chunk."""
    assert split_into_chunks("Short text.", max_chars=100, overlap=10) == ["Short text."]


def test_documented_example() -> None:
    """Match the examples in the function docstring."""
    assert split_into_chunks("abc", max_chars=2, overlap=1) == ["ab", "bc"]
    assert split_into_chunks("one two three four", max_chars=10, overlap=4) == [
        "one two",
        "two three",
        "three four",
    ]


def test_whitespace_and_line_breaks_are_collapsed() -> None:
    """Turn PDF line breaks and repeated spaces into single spaces."""
    assert split_into_chunks("a\n\nb   c\td", max_chars=100, overlap=0) == ["a b c d"]


def test_no_chunk_exceeds_max_chars() -> None:
    """Respect the size limit for every chunk."""
    chunks = split_into_chunks(LONG_SAMPLE, max_chars=80, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_words_are_never_cut_in_half() -> None:
    """Start and end every chunk on a whole word when the text has spaces."""
    words = set(SAMPLE.split())

    for chunk in split_into_chunks(LONG_SAMPLE, max_chars=80, overlap=20):
        assert set(chunk.split()) <= words


def test_neighbouring_chunks_overlap() -> None:
    """Repeat some text from the end of a chunk at the start of the next one."""
    chunks = split_into_chunks(" ".join([SAMPLE] * 3), max_chars=80, overlap=30)

    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert previous.split()[-1] in following.split()


def test_all_words_are_covered_in_order() -> None:
    """Lose no text: every word of the input appears in some chunk, in order."""
    chunks = split_into_chunks(SAMPLE, max_chars=60, overlap=15)
    seen = " ".join(chunks).split()
    position = 0
    for word in SAMPLE.split():
        position = seen.index(word, position)


def test_zero_overlap_chunks_do_not_repeat_text() -> None:
    """Rebuild the original text exactly when overlap is zero."""
    chunks = split_into_chunks(SAMPLE, max_chars=50, overlap=0)

    assert " ".join(chunks) == SAMPLE


def test_text_without_spaces_is_cut_hard_and_terminates() -> None:
    """Cut a single very long word at the limit instead of looping forever."""
    chunks = split_into_chunks("x" * 250, max_chars=100, overlap=20)

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "".join(chunks).count("x") >= 250


@pytest.mark.parametrize(
    ("max_chars", "overlap"),
    [(0, 0), (-5, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_parameters_raise_value_error(max_chars: int, overlap: int) -> None:
    """Reject sizes that would make the chunker unable to advance."""
    with pytest.raises(ValueError, match="max_chars|overlap"):
        split_into_chunks("some text", max_chars=max_chars, overlap=overlap)
