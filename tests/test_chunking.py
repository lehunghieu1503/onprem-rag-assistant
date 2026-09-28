import unicodedata

import pytest

from src.chunking import chunk_page
from src.ingest import clean_text
from tests.conftest import ws_offsets


def test_empty_page_gives_no_chunks():
    assert chunk_page("   \n ", ws_offsets, 10, 2) == []


def test_chunks_respect_token_limit_and_cover_text():
    text = " ".join(f"Sentence number {i} ends here." for i in range(30))
    chunks = chunk_page(text, ws_offsets, 20, 5)
    assert len(chunks) > 1
    assert all(c.token_count <= 20 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start <= a.end and b.start > a.start  # contiguous or overlapping, always advancing
    assert all(text[c.start:c.end] == c.text for c in chunks)


def test_sentence_longer_than_limit_is_split_by_tokens():
    text = " ".join(f"w{i}" for i in range(95))
    chunks = chunk_page(text, ws_offsets, 30, 5)
    assert [c.token_count for c in chunks] == [30, 30, 30, 5]
    assert "w94" in chunks[-1].text


def test_prefers_sentence_boundaries():
    text = "Alpha beta gamma. Delta epsilon zeta eta. Theta iota."
    chunks = chunk_page(text, ws_offsets, 5, 0)
    assert [c.text for c in chunks] == ["Alpha beta gamma.", "Delta epsilon zeta eta.", "Theta iota."]


def test_overlap_repeats_trailing_sentence():
    text = "One two three. Four five. Six seven eight nine."
    chunks = chunk_page(text, ws_offsets, 5, 2)
    assert chunks[0].text == "One two three. Four five."
    assert chunks[1].text.startswith("Four five.")


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_page("a b c", ws_offsets, 5, 5)


def test_clean_text_nfc_and_line_merge_keep_numbers_and_units():
    raw = unicodedata.normalize("NFD", "Áp suất thử\nbằng 1,5 lần\n\nQCVN 01:2008/BCT  ≥ 10 kg/cm²\r\n")
    out = clean_text(raw)
    assert out == unicodedata.normalize("NFC", "Áp suất thử bằng 1,5 lần\n\nQCVN 01:2008/BCT ≥ 10 kg/cm²")
    assert unicodedata.is_normalized("NFC", out)


def test_clean_text_keeps_hyphenated_codes_across_lines():
    assert clean_text("pump P-\n101A runs") == "pump P-101A runs"
