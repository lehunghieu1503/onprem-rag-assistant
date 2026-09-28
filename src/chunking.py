"""Split one cleaned page into token-bounded chunks, preferring paragraph/sentence boundaries."""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable

# A tokenizer here is anything that maps text -> list of (char_start, char_end) per token.
Offsets = Callable[[str], list]

_UNIT_END = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    start: int
    end: int
    token_count: int


def hf_offsets(tokenizer) -> Offsets:
    def offsets(text: str) -> list:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        return [tuple(o) for o in enc["offset_mapping"]]

    return offsets


def _units(text: str) -> list:
    """Character spans of sentences/paragraphs covering all non-space text."""
    spans, pos = [], 0
    for m in _UNIT_END.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def chunk_page(text: str, offsets: Offsets, max_tokens: int, overlap: int) -> list:
    if max_tokens <= 0 or not 0 <= overlap < max_tokens:
        raise ValueError("need max_tokens > 0 and 0 <= overlap < max_tokens")
    if not text.strip():
        return []
    token_spans = offsets(text)
    if not token_spans:
        return []
    starts = [s for s, _ in token_spans]

    def tok_index(char_pos: int) -> int:
        return bisect_left(starts, char_pos)

    # Pieces are token ranges [a, b): whole units, or slices of a unit longer than max_tokens.
    pieces = []
    for s, e in _units(text):
        a, b = tok_index(s), tok_index(e)
        while b - a > max_tokens:
            pieces.append((a, a + max_tokens))
            a += max_tokens
        if b > a:
            pieces.append((a, b))
    if not pieces:
        return []

    chunks, i = [], 0
    while i < len(pieces):
        a = pieces[i][0]
        j = i
        while j + 1 < len(pieces) and pieces[j + 1][1] - a <= max_tokens:
            j += 1
        b = pieces[j][1]
        start, end = token_spans[a][0], token_spans[b - 1][1]
        chunks.append(Chunk(len(chunks), text[start:end], start, end, b - a))
        if j + 1 >= len(pieces):
            break
        # Step back over whole pieces worth at most `overlap` tokens; always advance at least one piece.
        nxt = j + 1
        while nxt - 1 > i and b - pieces[nxt - 1][0] <= overlap:
            nxt -= 1
        i = nxt
    return chunks
