"""Split documents into chunks of whole paragraphs, keeping each chunk's offsets.

A chunk is always an exact slice text[start:end], so quotes can be checked against the
source. Each 8-K item heading ("Item 8.01 ...") starts a new chunk; paragraphs longer
than max_chars are cut after sentence ends, and hard-cut only as a last resort.
"""

import re

import pandas as pd

PARAGRAPH = re.compile(r"\S(?:[^\n]|\n(?![ \t]*\n))*")  # up to the next blank line
SENTENCE_END = re.compile(r"[.!?](?=\s)")
SECTION_START = re.compile(r"item\s+\d+\.\d+", re.IGNORECASE)

CHUNK_COLUMNS = ["chunk_id", "start", "end", "text"]


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _pieces(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Sentence spans of each paragraph, hard-cut where a sentence exceeds max_chars."""
    spans = []
    for paragraph in PARAGRAPH.finditer(text):
        start, end = paragraph.span()
        for cut in [m.end() for m in SENTENCE_END.finditer(text, start, end)] + [end]:
            while cut - start > max_chars:
                spans.append(_strip(text, start, start + max_chars))
                start += max_chars
            spans.append(_strip(text, start, cut))
            start = cut
    return [(s, e) for s, e in spans if s < e]


def chunk_text(text: str, max_chars: int) -> list[tuple[int, int]]:
    """(start, end) of each chunk: consecutive pieces packed up to max_chars."""
    chunks: list[tuple[int, int]] = []
    for start, end in _pieces(text, max_chars):
        if chunks and end - chunks[-1][0] <= max_chars and not SECTION_START.match(text, start):
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
    return chunks


def chunk_documents(docs: pd.DataFrame, max_chars: int) -> pd.DataFrame:
    """One row per chunk: the document's columns except text, plus CHUNK_COLUMNS."""
    rows = []
    for doc in docs.to_dict(orient="records"):
        text = doc.pop("text")
        for i, (start, end) in enumerate(chunk_text(text, max_chars)):
            rows.append(
                {
                    **doc,
                    "chunk_id": f"{doc['doc_ref']}#{i}",
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
    columns = [c for c in docs.columns if c != "text"] + CHUNK_COLUMNS
    return pd.DataFrame(rows, columns=columns)
