from rag.chunking import chunk_text

TEXT = (
    "Item 8.01 Other Events.\n\n"
    "First paragraph about the dividend.\n\n"
    "Second paragraph, still short.\n\n"
    "Item 9.01 Financial Statements and Exhibits.\n\n"
    "Exhibit 99.1 press release."
)


def test_chunks_are_exact_slices_within_the_limit():
    for max_chars in (20, 60, 200):
        for start, end in chunk_text(TEXT, max_chars):
            assert 0 <= start < end <= len(TEXT)
            assert end - start <= max_chars
            assert TEXT[start:end].strip() == TEXT[start:end]


def test_short_paragraphs_are_packed_and_items_start_a_new_chunk():
    chunks = [TEXT[s:e] for s, e in chunk_text(TEXT, 200)]
    assert len(chunks) == 2
    assert chunks[0].startswith("Item 8.01") and "Second paragraph" in chunks[0]
    assert chunks[1].startswith("Item 9.01") and chunks[1].endswith("press release.")


def test_long_paragraph_is_split_at_sentence_ends():
    text = "One sentence here. Another sentence there. A third one ends it."
    chunks = [text[s:e] for s, e in chunk_text(text, 45)]
    assert chunks == ["One sentence here. Another sentence there.", "A third one ends it."]


def test_text_without_any_break_is_hard_cut():
    text = "x" * 25
    assert chunk_text(text, 10) == [(0, 10), (10, 20), (20, 25)]


def test_blank_text_has_no_chunks():
    assert chunk_text("  \n\n ", 100) == []
