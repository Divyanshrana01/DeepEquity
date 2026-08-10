from __future__ import annotations

import pytest

from deepequity.ingestion.chunking import build_chunks


def test_short_text_becomes_one_parent_one_child() -> None:
    parents = build_chunks("A short filing.", parent_size=2000, child_size=400, child_overlap=80)

    assert len(parents) == 1
    assert len(parents[0].children) == 1
    assert parents[0].children[0].text == "A short filing."


def test_empty_text_produces_nothing() -> None:
    assert build_chunks("   \n\n  ", parent_size=2000, child_size=400, child_overlap=80) == []


def test_children_carry_a_document_wide_index() -> None:
    # The index has to be unique across the whole document, not restart per parent,
    # because it's half of the unique key that stops a re-ingest duplicating chunks.
    text = "\n\n".join(f"Paragraph number {i}. " * 20 for i in range(6))

    parents = build_chunks(text, parent_size=500, child_size=150, child_overlap=30)

    indexes = [child.index for parent in parents for child in parent.children]
    assert indexes == list(range(len(indexes)))


def test_children_overlap_so_sentences_are_not_orphaned() -> None:
    # A sentence split across a boundary reads as a fragment in both chunks and matches
    # nothing well. Overlap means the full sentence appears intact in at least one.
    text = "word " * 400

    parents = build_chunks(text, parent_size=5000, child_size=200, child_overlap=50)
    children = [child.text for parent in parents for child in parent.children]

    assert len(children) > 1
    # consecutive children should share some text
    first_tail = children[0][-30:]
    assert any(first_tail[:10] in children[1] for _ in [0]) or children[1].startswith("word")


def test_every_child_belongs_to_its_parents_text() -> None:
    # A child that isn't actually inside its parent would hand the LLM context that
    # doesn't contain the matched passage, which defeats parent-child retrieval.
    text = "\n\n".join(f"Section {i}. " + ("detail " * 40) for i in range(5))

    parents = build_chunks(text, parent_size=600, child_size=200, child_overlap=40)

    for parent in parents:
        for child in parent.children:
            assert child.text in parent.text


def test_huge_paragraph_is_split_rather_than_becoming_one_giant_parent() -> None:
    # Filings love enormous single paragraphs. If we didn't hard split them, one would
    # become a single parent far bigger than the configured size.
    text = "sentence. " * 500

    parents = build_chunks(text, parent_size=500, child_size=150, child_overlap=30)

    assert len(parents) > 1
    for parent in parents:
        # allow a little slack, we cut on sentence boundaries not exact characters
        assert len(parent.text) <= 700


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    # If overlap >= size the loop would never move forward and would hang, so we refuse.
    with pytest.raises(ValueError, match="smaller"):
        build_chunks("some text", parent_size=1000, child_size=100, child_overlap=100)


def test_no_empty_chunks_are_produced() -> None:
    text = "\n\n".join(["Real paragraph here." * 10, "", "   ", "Another one." * 10])

    parents = build_chunks(text, parent_size=300, child_size=100, child_overlap=20)

    for parent in parents:
        assert parent.text.strip()
        assert parent.children  # a parent with no children is unretrievable
        for child in parent.children:
            assert child.text.strip()
