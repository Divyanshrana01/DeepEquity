from __future__ import annotations

import re
from dataclasses import dataclass, field

from deepequity.core.config import get_settings

#split points in preference order: paragraph break first, then sentence end, then any
#whitespace. we'd rather cut where a human would than in the middle of a word
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChildChunk:
    text: str
    #position within the document, used to keep ordering stable and as part of the
    #unique key so re-ingesting the same document can't create duplicates
    index: int


@dataclass
class ParentChunk:
    text: str
    index: int
    children: list[ChildChunk] = field(default_factory=list)


#splits a filing into big parent chunks, each cut into smaller overlapping children.
#the children get embedded and searched (small = a hit points at the exact passage),
#the parent is what we hand the llm (big = enough context to actually use the hit).
#filings are the textbook case for this: a sentence about a risk means little without
#the paragraph around it.
def chunk_document(text: str) -> list[ParentChunk]:
    settings = get_settings()
    return build_chunks(
        text,
        parent_size=settings.parent_chunk_chars,
        child_size=settings.child_chunk_chars,
        child_overlap=settings.child_chunk_overlap_chars,
    )


#the real implementation, with sizes passed in so tests can use small numbers instead
#of the production ones
def build_chunks(
    text: str, parent_size: int, child_size: int, child_overlap: int
) -> list[ParentChunk]:
    if child_overlap >= child_size:
        raise ValueError("child_overlap must be smaller than child_size, or we never advance")

    text = text.strip()
    if not text:
        return []

    parents: list[ParentChunk] = []
    child_counter = 0

    for parent_index, parent_text in enumerate(_split_into_parents(text, parent_size)):
        parent = ParentChunk(text=parent_text, index=parent_index)
        for child_text in _split_into_children(parent_text, child_size, child_overlap):
            parent.children.append(ChildChunk(text=child_text, index=child_counter))
            child_counter += 1
        #a parent with no children would be an orphan nothing can ever retrieve, so drop it
        if parent.children:
            parents.append(parent)

    return parents


#groups paragraphs into parent-sized blocks. we accumulate whole paragraphs and start a
#new parent when adding the next one would overshoot, so parents break on real
#boundaries instead of arbitrary character counts
def _split_into_parents(text: str, parent_size: int) -> list[str]:
    paragraphs = [p.strip() for p in _PARAGRAPH.split(text) if p.strip()]
    if not paragraphs:
        return []

    parents: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        #a single paragraph bigger than a parent gets hard-split, otherwise one huge
        #block of text (common in filings, they love enormous paragraphs) would become
        #one giant parent and defeat the point
        if len(paragraph) > parent_size:
            if current:
                parents.append("\n\n".join(current))
                current, current_len = [], 0
            parents.extend(_hard_split(paragraph, parent_size))
            continue

        if current_len + len(paragraph) > parent_size and current:
            parents.append("\n\n".join(current))
            current, current_len = [], 0

        current.append(paragraph)
        current_len += len(paragraph) + 2  # +2 for the join

    if current:
        parents.append("\n\n".join(current))

    return parents


#slices a parent into overlapping children. the overlap matters: without it a sentence
#that straddles a boundary is split across two chunks and neither one reads as a
#complete thought, so neither matches the query well
def _split_into_children(text: str, child_size: int, overlap: int) -> list[str]:
    if len(text) <= child_size:
        return [text]

    children: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + child_size, len(text))

        #if we're mid-document, back the cut up to the nearest sentence end so chunks
        #don't start halfway through a word
        if end < len(text):
            end = _best_break(text, start, end)

        chunk = text[start:end].strip()
        if chunk:
            children.append(chunk)

        if end >= len(text):
            break
        #step forward by the chunk minus the overlap, so consecutive children share
        #their edges. max() guards against a tiny chunk leaving us stuck in place
        start = max(end - overlap, start + 1)

    return children


#finds a sentence boundary near the end of the window, falling back to a space, then to
#the hard cut if the text has neither (a long unbroken string of digits, say)
def _best_break(text: str, start: int, end: int) -> int:
    window = text[start:end]

    matches = list(_SENTENCE_END.finditer(window))
    if matches:
        candidate = start + matches[-1].end()
        #only take it if it isn't so far back that we'd produce a stub chunk
        if candidate > start + (end - start) // 2:
            return candidate

    space = window.rfind(" ")
    if space > (end - start) // 2:
        return start + space + 1

    return end


#chops an oversized paragraph into parent-sized pieces on whitespace where possible
def _hard_split(text: str, size: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            end = _best_break(text, start, end)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = max(end, start + 1)
    return pieces
