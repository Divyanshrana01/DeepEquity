from __future__ import annotations

import re

from selectolax.parser import HTMLParser

#tags whose contents are never readable text, so we throw the whole node away rather
#than letting javascript or css end up in the chunks we embed
_JUNK_TAGS = ("script", "style", "noscript", "head", "meta", "link")

#tags that end a block of text. we push a real newline in after these before parsing,
#so paragraphs and table rows stay separated. everything else (bold, italic, spans,
#font tags, which filings use constantly) is inline and must NOT break the line, or
#"Revenue <b>increased</b> 5%" would come out as three separate fragments and both the
#sentence splitting and the embeddings would suffer for it.
_BLOCK_END = re.compile(
    r"</(?:p|div|tr|table|thead|tbody|h[1-6]|li|ul|ol|section|article|blockquote)>"
    r"|<br\s*/?>",
    re.IGNORECASE,
)

#three or more newlines collapse to two, keeps paragraph breaks without huge gaps
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
#runs of spaces and tabs become one space. filings are full of layout whitespace
_EXCESS_SPACES = re.compile(r"[ \t\xa0]+")
#trailing/leading spaces on each line, left over once tags are gone
_LINE_EDGES = re.compile(r"[ \t]*\n[ \t]*")


#turns a raw sec filing into plain readable text. filings are html tables and inline
#styling wrapped around the actual words, and embedding that markup would waste most of
#every chunk on tags. selectolax is used over beautifulsoup because it's a rust parser
#and these documents run to megabytes.
def html_to_text(html: str) -> str:
    #mark where blocks end before parsing, so paragraph structure survives into the text
    html = _BLOCK_END.sub(lambda match: match.group(0) + "\n", html)

    tree = HTMLParser(html)

    for tag in _JUNK_TAGS:
        for node in tree.css(tag):
            node.decompose()

    body = tree.body if tree.body is not None else tree.root
    if body is None:
        return ""

    #a space between nodes, not a newline: inline tags shouldn't break a sentence, and
    #it still keeps neighbouring table cells from gluing into "RevenueCost of sales"
    text = body.text(separator=" ")
    return _tidy(text)


#squeezes out the mountain of whitespace html leaves behind, so chunks are mostly words
def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _EXCESS_SPACES.sub(" ", text)
    text = _LINE_EDGES.sub("\n", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()
