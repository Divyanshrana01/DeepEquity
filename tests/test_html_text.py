from __future__ import annotations

from deepequity.ingestion.html_text import html_to_text


def test_tags_are_stripped_but_words_survive() -> None:
    html = "<html><body><p>Revenue <b>increased</b> by 5%.</p></body></html>"

    assert html_to_text(html) == "Revenue increased by 5%."


def test_script_and_style_content_is_dropped() -> None:
    # Without this, javascript and css end up embedded as if they were filing text,
    # wasting chunk space and polluting search results.
    html = """
    <html><head><style>.x{color:red}</style></head>
    <body><script>var secret = 1;</script><p>Real content.</p></body></html>
    """

    text = html_to_text(html)

    assert "Real content." in text
    assert "color:red" not in text
    assert "var secret" not in text


def test_adjacent_cells_do_not_run_together() -> None:
    # Filings are mostly tables. Without a separator between block elements you get
    # "RevenueCost of sales" glued into one nonsense token.
    html = "<table><tr><td>Revenue</td><td>Cost of sales</td></tr></table>"

    text = html_to_text(html)

    assert "RevenueCost" not in text
    assert "Revenue" in text
    assert "Cost of sales" in text


def test_whitespace_is_collapsed() -> None:
    html = "<p>Lots     of\n\n\n\n\nspace   here</p>"

    text = html_to_text(html)

    assert "     " not in text
    assert "\n\n\n" not in text


def test_non_breaking_spaces_are_normalised() -> None:
    # Filings are full of &nbsp;, which is not a normal space and would otherwise stay
    # glued to words all the way into the embeddings.
    html = "<p>Net&nbsp;&nbsp;income</p>"

    assert html_to_text(html) == "Net income"


def test_empty_or_junk_html_gives_empty_string() -> None:
    assert html_to_text("") == ""
    assert html_to_text("<html><head><title>x</title></head><body></body></html>") == ""
