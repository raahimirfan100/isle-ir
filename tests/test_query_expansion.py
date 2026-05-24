from __future__ import annotations

from retrieval.query_expansion import expand_query


def test_expand_query_adds_related_terms_for_known_words() -> None:
    expanded = expand_query("bm25 retrieval")

    assert expanded
    assert expanded != "bm25 retrieval"
    assert "bm25" in expanded
    assert "retrieval" in expanded


def test_expand_query_falls_back_for_unknown_or_empty_input() -> None:
    assert expand_query("qwertyuiop") == "qwertyuiop"
    assert expand_query("") == "query"