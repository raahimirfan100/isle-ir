"""Unit tests for BibTeX/CSV export."""

from __future__ import annotations

from app.export import to_bibtex, to_csv


def test_bibtex_format():
    papers = [
        {
            "arxiv_id": "2301.07041",
            "title": "A Test Paper",
            "authors": ["Alice Smith", "Bob Jones"],
            "year": 2023,
            "categories": ["cs.LG", "cs.AI"],
            "cited_by_count": 5,
        }
    ]
    result = to_bibtex(papers)
    assert result.startswith("@article{2301.07041,")
    assert "title = {A Test Paper}" in result
    assert "author = {Alice Smith and Bob Jones}" in result
    assert "year = {2023}" in result
    assert "url = {https://arxiv.org/abs/2301.07041}" in result


def test_csv_format():
    papers = [
        {
            "arxiv_id": "2301.07041",
            "title": "A Test Paper",
            "authors": ["Alice Smith"],
            "year": 2023,
            "categories": ["cs.LG"],
            "cited_by_count": 5,
        }
    ]
    result = to_csv(papers)
    assert result.startswith(b"arxiv_id,title,authors")
    assert b"2301.07041" in result
    assert b"A Test Paper" in result


def test_empty_results():
    assert to_bibtex([]) == ""
    assert to_csv([]) == b""
