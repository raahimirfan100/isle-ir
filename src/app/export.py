"""BibTeX and CSV export for search results.

Owner:    Mujtaba Kamran (SSOT §2)
Phase:    Bonus (B-M3)
STATUS:   enhancement

One-click export of the current result set in researcher-friendly formats.
"""

from __future__ import annotations

import pandas as pd


def to_bibtex(results: list[dict]) -> str:
    """Render results as a BibTeX string.

    Each entry: @article{arxiv_id,
      title = {Title Here},
      author = {First Last and First Last},
      year = {2023},
      journal = {arXiv},
      url = {https://arxiv.org/abs/arxiv_id},
    }
    """
    entries: list[str] = []
    for paper in results:
        pid = paper.get("arxiv_id", "unknown")
        title = paper.get("title", "").replace("{", "{{").replace("}", "}}")
        authors = paper.get("authors", [])
        author_str = " and ".join(authors) if authors else "Unknown"
        year = str(paper.get("year", ""))
        categories = ", ".join(paper.get("categories", []))

        entry = (
            f"@article{{{pid},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{author_str}}},\n"
            f"  year = {{{year}}},\n"
            f"  journal = {{arXiv}},\n"
            f"  url = {{https://arxiv.org/abs/{pid}}},\n"
            f"  categories = {{{categories}}}\n"
            f"}}"
        )
        entries.append(entry)

    return "\n\n".join(entries)


def to_csv(results: list[dict]) -> bytes:
    """Render results as UTF-8 CSV bytes (for st.download_button).

    Columns: arxiv_id, title, authors, year, categories, cited_by_count, url.
    """
    if not results:
        return b""

    columns = ["arxiv_id", "title", "authors", "year", "categories", "cited_by_count", "url"]
    rows = []
    for paper in results:
        authors = "; ".join(paper.get("authors", []))
        categories = "; ".join(paper.get("categories", []))
        rows.append(
            {
                "arxiv_id": paper.get("arxiv_id", ""),
                "title": paper.get("title", ""),
                "authors": authors,
                "year": paper.get("year", ""),
                "categories": categories,
                "cited_by_count": paper.get("cited_by_count", 0),
                "url": f"https://arxiv.org/abs/{paper.get('arxiv_id', '')}",
            }
        )

    df = pd.DataFrame(rows, columns=columns)
    return df.to_csv(index=False).encode("utf-8")
