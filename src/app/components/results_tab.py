"""Results tab — ranked retrieval results.

Owner:   Abubakar Ahmed (SSOT §2)
Phase:   5 — Streamlit UI
STATUS:  core
"""

from __future__ import annotations


def render(results: list[tuple[str, float]], papers_lookup: dict[str, dict]) -> None:
    """Render ranked result cards: title, score, abstract, metadata."""
    # TODO(abubakar, phase-5): result cards.
    # TODO(abubakar, enhancement): per-card "More Like This" button.
    raise NotImplementedError
