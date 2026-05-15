"""Streamlit application entry point.

Owner:    Mujtaba Kamran (layout) · Raahim Irfan (integration review) (SSOT §2)
Contract: none — orchestrates Contracts B-F
Phase:    5 — Streamlit UI (IMPLEMENTATION.md Phase 5)
STATUS:   core

Run with:  streamlit run src/app/main.py

Loads the indexes and corpus (cached), runs the retrieval pipeline, models
topics, builds the knowledge graph, and renders the tabbed UI.
"""

from __future__ import annotations


def main() -> None:
    """Render the Streamlit app."""
    # TODO(mujtaba, phase-5): build the layout — sidebar inputs,
    #   @st.cache_resource for indexes, @st.cache_data for the corpus, and
    #   tab routing into app/components/*. See IMPLEMENTATION.md Phase 5.
    raise NotImplementedError


if __name__ == "__main__":
    main()
