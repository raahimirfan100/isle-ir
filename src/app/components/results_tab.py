"""Results tab — ranked retrieval results.

Owner:   Abubakar Ahmed (SSOT §2)
Phase:   5 — Streamlit UI
STATUS:  core
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from app.export import to_bibtex, to_csv

if TYPE_CHECKING:
    from retrieval.faiss_index import FAISSIndex

# Maximum number of result cards to display.
MAX_CARDS = 20
# Abstract truncation length.
_ABSTRACT_MAX = 300


def render(
    results: list[tuple[str, float]],
    result_papers: list[dict],
    result_df: pd.DataFrame,
    faiss_index: FAISSIndex,
) -> None:
    """Render ranked result cards with metadata, Rocchio feedback, and export.

    Parameters
    ----------
    results:
        List of ``(arxiv_id, rrf_score)`` tuples produced by RRF fusion.
    result_papers:
        List of paper metadata dicts, parallel to *results*.  Each dict has
        keys: ``arxiv_id``, ``title``, ``abstract``, ``authors`` (list[str]),
        ``year``, ``categories`` (list[str]), ``cited_by_count``,
        ``institutions`` (list[str]), ``countries`` (list[str]), ``doi``.
    result_df:
        DataFrame with at least an ``arxiv_id`` index and a ``_score``
        column containing the RRF score.
    faiss_index:
        Loaded FAISS index (reserved for "More Like This" — enhancement).
    """
    # -- Initialise session-state keys for Rocchio feedback ---------------
    if "rel_ids" not in st.session_state:
        st.session_state["rel_ids"]: list[str] = []
    if "nrel_ids" not in st.session_state:
        st.session_state["nrel_ids"]: list[str] = []

    # -- Empty-results guard ------------------------------------------------
    if not results or not result_papers:
        st.info("No results found. Try a different query.")
        return

    n_found = min(len(results), MAX_CARDS)
    st.subheader(f"Results ({n_found} paper{'s' if n_found != 1 else ''} found)")

    # -- Build score lookup from result_df ----------------------------------
    score_lookup: dict[str, float] = {}
    if not result_df.empty and "_score" in result_df.columns:
        score_lookup = result_df["_score"].to_dict()

    # -- Result cards ------------------------------------------------------
    for idx in range(min(len(results), MAX_CARDS)):
        arxiv_id, _rrf = results[idx]
        if idx >= len(result_papers):
            break
        paper = result_papers[idx]

        # Gather fields (graceful defaults if a key is missing).
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", [])
        year = paper.get("year", "")
        categories = paper.get("categories", [])
        abstract = paper.get("abstract", "")
        cited_by = paper.get("cited_by_count", 0)

        # Use the RRF score from result_df when available, fall back to
        # the raw score in the results tuple.
        score = score_lookup.get(arxiv_id, _rrf)

        # Truncate long abstracts.
        display_abstract = (
            abstract[:_ABSTRACT_MAX] + "..." if len(abstract) > _ABSTRACT_MAX else abstract
        )

        # -- Card header (rank + title) ------------------------------------
        rank_label = f"**{idx + 1}.**"
        title_link = f"[{title}](https://arxiv.org/abs/{arxiv_id})"
        st.markdown(f"{rank_label} {title_link}")

        # -- Metadata row ---------------------------------------------------
        meta_parts: list[str] = []
        if authors:
            author_str = ", ".join(authors)
            meta_parts.append(f"Authors: {author_str}")
        if year:
            meta_parts.append(f"Year: {year}")
        if categories:
            meta_parts.append(f"Categories: {', '.join(categories)}")
        st.caption(" | ".join(meta_parts))

        # -- Expandable details ---------------------------------------------
        with st.expander("Show details"):
            st.markdown(f"**RRF Score:** {score:.4f}")
            if display_abstract:
                st.markdown(f"**Abstract:** {display_abstract}")
            st.markdown(f"**Cited by:** {cited_by}")

        # -- Rocchio feedback buttons --------------------------------------
        rel_col, nrel_col = st.columns([1, 1])
        with rel_col:
            if st.button("👍", key=f"rel_{arxiv_id}"):
                if arxiv_id not in st.session_state["rel_ids"]:
                    st.session_state["rel_ids"].append(arxiv_id)
                # Remove from non-relevant if it was there.
                if arxiv_id in st.session_state["nrel_ids"]:
                    st.session_state["nrel_ids"].remove(arxiv_id)
                st.rerun()
        with nrel_col:
            if st.button("👎", key=f"nrel_{arxiv_id}"):
                if arxiv_id not in st.session_state["nrel_ids"]:
                    st.session_state["nrel_ids"].append(arxiv_id)
                # Remove from relevant if it was there.
                if arxiv_id in st.session_state["rel_ids"]:
                    st.session_state["rel_ids"].remove(arxiv_id)
                st.rerun()

        st.divider()

    # -- Export buttons -----------------------------------------------------
    st.subheader("Export Results")

    # Export only the papers we have metadata for.
    export_papers = result_papers[:MAX_CARDS]

    bib_str = to_bibtex(export_papers)
    csv_bytes = to_csv(export_papers)

    exp_col_bib, exp_col_csv = st.columns(2)
    with exp_col_bib:
        st.download_button(
            label="Export BibTeX",
            data=bib_str,
            file_name="results.bib",
            mime="text/plain",
            disabled=not bib_str,
        )
    with exp_col_csv:
        st.download_button(
            label="Export CSV",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
            disabled=not csv_bytes,
        )
