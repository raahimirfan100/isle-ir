"""Topics tab — topic distribution of the retrieved set.

Owner:   Abubakar Ahmed (SSOT §2)
Phase:   5 — Streamlit UI
STATUS:  core
"""

from __future__ import annotations

from collections import defaultdict

import plotly.express as px
import streamlit as st


def render(topic_info: list[dict], doc_topics: list[int], result_papers: list[dict]) -> None:
    """Render the topic distribution (pie chart) and per-topic keyword details.

    Parameters
    ----------
    topic_info:
        List of topic dicts, each with keys ``topic_id``, ``keywords``,
        ``doc_count``, and ``label``.
    doc_topics:
        Topic id assigned to each paper in *result_papers* (same length).
    result_papers:
        Paper dicts returned by the retrieval pipeline, each containing
        ``title`` and ``year`` at minimum.
    """
    # -- Header -----------------------------------------------------------
    if not topic_info:
        st.info(
            "Topics could not be computed for this result set. "
            "This can happen when very few papers are retrieved."
        )
        return

    num_topics = len(topic_info)
    st.subheader(f"Topic Analysis ({num_topics} topic{'s' if num_topics != 1 else ''} found)")

    # -- Build topic labels (first 3 keywords) ---------------------------
    def _label(topic: dict) -> str:
        kw = topic.get("keywords", [])
        return ", ".join(kw[:3])

    labels = [_label(t) for t in topic_info]

    # -- Pie chart --------------------------------------------------------
    fig = px.pie(
        values=[t["doc_count"] for t in topic_info],
        names=labels,
        title="Document Distribution by Topic",
    )
    fig.update_layout(margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # -- Group papers by topic -------------------------------------------
    topic_papers: dict[int, list[dict]] = defaultdict(list)
    for idx, topic_id in enumerate(doc_topics):
        if idx < len(result_papers):
            topic_papers[topic_id].append(result_papers[idx])

    # -- Per-topic sections -----------------------------------------------
    for topic in topic_info:
        tid = topic["topic_id"]
        papers = topic_papers.get(tid, [])

        with st.expander(
            f"**{_label(topic)}** — {topic['doc_count']} paper"
            f"{'s' if topic['doc_count'] != 1 else ''}"
        ):
            if papers:
                for paper in papers[:3]:
                    title = paper.get("title", "Untitled")
                    year = paper.get("year", "")
                    st.markdown(f"- {title}" + (f" ({year})" if year else ""))
            else:
                st.caption("No papers assigned to this topic in the current results.")
