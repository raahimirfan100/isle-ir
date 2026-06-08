"""Topics tab — topic distribution of the retrieved set.

Owner:   Abubakar Ahmed (SSOT §2)
Phase:   5 — Streamlit UI
STATUS:  core
"""

from __future__ import annotations

from collections import defaultdict

import plotly.express as px
import streamlit as st


def render(
    topic_info: list[dict],
    doc_topics: list[int],
    result_papers: list[dict],
    topic_model=None,
    docs: list[str] | None = None,
) -> None:
    """Render the topic distribution (pie chart), per-topic keyword details,
    and (when available) a topics-over-time line chart (B-R3).

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
    topic_model:
        Fitted BERTopic model (optional). Required for topics-over-time chart.
    docs:
        Raw text for each paper in *result_papers* (parallel list). Required
        for topics-over-time chart.
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

    # -- Group papers by topic (from current results) --------------------
    topic_papers: dict[int, list[dict]] = defaultdict(list)
    for idx, topic_id in enumerate(doc_topics):
        if idx < len(result_papers):
            topic_papers[topic_id].append(result_papers[idx])

    # Keep only topics that have at least one paper in the current results
    active_topics = [t for t in topic_info if topic_papers.get(t["topic_id"])]
    if not active_topics:
        st.info("No topics matched papers in the current result set.")
        return

    # Recount from current results so the displayed counts are accurate
    for topic in active_topics:
        topic["doc_count"] = len(topic_papers[topic["topic_id"]])

    labels = [_label(t) for t in active_topics]

    # -- Pie chart --------------------------------------------------------
    fig = px.pie(
        values=[t["doc_count"] for t in active_topics],
        names=labels,
        title="Document Distribution by Topic",
    )
    fig.update_layout(margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # -- Per-topic sections -----------------------------------------------
    for topic in active_topics:
        tid = topic["topic_id"]
        papers = topic_papers.get(tid, [])

        with st.expander(
            f"**{_label(topic)}** — {topic['doc_count']} paper"
            f"{'s' if topic['doc_count'] != 1 else ''}"
        ):
            for paper in papers[:5]:
                title = paper.get("title", "Untitled")
                year = paper.get("year", "")
                st.markdown(f"- {title}" + (f" ({year})" if year else ""))

    # -- Topics over time (B-R3) ─────────────────────────────────────────
    if topic_model is None or docs is None:
        return

    timestamps = [str(p.get("year", "")) for p in result_papers]
    distinct_years = {t for t in timestamps if t and t != "None"}
    if len(distinct_years) < 3:
        st.subheader("Topic Trends Over Time")
        st.info("Not enough year diversity in this result set for temporal analysis.")
        return

    try:
        st.subheader("Topic Trends Over Time")
        tot = topic_model.topics_over_time(docs, timestamps, nr_bins=6, global_tuning=True)
        fig = topic_model.visualize_topics_over_time(tot, top_n_topics=8)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.caption(f"Topics-over-time chart unavailable: {exc}")
