"""Semantic Map tab — UMAP 2D embedding visualization.

Owner:    Raahim Irfan (SSOT §2)
Phase:    Bonus (B-R2)
STATUS:   enhancement

Projects the top-50 retrieved papers into 2D using the fitted BERTopic UMAP
model, coloured by topic cluster. Interactive Plotly scatter with hover showing
title, year, and citation count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render(
    result_ids: list[str],
    topic_assignments: list[int],
    all_embeddings: np.ndarray,
    pid_to_idx: dict[str, int],
    topic_info: list[dict],
    papers_lookup: dict[str, dict] | None = None,
) -> None:
    """Render a 2D UMAP scatter of retrieved results coloured by topic.

    Calls BERTopic's umap_model.transform() on the result subset — does NOT
    refit. Falls back to PCA if the model is unavailable.
    """
    st.subheader("Semantic Map")

    if not result_ids:
        st.info("Run a search first to see the semantic map.")
        return

    # -- Gather embeddings for the result subset --------------------------
    subset_idxs = [pid_to_idx[pid] for pid in result_ids if pid in pid_to_idx]
    if len(subset_idxs) < 3:
        st.info("Not enough papers in the result set to plot a semantic map.")
        return

    subset_embs = all_embeddings[subset_idxs].astype(np.float32)
    subset_ids = [pid for pid in result_ids if pid in pid_to_idx]
    n = len(subset_ids)

    # -- 2D projection (BERTopic UMAP if available, else PCA) -------------
    coords_2d: np.ndarray | None = None

    try:
        from bertopic import BERTopic

        model_dir = None
        try:
            from app.config import BERTOPIC_DIR

            candidate = BERTOPIC_DIR / "model"
            if candidate.exists():
                model_dir = str(candidate)
        except Exception:
            pass

        if model_dir:
            topic_model = BERTopic.load(model_dir)
            umap_out = topic_model.umap_model.transform(subset_embs)
            coords_2d = umap_out[:, :2]
    except Exception:
        pass

    if coords_2d is None:
        # PCA fallback — no extra dependency
        centered = subset_embs - subset_embs.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords_2d = centered @ vt[:2].T

    # -- Build topic label per paper --------------------------------------
    topic_label_map: dict[int, str] = {}
    for t in topic_info:
        kw = t.get("keywords", [])
        label = ", ".join(str(k) for k in kw[:3]) if kw else f"Topic {t['topic_id']}"
        topic_label_map[t["topic_id"]] = label

    if len(topic_assignments) >= n:
        assigned = topic_assignments[:n]
    else:
        assigned = topic_assignments + [-1] * (n - len(topic_assignments))

    rows = []
    for i, pid in enumerate(subset_ids):
        paper = (papers_lookup or {}).get(pid, {})
        rows.append(
            {
                "x": float(coords_2d[i, 0]),
                "y": float(coords_2d[i, 1]),
                "topic": topic_label_map.get(assigned[i], "Unclustered"),
                "title": paper.get("title", pid)[:80],
                "year": paper.get("year", ""),
                "cited_by_count": paper.get("cited_by_count", 0),
            }
        )

    df = pd.DataFrame(rows)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="topic",
        hover_data={"title": True, "year": True, "cited_by_count": True, "x": False, "y": False},
        title=f"2D semantic map — {n} papers",
        labels={"x": "Dim 1", "y": "Dim 2"},
    )
    fig.update_traces(marker_size=8)
    fig.update_layout(legend_title_text="Topic", margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Positions via BERTopic UMAP projection (or PCA fallback). Colours = topic cluster.")
