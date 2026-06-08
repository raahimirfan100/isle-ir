"""Corpus statistics — Heaps' Law and Zipf's Law.

Owner:   Abubakar Ahmed (SSOT §2)
Phase:   6 — Evaluation (SSOT §13)
STATUS:  core

Produces the vocabulary-growth (Heaps) and term-frequency (Zipf) plots over
corpus.parquet for the report's Corpus Statistics section. Course tie-in:
Week 4 (Heaps' / Zipf's law).
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from PIL import Image, ImageDraw, ImageFont

from app.config import BERTOPIC_DIR, CORPUS_PATH, EVAL_RESULTS_DIR
from topics.bertopic_model import _get_topic_info
from topics.nmf_model import run_nmf

HEAPS_SAMPLE_INTERVAL = 5_000
COHERENCE_SAMPLE_SIZE = 1_000
ZIPF_TOP_N = 5_000
PLOT_SIZE = (1200, 800)


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _load_corpus() -> pd.DataFrame:
    return pd.read_parquet(CORPUS_PATH)


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _normalize_topic_keywords(keywords: list[object]) -> list[str]:
    cleaned: list[str] = []
    for keyword in keywords:
        word = str(keyword).strip()
        if word and " " not in word:
            cleaned.append(word)
    return cleaned


def _topic_keyword_ids(topics: list[list[str]], dictionary: Dictionary) -> list[list[int]]:
    topic_ids: list[list[int]] = []
    for topic in topics:
        ids = [dictionary.token2id[word] for word in topic if word in dictionary.token2id]
        if len(ids) >= 2:
            topic_ids.append(ids)
    return topic_ids


def _load_bertopic_model() -> BERTopic:
    model_path = BERTOPIC_DIR / "model"
    if model_path.exists():
        try:
            return BERTopic.load(str(model_path))
        except Exception:
            if model_path.is_dir():
                shutil.rmtree(model_path)
            else:
                model_path.unlink()
            for child in list(BERTOPIC_DIR.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    BERTOPIC_DIR.mkdir(parents=True, exist_ok=True)
    release_cmd = [
        "gh",
        "release",
        "download",
        "bertopic-index",
        "--repo",
        "raahimirfan100/isle-ir",
        "--dir",
        str(BERTOPIC_DIR),
    ]
    try:
        subprocess.run(release_cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "gh is required to download the published BERTopic model artifact."
        ) from exc

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find BERTopic model after download at {model_path}."
        )

    return BERTopic.load(str(model_path))


def _draw_series_plot(
    points: list[tuple[float, float]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    line_color: tuple[int, int, int],
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    width, height = PLOT_SIZE
    margin_left = 120
    margin_right = 40
    margin_top = 90
    margin_bottom = 110
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(26)
    label_font = _load_font(18)
    small_font = _load_font(14)

    draw.text((margin_left, 24), title, fill="black", font=title_font)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)

    if not points:
        draw.text(
            (margin_left, margin_top + 40),
            "No data available.",
            fill="black",
            font=label_font,
        )
        image.save(output_path)
        return

    transformed_x: list[float] = []
    transformed_y: list[float] = []
    for x_value, y_value in points:
        if log_x:
            x_value = math.log10(x_value)
        if log_y:
            y_value = math.log10(y_value)
        transformed_x.append(x_value)
        transformed_y.append(y_value)

    x_min = min(transformed_x)
    x_max = max(transformed_x)
    y_min = min(transformed_y)
    y_max = max(transformed_y)

    if x_max == x_min:
        x_max += 1.0
    if y_max == y_min:
        y_max += 1.0

    def map_x(value: float) -> float:
        return plot_left + ((value - x_min) / (x_max - x_min)) * (plot_right - plot_left)

    def map_y(value: float) -> float:
        return plot_bottom - ((value - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)

    mapped_points = [
        (map_x(x_value), map_y(y_value))
        for x_value, y_value in zip(transformed_x, transformed_y, strict=False)
    ]

    draw.line(mapped_points, fill=line_color, width=3, joint="curve")

    x_ticks = 5
    y_ticks = 5
    for index in range(x_ticks + 1):
        ratio = index / x_ticks
        x_value = x_min + (x_max - x_min) * ratio
        if log_x:
            original_x = 10 ** x_value
            tick_label = f"{original_x:,.0f}" if original_x >= 1000 else f"{original_x:.0f}"
        else:
            original_x = x_value
            tick_label = f"{original_x:,.0f}"
        pixel_x = map_x(x_value)
        draw.line((pixel_x, plot_bottom, pixel_x, plot_bottom + 6), fill="black", width=1)
        draw.text((pixel_x - 20, plot_bottom + 10), tick_label, fill="black", font=small_font)

    for index in range(y_ticks + 1):
        ratio = index / y_ticks
        y_value = y_min + (y_max - y_min) * ratio
        if log_y:
            original_y = 10 ** y_value
            tick_label = f"{original_y:,.0f}" if original_y >= 1000 else f"{original_y:.0f}"
        else:
            original_y = y_value
            tick_label = f"{original_y:,.0f}"
        pixel_y = map_y(y_value)
        draw.line((plot_left - 6, pixel_y, plot_left, pixel_y), fill="black", width=1)
        draw.text((20, pixel_y - 8), tick_label, fill="black", font=small_font)

    draw.text((width // 2 - 80, height - 50), x_label, fill="black", font=label_font)
    draw.text((18, height // 2 - 40), y_label, fill="black", font=label_font)
    image.save(output_path)


def _build_heaps_points(df: pd.DataFrame) -> tuple[list[tuple[float, float]], int, int]:
    all_tokens: list[str] = []
    vocab: set[str] = set()
    points: list[tuple[float, float]] = []

    for index, text in enumerate(df["text"], start=1):
        tokens = _tokenize(str(text))
        all_tokens.extend(tokens)
        vocab.update(tokens)

        if index % HEAPS_SAMPLE_INTERVAL == 0 or index == len(df):
            points.append((float(len(all_tokens)), float(len(vocab))))

    return points, len(all_tokens), len(vocab)


def _build_zipf_points(df: pd.DataFrame) -> tuple[list[tuple[float, float]], int]:
    freq = Counter()
    for text in df["text"]:
        freq.update(_tokenize(str(text)))

    sorted_freq = sorted(freq.values(), reverse=True)[:ZIPF_TOP_N]
    points = [(float(rank), float(count)) for rank, count in enumerate(sorted_freq, start=1)]
    return points, len(freq)


def _coherence_scores(
    docs: list[str], bertopic_model: BERTopic
) -> tuple[float, float, float, float]:
    tokenized_docs = [_tokenize(doc) for doc in docs]
    dictionary = Dictionary(tokenized_docs)

    nmf_topic_info, _doc_topics, _doc_probs = run_nmf(docs, n_topics=10)
    nmf_keywords = [
        _normalize_topic_keywords(topic["keywords"])
        for topic in nmf_topic_info
        if isinstance(topic.get("keywords"), list)
        and len(_normalize_topic_keywords(topic["keywords"])) >= 2
    ]

    bertopic_info = _get_topic_info(bertopic_model)
    bertopic_keywords = [
        _normalize_topic_keywords(topic["keywords"])
        for topic in bertopic_info
        if isinstance(topic.get("keywords"), list)
        and len(_normalize_topic_keywords(topic["keywords"])) >= 2
    ]

    nmf_topic_ids = _topic_keyword_ids(nmf_keywords, dictionary)
    bertopic_topic_ids = _topic_keyword_ids(bertopic_keywords, dictionary)

    if not nmf_topic_ids:
        raise ValueError(
            "No NMF topics contained at least two vocabulary tokens for coherence scoring."
        )
    if not bertopic_topic_ids:
        raise ValueError(
            "No BERTopic topics contained at least two vocabulary tokens for coherence scoring."
        )

    nmf_cv = float(
        CoherenceModel(
            topics=nmf_topic_ids,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        ).get_coherence()
    )
    nmf_npmi = float(
        CoherenceModel(
            topics=nmf_topic_ids,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_npmi",
            processes=1,
        ).get_coherence()
    )
    bertopic_cv = float(
        CoherenceModel(
            topics=bertopic_topic_ids,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        ).get_coherence()
    )
    bertopic_npmi = float(
        CoherenceModel(
            topics=bertopic_topic_ids,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_npmi",
            processes=1,
        ).get_coherence()
    )

    return nmf_cv, nmf_npmi, bertopic_cv, bertopic_npmi


def compute_corpus_stats() -> None:
    """Compute and save the Heaps' and Zipf's law plots to evaluation/results/."""
    df = _load_corpus()
    if df.empty:
        raise ValueError("corpus.parquet is empty.")

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    heaps_points, token_count, vocab_size = _build_heaps_points(df)
    _draw_series_plot(
        heaps_points,
        EVAL_RESULTS_DIR / "heaps_law.png",
        "Heaps' Law - Vocabulary Growth",
        "Tokens processed",
        "Vocabulary size",
        (79, 195, 247),
    )

    zipf_points, unique_terms = _build_zipf_points(df)
    _draw_series_plot(
        zipf_points,
        EVAL_RESULTS_DIR / "zipf_law.png",
        "Zipf's Law - Term Frequency Distribution",
        "Rank (log scale)",
        "Frequency (log scale)",
        (255, 138, 101),
        log_x=True,
        log_y=True,
    )

    sample_size = min(COHERENCE_SAMPLE_SIZE, len(df))
    coherence_docs = df["text"].sample(n=sample_size, random_state=42).tolist()
    bertopic_model = _load_bertopic_model()

    nmf_cv, nmf_npmi, bertopic_cv, bertopic_npmi = _coherence_scores(
        coherence_docs,
        bertopic_model,
    )

    print(f"Corpus: {len(df)} papers | {token_count:,} tokens | {vocab_size:,} unique terms")
    print(f"Heaps' Law plot: {EVAL_RESULTS_DIR / 'heaps_law.png'}")
    print(f"Zipf's Law plot: {EVAL_RESULTS_DIR / 'zipf_law.png'}")
    print(f"Zipf unique terms: {unique_terms}")
    print(f"Topic coherence sample size: {sample_size}")
    print(f"NMF c_v: {nmf_cv:.4f}")
    print(f"NMF NPMI: {nmf_npmi:.4f}")
    print(f"BERTopic c_v: {bertopic_cv:.4f}")
    print(f"BERTopic NPMI: {bertopic_npmi:.4f}")


if __name__ == "__main__":
    compute_corpus_stats()
