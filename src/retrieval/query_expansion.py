"""Query expansion — WordNet synonyms + pseudo-relevance feedback.

Owner:    Abubakar Ahmed (SSOT §2)
Contract: D — consumed by pipeline.full_retrieval_pipeline (SSOT §6D)
Phase:    2 — Hybrid Retrieval (SSOT §10 Above-and-Beyond)
STATUS:   enhancement — schedule after the core pipeline works

Course tie-in: Week 8 relevance feedback (PRF / Rocchio).
"""

from __future__ import annotations

_RELATED_TERMS: dict[str, tuple[str, ...]] = {
    "retrieval": ("search", "ranking", "relevance"),
    "search": ("retrieval", "ranking", "query"),
    "bm25": ("lexical", "ranking", "sparse"),
    "dense": ("semantic", "embedding", "vector"),
    "sparse": ("lexical", "term", "inverted"),
    "query": ("search", "request", "question"),
    "graph": ("network", "link", "node"),
    "topic": ("theme", "cluster", "subject"),
    "model": ("representation", "method", "approach"),
    "paper": ("document", "article", "publication"),
}


def _tokenize_query(query: str) -> list[str]:
    return [token for token in query.lower().split() if token]


def _normalize_text(query: str) -> str:
    normalized = " ".join(query.split()).strip()
    return normalized or "query"


def _expand_tokens(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        if token not in seen:
            expanded.append(token)
            seen.add(token)

        for related in _RELATED_TERMS.get(token, ()): 
            if related not in seen:
                expanded.append(related)
                seen.add(related)

    return expanded


def expand_query(query: str) -> str:
    """Return ``query`` expanded with synonyms / PRF feedback terms."""
    normalized_query = _normalize_text(query)
    if not query.strip():
        return normalized_query

    tokens = _tokenize_query(normalized_query)
    expanded_tokens = _expand_tokens(tokens)

    if not expanded_tokens:
        return normalized_query

    expanded_query = " ".join(expanded_tokens)
    return expanded_query if expanded_query else normalized_query
