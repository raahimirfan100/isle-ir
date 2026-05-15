# Phase 4 — Knowledge Graph

> Copy everything below the divider into your agent. Owner: Mujtaba.

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then do **Phase 4:
Knowledge Graph**.

## Read first
- `docs/IMPLEMENTATION.md` §4 — graph builder, analytics, visualization.
- `docs/SSOT.md` §6 Contract F.

## Goal
Turn a retrieved set (plus its topic assignments) into a heterogeneous
knowledge graph, with analytics and an interactive visualization.

## Deliverables
- `src/graph/builder.py` — `build_knowledge_graph` (Contract F).
- `src/graph/analytics.py` — `compute_paper_impact`, `graph_summary`.
- `src/graph/visualization.py` — `export_pyvis`.

## Contract to honor
Contract F (`docs/SSOT.md` §6F): a `networkx.DiGraph` with `node_type` on every
node — `paper`, `author`, `institution`, `country`, `topic`, `year` — and edge
`rel` attrs `authored`, `affiliated`, `from_country`, `published_in`,
`belongs_to`, `cites`.

## Steps
1. Implement `build_knowledge_graph` per IMPLEMENTATION.md §4.1. Add nodes for
   every type; keep `cites` edges intra-corpus; skip outlier topic `-1`.
2. Implement impact metrics (ISLE Eq. 19–20) and `graph_summary` in
   `analytics.py`.
3. Implement the PyVis HTML export, coloured by `node_type`.
4. Consume the Phase 3 topic output (Contract E) directly — do not reshape it.

## Exit criteria — verify every box before reporting done
- [ ] `build_knowledge_graph` returns a DiGraph with all six node types.
- [ ] `belongs_to` edges connect papers to their topics.
- [ ] `cites` edges are present and intra-corpus only.
- [ ] `graph_summary` and `compute_paper_impact` return sensible numbers.
- [ ] `export_pyvis` produces an interactive HTML file.
- [ ] `ruff check .` clean; the builder consumes Contract E output unchanged.

## Enhancements (defer — leave TODOs)
PageRank link analysis and the author collaboration network are above-and-beyond
(`docs/SSOT.md` §10). Leave `# TODO(mujtaba, enhancement): ...` markers where
they plug in; do not build them until the core graph works.

## Commit & branch
Branch `feat/mujtaba-kg` off `dev`. Small commits: `feat(graph): build paper
and author nodes`, `feat(graph): add citation impact analytics`.
