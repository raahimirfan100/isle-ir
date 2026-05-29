# Phase Prompts

Self-contained, copy-paste prompts that drive the build one phase at a time.
They work with any AI agent (Claude Code, Cursor, Codex, Copilot Chat) and
double as a checklist for a human doing the work.

## How to use

1. Open the phase file you are starting (`phase-0-setup.md` …
   `phase-6-evaluation.md`).
2. Copy everything **below the divider** into your agent as the task.
3. The agent reads `AGENTS.md` first (the prompt instructs it to), then works.
4. Do not start a phase until the previous phase's exit criteria are all met.

Claude Code users: `/phase <N>` loads the phase-N prompt directly.

## Phases

| Phase | File | Primary owner(s) | Status |
|---|---|---|---|
| 0 | `phase-0-setup.md` | All | ✅ Done |
| 1 | `phase-1-data.md` | Mujtaba | ✅ Done |
| 2 | `phase-2-retrieval.md` | Abubakar + Raahim | ✅ Done |
| 3 | `phase-3-topics.md` | Abubakar + Raahim | ✅ Done |
| 4 | `phase-4-knowledge-graph.md` | Mujtaba | ✅ Done |
| 5 | `phase-5-frontend.md` | Mujtaba (+ all) | ⏳ In progress |
| 6 | `phase-6-evaluation.md` | Abubakar | ⏳ Not started |

Phases 0–6 cover the **committed proposal scope**. All `SSOT.md §10`
above-and-beyond features (cross-encoder reranking, query expansion, More Like
This, PageRank, author collaboration network) are already implemented.

## Bonus features (new — beyond SSOT §10)

See `docs/BONUS_FEATURES.md` for the full spec. Each owner has a self-contained
agent prompt:

| File | Owner | Features |
|---|---|---|
| `bonus-raahim.md` | Raahim | MMR re-ranking · UMAP 2D viz · topics-over-time · SPECTER2 vs MiniLM |
| `bonus-abubakar.md` | Abubakar | Rocchio feedback · ILD metric · citation flow Sankey |
| `bonus-mujtaba.md` | Mujtaba | Freshness re-ranking · citation path explorer · BibTeX/CSV export |

Start bonus work only after Phase 5 and Phase 6 are complete.
