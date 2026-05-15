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

| Phase | Name | Primary owner(s) |
|---|---|---|
| 0 | Environment & project setup | All |
| 1 | Data acquisition | Mujtaba |
| 2 | Hybrid retrieval engine | Abubakar + Raahim |
| 3 | Topic modeling | Abubakar + Raahim |
| 4 | Knowledge graph | Mujtaba |
| 5 | Streamlit frontend | Mujtaba (+ all) |
| 6 | Evaluation | Abubakar |

Phases 0–6 cover the **committed proposal scope**. Above-and-beyond
enhancements (cross-encoder reranking, query expansion, More Like This,
PageRank, author collaboration network) are deferred — see `docs/SSOT.md` §10 —
and marked with `# TODO(..., enhancement)` in the code. Tackle them only after
the core system runs end to end.
