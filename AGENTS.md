# AGENTS.md — Working in the ISLE-IR Repository

Read this before doing any work here — human or AI agent. It is the contract
for **how** to work. **What** to build lives in [`docs/SSOT.md`](docs/SSOT.md)
(project spec) and [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)
(technical guide).

## The project in three sentences

ISLE-IR is a hybrid scientific-literature search system: it fuses BM25 and dense
retrieval with Reciprocal Rank Fusion, models the topics of the retrieved set,
builds a knowledge graph, and serves it through a Streamlit UI. It is a 5-week
university group project (3 teammates) that will become a public portfolio repo,
so engineering quality and clean history matter. It is based on the ISLE paper,
arXiv:2512.12760 (`docs/papers/isle-paper.pdf`).

```
query → expansion → [ BM25 ‖ dense/FAISS ] → RRF → rerank
                                                ├→ topic modeling
                                                └→ knowledge graph → Streamlit UI
```

## Where to find what

| You need… | Read… |
|---|---|
| Current status — what is done, what to do next | `docs/PROGRESS.md` |
| Roles, interface contracts, timeline, evaluation plan | `docs/SSOT.md` |
| Phase-by-phase code, libraries, known pitfalls | `docs/IMPLEMENTATION.md` |
| A ready-to-run task for a phase | `docs/agent-prompts/phase-N-*.md` |
| Git / branch / PR workflow | `CONTRIBUTING.md` |
| Paths, model names, hyperparameters | `src/app/config.py` |

Do not copy SSOT / IMPLEMENTATION content into code or new docs — link to it.

## Golden rules

1. **Honor the interface contracts.** `docs/SSOT.md` §6 (Contracts A–F) fixes
   the public signatures that connect each teammate's work. Implement *behind*
   a contract freely; never change a signature without updating `docs/SSOT.md`
   and flagging it in the PR — a silent change breaks someone else's module.
2. **`src/app/config.py` is the only home for paths and constants.** No
   hardcoded paths, model names or hyperparameters anywhere else.
3. **Committed scope first.** Build the proposal scope — BM25 + dense + RRF,
   topic modeling, knowledge graph, Streamlit UI — before any above-and-beyond
   feature. Enhancement modules are marked `STATUS: enhancement` in their
   docstrings; leave them stubbed until the core pipeline works end to end.
4. **Stay in your phase.** Work one phase at a time and satisfy its exit
   criteria before moving on.

## Git & commits

Full workflow in `CONTRIBUTING.md`. The essentials for an agent:

- **Conventional Commits** — `type(scope): summary`. Types: `feat`, `fix`,
  `docs`, `test`, `refactor`, `chore`, `build`, `ci`. Scope = module
  (`retrieval`, `topics`, `graph`, `ingestion`, `app`, `eval`).
- **Commit small and often.** One logical change per commit; a phase is many
  commits, not one. Avoid large diffs — rule of thumb, keep a commit under
  ~200 changed lines.
- **No machine enforcement.** There is no commit hook or commit-time linter.
  This convention is on you — following it keeps the public history clean.
- Branch by **feature, not developer**: `feat/<feature>` off `dev` (e.g.
  `feat/faiss-index`). The repo is public — keep personal names out of branch
  names. Never push to `dev` or `main` directly; never force-push.

### Raahim's personal workflow (Raahim's tasks only)

This is **not** a project-wide rule — other teammates commit however they
prefer. When an agent is doing **Raahim's** work specifically:

- Organize the work into short-lived sub-branches off the feature branch
  (e.g. `feat/faiss-index/encoding`, then `feat/faiss-index/index-build`).
- Commit *very* frequently in small, self-contained steps, so the history
  reads as a clear progression of the work.

## TODO conventions

- In code, mark deferred work as `# TODO(<owner>, <phase>): <what>` — e.g.
  `# TODO(raahim, phase-2): …` or `# TODO(mujtaba, enhancement): …`. The
  scaffold stubs already follow this; keep extending it.
- Where an above-and-beyond feature would plug in, leave a
  `# TODO(<owner>, enhancement): …` instead of building it now.
- Use your task-list tool (Claude Code: TodoWrite) for the current session's
  steps. That is session state — not a replacement for in-code TODOs.

## Verification & self-evaluation

Before reporting any unit of work as done:

1. **Contract check** — does the code match the relevant `docs/SSOT.md` §6
   signature exactly (names, argument order, return shape)?
2. **Lint** — `ruff check .` is clean.
3. **It runs** — imports succeed and the function does what its docstring
   claims on a small input.
4. **Phase exit criteria** — every checkbox in the phase prompt is genuinely
   satisfied. Do not tick a box you have not verified. If work is partial, say
   so plainly — never claim a phase is complete when it is not.

Streamlit UI work is not "done" until it has been opened in a browser and
exercised — a clean import and passing lint do not prove a feature works.

## Phase workflow

Work proceeds through phases 0–6, each with a self-contained prompt in
`docs/agent-prompts/`. Copy a phase prompt into any agent to start that phase.
**Phase 0 includes a required team decision** — the embedding model
(MiniLM/384 vs SPECTER2/768); the knob is `src/app/config.py` and the trade-off
is documented in the `docs/IMPLEMENTATION.md` header.

## For non-Claude-Code agents

This file is the canonical guide and is plain Markdown. Cursor and Codex read
`AGENTS.md` natively; GitHub Copilot reads `.github/copilot-instructions.md`,
which points here; Claude Code reads `CLAUDE.md`, which imports this file.
Everything above applies regardless of which agent or tool you use.
