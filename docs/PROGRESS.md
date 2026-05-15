# Project Status

> Session-handoff tracker — read this and `AGENTS.md` at the start of any
> session. It records *state*; the *plan* lives in `docs/SSOT.md`. Update the
> phase table, the date, and the open items whenever state changes.

**Last updated:** 2026-05-16 · **Stage:** bootstrap complete — Phase 0 is next.

## Phase status

| Phase | Scope | Status |
|---|---|---|
| Bootstrap | repo scaffold, docs, CI, branch protection | done |
| 0 — Setup | env, GPU, embedding-model decision | not started |
| 1 — Data | arXiv + OpenAlex -> corpus.parquet | not started |
| 2 — Retrieval | BM25 + FAISS + RRF + pipeline | not started |
| 3 — Topics | NMF + BERTopic | not started |
| 4 — Knowledge graph | builder + analytics + visualization | not started |
| 5 — Frontend | Streamlit app | not started |
| 6 — Evaluation | ablation, coherence, corpus stats | not started |

Each phase has a copy-paste prompt in `docs/agent-prompts/`. Mark a phase done
only when its exit-criteria checklist is fully met.

## Starting a new session

1. Read `AGENTS.md` (how to work here) and this file (where things stand).
2. Take the current phase from the table above; open its
   `docs/agent-prompts/phase-N-*.md` prompt and follow it.
3. Spec and contracts: `docs/SSOT.md`. Code and pitfalls: `docs/IMPLEMENTATION.md`.

## Infrastructure state

- **Branches:** `main` and `dev`, both protected (PR + 1 review, no force-push,
  no deletion; the repo admin can bypass). Feature work: `feat/<feature>` off
  `dev`. The local `chore/repo-bootstrap` branch is merged and disposable.
- **CI:** ruff lint + syntax + package-layout check on every PR and push.
- **Dependabot:** weekly pip and GitHub Actions update PRs.
- **Visibility:** private. Intended public after the project concludes — secret
  scanning and push protection auto-enable then.
- **Large artifacts:** GitHub Release assets, never the git tree (SSOT §15).

## Key decisions — index (detail lives where noted)

| Decision | Lives in |
|---|---|
| MIT license | `LICENSE` |
| Conventional Commits, no machine enforcement (no hooks) | `AGENTS.md` |
| Branch names by feature, not developer | `CONTRIBUTING.md` |
| Embedding model — OPEN, lock in Phase 0 (default MiniLM/384; SSOT contracts assume SPECTER2/768) | `src/app/config.py` |
| No Claude git attribution; no `claude/*` branch names | team preference |
| Canonical docs: `docs/SSOT.md` + `docs/IMPLEMENTATION.md` (older drafts dropped) | — |
| Committed scope first; enhancements deferred and TODO-marked | `AGENTS.md`, SSOT §10 |

## Open items

- [ ] Phase 0: team locks the embedding model in `src/app/config.py`.
- [ ] After CI runs once, optionally add "Lint & syntax" as a required status check.
- [ ] Delete the merged local `chore/repo-bootstrap` branch.
