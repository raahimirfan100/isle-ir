# Project Status

> Session-handoff tracker — read this and `AGENTS.md` at the start of any
> session. It records *state*; the *plan* lives in `docs/SSOT.md`. Update the
> phase table, the date, and the open items whenever state changes.

**Last updated:** 2026-05-22 · **Stage:** Phase 0 complete (Mujtaba). Awaiting OpenAlex key.

## Phase status

| Phase | Scope | Status |
|---|---|---|
| Bootstrap | repo scaffold, docs, CI, branch protection | done |
| 0 — Setup | env, GPU, embedding-model decision | done (Mujtaba) |
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

- **Branches:** `main` — protected: a PR is required but no review (the author
  can self-merge); no force-push or deletion. `dev` — direct merges allowed (no
  PR/review); no force-push or deletion. Feature work: `feat/<feature>` off
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
| `dev` accepts direct merges (no PR); `main` needs a PR but no review | `CONTRIBUTING.md`, SSOT §4 |
| Embedding model: **SPECTER2** (`allenai/specter2_base`, 768-dim); MiniLM kept as comparison baseline | `src/app/config.py` |
| Environment: **uv** (`uv.lock`, `cpu`/`gpu` extras) — conda dropped; PyTorch `cu126` + `faiss-gpu-cu12` | `pyproject.toml`, `setup.sh` |
| No Claude git attribution; no `claude/*` branch names | team preference |
| Canonical docs: `docs/SSOT.md` + `docs/IMPLEMENTATION.md` (older drafts dropped) | — |
| Committed scope first; enhancements deferred and TODO-marked | `AGENTS.md`, SSOT §10 |

## Open items

- [ ] Mujtaba: register OpenAlex key at https://openalex.org/api-key and set `OPENALEX_EMAIL` + `OPENALEX_API_KEY` in environment.
- [ ] After CI runs once, optionally add "Lint & syntax" as a required status check.
- [ ] Delete the merged local `chore/repo-bootstrap` branch.
