# Contributing to ISLE-IR

This guide covers the team workflow. AI agents (Claude Code, Cursor, Codex,
Copilot) should **also** read [`AGENTS.md`](AGENTS.md), which builds on this.

## Setup

See the [Quickstart in `README.md`](README.md#quickstart). All paths and
constants come from `src/app/config.py` — **never hardcode a path** anywhere
else.

## Branching model

```
main           protected — merge only via reviewed PR
  └── dev      integration branch — features merge here directly (no PR)
        ├── feat/bm25-rrf
        ├── feat/faiss-index
        └── feat/ingestion
```

- Name branches by **feature, not developer**: `feat/<feature>` (or `fix/`,
  `docs/`, `chore/`). The repo is public — branch names should not carry
  personal names. Example: `feat/faiss-index`, not `feat/raahim-faiss`.
- Branch off `dev`, never off `main`.
- Merge feature branches into `dev` directly — no PR or review required.
- Never force-push or delete `dev` or `main`.
- Raahim owns the final integration merge from `dev` to `main`.

## Commits — Conventional Commits

Every commit message uses the [Conventional Commits](https://www.conventionalcommits.org/)
format. This keeps history readable and is portfolio-visible.

```
<type>(<scope>): <short imperative summary>
```

| Type | Use for |
|---|---|
| `feat` | a new feature |
| `fix` | a bug fix |
| `docs` | documentation only |
| `test` | adding or fixing tests |
| `refactor` | code change that neither fixes a bug nor adds a feature |
| `chore` | tooling, scaffolding, deps, housekeeping |
| `build` / `ci` | build system or CI config |

Scope is the module: `retrieval`, `topics`, `graph`, `ingestion`, `app`, `eval`.

```
feat(retrieval): add BM25 index build and search
fix(graph): skip outlier topic -1 when adding belongs_to edges
docs: document OpenAlex API-key requirement
```

**Commit small and often.** One logical change per commit; avoid sprawling
diffs (rule of thumb: keep a commit under ~200 changed lines). This is a team
convention, not machine-enforced — there is no commit hook. Discipline is on you.

## Interface contracts

`docs/SSOT.md` §6 defines the exact function/class signatures that connect each
person's work (Contracts A–F). Your internal implementation is your own; your
**public signature is not**.

> Do not change a Contract signature without announcing it to the team and
> updating `docs/SSOT.md` in the same PR. A silent signature change breaks
> someone else's module.

## Merging

**Into `dev`** — merge your feature branch directly; a merge commit is fine,
no PR or review required. First make sure your branch runs without errors and
`ruff check .` passes (CI also runs on every push to `dev`).

**Into `main`** — open a PR from `dev`, fill in the PR template (summary,
contract impact, test plan), get **one** teammate's review, and confirm CI is
green. Raahim owns this final integration merge.

## Large files

`corpus.parquet`, `embeddings.npy`, and FAISS/BM25 indexes are **never**
committed — `.gitignore` blocks them. Distribute them as **GitHub Release
assets** (`gh release create data-v1 <files>` / `gh release download data-v1`);
see `docs/SSOT.md` §15.

## Issues

Use the **Task** issue template to track work items. Link the issue in your PR.
