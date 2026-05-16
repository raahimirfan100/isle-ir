---
description: Draft a Conventional Commits message for the staged changes and commit.
---

Review the staged changes with `git diff --cached` and any unstaged changes
with `git diff`.

Then:
1. Pick the `type`: feat, fix, docs, refactor, test, chore, build, ci, perf.
2. Pick the `scope` (the module): ingestion, retrieval, topics, graph, app, eval.
3. Write a concise description — imperative mood, no trailing period, ~60 chars.
4. For a non-trivial change, add a body explaining *why* (the diff shows what).
5. Commit: `git commit -m "type(scope): description"` (use a heredoc for a body).

Rules:
- **Do NOT add a `Co-Authored-By` trailer** — commits are attributed to the
  human author only.
- One logical change per commit. If more than ~5 files or ~200 lines are
  staged, suggest splitting into several commits instead.
- Examples:
  - `feat(retrieval): add BM25 index build and search`
  - `fix(graph): skip outlier topic -1 in belongs_to edges`

If nothing is staged, run `git status` and suggest what to stage based on the
recent work. After committing, show `git log -1 --oneline`.
