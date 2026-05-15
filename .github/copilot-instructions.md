# GitHub Copilot Instructions

The canonical agent guide for this repository is **[`AGENTS.md`](../AGENTS.md)**
at the repo root. Read it before generating or editing code.

Key rules (full detail in `AGENTS.md`):

- Honor the interface contracts in `docs/SSOT.md` §6 — never change a public
  signature without updating that doc.
- All paths, model names and hyperparameters come from `src/app/config.py` —
  no hardcoded values elsewhere.
- Use Conventional Commits; keep commits small and focused.
- Build the committed proposal scope before any above-and-beyond enhancement.
- Verify against the phase exit criteria in `docs/agent-prompts/` before
  calling work done.
