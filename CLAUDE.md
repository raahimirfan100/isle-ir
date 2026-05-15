# CLAUDE.md

@AGENTS.md

The full working guide is **`AGENTS.md`** (imported above) — read it first.
This file adds only Claude Code-specific notes.

## Claude Code specifics

- **Plan mode** — for any non-trivial change (a new module, a multi-file edit),
  plan before editing. Trivial fixes can go direct.
- **Slash commands** (`.claude/commands/`):
  - `/phase <N>` — load the phase-N prompt from `docs/agent-prompts/`.
  - `/contract-check` — audit the current code against the SSOT §6 contracts.
- **Subagents** — use the Explore subagent for codebase search to keep the main
  context lean; run independent research in parallel.
- **TodoWrite** — track multi-step work; mark each step done as you finish it.
- `.claude/settings.json` holds a shared permission allowlist for low-risk
  commands. Personal overrides go in `.claude/settings.local.json` (gitignored).

## Key files

[`docs/SSOT.md`](docs/SSOT.md) · [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)
· [`src/app/config.py`](src/app/config.py) ·
[`docs/agent-prompts/`](docs/agent-prompts/)
