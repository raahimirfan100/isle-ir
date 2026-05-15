---
description: Load a phase prompt and start that phase of the ISLE-IR build
---

Start phase $ARGUMENTS of the ISLE-IR build.

1. Read `AGENTS.md` (how to work here).
2. Read `docs/agent-prompts/phase-$ARGUMENTS-*.md` — the prompt for this phase.
3. Follow that prompt's steps. Honor the interface contracts it references.
4. Before reporting the phase done, verify every box in the prompt's
   exit-criteria checklist.

If the previous phase's exit criteria are not yet met, stop and tell me before
starting.
