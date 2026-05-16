---
description: List in-code TODO markers grouped by module, to plan remaining work.
---

Run:

```bash
grep -rn "# TODO" src/ evaluation/ --include="*.py" | sort
```

Group the hits by module (ingestion, retrieval, topics, graph, app, eval) and
present a table: file · line · owner/phase · what.

The repo convention is `# TODO(<owner>, <phase>): <what>` — flag any TODO that
does not follow it. List `enhancement` TODOs separately: those are
above-and-beyond work, deferred until the committed scope is done.

Then point out which TODOs sit on the current phase's critical path (check
`docs/PROGRESS.md` for the active phase) and ask which to pick up next.
