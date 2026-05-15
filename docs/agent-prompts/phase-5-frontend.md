# Phase 5 — Streamlit Frontend

> Copy everything below the divider into your agent. Owner: Mujtaba (layout);
> Abubakar and Mujtaba own their tab components; Raahim reviews integration.

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then do **Phase 5:
Streamlit Frontend**.

## Read first
- `docs/IMPLEMENTATION.md` §5 — the reference Streamlit app and caching model.
- `docs/SSOT.md` §6 (all contracts — the UI orchestrates them).

## Goal
A working Streamlit app that runs a query end to end and presents the results,
topics, knowledge graph and analytics in tabs.

## Deliverables
- `src/app/main.py` — layout, sidebar, cached resource loading, tab routing.
- `src/app/components/results_tab.py`, `topics_tab.py`, `graph_tab.py`,
  `analytics_tab.py` — one `render()` per tab.

## Steps
1. Build `main.py`: sidebar inputs, `@st.cache_resource` for indexes,
   `@st.cache_data` for the corpus, tab routing into the components.
2. Implement each tab's `render()`; tabs call the Phase 2–4 modules behind
   their contracts.
3. Add session-state handling so results persist across tab switches without
   re-running retrieval.
4. **Test in a browser.** Run `streamlit run src/app/main.py`, execute a query,
   click through every tab, check edge cases (empty query, no results).

## Exit criteria — verify every box before reporting done
- [ ] `streamlit run src/app/main.py` launches with no errors.
- [ ] A query runs end to end; Results, Topics, Knowledge Graph and Analytics
      tabs all display real content.
- [ ] Indexes/corpus are cached (no reload on every interaction).
- [ ] Edge cases handled gracefully (empty query, zero results).
- [ ] `ruff check .` clean.

## Reminder
A clean import and passing lint do not prove a UI works — it is only "done"
after you have exercised it in a browser (see `AGENTS.md` → Verification).

## Commit & branch
Branches `feat/streamlit-ui` (and tab branches) off `dev`. Small commits:
`feat(app): add main layout and caching`, `feat(app): add results tab`.
