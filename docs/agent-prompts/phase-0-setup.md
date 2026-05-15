# Phase 0 — Environment & Project Setup

> Copy everything below the divider into your agent. Owners: all three
> teammates, each on their own machine.

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then do **Phase 0:
Environment & Project Setup**.

## Read first
- `AGENTS.md` and `CONTRIBUTING.md` — how to work here.
- `docs/IMPLEMENTATION.md` §0 — environment setup.
- `docs/SSOT.md` §5 (config) and §6 (contracts).
- The `docs/IMPLEMENTATION.md` header — the embedding-model mismatch.

## Goal
Get a working development environment, confirm the scaffold imports cleanly,
and **lock the embedding-model team decision**.

## Steps
1. Create the Python 3.11 conda env; install PyTorch (CUDA wheel on the RTX
   rig, CPU wheel elsewhere), then `pip install -r requirements.txt` and
   `pip install -e .`.
2. Verify imports: `python -c "import app.config, retrieval.rrf, graph.builder"`.
3. On the GPU machine, verify CUDA: `torch.cuda.is_available()` and
   `faiss.get_num_gpus()`.
4. **Team decision — embedding model.** Discuss MiniLM/384 vs SPECTER2/768
   (trade-offs in the `docs/IMPLEMENTATION.md` header). Set `EMBEDDING_MODEL`
   and `EMBEDDING_DIM` in `src/app/config.py` accordingly, and update whichever
   planning doc still disagrees so the repo is internally consistent.
5. Register a free OpenAlex API key (https://openalex.org/api-key) and set
   `OPENALEX_EMAIL` / `OPENALEX_API_KEY` in your environment.
6. Create the `dev` branch off `main` if it does not exist; configure branch
   protection on `main` and `dev` (1 review required) on GitHub.

## Exit criteria — verify every box before reporting done
- [ ] `pip install -e .` succeeds; `import app.config` works from any directory.
- [ ] `ruff check .` is clean.
- [ ] GPU verified on the RTX rig (CUDA + FAISS see the GPU).
- [ ] `EMBEDDING_MODEL` / `EMBEDDING_DIM` decided and committed in config.py;
      no planning doc contradicts it.
- [ ] OpenAlex key registered and set in the environment.
- [ ] `dev` branch exists; `main` and `dev` are protected.

## Commit & branch
Conventional Commits, small logical commits. Config/doc changes:
`chore(app): set embedding model` / `docs: reconcile embedding model`.
