# Phase 0 — Environment & Project Setup

> Copy everything below the divider into your agent. Owners: all three
> teammates, each on their own machine.

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then do **Phase 0:
Environment & Project Setup**.

## Read first
- `AGENTS.md` and `CONTRIBUTING.md` — how to work here.
- `README.md` Quickstart and `docs/IMPLEMENTATION.md` §0 — environment setup.
- `docs/SSOT.md` §5 (config) and §6 (contracts).
- The `docs/IMPLEMENTATION.md` header — the embedding-model decision.

## Goal
Get a working development environment, confirm the scaffold imports cleanly,
and **confirm the embedding-model team decision**.

## Steps
1. Install **uv** (the project's package manager):
   `curl -LsSf https://astral.sh/uv/install.sh | sh`.
2. From the repo root, run `./setup.sh`. It detects your hardware and runs
   `uv sync` with the right extra — `gpu` on Linux/WSL with an NVIDIA GPU,
   `cpu` everywhere else (Mac, Intel, Colab). uv creates the `.venv`, installs
   the locked dependencies, and installs the project itself editable — there
   is no separate `pip install -e .`.
3. Verify imports:
   `uv run python -c "import app.config, retrieval.rrf, graph.builder"`.
4. On the RTX rig only, verify the GPU: `torch.cuda.is_available()` is `True`
   and `faiss.get_num_gpus()` is `1`. CPU machines skip this step.
5. **Verify the embedding model.** It is decided — SPECTER2
   (`allenai/specter2_base`, 768-dim), already set in `src/app/config.py`.
   Confirm it loads (needs the `adapters` library, installed by `uv sync`) and
   yields 768-dim vectors.
6. Register a free OpenAlex API key (https://openalex.org/api-key) and set
   `OPENALEX_EMAIL` / `OPENALEX_API_KEY` in your environment.

## Exit criteria — verify every box before reporting done
- [ ] `./setup.sh` (or `uv sync`) succeeds; `uv run python -c "import app.config"`
      works from any directory.
- [ ] `uv run ruff check .` is clean.
- [ ] GPU verified on the RTX rig (CUDA + FAISS see the GPU); CPU machines
      confirm `uv sync --extra cpu` installed `torch` + `faiss-cpu`.
- [ ] SPECTER2 (`allenai/specter2_base`) loads and yields 768-dim embeddings.
- [ ] OpenAlex key registered and set in the environment.
- [ ] `dev` branch exists; `main` and `dev` are protected (already configured
      by the repo admin — nothing to do here).

## Commit & branch
Environment setup is per-machine and mostly local. Any config/doc tweak goes in
small Conventional Commits, e.g. `docs: reconcile embedding model`.
