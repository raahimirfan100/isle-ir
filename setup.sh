#!/usr/bin/env bash
# ISLE-IR — development environment setup.
#
# Creates a uv-managed virtualenv (.venv, Python 3.11 from .python-version)
# and installs the project with the right hardware extra:
#   gpu  -> torch (CUDA 12.6) + faiss-gpu-cu12   (RTX rig, Google Colab)
#   cpu  -> torch (CPU / Apple MPS) + faiss-cpu  (M4 Mac, Intel laptop)
#
# Usage:
#   ./setup.sh          # auto-detect: nvidia-smi present -> gpu, else cpu
#   ./setup.sh gpu      # force GPU mode
#   ./setup.sh cpu      # force CPU mode
set -euo pipefail

cd "$(dirname "$0")"

# ── 1. uv must be installed ──────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed. Install it, then re-run this script:"
  echo "  macOS / Linux / WSL : curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  Windows PowerShell  : irm https://astral.sh/uv/install.ps1 | iex"
  exit 1
fi

# ── 2. Decide GPU vs CPU ─────────────────────────────────────────────────
MODE="${1:-auto}"
if [ "$MODE" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    MODE="gpu"
    echo "==> NVIDIA GPU detected — using GPU mode."
  else
    MODE="cpu"
    echo "==> No NVIDIA GPU detected — using CPU mode."
  fi
fi

case "$MODE" in
  gpu) EXTRA="gpu" ;;
  cpu) EXTRA="cpu" ;;
  *)   echo "ERROR: unknown mode '$MODE'. Use: ./setup.sh [gpu|cpu]"; exit 1 ;;
esac

# ── 3. Sync the environment ──────────────────────────────────────────────
echo "==> Installing ISLE-IR ($EXTRA) into .venv  —  uv sync --extra $EXTRA"
uv sync --extra "$EXTRA"

# ── 4. Done ──────────────────────────────────────────────────────────────
cat <<'EOF'

==> Environment ready.

   Activate    : source .venv/bin/activate        # Windows: .venv\Scripts\activate
   Run the app : uv run streamlit run src/app/main.py
   Verify      : uv run python -c "import app.config, retrieval.rrf, graph.builder; print('imports OK')"
EOF

if [ "$EXTRA" = "gpu" ]; then
  echo '   GPU check   : uv run python -c "import torch, faiss; print('"'"'CUDA'"'"', torch.cuda.is_available(), '"'"'| FAISS GPUs'"'"', faiss.get_num_gpus())"'
fi
