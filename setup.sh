#!/usr/bin/env bash
# ISLE-IR — development environment setup.
#
# Creates a uv-managed virtualenv (.venv, Python 3.11 from .python-version)
# and installs the project with the right hardware extra:
#   gpu  -> torch (CUDA 12.6) + faiss-gpu-cu12   (RTX rig, Google Colab)
#   cpu  -> torch (CPU / Apple MPS) + faiss-cpu  (M4 Mac, Intel laptop)
#
# Usage:
#   ./setup.sh          # auto-detect: Linux/WSL + NVIDIA GPU -> gpu, else cpu
#   ./setup.sh gpu      # force GPU mode (Linux/WSL only)
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
# The `gpu` extra (faiss-gpu-cu12) ships Linux x86-64 wheels only, so GPU
# mode is supported on Linux / WSL only — never on Windows or macOS, even if
# an NVIDIA GPU and nvidia-smi are present.
case "$(uname -s)" in
  Linux*) IS_LINUX=1 ;;
  *)      IS_LINUX=0 ;;
esac

MODE="${1:-auto}"
if [ "$MODE" = "auto" ]; then
  if [ "$IS_LINUX" -eq 1 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    MODE="gpu"
    echo "==> NVIDIA GPU on Linux/WSL detected — using GPU mode."
  else
    MODE="cpu"
    echo "==> Using CPU mode (GPU mode needs Linux/WSL with an NVIDIA GPU)."
  fi
fi

case "$MODE" in
  gpu)
    if [ "$IS_LINUX" -ne 1 ]; then
      echo "ERROR: GPU mode is supported on Linux / WSL only — faiss-gpu-cu12"
      echo "       has no Windows or macOS wheels. Re-run with: ./setup.sh cpu"
      exit 1
    fi
    EXTRA="gpu" ;;
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
