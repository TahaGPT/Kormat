#!/usr/bin/env bash
# One-command start: ./scripts/start_app.sh   (KORMAT_FAKE_ENGINE=1 for a model-free demo; PORT=8000 by default)
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
if [ "${KORMAT_FAKE_ENGINE:-0}" != "1" ]; then
  if [ ! -f checkpoints/adapter/adapter_config.json ] || ! ls ~/.cache/huggingface/hub 2>/dev/null | grep -q "Qwen--Qwen2.5-1.5B-Instruct"; then
    echo "[kormat] weights missing -> downloading (one time)..."
    uv run python scripts/download_weights.py
  fi
  echo "[kormat] real models load in the background (~1 min); /api/health shows progress."
else
  echo "[kormat] FAKE ENGINE: deterministic demo results, no model loaded."
fi
echo "[kormat] open http://localhost:${PORT}"
exec uv run uvicorn kormat.app.server:app --host 0.0.0.0 --port "$PORT"
