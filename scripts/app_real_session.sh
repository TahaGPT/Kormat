#!/usr/bin/env bash
# Locked real-model session: smoke test, then uvicorn on :8020 + screenshots. Run under flock .model.lock.
cd "$(dirname "$0")/.."
uv run python scripts/app_real_smoke.py
uv run uvicorn kormat.app.server:app --port 8020 > /tmp/claude-1000/uvicorn8020.log 2>&1 &
UV=$!
trap 'kill $UV 2>/dev/null; wait $UV 2>/dev/null' EXIT
until curl -s localhost:8020/api/health | grep -q '"status":"ready"'; do sleep 3; done
uv run --with playwright python scripts/app_screenshots.py http://localhost:8020 app_real
