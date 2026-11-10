#!/usr/bin/env bash
# Full before/after evaluation on CPU, chunked so other jobs can take the model lock between chunks.
# Order (deadline-driven; config frozen, nothing tuned on val): ft-test, base-test, ft-val, base-val. Resumable: re-running continues where it stopped.
set -u
cd "$(dirname "$0")/.."
CHUNK=${CHUNK:-40}
run() {  # model split outdir
  local model=$1 split=$2 out=$3 allow=0
  [ "$split" = test ] && allow=1
  while true; do
    KORMAT_ALLOW_TEST=$allow flock .model.lock uv run python -m kormat.eval.run \
      --model "$model" --split "$split" --prefix-cache --chunk "$CHUNK" --out "$out" || { echo "FAILED $model $split"; return 1; }
    if uv run python -c "import json,sys;sys.exit(0 if json.load(open('$out/metrics.json')).get('complete') else 1)"; then
      echo "DONE $model $split -> $out"; return 0; fi
    echo "chunk done $model $split ($(wc -l < "$out/predictions.jsonl") rows)"
  done
}
run ft   test results/eval/test_final/ft
run base test results/eval/test_final/base
run ft   val  results/eval/val_ft
run base val  results/eval/val_base
