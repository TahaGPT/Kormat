# All commands run on the CPU laptop. Model-loading targets take a lock so only one big model runs at a time.
UV := uv run
LOCK := flock .model.lock

.PHONY: eval-all setup test test-slow data download-weights probe baseline train-smoke eval-val eval-test bench-kv bench-attn bench-spec figures app paper

setup:            ; uv sync
test:             ; $(UV) pytest -q
test-slow:        ; $(LOCK) $(UV) pytest -q -m slow
data:             ; $(UV) python -m kormat.data.build
download-weights: ; $(UV) python scripts/download_weights.py
probe:            ; $(LOCK) $(UV) python scripts/probe_cpu.py
baseline:         ; $(LOCK) $(UV) python -m kormat.eval.run --model base --split val --prefix-cache --out results/eval/val_base
train-smoke:      ; $(LOCK) $(UV) python -m kormat.finetune.train --config configs/train_smoke.yaml --mode smoke
eval-val:         ; $(LOCK) $(UV) python -m kormat.eval.run --model ft --split val --prefix-cache --out results/eval/val_ft
eval-test:        ; KORMAT_ALLOW_TEST=1 $(LOCK) $(UV) python -m kormat.eval.run --model both --split test --prefix-cache --out results/eval/test_final
eval-all:         ; ./scripts/run_full_eval.sh
bench-kv:         ; $(LOCK) $(UV) python scripts/bench_kv.py
bench-attn:       ; $(UV) python scripts/bench_attn.py
bench-spec:       ; $(LOCK) $(UV) python scripts/bench_spec.py
figures:          ; $(UV) python scripts/make_figures.py
app:              ; ./scripts/start_app.sh
paper:            ; cd report && (command -v latexmk >/dev/null && latexmk -pdf -interaction=nonstopmode main.tex || $(HOME)/.local/bin/tectonic -X compile main.tex)
