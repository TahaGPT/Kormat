# All commands run on the CPU laptop. Model-loading targets take a lock so only one big model runs at a time.
UV := uv run
LOCK := flock .model.lock

.PHONY: setup test test-slow data download-weights probe baseline train-smoke eval-val eval-test bench-kv bench-attn bench-spec figures app paper

setup:            ; uv sync
test:             ; $(UV) pytest -q
test-slow:        ; $(LOCK) $(UV) pytest -q -m slow
data:             ; $(UV) python -m kormat.data.build
download-weights: ; $(UV) python scripts/download_weights.py
probe:            ; $(LOCK) $(UV) python scripts/probe_cpu.py
baseline:         ; $(LOCK) $(UV) python -m kormat.eval.run --model base --split val
train-smoke:      ; $(LOCK) $(UV) python -m kormat.finetune.train --config configs/train_smoke.yaml
eval-val:         ; $(LOCK) $(UV) python -m kormat.eval.run --model ft --split val
eval-test:        ; KORMAT_ALLOW_TEST=1 $(LOCK) $(UV) python -m kormat.eval.run --model both --split test
bench-kv:         ; $(LOCK) $(UV) python scripts/bench_kv.py
bench-attn:       ; $(UV) python scripts/bench_attn.py
bench-spec:       ; $(LOCK) $(UV) python scripts/bench_spec.py
figures:          ; $(UV) python scripts/make_figures.py
app:              ; ./scripts/start_app.sh
paper:            ; cd report && latexmk -pdf -interaction=nonstopmode main.tex
