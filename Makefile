.PHONY: save demo_deps demo_models demo_profile demo demo_summary demo_doc demo_clean

SIZES  ?= 224 320 640
RUNS   ?= 300
BATCH  ?= 1
WARMUP ?= 10
INTRA  ?= 1
INTER  ?= 1
RECENT ?= 5

BENCH_DOC ?= BENCHMARKS.md

# -------------------------
# Save & push (timestamped)
# -------------------------
save:
	git add -A
	git commit -m "Auto-save $$(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
	git push

# -------------------------
# Dependencies (Poetry env)
# -------------------------
demo_deps:
	poetry install
	poetry run python -m pip install -U pip
	poetry run python -m pip install torch onnxscript

# -------------------------
# Generate toy models
# -------------------------
demo_models: demo_deps
	mkdir -p models
	@set -e; \
	for s in $(SIZES); do \
		echo "==> Generating toy model: $${s}x$${s}"; \
		poetry run python scripts/make_toy_model.py --height $$s --width $$s --out models/toy$${s}.onnx; \
	done

# -------------------------
# Profile toy models (timestamped reports)
# -------------------------
demo_profile: demo_models
	mkdir -p reports
	@set -e; \
	ts=$$(date -u +%Y%m%d-%H%M%S); \
	for s in $(SIZES); do \
		echo "==> Profiling: models/toy$${s}.onnx"; \
		poetry run edgebench profile models/toy$${s}.onnx \
			--warmup $(WARMUP) \
			--runs $(RUNS) \
			--batch $(BATCH) \
			--height $$s --width $$s \
			--intra-threads $(INTRA) \
			--inter-threads $(INTER) \
			-o reports/toy$${s}__onnxruntime_cpu__b$(BATCH)__h$${s}w$${s}__r$(RUNS)__t$(INTRA)x$(INTER)__$${ts}.json; \
	done
	@echo "Done. Reports saved under ./reports/"

# -------------------------
# Summarize reports (stdout)
# -------------------------
demo_summary:
	@echo "==> Summarizing reports"
	poetry run edgebench summarize "reports/*.json" --format md --sort p99

# -------------------------
# Write summary to markdown file
# -------------------------
demo_doc:
	@echo "==> Writing benchmark doc: $(BENCH_DOC)"
	poetry run edgebench summarize "reports/*.json" --format md --sort p99 -o $(BENCH_DOC)

# -------------------------
# Update README Benchmarks block
# -------------------------
demo_readme: demo_doc
	@echo "==> Updating README.md Benchmarks block"
	poetry run python scripts/update_readme.py --readme README.md --bench $(BENCH_DOC)

# -------------------------
# One-shot demo
# -------------------------
demo: demo_profile demo_readme
	@echo "✅ demo complete (saved $(BENCH_DOC) + updated README)"
# -------------------------
# Clean generated artifacts
# -------------------------
demo_clean:
	rm -rf models reports