.PHONY: demo_deps demo_models demo_profile

SIZES ?= 224 320 640
RUNS  ?= 300
BATCH ?= 1
WARMUP ?= 10
INTRA ?= 1
INTER ?= 1

save:
	git add -A
	git commit -m "Auto-save $$(data -u +%Y-%m-%dT%H:%M:%SZ)" || true
	git push

demo_deps:
	poetry install
	poetry run python -m pip install -U pip
	poetry run python -m pip install torch onnxscript

demo_models:
	mkdir -p models

demo_profile: demo_deps demo_models
	mkdir -p reports
	@set -e; \
	for s in $(SIZES); do \
		echo "==> Generating toy model: $${s}x$${s}"; \
		poetry run python scripts/make_toy_model.py --height $$s --width $$s --out models/toy$${s}.onnx; \
		echo "==> Profiling: models/toy$${s}.onnx"; \
		poetry run edgebench profile models/toy$${s}.onnx \
			--warmup $(WARMUP) \
			--runs $(RUNS) \
			--batch $(BATCH) \
			--intra-threads $(INTRA) \
			--inter-threads $(INTER) \
			-o reports/toy$${s}__onnxruntime_cpu__b$(BATCH)__r$(RUNS)__t$(INTRA)x$(INTER).json; \
	done
	@echo "Done. Reports saved under ./reports/"