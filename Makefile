.PHONY: demo_deps demo_models demo_profile
demo_deps:
	pip install torch onnxscript

demo_models: demo_deps
	mkdir -p models
	python scripts/make_toy_model.py --height 224 --width 224 --out models/toy.onnx

demo_profile: demo_models
	edgebench profile models/toy.onnx --runs 300 --height 224 --width 224 --intra-threads 1 --inter-threads 1
	edgebench profile models/toy.onnx --runs 300 --height 320 --width 320 --intra-threads 1 --inter-threads 1
	edgebench profile models/toy.onnx --runs 300 --height 640 --width 640 --intra-threads 1 --inter-threads 1
