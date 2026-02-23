import typer

app = typer.Typer(help="EdgeBench CLI - Edge AI Profiling Tool")

@app.command()
def analyze(model_path: str):
    """
    Analyze ONNX model structure.
    """
    print(f"Analyzing model: {model_path}")

@app.command()
def profile(model_path: str):
    """
    Profile ONNX model inference latency.
    """
    print(f"Profiling model: {model_path}")

if __name__ == "__main__":
    app()
