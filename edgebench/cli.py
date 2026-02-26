from __future__ import annotations

import typer

from edgebench.commands.analyze import analyze_cmd
from edgebench.commands.profile import profile_cmd
from edgebench.commands.summarize import summarize

app = typer.Typer(help="EdgeBench CLI - Edge AI Profiling Tool")

app.command("analyze", help="Static analysis (params/IO/FLOPs)")(analyze_cmd)
app.command("profile", help="Runtime profiling (onnxruntime cpu)")(profile_cmd)
app.command("summarize", help="Summarize EdgeBench JSON reports")(summarize)

if __name__ == "__main__":
    app()