from __future__ import annotations

import typer
from edgebench.commands import app as commands_app

app = typer.Typer(help="EdgeBench CLI - Edge AI Profiling Tool")
app.add_typer(commands_app)

if __name__ == "__main__":
    app()