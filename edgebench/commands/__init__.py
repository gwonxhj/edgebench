from __future__ import annotations

import typer

from .analyze import app as analyze_app
from .profile import app as profile_app
from .summarize import app as summarize_app

app = typer.Typer(help="EdgeBench commands")

app.add_typer(analyze_app, name="analyze")
app.add_typer(profile_app, name="profile")
app.add_typer(summarize_app, name="summarize")