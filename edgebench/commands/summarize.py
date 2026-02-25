from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import typer
from rich import print as rprint

@dataclass
class Row:
    path: str
    model: str
    engine: str
    device: str
    h: Optional[int]
    w: Optional[int]
    batch: Optional[int]
    flops: Optional[int]
    mean_ms: Optional[float]
    p99_ms: Optional[float]

def _load_one(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_row(path: str, d: Dict[str, Any]) -> Row:
    model_path = (d.get("model") or {}).get("path") or ""
    model_name = model_path.split("/")[-1] if model_path else "unknown"

    runtime = d.get("runtime") or {}
    extra = runtime.get("extra") or {}
    static = d.get("static") or {}
    latency = runtime.get("latency_ms") or {}

    def _to_int(x) -> Optional[int]:
        return int(x) if x is not None else None

    def _to_float(x) -> Optional[float]:
        return float(x) if x is not None else None

    return Row(
        path=path,
        model=model_name,
        engine=str(runtime.get("engine") or "unknown"),
        device=str(runtime.get("device") or "unknown"),
        h=_to_int(extra.get("height")),
        w=_to_int(extra.get("width")),
        batch=_to_int(extra.get("batch")),
        flops=_to_int(static.get("flops_estimate")),
        mean_ms=_to_float(latency.get("mean")),
        p99_ms=_to_float(latency.get("p99")),
    )

def _md_table(rows: List[Row]) -> str:
    lines = []

    def _f3(x):
        return "-" if x is None else f"{x:.3f}"

    lines.append("| Model | Engine | Device | Input(HxW) | FLOPs | Mean (ms) | P99 (ms) |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        hw = f"{r.h}x{r.w}" if (r.h and r.w) else "-"
        lines.append(
            f"| {r.model} | {r.engine} | {r.device} | {hw} | {r.flops or '-'} | {_f3(r.mean_ms)} | {_f3(r.p99_ms)} |"
        )
    return "\n".join(lines)

def summarize(
    pattern: str = typer.Argument(..., help='예: reports/*.json'),
    format: str = typer.Option("md", "--format", help="md"),
    sort: str = typer.Option("p99", "--sort", help="p99/mean/flops"),
    top: int = typer.Option(0, "--top", help="0이면 전체, 아니면 상위 N개"),
):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise typer.BadParameter(f"no files matched: {pattern}")

    rows = [_to_row(p, _load_one(p)) for p in paths]

    key = {
        "p99": lambda r: (r.p99_ms is None, r.p99_ms),
        "mean": lambda r: (r.mean_ms is None, r.mean_ms),
        "flops": lambda r: (r.flops is None, r.flops),
    }.get(sort)

    if key is None:
        raise typer.BadParameter("--sort must be one of: p99, mean, flops")

    rows.sort(key=key)

    if top and top > 0:
        rows = rows[:top]

    if format != "md":
        raise typer.BadParameter("--format currently supports only: md")

    print(_md_table(rows))