from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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
    ts_iso: Optional[str]


def _load_one(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_int(x) -> Optional[int]:
    return int(x) if x is not None else None


def _to_float(x) -> Optional[float]:
    return float(x) if x is not None else None


def _parse_ts_iso(d: Dict[str, Any]) -> Optional[str]:
    ts = d.get("timestamp")
    if not ts:
        return None
    if isinstance(ts, str):
        return ts
    return None


def _ts_key(ts_iso: Optional[str]) -> Tuple[bool, datetime]:
    """
    sort key: (missing_ts, datetime)
    """
    if not ts_iso:
        return (True, datetime.fromtimestamp(0, tz=timezone.utc))
    # accept "2026-02-26T07:45:37Z"
    s = ts_iso.replace("Z", "+00:00")
    try:
        return (False, datetime.fromisoformat(s))
    except Exception:
        return (True, datetime.fromtimestamp(0, tz=timezone.utc))


def _to_row(path: str, d: Dict[str, Any]) -> Row:
    model_path = (d.get("model") or {}).get("path") or ""
    model_name = model_path.split("/")[-1] if model_path else "unknown"

    runtime = d.get("runtime") or {}
    extra = runtime.get("extra") or {}
    static = d.get("static") or {}
    latency = runtime.get("latency_ms") or {}

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
        ts_iso=_parse_ts_iso(d),
    )


def _fmt_int_commas(x: Optional[int]) -> str:
    return "-" if x is None else f"{x:,}"


def _fmt_f3(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"


def _hw(r: Row) -> str:
    return f"{r.h}x{r.w}" if (r.h and r.w) else "-"


def _md_table_latest(rows: List[Row]) -> str:
    """
    Latest-only table: one row per (model, engine, device, batch, h, w)
    """
    lines: List[str] = []
    lines.append("| Model | Engine | Device | Batch | Input(HxW) | FLOPs | Mean (ms) | P99 (ms) | Timestamp (UTC) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        lines.append(
            f"| {r.model} | {r.engine} | {r.device} | {r.batch or '-'} | {_hw(r)} | "
            f"{_fmt_int_commas(r.flops)} | {_fmt_f3(r.mean_ms)} | {_fmt_f3(r.p99_ms)} | {r.ts_iso or '-'} |"
        )
    return "\n".join(lines)


def _md_table_history(rows: List[Row]) -> str:
    """
    History table: all rows
    """
    lines: List[str] = []
    lines.append("| Model | Engine | Device | Batch | Input(HxW) | FLOPs | Mean (ms) | P99 (ms) | Timestamp (UTC) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        lines.append(
            f"| {r.model} | {r.engine} | {r.device} | {r.batch or '-'} | {_hw(r)} | "
            f"{_fmt_int_commas(r.flops)} | {_fmt_f3(r.mean_ms)} | {_fmt_f3(r.p99_ms)} | {r.ts_iso or '-'} |"
        )
    return "\n".join(lines)


def _group_key(r: Row) -> Tuple[str, str, str, Optional[int], Optional[int], Optional[int]]:
    return (r.model, r.engine, r.device, r.batch, r.h, r.w)


def _sort_rows(rows: List[Row], sort: str) -> List[Row]:
    key = {
        "p99": lambda r: (r.p99_ms is None, r.p99_ms),
        "mean": lambda r: (r.mean_ms is None, r.mean_ms),
        "flops": lambda r: (r.flops is None, r.flops),
        "time": lambda r: _ts_key(r.ts_iso),
    }.get(sort)

    if key is None:
        raise typer.BadParameter("--sort must be one of: p99, mean, flops, time")

    return sorted(rows, key=key)


def _latest_per_group(rows: List[Row]) -> List[Row]:
    """
    pick latest row per group by timestamp
    """
    best: Dict[Tuple[str, str, str, Optional[int], Optional[int], Optional[int]], Row] = {}
    for r in rows:
        k = _group_key(r)
        if k not in best:
            best[k] = r
            continue
        # compare timestamp
        if _ts_key(r.ts_iso) > _ts_key(best[k].ts_iso):
            best[k] = r
    return list(best.values())


def summarize(
    pattern: str = typer.Argument(..., help='예: reports/*.json'),
    format: str = typer.Option("md", "--format", help="md"),
    mode: str = typer.Option("latest", "--mode", help="latest/history (latest=중복 제거, history=전체)"),
    sort: str = typer.Option("p99", "--sort", help="p99/mean/flops/time"),
    recent: int = typer.Option(0, "--recent", help="0이면 전체, 아니면 최근 N개(시간 기준)"),
    top: int = typer.Option(0, "--top", help="0이면 전체, 아니면 상위 N개 (sort 기준)"),
    output: str = typer.Option("", "--output", "-o", help="출력 파일 경로(미지정 시 stdout)"),
):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise typer.BadParameter(f"no files matched: {pattern}")

    rows = [_to_row(p, _load_one(p)) for p in paths]

    if recent and recent > 0:
        rows = sorted(rows, key=lambda r: _ts_key(r.ts_iso))
        rows = rows[-recent:]  # newest N

    if mode not in ("latest", "history"):
        raise typer.BadParameter("--mode must be one of: latest, history")

    if format != "md":
        raise typer.BadParameter("--format currently supports only: md")

    if mode == "latest":
        rows = _latest_per_group(rows)

    rows = _sort_rows(rows, sort=sort)

    if top and top > 0:
        rows = rows[:top]

    if mode == "latest":
        text = "## Latest (recommended)\n\n" + _md_table_latest(rows) + "\n"
    else:
        text = "## History (raw)\n\n" + _md_table_history(rows) + "\n"

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
        rprint(f"[green]Saved[/green]: {output}")
    else:
        print(text, end="")