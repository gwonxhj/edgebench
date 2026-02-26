from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import typer
from rich import print as rprint

app = typer.Typer(help="Summarize EdgeBench JSON reports")

@dataclass
class Row:
    path: str
    ts: Optional[datetime]

    model: str
    engine: str
    device: str

    h: Optional[int]
    w: Optional[int]
    batch: Optional[int]

    flops: Optional[int]
    mean_ms: Optional[float]
    p99_ms: Optional[float]

@dataclass
class StatsRow:
    key: str
    n: int
    flops: Optional[int]
    mean_mean_ms: Optional[float]
    mean_p99_ms: Optional[float]
    best_mean_ms: Optional[float]
    worst_mean_ms: Optional[float]
    best_p99_ms: Optional[float]
    worst_p99_ms: Optional[float]

# -------------------------
# Parsing
# -------------------------

def _load_one(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _parse_ts(d: Dict[str, Any], path: str) -> Optional[datetime]:
    """
    Prefer report.timestamp
    fallback: file mtime
    """
    ts = d.get("timestamp")
    if isinstance(ts, str) and ts:
        try:
            if ts.endswith("Z"):
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(ts)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    except Exception:
        return None

def _to_row(path: str, d: Dict[str, Any]) -> Row:
    model_path = (d.get("model") or {}).get("path") or ""
    model_name = model_path.split("/")[-1] if model_path else "unknown"

    runtime = d.get("runtime") or {}
    extra = runtime.get("extra") or {}
    static = d.get("static") or {}
    latency = runtime.get("latency_ms") or {}

    def _to_int(x) -> Optional[int]:
        try:
            return int(x) if x is not None else None
        except Exception:
            return None

    def _to_float(x) -> Optional[float]:
        try:
            return float(x) if x is not None else None
        except Exception:
            return None
        
    ts = _parse_ts(d, path)

    return Row(
        path=path,
        ts=ts,
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


# -------------------------
# Grouping / Stats
# -------------------------
def _key(r: Row) -> Tuple[str, str, str, int, int, int]:
    # normalize None -> 0 for stable grouping
    b = r.batch or 0
    h = r.h or 0
    w = r.w or 0
    return (r.model, r.engine, r.device, b, h, w)


def _key_str(r: Row) -> str:
    b = r.batch or 0
    h = r.h or 0
    w = r.w or 0
    return f"{r.model} | {r.engine} | {r.device} | b{b} | {h}x{w}"


def _group(rows: List[Row]) -> Dict[Tuple[str, str, str, int, int, int], List[Row]]:
    g: Dict[Tuple[str, str, str, int, int, int], List[Row]] = {}
    for r in rows:
        g.setdefault(_key(r), []).append(r)

    # sort each group by timestamp asc (None last)
    for k in g:
        g[k].sort(key=lambda x: (x.ts is None, x.ts))
    return g


def _latest_of(group: List[Row]) -> Row:
    # group already sorted asc; take last non-None ts if possible
    # but if all None, last is fine
    return group[-1]


def _calc_stats(key_str: str, group: List[Row], recent: int) -> StatsRow:
    # take last N rows (most recent)
    use = group[-recent:] if recent > 0 else group[:]
    n = len(use)

    flops = None
    for r in reversed(use):
        if r.flops is not None:
            flops = r.flops
            break

    mean_vals = [r.mean_ms for r in use if r.mean_ms is not None]
    p99_vals = [r.p99_ms for r in use if r.p99_ms is not None]

    def _avg(xs: List[float]) -> Optional[float]:
        return (sum(xs) / len(xs)) if xs else None

    def _min(xs: List[float]) -> Optional[float]:
        return min(xs) if xs else None

    def _max(xs: List[float]) -> Optional[float]:
        return max(xs) if xs else None

    return StatsRow(
        key=key_str,
        n=n,
        flops=flops,
        mean_mean_ms=_avg(mean_vals),
        mean_p99_ms=_avg(p99_vals),
        best_mean_ms=_min(mean_vals),
        worst_mean_ms=_max(mean_vals),
        best_p99_ms=_min(p99_vals),
        worst_p99_ms=_max(p99_vals),
    )


# -------------------------
# Formatting
# -------------------------
def _f3(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"


def _fint(x: Optional[int]) -> str:
    return "-" if x is None else f"{x:,}"


def _fhw(h: Optional[int], w: Optional[int]) -> str:
    if h and w:
        return f"{h}x{w}"
    return "-"


def _md_latest_table(rows: List[Row]) -> str:
    lines: List[str] = []
    lines.append("| Model | Engine | Device | Batch | Input(HxW) | FLOPs | Mean (ms) | P99 (ms) | Timestamp (UTC) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        ts = r.ts.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z" if r.ts else "-"
        lines.append(
            f"| {r.model} | {r.engine} | {r.device} | {r.batch or '-'} | {_fhw(r.h, r.w)} | {_fint(r.flops)} | {_f3(r.mean_ms)} | {_f3(r.p99_ms)} | {ts} |"
        )
    return "\n".join(lines)


def _md_stats_table(rows: List[StatsRow]) -> str:
    lines: List[str] = []
    lines.append("| Key | N | FLOPs | Mean(mean) | Mean(p99) | Best(mean) | Worst(mean) | Best(p99) | Worst(p99) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.key} | {r.n} | {_fint(r.flops)} | {_f3(r.mean_mean_ms)} | {_f3(r.mean_p99_ms)} | {_f3(r.best_mean_ms)} | {_f3(r.worst_mean_ms)} | {_f3(r.best_p99_ms)} | {_f3(r.worst_p99_ms)} |"
        )
    return "\n".join(lines)


# -------------------------
# CLI
# -------------------------
@app.command("main")
def summarize(
    pattern: str = typer.Argument(..., help='예: reports/*.json'),
    mode: str = typer.Option("both", "--mode", help="latest/stats/both/history"),
    recent: int = typer.Option(5, "--recent", help="stats에서 최근 N개만 사용 (0이면 전체)"),
    sort: str = typer.Option("p99", "--sort", help="p99/mean/flops (latest 테이블 정렬 기준)"),
    top: int = typer.Option(0, "--top", help="0이면 전체, 아니면 상위 N개 (latest 테이블만 적용)"),
    output: str = typer.Option("", "--output", "-o", help="출력 파일 경로(미지정 시 stdout)"),
):
    """
    reports/*.json을 모아서 Markdown 표를 출력합니다.

    - mode=latest: 조합별 최신 1개만
    - mode=stats: 조합별 최근 N개 통계
    - mode=both: latest + stats
    - mode=history: 전체 히스토리(원하면 BENCHMARKS.md 같은 로그용)
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise typer.BadParameter(f"no files matched: {pattern}")

    rows = [_to_row(p, _load_one(p)) for p in paths]
    g = _group(rows)

    # Build "latest per key"
    latest_rows: List[Row] = [_latest_of(gr) for gr in g.values()]

    # Sorting for latest table
    sort_key = {
        "p99": lambda r: (r.p99_ms is None, r.p99_ms),
        "mean": lambda r: (r.mean_ms is None, r.mean_ms),
        "flops": lambda r: (r.flops is None, r.flops),
    }.get(sort)

    if sort_key is None:
        raise typer.BadParameter("--sort must be one of: p99, mean, flops")

    latest_rows.sort(key=sort_key)
    if top and top > 0:
        latest_rows = latest_rows[:top]

    # Stats rows
    stats_rows: List[StatsRow] = []
    for gr in g.values():
        stats_rows.append(_calc_stats(_key_str(gr[-1]), gr, recent=recent))
    # make stats sort by worst p99 (bigger is worse) descending? -> you want "기술적으로 인정"이면 worst 먼저 보는게 좋음
    stats_rows.sort(key=lambda x: (x.worst_p99_ms is None, -(x.worst_p99_ms or 0.0)))

    # History rows (raw)
    history_rows = sorted(
        rows,
        key=lambda r: (r.ts is None, r.ts),
    )

    chunks: List[str] = []

    mode = mode.lower().strip()
    if mode not in ("latest", "stats", "both", "history"):
        raise typer.BadParameter("--mode must be one of: latest, stats, both, history")

    if mode in ("latest", "both"):
        chunks.append("## Latest (per model/engine/device/batch/HxW)\n")
        chunks.append(_md_latest_table(latest_rows))
        chunks.append("")

    if mode in ("stats", "both"):
        chunks.append(f"## Recent Stats (per key, last {recent if recent > 0 else 'ALL'} runs)\n")
        chunks.append(_md_stats_table(stats_rows))
        chunks.append("")

    if mode == ("history"):
        chunks.append("## History (raw)\n")
        chunks.append(_md_latest_table(history_rows))
        chunks.append("")

    text = "\n".join(chunks).rstrip() + "\n"

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
        rprint(f"[green]Saved[/green]: {output}")
    else:
        print(text, end="")