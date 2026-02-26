#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


START = "<!-- EDGE_BENCH:START -->"
END = "<!-- EDGE_BENCH:END -->"


@dataclass(frozen=True)
class Key:
    model: str
    engine: str
    device: str
    batch: int
    h: int
    w: int


@dataclass
class Row:
    key: Key
    flops: Optional[int]
    mean_ms: Optional[float]
    p99_ms: Optional[float]
    ts: str  # ISO UTC "....Z"


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt_int_commas(x: Optional[int]) -> str:
    return "-" if x is None else f"{x:,}"


def _fmt_f3(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"


def _parse_row(d: Dict[str, Any]) -> Optional[Row]:
    model_path = (d.get("model") or {}).get("path") or ""
    model = model_path.split("/")[-1] if model_path else "unknown"

    runtime = d.get("runtime") or {}
    static = d.get("static") or {}
    extra = (runtime.get("extra") or {})
    lat = (runtime.get("latency_ms") or {})

    engine = str(runtime.get("engine") or "unknown")
    device = str(runtime.get("device") or "unknown")
    ts = str(d.get("timestamp") or "")

    batch = _to_int(extra.get("batch")) or 1
    h = _to_int(extra.get("height"))
    w = _to_int(extra.get("width"))
    if h is None or w is None:
        return None  # README는 해상도 없는 row는 제외

    key = Key(model=model, engine=engine, device=device, batch=batch, h=h, w=w)
    return Row(
        key=key,
        flops=_to_int(static.get("flops_estimate")),
        mean_ms=_to_float(lat.get("mean")),
        p99_ms=_to_float(lat.get("p99")),
        ts=ts,
    )


def _md_table_latest(latest: List[Row]) -> str:
    lines: List[str] = []
    lines.append("## Latest (deduplicated)")
    lines.append("")
    lines.append("| Model | Engine | Device | Batch | Input(HxW) | FLOPs | Mean (ms) | P99 (ms) | Timestamp (UTC) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in latest:
        k = r.key
        lines.append(
            f"| {k.model} | {k.engine} | {k.device} | {k.batch} | {k.h}x{k.w} | {_fmt_int_commas(r.flops)} | {_fmt_f3(r.mean_ms)} | {_fmt_f3(r.p99_ms)} | {r.ts} |"
        )
    return "\n".join(lines)


def _md_table_recent_stats(rows: List[Row], recent_n: int) -> str:
    """
    recent_n: 각 key별 최근 N개로 stats 요약
    """
    from collections import defaultdict

    by_key: Dict[Key, List[Row]] = defaultdict(list)
    for r in rows:
        by_key[r.key].append(r)

    lines: List[str] = []
    lines.append(f"## Recent stats (last {recent_n} runs per key)")
    lines.append("")
    lines.append("| Model | Engine | Device | Batch | Input(HxW) | N | Mean(avg) | P99(avg) | P99(max) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    keys_sorted = sorted(by_key.keys(), key=lambda k: (k.model, k.engine, k.device, k.batch, k.h, k.w))
    for k in keys_sorted:
        xs = sorted(by_key[k], key=lambda r: r.ts, reverse=True)[:recent_n]
        means = [r.mean_ms for r in xs if r.mean_ms is not None]
        p99s = [r.p99_ms for r in xs if r.p99_ms is not None]

        def avg(a: List[float]) -> Optional[float]:
            return None if not a else sum(a) / len(a)

        mean_avg = avg(means)
        p99_avg = avg(p99s)
        p99_max = None if not p99s else max(p99s)

        lines.append(
            f"| {k.model} | {k.engine} | {k.device} | {k.batch} | {k.h}x{k.w} | {len(xs)} | {_fmt_f3(mean_avg)} | {_fmt_f3(p99_avg)} | {_fmt_f3(p99_max)} |"
        )

    return "\n".join(lines)


def build_readme_block(bench_md_path: Path, recent_n: int) -> str:
    """
    BENCHMARKS.md(History raw)를 파싱해서 README용 요약 블록을 만든다.
    """
    # BENCHMARKS.md는 md지만, 우리는 원본 source로 reports/*.json을 권장.
    # 다만 지금 파이프라인 유지 위해: BENCHMARKS.md 옆에 reports 폴더가 있다고 가정하고 직접 json들을 읽는다.
    reports_dir = Path("reports")
    json_paths = sorted(reports_dir.glob("*.json"))
    if not json_paths:
        # fallback: bench_md에 아무것도 못 만들면 빈 블록
        return "## Latest (deduplicated)\n\n(no reports)\n"

    parsed: List[Row] = []
    for p in json_paths:
        try:
            d = _load_report(p)
            r = _parse_row(d)
            if r is not None:
                parsed.append(r)
        except Exception:
            continue

    # 최신순 정렬
    parsed.sort(key=lambda r: r.ts)

    # latest dedup
    latest_map: Dict[Key, Row] = {}
    for r in parsed:
        latest_map[r.key] = r
    latest = list(latest_map.values())
    latest.sort(key=lambda r: (r.key.model, r.key.engine, r.key.device, r.key.batch, r.key.h, r.key.w))

    # block 구성
    parts = []
    parts.append(_md_table_latest(latest))
    parts.append("")
    parts.append(_md_table_recent_stats(parsed, recent_n=recent_n))
    parts.append("")
    parts.append("> Full history: `BENCHMARKS.md`")
    return "\n".join(parts) + "\n"


def replace_block(readme_text: str, new_block: str) -> str:
    pattern = re.compile(rf"{re.escape(START)}.*?{re.escape(END)}", re.DOTALL)
    repl = f"{START}\n\n{new_block}\n{END}"
    if not pattern.search(readme_text):
        raise SystemExit(f"README does not contain markers: {START} / {END}")
    return pattern.sub(repl, readme_text, count=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", required=True)
    ap.add_argument("--bench", required=True, help="kept for compatibility (BENCHMARKS.md path)")
    ap.add_argument("--recent", type=int, default=5)
    args = ap.parse_args()

    readme_path = Path(args.readme)
    bench_path = Path(args.bench)

    new_block = build_readme_block(bench_path, recent_n=args.recent)
    updated = replace_block(readme_path.read_text(encoding="utf-8"), new_block)
    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated {readme_path} from reports/*.json (recent={args.recent}).")


if __name__ == "__main__":
    main()