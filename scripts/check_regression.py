from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass(frozen=True)
class Key:
    model: str
    engine: str
    device: str
    batch: int
    h: int
    w: int

@dataclass
class Measured:
    mean_ms: Optional[float]
    p99_ms: Optional[float]
    src: str

def load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_key(d: Dict[str, Any]) -> key:
    model_path = (d.get("model") or {}).get("path") or ""
    model = model_path.split("/")[-1] if model_path else "unknown"

    runtime = d.get("runtime") or {}
    extra = (runtime.get("extra") or {})
    engine = str(runtime.get("engine") or "unknown")
    device = str(runtime.get("device") or "unknown")

    batch = int(extra.get("batch") or 1)
    h = int(extra.get("height") or 0)
    w = int(extra.get("width") or 0)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid H/W in report: {path}")

    return Key(model=model, engine=engine, device=device, batch=batch, h=h, w=w)

def to_measured(d: Dict[str, Any], src: str) -> Measured:
    runtime = d.get("runtime") or {}
    lat = runtime.get("latency_ms") or {}
    maen_ms = lat.get("mean")
    p99_ms = lat.get("p99")
    return Measured(
        mean_ms=float(mean_ms) if mean_ms is not None else None,
        p99_ms=float(p99_ms) if p99_ms is not None else None,
        src=src,
    )

def load_baseline(path: str) -> Tuple[Dict[Key, Measured], float, float]:
    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)

    thr = b.get("thresholds") or {}
    p99_ratio = float(thr.get("p99_regression_ratio", 1.15))
    mean_ratio = float(thr.get("mean_regression_ratio", 1.20))

    table: Dict[Key, Measured] = {}
    for c in b.get("cases") or []:
        k = Key(
            model=str(c["model"]),
            engine=str(c["engine"]),
            device=str(c["device"]),
            batch=int(c["batch"]),
            h=int(c["h"]),
            w=int(c["w"]),
        )
        table[k] = Measured(
            mean_ms=float(c.get("mean_ms")) if c.get("mean_ms") is not None else None,
            p99_ms=float(c.get("p99_ms")) if c.get("p99_ms") is not None else None,
            src=path,
        )
    return table, p99_ratio, mean_ratio

def fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline json path")
    ap.add_argument("--reports", required=True, help='glob, e.g. "reports/*.json"')
    args = ap.parse_args()

    baseline, p99_thr, mean_thr = load_baseline(args.baseline)

    paths = sorted(glob.glob(args.reports))
    if not paths:
        print(f"[FAIL] no reports matched: {args.reports}", file=sys.stderr)
        return 2

    current: Dict[Key, Measured] = {}
    for p in paths:
        d = load_report(p)
        k = to_key(d)
        current[k] = to_measured(d, src=p)

    failed = False
    missing: List[Key] = []
    for k, b in baseline.items():
        c = current.get(k)
        if c is None:
            missing.append(k)
            continue

        # p99 check
        if b.p99_ms is not None and c.p99_ms is not None:
            ratio = c.p99_ms / b.p99_ms if b.p99_ms > 0 else 999.0
            if ratio > p99_thr:
                failed = True
                print(
                    f"[REGRESSION:p99] {k.model} {k.h}x{k.w} "
                    f"baseline={fmt(b.p99_ms)}ms current={fmt(c.p99_ms)}ms "
                    f"ratio={ratio:.3f} > {p99_thr:.2f} ({c.src})"
                )

        # mean check
        if b.mean_ms is not None and c.mean_ms is not None:
            ratio = c.mean_ms / b.mean_ms if b.mean_ms > 0 else 999.0
            if ratio > mean_thr:
                failed = True
                print(
                    f"[REGRESSION:mean] {k.model} {k.h}x{k.w} "
                    f"baseline={fmt(b.mean_ms)}ms current={fmt(c.mean_ms)}ms "
                    f"ratio={ratio:.3f} > {mean_thr:.2f} ({c.src})"
                )

    if missing:
        failed = True
        for k in missing:
            print(f"[MISSING] baseline case not found in reports: {k}", file=sys.stderr)

    if failed:
        print("[FAIL] performance regression detected", file=sys.stderr)
        return 1

    print("[OK] no regression vs baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())