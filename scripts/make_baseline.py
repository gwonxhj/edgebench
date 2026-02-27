from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict

from scripts.check_regression import _parse_report, _key_to_str  # reuse parser


def main():
    reports = os.environ.get("REPORTS_GLOB", "reports/*.json")
    out = os.environ.get("BASELINE_OUT", "benchmarks/baseline_codespaces_cpu.json")

    paths = sorted(glob.glob(reports))
    if not paths:
        raise SystemExit(f"no reports matched: {reports}")

    baseline: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        k, st = _parse_report(p)
        baseline[_key_to_str(k)] = {
            "mean_ms": st.mean_ms,
            "p99_ms": st.p99_ms,
            "flops": st.flops,
            "timestamp": st.ts,
        }

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Saved baseline: {out} (keys={len(baseline)})")


if __name__ == "__main__":
    main()