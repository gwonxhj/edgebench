from __future__ import annotations

import os
import re
import subprocess
import sys


MARK_START = "<!-- EDGE_BENCH:START -->"
MARK_END = "<!-- EDGE_BENCH:END -->"


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit(p.returncode)
    # edgebench가 rich 출력/경고 섞을 수 있으니, stdout만 사용
    return p.stdout


def replace_marked_block(readme_text: str, block: str) -> str:
    pat = re.compile(
        re.escape(MARK_START) + r".*?" + re.escape(MARK_END),
        flags=re.DOTALL,
    )
    repl = MARK_START + "\n\n" + block.strip() + "\n\n" + MARK_END
    if not pat.search(readme_text):
        raise SystemExit("README.md marker not found. Add markers first.")
    return pat.sub(repl, readme_text)


def main():
    # 1) produce latest+stats markdown (for README marker)
    md_both = run(
        [
            "poetry",
            "run",
            "edgebench",
            "summarize",
            "reports/*.json",
            "--mode",
            "both",
            "--recent",
            os.environ.get("EDGE_BENCH_RECENT", "5"),
            "--sort",
            "p99",
        ]
    )

    # 2) produce "history" markdown (for BENCHMARKS.md)
    md_history = run(
        [
            "poetry",
            "run",
            "edgebench",
            "summarize",
            "reports/*.json",
            "--mode",
            "history",
            "--sort",
            "p99",
        ]
    )

    # Write BENCHMARKS.md (full overwrite)
    with open("BENCHMARKS.md", "w", encoding="utf-8") as f:
        f.write(md_history)

    # Patch README markers
    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()

    readme2 = replace_marked_block(readme, md_both)

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme2)

    print("✅ Updated README.md (marker block) + BENCHMARKS.md")


if __name__ == "__main__":
    main()