from __future__ import annotations

import argparse
from pathlib import Path

START = "<!-- EDGE_BENCH:START -->"
END = "<!-- EDGE_BENCH:END -->"

def update_block(readme_text: str, new_block: str) -> str:
    si = readme_text.find(START)
    ei = readme_text.find(END)
    if si == -1 or ei == -1 or ei < si:
        raise SystemExit(
            f"README markers not found or invalid. Ensure README contains:\n{START}\n...\n{END}\n"
        )
    
    before = readme_text[: si + len(START)]
    after = readme_text[ei:]
    
    return before + "\n\n" + new_block.strip() + "\n\n" + after

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--bench", default="BENCHMARKS.md")
    args = ap.parse_args()

    readme_path = Path(args.readme)
    bench_path = Path(args.bench)

    readme_text = readme_path.read_text(encoding="utf-8")
    bench_text = bench_path.read_text(encoding="utf-8")

    out = update_block(readme_text, bench_text)
    readme_path.write_text(out, encoding="utf-8")
    print(f"Updated {readme_path} from {bench_path}.")

if __name__ == "__main__":
    main()