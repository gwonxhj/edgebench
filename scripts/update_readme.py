from __future__ import annotations

import argparse
import re
from pathlib import Path


START = "<!-- EDGE_BENCH:START -->"
END = "<!-- EDGE_BENCH:END -->"


def extract_latest_block(bench_text: str) -> str:
    """
    Return markdown for the 'Latest (recommended)' section only.
    """
    m = re.search(r"^## Latest \(recommended\)\s*\n\n(.*?)(?=^\#\#\s|\Z)", bench_text, flags=re.M | re.S)
    if not m:
        raise RuntimeError("BENCHMARKS.md does not contain '## Latest (recommended)' section.")
    body = m.group(1).strip() + "\n"
    return "## Latest (recommended)\n\n" + body


def replace_block(readme_text: str, new_block: str) -> str:
    pattern = re.compile(
        re.escape(START) + r".*?" + re.escape(END),
        flags=re.S,
    )
    replacement = START + "\n\n" + new_block.strip() + "\n\n" + END
    if not pattern.search(readme_text):
        raise RuntimeError("README.md does not contain EDGE_BENCH markers.")
    return pattern.sub(replacement, readme_text, count=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", required=True)
    ap.add_argument("--bench", required=True)
    args = ap.parse_args()

    readme_path = Path(args.readme)
    bench_path = Path(args.bench)

    readme_text = readme_path.read_text(encoding="utf-8")
    bench_text = bench_path.read_text(encoding="utf-8")

    latest_block = extract_latest_block(bench_text)
    updated = replace_block(readme_text, latest_block)

    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated {readme_path} from {bench_path}.")


if __name__ == "__main__":
    main()