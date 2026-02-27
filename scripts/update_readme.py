from __future__ import annotations

import argparse
import re
from pathlib import Path

START = "<!-- EDGE_BENCH:START -->"
END = "<!-- EDGE_BENCH:END -->"

def extract_between_markers(text: str, start: str, end: str) -> str:
    s = text.find(start)
    e = text.find(end)
    if s == -1 or e == -1 or e <= s:
        raise RuntimeError(f"Could not find marker block: {start} ... {end}")
    return text[s + len(start) : e].strip("\n")

def pick_latest_section(bench_text: str) -> str:
    """
    BENCHMARKS.md 전체에서 'Latest' 섹션을 최대한 관대하게 찾는다.
    - 우선: '## Latest' 로 시작하는 섹션을 찾고
    - 그 안에서 첫 번째 markdown table(| Model | ... )를 뽑는다.
    """
    # 1) Latest 섹션 덩어리 찾기 (## Latest ... 다음 ## 까지)
    m = re.search(r"^##\s+Latest.*?$([\s\S]*?)(?=^##\s+|\Z)", bench_text, re.MULTILINE)
    if not m:
        raise RuntimeError("BENCHMARKS.md does not contain a '## Latest' section.")
    block = m.group(0)

    # 2) 그 블록에서 첫 번째 md table 추출 (헤더라인 |...| 과 separator |---| 포함)
    t = re.search(
        r"(\|[^\n]*\|\n\|[-:| ]+\|\n(?:\|[^\n]*\|\n)+)",
        block,
        re.MULTILINE,
    )
    if not t:
        raise RuntimeError("Could not find a markdown table inside the Latest section.")
    return t.group(1).strip("\n")

def update_readme(readme_path: Path, bench_path: Path) -> None:
    readme = readme_path.read_text(encoding="utf-8")
    bench = bench_path.read_text(encoding="utf-8")

    latest_table = pick_latest_section(bench)

    # README 내 마커 블록 치환
    s = readme.find(START)
    e = readme.find(END)
    if s == -1 or e == -1 or e <= s:
        raise RuntimeError(f"README.md does not contain marker block: {START} ... {END}")

    new_block = f"{START}\n\n{latest_table}\n\n{END}"
    out = readme[:s] + new_block + readme[e + len(END):]

    readme_path.write_text(out, encoding="utf-8")
    print(f"Updated {readme_path} from {bench_path}.")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", required=True)
    ap.add_argument("--bench", required=True)
    args = ap.parse_args()

    update_readme(Path(args.readme), Path(args.bench))

if __name__ == "__main__":
    main()