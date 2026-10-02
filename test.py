"""Count records in a local sentence corpus without network dependencies."""

import argparse
import gzip
import json
from pathlib import Path


def count_sentences(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("data/eng-base.jsonl"))
    args = parser.parse_args()
    print(count_sentences(args.path))


if __name__ == "__main__":
    main()
