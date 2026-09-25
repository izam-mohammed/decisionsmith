"""Fail unless every module has at least --min % line + branch coverage (reads coverage.json)."""

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--min", type=float, default=95.0)
    p.add_argument("--file", default="coverage.json")
    args = p.parse_args(argv)
    with open(args.file) as f:
        files = json.load(f)["files"]
    low = {}
    for path, data in sorted(files.items()):
        s = data["summary"]
        total = s["num_statements"] + s.get("num_branches", 0)
        covered = s["covered_lines"] + s.get("covered_branches", 0)
        pct = 100.0 * covered / total if total else 100.0
        print("%6.1f%%  %s" % (pct, path))
        if pct < args.min:
            low[path] = pct
    if low:
        print("\nbelow %.0f%%: %s" % (args.min, ", ".join("%s (%.1f%%)" % kv for kv in low.items())))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
