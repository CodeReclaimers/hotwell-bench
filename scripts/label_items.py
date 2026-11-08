#!/usr/bin/env python3
"""Label a range of indices (runs, regeneration, confusers) and print the report.

Usage:
    python scripts/label_items.py --master N --split dev --indices 0-49
        [--workers W] [--cache DIR] [--out DIR] [--cap K]

Test-split use requires --cache and --out outside any git work tree (secrecy guard).
"""
from __future__ import annotations

import argparse
import sys

from hotwell_bench.pipeline import ATTEMPT_CAP, label_indices, summarize
from hotwell_bench.secrecy import REPO_ROOT


def parse_range(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=int, required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--indices", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cap", type=int, default=ATTEMPT_CAP)
    a = ap.parse_args()
    if a.split == "test" and (a.cache is None or a.out is None):
        ap.error("--split test needs explicit --cache and --out outside the repository")
    cache = a.cache or str(REPO_ROOT / "cache")
    out = a.out or str(REPO_ROOT / "out" / "labels" / a.split)
    res = label_indices(a.master, a.split, parse_range(a.indices), cache, out,
                        workers=a.workers, cap=a.cap)
    print(summarize(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
