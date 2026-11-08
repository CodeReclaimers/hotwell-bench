#!/usr/bin/env python3
"""Grade a directory of answers against a split's keys (B5; plan §6).

Usage:
    python scripts/grade.py --keys SPLIT/keys --answers DIR [--out graded.json]

DIR holds one `<item_id>.json` per item, each a single answer object (task.md).
A missing or unparseable file is graded invalid. Prints the report; --out writes
per-item results and the summary as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hotwell_bench.grade import grade, load_keys, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    keys = load_keys(a.keys)
    answers = {}
    for i in keys:
        p = Path(a.answers) / f"{i}.json"
        if p.exists():
            try:
                answers[i] = json.loads(p.read_text())
            except json.JSONDecodeError:
                answers[i] = None
    graded, summary = grade(keys, answers)
    print(f"{len(keys)} keys, {len(answers)} answer files")
    print(report(summary))
    if a.out:
        Path(a.out).write_text(json.dumps({"items": graded, "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
