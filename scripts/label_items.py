#!/usr/bin/env python3
"""Label a range of indices (runs, regeneration, confusers) and print the report.

Usage:
    python scripts/label_items.py --master N --split dev --indices 0-49
        [--workers W] [--cache DIR] [--out DIR] [--cap K]

Test-split use requires --cache and --out outside any git work tree (secrecy guard).

The environment gate (`provenance.check_environment`) runs first. A `run.json` is
written next to the label files: the environment record, the platform, the bench
commit (and whether the tree was dirty), the arguments, and the SHA-256 of the
master seed. The seed itself is never written there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from hotwell_bench.pipeline import ATTEMPT_CAP, label_indices, summarize
from hotwell_bench.provenance import check_environment
from hotwell_bench.secrecy import REPO_ROOT


def parse_range(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def bench_commit() -> dict:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *a], check=True,
                              capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


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
    env = check_environment()
    record = {"environment": env, "platform": platform.platform(),
              "python": platform.python_version(), "bench": bench_commit(),
              "split": a.split, "indices": a.indices, "cap": a.cap,
              "master_sha256": hashlib.sha256(str(a.master).encode()).hexdigest(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    print(f"environment OK: hotwell {env['hotwell']['commit_id'][:12]}, numpy "
          f"{env['numpy']}, bench {record['bench']}", flush=True)
    res = label_indices(a.master, a.split, parse_range(a.indices), cache, out,
                        workers=a.workers, cap=a.cap)
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (Path(out) / "run.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print(summarize(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
