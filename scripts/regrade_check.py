#!/usr/bin/env python3
"""G-B5v (PLAN-bench-0.2.md §8): answers regrade byte-identically under the current grader
and a reference revision's grader.

Usage:
    python scripts/regrade_check.py --keys SPLIT/keys --answers DIR [DIR ...] [--rev v0.1.0]

For each answer directory, grades it with `hotwell_bench.grade` as checked out at --rev
(loaded from `git show REV:src/hotwell_bench/grade.py`, importing the current taxonomy;
check that taxonomy.py is unchanged since REV) and with the working tree's grader. It
compares the bytes of json.dumps({"items", "summary"}, indent=1) (scripts/grade.py
--out) and of the printed report. Prints one line per directory with the
sha256 prefix and whether it matches, and only counts at the end. Answer contents are
never printed. Exit status 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from hotwell_bench import grade as G
from hotwell_bench.secrecy import REPO_ROOT


def grader_at(rev: str):
    src = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{rev}:src/hotwell_bench/grade.py"],
                         check=True, capture_output=True, text=True).stdout
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"grade_{rev.replace('.', '_')}.py"
        p.write_text(src)
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def taxonomy_changed(rev: str) -> bool:
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--quiet", rev, "--",
                        "src/hotwell_bench/taxonomy.py"])
    return r.returncode != 0


def load_answers(keys: dict, d: Path) -> dict:
    """As scripts/grade.py reads them."""
    out = {}
    for i in keys:
        p = d / f"{i}.json"
        if p.exists():
            try:
                out[i] = json.loads(p.read_text())
            except json.JSONDecodeError:
                out[i] = None
    return out


def outputs(mod, keys: dict, answers: dict) -> tuple[bytes, bytes]:
    graded, summary = mod.grade(keys, answers)
    return (json.dumps({"items": graded, "summary": summary}, indent=1).encode(),
            mod.report(summary).encode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--answers", required=True, nargs="+")
    ap.add_argument("--rev", default="v0.1.0")
    a = ap.parse_args()
    if taxonomy_changed(a.rev):
        print(f"taxonomy.py differs from {a.rev}; the reference grader would not be {a.rev}'s")
        return 1
    ref = grader_at(a.rev)
    keys = G.load_keys(a.keys)
    print(f"{len(keys)} keys, bench version {G.split_version(keys)}; reference grader {a.rev}",
          flush=True)
    n_ok = 0
    for n, d in enumerate(a.answers):
        ans = load_answers(keys, Path(d))
        (j0, r0), (j1, r1) = outputs(ref, keys, ans), outputs(G, keys, ans)
        same = j0 == j1 and r0 == r1
        n_ok += same
        print(f"  set {n:3d}: {len(ans):4d} answers, json {hashlib.sha256(j1).hexdigest()[:16]} "
              f"report {hashlib.sha256(r1).hexdigest()[:16]} "
              f"{'identical' if same else 'DIFFERS'}", flush=True)
    print(f"G-B5v: {n_ok} of {len(a.answers)} answer sets regrade byte-identically", flush=True)
    return 0 if n_ok == len(a.answers) else 1


if __name__ == "__main__":
    sys.exit(main())
