#!/usr/bin/env python3
"""Run-to-run noise and a paired bootstrap between two configurations (PLAN-bench-0.2.md §6).

Usage:
    python scripts/compare_runs.py --keys SPLIT/keys --a DIR [DIR ...] [--b DIR [DIR ...]]
        [--criterion full] [--n-boot 10000] [--seed 0] [--out compare.json]

Each DIR is one run's answers (one `<item_id>.json` per item, as for scripts/grade.py).
--a and --b are the runs of configurations A and B. Prints, per configuration, the
per-run mean and the run-to-run flips (with two or more runs). With --b, it also prints
the paired bootstrap of A − B over items. Item ids are printed only as counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hotwell_bench.compare import flips, paired_bootstrap
from hotwell_bench.grade import grade, load_keys


def graded_run(keys: dict, d: Path) -> list[dict]:
    answers = {}
    for i in keys:
        p = d / f"{i}.json"
        if p.exists():
            try:
                answers[i] = json.loads(p.read_text())
            except json.JSONDecodeError:
                answers[i] = None
    return grade(keys, answers)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", default=None)
    ap.add_argument("--criterion", default="full")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    keys = load_keys(a.keys)
    out = {"criterion": a.criterion}
    runs = {"A": [graded_run(keys, Path(d)) for d in a.a]}
    if a.b:
        runs["B"] = [graded_run(keys, Path(d)) for d in a.b]
    for name, rs in runs.items():
        if len(rs) >= 2:
            f = flips(rs, a.criterion)
            out[name] = f
            print(f"{name}: {f['n_runs']} runs on {f['n_items']} items, per-run "
                  f"{a.criterion} {[round(x, 4) for x in f['per_run_mean']]}; "
                  f"{f['n_unstable']} items not the same in every run; pair flips "
                  f"{f['pair_flips']}", flush=True)
        else:
            print(f"{name}: 1 run (run-to-run noise needs at least two)", flush=True)
    if a.b:
        bs = paired_bootstrap(runs["A"], runs["B"], a.criterion, a.n_boot, a.seed)
        out["A-B"] = bs
        print(f"A - B on {bs['n_items']} items: {bs['mean_a']:.4f} - {bs['mean_b']:.4f} = "
              f"{bs['diff']:+.4f}, {bs['level']:.0%} interval [{bs['ci'][0]:+.4f}, "
              f"{bs['ci'][1]:+.4f}], P(diff <= 0) {bs['p_diff_le_0']:.4f} "
              f"({bs['n_boot']} resamples, seed {bs['seed']})", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
