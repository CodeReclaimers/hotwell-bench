#!/usr/bin/env python3
"""Re-label existing label files under another preset's rules (diagnostic only).

PLAN-bench-0.2.md §2.1: re-label the 0.1 dev split under the windowed rule and report
how many t_det values move. The items and runs are the 0.1 ones, read from the label
files and the run cache. Only `label.label_runs` is re-applied:
- no item is resampled;
- confusers are not re-checked. The windowed rule can only separate confusers that
  0.1 accepted, never merge them.
Nothing is written.

Usage: python scripts/relabel_diagnostic.py --labels out/labels/dev --cache cache --preset P02
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from hotwell_bench.items import ItemRecord
from hotwell_bench.label import label_runs
from hotwell_bench.presets import PRESETS
from hotwell_bench.run import run_cached


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    a = ap.parse_args()
    preset = PRESETS[a.preset]
    rows, moved, rules_new = [], [], Counter()
    for f in sorted(Path(a.labels).glob("*-*.json")):
        r = json.loads(f.read_text())
        if r["label"] is None or r["status"] != "accepted":
            continue
        it = ItemRecord.from_dict(r["item"])
        old = r["label"]
        new = label_runs(it, run_cached(it, "faulted", a.cache),
                         run_cached(it, "twin", a.cache), preset=preset)
        rules_new[new.rule] += 1
        d = None if new.t_det is None else new.t_det - old["t_det"]
        if d:
            moved.append(d)
        rows.append((it.item_id, it.fault.category, it.fault.shape, old["t_inj"],
                     old["t_det"], old["rule"], new.t_det, new.rule, new.instrument, d))
    print(f"{'item':10s} {'category':28s} {'shape':5s} {'t_inj':>6s} {'old t_det':>9s} "
          f"{'rule':8s} {'new t_det':>9s} {'rule':8s} {'instrument':16s} {'shift':>7s}")
    for x in rows:
        print(f"{x[0]:10s} {x[1]:28s} {x[2]:5s} {x[3]:6d} {x[4]:9d} {x[5]:8s} "
              f"{x[6]:9d} {x[7]:8s} {x[8]:16s} {x[9]:7d}")
    n = len(rows)
    moved.sort()
    print(f"\naccepted items: {n}; t_det moved: {len(moved)} (all earlier: "
          f"{all(d < 0 for d in moved)}); new t_det rule shares: {dict(rules_new)}")
    if moved:
        print(f"shift [s] min / median / max: {moved[0]} / {moved[len(moved) // 2]} / "
              f"{moved[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
