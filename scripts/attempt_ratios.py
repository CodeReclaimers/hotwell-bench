#!/usr/bin/env python3
"""Observability of labelled attempts by magnitude ratio (C6 calibration diagnostic).

Usage:
    python scripts/attempt_ratios.py --labels out/labels/RUN --master SEED --preset P02
        [--split dev]

For every fault attempt in the label files, re-samples the item at that attempt (items
are a pure function of master, split, index, attempt and preset) and tallies whether the
labeller found it observable, by the fault's magnitude / windowed-threshold ratio. Shows
where the acceptance bar sits relative to MAG_RANGE. Dev labels only: it prints counts,
no item ids. Refuses (exit 1) unless every labelled item re-samples identically, i.e.
the preset and master are as they were at labelling time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hotwell_bench.items import sample_item
from hotwell_bench.presets import PRESETS

EDGES = ((0.5, "<0.5"), (0.75, "0.5-0.75"), (1.0, "0.75-1"), (1.5, "1-1.5"), (2.0, "1.5-2"),
         (3.0, "2-3"), (float("inf"), ">=3"))


def bucket(r: float | None) -> str:
    if r is None:
        return "n/a"
    return next(name for hi, name in EDGES if r < hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--master", type=int, required=True)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--split", default="dev", choices=["dev"])
    a = ap.parse_args()
    preset = PRESETS[a.preset]
    tally: dict[str, list[int]] = {}
    checked = 0
    for f in sorted(Path(a.labels).glob(f"{a.split}-*.json")):
        d = json.loads(f.read_text())
        idx = int(f.stem.split("-")[1])
        rec = d.get("item")
        if rec is not None:
            again = sample_item(a.master, a.split, idx, rec["attempt"], preset=preset)
            if json.loads(json.dumps(again.to_dict())) != rec:
                print(f"{f.name}: the item re-sampled under {a.preset} as it is now differs "
                      "from the labelled one (the preset or master changed since labelling); "
                      "run this at the labels' commit", file=sys.stderr)
                return 1
            checked += 1
        for att in d.get("attempts", []):
            item = sample_item(a.master, a.split, idx, att["attempt"], preset=preset)
            if item.fault is None:
                continue
            t = tally.setdefault(bucket(item.fault.magnitude_ratio), [0, 0])
            t[0] += 1
            t[1] += bool(att["observable"])
    print(f"{checked} labelled items re-sampled identically under {a.preset}")
    print(f"{'ratio':9s} {'attempts':>8s} {'observable':>10s}")
    for name in [n for _, n in EDGES] + ["n/a"]:
        if name in tally:
            n, o = tally[name]
            print(f"{name:9s} {n:8d} {o:10d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
