#!/usr/bin/env python3
"""Per-item table and summary from written label files (no simulation).

Usage: python scripts/pilot_table.py [--dir out/labels/dev]

Reads the JSON files `pipeline.label_indices` wrote and prints one row per index:
- category, component, sensor kind, shape;
- magnitude / threshold;
- t_inj, delay t_det − t_inj, rule/instrument;
- attempts, easy, the operator plan's swap times;
- confuser verdicts.

Then it prints `pipeline.summarize` over the same files. Test-split label
directories live outside the repository; pass them explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hotwell_bench.pipeline import summarize
from hotwell_bench.secrecy import REPO_ROOT


def row(r: dict) -> str:
    it, lab = r["item"], r["label"]
    f = it["fault"]
    name = f"{it['split']}-{it['index']:04d}"
    if f is None:
        return f"{name} none"
    if lab is None:
        return (f"{name} {f['category']} {f['component']} {r['status']} "
                f"attempts={len(r['attempts'])}")
    x = f"{f['magnitude'] / f['threshold']:.2f}" if f.get("threshold") else "-"
    d = None if lab["t_det"] is None else lab["t_det"] - lab["t_inj"]
    op = it["operator"]
    conf = ", ".join(f"{c['category']}/{c['component']}:"
                     f"{'CONFUSABLE' if c['confusable'] else c['reason']}"
                     for c in r["confusers"]) or "-"
    return (f"{name} {f['category']:28s} {f['component']!s:15s} {f.get('sensor_kind')!s:8s} "
            f"{f['shape']:4s} ramp={f['params'].get('ramp_s', 0):6.0f} x={x:>5s} "
            f"t_inj={lab['t_inj']:6d} d={d!s:>6s} {lab['rule']}/{lab['instrument']} "
            f"att={len(r['attempts'])} easy={lab['easy']} "
            f"cp_swap={op['cp_swap_s']} fp_swap={op['fp_swap_s']} conf=[{conf}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO_ROOT / "out" / "labels" / "dev"))
    a = ap.parse_args()
    files = sorted(Path(a.dir).glob("*.json"))
    results = {}
    for p in files:
        r = json.loads(p.read_text())
        results[r["item"]["index"]] = r
        print(row(r))
    print(summarize(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
