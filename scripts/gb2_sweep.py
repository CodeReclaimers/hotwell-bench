#!/usr/bin/env python3
"""G-B2 sweep with per-rule diagnostics.

Usage: python scripts/gb2_sweep.py [--workers N] [--cache DIR]

Runs hotwell's 0.4.0 detectability-sweep setup through the bench runner (AB + 3E,
flat 60 MW, no operator, seed 42, step at 300 s, 1 h). For every row it prints
hotwell's delay next to the bench level rule's, then every stuck-rule hit (with the
faulted run length that fired and the twin's longest run) and every variance-rule
hit. Rows hotwell found undetectable are the interesting ones: there, any stuck or
variance hit has to be justified as real evidence, not a statistics artifact.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_b3_label import SWEEP_AB, gb2_items  # noqa: E402

from hotwell_bench import label as L  # noqa: E402
from hotwell_bench.run import COLUMNS, run_cached, run_items  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    cache = a.cache or tempfile.mkdtemp(prefix="gb2-")
    items = gb2_items()
    run_items(items, cache, workers=a.workers, until=3600)
    for it, (mech, tgt, _p, val, delay, _inst) in zip(items, SWEEP_AB, strict=True):
        fa = run_cached(it, "faulted", cache, until=3600)
        tw = run_cached(it, "twin", cache, until=3600)
        ev = L.evidence(fa.values, tw.values)
        lv = [(t, COLUMNS[i]) for i, t in enumerate(ev["level"]) if t is not None]
        out = []
        for i, c in enumerate(COLUMNS):
            ts = ev["stuck"][i]
            if ts is not None:
                _, lt = L._runs(tw.values[:, i])
                _, lf = L._runs(fa.values[:, i])
                out.append(f"stuck {c}@{ts - 300:+d} (F run {int(lf.max())}, "
                           f"T max {int(lt.max())})")
            tv = ev["variance"][i]
            if tv is not None:
                out.append(f"var {c}@{tv - 300:+d}")
        lvl = min(lv) if lv else None
        print(f"{mech:18s} {tgt:9s} {val!s:>6s} hotwell {delay!s:>4s} | level "
              f"{'-' if lvl is None else f'{lvl[0] - 300} {lvl[1]}'} | "
              + ("; ".join(out) or "no stuck/variance hits"), flush=True)
    print(f"cache: {cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
