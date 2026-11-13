#!/usr/bin/env python3
"""CUSUM change-detector baseline (plan §6): train on a split with keys, answer another.

Usage:
    python scripts/cusum_baseline.py --train DEV_SPLIT --eval SPLIT --out ANSWERS_DIR

Writes ANSWERS_DIR/<item_id>.json (one answer each, the grader's input) and
ANSWERS_DIR/model.json (fitted parameters and training signatures). Grade with
`scripts/grade.py --keys SPLIT/keys --answers ANSWERS_DIR`. The eval split's keys are
never read here. For the test split, ANSWERS_DIR must be outside any git work tree.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from hotwell_bench import cusum as C
from hotwell_bench.secrecy import check_output_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    check_output_path(out, Path(a.eval).name)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, sigs = C.train_from_split(Path(a.train))
    print(f"trained on {a.train}: {len(model['names'])} features, {len(sigs)} training "
          f"signatures, {time.time() - t0:.0f} s", flush=True)
    (out / "model.json").write_text(json.dumps({"model": model, "signatures": sigs}))
    items = sorted(p.name for p in (Path(a.eval) / "items").iterdir())
    n_det = 0
    for n, i in enumerate(items, 1):
        mn = C.minutes(*C.read_instruments(Path(a.eval) / "items" / i))
        ans, det = C.answer(mn, model, sigs)
        n_det += ans["fault_present"]
        (out / f"{i}.json").write_text(json.dumps(ans))
        if n % 25 == 0 or n == len(items):
            print(f"  {n}/{len(items)} answered, {n_det} alarms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
