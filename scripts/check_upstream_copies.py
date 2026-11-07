#!/usr/bin/env python3
"""Check that constants copied from hotwell still equal their upstream source.

Usage: python scripts/check_upstream_copies.py PATH_TO_HOTWELL_CHECKOUT [--ref v0.4.0]

Reads `scripts/quiet_day.py` from the checkout at REF (via `git show`, so the
working tree's state does not matter) and compares QUIET_DAY_SENSOR_NOISE and
QUIET_DAY_LOAD_NOISE, parsed as literals (no import, no execution), against
hotwell_bench.presets. Exit status 1 on any difference.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys

from hotwell_bench import presets

COPIES = {"QUIET_DAY_SENSOR_NOISE": "SENSOR_NOISE", "QUIET_DAY_LOAD_NOISE": "LOAD_NOISE"}


def upstream_literals(checkout: str, ref: str) -> dict:
    src = subprocess.run(["git", "-C", checkout, "show", f"{ref}:scripts/quiet_day.py"],
                         check=True, capture_output=True, text=True).stdout
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in COPIES:
                out[name] = ast.literal_eval(node.value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkout")
    ap.add_argument("--ref", default="v0.4.0")
    a = ap.parse_args()
    up = upstream_literals(a.checkout, a.ref)
    ok = True
    for up_name, local_name in COPIES.items():
        same = up.get(up_name) == getattr(presets, local_name)
        print(f"{up_name} @ {a.ref} == presets.{local_name}: {same}", flush=True)
        ok &= same
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
