#!/usr/bin/env python3
"""Reference baselines and the G-B5 grader checks (B5; plan §6, §7).

Usage: python scripts/baselines.py --keys SPLIT/keys [--prior-keys DEV/keys]

Baselines graded against the keys:
- oracle: the key's first accepted answer, onset t_inj (G-B5: full = 1.0);
- always-none (G-B5: full = the null fraction, exactly);
- prior: the most frequent (category, component) of --prior-keys (default: the same
  keys), onset 12:00:00;
- shuffled keys: oracle answers graded against keys permuted by a fixed seed
  (G-B5: near chance).
Exit status 1 if a G-B5 check fails.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from hotwell_bench.grade import (
    always_none,
    grade,
    load_keys,
    oracle_answer,
    prior_answer,
    report,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--prior-keys", default=None)
    a = ap.parse_args()
    keys = load_keys(a.keys)
    ids = sorted(keys)
    ref = load_keys(a.prior_keys) if a.prior_keys else keys
    common = Counter((k["accepted_answers"][0]["category"], k["accepted_answers"][0]["component"])
                     for k in ref.values() if k["fault_present"]).most_common(1)[0][0]
    n_null = sum(not k["fault_present"] for k in keys.values())
    ok = True
    res = {}
    for name, ans in (
        ("oracle", {i: oracle_answer(keys[i]) for i in ids}),
        ("always-none", {i: always_none(keys[i]) for i in ids}),
        (f"prior {common}", {i: prior_answer(keys[i], *common, "12:00:00") for i in ids}),
    ):
        _, s = grade(keys, ans)
        res[name] = s
        print(f"\n== {name}\n{report(s)}", flush=True)
    perm = np.random.default_rng(12345).permutation(len(ids))
    shuffled = {ids[j]: oracle_answer(keys[ids[perm[j]]]) for j in range(len(ids))}
    _, s = grade(keys, shuffled)
    res["shuffled"] = s
    print(f"\n== oracle answers vs shuffled keys\n{report(s)}", flush=True)
    full = {k: v["overall"]["full"] for k, v in res.items()}
    print(f"\nG-B5: oracle {full['oracle']:.4f} (want 1); always-none "
          f"{full['always-none']:.4f} (want null fraction {n_null}/{len(ids)} = "
          f"{n_null / len(ids):.4f}); shuffled {full['shuffled']:.4f}", flush=True)
    ok &= full["oracle"] == 1.0
    ok &= abs(full["always-none"] - n_null / len(ids)) < 1e-12
    ok &= full["shuffled"] < 0.5
    print("G-B5 PASS" if ok else "G-B5 FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
