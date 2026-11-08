#!/usr/bin/env python3
"""Package one labelled split, add the manual, run the leak scan (B4; plan §5, §8.1).

Usage:
    python scripts/package_split.py --split dev --labels out/labels/dev --cache cache \
        --out DIR --canary-file G [--master-seed-file F] [--hotwell-repo ../hotwell]

Reads the label JSONs written by scripts/label_items.py and the faulted runs from
the cache. Writes DIR/<split>/ (layout in hotwell_bench.package). The manual is
taken from `git archive <pinned tag> docs/manual` of the hotwell repository.

Provenance comes from <labels>/run.json when present. The dev pilot predates
run.json, so its provenance is the current environment plus the commit that
labelled it, passed with --labels-commit.

--canary-file holds the release's canary GUID (kept outside git; plan §8.1).

With --master-seed-file, the seed is added to the scan's forbidden seeds and the
file is copied to keys/MASTER_SEED. Exit status 1 if the leak scan finds anything.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hotwell_bench import package as K
from hotwell_bench.items import ItemRecord
from hotwell_bench.provenance import PINNED_HOTWELL_TAG, check_environment
from hotwell_bench.run import run_cached


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--canary-file", required=True)
    ap.add_argument("--master-seed-file", default=None)
    ap.add_argument("--master", type=int, default=None, help="dev only: the dev master seed")
    ap.add_argument("--labels-commit", default=None)
    ap.add_argument("--hotwell-repo", default=str(Path(__file__).resolve().parents[2] / "hotwell"))
    a = ap.parse_args()
    env = check_environment()
    guid = K.load_guid(a.canary_file)
    t0 = time.time()
    labels = Path(a.labels)
    rj = labels / "run.json"
    if rj.exists():
        prov = json.loads(rj.read_text())
    else:
        if a.labels_commit is None:
            ap.error(f"{rj} missing: pass --labels-commit")
        prov = {"environment": env, "platform": platform.platform(),
                "python": platform.python_version(),
                "bench": {"commit": a.labels_commit, "dirty": False,
                          "note": "labelled before run.json existed"}}
    prov = {k: prov[k] for k in ("environment", "platform", "python", "bench")}
    results, runs = {}, {}
    for p in sorted(labels.glob(f"{a.split}-*.json")):
        r = json.loads(p.read_text())
        i = r["item"]["index"]
        results[i] = r
        if r["status"] != "excluded_at_cap":
            it = ItemRecord.from_json(json.dumps(r["item"]))
            runs[i] = run_cached(it, "faulted", a.cache)
    print(f"{len(results)} label files, {len(runs)} to package", flush=True)
    seeds = {r["item"]["sim_seed"] for r in results.values()}
    if a.master_seed_file:
        seeds.add(int(Path(a.master_seed_file).read_text().strip()))
    if a.master is not None:
        seeds.add(a.master)
    man = K.package_split(a.split, results, runs, a.out, prov, guid)
    split_dir = Path(a.out) / a.split
    tar = subprocess.run(["git", "-C", a.hotwell_repo, "archive", PINNED_HOTWELL_TAG,
                          "docs/manual"], check=True, capture_output=True).stdout
    man["files"].update(K.add_manual(split_dir, tar, guid))
    man["files"] = dict(sorted(man["files"].items()))
    (split_dir / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    if a.master_seed_file:
        shutil.copyfile(a.master_seed_file, split_dir / "keys" / "MASTER_SEED")
    problems = K.leak_scan(split_dir, seeds, guid) + K.scan_manual(split_dir, seeds, guid)
    n_files = len(man["files"])
    print(f"packaged {man['n_items']} items, {n_files} manifest files, "
          f"{time.time() - t0:.0f} s", flush=True)
    print(f"leak scan: {len(problems)} problems", flush=True)
    for p in problems[:50]:
        print("  " + p, flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
