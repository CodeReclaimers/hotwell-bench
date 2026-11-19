#!/usr/bin/env python3
"""Seed-pair null days for the windowed level rule's threshold (PLAN-bench-0.2.md §2.1).

Pair k is one fault-free background, the day profile and operator plan of
`sample_item(CAL_MASTER, "dev", k, preset=P01)` with its fault removed, run twice:
- run A with the item's own sim_seed;
- run B with a different sim_seed.

Every random stream in a hotwell run (sensor noise, load noise, operator reaction
delays) comes from the simulator's stream bank, seeded by sim_seed, so A and B differ
in every random stream and share everything else. A − B is then the day-to-day
variability an analyst must separate a fault from.

The split is fixed before any data is seen: even k calibrate the threshold and odd k
validate it.

For each pair and each instrument, and each window W in WINDOWS_S with sustain S, the
statistic is
    M = max over t of ( min over the S samples ending at t of |m_W| ),
where m_W(t) is the trailing W-sample mean of (A − B) over samples t−W+1 … t.
M is exactly the largest threshold θ that the windowed rule (sustained exceedance
for S samples) would exceed somewhere in the day. So the rule fires on a pair iff
M > θ for some instrument. The robust scale 1.4826·MAD of m_W is also recorded.

Usage:
    python scripts/null_pairs.py --pairs 0-199 [--workers W] [--cache DIR] [--out FILE]

Runs are cached (default `cache/null_pairs/`, gitignored), so a re-run recomputes the
statistics without simulating.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from hotwell_bench.items import ItemRecord, sample_item
from hotwell_bench.presets import P01
from hotwell_bench.provenance import check_environment
from hotwell_bench.run import COLUMNS, _task, run_cached
from hotwell_bench.secrecy import REPO_ROOT

#: Public calibration master seed. It differs from the dev master (20261107) and is
#: used only for threshold calibration, never for a split.
CAL_MASTER = 2026111901
WINDOWS_S = (300, 600, 1200)
SUSTAIN_S = 60


def pair(k: int) -> tuple[ItemRecord, ItemRecord]:
    base = dataclasses.replace(sample_item(CAL_MASTER, "dev", k, preset=P01), fault=None)
    ss = np.random.SeedSequence([CAL_MASTER, k, 1])
    seed_b = int(ss.generate_state(1, np.uint32)[0]) & (2**31 - 1)
    if seed_b == base.sim_seed:
        seed_b = (seed_b + 1) % 2**31
    return base, dataclasses.replace(base, sim_seed=seed_b)


def parse_range(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing w-sample mean along axis 0; row j covers rows j−w+1 … j (rows ≥ w−1 only)."""
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    return (c[w:] - c[:-w]) / w


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    out: dict = {"n_samples": int(n)}
    for w in WINDOWS_S:
        m = np.abs(trailing_mean(d, w))
        sus = np.lib.stride_tricks.sliding_window_view(m, SUSTAIN_S, axis=0).min(axis=-1)
        med = np.median(m, axis=0)       # |m| is folded; MAD of signed m below
        sm = trailing_mean(d, w)
        mad = 1.4826 * np.median(np.abs(sm - np.median(sm, axis=0)), axis=0)
        out[str(w)] = {"M": dict(zip(COLUMNS, map(float, sus.max(axis=0)), strict=True)),
                       "argmax_t": dict(zip(COLUMNS, map(int, sus.argmax(axis=0)
                                                       + w - 1 + SUSTAIN_S - 1),
                                            strict=True)),
                       "scale_mad": dict(zip(COLUMNS, map(float, mad), strict=True)),
                       "median_abs": dict(zip(COLUMNS, map(float, med), strict=True))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", default=str(REPO_ROOT / "cache" / "null_pairs"))
    ap.add_argument("--out", default=str(REPO_ROOT / "out" / "null_pairs" / "stats.json"))
    a = ap.parse_args()
    env = check_environment()
    ks = parse_range(a.pairs)
    items = [it for k in ks for it in pair(k)]
    workers = a.workers or max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(workers, len(items)),
                             mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(_task, (it.to_json(), "faulted", a.cache, None)) for it in items]
        for n, fut in enumerate(as_completed(futs), 1):
            iid, _, summ = fut.result()
            print(f"[{time.time() - t0:6.0f} s] [{n}/{len(futs)}] {iid}: t_end "
                  f"{summ['t_end']:.0f} s, dead {summ['plant_dead']}, wall "
                  f"{summ['wall_s']:.0f} s", flush=True)
    stats = {}
    for k in ks:
        A, B = pair(k)
        ra, rb = run_cached(A, "faulted", a.cache), run_cached(B, "faulted", a.cache)
        s = pair_stats(ra.values, rb.values)
        s.update(role="calibration" if k % 2 == 0 else "validation",
                 seeds=[A.sim_seed, B.sim_seed], flat=A.profile.flat,
                 cp_swap_s=A.operator.cp_swap_s, fp_swap_s=A.operator.fp_swap_s,
                 dead=[ra.summary["plant_dead"], rb.summary["plant_dead"]])
        stats[str(k)] = s
    rec = {"cal_master": CAL_MASTER, "windows_s": list(WINDOWS_S), "sustain_s": SUSTAIN_S,
           "background_preset": P01.name, "environment": env, "pairs": stats}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"wrote {a.out}: {len(stats)} pairs, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
