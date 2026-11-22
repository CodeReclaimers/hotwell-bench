"""C3: sampler changes for P02 (PLAN-bench-0.2.md §3) and the P01 guard."""
from __future__ import annotations

import hashlib

from hotwell_bench.items import sample_item
from hotwell_bench.presets import P01

#: sha256 over sample_item(...).to_json() for masters (20261107, 1, 987654321), splits
#: (dev, test), indices 0..399 and attempts 0..2 under P01 (7200 items). Taken at the C0
#: refactor (2026-11-19), when it equalled the pre-refactor sampler's hash. G-B0r covers
#: the whole 0.1 dev pipeline; this guards the sampler alone, fast.
P01_GOLDEN = "31b300dc767f67385703b3c8850ca1a0eb5e90f2ec344794f5601f2ff645faf1"


def test_p01_sampler_output_is_unchanged():
    h = hashlib.sha256()
    n = 0
    for master in (20261107, 1, 987654321):
        for split in ("dev", "test"):
            for i in range(400):
                for a in range(3):
                    h.update(sample_item(master, split, i, a, preset=P01).to_json().encode())
                    n += 1
    print(f"\nP01 sampler: {n} items, sha256 {h.hexdigest()}")
    assert n == 7200 and h.hexdigest() == P01_GOLDEN


def _p02_sample(n=6000, master=20261122):
    from hotwell_bench.presets import P02
    return [sample_item(master, "dev", i, 0, preset=P02) for i in range(n)]


def test_p02_sampling_matches_the_preset():
    """P02 (§3.1, B2-Q7, B2-Q11, B2-Q14, B2-Q15): null fraction, ramp share, magnitude
    ratios within MAG_RANGE on the C2 thresholds, frequency never bias/drift,
    noise_increase factor = 1 + r·(thr − 1)."""
    from hotwell_bench.presets import P02
    from hotwell_bench.window_detectability import (
        SENSOR_BIAS_THRESHOLD,
        SENSOR_NOISE_THRESHOLD,
        THRESHOLDS_V020,
    )
    items = _p02_sample()
    faults = [it.fault for it in items if it.fault is not None]
    null = 1 - len(faults) / len(items)
    thr = [f for f in faults if (f.mechanism, f.target) in THRESHOLDS_V020]
    ramps = sum(f.shape == "ramp" for f in thr) / len(thr)
    ratios = [f.magnitude_ratio for f in faults if f.magnitude_ratio is not None]
    sens = [f for f in faults if f.sensor_kind is not None]
    freq = {f.sensor_kind for f in sens if f.target == "frequency"}
    print(f"\nP02: {len(items)} items, null {null:.3f}, thresholded ramp share {ramps:.3f}, "
          f"ratio range {min(ratios):.3f}..{max(ratios):.3f}, frequency kinds {sorted(freq)}")
    assert abs(null - 0.30) < 0.02
    assert abs(ramps - 0.70) < 0.03
    assert 0.7 - 1e-9 <= min(ratios) and max(ratios) <= 3.0 + 1e-9
    assert freq and freq <= {"stuck", "noise_increase"}
    for f in thr:
        assert f.threshold == THRESHOLDS_V020[(f.mechanism, f.target)][0]
        if f.shape == "ramp":
            assert 1800.0 <= f.params["ramp_s"] <= 8 * 3600.0
    for f in sens:
        assert f.sensor_kind in P02.sensor_kinds
        if f.sensor_kind in ("bias", "drift"):
            assert f.threshold == SENSOR_BIAS_THRESHOLD[f.target][0]
        if f.sensor_kind == "noise_increase":
            t = SENSOR_NOISE_THRESHOLD[f.target][0]
            assert abs(f.params["factor"] - (1 + f.magnitude_ratio * (t - 1))) < 1e-12
