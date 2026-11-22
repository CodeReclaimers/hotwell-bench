"""C3: sampler changes for P02 (PLAN-bench-0.2.md §3) and the P01 guard."""
from __future__ import annotations

import hashlib

import numpy as np

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


def test_p02_decoys_are_independent_of_the_fault_and_placed_as_specified():
    """§3.2 / B2-Q13: decoys on fault and null days at the same rate; one or two
    distinct kinds; load excursions inside the day edges and clear of the daily trough;
    round-trip through JSON."""
    from hotwell_bench.items import ItemRecord
    from hotwell_bench.presets import DAY_S, P02
    dp = P02.decoys
    items = _p02_sample()
    has = [bool(it.decoys) for it in items]
    on_fault = [h for h, it in zip(has, items, strict=True) if it.fault is not None]
    on_null = [h for h, it in zip(has, items, strict=True) if it.fault is None]
    two = [len(it.decoys) == 2 for it in items if it.decoys]
    kinds = {}
    for it in items:
        for d in it.decoys:
            kinds[d.kind] = kinds.get(d.kind, 0) + 1
    print(f"\ndecoys: overall {sum(has) / len(has):.3f}, fault days "
          f"{sum(on_fault) / len(on_fault):.3f}, null days {sum(on_null) / len(on_null):.3f}, "
          f"two of them {sum(two) / len(two):.3f}, kinds {kinds}")
    assert abs(sum(has) / len(has) - 1 / 3) < 0.02
    assert abs(sum(on_fault) / len(on_fault) - sum(on_null) / len(on_null)) < 0.04
    assert abs(sum(two) / len(two) - 0.5) < 0.04
    for it in items:
        assert len({d.kind for d in it.decoys}) == len(it.decoys) <= 2
        assert ItemRecord.from_json(it.to_json()) == it
        for d in it.decoys:
            if d.kind == "load_excursion":
                end = d.t_s + 2 * d.params["ramp_s"] + d.params["hold_s"]
                assert dp.edge_s <= d.t_s and end <= DAY_S - dp.edge_s
                assert dp.excursion_mw[0] <= d.params["mw"] <= dp.excursion_mw[1]
                if not it.profile.flat:
                    bp = it.profile.breakpoints()
                    assert (end <= bp[1][0] - dp.trough_margin_s
                            or d.t_s >= bp[4][0] + dp.trough_margin_s)
            if d.kind == "pump_swap_back":
                base = it.operator.cp_swap_s if d.params["pair"] == "cp" else it.operator.fp_swap_s
                assert (d.params["swap_s"] is None) == (base is not None)
                start = base if base is not None else d.params["swap_s"]
                assert dp.swap_back_s[0] <= d.params["back_s"] - start <= dp.swap_back_s[1]
                assert d.params["back_s"] < DAY_S


def test_p02_decoys_build_into_valid_scenarios_and_schedules():
    """Builder (§3.2): excursions superpose on the profile as set_load events and pass
    hotwell's scenario validation; swap-backs and trim bursts appear in the operator
    schedule (validated against the procedures); the twin carries the same decoys."""
    from hotwell_bench.build import build, mw_breakpoints, operator_schedule
    from hotwell_bench.presets import RATED_MW
    items = [it for it in _p02_sample(600) if it.decoys][:60]
    n_ex = n_sw = n_tr = 0
    for it in items:
        b = build(it)                           # parses through hotwell's validators
        sched = operator_schedule(it)
        mw = mw_breakpoints(it)
        assert min(m for _, m in mw) >= RATED_MW - 15.0 - 24.0 - 1e-9
        for d in it.decoys:
            if d.kind == "load_excursion":
                n_ex += 1
                t_mid = d.t_s + d.params["ramp_s"] + d.params["hold_s"] / 2
                prof = float(np.interp(t_mid, *zip(*it.profile.breakpoints(), strict=True)))
                got = float(np.interp(t_mid, *zip(*mw, strict=True)))
                assert abs(got - (prof - d.params["mw"])) < 1e-6, (got, prof, d)
                assert b.scenario_yaml.count("set_load") == b.twin_yaml.count("set_load")
            if d.kind == "pump_swap_back":
                n_sw += 1
                back = [s for s in sched if s[0] == float(d.params["back_s"])]
                assert any(s[2].get("from") in ("cp2", "fp2") for s in back), back
            if d.kind == "trim_burst":
                n_tr += 1
                ts = {d.t_s + i * d.params["gap_s"] for i in range(d.params["n"])}
                assert ts <= {int(s[0]) for s in sched if s[1] == "NOP-03"}
    print(f"\nbuilt {len(items)} decoy items: {n_ex} excursions, {n_sw} swap-backs, "
          f"{n_tr} trim bursts")
    assert n_ex and n_sw and n_tr
