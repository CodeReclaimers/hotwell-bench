"""CUSUM baseline on synthetic days (no simulator, no release files).

A synthetic day: every instrument = base + gain x load(t) + white noise, with a load
trough and a pump swap in each pair. Checks what the method promises:
- no alarm on its own training days (thresholds are H_MARGIN x the training maxima);
  with the swap inside every day, this also shows a pump swap alone does not alarm;
- a step on one instrument is detected, first on that instrument, with the onset at
  the step.
Unseen fault-free days are NOT promised to stay silent: 96 one-sided statistics run
every minute, and a threshold taken from finite training maxima is exceeded on some
new days (on the 0.1 test split, 8 of 39 null days alarmed). The count is printed."""
from __future__ import annotations

import numpy as np

from hotwell_bench import cusum as C
from hotwell_bench.package import COLUMNS

DAY = 6 * 3600     # 6 h synthetic days keep the test fast


def day(seed: int, step: tuple[str, int, float] | None = None, swap_at: int = 3 * 3600):
    g = np.random.default_rng(seed)
    t = np.arange(DAY)
    trough = 1 - 0.25 * np.clip(np.minimum((t - 3600) / 1800, (5 * 3600 - t) / 1800), 0, 1)
    load = 60e6 * trough + g.normal(0, 3e5, DAY).cumsum() * 0.001
    a = np.empty((DAY, len(COLUMNS)))
    for j in range(len(COLUMNS)):
        base, gain = 10.0 + j, 0.5 + 0.1 * j
        a[:, j] = base + gain * load / 60e6 + g.normal(0, 0.01, DAY)
    for p1, p2 in C.PAIRS.values():                # p1 runs until the swap, then p2
        run1 = t < swap_at
        for q, on, off in (("current", 50.0, 0.2), ("npsh_margin", 1.0, 2.0)):
            a[:, COLUMNS.index(f"{p1}_{q}")] = np.where(run1, on, off) + g.normal(0, 0.01, DAY)
            a[:, COLUMNS.index(f"{p2}_{q}")] = np.where(run1, off, on) + g.normal(0, 0.01, DAY)
    a[:, COLUMNS.index("mw_out")] = load + g.normal(0, 5e4, DAY)
    if step:
        name, t0, size = step
        a[t0:, COLUMNS.index(name)] += size
    return C.minutes(list(COLUMNS), a)


def test_cusum_detects_a_step_and_ignores_swaps():
    train = [(day(s), DAY // 60) for s in range(30)]
    model = C.fit(train)
    own = [C.detect(mn, model).t_alarm_min for mn, _ in train]
    unseen = [C.detect(day(100 + s), model).t_alarm_min for s in range(10)]
    print(f"\nalarms on training days: {sum(a is not None for a in own)}/30; "
          f"on unseen fault-free days: {sum(a is not None for a in unseen)}/10 {unseen}")
    assert all(a is None for a in own)

    t0 = 45 * 60                                   # just after the 30-min warm-up
    det = C.detect(day(200, ("da_pressure", t0, 0.05)), model)
    name = model["names"][det.first]
    print(f"step 0.05 (5 sigma) at minute {t0 // 60}: alarm minute {det.t_alarm_min}, "
          f"onset minute {det.onset_min}, first feature {name}")
    assert det.t_alarm_min is not None and name == "da_pressure:mean"
    assert t0 // 60 <= det.t_alarm_min <= t0 // 60 + 5
    assert abs(det.onset_min - t0 // 60) <= 2


def test_raw_instrument_maps_pump_roles():
    mn = day(1)
    early, late = 10, DAY // 60 - 10            # before and after the swap
    assert C.raw_instrument("cp_run_current:mean", mn, early) == "cp1_current"
    assert C.raw_instrument("cp_run_current:mean", mn, late) == "cp2_current"
    assert C.raw_instrument("fp_sb_npsh_margin:logstd", mn, early) == "fp2_npsh_margin"
    assert C.raw_instrument("drum_level:mean", mn, early) == "drum_level"
    assert C.onset_hms(61) == "01:01:00"
