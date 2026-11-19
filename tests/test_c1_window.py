"""C1: the windowed level rule (PLAN-bench-0.2.md §2.1), on synthetic signals.

A test preset enables the rule with θ_i = 0.2 D_i on every instrument; the
calibrated P02 thresholds are tested separately once they exist."""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from hotwell_bench import label as L
from hotwell_bench.presets import P01
from hotwell_bench.run import COLUMNS

W, S = 600, 60
THETA = {c: 0.2 * float(L.SCALE[i]) for i, c in enumerate(COLUMNS)}
PW = dataclasses.replace(P01, name="test-window", window_s=W, window_sustain_s=S,
                         window_thresholds=THETA)
J = COLUMNS.index("drum_level")


def _noise(n=6000, seed=3):
    g = np.random.default_rng(seed)
    return g.normal(0.0, 1.0, (n, len(COLUMNS))) * L.SCALE / 3.0   # σ_i = D_i / 3


def test_p01_has_no_window_rule():
    T = _noise()
    ev = L.evidence(T, T.copy(), preset=P01)
    assert L.rules(P01) == ("level", "stuck", "variance") and "window" not in ev


def test_sub_threshold_step_missed_per_sample_caught_windowed():
    """A step of h = 0.5 D_i from t0: |F − T| = h < D_i, so the level rule never fires.
    The trailing mean of (F − T) is h·(t − t0 + 1)/W for t0 ≤ t < t0 + W, so it first
    exceeds θ = 0.2 D_i at t − t0 + 1 = floor(θW/h) + 1 and stays above: the window
    rule reports t* = t0 + floor(θW/h) exactly."""
    T = _noise()
    t0, h = 2000, 0.5 * float(L.SCALE[J])
    F = T.copy()
    F[t0:, J] += h
    ev = L.evidence(F, T, preset=PW)
    want = t0 + math.floor(THETA[COLUMNS[J]] * W / h)
    print(f"\nstep 0.5 D_i at {t0}: level {ev['level'][J]}, window {ev['window'][J]} "
          f"(expected {want})")
    assert ev["level"][J] is None
    assert ev["window"][J] == want
    lab_rules = L.rules(PW)
    assert lab_rules[-1] == "window"


def test_effect_at_half_theta_not_caught():
    T = _noise()
    F = T.copy()
    F[1000:, J] += 0.5 * THETA[COLUMNS[J]]
    ev = L.evidence(F, T, preset=PW)
    print(f"\nstep 0.5 θ: window {ev['window'][J]}")
    assert all(x is None for x in ev["window"])


def test_sustain_needs_s_consecutive_window_ends():
    """A pulse that keeps the trailing mean above θ for S − 1 window ends is not
    evidence; one more sample of pulse makes it evidence."""
    T = np.zeros((4000, len(COLUMNS)))
    th = THETA[COLUMNS[J]]
    # A block of height H over W + n − 1 samples: a window covers the whole block's
    # height on exactly n window ends (mean H), and at most W − 1 samples of it
    # elsewhere (mean ≤ H (W − 1)/W). With θ < H < θ W / (W − 1) only the n full
    # windows exceed θ, so the rule fires iff n ≥ S, at the first full window's end.
    H = th * (1.0 + 0.5 / (W - 1))
    for extra, fires in ((-1, False), (0, True)):
        F = T.copy()
        n = S + extra
        F[1000:1000 + W + n - 1, J] = H
        ev = L.window_rule(F, T, PW)
        print(f"\nplateau {n} window ends: window {ev[J]}")
        assert (ev[J] is not None) == fires
        if fires:
            assert ev[J] == 1000 + W - 1
