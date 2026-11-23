"""Constants copied from hotwell v0.4.0, plus bench-owned generation presets.

The noise presets live in hotwell's `scripts/quiet_day.py`, which is not part of
the installed package, so they are copied here verbatim. The copy is checked
against a hotwell checkout by `scripts/check_upstream_copies.py PATH`.

PROVISIONAL by decision (PLAN-bench-0.1.md §11, B-Q6): plausible orders of
magnitude, not calibrated instrument data. One documented artifact: noise on an
instrument resting at its range floor biases it upward. `makeup_flow` (σ 0.05,
LSB 0.1, floor 0) reads up to 0.2 kg/s with the valve shut; task.md says so.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from hotwell_bench import window_detectability as _WD
from hotwell_bench import window_thresholds as _WT
from hotwell_bench.taxonomy import SENSOR_KINDS

#: hotwell scripts/quiet_day.py::QUIET_DAY_SENSOR_NOISE at v0.4.0 (5f57636), verbatim.
#: sigma in instrument units; tau [s] (0 = white).
SENSOR_NOISE = {
    "drum_pressure": {"sigma": 1.5e4, "tau": 0.5},
    "chest_pressure": {"sigma": 1.5e4, "tau": 0.5},
    "drum_level": {"sigma": 3.0e-3, "tau": 1.0},
    "steam_flow": {"sigma": 0.3, "tau": 0.5},
    "feed_flow": {"sigma": 0.3, "tau": 0.5},
    "condensate_flow": {"sigma": 0.3, "tau": 0.5},
    "cond_pressure": {"sigma": 50.0, "tau": 0.0},
    "hotwell_level": {"sigma": 2.0e-3, "tau": 1.0},
    "da_level": {"sigma": 2.0e-3, "tau": 1.0},
    "da_pressure": {"sigma": 500.0, "tau": 0.0},
    "frequency": {"sigma": 0.005, "tau": 0.0},
    "mw_out": {"sigma": 5.0e4, "tau": 0.0},
    "cp1_current": {"sigma": 0.5, "tau": 0.0},
    "cp2_current": {"sigma": 0.5, "tau": 0.0},
    "fp1_current": {"sigma": 1.0, "tau": 0.0},
    "fp2_current": {"sigma": 1.0, "tau": 0.0},
    "fw_valve_pos": {"sigma": 0.002, "tau": 0.0},
    "throttle_pos": {"sigma": 0.002, "tau": 0.0},
    "firing_rate": {"sigma": 0.002, "tau": 0.0},
    "da_lcv_pos": {"sigma": 0.002, "tau": 0.0},
    "fp1_npsh_margin": {"sigma": 0.03, "tau": 0.0},
    "fp2_npsh_margin": {"sigma": 0.03, "tau": 0.0},
    "makeup_flow": {"sigma": 0.05, "tau": 0.0},
    "makeup_valve_pos": {"sigma": 0.002, "tau": 0.0},
    "cp1_npsh_margin": {"sigma": 0.03, "tau": 0.0},
    "cp2_npsh_margin": {"sigma": 0.03, "tau": 0.0},
}

#: hotwell scripts/quiet_day.py::QUIET_DAY_LOAD_NOISE at v0.4.0, verbatim.
LOAD_NOISE = {
    "nonvital": {"sigma_w": 3.0e5, "tau": 120.0},
    "vital": {"sigma_w": 5.0e4, "tau": 300.0},
}

#: B-Q7: AB drum + three-element feedwater only.
DRUM_MODEL = "astrom_bell"
THREE_ELEMENT = True
DAY_S = 86400.0
RATED_MW = 60.0

# --------------------------------------------------------------- generation presets


@dataclass(frozen=True)
class DecoyParams:
    """Decoy events (PLAN-bench-0.2.md §3.2, B2-Q1, B2-Q13): real, logged or visible
    non-fault events, on fault and null days alike. Ranges are uniform; times in s."""
    p_decoy: float                           # items with at least one decoy
    p_two: float                             # of those, items with two (distinct kinds)
    kinds: tuple[str, ...]
    #: load_excursion: customer (non-vital) load drop of `mw`, ramped in `ramp_s`,
    #: held `hold_s`, ramped back. Placed outside the daily trough ± `trough_margin_s`
    #: and inside [`edge_s`, DAY_S − `edge_s`].
    excursion_mw: tuple[float, float]
    excursion_ramp_s: tuple[float, float]
    excursion_hold_s: tuple[float, float]
    trough_margin_s: float
    edge_s: float
    #: pump_swap_back: a pump pair is swapped and swapped back `back_s` later. If the
    #: day already has a swap on that pair, only the swap back is added.
    swap_back_s: tuple[int, int]
    #: trim_burst: `n` NOP-03 requests `gap_s` apart.
    trim_burst_n: tuple[int, int]
    trim_burst_gap_s: tuple[int, int]


DECOY_KINDS = ("load_excursion", "pump_swap_back", "trim_burst")


@dataclass(frozen=True)
class Preset:
    """Everything that sets item difficulty: sampling ranges and labeller constants.

    A preset is passed explicitly to the sampler (`items.sample_item`), the
    labeller (`label.label_runs`) and the confuser check. A split is generated
    under exactly one preset. The plant constants above (noise, drum model, day
    length) are not part of it: they are hotwell configuration, shared by every
    preset. Times in s, power in MW (gross).
    """
    name: str

    # day profile (PLAN-bench-0.1.md §3)
    p_flat_day: float                        # fraction of days held flat at 60 MW
    trough_mw: tuple[float, float]
    ramp_down_start_s: tuple[float, float]
    ramp_s: tuple[float, float]              # each ramp
    trough_len_s: tuple[float, float]

    # operator plan
    nop03_cadence_s: tuple[int, int]         # integer seconds
    p_swap: float                            # per pump pair (CP, FP)
    swap_window_s: tuple[int, int]
    reaction_s: float                        # ProcedureRunner default
    #: lognormal sigma of the operator's reaction delay. Provisional: the plan asks
    #: only for sigma > 0. B3 must check whether reaction-time divergence between
    #: twins makes t_det spuriously early.
    reaction_sigma: float

    # faults
    p_null: float                            # fault-free items
    onset_s: tuple[int, int]                 # t_inj window, integer seconds
    p_ramp: float                            # ramped vs step onset (factor faults)
    onset_ramp_s: tuple[float, float]        # log-uniform
    #: × threshold, log-uniform. Also the multiple of the sensor bias scale (D_i, or
    #: `sensor_bias_thresholds` when set) for bias and drift faults (random sign).
    mag_range: tuple[float, float]
    #: step detectability thresholds per (mechanism, target): (deviation, source)
    thresholds: Mapping[tuple[str, str], tuple[float, str]]
    slow_rate_factor: tuple[float, float]    # actuator_slow, log-uniform
    sensor_drift_time_s: tuple[float, float]  # time to reach the final bias, log-uniform
    sensor_lag_tau_s: tuple[float, float]    # log-uniform
    sensor_noise_factor: tuple[float, float]  # log-uniform
    #: sensor fault kinds sampled for instrument_fault, uniformly (a subset of
    #: taxonomy.SENSOR_KINDS)
    sensor_kinds: tuple[str, ...]

    # labeller (label.py describes the rules)
    level_run: int
    var_window: int
    var_sustain: int
    stuck_factor: int
    stuck_margin: int
    var_lo: float
    var_hi: float
    #: windowed level rule (PLAN-bench-0.2.md §2.1); all three None disables it.
    #: window_s: trailing-mean window W [samples = s]; window_sustain_s: S;
    #: window_thresholds: θ_i per instrument, in instrument units.
    window_s: int | None = None
    window_sustain_s: int | None = None
    window_thresholds: Mapping[str, float] | None = None
    #: strict per-instrument stratum (Alan, 2026-11-21): stuck + variance + windowed
    #: rules at every W in strict_windows_s with θ_{W,i} = strict_thresholds[W][i],
    #: WITHOUT the per-sample level rule. Recorded on each label (Label.strict);
    #: never decides observability. All three None disables it.
    strict_windows_s: tuple[int, ...] | None = None
    strict_sustain_s: int | None = None
    strict_thresholds: Mapping[int, Mapping[str, float]] | None = None
    #: sensor fault scales from the C2 sweep (window_detectability.py), per instrument:
    #: (threshold or None, source). None (P01) keeps the 0.1 scales: D_i for bias and
    #: drift, `sensor_noise_factor` for noise_increase. When set:
    #: - bias and drift take b = ±r·thr_i with r from `mag_range`;
    #: - noise_increase takes factor = 1 + r·(thr_i − 1) (B2-Q14);
    #: - an instrument whose threshold is None is not sampled for that kind (bias and
    #:   drift share the bias threshold; B2-Q11, B2-Q15).
    sensor_bias_thresholds: Mapping[str, tuple[float | None, str]] | None = None
    sensor_noise_thresholds: Mapping[str, tuple[float | None, str]] | None = None
    #: decoy events; None (P01) samples none and draws nothing.
    decoys: DecoyParams | None = None


#: Step detectability thresholds, AB + 3E column of hotwell
#: scripts/fault_detectability.py at v0.4.0 (HANDOFF "Notes for the bench phase"),
#: expressed as the fault's DEVIATION from healthy: 1 − factor for degradations,
#: factor − 1 for fouling, kg/s for leaks. "swept": the smallest swept magnitude
#: that was detected (an upper bound on the true threshold). "assumed": the
#: component was not swept and borrows the named sibling's value.
THRESHOLDS_V040: Mapping[tuple[str, str], tuple[float, str]] = MappingProxyType({
    ("pump_degrade", "fp1"): (0.005, "swept"),
    ("pump_degrade", "fp2"): (0.005, "assumed: fp1"),
    ("pump_degrade", "cp1"): (0.01, "swept"),
    ("pump_degrade", "cp2"): (0.01, "assumed: cp1"),
    ("valve_degrade", "frv"): (0.02, "swept (0.01 not detected)"),
    ("valve_degrade", "lcv"): (0.02, "swept (0.01 not detected)"),
    ("valve_degrade", "makeup_valve"): (0.02, "assumed: frv/lcv; valve normally shut"),
    ("firing_degrade", ""): (0.01, "swept (0.005 not detected)"),
    ("condenser_fouling", ""): (0.25, "swept (0.1 not detected)"),
    ("tube_leak", ""): (0.1, "swept (smallest swept, detected at 689 s)"),
})

#: The 0.1 preset (PLAN-bench-0.1.md §3, §4). The 0.1 splits were generated with
#: these values; regenerating the 0.1 dev split under P01 must reproduce its
#: manifest hash (PLAN-bench-0.2.md G-B0r).
P01 = Preset(
    name="P01",
    p_flat_day=0.2,
    trough_mw=(36.0, 48.0),
    ramp_down_start_s=(1 * 3600.0, 5 * 3600.0),
    ramp_s=(1800.0, 5400.0),                 # each ramp 30–90 min
    trough_len_s=(4 * 3600.0, 8 * 3600.0),
    nop03_cadence_s=(240, 420),
    p_swap=0.7,
    swap_window_s=(2 * 3600, 22 * 3600),
    reaction_s=5.0,
    reaction_sigma=0.3,
    p_null=0.2,
    onset_s=(2 * 3600, 20 * 3600),
    p_ramp=0.5,
    onset_ramp_s=(600.0, 6 * 3600.0),
    mag_range=(0.5, 4.0),
    thresholds=THRESHOLDS_V040,
    #: actuator_slow was never detected at constant load (any factor 0.05–0.5); it
    #: is sampled across the swept range instead of from a threshold.
    slow_rate_factor=(0.05, 0.5),
    # Sensor faults (never swept in hotwell); scale D_i = max(3 σ_i, 2 LSB_i), the
    # same bound the twin criterion uses. Provisional; B3 measures.
    sensor_drift_time_s=(600.0, 6 * 3600.0),
    sensor_lag_tau_s=(5.0, 300.0),
    sensor_noise_factor=(2.0, 10.0),
    sensor_kinds=("bias", "drift", "stuck", "lag", "noise_increase"),
    level_run=10,
    var_window=600,
    var_sustain=60,
    stuck_factor=2,
    stuck_margin=60,
    var_lo=0.8,
    var_hi=1.25,
)

#: The 0.2 preset, built up milestone by milestone (PLAN-bench-0.2.md §9) and frozen
#: at the params tag. Differences from P01 so far:
#: - C1: `lag` is not sampled (§2.2);
#: - C1: the windowed level rule takes part in observability and t_det (§2.1), and the
#:   strict per-instrument stratum is recorded (Alan, 2026-11-21). Thresholds come
#:   from window_thresholds.py (generated by scripts/window_thresholds.py).
#: - C3: magnitudes scale on the C2 sweep's thresholds (window_detectability.py,
#:   generated by scripts/window_sweep.py) at 0.7–3× (§3.1); P_RAMP 0.7 and ramps of
#:   30 min–8 h (§3.1); P_NULL 0.30 (B2-Q7); decoys (§3.2, B2-Q13).
#: - C6 round 2 (§7; Alan, 2026-11-23): MAG_RANGE 0.7–3 → 0.3–1.5 and the decoy
#:   fraction 1/3 → 1/2, because round 1 was too easy (log-uniform median ratio
#:   1.45 → 0.67). P_RAMP and ONSET_RAMP_S unchanged.
P02 = replace(
    P01,
    name="P02",
    p_null=0.30,
    p_ramp=0.7,
    onset_ramp_s=(1800.0, 8 * 3600.0),
    mag_range=(0.3, 1.5),
    thresholds=_WD.THRESHOLDS_V020,
    sensor_bias_thresholds=_WD.SENSOR_BIAS_THRESHOLD,
    sensor_noise_thresholds=_WD.SENSOR_NOISE_THRESHOLD,
    # C3: decoys (§3.2; B2-Q13: fault and null days alike). p_decoy was the plan's
    # proposed 1/3 in C6 round 1; 1/2 from round 2 (a calibration knob, §7).
    decoys=DecoyParams(
        p_decoy=0.5,
        p_two=0.5,
        kinds=DECOY_KINDS,
        excursion_mw=(5.0, 15.0),
        excursion_ramp_s=(300.0, 900.0),
        excursion_hold_s=(1200.0, 5400.0),
        trough_margin_s=1800.0,
        edge_s=3600.0,
        swap_back_s=(900, 3600),
        trim_burst_n=(4, 8),
        trim_burst_gap_s=(30, 90),
    ),
    sensor_kinds=("bias", "drift", "stuck", "noise_increase"),
    window_s=_WT.WINDOW_S,
    window_sustain_s=_WT.SUSTAIN_S,
    window_thresholds=MappingProxyType(dict(_WT.WINDOW_THETA)),
    strict_windows_s=_WT.STRICT_WINDOWS_S,
    strict_sustain_s=_WT.SUSTAIN_S,
    strict_thresholds=MappingProxyType({w: MappingProxyType(dict(t))
                                        for w, t in _WT.STRICT_THETA.items()}),
)

PRESETS: Mapping[str, Preset] = MappingProxyType({P01.name: P01, P02.name: P02})

for _p in PRESETS.values():
    assert _p.sensor_kinds and set(_p.sensor_kinds) <= set(SENSOR_KINDS), _p.name
    _w = (_p.window_s, _p.window_sustain_s, _p.window_thresholds)
    assert all(x is None for x in _w) or all(x is not None for x in _w), _p.name
    _s = (_p.strict_windows_s, _p.strict_sustain_s, _p.strict_thresholds)
    assert all(x is None for x in _s) or all(x is not None for x in _s), _p.name
    assert _p.strict_thresholds is None or set(_p.strict_thresholds) == set(_p.strict_windows_s)
    assert all(v[0] is not None and v[0] > 0 for v in _p.thresholds.values()), _p.name
    for _t in (_p.sensor_bias_thresholds, _p.sensor_noise_thresholds):
        assert _t is None or set(_t) == set(SENSOR_NOISE), _p.name
    assert _p.decoys is None or (set(_p.decoys.kinds) <= set(DECOY_KINDS)
                                 and len(_p.decoys.kinds) >= 2), _p.name
    assert (_p.sensor_noise_thresholds is None
            or all(v[0] is None or v[0] > 1 for v in _p.sensor_noise_thresholds.values()))
