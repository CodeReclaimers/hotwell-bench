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
from dataclasses import dataclass
from types import MappingProxyType

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
    #: × threshold, log-uniform. Also the multiple of D_i for sensor bias and
    #: drift faults (random sign), so a bias of k·D_i is k× the oracle threshold
    #: on its own instrument.
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

PRESETS: Mapping[str, Preset] = MappingProxyType({P01.name: P01})

for _p in PRESETS.values():
    assert _p.sensor_kinds and set(_p.sensor_kinds) <= set(SENSOR_KINDS), _p.name
    _w = (_p.window_s, _p.window_sustain_s, _p.window_thresholds)
    assert all(x is None for x in _w) or all(x is not None for x in _w), _p.name
    _s = (_p.strict_windows_s, _p.strict_sustain_s, _p.strict_thresholds)
    assert all(x is None for x in _s) or all(x is not None for x in _s), _p.name
    assert _p.strict_thresholds is None or set(_p.strict_thresholds) == set(_p.strict_windows_s)
