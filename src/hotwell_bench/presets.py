"""Constants copied from hotwell v0.4.0, plus bench-owned sampling ranges.

The noise presets live in hotwell's `scripts/quiet_day.py`, which is not part of
the installed package, so they are copied here verbatim. The copy is checked
against a hotwell checkout by `scripts/check_upstream_copies.py PATH`.

PROVISIONAL by decision (PLAN-bench-0.1.md §11, B-Q6): plausible orders of
magnitude, not calibrated instrument data. One documented artifact: noise on an
instrument resting at its range floor biases it upward. `makeup_flow` (σ 0.05,
LSB 0.1, floor 0) reads up to 0.2 kg/s with the valve shut; task.md says so.
"""
from __future__ import annotations

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

# --------------------------------------------------------------- sampling ranges
# PLAN-bench-0.1.md §3. Times in s, power in MW (gross).
P_FLAT_DAY = 0.2                         # fraction of days held flat at 60 MW
TROUGH_MW = (36.0, 48.0)
RAMP_DOWN_START_S = (1 * 3600.0, 5 * 3600.0)
RAMP_S = (1800.0, 5400.0)                # each ramp 30–90 min
TROUGH_LEN_S = (4 * 3600.0, 8 * 3600.0)

NOP03_CADENCE_S = (240, 420)             # integer seconds
P_SWAP = 0.7                             # per pump pair (CP, FP)
SWAP_WINDOW_S = (2 * 3600, 22 * 3600)
REACTION_S = 5.0                         # ProcedureRunner default
#: lognormal sigma of the operator's reaction delay. Provisional: the plan asks
#: only for sigma > 0. B3 must check whether reaction-time divergence between
#: twins makes t_det spuriously early.
REACTION_SIGMA = 0.3

P_NULL = 0.2                             # fault-free items
ONSET_S = (2 * 3600, 20 * 3600)          # t_inj window, integer seconds
P_RAMP = 0.5                             # ramped vs step onset (factor faults)
ONSET_RAMP_S = (600.0, 6 * 3600.0)       # log-uniform
MAG_RANGE = (0.5, 4.0)                   # × threshold, log-uniform

#: Step detectability thresholds, AB + 3E column of hotwell
#: scripts/fault_detectability.py at v0.4.0 (HANDOFF "Notes for the bench phase"),
#: expressed as the fault's DEVIATION from healthy: 1 − factor for degradations,
#: factor − 1 for fouling, kg/s for leaks. "swept": the smallest swept magnitude
#: that was detected (an upper bound on the true threshold). "assumed": the
#: component was not swept and borrows the named sibling's value.
THRESHOLDS: dict[tuple[str, str], tuple[float, str]] = {
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
}
#: actuator_slow was never detected at constant load (any factor 0.05–0.5); it is
#: sampled across the swept range instead of from a threshold.
SLOW_RATE_FACTOR = (0.05, 0.5)           # log-uniform

# Sensor faults (never swept in hotwell). Instrument detection scale
# D_i = max(3 σ_i, 2 LSB_i), the same bound the twin criterion uses, so a bias of
# k·D_i is k× the oracle threshold on its own instrument. Provisional; B3 measures.
SENSOR_BIAS_X = MAG_RANGE                # × D_i, random sign
SENSOR_DRIFT_TIME_S = (600.0, 6 * 3600.0)  # time to reach the final bias, log-uniform
SENSOR_LAG_TAU_S = (5.0, 300.0)          # log-uniform
SENSOR_NOISE_FACTOR = (2.0, 10.0)        # log-uniform
