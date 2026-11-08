"""B3 confuser check: candidate table, fault mapping, and (slow) gate G-B3."""
from __future__ import annotations

import pytest

from hotwell_bench.build import build
from hotwell_bench.confusers import (
    Candidate,
    candidates,
    check_item,
    confuser_fault,
    running_at,
)
from hotwell_bench.items import SCHEMA, DayProfile, Fault, ItemRecord, OperatorPlan
from hotwell_bench.run import RunResult, run_built

FLAT = DayProfile(True, 48.0, 3600.0, 1800.0, 14400.0, 1800.0)


def _fp1(head: float, t_inj=600) -> Fault:
    return Fault(category="pump_performance_loss", component="FP-1", mechanism="pump_degrade",
                 target="fp1", t_inj=t_inj, shape="step", params={"head_factor": head},
                 magnitude=1 - head)


def _item(fault, fp_swap=None, index=9700) -> ItemRecord:
    return ItemRecord(SCHEMA, "dev", index, 0, 42, FLAT,
                      OperatorPlan(86400, None, fp_swap, 5.0, 0.0), fault)


def _fake_run(events) -> RunResult:
    import numpy as np
    return RunResult("x", "faulted", np.arange(2), np.zeros((2, 26)), events, [], {})


def test_running_pump_follows_swap_indications():
    ev = [{"t": 100.0, "kind": "indication", "id": "fp2_running", "value": True},
          {"t": 180.0, "kind": "indication", "id": "fp1_running", "value": False}]
    r = _fake_run(ev)
    assert running_at(r, 50, "fp") == "fp1"
    assert running_at(r, 200, "fp") == "fp2"
    assert running_at(r, 200, "cp") == "cp1"


def test_candidate_table():
    fa = _fake_run([])
    cases = {
        ("pump_degrade", "fp1", None): [Candidate("valve_degrade", "frv")],
        ("pump_degrade", "fp2", None): [],          # standby at onset: no settled signature
        ("valve_degrade", "lcv", None): [Candidate("pump_degrade", "cp1")],
        ("firing_degrade", "", None): [Candidate("sensor_fault", "firing_rate", "bias")],
        ("actuator_stuck", "throttle", None): [Candidate("sensor_fault", "throttle_pos",
                                                         "stuck", fitted=False)],
        ("sensor_fault", "da_lcv_pos", "stuck"): [Candidate("actuator_stuck", "da_lcv",
                                                            fitted=False)],
        ("tube_leak", "", None): [],
    }
    for (m, t, k), want in cases.items():
        f = Fault(category="x", component=None, mechanism=m, target=t, t_inj=10, shape="step",
                  params={"kind": k} if k else {}, sensor_kind=k)
        got = candidates(_item(f), fa)
        print(f"{m} {t} {k}: {got}")
        assert got == want


def test_confuser_fault_keeps_onset_and_shape():
    ramped = Fault(category="combustion_efficiency_loss", component=None,
                   mechanism="firing_degrade", target="", t_inj=5000, shape="ramp",
                   params={"eta_factor": 0.98, "ramp_s": 1200.0}, magnitude=0.02)
    cf = confuser_fault(ramped, Candidate("sensor_fault", "firing_rate", "bias"), 0.01)
    print(f"\n{cf}")
    assert cf.t_inj == 5000 and cf.sensor_kind == "drift"
    assert cf.params["max_bias"] == 0.01 and cf.params["rate"] == pytest.approx(0.01 / 1200)
    back = confuser_fault(cf, Candidate("firing_degrade", ""), 0.02)
    assert back.params == {"eta_factor": 0.98, "ramp_s": 1200.0}
    assert back.category == "combustion_efficiency_loss" and back.component is None


@pytest.mark.slow
@pytest.mark.parametrize(("name", "head", "fp_swap", "want"), [
    ("FP-1 -0.5 %, no swap", 0.995, None, True),
    ("FP-1 -0.5 %, FP swap after onset", 0.995, 2400, False),
    ("FP-1 -3 %, no swap (current separates)", 0.97, None, False),
])
def test_gb3_fp1_vs_frv(name, head, fp_swap, want):
    """G-B3: a seeded known confusion is found confusable; the same pair with an FP
    swap after onset is separable (the swap restores the FRV only if the pump was at
    fault). Near threshold only: at -3 % the FP-1 current difference (10 A against
    D = 3 A, hotwell M5) separates the pair even without a swap."""
    it = _item(_fp1(head), fp_swap=fp_swap)
    b = build(it)
    fa, tw = run_built(b, "faulted", until=4200), run_built(b, "twin", until=4200)
    res = check_item(it, fa, tw)
    for r in res:
        print(f"\n{name}: {r.category}/{r.component} {r.params} -> confusable {r.confusable} "
              f"({r.reason}); signature {r.signature} shift {r.target_shift} achieved "
              f"{r.achieved_shift} in {r.evaluations} evals; separated by {r.separated_by}")
    assert len(res) == 1 and res[0].category == "valve_capacity_loss"
    assert res[0].confusable is want
