"""B5: grader rules on hand-made keys (edge cases of validation and matching)."""
from __future__ import annotations

import pytest

from hotwell_bench.grade import grade, grade_item, onset_seconds, oracle_answer, validate

KEY_FAULT = {"item_id": "dev-0001", "fault_present": True, "t_inj": 36000, "t_det": 36500,
             "onset_window": [35400, 37100], "easy": False,
             "accepted_answers": [{"category": "pump_performance_loss", "component": "CP-1"},
                                  {"category": "valve_capacity_loss", "component": "DA LCV"}],
             "strata": {"category": "pump_performance_loss", "shape": "step", "ratio": 1.5,
                        "sensor_kind": None}}
KEY_PW = {**KEY_FAULT, "item_id": "dev-0002",
          "accepted_answers": [{"category": "boiler_tube_leak", "component": None}],
          "strata": {**KEY_FAULT["strata"], "category": "boiler_tube_leak"}}
KEY_NULL = {"item_id": "dev-0003", "fault_present": False, "t_inj": None, "t_det": None,
            "onset_window": None, "easy": False,
            "accepted_answers": [{"category": "none", "component": None}]}


def ans(cat="pump_performance_loss", comp="CP-1", onset="10:00:00", fp=True, conf=0.5):
    return {"fault_present": fp, "category": cat, "component": comp, "onset": onset,
            "confidence": conf}


CASES = [
    ("exact", KEY_FAULT, ans(), True),
    ("confuser accepted", KEY_FAULT, ans("valve_capacity_loss", "DA LCV"), True),
    ("component case and space", KEY_FAULT, ans(comp=" cp-1 "), True),
    ("wrong component", KEY_FAULT, ans(comp="CP-2"), False),
    ("confuser category, wrong component", KEY_FAULT, ans("valve_capacity_loss", "FRV"), False),
    ("window low edge 09:50:00", KEY_FAULT, ans(onset="09:50:00"), True),
    ("just before window", KEY_FAULT, ans(onset="09:49:59"), False),
    ("window high edge 10:18:20", KEY_FAULT, ans(onset="10:18:20"), True),
    ("just after window", KEY_FAULT, ans(onset="10:18:21"), False),
    ("onset missing", KEY_FAULT, ans(onset=None), False),
    ("fault_present false", KEY_FAULT, ans(fp=False), False),
    ("plant-wide null", KEY_PW, ans("boiler_tube_leak", None), True),
    ("plant-wide dash", KEY_PW, ans("boiler_tube_leak", "—"), True),
    ("plant-wide with component", KEY_PW, ans("boiler_tube_leak", "FP-1"), False),
    ("null correct", KEY_NULL, ans("none", None, None, False), True),
    ("null, onset ignored", KEY_NULL, ans("none", None, "01:00:00", False), True),
    ("null, fault claimed", KEY_NULL, ans(), False),
    ("null, none but fault_present", KEY_NULL, ans("none", None, None, True), False),
]


@pytest.mark.parametrize("name,key,a,full", CASES, ids=[c[0] for c in CASES])
def test_full_correctness(name, key, a, full):
    g = grade_item(key, a)
    print(f"\n{name}: {g}")
    assert g["valid"] and g["full"] is full


@pytest.mark.parametrize("bad", [
    None, [], {"fault_present": True},
    ans(cat="pump_failure"), ans(fp="yes"), ans(conf=1.5), ans(conf=True),
    ans(conf=float("nan")), ans(onset="25:00:00"), ans(onset="10:00"), ans(comp=3)])
def test_invalid_answers_score_zero(bad):
    g = grade_item(KEY_FAULT, bad)
    print(f"\n{bad!r}: {g['error']}")
    assert not g["valid"] and not g["full"] and g["brier"] is None


def test_onset_parse_and_brier():
    assert onset_seconds("00:00:00") == 0 and onset_seconds("23:59:59") == 86399
    assert onset_seconds("10:00:00.5") == 36000
    assert validate(ans())[1] is None
    g = grade_item(KEY_FAULT, ans(conf=0.8))
    assert g["brier"] == pytest.approx(0.04)


def test_oracle_and_missing_answers():
    keys = {k["item_id"]: k for k in (KEY_FAULT, KEY_PW, KEY_NULL)}
    graded, s = grade(keys, {i: oracle_answer(k) for i, k in keys.items()})
    assert s["overall"]["full"] == 1.0
    graded, s = grade(keys, {})
    print(f"\nall missing: {s['overall']}")
    assert s["overall"]["full"] == 0.0 and s["overall"]["invalid"] == 3
