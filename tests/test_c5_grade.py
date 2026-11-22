"""C5: grader 0.2 (PLAN-bench-0.2.md §6), G-B5v on hand-made keys, run comparison."""
from __future__ import annotations

import json
import math
import sys

import pytest
from test_b5_grade import CASES, KEY_FAULT, KEY_NULL, KEY_PW, ans

from hotwell_bench import grade as G
from hotwell_bench.compare import flips, paired_bootstrap
from hotwell_bench.secrecy import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import regrade_check  # noqa: E402


def v02(key: dict, decoys=(), strict=True) -> dict:
    k = {**key, "decoys": list(decoys)}
    if k["fault_present"]:
        k["strata"] = {**k["strata"], "strict_observable": strict}
    return k


K2_FAULT = v02(KEY_FAULT, ["trim_burst"])
K2_PW = v02({**KEY_PW, "strata": {**KEY_PW["strata"], "ratio": 0.8}}, strict=False)
K2_NULL = v02(KEY_NULL, ["load_excursion", "pump_swap_back"])
K2_NULL0 = v02({**KEY_NULL, "item_id": "dev-0005"})


def test_onset_score():
    """exp(−|onset − t_inj| / 1800 s); 0 for a missing onset or an invalid answer on a
    fault item; None on a null item. 0.1 fields are unchanged by the addition."""
    cases = [("exact", ans(onset="10:00:00"), 1.0),
             ("30 min late", ans(onset="10:30:00"), math.exp(-1)),
             ("1 h early", ans(onset="09:00:00"), math.exp(-2)),
             ("missing onset", ans(onset=None), 0.0),
             ("invalid", ans(conf=3.0), 0.0)]
    for name, a, want in cases:
        g = G.grade_item(K2_FAULT, a)
        g1 = G.grade_item(KEY_FAULT, a)
        print(f"\n{name}: onset_score {g['onset_score']:.6f} (want {want:.6f}), "
              f"onset {g['onset']}, onset_minus_tdet_s {g.get('onset_minus_tdet_s')}")
        assert g["onset_score"] == pytest.approx(want, abs=1e-15)
        assert {k: v for k, v in g.items() if k != "onset_score"} == g1
        assert "onset_score" not in g1
    assert G.grade_item(K2_NULL, ans("none", None, None, False))["onset_score"] is None
    assert G.grade_item(K2_NULL, None)["onset_score"] is None


def test_full_correctness_unchanged_on_v02_keys():
    for name, key, a, full in CASES:
        k2 = v02(key)
        assert G.grade_item(k2, a)["full"] is full, name


def test_v02_strata_and_summary():
    keys = {k["item_id"]: k for k in (K2_FAULT, K2_PW, K2_NULL, K2_NULL0)}
    answers = {i: G.oracle_answer(k) for i, k in keys.items()}
    answers["dev-0001"] = ans(onset="10:30:00")          # outside the window, onset score e^-1
    answers["dev-0003"] = ans()                          # false alarm on a decoy null day
    _, s = G.grade(keys, answers)
    st = s["strata"]
    show = {d: {v: (x["n"], x["detection"]) for v, x in st[d].items()}
            for d in ("decoys", "decoy_kind", "ratio_w", "strict")}
    print(f"\nstrata {json.dumps(show)}\nonset_minus_tdet_s {s['onset_minus_tdet_s']}; "
          f"fault onset_score {s['fault_items']['onset_score']:.6f}")
    print(G.report(s))
    assert s["bench_version"] == "0.2"
    assert set(st["decoys"]) == {"fault, decoy", "fault, no decoy", "null, decoy",
                                 "null, no decoy"}
    assert st["decoys"]["null, decoy"]["detection"] == 0.0
    assert st["decoys"]["null, no decoy"]["detection"] == 1.0
    assert st["decoy_kind"]["null, load_excursion"]["n"] == 1
    assert st["decoy_kind"]["null, pump_swap_back"]["n"] == 1
    assert set(st["decoy_kind"]) == {"fault, trim_burst", "null, load_excursion",
                                     "null, pump_swap_back"}
    assert set(st["ratio_w"]) == {"1.5-2", "<1", "null"}
    assert st["strict"]["lenient only"]["n"] == 1 and st["strict"]["null"]["n"] == 2
    assert s["fault_items"]["onset_score"] == pytest.approx((math.exp(-1) + 1) / 2)
    assert s["null_items"]["onset_score"] is None
    assert s["onset_minus_tdet_s"]["n"] == 2
    assert "oscor" in G.report(s) and "onset - t_det" in G.report(s)


def test_ratio_w_bucket_edges():
    got = [G._ratio_bucket_w(r) for r in (None, 0.7, 0.9999, 1.0, 1.4999, 1.5, 2.0, 2.999, 3.0)]
    print(f"\nratio_w buckets {got}")
    assert got == ["n/a", "<1", "<1", "1-1.5", "1-1.5", "1.5-2", "2-3", "2-3", ">=3"]


def test_mixed_versions_are_refused():
    with pytest.raises(ValueError):
        G.grade({"a": {**KEY_FAULT, "item_id": "a"}, "b": v02({**KEY_NULL, "item_id": "b"})}, {})


def test_v01_keys_regrade_identically_to_v010_grader():
    """G-B5v on hand-made 0.1 keys: the v0.1.0 grader (from git) and the current one give
    the same JSON and report bytes, for oracle, missing, invalid and mixed answers. On
    0.2 keys they differ (the control)."""
    if regrade_check.taxonomy_changed("v0.1.0"):
        pytest.fail("taxonomy.py changed since v0.1.0")
    ref = regrade_check.grader_at("v0.1.0")
    keys = {k["item_id"]: k for k in (KEY_FAULT, KEY_PW, KEY_NULL)}
    sets = {"oracle": {i: G.oracle_answer(k) for i, k in keys.items()},
            "missing": {},
            "mixed": {"dev-0001": ans(onset="10:30:00", conf=0.3), "dev-0002": ans(conf=9),
                      "dev-0003": ans()}}
    for name, a in sets.items():
        o0, o1 = regrade_check.outputs(ref, keys, a), regrade_check.outputs(G, keys, a)
        print(f"\n{name}: json {len(o1[0])} bytes, identical {o0 == o1}")
        assert o0 == o1
    k2 = {k["item_id"]: v02(k) for k in keys.values()}
    assert regrade_check.outputs(ref, k2, sets["oracle"]) != regrade_check.outputs(
        G, k2, sets["oracle"])


def _run(scores: dict[str, bool]) -> list[dict]:
    return [{"item_id": i, "full": v} for i, v in sorted(scores.items())]


def test_flips_and_paired_bootstrap():
    ids = [f"dev-{n:04d}" for n in range(40)]
    a1 = _run({i: n % 2 == 0 for n, i in enumerate(ids)})
    a2 = _run({i: (n % 2 == 0) != (n < 4) for n, i in enumerate(ids)})    # 4 flipped
    b = _run(dict.fromkeys(ids, False))
    f = flips([a1, a2])
    bs = paired_bootstrap([a1, a2], [b, b], n_boot=2000, seed=1)
    same = paired_bootstrap([a1], [a1], n_boot=200)
    print(f"\nflips {f['n_unstable']} {f['pair_flips']}; A-B {bs['diff']:.4f} {bs['ci']}; "
          f"A-A {same['diff']} {same['ci']}")
    assert f["n_unstable"] == 4 and f["pair_flips"] == {"0-1": 4}
    assert bs["diff"] == pytest.approx(0.5) and bs["ci"][0] < 0.5 < bs["ci"][1]
    assert bs["p_diff_le_0"] == 0.0
    assert same["diff"] == 0.0 and same["ci"] == [0.0, 0.0]
    assert paired_bootstrap([a1, a2], [b], n_boot=2000, seed=1) == bs
    with pytest.raises(ValueError):
        flips([a1])
