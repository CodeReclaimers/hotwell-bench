"""B3 pipeline mechanics (slow): regeneration keeps category/component, the cap
excludes, nulls are labelled once, outputs are written per index."""
from __future__ import annotations

import json

import pytest

from hotwell_bench.items import sample_item
from hotwell_bench.pipeline import label_indices, summarize

MASTER = 20261107


@pytest.mark.slow
def test_pipeline_regenerates_excludes_and_writes(tmp_path):
    """Runs are cut to 3 h, so most onsets (U(2 h, 20 h)) fall outside the run: those
    items are unobservable and exercise regeneration and the cap."""
    idx = [0, 1, 2, 3, 4, 5]
    res = label_indices(MASTER, "dev", idx, tmp_path / "cache", tmp_path / "labels",
                        workers=12, cap=2, until=10800)
    print("\n" + summarize(res))
    for i in idx:
        r = res[i]
        written = json.loads((tmp_path / "labels" / f"dev-{i:04d}.json").read_text())
        assert written["status"] == r["status"]
        s0 = sample_item(MASTER, "dev", i, 0)
        if s0.fault is None:
            assert r["status"] == "null" and len(r["attempts"]) == 1
            continue
        cats = {(r["item"]["fault"]["category"], r["item"]["fault"]["component"])}
        assert cats == {(s0.fault.category, s0.fault.component)}
        assert 1 <= len(r["attempts"]) <= 2
        if r["status"] == "accepted":
            assert r["attempts"][-1]["observable"] is True
            assert r["accepted_answers"][0] == {"category": s0.fault.category,
                                                "component": s0.fault.component}
        else:
            assert r["status"] == "excluded_at_cap"
            assert all(a["observable"] is False for a in r["attempts"])
