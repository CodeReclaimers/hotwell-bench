"""B4 / G-B4: packaging, leak scan, mutation tests, byte-identical re-packaging."""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from test_b3_label import gb2_items

from hotwell_bench import package as K
from hotwell_bench.items import sample_item
from hotwell_bench.label import label_runs
from hotwell_bench.presets import P01
from hotwell_bench.run import run_cached, run_items
from hotwell_bench.secrecy import REPO_ROOT
from hotwell_bench.taxonomy import FAULT_CATEGORIES

UNTIL = 600
#: a test-only GUID; the release GUID is a run-time input and never in git
GUID = "00000000-0000-4000-8000-000000000000"
CANARY = K.canary_text(GUID)
PROV = {"environment": {"hotwell": {"requested_revision": "v0.4.0", "commit_id": "c0ffee"},
                        "numpy": "2.5.1"}, "platform": "test", "bench": {"commit": "beef"}}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A null item and an FP-1 −3 % step at 300 s (detected at once), 600 s each."""
    cache = tmp_path_factory.mktemp("cache")
    null = next(it for i in range(20)
                if (it := sample_item(20261107, "dev", i, preset=P01)).fault is None)
    fault = gb2_items()[3]
    items = [null, fault]
    run_items(items, cache, workers=2, until=UNTIL)
    results, rr, twins = {}, {}, {}
    for n, it in enumerate(items):
        fa = run_cached(it, "faulted", cache, until=UNTIL)
        tw = run_cached(it, "twin", cache, until=UNTIL)
        lab = label_runs(it, fa, tw, preset=P01)
        f = it.fault
        results[n] = {"status": "null" if f is None else "accepted",
                      "item": json.loads(it.to_json()), "label": lab.to_dict(),
                      "attempts": [], "confusers": [],
                      "accepted_answers": [{"category": "none" if f is None else f.category,
                                            "component": None if f is None else f.component}]}
        rr[n] = fa
        twins[n] = tw
    seeds = {r["item"]["sim_seed"] for r in results.values()} | {20261107}
    return results, rr, seeds, twins


def _package(tmp: Path, runs) -> Path:
    results, rr, _, _ = runs
    K.package_split("dev", results, rr, tmp, PROV, GUID, version="0.1")
    return tmp / "dev"


def _hashes(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def test_canary_guid_validation_and_wrong_guid_fails_the_scan(tmp_path, runs):
    with pytest.raises(ValueError):
        K.canary_text("not-a-guid")
    d = _package(tmp_path / "g", runs)
    problems = K.leak_scan(d, runs[2], "11111111-1111-4111-8111-111111111111")
    print(f"\nwrong GUID: {len(problems)} problems, e.g. {problems[:2]}")
    assert any("canary" in p for p in problems)


def test_formats():
    assert K.hms(0) == "00:00:00" and K.hms(86399) == "23:59:59"
    assert K.hms(47725.6, 2) == "13:15:25.60" and K.hms(59.999, 2) == "00:01:00.00"
    assert [K.decimals_for(r) for r in (0.005, 0.05, 0.5, 0.1, 0.001, 0.01, 1.0, 1e4)] \
        == [3, 2, 1, 1, 3, 2, 0, 0]


def test_task_statement_is_clean_and_complete():
    t = K.task_md(CANARY, "0.1").decode()
    hits = [m.group(0) for m in K._token_re(K.forbidden_tokens()).finditer(t)]
    print(f"\ntask.md {len(t)} chars; forbidden hits {hits}")
    assert not hits
    assert CANARY in t.split("\n", 1)[0]
    for c in FAULT_CATEGORIES:
        assert f"`{c}`" in t, c


def test_package_passes_scan_and_is_byte_identical(tmp_path, runs):
    """G-B4: the leak scan passes on a packaged split, and re-packaging into a fresh
    directory reproduces every file byte for byte."""
    a = _package(tmp_path / "a", runs)
    b = _package(tmp_path / "b", runs)
    problems = K.leak_scan(a, runs[2], GUID)
    ha, hb = _hashes(a), _hashes(b)
    print(f"\nfiles {len(ha)}; scan problems {problems}")
    for k in sorted(ha):
        print(f"  {k:45s} {ha[k][:16]}")
    assert problems == []
    assert ha == hb
    key = json.loads((a / "keys" / "dev-9603.json").read_text())
    print(f"  key dev-9603: window {key['onset_window']} t_det {key['t_det']}")
    assert key["onset_window"] == [300 - 600, key["t_det"] + 600]
    man = json.loads((a / "manifest.json").read_text())
    assert not any(k.startswith("keys/") for k in man["files"])
    assert "accepted_answers" not in json.dumps(man)


def _mutations():
    def append(rel, text):
        def f(d: Path):
            p = d / rel
            p.write_text(p.read_text() + text)
        return f

    def rewrite_csv(fn):
        def f(d: Path):
            p = next((d / "items").iterdir()) / "instruments.csv.gz"
            p.write_bytes(gzip.compress(fn(gzip.decompress(p.read_bytes()).decode()).encode()))
        return f

    def first_item(d: Path) -> Path:
        return next(p for p in sorted((d / "items").iterdir()))

    def item_append(name, text):
        def f(d: Path):
            p = first_item(d) / name
            p.write_text(p.read_text() + text)
        return f

    def drop_canary(d: Path):
        p = first_item(d) / "shift_log.txt"
        p.write_text(p.read_text().split("\n", 1)[1])

    def extra_file(d: Path):
        (first_item(d) / "notes.txt").write_text("x")

    return {
        "event name in shift log": (item_append("shift_log.txt",
                                    "12:00:00.00  NOP-03 done: pump_degrade\n"), "forbidden"),
        "truth signal in task.md": (append("task.md", "m_leak\n"), "forbidden"),
        "seed in events": (item_append("events.csv", "12:00:00.00,43200.00,alarm,"
                                       "FREQ_LO,{seed}\n"), "seed"),
        "unknown event id": (item_append("events.csv",
                                         "12:00:00.00,43200.00,alarm,FAULT_X,active\n"),
                             "not whitelisted"),
        "unknown procedure": (item_append("shift_log.txt", "12:00:00.00  XOP-9 done: x\n"),
                              "unknown procedure"),
        "item task.md edited": (item_append("task.md", "hint\n"), "differs"),
        "extra column": (rewrite_csv(lambda s: s.replace("t_s,", "t_s,fault,", 1)), "header"),
        "text in csv body": (rewrite_csv(lambda s: s + "1,pump_degrade\n"), "non-numeric"),
        "decoy kind in shift log": (item_append("shift_log.txt",
                                    "12:00:00.00  NOP-03 done: trim_burst\n"), "forbidden"),
        "decoy parameter in shift log": (item_append("shift_log.txt",
                                         "12:00:00.00  NOP-01 done: back_s\n"), "forbidden"),
        "canary missing": (drop_canary, "canary"),
        "extra file": (extra_file, "files"),
    }


@pytest.mark.parametrize("name", list(_mutations()))
def test_mutation_fails_the_scan(tmp_path, runs, name):
    """G-B4 mutation test: each planted leak is caught, with the expected message."""
    base = _package(tmp_path / "base", runs)
    d = tmp_path / "mut"
    shutil.copytree(base, d)
    fn, expect = _mutations()[name]
    seed = sorted(runs[2])[0]
    if name == "seed in events":
        p = next(p for p in sorted((d / "items").iterdir())) / "events.csv"
        p.write_text(p.read_text() + f"12:00:00.00,43200.00,alarm,FREQ_LO,{seed}\n")
    else:
        fn(d)
    problems = K.leak_scan(d, runs[2], GUID)
    print(f"\n{name}: {problems[:3]}")
    assert any(expect in p for p in problems), problems


def test_test_split_package_is_refused_inside_the_repository(runs):
    results, rr, _, _ = runs
    with pytest.raises(PermissionError):
        K.package_split("test", results, rr, REPO_ROOT / "out" / "pkg", PROV, GUID,
                        version="0.1")



def test_v02_package_text_keys_and_scan(tmp_path, runs):
    """C4 (PLAN-bench-0.2.md §5): a split labelled under P02 packages as bench 0.2.
    task.md carries the 0.2 load sentence; every key has `decoys`; fault keys carry
    `strata.strict_observable`; the manifest records the version; the scan passes,
    re-packaging is byte-identical, and a 0.1 task.md inside a 0.2 split is flagged."""
    from hotwell_bench.items import ItemRecord
    from hotwell_bench.presets import P02
    results, rr, seeds, twins = runs
    r2 = {}
    for n, r in results.items():
        it = ItemRecord.from_json(json.dumps(r["item"]))
        r2[n] = dict(r, label=label_runs(it, rr[n], twins[n], preset=P02).to_dict())
    dirs = []
    for sub in ("a", "b"):
        K.package_split("dev", r2, rr, tmp_path / sub, PROV, GUID, version="0.2")
        dirs.append(tmp_path / sub / "dev")
    a, b = dirs
    problems = K.leak_scan(a, seeds, GUID)
    man = json.loads((a / "manifest.json").read_text())
    task = (a / "task.md").read_text()
    keys = {p.name: json.loads(p.read_text()) for p in (a / "keys").glob("dev-*.json")}
    print(f"\n0.2 package: scan {problems}; bench_version {man.get('bench_version')}; keys "
          + "; ".join(f"{k}: decoys {v['decoys']} strata {v.get('strata')}"
                      for k, v in sorted(keys.items())))
    assert problems == [] and _hashes(a) == _hashes(b)
    assert man["bench_version"] == "0.2" and "customer demand" in task
    assert task.encode() == K.task_md(CANARY, "0.2") != K.task_md(CANARY, "0.1")
    assert all(v["decoys"] == [] for v in keys.values())
    assert all("strict_observable" in v["strata"] for v in keys.values() if v["fault_present"])
    p = next(p for p in sorted((a / "items").iterdir())) / "task.md"
    p.write_bytes(K.task_md(CANARY, "0.1"))
    assert any("differs" in x for x in K.leak_scan(a, seeds, GUID))
