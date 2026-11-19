"""Labelling pipeline with regeneration (B3; PLAN-bench-0.1.md §4.2, B-Q5).

For each index: attempt 0 is run (faulted + twin) and labelled. A fault item with
no evidence by any rule (unobservable) is regenerated: the next attempt keeps its
category and component (STRUCTURE stream) and resamples background and details.
This repeats up to ATTEMPT_CAP attempts. An index still unobservable at the cap is
recorded as `excluded_at_cap`, not filled. How to fill a split is decided when it
is frozen (B6). Accepted fault items then get the confuser check. Fault-free
items are labelled once.

Scheduling (per-item pipelining, PLAN-bench-0.2.md §4). One process pool runs every
simulation, and each item advances as soon as its own previous step finishes:
faulted run → twin run → label (in this process) → either the next attempt's faulted
run, or the confuser check for an accepted item, or done. 0.1 ran barrier-synchronized
rounds (every attempt-a run, then every label, then attempt a + 1, confusers last), which
left cores idle while the slowest item of a round finished. Results do not depend on the
schedule: each index's attempts are still sequential, and every run is a pure function
of its item record.

One JSON per index is written to `out_dir` (secrecy-checked for the test split):
item record, label, attempt history, confuser results, accepted answers.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

from hotwell_bench.confusers import candidates, check_item
from hotwell_bench.items import ItemRecord, sample_item
from hotwell_bench.label import label_runs
from hotwell_bench.presets import PRESETS, Preset
from hotwell_bench.run import _task, run_cached
from hotwell_bench.secrecy import check_output_path

ATTEMPT_CAP = 5


def _answer(category: str, component: str | None) -> dict:
    return {"category": category, "component": component}


def _confuser_task(args) -> tuple[int, list[dict]]:
    index, item_json, cache_dir, until, preset_name = args
    item = ItemRecord.from_json(item_json)
    fa = run_cached(item, "faulted", cache_dir, until)
    tw = run_cached(item, "twin", cache_dir, until)
    return index, [r.to_dict() for r in check_item(item, fa, tw, preset=PRESETS[preset_name])]


def label_indices(master: int, split: str, indices: list[int], cache_dir: str | Path,
                  out_dir: str | Path, workers: int | None = None,
                  cap: int = ATTEMPT_CAP, until: float | None = None, *,
                  preset: Preset) -> dict[int, dict]:
    """Generate and label `indices` under `preset` (one of presets.PRESETS, by name,
    so worker processes can look it up). `until` shortens every run (tests only;
    real items run the full day). Returns {index: result}, sorted by index."""
    if PRESETS.get(preset.name) is not preset:
        raise ValueError(f"preset {preset.name!r} is not registered in presets.PRESETS")
    out = check_output_path(out_dir, split)
    check_output_path(cache_dir, split)
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    results: dict[int, dict] = {}
    history: dict[int, list] = {i: [] for i in indices}
    counts = {"faulted": 0, "twin": 0, "confusers": 0}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        running: dict[Future, tuple] = {}

        def submit_run(i: int, it: ItemRecord, variant: str) -> None:
            fut = ex.submit(_task, (it.to_json(), variant, str(cache_dir), until))
            running[fut] = ("run", i, it, variant)

        def on_labelled(i: int, it: ItemRecord) -> None:
            fa = run_cached(it, "faulted", cache_dir, until)
            tw = run_cached(it, "twin", cache_dir, until)
            lab = label_runs(it, fa, tw, preset=preset)
            history[i].append({"attempt": it.attempt, "observable": lab.observable,
                               "t_det": lab.t_det, "rule": lab.rule,
                               "instrument": lab.instrument})
            if it.fault is None:
                results[i] = {"status": "null", "item": json.loads(it.to_json()),
                              "label": lab.to_dict(), "attempts": history[i],
                              "confusers": [],
                              "accepted_answers": [_answer("none", None)]}
            elif lab.observable:
                if candidates(it, fa):
                    fut = ex.submit(_confuser_task, (i, it.to_json(), str(cache_dir), until,
                                                     preset.name))
                    running[fut] = ("confusers", i, it, lab)
                else:
                    _accept(results, i, it, lab, history[i], [])
            elif it.attempt + 1 < cap:
                nxt = sample_item(master, split, i, it.attempt + 1, preset=preset)
                assert nxt.fault is not None    # null-ness comes from the STRUCTURE stream
                submit_run(i, nxt, "faulted")
            else:
                it0 = sample_item(master, split, i, 0, preset=preset)
                results[i] = {"status": "excluded_at_cap", "item": json.loads(it0.to_json()),
                              "label": None, "attempts": history[i], "confusers": [],
                              "accepted_answers": []}

        for i in indices:
            submit_run(i, sample_item(master, split, i, 0, preset=preset), "faulted")
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                kind, i, it, x = running.pop(fut)
                if kind == "run":
                    iid, variant, summ = fut.result()
                    counts[variant] += 1
                    print(f"[{time.time() - t0:6.0f} s] {iid} a{it.attempt} {variant}: "
                          f"t_end {summ['t_end']:.0f} s, dead {summ['plant_dead']}, "
                          f"first_out {len(summ['first_out'])}, wall {summ['wall_s']:.0f} s",
                          flush=True)
                    if variant == "faulted":
                        submit_run(i, it, "twin")
                    else:
                        on_labelled(i, it)
                else:
                    _, res = fut.result()
                    counts["confusers"] += 1
                    print(f"[{time.time() - t0:6.0f} s] confusers {it.item_id}: "
                          + ", ".join(f"{r['category']}/{r['component']} "
                                      f"{'CONFUSABLE' if r['confusable'] else r['reason']}"
                                      for r in res), flush=True)
                    _accept(results, i, it, x, history[i], res)
    print(f"== done in {time.time() - t0:.0f} s: {counts['faulted']} faulted runs, "
          f"{counts['twin']} twin runs, {counts['confusers']} confuser checks", flush=True)
    results = dict(sorted(results.items()))
    for i, r in results.items():
        (out / f"{split}-{i:04d}.json").write_text(json.dumps(r, sort_keys=True, indent=1))
    return results


def _accept(results: dict, i: int, it: ItemRecord, lab, attempts: list, conf: list) -> None:
    f = it.fault
    answers = [_answer(f.category, f.component)] + [
        _answer(r["category"], r["component"]) for r in conf if r["confusable"]]
    results[i] = {"status": "accepted", "item": json.loads(it.to_json()),
                  "label": lab.to_dict(), "attempts": attempts,
                  "confusers": conf, "accepted_answers": answers}


def summarize(results: dict[int, dict]) -> str:
    """Plain-text pilot report: per-category outcomes and labelling statistics."""
    from collections import Counter, defaultdict
    lines = []
    by_cat: dict[str, list] = defaultdict(list)
    for r in results.values():
        f = r["item"]["fault"]
        by_cat["none" if f is None else f["category"]].append(r)
    lines.append(f"{'category':30s} {'n':>3s} {'acc':>4s} {'excl':>4s} {'attempts':>18s} "
                 f"{'unobs.att':>9s} {'easy':>4s} {'conf':>4s} "
                 f"{'delay p50/p90 [s]':>18s}")
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        acc = [r for r in rs if r["status"] == "accepted"]
        excl = [r for r in rs if r["status"] == "excluded_at_cap"]
        att = Counter(len(r["attempts"]) for r in rs if r["status"] != "null")
        unobs = sum(1 for r in rs for a in r["attempts"] if a["observable"] is False)
        tot = sum(len(r["attempts"]) for r in rs if r["status"] != "null")
        easy = sum(r["label"]["easy"] for r in acc)
        nconf = sum(any(c["confusable"] for c in r["confusers"]) for r in acc)
        delays = sorted(r["label"]["t_det"] - r["label"]["t_inj"] for r in acc)
        pct = (f"{delays[len(delays) // 2]}/{delays[min(len(delays) - 1, int(0.9 * len(delays)))]}"
               if delays else "-")
        lines.append(f"{cat:30s} {len(rs):3d} {len(acc):4d} {len(excl):4d} "
                     f"{str(dict(sorted(att.items()))):>18s} "
                     f"{(f'{unobs}/{tot}' if tot else '-'):>9s} {easy:4d} {nconf:4d} "
                     f"{pct:>18s}")
    rules = Counter(r["label"]["rule"] for r in results.values() if r["status"] == "accepted")
    lines.append(f"t_det rule shares (accepted): {dict(rules)}")
    excl = [(r["item"]["fault"]["category"], r["item"]["fault"]["component"])
            for r in results.values() if r["status"] == "excluded_at_cap"]
    lines.append(f"excluded at cap: {excl}")
    return "\n".join(lines)
