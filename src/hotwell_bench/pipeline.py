"""Labelling pipeline with regeneration (B3; PLAN-bench-0.1.md §4.2, B-Q5).

For each index: attempt 0 is run (faulted + twin) and labelled. A fault item with
no evidence by any rule (unobservable) is regenerated: the next attempt keeps its
category and component (STRUCTURE stream) and resamples background and details.
This repeats up to ATTEMPT_CAP attempts. An index still unobservable at the cap is
recorded as `excluded_at_cap`, not filled. How to fill a split is decided when it
is frozen (B6). Accepted fault items then get the confuser check. Fault-free
items are labelled once.

Phases (so simulations parallelize across items):
1. run attempt a for every pending index;
2. label;
3. carry the unobservable ones to attempt a + 1;
4. run confuser checks for accepted items in a process pool.

One JSON per index is written to `out_dir` (secrecy-checked for the test split):
item record, label, attempt history, confuser results, accepted answers.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from hotwell_bench.confusers import candidates, check_item
from hotwell_bench.items import ItemRecord, sample_item
from hotwell_bench.label import label_runs
from hotwell_bench.run import run_cached, run_items
from hotwell_bench.secrecy import check_output_path

ATTEMPT_CAP = 5


def _answer(category: str, component: str | None) -> dict:
    return {"category": category, "component": component}


def _confuser_task(args) -> tuple[int, list[dict]]:
    index, item_json, cache_dir, until = args
    item = ItemRecord.from_json(item_json)
    fa = run_cached(item, "faulted", cache_dir, until)
    tw = run_cached(item, "twin", cache_dir, until)
    return index, [r.to_dict() for r in check_item(item, fa, tw)]


def label_indices(master: int, split: str, indices: list[int], cache_dir: str | Path,
                  out_dir: str | Path, workers: int | None = None,
                  cap: int = ATTEMPT_CAP, until: float | None = None) -> dict[int, dict]:
    """`until` shortens every run (tests only; real items run the full day)."""
    out = check_output_path(out_dir, split)
    check_output_path(cache_dir, split)
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    results: dict[int, dict] = {}
    history: dict[int, list] = {i: [] for i in indices}
    pending = list(indices)
    accepted: dict[int, tuple] = {}
    for attempt in range(cap):
        if not pending:
            break
        items = {i: sample_item(master, split, i, attempt) for i in pending}
        if attempt > 0:
            items = {i: it for i, it in items.items() if it.fault is not None}
        print(f"== attempt {attempt}: {len(items)} items", flush=True)
        run_items(list(items.values()), cache_dir, workers=workers, until=until)
        nxt = []
        for i, it in items.items():
            fa = run_cached(it, "faulted", cache_dir, until)
            tw = run_cached(it, "twin", cache_dir, until)
            lab = label_runs(it, fa, tw)
            history[i].append({"attempt": attempt, "observable": lab.observable,
                               "t_det": lab.t_det, "rule": lab.rule,
                               "instrument": lab.instrument})
            if it.fault is None:
                results[i] = {"status": "null", "item": json.loads(it.to_json()),
                              "label": lab.to_dict(), "attempts": history[i],
                              "confusers": [],
                              "accepted_answers": [_answer("none", None)]}
            elif lab.observable:
                accepted[i] = (it, lab)
            else:
                nxt.append(i)
        pending = nxt
    for i in pending:
        it = sample_item(master, split, i, 0)
        results[i] = {"status": "excluded_at_cap", "item": json.loads(it.to_json()),
                      "label": None, "attempts": history[i], "confusers": [],
                      "accepted_answers": []}
    conf: dict[int, list] = {i: [] for i in accepted}
    tasks = [(i, it.to_json(), str(cache_dir), until) for i, (it, _) in accepted.items()
             if candidates(it, run_cached(it, "faulted", cache_dir, until))]
    if tasks:
        print(f"== confusers: {len(tasks)} items", flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                 mp_context=get_context("spawn")) as ex:
            futs = [ex.submit(_confuser_task, t) for t in tasks]
            for n, fut in enumerate(as_completed(futs), 1):
                i, res = fut.result()
                conf[i] = res
                print(f"  [{n}/{len(tasks)}] index {i}: "
                      + ", ".join(f"{r['category']}/{r['component']} "
                                  f"{'CONFUSABLE' if r['confusable'] else r['reason']}"
                                  for r in res), flush=True)
    for i, (it, lab) in accepted.items():
        f = it.fault
        answers = [_answer(f.category, f.component)] + [
            _answer(r["category"], r["component"]) for r in conf[i] if r["confusable"]]
        results[i] = {"status": "accepted", "item": json.loads(it.to_json()),
                      "label": lab.to_dict(), "attempts": history[i],
                      "confusers": conf[i], "accepted_answers": answers}
    for i, r in results.items():
        (out / f"{split}-{i:04d}.json").write_text(json.dumps(r, sort_keys=True, indent=1))
    return results


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
