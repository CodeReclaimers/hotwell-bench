"""Run an item and its fault-free twin (B2; PLAN-bench-0.1.md §3–§4).

`run_built(built, which)` drives one 24-h run with the background operator and
captures:
- `values`: the 26 OBSERVATION_SPEC instruments, sensed, at every whole second
  (row k is the observation at t = k s; row 0 is the noise-free init sample);
- `events`: every alarm-state and indication change at full 20 Hz time
  resolution, from diffing `Observation.alarms` and `Observation.indications`
  step over step (as hotwell's run_scenario.py does);
- `shift_log`: the ProcedureRunner's log entries;
- `summary`: end time, trips, first-out, rejected actions, hashes, wall time.

The drive loop is hotwell's quiet-day loop (scripts/quiet_day.py at v0.4.0):
start due procedures, `runner.tick`, then `sim.step`. A record that mirrors
the quiet day reproduces its state hash (tests/test_b2_run.py, slow).

Runs stop at `plant_dead`, as hotwell's own `Simulator.run` does. How a dump
packages a dead plant is a B4 decision. Results are cached on disk under a key
covering the environment record, RUNNER_VERSION, the item record, the variant
and the run length.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from hotwell.procedures import ProcedureRunner, load_procedures
from hotwell.sensors import OBSERVATION_SPEC
from hotwell.simulator import Simulator

from hotwell_bench.build import BuiltItem, build
from hotwell_bench.items import ItemRecord
from hotwell_bench.provenance import environment_record
from hotwell_bench.secrecy import check_output_path

#: bump whenever capture semantics change; part of the cache key
RUNNER_VERSION = 1
CAPTURE_PERIOD_S = 1.0
COLUMNS = tuple(s.name for s in OBSERVATION_SPEC)
VARIANTS = ("faulted", "twin")


@dataclass
class RunResult:
    item_id: str
    variant: str
    t: np.ndarray            # int64 seconds, shape (n,)
    values: np.ndarray       # float64, shape (n, len(COLUMNS))
    events: list             # [{"t", "kind": "alarm"|"indication", "id", "value"}]
    shift_log: list          # [LogEntry as dict]
    summary: dict

    def column(self, name: str) -> np.ndarray:
        return self.values[:, COLUMNS.index(name)]


def _diff(prev: dict, cur: dict, t: float, kind: str, cleared: str | None) -> list[dict]:
    out = [{"t": t, "kind": kind, "id": k, "value": v}
           for k, v in cur.items() if prev.get(k) != v]
    if cleared is not None:
        out += [{"t": t, "kind": kind, "id": k, "value": cleared}
                for k in prev if k not in cur]
    return out


def run_built(built: BuiltItem, variant: str, until: float | None = None,
              progress: bool = False) -> RunResult:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    sc = built.scenario if variant == "faulted" else built.twin
    max_t = sc.max_t if until is None else min(float(until), sc.max_t)
    sim = Simulator(copy.deepcopy(built.config), scenario=sc)
    sim.initialize_steady_state(sc.load_mw)
    op = built.item.operator
    runner = ProcedureRunner(load_procedures(), streams=sim.streams,
                             reaction_s=op.reaction_s, reaction_sigma=op.reaction_sigma)
    pending = deque(s for s in built.schedule if s[0] < max_t)
    stride = int(round(CAPTURE_PERIOD_S / sim.dt))
    obs = sim.last_obs
    rows, ts = [[obs.values[c] for c in COLUMNS]], [0]
    prev_alarms, prev_ind = dict(obs.alarms), dict(obs.indications)
    events: list[dict] = []
    wall0 = time.perf_counter()
    k = 0
    next_report = 3600.0
    while sim.t < max_t - 0.5 * sim.dt:
        while pending and pending[0][0] <= sim.t + 1e-9:
            _, pid, params = pending.popleft()
            runner.start(pid, params, sim.t)
        actions = runner.tick(sim.last_obs, sim.loop_state(), sim.t) or None
        obs = sim.step(actions)
        k += 1
        t = round(obs.t, 2)
        cur_alarms, cur_ind = dict(obs.alarms), dict(obs.indications)
        if cur_alarms != prev_alarms:
            events += _diff(prev_alarms, cur_alarms, t, "alarm", "clear")
            prev_alarms = cur_alarms
        if cur_ind != prev_ind:
            events += _diff(prev_ind, cur_ind, t, "indication", None)
            prev_ind = cur_ind
        if k % stride == 0:
            rows.append([obs.values[c] for c in COLUMNS])
            ts.append(k // stride)
        if progress and sim.t >= next_report:
            print(f"  [{built.item.item_id} {variant}] t={sim.t / 3600:5.2f} h  "
                  f"wall {time.perf_counter() - wall0:6.1f} s", flush=True)
            next_report += 3600.0
        if obs.plant_dead:
            break
    summary = {
        "item_id": built.item.item_id, "variant": variant, "scenario": sc.name,
        "t_end": sim.t, "steps": k, "plant_dead": bool(sim.disc.plant_dead),
        "mft": bool(sim.disc.mft), "turbine_trip": bool(sim.disc.turbine_trip),
        "pumps_tripped": sorted(p for p, m in sim.disc.pump_modes.items() if m == "tripped"),
        "first_out": [[round(t, 2), cid] for t, cid in sim.engine.first_out],
        "rejected": [[r.t, json.dumps(r.action, sort_keys=True), r.reason]
                     for r in sim.rejected],
        "state_hash": sim.state_history_hash, "observation_hash": sim.observation_hash,
        "wall_s": time.perf_counter() - wall0,
    }
    return RunResult(built.item.item_id, variant, np.asarray(ts, dtype=np.int64),
                     np.asarray(rows, dtype=np.float64), events,
                     [asdict(e) for e in runner.log], summary)


# ------------------------------------------------------------------ cache

def cache_key(item: ItemRecord, variant: str, until: float | None) -> str:
    blob = json.dumps({"env": environment_record(), "runner_version": RUNNER_VERSION,
                       "item": item.to_json(), "variant": variant, "until": until},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _paths(cache_dir: Path, item: ItemRecord, variant: str, until) -> tuple[Path, Path]:
    stem = f"{item.item_id}-a{item.attempt}-{variant}-{cache_key(item, variant, until)[:16]}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def save(res: RunResult, item: ItemRecord, cache_dir: str | Path, until=None) -> None:
    d = check_output_path(cache_dir, item.split)
    d.mkdir(parents=True, exist_ok=True)
    npz, js = _paths(d, item, res.variant, until)
    tmp = npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t=res.t, values=res.values)
    os.replace(tmp, npz)
    meta = {"key": cache_key(item, res.variant, until), "columns": COLUMNS,
            "item": json.loads(item.to_json()), "events": res.events,
            "shift_log": res.shift_log, "summary": res.summary}
    js.write_text(json.dumps(meta, sort_keys=True))


def load(item: ItemRecord, variant: str, cache_dir: str | Path, until=None) -> RunResult | None:
    npz, js = _paths(Path(cache_dir), item, variant, until)
    if not (npz.exists() and js.exists()):
        return None
    meta = json.loads(js.read_text())
    if meta["key"] != cache_key(item, variant, until) or tuple(meta["columns"]) != COLUMNS:
        return None
    with np.load(npz) as z:
        return RunResult(item.item_id, variant, z["t"], z["values"], meta["events"],
                         meta["shift_log"], meta["summary"])


def run_cached(item: ItemRecord, variant: str, cache_dir: str | Path | None,
               until: float | None = None, progress: bool = False) -> RunResult:
    if cache_dir is not None:
        check_output_path(cache_dir, item.split)
        hit = load(item, variant, cache_dir, until)
        if hit is not None:
            return hit
    res = run_built(build(item), variant, until=until, progress=progress)
    if cache_dir is not None:
        save(res, item, cache_dir, until)
    return res


def _task(args) -> tuple[str, str, dict]:
    item_json, variant, cache_dir, until = args
    item = ItemRecord.from_json(item_json)
    res = run_cached(item, variant, cache_dir, until)
    return item.item_id, variant, res.summary


def run_items(items: list[ItemRecord], cache_dir: str | Path, workers: int | None = None,
              until: float | None = None) -> dict[tuple[str, str], dict]:
    """Run every item's faulted and twin variants in a process pool (spawned
    workers), caching results. Returns {(item_id, variant): summary}."""
    for it in items:
        check_output_path(cache_dir, it.split)
    tasks = [(it.to_json(), v, str(cache_dir), until) for it in items for v in VARIANTS]
    out = {}
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                             mp_context=get_context("spawn")) as ex:
        futures = [ex.submit(_task, a) for a in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            iid, v, summ = fut.result()
            out[(iid, v)] = summ
            print(f"[{n}/{len(tasks)}] {iid} {v}: t_end {summ['t_end']:.0f} s, "
                  f"dead {summ['plant_dead']}, first_out {len(summ['first_out'])}, "
                  f"wall {summ['wall_s']:.0f} s", flush=True)
            sys.stdout.flush()
    return out
