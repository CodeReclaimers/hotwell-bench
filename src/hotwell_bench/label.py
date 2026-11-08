"""Onset labelling from twin runs (B3; PLAN-bench-0.1.md §4).

The faulted run F and its fault-free twin T share every noise draw (hotwell I2),
so any difference between them is caused by the fault. Three rules look for
evidence in the 1 Hz sensed data. Each is computed per instrument and reports
the time of the FIRST sample of the qualifying stretch, the convention of
hotwell `scripts/fault_detectability.py::detect`:

- level    |F − T| > D_i for LEVEL_RUN consecutive samples, D_i = max(3σ_i, 2 LSB_i).
           This is the plan's original oracle bound, and the 0.4.0 sweep's rule.
- stuck    a run of identical consecutive readings in F that is at least STUCK_FACTOR ×
           and STUCK_MARGIN samples longer than the longest such run anywhere in T.
- variance the ratio of rolling standard deviations of first differences
           (VAR_WINDOW samples) leaves [VAR_LO, VAR_HI] for VAR_SUSTAIN samples.
           First differences remove level shifts, so this sees noise changes,
           freezes and dithering stopping.

The last two exist because the level rule structurally cannot see faults that
change variance rather than level (plan §4.1, B2 finding). t_det is the earliest
evidence over all rules and instruments.

Before the first sample where F ≠ T all three statistics are exactly equal,
so no rule can fire before the fault acts. One exception is clamped: a frozen
value can extend a run that began a few samples before onset, so each stuck-rule
time is clamped to the instrument's first divergence.

Thresholds are provisional oracle-bound parameters, like the level rule's 3σ / 10 s.
The margins come from the G-B2 sweep (scripts/gb2_sweep.py). A tiny truth
perturbation re-quantizes a reading, so F's run lengths and difference
statistics drift from T's by chance. Under "any run longer than the twin's
longest" this fired on faults hotwell finds undetectable: 4 vs 3, 8 vs 7, 21 vs 16
samples, at most 6 vs 3. A 120-sample variance window gave late isolated hits.
A genuinely stuck sensor runs for hundreds of samples. Each rule has a seeded
gate test, and the slow G-B2 test checks there are no hits on rows hotwell
found undetectable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from hotwell_bench.items import ItemRecord, detection_scale
from hotwell_bench.run import COLUMNS, RunResult

LEVEL_RUN = 10
VAR_WINDOW = 600
VAR_SUSTAIN = 60
STUCK_FACTOR = 2
STUCK_MARGIN = 60
VAR_LO, VAR_HI = 0.8, 1.25
RULES = ("level", "stuck", "variance")
SCALE = np.array([detection_scale(c) for c in COLUMNS])


def _first_run_start(flag: np.ndarray, length: int) -> int | None:
    """Index of the first sample that starts `length` consecutive True values."""
    if flag.size < length:
        return None
    c = np.convolve(flag.astype(np.int64), np.ones(length, np.int64), mode="valid")
    hits = np.flatnonzero(c == length)
    return int(hits[0]) if hits.size else None


def level_rule(F: np.ndarray, T: np.ndarray) -> list[int | None]:
    over = np.abs(F - T) > SCALE
    return [_first_run_start(over[:, i], LEVEL_RUN) for i in range(F.shape[1])]


def _runs(col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start index, length) of every run of identical consecutive values."""
    change = np.flatnonzero(col[1:] != col[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [col.size])))
    return starts, lengths


def stuck_rule(F: np.ndarray, T: np.ndarray, first_div: list[int | None]) -> list[int | None]:
    out: list[int | None] = []
    for i in range(F.shape[1]):
        if first_div[i] is None:
            out.append(None)
            continue
        _, lt = _runs(T[:, i])
        sf, lf = _runs(F[:, i])
        need = max(STUCK_FACTOR * int(lt.max()), int(lt.max()) + STUCK_MARGIN)
        hit = np.flatnonzero(lf >= need)
        out.append(None if not hit.size else max(int(sf[hit[0]]), first_div[i]))
    return out


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing-window population std; entry j covers x[j-w+1 .. j] (NaN for j < w-1)."""
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    n = x.size
    out = np.full(n, np.nan)
    if n >= w:
        s1 = c1[w:] - c1[:-w]
        s2 = c2[w:] - c2[:-w]
        out[w - 1:] = np.sqrt(np.maximum(s2 / w - (s1 / w) ** 2, 0.0))
    return out


def variance_rule(F: np.ndarray, T: np.ndarray) -> list[int | None]:
    out: list[int | None] = []
    dF, dT = np.diff(F, axis=0), np.diff(T, axis=0)
    for i in range(F.shape[1]):
        if np.array_equal(dF[:, i], dT[:, i]):
            out.append(None)
            continue
        sf, st = _rolling_std(dF[:, i], VAR_WINDOW), _rolling_std(dT[:, i], VAR_WINDOW)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(st > 0, sf / st, np.where(sf > 0, np.inf, 1.0))
        # identical windows give identical sums, so the ratio is exactly 1 there
        same = np.isclose(sf, st, rtol=0.0, atol=0.0)
        ratio = np.where(same, 1.0, ratio)
        flag = (ratio > VAR_HI) | (ratio < VAR_LO)
        flag &= ~np.isnan(sf)
        j = _first_run_start(flag, VAR_SUSTAIN)
        out.append(None if j is None else j + 1)    # diff index j <-> sample j+1
    return out


def evidence(F: np.ndarray, T: np.ndarray) -> dict[str, list[int | None]]:
    """Per rule, per instrument: index of the first evidence sample (None = none).
    Rows are 1 Hz samples starting at t = 0, so an index is also a time in s."""
    n = min(len(F), len(T))
    F, T = F[:n], T[:n]
    neq = F != T
    first_div = [int(np.argmax(neq[:, i])) if neq[:, i].any() else None
                 for i in range(F.shape[1])]
    return {"level": level_rule(F, T), "stuck": stuck_rule(F, T, first_div),
            "variance": variance_rule(F, T), "first_divergence": first_div}


@dataclass
class Label:
    item_id: str
    attempt: int
    fault_present: bool
    t_inj: int | None
    observable: bool | None          # None for fault-free items
    t_det: int | None
    rule: str | None
    instrument: str | None
    per_rule: dict = field(default_factory=dict)    # rule -> {instrument: t}
    t_first_divergence: int | None = None
    easy: bool = False                # a protective trip or plant death caused by the fault
    first_out: list = field(default_factory=list)
    plant_dead_t: float | None = None
    first_alarm_t: float | None = None
    t_end: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def label_runs(item: ItemRecord, fa: RunResult, tw: RunResult) -> Label:
    ev = evidence(fa.values, tw.values)
    per_rule = {r: {COLUMNS[i]: t for i, t in enumerate(ev[r]) if t is not None}
                for r in RULES}
    cands = [(t, r, inst) for r in RULES for inst, t in per_rule[r].items()]
    divs = [t for t in ev["first_divergence"] if t is not None]
    t_det, rule, inst = min(cands) if cands else (None, None, None)
    twin_out = {tuple(x) for x in tw.summary["first_out"]}
    new_out = [x for x in fa.summary["first_out"] if tuple(x) not in twin_out]
    twin_acts = {(e["t"], e["id"]) for e in tw.events
                 if e["kind"] == "alarm" and e["value"] == "active"}
    new_acts = [e["t"] for e in fa.events if e["kind"] == "alarm"
                and e["value"] == "active" and (e["t"], e["id"]) not in twin_acts]
    f = item.fault
    return Label(
        item_id=item.item_id, attempt=item.attempt, fault_present=f is not None,
        t_inj=None if f is None else f.t_inj,
        observable=None if f is None else t_det is not None,
        t_det=t_det, rule=rule, instrument=inst, per_rule=per_rule,
        t_first_divergence=min(divs) if divs else None,
        easy=bool(new_out) or (fa.summary["plant_dead"] and not tw.summary["plant_dead"]),
        first_out=new_out,
        plant_dead_t=fa.summary["t_end"] if fa.summary["plant_dead"] else None,
        first_alarm_t=min(new_acts) if new_acts else None,
        t_end=fa.summary["t_end"])
