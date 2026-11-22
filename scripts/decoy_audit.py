#!/usr/bin/env python3
"""G-Bd: hand-audit material for decoy days (PLAN-bench-0.2.md §3.2, §8).

Samples P02 items under a public audit master seed (not a split's master). It picks
the first fault-free items that carry decoys until every decoy kind appears at least
twice and N items are chosen. It runs each one's full day (faulted variant; there is
no fault) and writes, per item:
- `<out>/<item>.png`: load and frequency, pump currents, flows, levels and valve
  positions. Decoy windows are shaded and shift-log procedure starts are marked.
- a report on stdout:
  - the decoys;
  - the shift-log entries inside each decoy's window, which are what "explained by the
    logs" means for swaps and trims;
  - new alarms in the window;
  - whether the plant survived;
  - the answer key, which must be `none`;
  - the CUSUM baseline's verdict (trained on the 0.1 dev split, `--cusum-train`) and
    whether its alarm falls in a decoy window.

Load excursions have no shift-log note (B2-Q16). They are visible as a load change in
mw_out, steam_flow and firing_rate.

Usage: python scripts/decoy_audit.py --cusum-train DEV_SPLIT_DIR [--n 6] [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hotwell_bench import cusum as C  # noqa: E402
from hotwell_bench.items import answer_key, sample_item  # noqa: E402
from hotwell_bench.presets import DECOY_KINDS, P02  # noqa: E402
from hotwell_bench.run import COLUMNS, _task, run_cached  # noqa: E402
from hotwell_bench.secrecy import REPO_ROOT  # noqa: E402

#: public, used only for this audit (not a split master)
AUDIT_MASTER = 2026112201
PANELS = [("mw_out", "frequency"), ("cp1_current", "cp2_current", "fp1_current", "fp2_current"),
          ("steam_flow", "feed_flow", "condensate_flow"),
          ("drum_level", "hotwell_level", "da_level"),
          ("fw_valve_pos", "da_lcv_pos", "throttle_pos", "firing_rate")]


def window(d, it) -> tuple[float, float]:
    if d.kind == "load_excursion":
        return d.t_s, d.t_s + 2 * d.params["ramp_s"] + d.params["hold_s"]
    if d.kind == "pump_swap_back":
        start = d.params["swap_s"]
        if start is None:                       # the day's own swap on that pair
            op = it.operator
            start = op.cp_swap_s if d.params["pair"] == "cp" else op.fp_swap_s
        return float(start), d.params["back_s"] + 600.0
    return d.t_s, d.t_s + d.params["n"] * d.params["gap_s"] + 600.0


def pick(n: int) -> list:
    out, seen, i = [], dict.fromkeys(DECOY_KINDS, 0), 0
    while len(out) < n or min(seen.values()) < 2:
        it = sample_item(AUDIT_MASTER, "dev", i, 0, preset=P02)
        i += 1
        if it.fault is not None or not it.decoys:
            continue
        need = any(seen[d.kind] < 2 for d in it.decoys)
        if need or len(out) < n and min(seen.values()) >= 2:
            out.append(it)
            for d in it.decoys:
                seen[d.kind] += 1
    return out


def plot(it, res, path: Path) -> None:
    t = res.t / 3600.0
    fig, axs = plt.subplots(len(PANELS), 1, figsize=(14, 2.4 * len(PANELS)), sharex=True)
    for ax, cols in zip(axs, PANELS, strict=True):
        for k, c in enumerate(cols):
            y = res.values[:, COLUMNS.index(c)]
            if c == "frequency":
                ax2 = ax.twinx()
                ax2.plot(t, y, lw=0.5, color="tab:red", label=c)
                ax2.set_ylabel("Hz")
                ax2.legend(loc="upper right", fontsize=7)
            else:
                ax.plot(t, y / (1e6 if c == "mw_out" else 1.0), lw=0.6, label=c, color=f"C{k}")
        ax.legend(loc="upper left", fontsize=7)
        for d in it.decoys:
            a, b = window(d, it)
            ax.axvspan(a / 3600, b / 3600, color="gold", alpha=0.25)
        for s in res.shift_log:
            if s["step_id"] == "-" and s["outcome"] != "queued":
                ax.axvline(s["t"] / 3600, color="grey", lw=0.4, alpha=0.5)
    axs[0].set_title(f"{it.item_id} (audit master {AUDIT_MASTER}, P02): "
                     + "; ".join(f"{d.kind} @ {d.t_s / 3600:.2f} h {d.params}" for d in it.decoys),
                     fontsize=8)
    axs[-1].set_xlabel("t [h]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cusum-train", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cache", default=str(REPO_ROOT / "cache" / "decoy_audit"))
    ap.add_argument("--out", default=str(REPO_ROOT / "out" / "decoy_audit"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = pick(a.n)
    print(f"audit items: {[it.item_id for it in items]}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(items), (os.cpu_count() or 2) - 2),
                             mp_context=get_context("spawn")) as ex:
        list(ex.map(_task, [(it.to_json(), "faulted", a.cache, None) for it in items]))
    print(f"runs done in {time.time() - t0:.0f} s", flush=True)
    model, sigs = C.train_from_split(Path(a.cusum_train))
    n_cusum = 0
    for it in items:
        res = run_cached(it, "faulted", a.cache)
        key = answer_key(it)
        det = C.detect(C.minutes(list(COLUMNS), res.values), model)
        print(f"\n== {it.item_id}: key {key['category']}, t_end {res.summary['t_end']:.0f} s, "
              f"plant_dead {res.summary['plant_dead']}, profile flat {it.profile.flat}")
        for d in it.decoys:
            lo, hi = window(d, it)
            log = [f"{s['t']:.0f} {s['proc_id']} {s['step_id']} {s['outcome']}"
                   for s in res.shift_log if lo <= s["t"] <= hi and s["step_id"] == "-"]
            al = [f"{e['t']:.0f} {e['id']}" for e in res.events
                  if e["kind"] == "alarm" and e["value"] == "active" and lo <= e["t"] <= hi]
            print(f"  {d.kind} {d.t_s} {d.params}  window {lo:.0f}-{hi:.0f} s")
            print(f"    shift log: {log}")
            print(f"    new alarms: {al}")
        if det.t_alarm_min is None:
            print("  CUSUM: no alarm")
        else:
            ta = det.t_alarm_min * 60
            inwin = [d.kind for d in it.decoys
                     if window(d, it)[0] - 600 <= ta <= window(d, it)[1] + 1800]
            n_cusum += 1
            print(f"  CUSUM: alarm at {ta} s on {model['names'][det.first]}; near decoy: {inwin}")
        assert key["category"] == "none"
        plot(it, res, out / f"{it.item_id}.png")
    print(f"\nCUSUM alarms on {n_cusum} of {len(items)} decoy null days; plots in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
