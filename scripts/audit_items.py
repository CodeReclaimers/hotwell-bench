#!/usr/bin/env python3
"""G-B1 hand-audit material: run items (cached), print a per-item summary, plot.

Usage:
    python scripts/audit_items.py --master N [--split dev] [--per-category 2]
        [--categories c1,c2,...] [--nulls 2] [--workers W] [--cache DIR] [--out DIR]

Selection is unbiased: the first indices (0, 1, 2, …) whose sampled category is in
the requested set, `--per-category` each, plus the first `--nulls` fault-free
items. For each item the summary gives:
- the key (category, component, mechanism, params, onset, shape);
- the background (profile breakpoints, swap times, NOP-03 cadence);
- the first second where faulted ≠ twin, and the instruments that differ there;
- the top instruments by max |faulted − twin| / D_i, with D_i = max(3σ, 2 LSB);
- trips, first-out, rejected actions, alarm activations, and non-trim shift-log lines.

The plot shows MW and frequency with the background marked, then the top-4
instruments faulted vs twin. The dev split only: this writes key information.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hotwell_bench.items import detection_scale, sample_item
from hotwell_bench.run import COLUMNS, run_cached, run_items
from hotwell_bench.secrecy import REPO_ROOT
from hotwell_bench.taxonomy import FAULT_CATEGORIES

DEFAULT_CATS = ("pump_performance_loss", "stuck_actuator", "boiler_tube_leak",
                "instrument_fault")


def select(master: int, split: str, cats: tuple[str, ...], per_cat: int, nulls: int,
           limit: int = 5000) -> list:
    want = {c: per_cat for c in cats}
    want_null = nulls
    out = []
    for i in range(limit):
        it = sample_item(master, split, i)
        c = None if it.fault is None else it.fault.category
        if c is None and want_null > 0:
            out.append(it)
            want_null -= 1
        elif c in want and want[c] > 0:
            out.append(it)
            want[c] -= 1
        if want_null == 0 and not any(want.values()):
            break
    return out


def hms(t: float) -> str:
    s = int(round(t))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def summarize(it, fa, tw) -> dict:
    n = min(len(fa.t), len(tw.t))
    d = np.abs(fa.values[:n] - tw.values[:n])
    scale = np.array([detection_scale(c) for c in COLUMNS])
    rel = d / scale
    differs = np.any(d > 0, axis=1)
    first = int(np.argmax(differs)) if differs.any() else None
    top = sorted(((float(rel[:, j].max()), COLUMNS[j]) for j in range(len(COLUMNS))),
                 reverse=True)[:8]
    return {"n": n, "first_diff_s": first,
            "first_diff_cols": [] if first is None else
            [COLUMNS[j] for j in np.nonzero(d[first] > 0)[0]],
            "top": top}


def report(it, fa, tw, s) -> None:
    f = it.fault
    print(f"\n=== {it.item_id} (attempt {it.attempt}, sim seed {it.sim_seed}) ===", flush=True)
    if f is None:
        print("key: no fault")
    else:
        print(f"key: {f.category} / {f.component}  [{f.mechanism} {f.target} {f.params}]  "
              f"onset {hms(f.t_inj)} ({f.t_inj} s), {f.shape}; magnitude {f.magnitude} "
              f"thr {f.threshold} ({f.threshold_source})")
    bp = it.profile.breakpoints()
    print("profile: " + ("flat 60 MW" if it.profile.flat else
                         ", ".join(f"{hms(t)} {m:.1f}" for t, m in bp)))
    op = it.operator
    print(f"operator: NOP-03 every {op.nop03_cadence_s} s; CP swap "
          f"{hms(op.cp_swap_s) if op.cp_swap_s is not None else '-'}; FP swap "
          f"{hms(op.fp_swap_s) if op.fp_swap_s is not None else '-'}; reaction "
          f"{op.reaction_s} s, sigma {op.reaction_sigma}")
    print(f"first faulted != twin: "
          f"{'never' if s['first_diff_s'] is None else hms(s['first_diff_s'])} "
          f"{s['first_diff_cols'][:8]}")
    print("max |faulted - twin| / D_i: " +
          ", ".join(f"{c} {v:.2f}" for v, c in s["top"]))
    for v, r in (("faulted", fa), ("twin", tw)):
        sm = r.summary
        acts = [(e["t"], e["id"]) for e in r.events
                if e["kind"] == "alarm" and e["value"] == "active"]
        ops = [(e["t"], e["proc_id"], e["outcome"]) for e in r.shift_log
               if e["proc_id"] != "NOP-03" and e["step_id"] == "-"]
        trims = sum(1 for e in r.shift_log if e["proc_id"] == "NOP-03"
                    and e["outcome"] == "completed")
        print(f"  {v:8s} t_end {hms(sm['t_end'])} dead {sm['plant_dead']} mft {sm['mft']} "
              f"turbine_trip {sm['turbine_trip']} pumps_tripped {sm['pumps_tripped']} "
              f"first_out {sm['first_out']} rejected {len(sm['rejected'])}")
        print(f"           alarm activations {len(acts)} {[(hms(t), a) for t, a in acts[:6]]}"
              f"; ops {[(hms(t), p, o) for t, p, o in ops]}; NOP-03 completed {trims}")


def plot(it, fa, tw, s, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = [c for v, c in s["top"] if v > 0 and c not in ("mw_out", "frequency")][:4]
    for c in ("drum_level", "drum_pressure", "fw_valve_pos", "hotwell_level"):
        if len(top) < 4 and c not in top:   # pad with key process variables
            top.append(c)
    panels = ["mw_out", "frequency", *top]
    fig, axes_all = plt.subplots(len(panels) + 1, 1, figsize=(12, 2.0 * (len(panels) + 1)),
                                 sharex=True)
    axes, ax_diff = axes_all[:-1], axes_all[-1]
    th, tt = fa.t / 3600.0, tw.t / 3600.0
    for ax, c in zip(axes, panels, strict=True):
        j = COLUMNS.index(c)
        k = 1e-6 if c == "mw_out" else 1.0
        # faulted underneath, twin on top and semi-transparent: where they coincide
        # (always before onset) the twin stays visible instead of being painted over
        ax.plot(th, fa.values[:, j] * k, lw=0.6, color="tab:red", label="faulted", zorder=1)
        ax.plot(tt, tw.values[:, j] * k, lw=0.5, color="0.15", alpha=0.45,
                label="twin (no fault)", zorder=2)
        ax.set_ylabel(c + (" [MW]" if c == "mw_out" else ""), fontsize=8)
        ax.tick_params(labelsize=7)
        if it.fault is not None:
            ax.axvline(it.fault.t_inj / 3600, color="tab:red", lw=0.8, ls="--")
            if it.fault.shape == "ramp" and "ramp_s" in it.fault.params:
                ax.axvline((it.fault.t_inj + it.fault.params["ramp_s"]) / 3600,
                           color="tab:red", lw=0.6, ls=":")
        for ts, lab in ((it.operator.cp_swap_s, "CP swap"), (it.operator.fp_swap_s, "FP swap")):
            if ts is not None:
                ax.axvline(ts / 3600, color="tab:blue", lw=0.6, ls="--")
                if c == "mw_out":
                    ax.text(ts / 3600, ax.get_ylim()[1], lab, fontsize=7, color="tab:blue",
                            va="top")
    if not it.profile.flat:
        bp = it.profile.breakpoints()
        axes[0].plot([t / 3600 for t, _ in bp], [m for _, m in bp], lw=0.8, color="k",
                     ls=":", label="profile (gross)")
    axes[0].legend(fontsize=7, loc="lower left")
    f = it.fault
    axes[0].set_title(f"{it.item_id}: " + ("no fault" if f is None else
                      f"{f.category} / {f.component} — {f.mechanism} {f.params}, onset "
                      f"{hms(f.t_inj)} ({f.shape})"), fontsize=9)
    # (faulted - twin) / D_i: shows the fault's effect even where traces overlap exactly
    n = min(len(fa.t), len(tw.t))
    for c, color in zip(top, ("tab:red", "tab:blue", "tab:green", "tab:purple"), strict=False):
        j = COLUMNS.index(c)
        ax_diff.plot(fa.t[:n] / 3600.0, (fa.values[:n, j] - tw.values[:n, j])
                     / detection_scale(c), lw=0.5, color=color, label=c)
    for y in (-1.0, 1.0):
        ax_diff.axhline(y, color="0.3", lw=0.6, ls=":")
    if it.fault is not None:
        ax_diff.axvline(it.fault.t_inj / 3600, color="tab:red", lw=0.8, ls="--")
    ax_diff.set_ylabel("(faulted − twin) / D_i", fontsize=8)
    ax_diff.tick_params(labelsize=7)
    leg = ax_diff.legend(fontsize=7, loc="upper left", ncol=4)
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    ax_diff.set_xlabel("time [h]")
    ax_diff.set_xlim(0, 24)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{it.item_id}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=int, required=True)
    ap.add_argument("--split", default="dev", choices=["dev"])
    ap.add_argument("--categories", default=",".join(DEFAULT_CATS))
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--nulls", type=int, default=2)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", default=str(REPO_ROOT / "cache"))
    ap.add_argument("--out", default=str(REPO_ROOT / "out" / "audit"))
    a = ap.parse_args()
    cats = tuple(a.categories.split(","))
    bad = [c for c in cats if c not in FAULT_CATEGORIES]
    if bad:
        ap.error(f"unknown categories {bad}")
    items = select(a.master, a.split, cats, a.per_category, a.nulls)
    print(f"selected {[it.item_id for it in items]}", flush=True)
    run_items(items, a.cache, workers=a.workers)
    for it in items:
        fa, tw = run_cached(it, "faulted", a.cache), run_cached(it, "twin", a.cache)
        s = summarize(it, fa, tw)
        report(it, fa, tw, s)
        print(f"plot: {plot(it, fa, tw, s, Path(a.out))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
