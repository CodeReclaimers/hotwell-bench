"""Comparing graded runs (PLAN-bench-0.2.md §6, protocol for model runs).

A configuration is one (model, settings) pair run several times on the same items; a run
is the per-item output of `grade.grade` (a list of dicts with `item_id`). Primitives:
- `item_scores`: item id -> score of one run on one criterion (bool criteria as 0/1;
  `onset_score` as is; items where the criterion is None, e.g. onset on a null item,
  are left out);
- `flips`: run-to-run noise within a configuration, i.e. the items whose score is not
  the same in every run, and the flip count for each pair of runs;
- `paired_bootstrap`: the difference between two configurations, mean over items of
  (mean over A's runs − mean over B's runs), with a percentile interval from resampling
  items with replacement (a fixed seed).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def item_scores(graded: list[dict], criterion: str = "full") -> dict[str, float]:
    return {g["item_id"]: float(g[criterion]) for g in graded if g.get(criterion) is not None}


def _common(runs: list[list[dict]], criterion: str) -> tuple[list[str], np.ndarray]:
    """Items scored in every run, and the (runs × items) score matrix."""
    sc = [item_scores(r, criterion) for r in runs]
    ids = sorted(set.intersection(*(set(s) for s in sc)))
    return ids, np.array([[s[i] for i in ids] for s in sc])


def flips(runs: list[list[dict]], criterion: str = "full") -> dict:
    if len(runs) < 2:
        raise ValueError("run-to-run noise needs at least two runs")
    ids, m = _common(runs, criterion)
    unstable = [i for i, col in zip(ids, m.T, strict=True) if np.ptp(col) > 0]
    pairs = {f"{a}-{b}": int(np.sum(m[a] != m[b])) for a, b in combinations(range(len(runs)), 2)}
    return {"n_items": len(ids), "n_runs": len(runs), "per_run_mean": m.mean(axis=1).tolist(),
            "unstable_items": unstable, "n_unstable": len(unstable), "pair_flips": pairs}


def paired_bootstrap(runs_a: list[list[dict]], runs_b: list[list[dict]],
                     criterion: str = "full", n_boot: int = 10000, seed: int = 0,
                     level: float = 0.95) -> dict:
    ids_a, ma = _common(runs_a, criterion)
    ids_b, mb = _common(runs_b, criterion)
    ids = sorted(set(ids_a) & set(ids_b))
    ia = {i: k for k, i in enumerate(ids_a)}
    ib = {i: k for k, i in enumerate(ids_b)}
    a = ma.mean(axis=0)[[ia[i] for i in ids]]
    b = mb.mean(axis=0)[[ib[i] for i in ids]]
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return {"n_items": len(ids), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(d.mean()), "ci": [float(lo), float(hi)], "level": level,
            "p_diff_le_0": float(np.mean(boots <= 0)), "n_boot": n_boot, "seed": seed}
