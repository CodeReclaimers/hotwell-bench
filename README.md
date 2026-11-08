# hotwell-bench

Benchmark generation and grading for a 24-hour plant data-dump fault-identification task,
built on the public [`hotwell`](https://github.com/CodeReclaimers/hotwell) simulator.

**This repository will be public; the test split is not.** The code, plan, and dev split
can be published. The test split's master seed, generated items and answer key must
never be committed. Generators take the master seed at run time, and the test-split
writers refuse any output path inside this repository tree (PLAN-bench-0.1.md §1, §11).

Plan and decisions: `PLAN-bench-0.1.md` (approved 2026-11-07; answers in §11).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -e '.[dev]'     # pulls hotwell @ v0.4.0 from GitHub and numpy==2.5.1
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
```

`hotwell_bench.provenance.check_environment()` refuses anything but the pinned,
non-editable hotwell tag commit and the pinned numpy. Every generation run must call it.
Never `pip install -e ../hotwell` here.

## What this repository provides

**Provided:** code to generate, package, grade and verify item sets. Sets are
distributed privately: no generated item, key or seed of the test split is in this
repository.

| Stage | Entry point | Output |
|---|---|---|
| Generate + label | `scripts/label_items.py --master N --split S --indices 0-199` | `<out>/<split>-NNNN.json` per index (item record, label, confusers, accepted answers) + `run.json` |
| Package + leak scan | `scripts/package_split.py --split S --labels L --cache C --out DIR --canary-file G` | `DIR/<split>/` (below); exit 1 if the leak scan finds anything |
| Verify | `scripts/verify_release.py DIR/<split> [--regenerate N]` | integrity vs manifest; optional re-simulation of N items |
| Grade | `scripts/grade.py --keys DIR/<split>/keys --answers ANS [--out graded.json]` | report table; per-item JSON |
| Baselines / G-B5 | `scripts/baselines.py --keys DIR/<split>/keys` | oracle, always-none, prior, shuffled-keys |

**A packaged split:**
```
<split>/
  README.md  task.md  manifest.json        (manifest: SHA-256 of README, task.md, manual/, items/; no answers)
  manual/*.md                              hotwell v0.4.0 operator manual (optional input)
  items/<item_id>/
    instruments.csv.gz   1 Hz, t_s = 1 … 86400 (shorter if the unit was lost); 26 instruments, SI
    events.csv           alarm transitions and indication changes
    shift_log.txt        operator procedure log
    task.md              task statement + answer schema + closed answer list (same for all items)
  keys/                                    answers — never shown to a model
    <item_id>.json  manifest.json  [MASTER_SEED]   (keys/manifest.json: SHA-256 of each key)
```
The first line of every file under `items/` (of the decompressed CSV for
`instruments.csv.gz`) and `manual/`, and of `task.md` and `README.md`, is a comment
carrying the canary. `scripts/verify_release.py SPLIT_DIR` checks this layout and the
hashes using only the standard library.

**What a benchmark harness must supply (not in this repository):**
- the agent loop: one item per episode;
- the sandbox (plan §11 B-Q1): the model gets the files of `items/<item_id>/`, plus
  `manual/` in the with-manual condition, and sandboxed Python with **no network**
  (no `pip install hotwell`, no re-simulation). It must never see `keys/`;
- extracting one JSON answer per item into `ANS/<item_id>.json`.

The answer format is `{"fault_present": bool, "category": str, "component": str|null,
"onset": "HH:MM:SS"|null, "confidence": 0–1}`, where `confidence` is the probability
that the whole answer is correct. `task.md` gives the rest. A missing or malformed
answer scores 0 and is counted as invalid.

**Scoring** (`hotwell_bench.grade`):
- detection; category; component; onset, which must fall within [t_inj − 10 min,
  t_det + 10 min]; and full (all of them). Look-alike faults recorded in the key also
  count as correct;
- Brier score on `confidence` against full;
- each score stratified by category, magnitude/threshold, step vs ramp, and easy
  (the fault tripped the plant).

**Not yet provided:** the CUSUM change-detector baseline (plan §6).

## License

MIT. See [LICENSE](LICENSE).

## Status

B0–B5 complete: generation, labelling, packaging with leak scan, grader and baselines.
Gates G-B1 to G-B5 pass (G-B5 on the dev split). The 0.1 dev split (50 items) is
packaged; the test split (200 items) is being generated (B6). Plan:
`PLAN-bench-0.1.md`. Progress logs: `progress-*.md`.
