# hotwell-bench (private)

Benchmark generation and grading for a 24-hour plant data-dump fault-identification task,
built on the public [`hotwell`](https://github.com/CodeReclaimers/hotwell) simulator.
**Keep this repository private.** The test split's secrecy rests on its held-out seeds
and answer key, which live only here.

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

## Status

B0 (skeleton, pins, determinism gates) complete. B1 (item record schema and sampler) is
next.
