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

## License

MIT. See [LICENSE](LICENSE).

## Status

B0 (skeleton, pins, determinism gates), B1 (item records, sampler, builder), B2
(runner, cache, parallel runs) and B3 (labelling, regeneration, confusers,
action-replay twins) complete. Gates G-B1 (hand audit of 10 dev items,
`scripts/audit_items.py --master 20261107`), G-B2 and G-B3 passed. B4 (packaging and
leak scan) is next.
