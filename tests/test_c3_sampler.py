"""C3: sampler changes for P02 (PLAN-bench-0.2.md §3) and the P01 guard."""
from __future__ import annotations

import hashlib

from hotwell_bench.items import sample_item
from hotwell_bench.presets import P01

#: sha256 over sample_item(...).to_json() for masters (20261107, 1, 987654321), splits
#: (dev, test), indices 0..399 and attempts 0..2 under P01 (7200 items). Taken at the C0
#: refactor (2026-11-19), when it equalled the pre-refactor sampler's hash. G-B0r covers
#: the whole 0.1 dev pipeline; this guards the sampler alone, fast.
P01_GOLDEN = "31b300dc767f67385703b3c8850ca1a0eb5e90f2ec344794f5601f2ff645faf1"


def test_p01_sampler_output_is_unchanged():
    h = hashlib.sha256()
    n = 0
    for master in (20261107, 1, 987654321):
        for split in ("dev", "test"):
            for i in range(400):
                for a in range(3):
                    h.update(sample_item(master, split, i, a, preset=P01).to_json().encode())
                    n += 1
    print(f"\nP01 sampler: {n} items, sha256 {h.hexdigest()}")
    assert n == 7200 and h.hexdigest() == P01_GOLDEN
