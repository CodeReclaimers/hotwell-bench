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
| CUSUM baseline | `scripts/cusum_baseline.py --train DIR/dev --eval DIR/<split> --out ANS` | non-LLM change detector (plan §6): one answer per item, graded with `grade.py` |

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

**CUSUM baseline** (`src/hotwell_bench/cusum.py`): per-feature CUSUM on load-normalized
1-minute residuals, trained on dev (fault-free statistics from null days and the
pre-fault part of fault days), category from the nearest dev training signature.
On the 0.1 test split it scores 0.683 fully correct (detection 0.994 on fault items;
8 of 39 fault-free days raise a false alarm). By the plan's own criterion (§6: if the
change detector solves most items, the set is too easy), most of 0.1 is solvable by
change detection plus signature lookup; instrument faults and stuck actuators are
where it fails.

## License

MIT. See [LICENSE](LICENSE).

## Status

B0–B5 complete: generation, labelling, packaging with leak scan, grader and baselines.
Gates G-B1 to G-B5 pass on both splits. 0.1 splits (generated at commit `34f9560`,
packaged at `v0.1.0`; the files are distributed directly, not from this repository):

| split | items | of which null | `manifest.json` SHA-256 |
|---|---|---|---|
| dev | 50 | 9 | `c83fea28ec264117bdfa7d5621a060dafd6d21c571f3b8e4fa1f803232463b3d` |
| test | 199 | 39 | `88b11882cc7839074c596fdbdb6d44f8c5eae22a838316695e5cfd75e8df5cb0` |

Test was sampled at 200; one item (an instrument fault on `makeup_flow`) stayed
unobservable through the attempt cap and is excluded. Plan: `PLAN-bench-0.1.md`.
Progress logs: `progress-*.md`.

## Future directions

Candidate scope for later versions; none of this is planned or approved yet. Early
results on 0.1 suggest frontier models lose little once they have the whole day of data
in front of them. The ideas below target that, roughly from cheapest to largest.

**1. Harder offline items** (same format as 0.1):
- Ramped onsets and magnitudes near the detectability limit. 0.1 sizes faults at
  0.5–4× a per-sample threshold, but an analyst who averages over a window gains
  roughly √N; the twin-run difference, judged with a windowed rule, keeps every item
  solvable.
- More fault-free days with convincing decoys: an operator step change, or a nuisance
  alarm that coincides with a real-looking drift. These test whether a model can say
  "nothing here."
- Zero, one or two faults per item, with the model required to say how many (0.1
  assumes at most one; plan §9).
- A misleading shift log: an operator compensates by hand for a degrading pump and logs
  it as routine, so the telemetry contradicts the log.
- A misleading instrument paired with a real fault: a biased sensor that masks or mimics
  a physical fault, so the model has to decide which evidence to trust.
- Before treating an item that every model misses as hard, check that a reference
  solver can find it; otherwise it is an item defect.

**2. Streaming detection.** Feed the day in chunks (for example an hour at a time) and
require an alarm as early as the model is confident. Score detection delay against
false alarms. This reuses the 0.1 items and ground truth; the onset tolerance becomes a
delay metric.

**3. Live operation.** The model operates the plant through a fault instead of reading
a finished record.
- Time pressure in simulated time, not wall-clock time: the simulation advances a fixed
  amount per tool call or per output token, so pressure is reproducible and the same for
  every model, and does not measure API latency.
- Outcomes are scored first (energy delivered, limit violations judged on true rather
  than sensed values, trips avoided); diagnosis is secondary.
- Information costs operating margin: diagnostic probes such as manual valve steps or
  load changes use it up, and the drum's inverse response (shrink and swell) makes the
  obvious probe dangerous.
- Grading by open-loop replay of the agent's logged actions against every fault
  hypothesis gives the set of faults that were genuinely indistinguishable; a scripted
  reference policy shows which cases are solvable at all.

**4. What is scored** (possible on 0.1 data now): accuracy against tokens used as a
frontier, calibration, and time to a confident answer, rather than correctness alone.
Stricter grading is also available without new data: a tighter onset window, no credit
for the confusable alternative, and grading the instrument-fault kind (which needs one
more answer field).

**5. Other plant variants: a pluggable heat source.** Instead of replacing the fossil
boiler, make the heat source a component behind a clean interface to the steam side,
with the boiler as the default. 0.1 then stays reproducible and comparable across model
generations. A new source becomes a second plant variant, not a rewrite: the steam
side, the condensate and feed train, the noise model, the historian and most of this
repository's tooling carry over. The interface is a change to `hotwell`, made there as
its own task. Two coupling styles need to fit behind it: heat released directly into the
drum (fired boiler), and a hot fluid stream through an exchanger (reactor coolant,
turbine exhaust, molten salt).

- *Sodium-cooled fast reactor.* Its main value is that models know much less about it
  than about fossil boilers, so they have to reason from the manual and first
  principles rather than recognise familiar fault signatures.
  - Unusual dynamics: delayed-neutron kinetics; reactivity feedbacks (Doppler, sodium
    density, thermal expansion) with interacting signs and time constants; pump
    coastdown into natural circulation.
  - Fault families with no boiler equivalent: sodium–water reaction in the steam
    generator, intermediate heat exchanger fouling, subassembly flow blockage,
    control-rod drift, fuel cladding failure.
  - A three-loop plant (primary sodium, intermediate sodium, water/steam), so heat-
    transport delays make onset harder to place.
  - Costs: the kinetics are too stiff for a fixed-step explicit integrator, so they need
    an implicit method or the prompt-jump approximation, within the determinism
    contract. Validation needs published transient benchmarks (for example the EBR-II
    shutdown heat removal tests). The manual, taxonomy, detectability sweeps and
    look-alike pairs are all new. Some fast reactors paired sodium-heated evaporators
    with a steam drum; check that against a specific reference plant before relying on
    it.
  - This variant pays off most in live operation, where reactivity feedback makes
    control actions matter. It belongs after the harder offline items and streaming
    detection.
- *Gas turbine with a heat-recovery steam generator (combined cycle).*
  - Exhaust flow and temperature follow turbine load and ambient temperature, which gives
    realistic benign variability, a natural source of decoys for fault-free days.
  - Turbine faults: compressor fouling, hot-section degradation.
  - Supplementary firing as a second, independent heat input.
  - A turbine trip removes heat in seconds.
  - Well documented, so it adds coupling rather than unfamiliarity.
- *Cogeneration.* Process-steam export is a property of the steam side, not of the heat
  source, and could pair with any source. It adds a second demand (electricity against
  process steam). In live operation the agent then has to trade one off against the
  other.
- *Solar thermal with molten-salt storage.* Cloud transients are a benign disturbance,
  and a storage inventory sets the operating margin.
- *Geothermal flash.* No boiler. Non-condensable gases from the well load the condenser,
  which makes the existing condenser-vacuum faults harder to tell apart.
- *Other reactor types* (gas-cooled, molten salt). These are unfamiliar to models in the
  same way as the fast reactor, and share most of its costs.
