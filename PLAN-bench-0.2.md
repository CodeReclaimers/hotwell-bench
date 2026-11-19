# hotwell-bench 0.2: harder offline items: Plan

Status: **DRAFT** (written 2026-11-18; open questions B2-Q1…B2-Q9 answered by Alan on
2026-11-19, recorded in §10). Work has not started.
- Where this plan and PLAN-bench-0.1.md disagree, the 0.1 plan governs until this one is
  approved.

## 0. Context and objective

0.1 is frozen: tag v0.1.0, dev split of 50 items, test split of 199 items. The 0.1 results
point one way: the items are too easy, in specific ways.

- **Detection is nearly free.** The no-AI CUSUM baseline detects 0.994 of the faulted
  test days (progress-20261113.md). The baseline does not identify faults well (full
  correct 0.683), but it almost never misses that something changed. The faults are
  large relative to what an analyst who averages over a window can resolve.
- **Fault-free days are too easy.** Nothing on a null day resembles a fault, so "none" is
  safe whenever nothing looks odd. The only null-day errors seen are a simple detector's
  false alarms (CUSUM: 8 of 39).
- **Model evidence is held elsewhere.** Language-model runs on 0.1 are made and
  recorded outside this repository. The README's "Future directions" summarizes them only
  in general terms. This plan's scope is consistent with that evidence, but cites no model
  results.
- **One labelling defect class.** One test item with a sensor `lag` fault was missed by
  every method. The fault is visible only in a transient that an observer without the
  twin cannot interpret (progress-20261113.md, "Generator finding"). 0.2 drops `lag`
  from sampling (§2.2).

**Objective for 0.2:**
- Keep 0.1's task format: a whole day offline, one answer, the same answer schema, and
  the same `task.md` apart from the additions in §5. Any harness that ran 0.1 then runs
  0.2 unchanged, and a score change between 0.1 and 0.2 measures difficulty alone.
- Change the items, so that:
  1. detection is no longer free;
  2. fault-free days need judgment;
  3. the headline score spreads models across the range rather than at the ceiling.

**Target, set before any calibration run (§7):**
- On the 0.2 dev split, the strongest model available lands at roughly 0.6–0.75 full
  correct. It is checked by one strongest-model dev run in the last calibration round
  (§7, B2-Q9).
- CUSUM detection on faulted days falls well below 0.9.
- Every item stays solvable by the oracle-twin criterion (§2).

**Out of scope for 0.2 (deferred):**
- streaming detection: it changes the task format, so it is 0.3;
- multiple faults per item: it changes the answer schema and grader;
- live operation;
- other plant variants (a pluggable heat source);
- a misleading instrument paired with a real fault: a two-fault item, so it waits for
  multi-fault support;
- decoys that need new hotwell procedures (soot blowing, setpoint changes, condenser
  backwash, instrument calibration checks; B2-Q1);
- misleading operator notes, both misattributing and red-herring kinds (B2-Q6);
- sensor `lag` faults (B2-Q2).

## 1. Constraints carried over

All of PLAN-bench-0.1.md §1 still holds:
- hotwell is consumed as a pinned tag and never edited from here. 0.2 stays on hotwell
  0.4.0;
- numpy is part of the determinism contract;
- the test seed, test items and test keys never enter git;
- reported numbers come from committed code;
- each stage is gated on an audit of its output before the next stage uses it.

Added for 0.2:
- **0.1 stays reproducible from the 0.2 code.** Regenerating the 0.1 dev split from
  the 0.2 tree with the 0.1 preset must reproduce the 0.1 manifest hash
  `c83fea28ec264117bdfa7d5621a060dafd6d21c571f3b8e4fa1f803232463b3d`. This is a test, not
  a promise (§9, C0).
- **Generation settings freeze mechanically before the test split exists.** The 0.2
  sampling preset and labeller constants are committed and tagged. The test-split
  generator refuses to run unless HEAD is that tag and the tree is clean.
- **The grader is versioned.** The 0.2 grader still grades 0.1 answers with 0.1
  semantics byte-for-byte. New scores are added fields, not changed ones.

## 2. Labelling: fix validity before raising difficulty (C1)

The labeller change comes first, because shrinking faults without it produces items
whose labels are wrong.

### 2.1 A windowed level rule

**Today:** the level rule (`label.level_rule`) fires when
|F − T| > D_i = max(3σ_i, 2 LSB_i) for `LEVEL_RUN` = 10 consecutive samples.
- F − T is the fault's effect alone, because the twins share every noise draw.
- D_i asks whether the effect stands out from one sample of noise.
- An analyst who averages over a window resolves much less than that. 0.1 sized
  faults at `MAG_RANGE` = 0.5–4× this per-sample bound, which is why detection was
  nearly free.

**Decided (B2-Q8):** add a windowed rule, and take t_det as the earliest of the existing
rules and this one. The rule fires at the first t where:
- the trailing-window mean of (F − T) over [t − W, t] exceeds a threshold θ_i(W);
- this holds for a sustain time S.

t_det is the end of that first window, so the rule is causal.

**Threshold reference: empirical day-to-day variability.** Run pairs of fault-free days
that share the item's background but differ in every random stream.
- Load noise and sensor noise then both differ, and the plant's response to them is
  included.
- Take θ_i(W) as the (1 − α) quantile of |trailing W-mean of the difference| over
  those pairs.
- α is set so that the windowed rule's false-positive rate over a whole day, across
  all 26 instruments, is at most 1% on seed-pair nulls.

*Rejected:* a sensor-noise-only threshold, θ_i = z·σ_i·√(c_i/W), with c_i the noise
correlation factor (1 for white noise; ≈2τ for the first-order `tau` presets).
- It is the oracle bound for an analyst facing sensor noise alone.
- It ignores the load-driven variation a real analyst must separate the fault from, so
  it would label as observable faults that no practical method can find.

**Window and sustain:** W = 600 s, S = 60 s. G-B2w (§8) checks the rule on seeded cases.

**Consequence for 0.1 items:** re-label the 0.1 dev split under the new rule as a
diagnostic only, and report how many t_det values move. 0.1 itself is not re-released.

### 2.2 Sensor lag faults are dropped from 0.2

**Decided (B2-Q2):** `lag` is removed from 0.2 sampling, as 0.1 removed
`actuator_sluggish`.

**Why lag is a validity problem:**
- hotwell applies `lag` to the true value before adding noise (sensors.py: truth → lag →
  bias/drift → noise). A lag fault therefore changes nothing in a steady signal, and
  shows only during transients.
- The level rule can fire on a single transient, for example a pump stop. But an
  observer has no twin to compare that one transient against.

**Alternatives considered:**
- *A visibility criterion:* count a lag item as observable only after at least 3
  separate transients, at least 10 min apart, each exceeding the level rule for ≥ 10 s.
  Deferred, not rejected; it can return with lag in a later version.
- *Raising `SENSOR_LAG_TAU_S` from (5, 300) s.* Rejected: a long lag on a slow
  instrument is still invisible at steady state.

### 2.3 Sampled categories unchanged

**Decided (B2-Q3):** 0.1's exclusions stay: makeup-valve faults and
`actuator_sluggish`. With `lag` also removed (§2.2), the category list in `task.md` is
unchanged. Instrument faults are still sampled, with the remaining sensor fault kinds.

## 3. Item difficulty (C2–C3)

### 3.1 Magnitudes against the windowed threshold

- **Detectability sweep.** `scripts/gb2_sweep.py` and `scripts/ramp_sweep.py` exist for
  the per-sample rule. A bench-side sweep re-measures every mechanism's step threshold
  under the windowed rule.
  - It covers the `THRESHOLDS` table rows plus the sensor kinds.
  - It produces a new threshold table with the same provenance strings as 0.1.
  - hotwell's `fault_detectability.py` stays the 0.1 source and is not edited.
- **Magnitude range.** `MAG_RANGE` becomes a log-uniform range in multiples of the
  windowed threshold.
  - Proposed: 0.7–3×. The lower end still makes some items unobservable, and the
    existing drop-and-regenerate path (0.1 §4.2) handles them.
  - The range is a calibration knob (§7), not a fixed choice.
- **Shape mix.** `P_RAMP` goes from 0.5 to a proposed 0.7, and `ONSET_RAMP_S` widens to
  (30 min, 8 h).
  - A slow ramp is where an analyst's choice of onset is genuinely uncertain, which
    the onset window (t_det + 600 s) already accounts for.

### 3.2 Decoy events on fault-free days

- **Null fraction (B2-Q7):** `P_NULL` rises from 0.2 to 0.30. Over 200 test items that
  is about 60 null days, about 20 of them with decoys. 0.30 is the top of the agreed
  25–30% range, chosen so the decoy stratum has about 20 items.
  - `P_NULL` is fixed, not a calibration knob: changing it changes the score's
    composition, not item difficulty.
- **Decoy fraction:** a proposed one third of null items carry one or two decoys. A
  decoy is a real, logged, non-fault event whose signature resembles a fault.
- **The key stays `none`.** A decoy is acceptable only if the shift log or event log
  explains it.

**Decided (B2-Q1): only decoys the bench can build with hotwell 0.4.0** (no core change):
- An unscheduled load excursion outside the daily profile:
  - a dispatch step of −5 to −15 MW held 20–90 min;
  - or a second, shallower trough.
  - The day profile is bench-owned (`DayProfile`), so this is a sampler and builder
    change.
- A pump swap at an unusual time, or swapped back within the hour (NOP-01/NOP-02 exist).
  - Swaps happen on 0.1 null days too. What is new is placing them where they mimic a
    pump fault's timing.
- A frequency-trim burst: several NOP-03 trims in quick succession.

**Deferred:** decoys that need new hotwell procedures (soot blowing, setpoint changes,
condenser backwash, instrument calibration checks).

### 3.3 Explanatory operator notes

**Decided (B2-Q6):** no misleading operator notes in 0.2.

- **Today:** the shift log is the ProcedureRunner's log of procedure starts and
  completions. It is truthful and contains no free text.
- **Open point in C3:** whether every decoy is already explained by an existing log
  entry.
  - A pump swap or trim is logged by its procedure.
  - A load excursion may not be explained by any existing entry.
- **Where one is not:** add a true explanatory note, a short templated entry at the
  decoy's start, e.g. "load reduced per dispatch".
  - Notes only ever describe decoys correctly.
  - They appear on decoy days only, so a note's presence does not reveal a fault.
- **Leak scan.** Notes come from a committed template table. The leak scan whitelists
  the rendered templates exactly, so free text cannot carry an event, mechanism or
  truth-signal name (G-B4 extends to notes).
- **Deferred:** misattributing notes, which blame a fault's early symptom on a benign
  cause, and red-herring notes, which report a concern the telemetry contradicts.

## 4. Generation hygiene (C0)

These are separate commits, each before any behaviour change:
- **Versioned presets (refactor only).** Move the sampling ranges and labeller constants
  from module constants (`presets.py`, `label.py`) into a named preset object (`P01`,
  `P02`). Pass it explicitly through sampler, builder and labeller.
  - Gate: the 0.1 dev split regenerates to the 0.1 manifest hash under `P01`.
  - No behaviour change in the same commit.
- **Master seed from a file.** `scripts/label_items.py --master-seed-file`. The test seed
  was on the process command line in 0.1.
- **Per-item retry pipelining.** Retry an item as soon as its attempt is rejected,
  instead of in barrier-synchronized rounds (about 40 min of idle cores in 0.1). This is
  a performance change, separate from correctness.
- **`pyproject.toml` `requires-python`.** It says `>=3.11`, but numpy 2.5.1 requires 3.12,
  so `uv run` can no longer resolve the project (seen 2026-11-18). Set `>=3.12`.

## 5. Packaging (C4)

- Layout, file set and the answer schema are unchanged from 0.1.
- If C3 adds explanatory notes, they appear in `shift_log.txt` as their own entry type,
  with timestamp and text.
  - `task.md` describes that entry type only as far as its file-format section needs.
  - `task.md` gains no sentence about notes being wrong, since none are (B2-Q6).
- The leak scan's whitelist is extended by the rendered note templates (§3.3). A
  mutation test plants a forbidden token inside a note and must fail.
- The category list in `task.md` is unchanged (§2.3).

## 6. Grader and protocol (C5)

- **Grader 0.2.** It adds fields and changes no 0.1 field:
  - a continuous onset score, exp(−|onset − t_inj| / 1800 s), reported alongside the
    window rule;
  - per-item detection delay relative to t_det;
  - strata for decoy days and windowed-threshold ratio buckets.

  Regrading the 0.1 dev answers must give byte-identical 0.1 summaries.
- **Protocol for model runs** (guidance for whoever runs models; runners live outside
  this repository):
  - at least two independent runs per configuration;
  - report tokens and cost alongside accuracy;
  - report run-to-run noise (items flipping between runs) and a paired bootstrap over
    items for differences between configurations.
- **CUSUM baseline.** Re-run unchanged on 0.2 and reported. Its false-alarm rate on
  decoy days is a direct check that the decoys do their job.

## 7. Difficulty calibration on dev (C6)

**Risk:** tuning items by whether a particular model fails them builds a benchmark around
that model's blind spots (adversarial filtering).

Rules:
1. **Knobs only, never items.** Calibration changes only these preset values:
   - `MAG_RANGE`;
   - `P_RAMP` and `ONSET_RAMP_S`;
   - the decoy fraction.

   No item is kept or dropped because of a model's answer.
2. **Dev only.** Calibration runs use a fresh 0.2 dev split. The test split does not
   exist until the preset is frozen.
3. **Few rounds.** At most 3 calibration rounds. Each round is recorded in a progress log
   with the knob values and the scores.
4. **Probe models and budget (B2-Q4).**
   - Two cheap probe models.
   - A fixed total spend cap across all rounds.
   - The models and the cap are recorded with the runner, outside this repository.
   - In the last round only, one dev run of the strongest available model, outside the
     probe cap, checks the §0 target (B2-Q9).
5. **Oracle and CUSUM every round.** Oracle answers must still score 1.0, and CUSUM
   results are reported each round.

Then freeze: commit the preset, tag `v0.2.0-params`, and generate the test split from
that tag (§1).

## 8. Validity gates (each blocks the next stage)

The 0.1 gates G-B1…G-B5 carry over and are re-run on 0.2 output. New gates:
- **G-B0r (refactor):** the 0.1 dev split regenerates byte-identically under `P01`.
- **G-B2w (windowed rule):**
  - a seeded sub-threshold step that the per-sample rule misses and the windowed rule
    catches;
  - a seeded effect at 0.5× θ_i that it does not catch;
  - the false-positive rate on 20 seed-pair null days is at most 1%.
- **G-Bd (decoys):** hand-audit plots of 6 decoy null days (§3.2).
  - Every decoy is explained by the logs.
  - Null keys are `none`.
  - The CUSUM baseline alarms on at least some decoy days.
- **G-B5v (grader versioning):** 0.1 dev answers regrade byte-identically.
- **G-freeze:** the test-split generator refuses when HEAD is not the params tag or the
  tree is dirty, checked by a test.

## 9. Milestones

| ID | Work | Gate |
|---|---|---|
| C0 | Hygiene (§4), each item its own commit; the preset refactor first | G-B0r |
| C1 | Windowed level rule (§2.1); re-label the 0.1 dev split as a diagnostic; drop `lag` from sampling (§2.2) | G-B2w |
| C2 | Windowed detectability sweep, giving a new threshold table | G-B2 (re-run) |
| C3 | Sampler: magnitudes, shapes, null fraction, decoys, explanatory notes if needed (§3) | G-B1 (re-run), G-Bd |
| C4 | Packaging and leak scan for notes (§5) | G-B4 |
| C5 | Grader 0.2 (§6) | G-B5, G-B5v |
| C6 | Dev split plus calibration rounds (§7) | §0 target, oracle 1.0 |
| C7 | Freeze tag, test split generation, packaging, verification, transfer as in 0.1 §8.1 | G-freeze, G-B4, G-B5 |

**Split sizes (B2-Q5):** dev 50 items, test 200 items (indices 0–199).
- As in 0.1, an item rejected at the attempt cap is excluded and reported, not
  backfilled. So the packaged test split can come out slightly under 200; 0.1 packaged
  199.

**Compute:**
- The 0.1 plan estimated about 2 h on 32 cores for 200 items, counting twins and
  confusers.
- §2.1 adds seed-pair null runs. These are one-off per background profile family,
  not per item, and are cached.
- The detectability sweep (C2) is the largest new job; it will be estimated before it
  runs.

## 10. Decisions and open questions

Answered by Alan, 2026-11-19:

| ID | Question | Decision |
|---|---|---|
| B2-Q1 | Core decoys (new hotwell procedures) or bench-only decoys? | Bench-only decoys. hotwell stays at 0.4.0; core decoys are deferred (§3.2). |
| B2-Q2 | Lag: three-transient criterion, or drop `lag`? | Drop `lag` from 0.2 (§2.2). |
| B2-Q3 | Keep 0.1's category exclusions? | Yes: makeup-valve faults and `actuator_sluggish` stay excluded (§2.3). |
| B2-Q4 | Calibration models and budget | Two probe models with a total spend cap, recorded with the runner (§7). |
| B2-Q5 | Split sizes | Dev 50, test 200 (§9). |
| B2-Q6 | Misleading operator notes, and a task.md sentence about them | No misleading notes in 0.2, and no such sentence. True explanatory notes only, where a decoy needs one (§3.3). |
| B2-Q7 | Null fraction | Raised to 25–30%; this plan fixes `P_NULL` = 0.30 (§3.2). |
| B2-Q8 | Windowed rule | Empirical seed-pair threshold; W = 600 s, S = 60 s; α for a ≤ 1% per-day false-positive rate (§2.1). |
| B2-Q9 | The §0 target names the strongest model, but calibration runs only the probe models: how is the target checked? | One strongest-model dev run in the last calibration round, outside the probe cap (§7). Rejected: restating the target as a probe-model band; checking only after freeze, when nothing can be tuned. |
