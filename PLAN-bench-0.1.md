# hotwell-bench 0.1 — 24-hour fault identification ("Idea 2") — Plan

Status: **APPROVED** (Alan, 2026-11-07; B-Q1…B-Q8 answered, see §11). The questions in
§10 are kept as asked; §11 records the answers, which override §2–§6 where they differ.

## 0. Context and objective

A model gets a 24-hour data dump from the hotwell plant with **at most one** injected
fault. It must say (a) whether a fault is present, (b) what it is and on which
component, and (c) when it started. hotwell 0.4.0 supplies the mechanisms: named rng
streams with fixed draws per step (I2), sensor noise and sensor faults, load ramps and
load noise, hidden plant faults, the closed-loop background operator, and the manual.
This repo holds what 0.4.0 left out on purpose (its §3): the **item generator,
twin-run onset labelling, dump packaging, grader, and item taxonomy**.

Deliverable for 0.1: a frozen, versioned item set (a public-safe dev split plus a
held-out test split), its answer key, a deterministic grader, trivial and non-LLM
baselines, and a validity report. Running frontier models against the set is a
separate step (it needs API keys and spend; §9).

## 1. Constraints carried over

- `hotwell` is consumed as an installed dependency pinned to the GitHub tag
  (`hotwell @ git+https://github.com/CodeReclaimers/hotwell@v0.4.0`) and never edited.
  A finding that needs a core change goes back to the core repo as its own task.
- **numpy is part of the determinism contract** (0.4.0 M1):
  `test_pinned_raw_draws_numpy_version_gate` fails on a numpy that changes
  PCG64/SeedSequence or `normal()`. The bench pins numpy exactly, and every released
  item set records `hotwell` commit, numpy version, and per-item state and observation
  hashes.
- **The repo will be public (Alan, 2026-11-07, superseding "stays private"); the test
  split is not.** hotwell (simulator, fault catalog, manual) is public too. The test
  split's secrecy therefore rests on material that **never enters git**: its master
  seed, generated items and answer key. Mechanisms:
  - generators take the master seed as a run-time input, never a committed constant;
  - test-split writers refuse output paths inside the repository tree (B4);
  - the dev split may be published.

  Storage for the test seed and keys: §8.1 (decided 2026-11-07).
- Reported numbers come from committed code, and every pipeline stage is gated on an
  audit of its output before the next stage consumes it (§7).

## 2. Task definition

**What the model receives** (all on a clock starting 00:00:00; no simulator vocabulary):

1. `instruments.csv`: 1 Hz samples of the 26 `OBSERVATION_SPEC` instruments, sensed
   (noise, quantization, any sensor fault), SI units in the header. 86 400 rows. No
   `TRUTH_SIGNALS` (`m_peg`, `m_relief`, `q_fire`, `p_mech`, `m_leak`), no controller
   internals.
2. `events.csv`: the sequence-of-events log: alarm transitions (activate / ack /
   clear), indication changes (pump run/trip lamps, breaker, MFT, safety valve), with
   timestamps. Derived by diffing `Observation.alarms` and `indications` step over
   step, as `scripts/run_scenario.py` does.
3. `shift_log.txt`: the operator's log (procedure starts and completions, trims) in
   plain words. See B-Q4.
4. Optionally the manual (`hotwell/docs/manual/`). The with/without condition is an
   ablation the 0.4.0 manual was built for.
5. A task statement with the answer schema and the taxonomy (§2.1) as a closed list.

**What the model answers** (JSON, validated by the grader):
`{"fault_present": bool, "category": str, "component": str|null, "onset": "HH:MM:SS"|null,
"confidence": 0–1}`.

### 2.1 Answer taxonomy (operator vocabulary, not event names)

| Category (answer key) | Components | hotwell mechanism |
|---|---|---|
| `none` | — | no fault |
| `pump_performance_loss` | CP-1, CP-2, FP-1, FP-2 | `pump_degrade` (head factor) |
| `valve_capacity_loss` | FRV, DA LCV, makeup valve | `valve_degrade` (Cv factor) |
| `stuck_actuator` | FRV, throttle, firing, DA LCV, makeup valve | `actuator_stuck` |
| `actuator_sluggish` | FRV, throttle, firing | `actuator_slow` |
| `combustion_efficiency_loss` | — | `firing_degrade` |
| `condenser_vacuum_degradation` | — | `condenser_fouling` |
| `boiler_tube_leak` | — | `tube_leak` |
| `instrument_fault` | any of the 26 instruments | sensor `bias` / `drift` / `stuck` / `lag` / `noise_increase` |

Whether `instrument_fault` asks for the sub-kind (bias vs drift…) is B-Q3.

*B1 deviations (recorded):*
- `actuator_stuck` is renamed `stuck_actuator`, because the original is a hotwell
  event name and the B4 leak scan forbids event names in task.md.
- "LCV" becomes "DA LCV": it is the same valve as the stuck-actuator row.

## 3. Item generation (B1–B2)

An **item** is a parameter record; the generator turns it into a hotwell config + a
scenario + an operator schedule, and runs it. Every random choice comes from a
per-item seed derived from a split's master seed (`SeedSequence([master, item_index])`),
never from the simulator's streams, so item parameters cannot shift noise draws.

Per-item randomized background (proposed ranges; all recorded in the key):
- **Day profile:** trough MW U(36, 48); ramp-down start U(01:00, 05:00); ramp durations
  U(30, 90) min; trough length U(4, 8) h. A fraction of days flat at 60 MW.
- **Operator:** NOP-03 cadence U(240, 420) s; CP swap and FP swap each with p = 0.7 at
  U(02:00, 22:00); reaction delay σ > 0 (the runner's `reaction_sigma`, operator
  stream). Swaps matter diagnostically: a degraded pump is visible only while it runs
  (0.4.0 M5: swapping off a degraded FP-1 restores the FRV position).
- **Noise:** the 0.4.0 quiet-day preset is labelled PROVISIONAL; the bench owns item
  noise levels (B-Q6).

Fault sampling:
- Fault-free items: proposed 20 % of each split.
- Category uniform over the taxonomy, then component uniform within it.
- Onset t_inj U(02:00, 20:00), so there are at least 2 h of baseline before and 4 h of
  evolution after.
- Shape: step, or linear ramp with `ramp_s` log-uniform in [10 min, 6 h] (50/50).
  Ramped onsets were never swept in 0.4.0; B3 measures them.
- Magnitude: log-uniform between 0.5× and 4× that fault's **step** detectability
  threshold, from the 0.4.0 sweep (`scripts/fault_detectability.py`). The lower end
  deliberately produces some unobservable items, which §4 then filters or relabels.

Fidelity: **AB drum + three-element** only (B-Q7). It is the panel default and the
richer plant. two_state + 1E stays available as an ablation.

Compute: a closed-loop AB 24 h run is ~8 min of wall on one core (measured ≈19 s per
simulated hour in the 0.4.0 release quiet-day run, with other jobs on the machine). An item costs 2 runs
(faulted + fault-free twin), plus confuser runs (§4.3). 200 items is about 2 h on 32
cores. Runs are cached by (hotwell commit, numpy version, item record) hash.

## 4. Labelling (B3)

### 4.1 Onset

The true onset t_inj is known exactly. What is *observable* starts later. The
**fault-free twin** shares every noise draw with the faulted run (I2), so their
sensed difference is the fault's effect alone. Per item we record:
- t_det: the first time some instrument's |faulted − twin| exceeds max(3σ, 2 LSB)
  for 10 consecutive 1 Hz samples, and which instrument. This is the 0.4.0
  oracle-twin bound, an optimistic floor: no model sees the twin.
- the per-instrument first-exceedance table (for the validity report and for difficulty
  strata).
- whether any protective trip or alarm occurs in the faulted run, and when.

*B2 finding: the criterion above sees only level changes (required change for B3).*
For faults that change a signal's variance or dynamics, the faulted − twin difference
is zero-mean noise, so the sustained-exceedance rule structurally cannot fire:
- **`stuck` sensor:** the faulted reading is flat and the twin noisy, so the
  difference is ≈ 1σ of noise and rarely exceeds 3σ for 10 s. Every stuck sensor on a
  noisy instrument would be dropped as unobservable, though a flatline is among the
  most visible faults.
- **`noise_increase`:** the difference has σ = (f − 1)σ_i. At dev-0017's f = 2.66 it
  exceeds 3σ_i ≈ 7 % of the time, so 10 consecutive seconds essentially never occur.
- **`lag`:** level differences appear only during transients, so the rule catches them
  only if the day has one.

B3 must add kind-appropriate statistics on the faulted run alone (twin as reference)
and take t_det as the earliest of them, each with a gate test on a seeded case:
- a run of identical quantized readings longer than the twin ever shows, for stuck;
- a rolling standard-deviation ratio, for noise_increase;
- the existing difference rule, for level and lag effects.

### 4.2 Unobservable items

If t_det does not exist (no exceedance before 24:00), the item has no observable fault.
Proposed: drop it and regenerate (B-Q5), and report the drop rate per category.

*Expected drop sources, recorded at B1 for B3 to measure:*
- makeup-valve faults (the valve is normally shut, so they show only if makeup is
  demanded);
- standby-pump faults on CP-2/FP-2 (they show only after a swap onto that pump);
- `actuator_sluggish` (never detected at constant load; it needs a ramp or a swap);
- `stuck` sensor faults on near-constant instruments;
- sensor faults on idle-pump instruments.

Regeneration keeps the category and component and resamples background and
details, so a component that is nearly always unobservable needs an attempt cap.
B3 sets the cap and reports per-component attempt counts.

*B3 risk, operator divergence:* the runner draws reaction delays from the `operator`
stream only when a step executes. After onset, a fault that changes how many trim
moves NOP-03 makes shifts the operator's later draws between the twins. The
resulting difference is causally downstream of the fault, but it could make t_det
earlier than the fault's own signature warrants. B3 must check whether the first
exceedances coincide with operator actions.
*Resolved by construction (§4.4):* the twin replays the faulted run's actions, so no
operator difference between the twins can exist.

### 4.3 Confusable faults

Two mechanisms can produce the same sensed signature. Example: FP-1 head loss and FRV
Cv loss both open the FRV at the same feed flow. FP-1 current and a later pump swap
separate them, and only sometimes.

Proposed mechanized check: for each fault item, run a short list of a-priori
**confusers** (a table in this repo) at a magnitude fitted to match the primary
signature. Accept a confuser as an equally correct answer if no instrument
distinguishes it from the true fault beyond the same max(3σ, 2 LSB) / 10 s bound.
The alternative, hand-written per-item rules, is rejected: rules cover only the
confusions someone thought of.

### 4.4 B3 decisions after the pilot (Alan, 2026-11-08)

- **Action-replay twins.** The twin and every confuser run are driven open-loop by the
  faulted run's recorded operator actions (per tick). A difference then means the
  plant responds differently to the same actions, which is what an observer with the
  shift log can judge. This replaces the closed-loop twin, whose own operator
  produced timing artifacts: 3 of 33 pilot acceptances came from trim timing alone.
- **Dropped from 0.1 sampling:** makeup-valve faults (valve capacity loss and stuck
  actuator; the valve is normally shut) and the actuator_sluggish category (load ramps
  are too slow to reach slew limits). They hit the attempt cap 8 of 8 times in the
  pilot. To revisit when the day design can reveal them (e.g. a makeup demand or a fast
  load step).

**Pilot 2 (2026-11-08, dev 0–49, master 20261107, replay twins, exclusions;
`scripts/label_items.py`, table via `scripts/pilot_table.py`):**
- 41 accepted, 9 null, 0 excluded at the cap. Regenerations: 6 unobservable attempts
  out of 47 fault attempts (pump 4/11, valve 1/7, combustion 1/7).
- Easy (plant death caused by the fault): 5, all stuck_actuator.
- t_det rule: level 36, variance 5. All 5 instrument faults happened to be
  `noise_increase` (the 4000-sample sampler test gives 104/82/86/90/79 over the five
  kinds). So the pilot exercised no bias, drift, stuck or lag sensor fault.
- Confusers: 29 items checked, 10 confusable. The condensate pair (CP head loss ↔ DA
  LCV Cv loss) was confusable in 5/5, in both directions. Feed pair (FP ↔ FRV) 1/6;
  combustion ↔ firing-rate bias 3/6; fouling ↔ cond-pressure bias 1/5; stuck
  actuator ↔ stuck position sensor 0/7 (5 of those separated by plant death).
- Delays over 10⁴ s, audited:
  - dev-0003 (FP-2, standby until the FP swap at 76697 s): detected 17 s after the swap.
  - dev-0026 (FRV −2.4 %, injected in the 38 MW trough at 25066 s): detected at
    44828 s, during the ramp back to 60 MW (43626–46321 s).

## 5. Packaging (B4)

Per item: `instruments.csv`, `events.csv`, `shift_log.txt`, `task.md`. Separately,
under `keys/`: the answer key with t_inj, t_det, accepted confusers, item record, and
hashes.

**Leak scan (mechanical gate, blocks release).** No packaged file may contain:
- a fault-event name (`pump_degrade`, `trip_pump`, …);
- a truth-signal name;
- a seed, scenario name, or hotwell internal identifier;
- a non-instrument column.

Byte-identical packaging is also required: re-packaging an item reproduces its
file hashes.

## 6. Grader and baselines (B5)

Scoring, per item (B-Q2 settles the onset window):
- **Detection:** `fault_present` correct.
- **Identification:** category correct (including accepted confusers); component
  correct as a second level.
- **Onset:** correct if within [t_inj − a, t_det + b], with a and b proposed as
  10 min. Also reported: |onset − t_inj| and onset − t_det.
- **Headline score:** the fraction of items fully correct (detection + category +
  component + onset). Each component is also reported, stratified by category,
  magnitude / threshold, step vs ramp, and with/without the manual.
- **Calibration:** Brier score on `confidence`.

Baselines, to anchor difficulty:
- always-`none`;
- class-prior guess;
- a **non-LLM change detector**: per-instrument CUSUM against statistics from
  fault-free training days, plus the category with the most likely signature.

If the change detector solves most items, the set is too easy. If no method beats
chance on a stratum, that stratum is suspect.

## 7. Validity gates (each blocks the next stage)

- **G-B1, generator:** hand-audit plots of 10 items (2 per major category + 2 null).
  **PASSED 2026-11-07 (Alan's review of `scripts/audit_items.py --master 20261107`):**
  every fault item shows its change except the stuck makeup valve (dev-0014), which
  B3 is to drop as unobservable; the nulls show none.
  The fault is where the key says it is, the background matches the record, and
  nothing else happens.
- **G-B2, labelling:** for step items at the 0.4.0 sweep magnitudes, t_det − t_inj
  reproduces `fault_detectability.py` within 1 s. The twin's state hash equals a
  standalone fault-free run's. **PASSED** (`tests/test_b3_label.py`, slow; 41/41 rows
  exact; re-passed under replay twins, progress-20261108.md).
- **G-B3, confusers:** a seeded known confusion (FP-1 vs FRV with no swap after
  onset) is detected as confusable. A known separable pair (the same pair with an FP
  swap after onset) is not. **PASSED** (`tests/test_b3_confusers.py`, slow; re-passed
  under replay twins, progress-20261108.md).
- **G-B4, packaging:** the leak scan passes, and a mutation test plants a forbidden
  token and fails it. Re-packaging is byte-identical.
- **G-B5, grader:** oracle answers score 1.0; always-`none` scores the null fraction
  exactly; shuffled keys score near chance.

## 8. Milestones

- **B0:** repo skeleton (uv, ruff, pytest), hotwell tag pin, numpy pin; a determinism
  smoke test (one tiny item regenerates its pinned hashes).
- **B1:** item record schema plus sampler (no simulation); config/scenario/schedule
  builder with load-time validation.
- **B2:** runner (faulted + twin, 1 Hz capture, events and shift log), process-pool
  parallelism, cache. G-B1.
- **B3:** labelling (t_det table, trips/alarms, unobservable filter, confusers);
  ramped-onset detectability sweep. G-B2, G-B3. **Done 2026-11-08** (pilot 2, §4.4).
- **B4:** packaging plus leak scan. G-B4.
- **B5:** grader, baselines, metrics report. G-B5.
- **B6:** pilot (≈40 dev items), end-to-end validity report, then freeze 0.1 (sizes per
  B-Q8), and transfer per §8.1.

### 8.1 B6 freeze and transfer (Alan, 2026-11-07)

**Generate once, here; transfer the artifacts.** The generated files are the set of
record. They are copied directly to Alan's laptop, where the benchmark is run. They
are not regenerated there. Reasons:
- hotwell's determinism holds only for the same seed, scenario, *platform* and numpy
  version. Another machine (e.g. an ARM laptop) may produce different floating-point
  trajectories, and so different dumps, t_det and confusers.
- Dev 50 + test 200 is ≈ 650 day-long runs with regeneration and confusers,
  ≈ 90 core-hours: 3–4 h on this 32-core machine against ≈ 15–20 h on a laptop.

**Manifest and checks.**
- The set carries a manifest, per item: the item record, hotwell commit, numpy
  version, platform string, the run's state and observation hashes, and the SHA-256
  of every file.
- `scripts/verify_release.py` (new in B6) has two modes:
  - check every file against the manifest after the transfer (integrity);
  - optionally regenerate a sample of items and report hash agreement
    (reproducibility). A disagreement on another platform is reported, not treated
    as a failure of the set.

**Storage and secrecy.**
- Both splits are written outside any git work tree (the secrecy guard enforces
  this for the test split) and transferred to the laptop. No public distribution in
  0.1.
- The test master seed is never committed. It travels with the test split or is kept
  offline.

**Packaging.**
- Compressed per-item files: gzip CSV or Parquet. Roughly 17 MB raw per item, ≈ 4 GB
  for 250, expected under 1 GB compressed.
- A set README: task, schema, noise caveats (provisional levels, zero-floor bias) and
  generation provenance.
- **Canary string:** a unique GUID (BIG-bench convention) in every packaged file of
  both splits and in the README, so contamination can be tested if any file ever
  leaks. Generated once at B6 and recorded here. The B4 leak scan must allow it, and
  the packaging test must check it is present. Kept although nothing is published,
  because it is cheap. Drop it if unwanted.

**Steps:**
1. Freeze the code (tag).
2. Generate dev and test outside the repository.
3. Validity report.
4. Package with the canary and manifest.
5. Run `verify_release.py` on a sample, here.
6. Transfer to the laptop.
7. Run `verify_release.py` integrity mode there.
8. Record the manifest hash in the plan and README.

*Superseded the same day:* the first version of this section distributed dev
publicly and test privately on HuggingFace. Alan removed that: "I will transfer
directly to my laptop where I'll be running the actual benchmark."

## 9. Out of scope (0.1)

- Running frontier models (needs API access and a budget decision).
- F01 live operation.
- Multi-fault items.
- Faults on indications/lamps (none exist in 0.4.0, invariant I4).
- Human expert baseline: worth having, needs people.

## 10. Open questions

- **B-Q1 — How the model consumes the dump.**
  - (a) **Agentic**: files plus a sandboxed Python with no network (recommended). A
    24 h × 1 Hz × 26 CSV is ~17 MB, the realistic form of "a data dump".
  - (b) In-context: decimated to 1-min means (~1 440 rows, ~150–250 k tokens).
  - (c) Both, as two tracks.
  - A sandbox without network also settles whether the model may `pip install hotwell`
    and simulate a reference day: it may not.
- **B-Q2 — Onset scoring window.** Proposed: correct iff t ∈ [t_inj − 10 min,
  t_det + 10 min]. Since t_det is an oracle floor, a model may legitimately name a time
  later than t_det. Alternative: credit decaying with |t − t_inj| / max(t_det − t_inj,
  10 min).
- **B-Q3 — Instrument-fault granularity.** Ask only "instrument fault on X"
  (recommended for 0.1), or also the sub-kind (bias / drift / stuck / lag /
  noise_increase)?
- **B-Q4 — Shift log in the dump.** Include it (recommended: a real control room has
  one, and swaps are diagnostic), or make the model infer operations from the data?
- **B-Q5 — Unobservable items.** Drop and regenerate (recommended), or keep them
  labelled `none`?
- **B-Q6 — Noise levels.** Keep the provisional quiet-day preset for 0.1
  (recommended, documented as provisional), or calibrate to instrument specs first?
  Either way, one artifact needs a decision. Noise on an instrument clamped at its
  range floor biases it: `makeup_flow` (σ 0.05, LSB 0.1, floor 0) reads up to
  0.2 kg/s with the valve shut. Options: zero noise on floor-resting instruments, or
  document it in `task.md`.
- **B-Q7 — Fidelity.** AB + 3E only (recommended), or mix in two_state + 1E? Evidence
  from the 0.4.0 release quiet day: under two_state + 1E a routine FP swap takes drum
  pressure to 8.38 MPa, 2 LSB under DRUM_P_HI, so null items there would sometimes
  alarm on background operations alone. Under AB + 3E the swap stays inside the
  day's ordinary range.
- **B-Q8 — Sizes.** Proposed: dev 50 items (shareable), test 200 items (held out).
  Fault items that trip the plant: keep them as an "easy" stratum (recommended), or
  exclude them?

## 11. Decisions (Alan, 2026-11-07)

- **B-Q1 — agentic.** Files plus sandboxed Python with **no network**, so no
  `pip install hotwell` and no re-simulation from the public core. The in-context
  track is dropped for 0.1.
- **B-Q2 — onset window as proposed.** Correct iff t ∈ [t_inj − 10 min, t_det + 10 min].
  |onset − t_inj| and onset − t_det are reported as secondary metrics.
- **B-Q3 — `instrument_fault` + instrument name** is enough for 0.1. The sub-kind
  (bias / drift / stuck / lag / noise_increase) is not graded; the key records it.
- **B-Q4 — the shift log is included** in every dump.
- **B-Q5 — unobservable items are dropped and regenerated.** The drop rate is reported
  per category.
- **B-Q6 — provisional noise levels kept** (the 0.4.0 quiet-day preset). The
  zero-floor bias (e.g. `makeup_flow` reads up to 0.2 kg/s with the valve shut) is
  kept and **documented in `task.md`**.
- **B-Q7 — AB drum + three-element only.** two_state + 1E is not used in 0.1.
- **B-Q8 — dev 50, test 200.** Fault items that trip the plant stay in, as an
  **easy** stratum, reported separately.

