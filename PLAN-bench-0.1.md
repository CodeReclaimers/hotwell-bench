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
- **This repo stays private.** hotwell (simulator, fault catalog, manual) is public;
  the secrecy of the test split rests on held-out seeds and the answer key, both of
  which live only here.
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
| `valve_capacity_loss` | FRV, LCV, makeup valve | `valve_degrade` (Cv factor) |
| `actuator_stuck` | FRV, throttle, firing, DA LCV, makeup valve | `actuator_stuck` |
| `actuator_sluggish` | FRV, throttle, firing | `actuator_slow` |
| `combustion_efficiency_loss` | — | `firing_degrade` |
| `condenser_vacuum_degradation` | — | `condenser_fouling` |
| `boiler_tube_leak` | — | `tube_leak` |
| `instrument_fault` | any of the 26 instruments | sensor `bias` / `drift` / `stuck` / `lag` / `noise_increase` |

Whether `instrument_fault` asks for the sub-kind (bias vs drift…) is B-Q3.

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

### 4.2 Unobservable items

If t_det does not exist (no exceedance before 24:00), the item has no observable fault.
Proposed: drop it and regenerate (B-Q5), and report the drop rate per category.

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
  The fault is where the key says it is, the background matches the record, and
  nothing else happens.
- **G-B2, labelling:** for step items at the 0.4.0 sweep magnitudes, t_det − t_inj
  reproduces `fault_detectability.py` within 1 s. The twin's state hash equals a
  standalone fault-free run's.
- **G-B3, confusers:** a seeded known confusion (FP-1 vs FRV with no swap after
  onset) is detected as confusable. A known separable pair (the same pair with an FP
  swap after onset) is not.
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
  ramped-onset detectability sweep. G-B2, G-B3.
- **B4:** packaging plus leak scan. G-B4.
- **B5:** grader, baselines, metrics report. G-B5.
- **B6:** pilot (≈40 dev items), end-to-end validity report, then freeze 0.1 (sizes per
  B-Q8).

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

