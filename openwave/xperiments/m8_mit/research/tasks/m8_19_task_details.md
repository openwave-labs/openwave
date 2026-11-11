# M8.19: THE CENSUS'S PERSISTENCE TESTS (S5), a delta on M8.15's persistence test

Pre-registration, for review. It governs nothing until the maintainer's go. **Sealed until the freeze:** no finite-element solve on a target or on the coherent row in `R5`, no spectrum on either, and no evolution of any state but the controls' recorded ones. § Known outcomes names what is already on record.

## TASK PLANNING

### The question

**What S5 asks.** "Every Elliptic stretch has at least one pre-registered persistence test" (the census, [§ VI](https://github.com/dmobius3/mode-identity-theory/blob/c0297f9910e4989ee4492226e1035032ed74f791/files/framework/files/working/files/slot-dynamics-census.md#vi-when-it-is-done)). The answer of 2026-11-07 on #512 ([19464325](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-19464325)) gives S5 an ending: a stretch whose instrument cannot be shown adequate under the tests fixed before any target evolves ends untested, with its cause named and no persistence claimed, and S5 counts it for the census's close. How a failed adequacy test on a control or a target is treated is for the review of these terms; § THE GATES states it.

**What this task asks:** for every Elliptic stretch on OpenWave's record that has no persistence test, fix the adequacy tests before any target evolves, run them on this registration's instrument, and run one persistence test where they pass; where they cannot pass, record the stretch untested with its cause. Separately, run the coherent row's ladder in `R5` above `ε = 2` and record its outcome on its own.

**The route.** The tests run on M8.15's finite-element dynamics, outside the harmonic admission, as [our request of 2026-10-29](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-19277944) set out: "we will ask for no further harmonic remedy after the first", and the boundary of 2026-10-15 ([18955679](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-18955679), § 3) places time integration in a harmonic basis outside its admission.

### The stretches

From OpenWave's adjudicated record (the census, § III), with each unclassified point treated as splitting a stretch:

| Stretch | Classified Elliptic points | Gaps | Persistence test on record |
| --- | --- | --- | --- |
| T1/R4, lower | 0.25 to 1 (5 points) | 1.41 unclassified | none |
| T1/R4, upper | 2 to 22.63 (8 points) | 32 to 72 unclassified | none |
| T1/R5 | 0.25 to 22.63 (14 points) | 32 unclassified | none |
| T4/R4 | 0.25 to 2.83 (8 points) | 4, 8 and 11.31 unclassified; Hyperbolic at 5.66 | none |
| T4/R5 | 0.25 to 8 (11 points) | change located at 8.720990 to 8.724874 | none |
| The coherent row in `R5` | 0.25, 0.5, 1 and 2 (M8.16.1's control C2a/R5) | no ladder above 2 on the record | none |

**Covered already,** by M8.15's tests ([`RECORD.md`](https://github.com/dmobius3/mode-identity-theory/blob/c0297f9910e4989ee4492226e1035032ed74f791/files/framework/files/working/files/scripts/soft-slot-dynamics/RECORD.md) § 2, adjudicated on #620), counted at the amplitudes they ran: the prism, T2/R4 at 5.657 and T2/R5 at 4 and 16; the hexagon, T3/R4 at 2.828 and T3/R5 at 4 and 11.31; T5/R2 at 16 and 64, one on each side of its unclassified 45.25. All eight Persist. C2a and C2b carry two control tests each, all Persist, and stay apart from the scored targets.

### What this task is not

- **Not a reclassification.** No type on OpenWave's record is read again, and no point of T1 or T4 is classified. The coherent row's ladder above 2 is the one classification here (§ THE COHERENT ROW IN `R5`).
- **Not a rescoring** of M8.15, M8.15.1, M8.15.2, M8.16.1, M8.17 or M8.18. Their outcomes stand as adjudicated.
- **No new instrument numerics.** The solver is M8.15.x's, frozen at MIT `97a8c08`; the changes below are to which points are tested, for how long, and what is recorded.
- **No time integration in a harmonic basis.** MIT's own computation selects no stretch here and is not read.
- **No `MODELS.md` cell, M8.4 object, slot relation, mass or particle state;** M8.7's gate is unchanged.

### Known outcomes

These are on record or follow from it. This task names them and claims no new evidence from them.
- **T1's cluster check fails at every ladder point** on M8.15's instrument, 18 in `R4` and 15 in `R5` ("cluster check (`G_tol` dimension 6, frozen 4)", `RECORD.md` § 2, adjudicated on #620). The record reads this as the 0.9 overlap rule taking the Krein-negative slow mode into `G_tol`; #620 left that reading unratified (the census, § IV). M8.15.x's replacement rule "fails as registered" ([`RESULT_2.md`](https://github.com/dmobius3/mode-identity-theory/blob/c0297f9910e4989ee4492226e1035032ed74f791/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-15x/qualification/RESULT_2.md)), and the module was left out of the freeze (the qualification's [`README.md`](https://github.com/dmobius3/mode-identity-theory/blob/c0297f9910e4989ee4492226e1035032ed74f791/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-15x/qualification/README.md)). The same failure is expected on this registration's instrument.
- **T4 has no finite-element solution on record:** Newton failed at the seed (`RECORD.md` § 3). A post-run diagnosis outside the record attributes it to the stopping test (INFERRED, the census, § IV); M8.15.x's frozen solver repairs that test.
- **The slow time scales,** from the adjudicated spectra (M8.17's readings for T1 and T4, M8.16.1's for the coherent row), recomputed by [`tools/periods.py`](https://github.com/dmobius3/mode-identity-theory/blob/0b98e28e2f234753bcef370a4df5c3cf383b2664/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/registration/tools/periods.py): at every Elliptic point of T1 and T4 the slowest genuine slow mode is Krein-negative, with period `P = 2π/|λ|`:

| Stretch | Point with the shortest `P` | `P` | Periods in `T` = 1000 |
| --- | --- | --- | --- |
| T1/R4, lower | 1 | 8481 | 0.12 |
| T1/R4, upper | 22.63 | 644 | 1.55 |
| T1/R5 | 22.63 | 1210 | 0.83 |
| T4/R4 | 2.83 | 37925 | 0.03 |
| T4/R5 | 5.66 | 15658 | 0.06 |
| The coherent row in `R5` | 2 | 2507 | 0.40 |

- **M8.15's controls** (`m8_15_solver/out/`): C1, C2a and C2b Persist; C3 Fails at `t = 583.28`; C3's rate run fits 0.009741 against a linear 0.009737; on the full ladders, C2a/R4 Persists at 8 and 32 and C2b/R2 at 16 and 64.
- **M8.15.x's control 1a** (`P5a` in `R4`, level 2): the repaired stopping test passes at iteration 3, at 9.6×10⁻¹³ ([`RESULT_1.md`](https://github.com/dmobius3/mode-identity-theory/blob/c0297f9910e4989ee4492226e1035032ed74f791/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-15x/qualification/RESULT_1.md)).
- **The margin rate `r`** on M8.15's recorded tests runs from about 10⁻¹⁶ on the controls and 10⁻¹² on T5 to 4.05×10⁻³ on T2/R4 (`m8_15_solver/out/`).
- **C2a/R4's slowest period at `ε` = 1** on M8.16.1's reading is 2740, so M8.15's C2a test at 1, at `T` = 1000, spanned 0.37 of it.
- **A disclosure.** M8.15's persistence code computes, beside its validity rule, a second energy rule, `valid_under_v6_energy_rule` (charge drift ≤ 10⁻¹⁰ and energy drift ≤ 10⁻⁶ of the total energy). It is false on every test on record, all valid under the frozen rule; only C3's rate run, at `η` = 10⁻⁷, is the reverse. The frozen rule governs here, as in M8.15; the second is recorded and not scored.

### Ownership and run format

Author-proposed, author-run, maintainer-adjudicated, as M8.15 to M8.18. One registered run on the author's machine, on mains power with sleep off, after the maintainer's go and the freeze block's pull request. At most four jobs run at once, each on one thread, as M8.15's jobs ran, matching the machine's four performance cores; § THE FREEZE, THE GUARD AND THE RUN fixes which jobs may share a wave.

## THE INSTRUMENT

M8.15.x's solver, frozen at MIT `97a8c08` ([`soft-slot-dynamics-m8-15x/`](https://github.com/dmobius3/mode-identity-theory/tree/c0297f9910e4989ee4492226e1035032ed74f791/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-15x), `SHA256SUMS` and `m8_15x_solver/out/MANIFEST.json`), byte for byte: M8.15's solver with the stopping test reading the full bordered residual, every multiplier included, and a record written on every end. M8.15.1 and M8.15.2 ran on it (#625, #632). Its persistence code is M8.15's, recording three more fields.

**The changes this registration makes,** all in a new driver beside the frozen solver, shown as a diff against M8.15.x's `m815_branch.py` and tested before the freeze (§ THE TESTS BEFORE THE FREEZE):
1. A test point is chosen from OpenWave's adjudicated stretch by the gates below, not from the finite-element instrument's own linear types.
2. A test's duration is set by gate 6: one period of the identified mode at least, and never under 1000.
3. The perturbation's share on the identified mode is read before evolution (gate 5).
4. The gates' readings are written to the record before any target evolves.
5. The standing wave and the level-2 spectrum at each tested point are written to disk.

## THE PERSISTENCE TEST

M8.15's, restated so that these terms stand alone:
- **Level 2;** Störmer–Verlet with the consistent mass matrix, at the fixed step `dt = 0.8 · 2/ω_max`; observations every `round(1/dt)` steps; the reference is the exact discrete wave.
- **The perturbation:** a fresh `default_rng(20261006)` per test draws a static vector; its slow half `(P_S − P_G)v` and fast half `(1 − P_S)v`, Ω-orthogonal projections, are each scaled to energy norm `(η/√2)‖Φ‖_M`, with `η` = 10⁻³. With no genuine slow mode, all of `η` goes to the fast half. The seed is frozen, and no state is drawn again.
- **d_⊥:** the relative M-norm of the position part, aligned modulo U(1), taken into Φ's rotating frame and projected off `G_tol`. **d_rot:** the relative M-norm of its projection onto the rotation tangents.
- **The verdict,** the frozen evaluator's, the first event deciding: Fails if d_⊥ exceeds 100η first; Invalid (drift) if d_rot exceeds √η first, a tie counting as Invalid; otherwise Persists if max d_⊥ ≤ 10η over `[0, T]`, and Unresolved if not.
- **Validity, as frozen:** the reference's distance from the family ≤ η, charge drift ≤ 10⁻¹⁰, energy oscillation ≤ 0.25 of the perturbation's energy, and secular drift ≤ 10⁻² of it. A run that fails it is recorded as Invalid, with its verdict beside it.

## THE GATES

Fixed here, before any target evolves. Every stretch's points are solved and their gates read and written to the record before any target evolves. A stretch's test point is its highest classified point at which gates 2 to 7 all pass; the search walks down the stretch from the top, to its lowest classified point. If none passes, the stretch ends untested, its cause named from the first gate that failed at each classified point, all recorded.

1. **The controls,** before anything else (§ THE CONTROLS).
2. **The standing wave:** an accepted Newton solution at the point, at levels 3 and 2, on the continuation path from the seed.
3. **The cluster and the mode.** `G_tol` passes the frozen cluster check at levels 2 and 3, as `Point.cluster_ok` reads it: its algebraic dimension the frozen one, closed under λ → −λ, λ̄ and −λ̄, and an Ω-Gram matrix whose smallest singular value is at least 10⁻⁶ of its largest. And the target mode is identified outside `G_tol`: the slowest genuine slow mode with the reference's Krein sign and multiplicity, found exactly once among the genuine modes.
4. **Agreement on that mode:** its eigenvalue at levels 2 and 3 within 5% of each other, with the same Krein sign; and, where OpenWave's record holds a harmonic reading at the point, within 5% of that reading's slowest genuine slow mode, with the same Krein sign and multiplicity. On the coherent row's ladder above 2, the reference is level 3 alone.
5. **Excitation:** with `v_s` the frozen perturbation's slow half, `B` the real invariant subspace of the identified mode and `P_B` its Ω-orthogonal projector, `‖P_B v_s‖² / ‖v_s‖² ≥ 0.01`, both in the positive-definite energy norm the frozen `persistence.py` scales the perturbation in. It is a projection measure, not a share of the signed Hamiltonian energy, which is negative on a Krein-negative mode. It is read and recorded before evolution, with the frozen seed and no redraw.
6. **The time scale:** `T = max(1000, P)`, rounded up to the next multiple of 250, with `P = 2π/|λ|` the identified mode's period at level 2, so that the test spans at least one period of the mode it is meant to probe.
7. **The margin:** `r · T ≤ ln(1/(10η))`, with `r` the largest real part among the level-2 eigenvalues the frozen code classes lifted or rigid, as in M8.15. It is the instrument's own bound on how long it can run at that point.

**What a failed gate does.**
- **A failed control (gate 1)** stops the run before any target. Its cause and the operations it stopped are recorded. Whether it shows the instrument inadequate for each stretch, and so gives S5's untested ending, is for the maintainer's adjudication; the driver assigns no ending.
- **A failed gate 2 to 7** at a point moves the search down the stretch. No evolution runs at a point whose gates fail.
- **No stretch is tested twice,** no verdict is turned into another, and nothing is repaired and rerun under these terms; any other route would be a separate request.

## THE CONTROLS

Before any target, on this registration's instrument:
- **M8.15's dynamics gate, whole** (`m815_controls.py`): C1 at `ε` = 6, and C2a and C2b at 1, Persist; C3 at 1 Fails; and C3's rate run fits within 5% of the linear rate.
- **The slice control** M8.15's record asks for (`RECORD.md` § 5): M8.15.x's control 1a, `P5a`, whose Newton must stop under the repaired test below 10⁻¹¹.
- **A control at an amplitude no record holds:** C3, the octahedron in `R4`, at `ε` = 3, which must Fail.
- **A control at a duration no record holds:** C2a/R4 at `ε` = 1, read through gates 3, 5, 6 and 7 as a target is, with `T` = 2750 by gate 6, which must Persist, valid.
- **The longest test.** A target test whose `T` exceeds 2750 runs only after C2a/R4 at 1 has Persisted, valid, at that `T`. That control runs once, at the longest `T` any target's gates set, after every gate is read and before any test longer than 2750 evolves. If it does not Persist, valid, the tests it was to cover do not run, the cause is recorded, and their stretches' endings are for the maintainer's adjudication, as for any failed control.

Each control's verdict must be the one stated.

## THE COHERENT ROW IN `R5`

Two gates, each with an outcome of its own.
- **The ladder.** The coherent orbit in `R5` from its seed along `R5`'s ladder to 32, with M8.15's frozen linear types at level 3, cross-checked at level 2, as C2a's in `R4`. At 0.25, 0.5, 1 and 2 the types must agree with M8.16.1's. The outcome: either the ladder establishes the types it reads, or it is not established, its cause named. **An unestablished ladder stays its own outcome:** S5's untested ending does not close it, and the row stays closed enough on Theorem A, as the census records.
- **The test.** On the stretch the ladder establishes, by the gates above. If the ladder is not established, on the stretch on record, 0.25 to 2, where gate 6 sets `T` at 2500 or 2750 at 2, inside gate 4's band, the durations C2a/R4's control covers.

## THE OUTCOMES

Per stretch, with no aggregate, the frozen evaluator's verdict at the test point, with `T`, the periods spanned, max d_⊥, max d_rot and the reference floor:
- **Persists** or **Fails:** the stretch has its test, as S5 asks.
- **Unresolved:** an adequately executed test that shows neither persistence nor failure under the thresholds. These terms read it as the test S5 asks for, earning no persistence conclusion, and ask the maintainer to confirm that reading.
- **Invalid,** for drift or for a run that fails the validity rule: these terms read it as S5's untested ending, with that cause, and ask the maintainer to confirm that reading, since an Invalid verdict is read after the target has evolved.
- **Untested,** where no point passes the gates, with each point's cause.

The ladder's outcome is recorded apart. No outcome claims orbital stability or instability: a test that Persists is evidence to `T`, not beyond, and one that Fails shows non-persistence at `η` to `T` on this instrument.

**Expected, and claimed as no evidence:** T1's three stretches untested at the cluster check, as on M8.15's record; the coherent row tested. T4 has no expected outcome: no finite-element state of it is on record, and with the margin rate on record running from about 10⁻¹⁶ to 4×10⁻³, gate 7 could go either way at T4's periods (INFERRED).

## STATED LIMITS

- One perturbation draw per test, at `η` = 10⁻³, at level 2, with the frozen seed; a finite `T`, at least one period of the identified mode, bounded at each point by the instrument's margin.
- The finite-element mesh breaks the continuous symmetry; the gates test whether that matters at the tested point, and nothing beyond it.
- A test spans one period of its mode or a few; it does not establish behaviour over many periods.

## THE TESTS BEFORE THE FREEZE

On the bench, before the freeze, none on a target or on the coherent row in `R5`:
- the driver's diff, reproducing M8.15.x's records of C2a/R4 at 8 and of control 1a bit for bit;
- that the driver reaches C3 at `ε` = 3 and reads its gates there, without evolving it;
- a smoke test of four jobs at once on the controls' recorded points, at a short `T` and read for nothing: one thread each, each record under its own path, and the orchestrator's order;
- each gate's mutation arm, turning it red for its named reason on the controls at their recorded points;
- `tools/periods.py`, reproducing the time-scale table from the committed records.

## COMPUTE

Inferred from M8.15's times on one thread (`m8_15_solver/out/`), 1.7 to 2.7 hours per 1000 of `T`:

| Stage | Hours, one thread |
| --- | --- |
| The controls: M8.15's gate, `P5a` and C3 at 3 | about 10 |
| C2a/R4 at 1, `T` = 2750 | 4.7 to 7.4 |
| T1's continuations and gates, both slots | about 3.5 |
| T4's continuations and gates, both slots | a few |
| The coherent row's ladder | 2 to 3 |
| The coherent row's test | 1.7 to 7.4 |
| Expected, in all | about 25 to 35 |

At four jobs at once, in the waves below, the expected run takes about 15 to 25 hours of wall time.

If T4's gates all pass, which needs a margin rate below about 2.9×10⁻⁴ in `R5` and 1.2×10⁻⁴ in `R4`, its tests add about 27 to 53 hours in `R5` (`T` from 15750 to 19750) and 65 to 103 in `R4` (`T` = 38000), with C2a/R4's control at the longer `T` first, another 65 to 103: about 6 to 9 days of wall time, the two tests running side by side after the control. That is the worst case at T4's highest points. The search walks down to each stretch's lowest point, where T4's period reaches 296267 in `R4` and 450558 in `R5`; if T4's gates pass only there, which needs a margin rate below about 1.6×10⁻⁵ and 1.0×10⁻⁵, `T` rises to 296500 and 450750, and the longest control and the tests take about 2 to 3.5 months of wall time (INFERRED). Every gate is read before any test runs, so that cost is known before it starts.

## THE FREEZE, THE GUARD AND THE RUN

As M8.18: a freeze block pins every file by SHA-256 in `FROZEN.json` before the go; the driver refuses to run without it, refuses any target before wave 1's controls have passed, and refuses any test longer than 2750 before the longest control has Persisted, valid. The orchestrator that starts the waves is frozen with the driver. Each job writes its own record, atomically, under its own path, the two C2a/R4 controls included, and no record overwrites another. A job that three launches start and leave unfinished ends its part of the run, its cause recorded, as in M8.18, and its stretch's ending is for the maintainer's adjudication, as for a failed control; a relaunch runs the same frozen job from its start, and a launch that ends early writes nothing that is read. The run goes in waves of at most four single-thread jobs, a job starting as another ends, and no wave starts before the one it depends on has ended:
1. **The controls:** M8.15's dynamics gate, C1, C2a, C2b and C3; then C3's rate run, `P5a`, C3 at `ε` = 3 and C2a/R4 at 1 with `T` = 2750.
2. **The gates:** the four continuations, T1 and T4 in both slots, and the coherent row's ladder, each with its gates, all written before any target evolves.
3. **The longest control,** C2a/R4 at 1 at the longest `T` the gates set, if any test needs more than 2750. Tests with `T` up to 2750 may run beside it.
4. **The tests,** up to four at once; a test longer than 2750 starts only after the longest control has Persisted, valid.
5. **The record.**

## DEFINITION OF DONE

Every stretch in § The stretches has its outcome, the coherent row's ladder has its own, and the record is filed for adjudication. On adjudication, the census page records where S5 stands, in a dated note.

## Dated note, 2026-11-10: after the adjudication

Everything above this heading is the filed pre-registration, byte for byte: its 21,303 bytes are a prefix of this file and hash to the governing `f2295e0b…`.

The maintainer adjudicated the run on [#705](https://github.com/openwave-labs/openwave/pull/705#pullrequestreview-5849541157) (2026-11-10): approve and merge, no findings, no maintainer edits; the record stands as filed. Gate 1 passed. T1/R4 lower and upper, T1/R5, T4/R4 and T4/R5 end Untested, gate 3 the first to fail at all 46 of their points; under the answer of 2026-11-07, S5 counts each of them for the census's close, with no persistence claimed either way. The coherent row in `R5` has its test: its ladder is established, Elliptic at all 15 points from 0.25 to 32, and its test at 32, with `T` = 1000, Persists, valid. How the census counts the ladder's types above 2 is not ruled. The roadmap's Done row states the outcome, and the author's dated note records where S5 stands on MIT, in § V of [the census page](https://github.com/dmobius3/mode-identity-theory/blob/d3083e8560d5dd81167b4d919a7cd91af7db5cf7/files/framework/files/working/files/slot-dynamics-census.md#v-the-program), as § DEFINITION OF DONE asks.

No rule, threshold or outcome changes.
