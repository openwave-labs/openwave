# M8.19 method note: the census's persistence tests (S5)

Task: [`tasks/m8_19_task_details.md`](../tasks/m8_19_task_details.md), governing SHA-256 `f2295e0b7f1d086a…`, pre-registered in [#698](https://github.com/openwave-labs/openwave/pull/698), run from 2026-11-10T03:53:24Z to 2026-11-10T14:42:59Z on the folder frozen at MIT `94a679ac`, with M8.15.x's solver at MIT `97a8c08`, under the maintainer's [go](https://github.com/openwave-labs/openwave/pull/698#issuecomment-6749289423). The first launch refused at its environment check, and under the [ruling](https://github.com/openwave-labs/openwave/pull/698#issuecomment-6755529199) the freeze was written once more; the [first](../m8_19/freeze_block.md) and [second](../m8_19/freeze_block_2.md) freeze blocks, of [#702](https://github.com/openwave-labs/openwave/pull/702) and [#703](https://github.com/openwave-labs/openwave/pull/703), record both. The record is [`RECORD.md`](https://github.com/dmobius3/mode-identity-theory/blob/a39604d29d22efbbeb14bb3d52d39b0f173ea22c/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/RECORD.md) beside the code. The run completed in one launch, with no stop. Gate 1 passed. The coherent row's ladder in `R5` is established, Elliptic at all 15 points to 32, and its test at 32 **Persists, valid**. T1's three stretches and T4's two end **Untested**, gate 3 the first to fail at every one of their 46 points. This note states the gates, the test and the outcomes as equations, maps each to its code, gives the results against the frozen terms, and says what the run does not verify. Adjudication is the maintainers'.

**Scope**: one registered run of M8.15.x's frozen finite-element dynamics under M8.19's frozen driver, on one machine: gate 1's eight controls; the gates on T1 and T4 in both slots and on the coherent row's ladder in `R5`; the plan; the one test the gates allow; and the record. It licenses each stretch's outcome and the ladder's as the record states them, and does not license orbital stability or instability, persistence beyond `T` or at another draw, a type at any point of T1 or T4, an ending under S5, a `MODELS.md` cell or a change to M8.7's gate.

## 1. Equations first

The instrument is M8.15.x's solver at MIT `97a8c08`, byte for byte: [M8.15's](m8_15_method_note.md) finite-element dynamics, with the stopping test its qualification repaired. The standing waves, their linearization, the frozen cluster check, the frozen types and the persistence test are its own. M8.19 adds the gates, the search, the plan, the coherent row's ladder and the record, and changes no instrument number.

### 1.1 The gates

```text
at each classified point ε of a stretch, read before any target evolves:
gate 2:   an accepted Newton solution at levels 3 and 2, on the continuation path from the seed
gate 3:   cluster_ok at levels 3 and 2:   dim G_tol = the frozen dimension,  G_tol closed under λ → −λ, λ̄, −λ̄,
                                          σ_min(Ω-Gram) ≥ 10⁻⁶ σ_max
          the mode, at both levels:       the slowest genuine imaginary mode of the reference's Krein sign;
                                          the candidates within 5% of its modulus counted;
                                          identified when the count is the reference's multiplicity
gate 4:   λ₂ and λ₃ within 5% of each other, one Krein sign;  where a harmonic reading exists, each within 5% of it
          (on the coherent row above 2, the reference is level 3 alone)
gate 5:   ‖P_B v_s‖² / ‖v_s‖² ≥ 0.01,  B the mode's real invariant subspace, in the energy norm
gate 6:   T = max(1000, P), rounded up to a multiple of 250,  P = 2π/|λ₂|
gate 7:   r T ≤ ln(1/(10η)),  r the largest real part among level 2's lifted or rigid eigenvalues
the search:  the stretch's highest point where gates 2 to 7 pass;  none: Untested, each point's first failed gate
```

### 1.2 The persistence test

```text
one draw at η = 10⁻³, the frozen seed, at level 2;  Verlet steps at dt = 0.8 · 2/ω_max, to T
observed every round(1/dt) steps, each distance relative to ‖Φ‖_M:
    d      the aligned position's deviation from Φ, in the M-norm
    d_⊥    the position's norm after the Ω-projection off G_tol
    d_rot  its M-projection onto the symmetry tangents
    d_ref  the unperturbed reference's deviation;   N and E, the charge and the energy
the first event:  d_rot > √η → Invalid (drift);  d_⊥ > 100η → Fails;  a tie is Invalid
otherwise:        Persists if max d_⊥ ≤ 10η over [0, T];  Unresolved if not
valid:            max d_ref ≤ η,  |ΔN|/N ≤ 10⁻¹⁰,  E_η > 0,  max|ΔE|/E_η ≤ 0.25,  secular drift ≤ 10⁻² E_η
```

### 1.3 The coherent row, the plan and the outcomes

```text
the ladder:     C2a's case in R5 from its seed along ladder(0.25, 32);  M8.15's frozen types at level 3,
                cross-checked at level 2
established:    the validity check at 0.25 passes, and the stretch on record, 0.25 to 2, reads M8.16.1's types
its stretch:    the unbroken run of Elliptic ladder points that contains 2;  any other run is named apart
the plan:       each stretch's test point and T from the gates' records;  T > 2750 waits for C2a/R4 at 1 at that T
the outcomes:   Persists or Fails: tested;  Unresolved: tested, no persistence conclusion;
                Invalid: S5's untested ending, the verdict beside it;  Untested: each point's cause
```

Gate 1 is the controls, before any target: M8.15's dynamics gate, C1, C2a and C2b Persisting and C3 Failing, with C3's rate run within 5% of the linear rate; control 1a's seed solve under the repaired test below 10⁻¹¹; C3 at `ε` = 3 Failing; and C2a/R4 at `ε` = 1, read through gates 3, 5, 6 and 7, Persisting, valid, at `T` = 2750.

## 2. Equation-to-code map

The driver's links are pinned to MIT `94a679ac`, the frozen folder, and the instrument's to `97a8c08`.

| Equation | Code |
| --- | --- |
| § 1.1 gate 3: the cluster check | [`Point`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L73), [`level_data`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L540) |
| § 1.1 gate 3: the mode | [`candidates`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_gates.py#L83), [`identify`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_gates.py#L90), [`level3_reference`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_gates.py#L109), [`mode_at`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_gates.py#L132) |
| § 1.1 gates 2 to 7 at a point | [`evaluate`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_gates.py#L149), [`read_points`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L544) |
| § 1.1 gate 5 | [`excitation`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L500) |
| § 1.1 gates 6 and 7 | [`test_T`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_gates.py#L120), [`margin`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_gates.py#L126), [`margin_ok`](https://github.com/dmobius3/mode-identity-theory/blob/97a8c0884c5e3a0ed25afb345f3d668340c5b537/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-15x/m8_15x_solver/persistence.py#L61) |
| § 1.1 the search | [`search`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_gates.py#L219), [`stretch_tests`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L590), [`job_gates`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L845) |
| § 1.2 the test | [`job_test`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L912), [`run_test`](https://github.com/dmobius3/mode-identity-theory/blob/97a8c0884c5e3a0ed25afb345f3d668340c5b537/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-15x/m8_15x_solver/persistence.py#L92), [`evaluate`](https://github.com/dmobius3/mode-identity-theory/blob/97a8c0884c5e3a0ed25afb345f3d668340c5b537/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-15x/m8_15x_solver/persistence.py#L161) |
| § 1.3 the ladder and its stretch | [`job_coherent`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L868), [`run_ladder`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L201), [`coherent_stretch`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_gates.py#L225), [`other_elliptic_runs`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_gates.py#L244) |
| § 1.3 the plan | [`job_plan`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L892), [`guard_long`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L479) |
| § 1.3 the outcomes | [`job_record`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L959) |
| gate 1 | [`job_dyn`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L605), [`job_p5a`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L617), [`job_c3e3`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L718), [`job_c2a`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L758), [`job_controls`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L793) |
| the run: the waves, the starts and the log | [`main`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_run.py#L351), [`launch`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_run.py#L374), [`Run`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_run.py#L80) |
| the freeze, admitted | [`validate_frozen`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L426), [`pinned_files`](https://github.com/dmobius3/mode-identity-theory/blob/94a679acc0c8ae9b7e002a063f0c31da8f82d538/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/run/m819_branch.py#L455) |

## 3. The run

- **The environment.** CPython 3.13.13, NumPy 2.5.0 and SciPy 1.18.0, on Accelerate's BLAS and LAPACK as each reports them, under macOS 15.8.2 (24H232) on one Apple M3, with the four threading variables at `1`, as `run/ENVIRONMENT.json` records it. The launch read it again by its own route, with no difference, and so did a reading after the run.
- **The order of the freeze.** The go at 2026-11-09T19:25:45Z; `FROZEN.json` written at 2026-11-09T19:53:23Z; #702 opened at 2026-11-09T20:36:41Z; the first launch refused at 2026-11-09T20:46:43Z, before its first job, since `main()` imports the driver, which sets the four threading variables, before it runs the environment check that the shell's record could not then pass; the ruling at 2026-11-10T00:41:59Z; the environment recorded by the launch's route and `FROZEN.json` written once more at 2026-11-10T00:55:03Z, pin `88940b293d440dd0…`, differing from the first in the environment record's pin, `written` and `go` alone; #703 opened at 2026-11-10T02:07:48Z and merged at 2026-11-10T03:07:14Z; then the run. The refused launch started no job and wrote no log.
- **The launch.** One launch on the second freeze, on mains power with sleep held off. `main()` ran `validate_frozen` and the environment check before `launch()`, and the console shows "differences: none".
- **The run.** Its log opened at 2026-11-10T03:53:24Z at the second freeze's pin and ended at 2026-11-10T14:42:59Z, 38975 s later: 17 jobs, each started once and completed, at most four at once and each on one thread, no job ended and no stop. The plan read the gates and added one test. Of the 64 outputs it logs, the 62 published unmodified match their logged SHA-256 directly; the 2 path-redacted console logs are published as archival copies, at SHA-256 values of their own, with their originals' logged SHA-256 in `ARCHIVAL.json`.
- **Deviations.** None in the run.

## 4. Results against the frozen terms

[`RECORD.md`](https://github.com/dmobius3/mode-identity-theory/blob/a39604d29d22efbbeb14bb3d52d39b0f173ea22c/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/RECORD.md) gives custody (§ 1), gate 1 (§ 2), every point's gates (§ 3), the coherent row (§ 4), the test (§ 5) and each stretch's outcome (§ 6). It recomputes every gate from its stored input with the frozen gates, and every dynamics verdict from its stored series with the frozen evaluator, and each equals its record.

| Stretch | Points | Outcome | Test | max d_⊥ | max d_rot | Reference floor |
| --- | --- | --- | --- | --- | --- | --- |
| T1/R4, lower | 5, 0.25 to 1 | Untested: gate 3 first at every point | none | | | |
| T1/R4, upper | 8, 2 to 22.63 | Untested: gate 3 first at every point | none | | | |
| T1/R5 | 14, 0.25 to 22.63 | Untested: gate 3 first at every point | none | | | |
| T4/R4 | 8, 0.25 to 2.83 | Untested: gate 3 first at every point | none | | | |
| T4/R5 | 11, 0.25 to 8 | Untested: gate 3 first at every point | none | | | |
| The coherent row in R5 | 15, 0.25 to 32 | Persists, valid | `ε` = 32, `T` = 1000, 3.33 periods | 9.96×10⁻⁴ | 1.68×10⁻⁶ | 1.68×10⁻¹³ |

- **Gate 1 passed.** C1, C2a and C2b Persist and C3 Fails at `t` = 583.28, with C3's rate run at 0.009741 against a linear 0.009737, M8.15's record to its digits; control 1a's seed solve stops below 10⁻¹¹; C3 at 3 Fails; and C2a/R4 at 1 Persists, valid, at `T` = 2750, a duration no record held.
- **Gate 2 passed at all 61 points,** T4's 19 among them: T4 now has finite-element standing waves on record, where M8.15's Newton failed at the seed.
- **Gate 3 failed first at every point of T1 and T4.** The level-3 cluster check failed at all 46. At T1's 27 the level-2 check failed too and no genuine imaginary mode of Krein sign − was found at either level, as the terms expected from M8.15's record. At T4's 19 the level-2 check failed at 15, and at 6 the level-2 mode was not identified: two candidates of Krein sign − of one recorded modulus, where the reference's multiplicity is 1. For T4 the terms named no expected outcome.
- **The coherent row in `R5`.** The ladder is established: 15 points from 0.25 to 32, every one Elliptic, no change of type, the validity check at 0.25 passing and the stretch on record reading M8.16.1's types; its stretch is the whole ladder, so no other Elliptic run is named. Gate 3 failed at 0.25, gate 4 at 0.3536, and gates 2 to 7 passed from 0.5 to 32, so the test went to 32.
- **The test.** At `ε` = 32, `T` = 1000, 3.33 periods of `P` = 300.3: **Persists, valid**. Its largest d_⊥, 9.96×10⁻⁴, is the perturbation's own size at `t` = 0, a tenth of the 10η that Persists allows; max d_rot is 1.68×10⁻⁶ against √η; the reference floor 1.68×10⁻¹³, the charge drift 1.68×10⁻¹³, the energy oscillation 0.0898 and the secular drift 9.60×10⁻⁴ of the perturbation's energy, each inside its bound.

## 5. Findings after the run

The readings in this section rest on the stored records. They are not further tests, score nothing and move no verdict.

### 5.1 T4 on the finite-element instrument

T4 reached gate 3 at all 19 points and stopped there: the level-3 cluster check fails at every point; at level 2 the check passes only at T4/R5's four lowest points, where two candidates of one recorded modulus leave the mode not identified. The census gives the leading diagnosis for T4 as the mesh's breaking of the continuous symmetry, INFERRED there. This run does not test it.

### 5.2 The fast-spectrum search

At level 2 the Krein count does not balance at 8 of T4's points (T4/R4 at 0.25, 0.3536 and 0.5; T4/R5 at 0.25, 0.3536, 0.5, 0.7071 and 8), and there the frozen code ran its fast-spectrum search, which did not balance it either. The search carries the defect M8.15.x's `REPRODUCE.md` records and leaves unrepaired, at its two real shifts, and its trace is a `ComplexWarning` in T4's two console logs. No gate reads the count or the search: the gates read each level's cluster check, `tau_re` and genuine modes alone, so the defect moves no gate (DERIVED from the frozen code).

### 5.3 The published files

The run's outputs and the freeze's files that `RECORD.md` § 1.7 lists are published in the record's folder; the bench's DEVLOG writers, its packet cover and the pull requests' text builders, which wrote prose, are left out. The files that name a local path go out as marked archival copies, with each original's SHA-256 and the copy's own in `ARCHIVAL.json`: the scripts and outputs the second block names, `freeze_block.py` and `freeze_block_2.py`, the first freeze's scripts, the second writing's `write_guard.out`, and 2 of T4's console logs, their originals at the SHA-256 the run's log records. `write_guard.out` corrects the second block, which says the other outputs go out as they are.

## 6. Provenance

- **Terms:** `tasks/m8_19_task_details.md` at OpenWave `8351bad3`, SHA-256 `f2295e0b7f1d086a…`.
- **Code:** MIT `94a679ac`, the folder the go names, and M8.15.x's solver at `97a8c08`, its manifest at `2b9864ab185e47f9…`.
- **Freeze:** `run/FROZEN.json`, pin `88940b293d440dd0…`, carried in full by [the second block](../m8_19/freeze_block_2.md); the first, pin `1e5107d0c6117a7d…`, kept and published, by [the first](../m8_19/freeze_block.md).
- **Record:** MIT [`a39604d`](https://github.com/dmobius3/mode-identity-theory/tree/a39604d29d22efbbeb14bb3d52d39b0f173ea22c/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19). It holds `RECORD.md`, `run/out/`, the freeze's files, `freeze/` and `freeze2/`, and the post-run files.
- **Reproduction:** `RECORD.md` § 11.

## 7. What this run does not verify

- **Orbital stability or instability.** A test that Persists is evidence to `T`, one draw at one `η`, not beyond.
- **The persistence of T1 and T4.** Their stretches are untested, each point's cause named; why gate 3 fails there is not tested.
- **A type at any point of T1 or T4.** None is read; their types stand as adjudicated.
- **The coherent row beyond its test point.** The test is at 32 alone.
- **An ending under S5.** The adjudication gives it, stretch by stretch.
- **Other machines.** The run is on one machine, and bitwise agreement is not claimed.
- **The platform's criteria.** No `MODELS.md` cell moves, and M8.7's gate is unchanged.

## 8. Audit record

Before filing, the record and this note were checked against the frozen terms, the frozen code and the run's files. `tools/build_record.py` rebuilt `RECORD.md` from the files with its refusals armed: the 62 unmodified outputs at their logged SHA-256, and the 2 path-redacted console logs at their copies' SHA-256, the index citing their originals' logged SHA-256; the freeze validating; the two freezes and the two environment records differing only as the ruling has them; every gate at every point recomputed from its stored input, with the search and the plan; every dynamics verdict recomputed from its stored series; `RECORD.json` against the folders; and each fact taken from `DEVLOG.md`. C3's rate is refitted as gate 1 fits it and compared within 10⁻¹² relative, since a fit's last digits move with the platform's linear algebra; every other field is compared exactly. This note's 35 code links resolve to their definitions at the pinned commits.
