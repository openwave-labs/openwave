# M8.18 method note: the census's two remedies

Task: [`tasks/m8_18_task_details.md`](../tasks/m8_18_task_details.md), governing SHA-256 `924e7ee363a756a8…`, pre-registered in [#687](https://github.com/openwave-labs/openwave/pull/687), run from 2026-10-31T21:51:23Z to 2026-11-01T13:17:32Z on the code frozen at MIT `4bd8bbd`, under the maintainer's [go](https://github.com/openwave-labs/openwave/pull/687#issuecomment-6548866246) and the [freeze block](../m8_18/freeze_block.md) of [#688](https://github.com/openwave-labs/openwave/pull/688). The record is [`RECORD.md`](https://github.com/dmobius3/mode-identity-theory/blob/5241a2d50baae390f8b90a2c8ae0ba08366236d4/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/RECORD.md) beside the code. The run completed, with no stop. Arm B located T4/R5's change of type, so that row now ends Each change of type located. Arm A ended at the control gate: the kernel at `N = 40` passed on all seven controls, and 22 of the 24 control points read converged, but P5 and C3R/R4 at `ε = 0.25` read Unresolved, so none of the twelve was read. This note states Arm B's one change, Arm A's gates and the row outcomes as equations, maps each to its code, gives the results against the frozen terms, and says what the run does not verify. Adjudication is the maintainers'.

**Scope**: one registered run of M8.16.1's frozen harmonic Galerkin instrument, with two arms: Arm B, the revised locator on T4/R5's bracket at `N = 16, 24` and 32, and Arm A's gates, the kernel at `N = 40` on seven controls and the control gate on six, on one machine. It licenses T4/R5's location and the gate's outcome as the record states them, and does not license a type at any of the twelve, a reading of the rule above `ε = 2` on the ladder 24, 32 and 40, the location's identification with the genuine index crossing, the persistence of any Elliptic stretch, an S5 or S6 ruling, a `MODELS.md` cell or a change to M8.7's gate.

## 1. Equations first

The instrument is M8.16.1's, byte for byte, as [M8.16.1's method note](../findings/m8_16_1_method_note.md) and, through it, [M8.16's](../findings/m8_16_method_note.md) give it: the law, the discretization, the exact kernel, the bordered Newton, the seeds and continuation, the linear record, K1, the convergence rule with its band-edge guard, and branch identity. `N = 40` is an input the pinned code accepts. The locator and the row outcomes are [M8.17's](../findings/m8_17_method_note.md), §§ 1.3 and 1.4, with the changes below.

### 1.1 Arm B: the one change

```text
the bracket:   T4/R5's (8, 11.31), and no other; N = 24 and 32 decide, N = 16 is reported
the path:      each cutoff's, recomputed from its seed to 11.31, each point equal to its record in M8.17's outputs
bisection, at each N:
    m = (a + b)/2, solved by continuation from a and from b
    if both solves are accepted and K1 fails on either:
        m' = a + 0.4 (b − a), solved from a and from b, once;  m stays in the record and decides nothing
        K1 failing at m' is an unsuccessful stop, as at any midpoint
    the two solutions agree:  aligned distance ≤ 10⁻² ‖φ‖,  n_N equal;   K1 holds;   n_N read by inertia
    keep the half whose ends differ in n_N
    complete when  b − a < (w/2) · a,   w = 10⁻³
```

The fraction and the direction were chosen from M8.17's locator record, as the terms state. Each of M8.17's midpoints must equal its stored record, timings aside. The location and its guards are M8.17's: `δ = 0.01`, then 0.05 once, each guard read at all three cutoffs at one `ε`, on the band above `N = 16`, and classified with its end's type.

### 1.2 Arm A's gates

```text
the kernel at N = 40:   each control's seed passes its check;  at each registered ε, accepted(ε) and K1(ε),
                        K1 as M8.16.1's terms freeze it, its calibration constants unchanged;
                        C1:  checks.check_c1 over its four cutoffs, its record equal to M8.16.1's
the control gate:       for C2a, C2b, C3, C2a/R5, P5 and C3R/R4, at ε = 0.25, 0.5, 1 and 2:
                        the rule on N = 24, 32 and 40, on the band above N = 16, reads converged,
                        with the type and (n(L), n(D), K) on the control's record;
                        identity(ε; 24, 32) and identity(ε; 32, 40)
a failure at either:    Arm A ends;  no target is solved or read, at any cutoff;  the twelve stay unclassified
```

### 1.3 The row outcomes

```text
Elliptic throughout:           every window point classified Elliptic
Each change of type located:   every point classified, or between a located change's guard readings;
                               every type bracket located
Not settled by this run:       otherwise
```

A row's classified points are M8.17's and this run's. Where a gate stops the targets, each of the twelve keeps M8.17's cause with the gate named, and that is not a second outcome with that cause. Whether a row meets an S4 completion condition is stated separately.

## 2. Equation-to-code map

The links are pinned to MIT `4bd8bbd`, the frozen code. The instrument's map is [M8.16's note](../findings/m8_16_method_note.md), § 2, and M8.16.1's driver and checks are mapped in [its note](../findings/m8_16_1_method_note.md), § 2.

| Equation | Code |
| --- | --- |
| § 1.1 the bisection and the moved midpoint | [`bisect`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L412), [`aligned`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L189) |
| § 1.1 M8.17's midpoints reproduced | [`gate_midpoints`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L240) |
| § 1.1 the location and the guards | [`location`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L455), [`guard_offsets`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L486), [`guards`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L494), [`span`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L518) |
| § 1.1 Arm B, end to end | [`locate`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L542), [`settle`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L573), [`Run.arm_b`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L421) |
| § 1.2 the controls' paths at `N = 24` and 32 | [`Run.lower`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L404), [`gate_cutoff`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L207), [`gate_identity`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L218) |
| § 1.2 the kernel | [`kernel_failures`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L319), [`Run.c1`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L511) |
| § 1.2 the control gate | [`control_failures`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L331), [`converged`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/checks.py#L224), [`rule`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L212), [`Run.case_job`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L387), [`Run.gates`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L465) |
| the band | [`band_value`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L143), [`band_ok`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L149), [`band_failures`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L347) |
| § 1.3 the row outcomes | [`classify_rows`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L667), [`assemble`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L713), [`row_outcome`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L623), [`meets_s4`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census_lib.py#L641) |
| the run: its order, the stop and the resume | [`main`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L677), [`Run.go`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L615), [`Run.stop`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L293), [`Run.resume`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L592) |
| the freeze: written once, then admitted | [`write`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/write_guard.py#L69), [`frozen`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/census.py#L89) |
| the freeze block | [`build`](https://github.com/dmobius3/mode-identity-theory/blob/4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/run/freeze_block.py#L39) |

## 3. The run

- **The environment.** Python 3.13.13, NumPy 2.5.0 and SciPy 1.18.0, on Accelerate's BLAS and LAPACK as each reports them, under macOS 15.8.2 (24H232) on one Apple M3, as `run/ENVIRONMENT.json` records it. It was read just before the launch, by the launch itself, and again after the run, with no difference.
- **The order of the freeze.** The code on MIT `main` at `4bd8bbd`, with its manifest pin, `6d244f4af56b71d6…`; the terms merged in #687; the maintainer's go; `run/write_guard.py`'s `FROZEN.json` at 2026-10-31T21:13:28Z, with its pin, `b5902494f927ec5d…`; the freeze block, whose pull request was opened at 2026-10-31T21:17:17Z; then the run. The freeze block merged at 2026-10-31T22:01:22Z, after the run started, as its pull request says it may. `census.py` refuses to start unless the freeze validates.
- **The launch.** One launch, on mains power with sleep held off. The command checked a read-only resume check against its reviewed SHA-256, `9405929a8518f5f5…`, ran it, ran `record_environment.py --check`, and only then `run/census.py`. The check sat outside the pinned tree and changes nothing in it. The console log opens with the hash check's `OK`, the check's `PASS` and `differences: none`, then holds the run's transcript line for line.
- **The run.** It started at 2026-10-31T21:51:23Z and wrote its record at 2026-11-01T13:17:32Z, 55569 s later. Arm B's record was written after 19217 s, before any solve at `N = 40`. Its log records the manifest pin at the start, one start, 57 jobs each started once and completed, no resume and no stop.
- **Deviations.** None in the run.

## 4. Results against the frozen terms

[`RECORD.md`](https://github.com/dmobius3/mode-identity-theory/blob/5241a2d50baae390f8b90a2c8ae0ba08366236d4/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18/RECORD.md) gives Arm B in full (§ 2), the kernel and each control's reading (§ 3), the twelve with their causes (§ 5) and each row's standing (§ 7).

| Row | Slot | M8.17's outcome | This run's | Meets an S4 completion condition | Classified | Open |
| --- | --- | --- | --- | --- | --- | --- |
| T1/R4 | `R4` | Not settled by this run | Not settled by this run | no | 13 of 18 | 1.41, 32, 45.25, 64 and 72 |
| T4/R4 | `R4` | Not settled by this run | Not settled by this run | no | 13 of 18 | 4, 8, 11.31, 64, 72 and the bracket (2.83, 5.66) |
| T1/R5 | `R5` | Not settled by this run | Not settled by this run | no | 14 of 15 | 32 |
| T4/R5 | `R5` | Not settled by this run | Each change of type located | yes | 15 of 15 | none |
| T5/R2 | `R2` | Not settled by this run | Not settled by this run | no | 17 of 20 | 45.25, 128 and 132 |

- **Arm B: located.** All 12 recomputed path points and all of M8.17's midpoints reproduced. K1 refused `ε = 8.721638` at both deciding cutoffs, on both solutions, and held at the moved point, `ε = 8.720990`, where `n(L)` reads 7. So `N = 24` and 32 complete on [8.720990, 8.724874], 4.5×10⁻⁴ of its lower end, with `n(L)` changing by −1. `N = 16`'s interval, [8.851079, 8.854315], is reported and does not decide. The run names it: "a change of type located between the guard readings, with n(L) changing by -1 inside the interval at N = 24 and 32". T4/R5's row now ends Each change of type located, every point classified.
- **The guards.** The lower, at 8.633781, reads Elliptic. The upper at `δ = 0.01`, 8.812122, reads Unresolved, since `N = 16`'s change lies above it, with `n(L)` 7 at `N = 16` against 6 at `N = 24` and 32, as the terms inferred. The retry at `δ = 0.05`, 9.161117, reads Hyperbolic, and counts. Each guard's stored band is `R5`'s frozen level, 18, as `RECORD.md` § 2 shows for the third note of #687's approval.
- **The kernel at `N = 40`** passed on all seven controls. C1's record, `N = 40` included, equals M8.16.1's.
- **The control gate did not pass.** All six controls were read and recorded before it decided. Branch identity held on both pairs at all 24 points, and 22 points read converged with their recorded type and indices. The two that did not: P5 at `ε = 0.25`, "tested set at 8.67568e-16+4.02029j (+1): non-contracting above the floor"; and C3R/R4 at `ε = 0.25`, "structure: a record is unresolved", from its `N = 40` record's "n(D): an eigenvalue of D lies in its gray zone". As the terms state, "A failure inside Arm A's gates ends Arm A. The twelve stay unclassified, and the record is written."
- **The twelve.** None was solved or read, at any cutoff. Each stays unclassified and keeps M8.17's cause, with the gate named in `run/out/CENSUS.json`, and that is not a second outcome with that cause.
- **The brackets.** T4/R4's change of type, Elliptic at 2.83 to Hyperbolic at 5.66, lies across the unclassified grid point 4. It has no type bracket and is not located, and no new bracket exists.

## 5. Findings after the run

The readings in this section are post-run findings, not further tests. They rest on the stored records and on `dev/runs/019_gate_cause/` in the record, a diagnostic made after the run on a copy of the folder. It scores nothing and moves no verdict. No outcome is assigned here.

### 5.1 Why the gate failed

In both cases a tolerance that grows with the cutoff reaches a value that does not move (DERIVED from the records and the diagnostic's reads of stored solutions, which reproduce the stored records on the recorded machine).

- **P5 at `ε = 0.25`.** Two Krein-positive eigenvalues near 4.0203 lie 1.1345×10⁻⁵ apart at all three cutoffs, read before they are linked. The record links eigenvalues within `τ(λ) = max(margin, 10 sqrt(u ‖A‖ |λ|))`, and `τ` is 7.173×10⁻⁶, 9.636×10⁻⁶ and 1.207×10⁻⁵ at the three cutoffs, so it passes the gap at `N = 40` alone, by 6.4%. There the pair is stored as one group at its mean, and the rule compares that one centre with the two below it: a change of half the gap, 5.673×10⁻⁶, against 5.390×10⁻¹¹ from 24 to 32. The pair's mean moves by 5×10⁻¹⁴, and its members, read before they are linked, by less than 10⁻¹³, the size of rounding. It is the only such pair inside the tested band at any of the 24 control points.
- **C3R/R4 at `ε = 0.25`.** D's negative eigenvalue is -0.01596075 at all three cutoffs. `n(D)`'s gray zone reaches `10⁻⁵ ‖B‖`, which is 0.01040 at `N = 32` and 0.01632 at `N = 40`, so it passes 0.01596075 at `N = 40` alone, by 2.3%. D's zero eigenvalues read below 10⁻¹⁷ of ‖B‖ there.

At both points the `N = 40` record reads Hyperbolic with (5, 1, 4), the recorded type and indices, and both failures are at the controls' smallest amplitude. The gate asks for a converged reading, and it was applied as the terms state. The tolerances are not new. What is new is the cutoff: ‖A‖ and ‖B‖ grow about as `N(N + 2) - λ`, and `N = 40` is the first rung where these two features fall inside them. Neither crossing needed a sealed reading, and no check of the frozen tolerances at `N = 40`'s norms was run before the filing.

### 5.2 T4/R5's moved midpoint

The terms' interpolation (INFERRED) put the nearest eigenvalue at the moved point at about 1.16×10⁻⁵ and 1.26×10⁻⁵, and the run reads 1.16×10⁻⁵ and 1.26×10⁻⁵ at `N = 24` and 32. `n(L)` reads 7 there, the branch the terms give a 3.88×10⁻³ interval, narrower than the 4.36×10⁻³ that completes the bisection. The answer of 2026-10-29 foresaw this, "since one midpoint that K1 accepts can finish that bisection", and the terms state that the fraction and the direction were chosen from M8.17's locator record. The location rests on that one midpoint, as a location's interval rests on its last, and the guards read on either side of it.

### 5.3 S6

This run makes no S6 ruling. The census page applies S6 to each item after the adjudication, with the cause the adjudication names. The gate's stop is not a second outcome with M8.17's causes, which is S6's first sentence. S6's last sentence closes as instrument-limited the rows a root cause blocks, where the maintainer admits no instrument that removes it, and the answer of 2026-10-29 leaves a point this run does not classify not offered to this instrument again. That answer also applies S6 to T5's row when its point at 45.25 has been read again, and this run did not read it, which is a question for the adjudication. The twelve, and T4/R4's change across 4, are in neither case the census page's note of 2026-10-30 names: a remedy was admitted for them and ran, and its gate stopped before it read them. The request of 2026-10-29 asked: "if only one remedy is admitted, or an admitted remedy ends without resolving its obstruction, may the remaining items close the same way, or does each cause need its own ruling?", and read the ruling of 2026-10-20 as covering these cases, "through S6's first sentence where a remedy meets the same cause again". The answer reads "Yes, your reading of S6 holds", and names two cases: no remedy admitted for an item, and a remedy that meets the same cause again. How that reaches them is a question for the adjudication too.

## 6. Provenance

- **Terms:** `tasks/m8_18_task_details.md` at OpenWave `a81d50c9`, SHA-256 `924e7ee363a756a8…`.
- **Code:** MIT `4bd8bbd`. Its manifest pins 37 files, `6d244f4af56b71d6…`, and `SHA256SUMS`, regenerated at the record's commit, lists every file of the folder.
- **Freeze:** `run/FROZEN.json`, pin `b5902494f927ec5d…`, carried in full by [the freeze block](../m8_18/freeze_block.md).
- **The resume check:** `launch/resume_check.py` in the record, SHA-256 `9405929a8518f5f5…`, with its 21 arms.
- **Record:** MIT [`5241a2d`](https://github.com/dmobius3/mode-identity-theory/tree/5241a2d50baae390f8b90a2c8ae0ba08366236d4/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-18). It holds `RECORD.md`, `run/out/`, the freeze's files, the resume check and the post-run files.
- **Reproduction:** `REPRODUCE.md` beside the code, and `RECORD.md` § 12.

## 7. What this run does not verify

- **A type at any of the twelve.** None was read. No type is claimed there.
- **Whether `N = 40` resolves the twelve's causes.** The gate stopped the run before any of them.
- **The rule above `ε = 2` on the ladder 24, 32 and 40.** No registered control is read there, as the terms' stated limits say.
- **Why K1 fails near T4/R5's change.** The record shows where it fails and that the moved point holds. The cause is OPEN.
- **The location's identification** with the genuine index crossing. T4/R5's location is an `n(L)` interval with two guard readings. The Hamiltonian–Krein Identification's Corollary I is a conditional test, and it is not applied here.
- **The persistence of any Elliptic stretch.** S5 needs a pre-registration of its own, outside this admission.
- **The calibration's range.** `N = 40` sits at the edge of K1's calibrated range, and `R5` lies outside the fitted slots (the terms, § STATED LIMITS: WHAT IS ADDED).
- **Other machines.** The run is on one machine, and bitwise agreement is not claimed.
- **The platform's criteria.** No `MODELS.md` cell moves, and M8.7's gate is unchanged.

## 8. Audit record

Before filing, the record and this note were checked in three rounds against the frozen terms, the frozen code and the run's files. The checks rebuilt `RECORD.md` from the files and reran its 18 refusal arms. They recomputed from `run/out/` all 24 gate rows, with the rule's verdict, the type and indices at `N = 40` and both aligned distances; the kernel table; each row's counts and open items; and the twelve's causes against M8.17's. They traced Arm B through its 12 path points, its midpoints, K1's refusal at 8.721638, the moved point and the three guards. They checked all 78 logged output files against their SHA-256, the freeze against its pin and the freeze block, the terms and the manifest against their pins, and the run's log for one start, 57 jobs each started once, and no stop, no ended job and no target job. They reran the pre-freeze tests on a copy of the post-run folder, and reran the tolerance reads and the read of P5's pair on a second machine, where the pair's gap and its linking agree and `margin` and ‖A‖ differ only in their last bits. They walked this note's 34 code links, each on its definition, and checked each quotation against its source. They led to five changes:
- P5's pair read before linking at all three cutoffs, so that § 5.1's grade holds at both control points, with the builder's check on that read and its arms;
- § 5.3's statement that the twelve, and T4/R4's change across 4, are in neither case the census page's note of 2026-10-30 names, with the request's question given in full;
- the movement of P5's members stated as less than 10⁻¹³, the size of rounding, not in digits that differ between machines;
- two wording corrections in the record, in its §§ 1 and 4;
- `REPRODUCE.md`'s line on where the tests write.

All were adopted, and no reading changed.
