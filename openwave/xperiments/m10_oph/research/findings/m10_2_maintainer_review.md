# M10.2 maintainer review: the static results recomputed, and the three row decisions

> Task: [M10.2](../tasks/m10_2_task_details.md). The author's record is the
> [method note](m10_2_method_note.md) and the [submission audit](m10_2_submission_audit.md).
> This note is the maintainers' side: an independent recomputation of the task's
> headline numbers, and the decision on rows 1, 2 and 7 with what would move each.
> The review as posted is on
> [pull request #723](https://github.com/openwave-labs/openwave/pull/723).
>
> **Verdict.** Every number this note recomputes agrees with the author's: the
> graph, the compact energies, the moments and the stopping times. The author's
> counts of calls, columns and branches were not recomputed. Rows 1, 2 and 7 stay
> not yet tested.
>
> **Scope**: finite arithmetic on the candidate's declared graph and endpoint rule,
> from one data file and the formulas of the author's method note; it licenses "the
> author's values for the quantities recomputed here are correct" and does not license any physical
> dictionary, any statement about the author's package (which was not run), or any
> `MODELS.md` icon.

## 1. The equations

All of these are the author's, from the M10.2 method note.

**The endpoint rule.** On a seam `(a, b)` with a fair coin `c ∈ {0, 1}` and integer
loads `q`:

```text
d = q_a − q_b
j = (d − (2c − 1)(|d| mod 2)) / 2
q' = q − j e_a + j e_b
```

**The mean one-event map and the energy convention.** On a graph with `|E|` seams
and Laplacian `L`, one event picks a seam and a coin uniformly, so the mean load
moves by

```text
T = I − L / (2|E|)            H = −2|E| log T
```

and a mode with `L f = x f` is assigned the energy `E = −2|E| log(1 − x / 2|E|)`.
The candidate has `|E| = 630`.

**The first-current moments.** On one cell, with port vectors `v` at
`(0, ±1, ±φ)` and cyclic, `φ = (1 + √5)/2`, and the source `+1` on a port `v`,
`−1` on its antipode, the sum runs over all 30 seams and both coins:

```text
Δ = j (v_b − v_a)        m = E[Δ]        C = E[(Δ − m)(Δ − m)ᵀ]
```

**The stopping time.** For a `+1` on port `p` and a `−1` on port `r` of one cell,
the pair cancels when an event moves them onto one port. The expected number of
events `t(p, r)` solves the linear system of the absorbing chain,
`t = 1 + Σ P(p, r → p', r') t(p', r')`, over the 132 ordered pairs.

**What is reported.** The structure of the graph; the residual of `L f = 6 f` for
the two compact rays; `E₆`, `E₁₂` and the pair energy at the 511 keV anchor; how
many modes of `L` lie below 6; `m`, `C∥`, `C⊥` on all twelve axes; the action of
the half turn `diag(−1, −1, 1)`; the stopping times; and the same structure under
two changed rules.

## 2. Equation-to-code map

One file,
[`m10_2_maintainer_recompute.py`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py).
Its docstring carries the same equations.

| Object | Function | Lines |
| --- | --- | --- |
| The cell: twelve ports, thirty seams | `cell` | [58-72](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py#L58-L72) |
| The endpoint rule's current `j` | `native` | [75-77](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py#L75-L77) |
| The two changed rules | `always_move`, `orientation_biased` | [80-87](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py#L80-L87) |
| `m` and `C` | `moments` | [90-98](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py#L90-L98) |
| Mean along `v`, `C` uniaxial, the half turn | `structure` | [101-118](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py#L101-L118) |
| The stopping times | `stopping_times` | [121-137](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py#L121-L137) |
| The cell's results assembled: the claimed moments on twelve axes, the half turn, the three rules | `part_a` | [140-168](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py#L140-L168) |
| The energy convention | `energy` | [171-172](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py#L171-L172) |
| The graph, the cells' linking pattern, `L f = 6 f`, the spectrum | `part_b` | [175-227](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py#L175-L227) |
| The comparisons and their wrong-value arms | `verdicts`, `arms` | [230-282](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py#L230-L282) |

## 3. Reproduction

Part A needs no input. Part B reads the candidate's graph definition, the file
`evidence/openwave/shared/localized_rest_join/evidence/graph.json` of the author
package at commit `c330c6e7f23506b5f87c339cd73906928c1a7cd9`. The script checks
that file's SHA-256 before computing anything from it, and reads it as JSON. It
imports nothing from the author's package.

```sh
python3 -I openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py
python3 -I openwave/xperiments/m10_oph/research/scripts/m10_2_maintainer_recompute.py \
  /path/to/graph.json --output /path/to/out.json
```

The graph file is not vendored: the author's record stays in the author's
repository, and the pin is the commit plus the hash.

## 4. Results

Stored in [`m10_2_maintainer_recompute.json`](../data/m10_2_maintainer_recompute.json).
Each line is compared with the author's value; each comparison turns false when it
is fed a wrong value (fifteen of fifteen).

| Claim | Result | How compared |
| --- | --- | --- |
| 240 positions, 630 seams | 20 cells of 12 ports and 30 seams, joined by 30 seams, three to a cell, each cell seeing the others at the distances of a dodecahedron's vertices; 60 ports carry an outside seam | exact |
| `L f = 6 f` for the rays `f` and `g` | residual 0 for both; neither support touches an outside seam | below 1e-12 |
| `E₆ = 6.014331228348673` | 6.014331228348651 | below 1e-12 |
| `E₁₂ = 12.057508280349891` | 12.057508280349847 | below 1e-12 |
| Pair energy, 1024.4508487023415 keV | 1024.4508487023415 keV | below 1e-9 |
| `m = (φ−3)/15 v`, `C∥ = (3−φ)/9`, `C⊥ = (φ+2)/15` | all twelve axes, 60 alternatives each | exact radicals |
| The half turn | a symmetry of the twelve ports; it sends four to their antipodes | exact |
| Expected stopping time, 66 to 90 events | 66, 84 or 90 over the 132 ordered pairs | exact rationals |

Three further facts the task does not state as headline numbers. They are reported, not compared with a claim:

| Fact | Value |
| --- | --- |
| Modes of `L` below 6, counting the constant mode | 80 of 240; the lowest nonzero is 0.0441. A further 70 sit exactly at 6 |
| Pair energy minus twice the anchor | +2.4508 keV, for two rays on disjoint, unconnected supports |
| The structure of the moments under a changed rule | a rule that moves a unit on every touched seam keeps it on all twelve axes, with a mean twice the native one; a rule whose weight depends on a seam's label order loses it on all twelve |

The excess of the pair comes from the logarithm in `H` alone: with `H = L` the two
energies add exactly.

## 5. The three rows

**The bar is the published one.** A row on a declared source stays not yet tested,
and a row moves when its observable is computed on the carrier's own process
([`m10_theory_canonical.md`](../m10_theory_canonical.md) § 8). In each row below
the native process does run. What the row names comes from what the candidate
declares, not from the run.

| Row | Decision | Why | What would move it |
| --- | --- | --- | --- |
| 1, charge quantization | stays | The loads are integers by the carrier's definition and the rule's current is an integer by its formula, so no run could return half a unit. The author's proof identifies the relative class with the load vector, `Z³⁰/im(D) = H₁(S², V; Z) = reduced H₀(V; Z)`, so the topological statement and the integer bookkeeping are one statement. What the law adds is that an opposite pair meets and cancels | The unit as an outcome of the law: larger loads allowed, with single units the only settled records against a changed-law control; or fractional loads native, with the law confining settled records to whole units; or an integer class the law cannot change, computed where other values are representable. Each is a new candidate, since this source holds one `+1` and one `−1` |
| 2, electron rest energy | stays | One anchor fixes one number and the row offers one number. The anchored mode is a chosen one, with 80 modes below it. Two separated copies come out 2.45 keV above twice the anchor. `T` is a decay factor of the mean walk, 209/210 per event, and the note establishes no physical clock | A state the dynamics holds or returns to, and the anchored quantity doing a second job that can fail: exact additivity for separated copies, or a second unanchored number that agrees with measurement |
| 7, spin-½ statistics | stays | For any rule that is equivariant under the cell's rotations, the five-fold symmetry about `v` puts the mean along `v` and makes the covariance uniaxial about `v`; the half turn then reverses the first and returns the second on the four axes it exchanges end for end. The native law contributes the three magnitudes and the sign of the mean. The author's note excludes spinor phase, exchange statistic and continuous rotation | A factor of two that the process produces and that the cell's symmetry does not fix: a quantity that changes sign under a full turn and returns after two, read by interference (a new candidate, since this memory law has positive real amplitudes); or a field whose return under a half turn is an outcome, against a control law that keeps `v` and `−v` apart |

## 6. Not computed, not run

- The author's package was not executed. Its two platform scripts fetch and run it;
  it holds 329 files, too many to read in full before a run
  ([`PR_REVIEW_STANDARDS.md` § 3.1](../../../../../dev_docs/PR_REVIEW_STANDARDS.md#31-the-safety-gate-before-anything-runs)).
  The package's counts of calls, columns and branches are therefore not confirmed.
- The gates P1 to P6 of the task record are the author's replay, on the author's run.
- The Lean statements, the spanning-tree certificate, the stopped-response formula
  and the eigenray certificate.
- Anything that uses the coherent memory law. The recomputed quantities are
  averages over the classical branches.

## 7. Minimal inspection set

1. [`m10_2_maintainer_recompute.py`](../scripts/m10_2_maintainer_recompute.py), the docstring first.
2. [`m10_2_maintainer_recompute.json`](../data/m10_2_maintainer_recompute.json).
3. The author's [method note](m10_2_method_note.md), for the equations as the author states them.

## 8. Audit

The review as posted was fact-checked twice by a second model before it went out.
The first pass confirmed every number by its own computation and found four
wording errors in the draft, the largest in the argument for row 7; the second
pass found one more, in the first way forward offered for row 1. All five were
corrected before posting. This note, its script and the three cells were then
audited again by a second model before they landed.
