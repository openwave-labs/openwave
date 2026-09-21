# M8.12 method note: the reduced Morse census of the level-6 quartic

The blind-run record for [M8.12](../tasks/m8_12_task_details.md), pre-registered at
[#571](https://github.com/openwave-labs/openwave/pull/571) and run on the merged text. It states the
equations first, maps each one to the code that computed it, and sets the returns against the frozen
claims. It follows [`dev_docs/METHOD_NOTE.md`](../../../../../dev_docs/METHOD_NOTE.md).

## 1. Equations first

### 1.1 The object

Everything happens in one seven-dimensional complex space, with no quotient and no bundle. Write
`V₃ = ℂ⁷` with basis `v₃, v₂, v₁, v₀, v₋₁, v₋₂, v₋₃`, Hermitian product `⟨u, w⟩ = Σ_m conj(c_m) d_m`,
and real inner product `Re⟨·,·⟩`. Rotations act through the spin-3 representation, fixed by

```text
J_z v_m = m·v_m,   J_± v_m = √(12 − m(m ± 1))·v_{m±1},   D³(n, θ) = exp(−iθ·(n·J)).
```

Time reversal is `(Θu)_m = (−1)^{3−m}·conj(c_{−m})`. With the Clebsch-Gordan coefficients
`⟨3 m₁; 3 m₂ | 6 Q⟩` in the Condon-Shortley convention, fixed by `⟨3 3; 3 3 | 6 6⟩ = +1`,

```text
ρ₆(u)_Q = Σ_{m₁} ⟨3 m₁; 3 (Q − m₁) | 6 Q⟩ · c_{m₁} · (Θu)_{Q − m₁},

r̂₆(u)  = Σ_Q |ρ₆(u)_Q|² / ‖u‖⁴.
```

`r̂₆` is real, invariant under `u ↦ e^{iφ}u` and under every rotation, and homogeneous of degree 0,
so it descends to `ℙ(V₃)`.

### 1.2 The loci, and why a critical point of a restriction is critical

For a closed subgroup `H` of `SO(3)` and a one-dimensional character `χ : H → U(1)`,

```text
V₃^{(H, χ)} = {u ∈ V₃ : D³(h)u = χ(h)·u for every h in H}
```

is the fixed space of the graph subgroup `{(χ(h)⁻¹, h)}` of `U(1) × SO(3)`, which acts linearly and
preserves `r̂₆`. By Palais' principle of symmetric criticality, a critical point of `r̂₆` restricted to
such a fixed space is critical for `r̂₆` on the whole sphere. That is what makes a finite search over
subgroups and characters into a census rather than a sampling: the question M8.12 asks is which of
these loci have projective dimension at most one, and what happens at their critical points.

### 1.3 The second variation, and the space it is read on

For a unit critical `u`, set `H_u(e, e) = d²/ds² r̂₆(u·cos s + e·sin s)` at `s = 0`, extended by
polarization. It is a self-adjoint form on the 13-real-dimensional `T_u = {e : Re⟨u, e⟩ = 0}`. Writing
`N(x) = ‖ρ₆(x)‖²`, so that `r̂₆ = N·D^{−2}` with `D = ‖x‖²`, degree-0 homogeneity makes the full
gradient vanish at a critical `u` and gives

```text
Hess(N·D^{−2})(u) = Hess N(u) − 4N(u)·I − 8N(u)·u uᵀ,
```

whose last term drops on `T_u`. So on `T_u` the second variation is the matrix

```text
M_u = Hess N(u) − 4N(u)·I.
```

The phase and rotation directions `O_u = span_ℝ{i·u, −i·J_x u, −i·J_y u, −i·J_z u}` are null for it,
being tangent to an orbit along which `r̂₆` is constant, and the census is read on the transverse
space `N_u = T_u ⊖ O_u`. `dim O_u` is 3 at a weight state and 4 elsewhere, so `dim N_u` is 10 or 9.

⚠️ `M_u` and `H_u` agree on `T_u` only where `∇N(u)` is orthogonal to the direction tested: the two
differ by `−4u(∇N(u)·d)`. At a critical point this is vacuous, which is exactly why the run carries a
deliberately non-critical control point where it is not.

### 1.4 What one signature gives

For a reduced energy proportional to `g·r̂₆` on the unit sphere, the signature `(n₋, n₀, n₊)` of `M_u`
on `N_u` gives the reduced Morse index at both signs of the coupling in one computation: `n₋` for
`g > 0` and `n₊` for `g < 0`. A kernel is a genuine degeneracy of the reduced quartic, not a numerical
artifact, only if it survives an exact route.

### 1.5 The sector bridge

The pinned pointwise law enters only through M8.1.2's D7. In each sector the reduced quartic on the
block is

```text
Q_σ = 1 + w₆(σ)·r̂₆,   w₆(3′) = 28/39,   w₆(4) = 21/52,
```

both positive. So the signature is sector-independent, and in a sector the form is `w₆(σ)` times the
one computed here. Nothing in this note re-derives sections, and nothing uses `m8_5c/design_inputs/`.

### 1.6 What this note does not claim

It computes the Morse index of the leading reduced quartic on the block, transverse to the
phase-and-rotation orbit. That is not the stability of a branch germ, and a branch germ's stability
is not particle stability. No MODELS.md cell moves and M8.7's gate is unchanged. Orbits whose
stabilizer is `C₂`, or `C₃` acting on its three-dimensional fixed space, or trivial, lie outside the
claim.

## 2. What the agents derived, none of it supplied

Two rooms received one packet: `worklist.md` byte-identical to the merged file, and a brief carrying
only room mechanics. Neither received this document, the frozen values, the loci, the isotropy
answers, or any statement about the two ends. Both derived the classification themselves.

| Derived in both rooms | Supplied |
|---|---|
| The classification of the character-fixed loci, and which have projective dimension at most one | The conventions of § 1.1, as formulas |
| The complete critical set on each locus, by elimination | `Q_σ = 1 + w₆(σ)·r̂₆` with both weights, stated as an input and not to be derived |
| The union of the critical points modulo rotation and phase, and the separation of the value ties | The definition of `H_u` as a second derivative along a great circle |
| The transverse dimension, signature and characteristic polynomial at each orbit | Nothing else |
| The kernel, and its identification | |
| Both ends of `r̂₆` on the sphere | |

The run was **offline**, the stricter of the two postures the pre-registration admits, so items 1 to
4 and item 8 are derived rather than located. Both rooms declared what they recalled rather than
derived: the classification of the closed subgroups of `SO(3)`, the character formula, Schur's lemma
and Poincaré-Hopf. Both also called a library's Clebsch-Gordan routine and reported it as a
cross-check against their own construction, agreeing on all coefficients, which is what the brief
asked for.

## 3. Equation-to-code map

The equation numbers below are § 1's.

### 3.1 The two solver rooms

| Equation | solver_a | solver_b |
|---|---|---|
| `ρ₆`, `r̂₆` (§ 1.1) | [`common.py`](../scripts/m8_12_solver_a/common.py), [`item0.py`](../scripts/m8_12_solver_a/item0.py) | [`core.py`](../scripts/m8_12_solver_b/core.py), [`item0.py`](../scripts/m8_12_solver_b/item0.py) |
| The fixed loci (§ 1.2) | [`classes.py`](../scripts/m8_12_solver_a/classes.py), [`item1.py`](../scripts/m8_12_solver_a/item1.py) | [`item1.py`](../scripts/m8_12_solver_b/item1.py), [`rot.py`](../scripts/m8_12_solver_b/rot.py) |
| The critical sets on the loci | [`item2.py`](../scripts/m8_12_solver_a/item2.py), [`item3.py`](../scripts/m8_12_solver_a/item3.py) | [`item2.py`](../scripts/m8_12_solver_b/item2.py), [`item3.py`](../scripts/m8_12_solver_b/item3.py) |
| The union modulo rotation and phase | [`item4.py`](../scripts/m8_12_solver_a/item4.py), [`orbits.py`](../scripts/m8_12_solver_a/orbits.py) | [`item4.py`](../scripts/m8_12_solver_b/item4.py) |
| `M_u = Hess N − 4N·I` on `N_u` (§ 1.3) | [`hess.py`](../scripts/m8_12_solver_a/hess.py), [`item5.py`](../scripts/m8_12_solver_a/item5.py) | [`hess.py`](../scripts/m8_12_solver_b/hess.py), [`item5.py`](../scripts/m8_12_solver_b/item5.py) |
| The in-locus second variations | [`item5b.py`](../scripts/m8_12_solver_a/item5b.py) | [`item5b.py`](../scripts/m8_12_solver_b/item5b.py) |
| The Morse index at both signs (§ 1.4) | [`item5.py`](../scripts/m8_12_solver_a/item5.py) | [`item6_10.py`](../scripts/m8_12_solver_b/item6_10.py) |
| The kernel | [`item5.py`](../scripts/m8_12_solver_a/item5.py) | [`item7.py`](../scripts/m8_12_solver_b/item7.py) |
| The two ends | [`sos.py`](../scripts/m8_12_solver_a/sos.py), [`item8.py`](../scripts/m8_12_solver_a/item8.py) | [`item8.py`](../scripts/m8_12_solver_b/item8.py) |
| The sector bridge (§ 1.5) | [`item10.py`](../scripts/m8_12_solver_a/item10.py) | [`item6_10.py`](../scripts/m8_12_solver_b/item6_10.py) |
| The control point (§ 1.3 ⚠️) | [`item11.py`](../scripts/m8_12_solver_a/item11.py) | [`item11.py`](../scripts/m8_12_solver_b/item11.py) |
| The two routes to `H_u` | [`item12.py`](../scripts/m8_12_solver_a/item12.py) | [`item12.py`](../scripts/m8_12_solver_b/item12.py) |

### 3.2 The audit room, [`../scripts/m8_12_audit/`](../scripts/m8_12_audit/)

Built its own `ρ₆` by lowering from `|6 6⟩ = v₃ ⊗ v₃` with an exact `J² = 42` check
([`aud_core.py`](../scripts/m8_12_audit/aud_core.py)), and its own rational weighted coordinates
([`aud_acoords.py`](../scripts/m8_12_audit/aud_acoords.py)). It imported, called and copied no solver
script to produce any number of its own.

### 3.3 The maintainer's route, [`../scripts/m8_12_maintainer/`](../scripts/m8_12_maintainer/)

Run before any room opened, so that a room reported as differing differs from a table that has itself
been tested.

| What | Script | Result |
|---|---|---|
| The frozen values, machine-readable | [`frozen_claims.json`](../scripts/m8_12_maintainer/frozen_claims.json) | the encoding under test |
| Internal consistency of that encoding | [`check_frozen.py`](../scripts/m8_12_maintainer/check_frozen.py) | 131 of 131, and five planted defects each fire |
| Independent recomputation from § 1's definitions | [`recompute.py`](../scripts/m8_12_maintainer/recompute.py) | 66 of 66 |
| The line restrictions against `r̂₆` itself | [`verify_lines.py`](../scripts/m8_12_maintainer/verify_lines.py), [`verify_frozen_lines.py`](../scripts/m8_12_maintainer/verify_frozen_lines.py) | 14 room restrictions and 7 frozen ones, all correct in their own charts |
| The room returns against the frozen table | [`compare.py`](../scripts/m8_12_maintainer/compare.py) | 70 of 70 on a synthetic perfect return, five planted defects fire |

## 4. Results against the frozen claims

Every claim is **REPRODUCED**. No claim is defective and none is unresolved.

| ID | Verdict | Route |
|---|---|---|
| P1 | ✅ reproduced | all eight values, both rooms, the audit and the maintainer |
| P2 | ✅ reproduced | both rooms in their own charts, verified against `r̂₆` directly; solver_b additionally prints the second chart, which lands on the frozen numerator exactly |
| P3 | ✅ reproduced | both rooms |
| P4 | ✅ reproduced | solver_a and the audit under the pass condition's convention, and independently by the maintainer, including the cross term computed at `−1.7e−51` and the pyramid's orbit-null direction at `−9.7e−42`. ⚠️ See § 5.3: solver_b reports different numbers under a different, declared convention, and both are right |
| P5 | ✅ reproduced | `−224/165` and `−42/55` are roots of the sector-scaled spectra in both rooms |
| L0 | ✅ reproduced | six points, seven lines, four spaces of projective dimension ≥ 2 from three groups, both rooms, argument SOUND in both |
| L1 | ✅ reproduced | every row, completeness by elimination in both rooms, cases re-worked independently by the audit |
| L2 | ✅ reproduced | ten orbits; solver_a separates the ties by invariants and isotropy, solver_b by exact `r̂₆` ranges alone, and the audit by both |
| H1 | ✅ reproduced | ten signatures, exact inertia with a numerical cross-check, in both rooms, the audit and the maintainer |
| H2 | ✅ reproduced | ten characteristic polynomials, and the audit confirms them by two independent bases |
| H3 | ✅ reproduced | `v₁` is the only degenerate orbit; the kernel is `span{v₋₃, i·v₋₃}`, identified with the line direction |
| H4 | ✅ stated and used | both rooms |
| G1 | ✅ reproduced | an argument in both rooms and in the audit |
| G2 | ✅ reproduced | ⚠️ as an **argument** by solver_a and, independently, by the audit's own certificate. solver_b reached the same value by a 400-start search, which establishes nothing about the maximum. See § 5.2 |
| G3 | ✅ reproduced | both ends transversely nondegenerate |
| N1 | ✅ reproduced | all six quantities, both rooms, the audit and the maintainer, agreeing to 25 digits; all four residuals nonzero |
| N2 | ✅ reproduced | no interior critical point, by the vanishing linear coefficient, in both rooms and in the audit |
| D1, D2 | recorded | as filed, not adjudicated |

## 5. Adversarial audit record

### 5.1 What the audit did

It received both solver rooms complete and was told that running a solver's script and getting the
solver's answer is not verification. Every number it asserts comes from code it wrote in its room.
It graded the arguments of items 1, 2, 3, 5, 7, 8 and 9 for each solver, adjudicated every
disagreement, and attacked the claims it judged most costly to get wrong.

**No mathematical defect was found in either solver.** Two artifact defects were demonstrated, both
in solver_b's files on disk and neither touching its mathematics.

| ID | What |
|---|---|
| D-B1 | `out/item0.json` holds the values of solver_b's own planted `cg_sign` defect run, not its result |
| D-B2 | `out/item5.json` holds the output of its planted `hess_no4N` run, whose annihilation residual `1/231` is exactly `4N(v₃)`, the signature of the missing term |

⭐ The cause is worth naming, because the mechanism that caused it is a safety mechanism. solver_b's
`run_all.py` runs `collect.py` before its planted defects, and the plants then re-execute the item
scripts and overwrite the same `out/` paths. Its machinery for proving its checks can fail
contaminated the artifacts it left behind. `results.json` and the return are correct, so the claims
stand, but anyone re-running `collect.py` over the landed `out/` would produce wrong numbers.
solver_a, whose defect script runs before its collector, has clean artifacts.

### 5.2 The arguments, graded

| item | solver_a | solver_b |
|---|---|---|
| 1 classification | SOUND | SOUND |
| 2 completeness on the loci | SOUND, by elimination | SOUND, by elimination |
| 3 criticality | SOUND | SOUND |
| 5 the second-variation formula | SOUND | SOUND |
| 7 the kernel | SOUND | SOUND |
| 8 the two ends | SOUND at both | minimum SOUND, maximum **INCOMPLETE** |
| 9 the computed zeros | SOUND | SOUND |

⭐ **The requirement that completeness rest on elimination rather than on a solver was not
pedantry, and the audit demonstrated it.** Applying `sympy.solve` to the exact Gröbner bases returned
no circle points for four of the loci and **no real point at all** for the locus carrying the
degenerate orbit. A room that had trusted a solver would have lost `v₁`, which is the one orbit with
a kernel and therefore the one the census most depends on.

### 5.3 Where the two rooms differ, and why both are right

| Item | The difference | Resolution |
|---|---|---|
| 5b | solver_a reports `diag(64/33, 10/11)` where solver_b reports `diag(8/11, 10/11)` for the same orbit | Both correct. solver_a's tangents are orthogonal to the orbit directions; solver_b's chart derivatives carry a phase component, and the two are related by exact projection factors the audit recomputed (`8/11 = (3/8)·(64/33)`). The worklist fixes neither the basis nor whether to normalize before or after projecting, which is why it asks for the Gram matrix, and each room stated its normalization |
| 1 | the stabilizer column | Different readings: the largest subgroup acting by a character, versus the setwise stabilizer. The class lists coincide |
| 12 | different orbits and directions | The item allows the choice; both verified, exact route difference zero |
| 8 | the maximum | Not a difference of value. solver_a argued it, solver_b searched for it |

### 5.4 What the audit could not check

It says so itself, at length, and the honesty is worth preserving: solver_a's own degree-6 identity
term by term, because the cubics are not stored, so it built an independent certificate instead;
whether any orbit other than the hexagon also attains the maximum, which neither room settled;
several tables it did not recompute; and subgroups beyond order 60, which rest on an argument it
re-derived rather than a computation. It also reports that one flag in its own script is faulty and
that it did not rely on it.

## 6. Containment record

| Guard | Outcome |
|---|---|
| Packet | `worklist.md` byte-identical to the merged file; both briefs cleared the handout gate, whose selftest passes by planting a frozen value and a withheld term |
| Session | three one-shot `--restricted` sessions, four tools each, no MCP server, no skill, no instruction file, no network |
| Denials | solver_a 1, solver_b 2, audit 1, each an automatic refusal by the permission layer |
| Out-of-room reads | solver_b read one of its own background-command outputs, in a folder holding only its own four files. Same class as M8.10 and M8.11, and nothing answer-bearing is reachable by that route |
| Shell shapes outside the sandbox wrapper | solver_b used `time ./py …`; solver_a began one command with a `cd` into its own room and disclosed it. Every path touched is in-room |
| Copy-out | one **disclosed redaction**: solver_b printed the absolute interpreter path into its return, and reported doing so. The maintainer replaced it in the landed copy and records the replacement here |
| Leak sweep | no absolute path and no withheld term in any landed file |

## 7. Provenance

| Item | Value |
|---|---|
| Pre-registration | [#571](https://github.com/openwave-labs/openwave/pull/571), merged `171feca9`, with [#572](https://github.com/openwave-labs/openwave/pull/572) `a94f5f8b` and [#575](https://github.com/openwave-labs/openwave/pull/575) `9f952247` |
| Worklist as run | SHA-256 `c07f9bc64d39ffc99317188e28a91378be18c0fc8902490bd44bed76bba4efdc` |
| Solver brief | SHA-256 `daabf276ad4c55e01c307d9846c9356f029d3d2780c36a364c7cc25d531a9f95` |
| Rooms | solver_a 119 turns in 59 min; solver_b 116 turns in 77 min; audit 64 turns in 22 min |
| Model | `claude-opus-5` in all three rooms |

## 8. What this run does NOT verify

- **Stability of anything.** It computes the reduced Morse index of the leading quartic on the block,
  transverse to the phase-and-rotation orbit. That is not the stability of a branch germ, and a branch
  germ's stability is not particle stability.
- **The three orbits outside the claim.** Orbits whose stabilizer is `C₂`, or `C₃` on its
  three-dimensional fixed space, or trivial, were not classified. Their indices remain unknown, and
  the pre-registration discloses three of them.
- **That the maximum is attained only at the hexagon.** The value is established; uniqueness of the
  attaining orbit is not, and the audit names this as unchecked.
- **`D1` and `D2`.** Recorded as diagnostics. `D1` would be forced if the whole critical set were
  Morse-Bott, which it is not, since `v₁` is degenerate.
- **Anything about sections, radii, or `MODELS.md`.** No cell moves and M8.7's gate is unchanged.
