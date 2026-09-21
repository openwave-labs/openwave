# S1: THE REDUCED MORSE CENSUS OF THE LEVEL-6 QUARTIC, and the classification of its critical orbits with a fixed locus of projective dimension at most one

Pre-registration, for review. It governs nothing until the maintainer's go.

## TASK PLANNING

### The question

M8.11 established, as an audited argument, that local branch germs exist at six level-6 critical rays in each of sectors `3′` and `4`, and it made no stability claim. The first question after it is finite algebra on spin 3:

- **which critical orbits of the level-6 reduced quartic `r̂₆` have a projective stabilizer whose fixed locus has projective dimension at most one**, that is, which sit at an isolated fixed point or on a fixed line of a rotation group; and
- **at each of those, what is the signature of `r̂₆`'s second variation transverse to the symmetry orbit.**

One signature gives the reduced Morse index for both signs of `g`: `n₋` for `g > 0` and `n₊` for `g < 0`.

Why the block carries it: sectors `3′` and `4` first occur at level 6, so the block is the bottom of the spectrum on both bundles (the fourth bedrock paper, § 6.1). At small amplitude the negative directions of the stationary equation's constrained second variation can therefore come, at leading order, only from the block. That is the motivation for the task. It is not claimed here, and what the count means for orbital or spectral stability is not this task's question.

### Standing: the #512 ruling

- **Run-before-write** is met on its plain reading. The last pre-registration, M8.11's, governed a run that has happened (#564) and closed with its package (#566, merged 2026-09-19). The maintainer confirms when a pre-registration is filed.
- **One program, one pre-registration**, within the § 12.2 budget, and nothing further lands on it until it has run.
- **A claim that frozen text is defective** is reproduced by the maintainer with independent code before ratification.

M8.12 is allocated here, as the next free ID in creation order, under `ROADMAP_STANDARDS.md` § 6: whoever creates the row allocates it, and the reviewer verifies rather than assigns. If a row taking M8.12 lands first, whoever merges second renumbers this one, which edits this document and so moves its hash.

### What this task is not

It computes the Morse index of the leading reduced quartic on the block, transverse to the phase-and-rotation orbit. It is not the stability of a branch germ, which is not particle stability. No MODELS.md cell moves and M8.7's gate is unchanged. It does not classify the whole critical set: orbits whose stabilizer is C2, or C3 acting on its three-dimensional fixed space, or trivial, lie outside the claim, and three such orbits are known numerically and disclosed below.

**Short forms.** The BACKLOG row, the PR body and any comment say "the second variation of the level-6 reduced quartic", or "reduced Morse index". Never "stable" or "unstable" alone, and never "the stability of the branches".

### Ownership and run format

The M8.1.x mechanics. The author freezes the values below from a fresh exact derivation; two maintainer-run blind agents recompute them from the worklist; an adversarial audit follows, which also grades the arguments L0, L1, L2, G1 and G2; then adjudication.

### Sources of record

| Source | What it supplies |
|---|---|
| The fourth bedrock paper, § 5.8 | The four inert rays and their values, the pyramid and prism loci with their values, and the D₃ chart formula |
| The fourth bedrock paper, § 7.2 | Every weight state is critical, and the seven of them realize four values of `924·r̂₆`: 1, 36, 225 and 400 |
| M8.1.2 (#540) | D1, D3, D4 and D7, verified blind |
| M8.11 (#554, #564, #566) | `L_T = (w₆/4)·Hess_T r̂₆` and the in-locus second variations at the pyramid and the prism |
| This task | L0, L1, L2, H, G1, G2 |

## THE LAW, PINNED

The pinned pointwise law enters only through D7: in each sector the reduced quartic on the block is

`Q_σ = 1 + w₆(σ)·r̂₆`, with `w₆(3′) = 28/39` and `w₆(4) = 21/52`, both positive.

So every claim below is sector-independent, and in a sector the second variation is `w₆(σ)` times the one computed here. S1 re-derives nothing about sections, and uses nothing from `m8_5c/design_inputs/`.

## SETTING AND DERIVATION

### The quartic

`V₃ = ℂ⁷` with basis `v₃, …, v₋₃`, Hermitian product `⟨·,·⟩`, real inner product `Re⟨·,·⟩`. With `(Θu)_m = (−1)^{3−m}·conj(u₋ₘ)` and the Clebsch-Gordan coupling to spin 6,

`ρ₆(u) = [u ⊗ Θu]₆`,  `r̂₆(u) = ‖ρ₆(u)‖² / ‖u‖⁴`.

### The fixed loci

For a closed subgroup `H` of `SO(3)` and a one-dimensional character `χ`, the set `V₃^{(H,χ)} = {u : D³(h)u = χ(h)u}` is the fixed space of the graph subgroup `{(χ(h)⁻¹, h)}` of `U(1) × SO(3)`, which acts linearly. `r̂₆` is invariant under it, so Palais' principle applies: a critical point of `r̂₆` restricted to such a fixed space is critical on the whole sphere.

### The second variation

For a unit critical `u`, `H_u(e, e) = d²/ds² r̂₆(u cos s + e sin s)` at `s = 0`, a self-adjoint form on `T_u = {e : Re⟨u, e⟩ = 0}`, of 13 real dimensions. Because `r̂₆` has degree 0, the full gradient vanishes at `u`, and with `N(x) = ‖ρ₆‖²` and `‖u‖ = 1`,

`Hess(N·D^{−2})(u) = Hess N(u) − 4N(u)·I − 8N(u)·u uᵀ`,

whose last term drops on `T_u`. So `H_u = Hess N(u) − 4N(u)·I` there. The orbit directions `O_u = span{iu, −iJ_x u, −iJ_y u, −iJ_z u}` are null for it, and the transverse space `N_u = T_u ⊖ O_u` has dimension 10 at the weight states and 9 elsewhere.

### Method

- The signature comes from **exact inertia**, by symmetric elimination with a 2×2 block whenever a pivot vanishes. By Sylvester's law it does not depend on the basis.
- The **exact characteristic polynomial** of the transverse operator is computed in an exact orthonormal basis.
- A **30-digit numerical diagonalization** cross-checks both.
- On the lines, completeness comes from an elimination argument: each dihedral restriction is even in `x` and in `y`, so `f_x = x·A(X, Y)` and `f_y = y·B(X, Y)` with `X = x²`, `Y = y²`, leaving four exhaustive cases, of which the off-axis one is a resultant with no positive solution. A solver corroborates it. The Poincaré-Hopf index sum is a consistency check only, since a missed maximum and saddle would cancel in it.

## THE FIREWALL

The rooms have no access to: this document; the S0 memo, its lemma note and the `M8_S/` aids and logs; `M8_DYNAMICS/`; and M8.11's records, whose in-locus values are graded again here. The worklist works in the fibre model only, defines every quantity by formula, and states D7 as an input. It gives no values, no global-extremum statement, no list of loci and no isotropy answers: the agents derive the classification themselves.

**Network posture, which this task needs stated.** M8.11's object existed in no literature, so blindness was automatic. This one does not: S0 established that `r̂₆` is the standard spin-3 condensate function at one point of the published four-coupling family, and that Kawaguchi and Ueda's Table II carries closed forms for two of the orbits below. An agent with search could locate rather than derive the loci, their critical sets and the union of them, which are worklist items 1 to 4, and the two ends, item 8. The transverse signatures, spectra and kernel, items 5 to 7, are new: S0's search found no source computing them.

The run is therefore offline, as M8.11's clean room was. If the maintainer runs it with search instead, items 1 to 4 and item 8 are scored "reproduced or located" rather than derived blind, and the worklist's standing request for anything looked up rather than derived is the discriminator. Either posture is acceptable; leaving it unstated is not, because afterwards nobody can tell which held.

## DISCLOSURE

The author's exploratory work before this task is inventoried file by file in `DISCLOSURE_INVENTORY.md`, generated by a sweep that fails if a swept record is unclassified: 57 records. That inventory lands with the author's package, and earlier on request. The load-bearing items:

- ascent and descent runs (2026-08-31 and 09-01) that reached the hexagon and the coherent orbit;
- the numerical Hessian of `Q` at the hexagon in both sectors, with its full negative spectrum: 9 negative, 5 null, 0 positive of 14 ambient directions;
- the hexagon branch by Newton continuation to `a ≈ 8.7`;
- a random census (2026-09-04, 09-05) that found twelve critical orbits, and a seeded census the same day that recorded thirteen, with block-operator spectra, noting that thirteen is a lower bound;
- a C3-plane search whose candidates are unverified, from a finite-difference gradient at its noise floor;
- the S0 memo, its lemma note and their aids (2026-09-19);
- this task's own derivation, in three steps with their logs.

**A dated correction.** M8.11's disclosure named "exact spectra of the block multiplication operator at six critical orbits". Those are the six integral spectra of the seeded census. It did not name either census, nor their seven other spectra. That is stated here rather than by editing M8.11's frozen text.

**What was already visible, row by row.** Part of the census below was on the record before the inertia computation ran.

| orbit | already visible | new here |
|---|---|---|
| coherent `v₃` | a global minimum, so `n₋ = 0` | the nullity and the spectrum |
| hexagon | a global maximum; M8.11's in-locus `(−, −)`; the exploratory count 9 negative, 5 null of 14 | the exact spectrum, and that the count is exact |
| octahedron | M8.11's in-locus `(−, +)` | the full signature and spectrum |
| prism | M8.11's `920/473` and `8/11` | the other seven directions |
| pyramid | M8.11's `−104/55`, and the null orbit direction | the other eight directions |
| zonal `v₀` | a saddle, from two in-line types | the exact signature |
| C3 ray, D2 ray | in-line types only | the full signatures |
| `v₂`, `v₁` | nothing beyond block-operator spectra | everything, including `v₁`'s degeneracy |

**Three orbits outside the claim.** The censuses found three critical orbits whose stabilizer is C2 or trivial: at `924·r̂₆ ≈ 215.775792502`, `238.016528926` and `238.356854743`. Their saved states are in the inventory. They are not claimed here, and their indices are unknown.

## CANDIDATE PRE-REGISTERED CLAIMS

### Group P: parents, reproduced rather than new

| ID | Claim | Standing | Pass condition | Fail condition |
|---|---|---|---|---|
| P1 | `924·r̂₆` is 1, 36, 225 and 400 at the weight states, 288 at the octahedron and 463 at the hexagon, `1188/5` at the pyramid, recorded as `9/35`, and `8800/43` at the prism, recorded as `200/903` | the weight states from paper § 7.2; the rest from paper § 5.8 and M8.1.2 D1, D3, D4 | all eight reproduce exactly | any differs |
| P2 | On the D₃ locus, the chart restriction is `(100\|z\|⁴ − 20x² + 148y² + 463)/(231(\|z\|² + 2)²)` | paper § 5.8 | reproduced exactly in the paper's chart, `b₁ = v₃ + v₋₃` and `b₂ = v₀` unnormalized, or an equivalent restriction in the chart the room states, carried onto it by the room's own stated change of chart | differs |
| P3 | On the C₅ locus, the restriction is `−(125/132)s² + (10/11)s + 3/77` in `s = \|c\|²`, up to the flip | paper § 5.8 | reproduced exactly | differs |
| P4 | M8.11's in-locus second variations of `r̂₆`: `−104/55` along the C₅-line tangent at the pyramid, `920/473` and `8/11` along the prism's chart tangents, no cross term, and the null orbit direction at the pyramid | M8.11 T4 | all reproduce from the full transverse form, with the tangents orthogonal to the orbit directions | any differs |
| P5 | The sector bridge `L_T = (w₆/4)·H` gives M8.11's `−56/165` in `3′` and `−21/110` in `4` at the pyramid | M8.11 T4 with D7 | the rooms report `w₆·H`, which is `4·L_T`, so `−224/165` and `−42/55` reproduce | either differs |

### Group L: the classification

| ID | Claim | Standing | Pass condition | Fail condition |
|---|---|---|---|---|
| L0 | Up to rotation, the character-fixed loci in `ℙ(V₃)` of projective dimension at most one are exactly six points and seven lines, as tabulated below; every other nonzero character-fixed space of a finite subgroup has projective dimension at least 2, and there are four such, from three groups | new; scored as an argument, with the enumeration computed | the same six classes and seven classes, derived by characters and by explicit joint eigenspaces, with each locus carried to a canonical representative and the classes shown distinct | a different list, a missing class, or distinctness argued only numerically |
| L1 | The complete critical set of `r̂₆` on each of the seven lines is as tabulated: five quadratics with their vertices, and two dihedral charts with five points each plus the point at infinity | new | every row reproduces exactly, a dihedral row either in the paper's chart or as an equivalent restriction the room carries onto it by its own stated change of chart, and completeness rests on an elimination argument rather than on a solver | any row differs, or completeness rests on a solver |
| L2 | United modulo rotation, those loci carry exactly ten critical orbits, with the values of the census table; the two value ties are separated, one by invariants and isotropy and one by isotropy alone | new; scored as an argument | the same ten orbits with the same values, and both ties separated | a different count, or a tie left unseparated |

### Group H: the census

| ID | Claim | Standing | Pass condition | Fail condition |
|---|---|---|---|---|
| H1 | At each of the ten orbits, the signature `(n₋, n₀, n₊)` of `H_u` on `N_u` is as tabulated, with `dim N_u` 10 at the weight states and 9 elsewhere | new | every row reproduces, by exact inertia, with a numerical cross-check agreeing | any row differs |
| H2 | The characteristic polynomial of the transverse operator at each orbit is as tabulated, which the table prints up to a positive constant | new | every polynomial reproduces exactly after monic normalization | any differs |
| H3 | `v₁` is the only degenerate orbit, and its two-dimensional kernel is exactly the C₄ line direction `span{v₋₃, i·v₋₃}` | new | the kernel reproduces and is identified with that direction | a different kernel, or another orbit degenerate |
| H4 | The signature is sector-independent, and in a sector the form is `w₆(σ)` times it | D7 | stated and used | a sector-dependent signature |

### Group G: the two ends, scored as arguments

| ID | Claim | Standing | Pass condition | Fail condition |
|---|---|---|---|---|
| G1 | The coherent orbit is the unique minimum of `r̂₆` on the unit sphere, with value `1/924` | new; argument | `r̂₆ = Σ_F c_F‖[u⊗u]_F‖²` with `c_F = 13·{3 3 6; 3 3 F}`, every `c_F` with `F < 6` above `c₆ = 1/924`, and equality only for coherent `u` | the argument fails, or a smaller value is found |
| G2 | The hexagon orbit is the unique maximum, with value `463/924` | new; argument | the invariant form `r̂₆ = −5/231 − \|f\|²/22 + (7/11)\|a₀₀\|² + TrN̄²/198`, the bounds `\|a₀₀\|² ≤ 1/7` and `TrN̄² ≤ 171/2` with its equality case, and the hexagon as the only orbit meeting all three | the argument fails, or a larger value is found |
| G3 | With H1, both ends are transversely nondegenerate: `(0, 0, 10)` at the coherent orbit and `(9, 0, 0)` at the hexagon | new; derived | both signatures reproduce | either has a kernel |

### Negative controls

| ID | Claim | Standing | Pass condition | Fail condition |
|---|---|---|---|---|
| N1 | At the non-critical point the worklist supplies, `u = (v₃ + v₁ − v₋₂)/√3`, `924·r̂₆ = 204`, the tangential gradient of `r̂₆` on the unit sphere has norm `2√1002/297`, and the orbit directions are not annihilated by `M_u = Hess N(u) − 4N(u)·I`: taking the four generators of § 1.5 as written, un-normalized, `‖M_u d‖` is `2√1002/297` at `i·u`, `2√(312√15 + 4170)/297` at `−i·J_x u`, `2√(4170 − 312√15)/297` at `−i·J_y u` and `2√3684/297` at `−i·J_z u`, the four squares summing to `17368/29403`. The first equals the gradient norm because `N` is phase-invariant, so `M_u(i·u) = i·(∇N(u) − 4N(u)·u)`. On orbit directions `M_u` agrees with the second variation `H_u` of § 1.5, because the two differ on `T_u` by `−4u(∇N(u)·d)` and `∇N(u)` is orthogonal to the orbit | new | all six reported quantities reproduce, of which five are independent, and all four residuals are nonzero | any of the six differs. The control itself ceases to discriminate only if the tangential gradient and all four residuals vanish together, which is what happens at a critical point and what `s1_control.py` checks as its arm |
| N2 | The C₄ line `span{v₃, v₋₁}` has no interior critical point, because its restriction's linear coefficient is exactly zero | new | reproduced exactly | a nonzero coefficient, or an interior point |

### Diagnostics, not claims

| ID | What it records |
|---|---|
| D1 | The parity sum over the weight-state orbits, `2 + 2 + 2 + 1 = 7 = χ(ℂℙ⁶)`, which would be forced if the whole critical set were Morse-Bott. It is not established, since `v₁` is degenerate, so this is recorded and not adjudicated |
| D2 | Among the ten classified orbits, the `g > 0` indices in increasing critical value are 0, 2, 3, 4, 5, 5, 5, 6, 8, 9. Indices 1 and 7 do not occur within the classified set. The deferred orbits' indices are unknown and may break this ordering |

## FROZEN VALUES

### The census

| orbit | representative | `924·r̂₆` | `dim N_u` | `(n₋, n₀, n₊)` | index, `g > 0` | index, `g < 0` |
|---|---|---|---|---|---|---|
| coherent `v₃` | `v₃` | 1 | 10 | (0, 0, 10) | 0 | 10 |
| `v₂` | `v₂` | 36 | 10 | (2, 0, 8) | 2 | 8 |
| prism | `v₃ + √(23/10)·v₀ + v₋₃` | `8800/43` | 9 | (3, 0, 6) | 3 | 6 |
| `v₁` | `v₁` | 225 | 10 | (4, 2, 4) | 4 | 4 |
| D2 ray | `v₂ + i√(6/5)·v₀ + v₋₂` | 225 | 9 | (5, 0, 4) | 5 | 4 |
| C3 ray | `v₂ + 2·v₋₁` | `1188/5` | 9 | (5, 0, 4) | 5 | 4 |
| pyramid | `√12·v₃ + √13·v₋₂` | `1188/5` | 9 | (5, 0, 4) | 5 | 4 |
| octahedron | `v₂ + v₋₂` | 288 | 9 | (6, 0, 3) | 6 | 3 |
| zonal `v₀` | `v₀` | 400 | 10 | (8, 0, 2) | 8 | 2 |
| hexagon | `v₃ + v₋₃` | 463 | 9 | (9, 0, 0) | 9 | 0 |

### The seven lines

| line | restriction | critical set |
|---|---|---|
| C₃ `{v₂, v₋₁}` | `−(15/44)s² + (3/22)s + 75/308` | endpoints; interior circle at `s* = 1/5` |
| C₄ `{v₃, v₋₁}` | `−(8/33)s² + 75/308` | endpoints only; the vertex is the endpoint `s = 0` |
| C₄ `{v₂, v₋₂}` | `−(12/11)s² + (12/11)s + 3/77` | endpoints; interior circle at `s* = 1/2` |
| C₅ `{v₃, v₋₂}` | `−(125/132)s² + (10/11)s + 3/77` | endpoints; interior circle at `s* = 12/25` |
| C₆ `{v₃, v₋₃}` | `−2s² + 2s + 1/924` | endpoints; interior circle at `s* = 1/2` |
| D₃ `{v₃ + v₋₃, v₀}` | `(100\|z\|⁴ − 20x² + 148y² + 463)/(231(\|z\|² + 2)²)` | `z = 0`; `z = ±√230/10`; `z = ±i√10/2`; `z = ∞` |
| D₂ `{v₂ + v₋₂, v₀}` | `4(25\|z\|⁴ + 142x² + 30y² + 72)/(231(\|z\|² + 2)²)` | `z = 0`; `z = ±√30/3`; `z = ±i√30/5`; `z = ∞` |

### The transverse spectra

| orbit | characteristic polynomial, up to a positive constant |
|---|---|
| coherent `v₃` | `(λ − 4)²(11λ − 3)²(11λ − 2)²(33λ − 65)²(33λ − 32)²` |
| `v₂` | `(3λ − 4)²(11λ − 24)²(11λ − 20)²(11λ − 12)²(33λ + 8)²` |
| `v₁` | `λ²(3λ + 5)²(11λ − 3)²(363λ² − 374λ − 600)²` |
| zonal `v₀` | `(11λ − 8)²(11λ + 12)²(11λ + 20)²(33λ + 40)²(33λ + 100)²` |
| octahedron | `(11λ + 8)³(11λ + 24)³(33λ − 40)³` |
| hexagon | `(λ + 4)(11λ + 10)²(11λ + 15)(11λ + 23)(33λ + 64)²(33λ + 67)²` |
| pyramid | `(55λ + 24)²(55λ + 104)(165λ − 4)²(1815λ² − 1012λ − 3360)²` |
| prism | `(11λ − 8)(473λ − 920)(473λ + 696)(1419λ − 560)²(671187λ² + 438944λ − 339200)²` |
| C3 ray | `(55λ + 24)(16471125λ⁴ + 9583200λ³ − 45776720λ² − 23015168λ + 998400)²` |
| D2 ray | `(11λ − 10)(11λ + 7)(33λ − 64)(31944λ³ + 46948λ² − 13530λ − 7125)²` |

### The classification

Six points: `v₃`, `v₂`, `v₁`, `v₀`, the octahedron `v₂ + v₋₂`, the hexagon `v₃ + v₋₃`.

Seven lines: C₃ on `{v₂, v₋₁}`; C₄ on `{v₃, v₋₁}`; C₄ on `{v₂, v₋₂}`; C₅ on `{v₃, v₋₂}`; C₆ on `{v₃, v₋₃}`; D₃ on `{v₃ + v₋₃, v₀}`; D₂ on `{v₂ + v₋₂, v₀}`.

Spaces of projective dimension at least 2, from three groups: `C₁` (7), `C₂` (4 and 3), `C₃` (3).

## FEASIBILITY, AND THE AUTHOR'S DERIVATION

The derivation ran in three steps before this document, each reviewed by two author-side units, which is review and not verification:

| step | what it settled | script | log |
|---|---|---|---|
| L0 | the classification, by two routes over 26 groups | `s1_L0.py` | 128 checks |
| L1 | the critical set on the seven lines, and the union | `s1_L1.py` | 43 checks |
| H | the ten signatures, spectra and the `v₁` kernel | `s1_H.py` | 29 checks |
| N1 | the control point, measured with the step-3 machinery | `s1_control.py` | 7 checks, on top of re-running step 3 |

**Correction, 2026-09-20, after filing.** The N1 row above records `s1_control.py` as filed, at 7 checks. The edits agreed in #571 grew N1 from three scored quantities to six, so the control was extended to measure each of the four orbit residuals against its frozen expression, the phase identity, the orthogonality of `∇N` to the orbit, and both sum rules. It now runs 16 checks. No frozen value moves, the worklist is untouched, and the filed row is left as filed rather than rewritten.

Four defects were caught by those steps' own gates and are recorded in the step notes: an `arccos` precision loss that made every half-turn read as the identity; a fixing-group predicate that accepted a zero matrix as a scalar; a tangent convention that projected out only the radial direction; and a kernel gate that pointed at the wrong end of a line.

## TO BE FIXED AT GO

- The conventions of the worklist, with the Clebsch-Gordan and time-reversal choices displayed.
- The exactness rule: a value is accepted as exact when an exact route returns it and a 30-digit route agrees; a value that a computation did not reach is reported as missing, never as zero.
- The claims tables above, with no value edited.

## DEFINITION OF DONE

Two blind agents return the classification, the critical sets, the ten signatures and the spectra. The audit grades L0, L1, L2, G1 and G2 as arguments, and checks the negative controls. Adjudication records each claim as reproduced, defective or unresolved. No radius, no stability claim, no MODELS.md cell, and no change to M8.7's gate.
