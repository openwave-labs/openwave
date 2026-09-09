# M8.1.2: THIRD BLIND RUN, the fourth bedrock paper (channel selection on S³/2I)

> Roadmap row: [`../m8_roadmap.md`](../m8_roadmap.md) M8.1.2 (**maintainer-run**).
> Parent template: [`m8_1_task_details.md`](m8_1_task_details.md) (the worked blind-run
> protocol); immediate precedent [`m8_1_1_task_details.md`](m8_1_1_task_details.md);
> firewall and packet-audit precedent
> [`../findings/m8_5a_reproduction_protocol.md`](../findings/m8_5a_reproduction_protocol.md)
> and [`m8_8_task_details.md`](m8_8_task_details.md).
> Status: 🔷 PROPOSED (author-drafted 2026-09-08, revision 14, ready for registration).

## TASK PLANNING (2026-09-08, author-proposed; registration and go pending)

### Scope

Blind independent verification, in the M8.1 sense, of a fourth MIT bedrock paper,
*The Surviving Ray: channel selection for a cubic self-interaction on S³/2I*. The paper
establishes that binary-icosahedral symmetry restricts the four-dimensional family of
equivariant cubic self-maps of spin 3 to two channels, one of them radial, so that modulo
the radial direction the surviving interaction is a **single projective ray**; and,
separately and with no icosahedral input, that the spin-8 cubic channel built from the same
data closes exactly on the time-reversal-invariant rays.

This is the fourth of the arena's math papers and the first added since the three verified
under M8.1 and M8.1.1.

### Why this task exists (the principled trigger, not courtesy)

| Point | Statement |
| --- | --- |
| The rule | M8.1.1 established it: any analytic number that becomes a pre-registered target of the dynamics program gets blind-verified BEFORE the task that would use it locks its targets. |
| Platform relevance | If a future dynamics candidate carries a cubic self-interaction on this arena, this paper constrains what that interaction can select: the leading cubic term is confined to `span{M_0, M_6}`, `M_0` is radial, and the tangential content is one ray. That is target structure of the kind M8.2 § 2 already freezes for the gap ratios. |
| Timing, stated conditionally | **No authorized dynamics route exists in this column at drafting time**: M8.4 closed unresolved, M8.6 closed on the M5 side, and M8.5-C2 was adjudicated FAILED, closing the spectral route and the M8.4 reopening path. This task therefore claims no dependency on a successor that does not yet exist. It is prospective infrastructure: **if** a future authorized dynamics task proposes to freeze the channel-selection result as target structure, this run must complete before that task's first target-bearing lock or run, or the result cannot bear on that verdict. That is M8.2's own T-b2 timing guard applied to a new benchmark, not a new rule. |
| Why maintainer-run BY CONSTRUCTION | The author and the author's agents have read, edited and audited this paper and hold a verification package for it. A self-run check is a reproducer, not an independent recompute. Blindness is the one thing the platform can supply that the author structurally cannot. |
| What this task is NOT | A gate on anything running: the column's IN PROGRESS row is empty. Nor a claim that the platform owes free validation. |

### Ownership and what the author is NOT doing

The author proposes this task, supplies the sources and a candidate instrument, and then
stops. The author does not run the solvers, does not audit them, does not see agent output
before the designer does, and does not participate in the comparison to the claims. If the
maintainers prefer to build the spec sheet from scratch rather than adopt the offered one,
that is strictly better for independence and the author has no objection.

Protocol authorship by an answer-holding author is platform-native under M8.5-A, which
records that protocol authorship does not compromise independence but target-aware
implementation does, provided a maintainer reviews and freezes the protocol and a fresh
context implements behind a firewall. This task is written to that standard.

### Sources of record

| Item | Pin |
| --- | --- |
| Paper (deposit) | Zenodo, [10.5281/zenodo.22681502](https://doi.org/10.5281/zenodo.22681502), deposited 2026-09-09. This is the **version** DOI (v1), not the concept DOI, so it is pinned to this text and cannot drift to a later deposit. That is right for a pin and only for a pin: anywhere the paper is cited to be *read* rather than fetched at a fixed state, the concept DOI is the better target, since a reader should land on the current version. Two DOIs, two jobs; do not read this row as a blanket preference for the version DOI. |
| Working text | `files/framework/files/bedrock/files/surviving-ray.md` in `dmobius3/mode-identity-theory`, commit `a4cb92c25b06f81d4068915c56c88667759cc1a6`, blob SHA-256 `1bace5521f795abe4a3f22cbda06d286f9de5361a826eb74ef7faebc89cfb3d0` |
| Typeset source | `the-surviving-ray.tex` accompanying the deposit, amsart |
| How the two relate, checkably | The deposit was not rendered from the pinned blob; both were generated from the same working file, so neither is downstream of the other. What IS checkable, and what a designer should check rather than take on faith: their **bodies are byte-identical**, from the abstract through the references, 15,134 words each. They differ only in the chrome each format carries, the repo copy having a banner, a glyph title and a generated abstract block, the `.tex` having amsart front matter. Strip both to the body and compare. |

## THE FIREWALL

### Context isolation (not merely a withheld-word list)

The deposit will be **public**. Withholding a title from a prompt does not prevent an agent
with ordinary web access from searching an equation and finding the paper, so a word list
alone does not earn the word "blind."

| Requirement | Statement |
| --- | --- |
| Repository access | Solver and audit contexts have NO access to the OpenWave M8 tree, the `mode-identity-theory` repository, or the deposited paper in any form. |
| Network | No unrestricted web search. Any external reference is a generic representation-theory or computer-algebra source approved in advance by the designer. |
| Manifest | Each agent returns a consulted-material manifest listing every source it read. An empty or unreturned manifest is a protocol failure, recorded as such. |
| Workspace | Agents write into the session scratchpad, never into the repository. The designer copies scripts into `research/scripts/` unmodified at FINISH. |
| Audit ordering | The auditor receives the clean construction packet FIRST and commits its own method and results before the solver's script or output is shown to it. Only then does it receive them, for refutation and comparison. M8.1.1 let the auditor see the solver's work from the start; this ordering removes a convergence pressure, since "own implementation" written while reading someone else's can drift toward it without anything being copied. |

### Where the frozen claims live

The claims table below is an explicit answer key. It states every constant in plain prose.
Once this document is registered it lives in the platform repository and is linked from the
roadmap row, so **the leak path of record is this file, not the handout**.

At go time the designer must state, in the pre-registration, where the frozen claims are
held and what prevents a solver session from reading that path. The handout is clean by
construction; this file is not, and it is the artifact that has to be walled off. A run in
which an agent could have read `research/tasks/m8_1_2_task_details.md` is not blind,
whatever the handout contained.

### Withheld terms: attribution, not vocabulary

The distinction is between pointers to the source and the technical vocabulary the
computation needs.

| Withheld | Permitted |
| --- | --- |
| the author, the model, the repository, "the surviving ray", "McKay", every claimed constant, every theorem number | the technical vocabulary required to pose the problem, including "Majorana" where the handout fixes that normalization |

"Icosahedral" is the interesting case. It cannot be withheld while the task stays posable
**if the group is named**. It can be withheld if the group is instead specified by its
generators, and the designer should prefer that: give the solver the two generating unit
quaternions and never name it. Adopt that route or drop "icosahedral" from the withheld list;
do not list it as withheld while naming the group.

**Be exact about what the generator route buys, because it is easy to overclaim.** It removes
the strongest search term, so an agent cannot reach the source paper by searching the group's
name. It does NOT make B1 and B2 unavailable by lookup: order 120, perfect, and a subgroup of
`SU(2)` identifies the group up to conjugacy through the ADE classification, so any agent that
knows that classification can name it in one step, after which the invariant degrees and the
branching are textbook. The packet's own X0 sanity check hands over exactly the data that
permits this, and it is still worth having.

The remaining work therefore falls to the manifest. **B1 and B2 must be reported as DERIVED or
as RECOGNISED**, with the consulted sources listed either way. A character table that agrees is
a different evidential object from a computation that agrees, and the designer should record
which one the run produced rather than scoring them alike.

### Designer-only material, quarantined from every agent

The author holds a verification package: `verify.py` (88 exact checks), a second
implementation on disjoint primitives, and two mutation harnesses. **None of it may reach a
solver or an audit agent, and none of it is part of adjudication.**

Following M8.1.1, which quarantined the author's own test script and did not use it as the
check, and M8.8, which allowed author artifacts beside the result afterward: **the designer
adjudicates against the frozen claims first and records that verdict. Only then may the
author's package be opened, for provenance-level comparison.**

State the asymmetry when reporting it. Agreement between the author's code and the agents is
**weak** evidence, since a shared convention error survives it. Disagreement is **strong**
evidence that something is wrong somewhere. A three-way agreement is a provenance statement
under the roadmap's own standing rule, not a stronger verification label.

### Offered instrument (adopt, modify, or discard)

| File | What it is |
| --- | --- |
| [`../m8_1_2/conventions-and-worklist.md`](../m8_1_2/conventions-and-worklist.md) | A conventions extract plus a worklist of 22 items (0(a)-(c) and 1-21), carrying the definitions and NONE of the results. It fixes the Clebsch-Gordan phase, the time-reversal convention, the Majorana normalization and the general-spin forms, so a disagreement is traceable to content rather than to an unstated convention. This is the file a solver receives, and the only one. |
| [`../m8_1_2/leakaudit.py`](../m8_1_2/leakaudit.py) | An author-written gate over that handout, demoted to a diagnostic below. **It must never enter the room.** Its forbidden list names nine of the claimed values outright, so beside the handout it is a partial answer key. Run it from outside and copy only the handout across. |

**The leak gate is an offered diagnostic only and cannot certify the handout.** Re-running
an author-written detector proves the handout passes the author's detector; it says nothing
about semantic leakage. M8.5-A records that the dominant risk is semantic and mathematical
circularity rather than an imported answer file, and M8.8 required the packet audit to be
maintainer-side, independent and mechanical for exactly this reason. **Before the room
opens, the designer independently audits the handout against the frozen claims and the
forbidden-input list, manually or with a maintainer-written gate. Passing the author's
checker is not sufficient evidence of non-leakage.**

### Disclosure: the author's prior exercise, and the paper's defect history

The author ran an internal two-unit exercise against the candidate worklist. It is NOT a
verification and is not offered as one: the units were the author's and the author wrote the
instrument.

That exercise, run against the finished paper, found only a defect in the author's own
worklist wording at item 8. **That sentence should not be read as evidence that the paper
came through adversarial review clean.** The paper's defect history is long and is in the
redline record. Three math-audit rounds in the drafting session alone applied seventeen
repairs, among them a genuine proof gap in the spin-8 argument (one-dimensionality of the
target space gives a proportionality constant, not a nonzero one), an invalid inference in
the unequal-degree Jacobian step (degree-zero homogeneity was doing work it cannot do), a
sign error at half-integer spin that the author's own harness was pinning in place because
the harness encoded the same error, and a representation-theory multiplicity error in the
`Sym³V_3` decomposition. Earlier rounds found more, including a false clause in the
selection theorem and a notation collision running through every section.

The honest summary is that this paper has been heavily corrected and the corrections are
recorded. That is a stronger disclosure than a clean-run claim, and it is the reason a blind
recompute is worth the maintainers' time.

## CANDIDATE PRE-REGISTERED CLAIMS

Frozen by the designer at go, not by the author, and grouped to follow the paper's proof
dependency graph rather than a list of interesting numbers. Conventions: the Condon-Shortley
phase and the time-reversal and Majorana conventions of the handout; states in
`V_3 = Sym⁶ℂ²`; `d = dim σ`; `Λ_m = (−1)^(3+m) C(6, 3+m)`. "Blind" means computed by an
agent that never saw the value.

### Group A: the ambient spin-3 results (no group input)

| ID | Claim | Pass condition | Fail condition |
| --- | --- | --- | --- |
| A1 | `Sym³V_3 = V_1 ⊕ 2V_3 ⊕ V_4 ⊕ V_5 ⊕ V_6 ⊕ V_7 ⊕ V_9`, dimension 84, containing NO `V_8` | blind decomposition reproduces the multiset including multiplicity 2 at spin 3, and reports no `V_8` | any multiplicity differs, or `V_8` appears |
| A2 | `dim Hom_SU(2)(Sym²V_3 ⊗ V̄_3, V_J) = 1` at `J = 8`, `= 4` at `J = 3` | both blind weight counts match | either differs |
| A3 | `M_0(u) = −‖u‖²u/√7` | blind evaluation returns a multiple of `u` with constant `−1/√7` | not radial, or a different constant |
| A4 | `M_6(v_m) = c Λ_m² v_m` on the weight basis, `Λ² = (1, 36, 225, 400, 225, 36, 1)`, `c = −C(12,6)⁻¹√(7/13) = −√91/12012` | blind `M_6` diagonal on the weight basis with that row and constant | non-diagonal, different row, or different constant |
| A5 | For `u ≠ 0`, the spin-8 channel `C(u) = 0` **if and only if** `[u]` is time-reversal invariant, equivalently the Majorana constellation is antipodally symmetric as a multiset | an EXACT argument: derive the covariant and characterize its kernel symbolically (a sampled family corroborates but cannot pass this claim) | a zero outside the set, a nonzero inside it, or no exact argument produced |

### Group B: the two group inputs

The paper states that Theorem 5.1 uses exactly two facts about the group. Both are graded
here as headline claims rather than handed to the solver as conventions. **The solver
receives the group as two generating
unit quaternions; whether B1 and B2 are then derived or recognised is recorded, not assumed.**

| ID | Claim | Pass condition | Fail condition |
| --- | --- | --- | --- |
| B1 (invariant-degree filter) | `dim(V_K)^Γ = (1, 0, 0, 0, 0, 0, 1)` for `K = 0..6`, so exactly two ranks carry an invariant **within the closed window `K ≤ 6`**, which is the density's whole reach | blind result from the supplied packet reproduces the row on `K = 0..6`, with DERIVED / RECOGNISED provenance stated | any entry differs on `K = 0..6`. NOT a failure: an agent that computes past the window and reports a further invariant at `K = 10` is CORRECT and agrees with this claim; the window is the claim's scope, not an assertion that no invariant exists above it |
| B2 (multiplicity-free complementary branching) | `V_3` restricted to `Γ` is multiplicity-free with exactly two constituents, of dimensions 3 and 4, complementary in `dim V_3 = 7` | blind branching returns two constituents, each multiplicity one, dimensions 3 and 4, with the same DERIVED / RECOGNISED declaration as B1 | a repeated constituent, a different count, or different dimensions |

### Group C: the selection theorem

| ID | Claim | Pass condition | Fail condition |
| --- | --- | --- | --- |
| C1 | Building the isotypic projector `P` from the group itself: `‖R_0(P)‖² = d²/7`; `‖R_6(P)‖² = d(7−d)/7 = 12/7` in BOTH sectors; `R_K(P) = 0` for `K = 1..5` | blind projector construction reproduces all three, with `12/7` common to both sectors | any rank survives that should vanish, or the sectors differ at `K = 6` |
| C2 | `w_0 = 7`; `w_6/w_0 = (7−d)/(13d) > 0`; `N = C(13,6)·d/(7−d) = 1287` at `d = 3`, `2288` at `d = 4` | all blind values match | any differs |
| C3 (the interaction populates the channel) | The self-interaction is `𝓝(u) = (d/7)‖u‖²u − ((7−d)/√91) M_6(u)` (script `𝓝`, NOT the sector normalization `N` of C2; the handout must keep the two symbols apart); in particular its `M_6` coefficient `t_6` is NONZERO, which is what makes the surviving ray a ray rather than nothing | blind derivation returns both coefficients and establishes `t_6 ≠ 0` | either coefficient differs, or `t_6 ≠ 0` is assumed rather than derived |
| C4a (right-side population) | From the raw group data and the constructed projector, `R_6(P) ≠ 0`, and the induced right coupling `V_3 → V_8` is injective on BOTH sectors. The route: `R_6(P)` read as a degree-12 binary form has twelve SIMPLE roots, and `v ↦ (I_12, f_v)_1` is therefore injective | blind agent establishes nonvanishing AND injectivity, with the simple-root fact derived rather than assumed | `R_6(P) = 0`, a repeated root, or injectivity asserted without argument |
| C4b (left-side population) | `[ρ_6(v_3) ⊗ v_3]_(8,3) = √273/1092 ≠ 0`, so the remaining proportionality constant is nonzero | blind evaluation matches exactly | any other value, or zero |

`(A3 + A4) → B1 → B2 → C1 → C3` is the chain the headline rests on. Nonradiality needs BOTH
halves: A3 fixes that the radial direction is `M_0`, and A4 supplies the nonconstant `Λ²`
that keeps `M_6` off it. Then B1 and B2 are the two group inputs, C1 the weight, and
`t_6 ≠ 0` (C3) is what populates the channel. A designer trimming for cost should trim
elsewhere. "Populated on the quotient" is earned only by C4a and C4b TOGETHER: the left
evaluation alone establishes a proportionality constant, not a populated channel.

### Group D: critical geometry and the corollaries

| ID | Claim | Pass condition | Fail condition |
| --- | --- | --- | --- |
| D1 | `C(12,6)·‖ρ_6(u)‖² = 1, 400, 288, 463` at the unit states `v_3`, `v_0`, `(v_2+v_−2)/√2`, `(v_3+v_−3)/√2`, **and each of those four rays is a critical point** of the ray invariant on `P(V_3)` | all four blind values match AND criticality is established at each | any value differs, or a ray is not shown critical |
| D2 | The full multipole row at `(v_2+v_−2)/√2` is `(1/7, 0, 0, 0, 6/11, 0, 24/77)`, so rank 4 is nonzero and the state is anticoherent of order exactly 3, not merely at least 3 | blind row matches entry by entry | any entry differs |
| D3 | On `u = cos t·v_2 + sin t·v_−3`, `‖ρ_6‖² = −(125/132)sin⁴t + (10/11)sin²t + 3/77`, stationary in the interior at `sin²t = 12/25` with value `9/35` | the blind agent reproduces the WHOLE quartic and its stationary point and value | a different polynomial, even one with the same stationary point, or a different value |
| D4 | On the chart `u = v_3 + z v_0 + v_−3`, `z = x + iy`, the top multipole is `(100\|z\|⁴ − 20x² + 148y² + 463) / (231(\|z\|²+2)²)`, whose critical set on the chart is **exactly five points**: `z = 0` at `463/924`; `z = ±√(23/10)` at `200/903`; `z = ±i√(5/2)` at `24/77` | blind agent reproduces the rational function and establishes exactly those five, by elimination rather than by sampling | a sixth point, a missing one, any value differing, or exhaustiveness not established |
| D5 | Writing the chart's Majorana polynomial in `w = z³` as a quadratic, the two roots multiply to 1 (reciprocal radii, so heights `±h`), and their argument decides the shape: **both arguments equal** at `z = √(23/10)`, so the triangles share azimuths, a trigonal PRISM; **arguments `±π/2`** at `z = i√(5/2)`, so azimuths differ by 60°, the ANTIPRISM, which is the regular octahedron. The D3 interior ray is a pentagonal pyramid, one pole plus a ring of five | blind agent computes the two arguments and the root product at both rays, and the pyramid's constellation, matching. **Naming the shapes is required, not optional**: latitude and azimuth data alone does not discharge this claim, since the prism/antiprism call is the content | a wrong argument pair, a non-reciprocal product, or a mis-split ring |
| D6 | Below level 6 the projected self-interaction is radial at every level `ℓ < 6`, so no projective direction is selected there | blind check reproduces radiality at each level `ℓ = 1..5` exhaustively | a nonradial level below 6 |
| D7 | `Q_3 = 1 + (28/39)r̂_6` and `Q_4 = 1 + (21/52)r̂_6`: the two sectors' reduced quartics are the same increasing affine function of `r̂_6` up to a positive slope, hence share a critical set on `P(V_3)` with the same ordering of values | blind derivation returns both slopes, both positive, and concludes the shared critical set and ordering ALGEBRAICALLY from the affine form | a different slope, a non-positive slope, or an attempt to establish the conclusion by enumerating the critical set |
| D8 | The two sectors' nonlinear shifts differ by `49/156` in the section normalization | blind value matches | any other value |

### Group E: the scope ceiling

| ID | Claim | Pass condition | Fail condition |
| --- | --- | --- | --- |
| E1 | The selection is INTERACTION-RELATIVE, not a universal consequence of the group. The density-type interaction lands in `span{M_0, M_6}`; a second local `Γ`-invariant quartic built from `ψψᵀ` is filtered by the same mechanism into a DIFFERENT plane `span{N_0, N_6}`, and the two planes are distinct. No claim is made that the second plane reduces to a single ray | blind agent constructs both interactions, shows both are filtered, and shows the planes differ | the planes coincide, or the second is claimed to reduce to a ray |

E1 is the paper's own honest ceiling and it is stated in the abstract. Verifying it makes
the headline stronger, not weaker: it establishes that the single surviving ray is a
property of the chosen interaction rather than a secretly universal group result.

### Diagnostics on the run (NOT claims about the paper)

Recorded as diagnostics, never graded as pass or fail against the paper.

| ID | What it detects |
| --- | --- |
| X0 (packet sanity, run BEFORE solving) | **The generator packet is load-bearing the moment B1 and B2 are derived from it.** Before any claim is attempted the agent confirms the packet itself: each supplied quaternion is a unit, `‖q_i‖ = 1`, the group they generate has order exactly 120, AND that group equals its own derived subgroup. Order alone does not pin it: cyclic `C_120` and the binary dihedral group of order 120 are also subgroups of `SU(2)`, and they fail the perfectness test (derived subgroups of order 1 and 30 respectively), so the third condition is doing real work. A wrong or corrupted packet otherwise yields a beautifully verified downstream answer to the wrong question. Not a claim about the paper; a gate on the input, and a protocol failure if it does not hold. |
| X1 | **Affine versus projective.** Worklist item 9 asks for two critical sets on the D4 chart: that of the ray invariant `r̂_6 = ‖ρ_6‖²/‖u‖⁴`, which is what D4 states, and separately that of `‖ρ_6‖²` read on the affine slice, which the paper does not discuss. They differ, because criticality of a homogeneous function on an affine slice is not criticality of its projectivization. A solver that conflates them returns a plausible set that is not a statement about rays. |
| X2 | **Holomorphic square versus density multipole.** `B_J = [u ⊗ u]_J` and `ρ_K = [u ⊗ Θu]_K` agree on `Fix(Θ)` and diverge over `ℂ`. An agent that builds the holomorphic square where the density is wanted returns a SELF-CONSISTENT wrong answer, which is the failure a comparison step is least able to catch. Witness, stated as the discriminating FACT rather than as a signed value: at `u = v_3` the holomorphic square vanishes identically, `B_2 = 0`, while the density multipole does not, `\|ρ_2\| = 5√21/42`, `‖ρ_2‖² = 25/84`. The sign is convention-dependent and the same magnitude appears elsewhere as `B_2` at the hexagonal state, so a diagnostic that hands over a signed value invites pattern-matching; give the magnitude and the vanishing, or state the sign in the handout's own convention explicitly. This misreading has cost real revisions in the author's own drafting, so its yield is known rather than speculated. |

### Feasibility triage, stated honestly

Most claims are finite evaluations. Three are not, and a designer budgeting from this
document should know which:

| Claim | Why it is not a spot check |
| --- | --- |
| A5 | A global "if and only if" over all nonzero `u`. Requires an exact kernel characterization; sampling corroborates but cannot pass it. The author's dry run found both units gave the forward direction cleanly and sampled the converse, because the handout was missing the classical Jacobian criterion for binary forms; that has been added to the conventions block as a supplied input, alongside the Clebsch-Gordan phase. It leaks nothing graded: the solver must still discover that the covariant factors through the first transvectant, which is the claim's actual content. |
| D4 | "Exactly five" is a global critical-point claim on a two-real-dimensional chart. Requires elimination, not evaluation at candidate points. |
| D7 | A brute-force comparison of critical sets would need a classification the paper does not have. The affine-form proof avoids enumeration entirely: two positive slopes give identical critical loci and identical ordering algebraically, on every pair of rays, whether or not the critical set is finite or ever enumerated. That is why the pass condition requires the algebraic route and fails an enumeration. |

The general-spin rank-zero constant `ε_j/√(2j+1)`, valid at integer and half-integer `j`
alike, is a general formula and is deliberately NOT in the claims table: testing several `j`
does not verify it. If the designer wants it, require a symbolic derivation and grade that,
or narrow it to the finitely many levels actually tested and say so.

### Instrument coverage

The worklist now reaches every claim: A5 through item 19 (the exact zero set, with the
scale-free
half separated from the normalized `L_8` evaluation that C4b grades) and E1 through item 21,
which
requires the solver to derive the `psi psi^T` to `B_J` bridge rather than being handed it. An
earlier revision did not reach either, and the designer should still confirm coverage at go
rather
than taking this paragraph for it.

### To be fixed at go time (BEFORE numerics)

| Item | Note |
| --- | --- |
| Deposit pin | done: the version DOI is pinned in Sources of record |
| Claims frozen | the designer freezes the tables above, edited freely, before any solver launches |
| Answer-key containment | state where the frozen claims are held and what prevents a solver reading that path |
| Instrument decision | adopt the offered worklist after confirming coverage, modify it, or build a fresh spec sheet |
| Handout audit | maintainer-side independent audit of the handout, not the author's gate alone |
| Group presentation | decide whether the group is handed over by generators (preferred) or by name |
| Citations sync | add the paper to [`../../theory/_CITATIONS.md`](../../theory/_CITATIONS.md) |

### Definition of done (skeleton, finalized at go)

| # | Item |
| --- | --- |
| 1 | Solver runs with exact values, scripts + JSON in the repo (`m8_1_2_` prefixes) |
| 2 | Adversarial audit with its own method, per-claim verdicts |
| 3 | Designer comparison against the frozen claims, all numbers stated, including any that landed elsewhere; diagnostics X0, X1 and X2 recorded separately from the claim verdicts, X0 as a precondition rather than a result |
| 4 | Method note `findings/m8_1_2_method_note.md` (equations first, eq-to-code map, audit record, consulted-material manifests) |
| 5 | Author package opened for provenance comparison ONLY after the verdict is recorded, with the agreement/disagreement asymmetry stated |
| 6 | Doc sync: canonical + briefing + roadmap row; MODELS.md only if a cell actually moves |
| 7 | Doc checker exit 0; TASK REVIEW presented |

### Instrument qualification, and what is frozen

This packet was exercised before proposal. Three questions were repaired as a result. No
expected value changed and no result-bearing statement was added:

| change | why |
| --- | --- |
| Added the classical Jacobian criterion for binary forms to the conventions block | It had fallen out of the handout as collateral when a result-bearing lemma was removed, and the conventions block should have carried it alongside the Clebsch-Gordan phase. **Read the residual before relying on it:** in the qualification run both units reached the correct characterization for A5 and both established the forward direction by a route this paper does not use (Sym^3 V_3 contains no V_8), needing no Jacobian input at all; neither reached the factorization through the first transvectant, which is what the converse turns on, and both sampled instead. The criterion therefore repairs a gap the run did not demonstrate was the binding one, and may not change the outcome. It was not repaired further, because finding that factorization is A5's content and supplying it would hand over the claim. |
| Item 20 now requires the simple-root fact, and requires a numerical-rank route to be declared as such | C4a's exact injectivity argument turns on simplicity, and the question asked only for root positions. |
| Item 15 now requires the shapes to be named | D5 grades the prism-versus-antiprism discrimination, and the question left it optional. |

One residual the designer should weigh rather than discover: the Jacobian criterion sits in a
conventions block that already carries the Majorana dictionary, so together they point a reader
toward binary forms of equal degree, which is the route A5 travels. The alternative is to grade
recall of a classical lemma rather than verification of this paper, and finding that the
covariant factors through the first transvectant remains the whole job.

**What is frozen.** From the maintainer launch the packet stops moving on the author's side:
no edit because a computed result disagrees with a frozen claim. A wrong answer is a result.
This binds the author, not the maintainers. If the designer wants to change the packet, that
is a design decision and theirs to make.

### Scheduling

Maintainer-run at maintainer pace. Not on the author's critical path and not blocking the
column's live front, which is empty. The one ordering constraint is the conditional timing
guard above.

---

## GO-TIME PRE-REGISTRATION

*To be written by the designer at go, freezing the claims tables and the firewall before any
numerics run. Nothing above this line is a substitute: the author drafted it and the author
has read the paper.*
