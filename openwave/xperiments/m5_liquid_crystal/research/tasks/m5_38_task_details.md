# M5.38: The author's 10 November suggestions: the inertia gate, two clock checks, the moment bound

> Opened 2026-11-11 from the author's post of 2026-11-10 17:58 UTC on discussion #186 ([discussioncomment-19525348](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19525348)): six suggestions for the next rungs of this column, ranked by the author by what each would unlock. The post ships no release and no script. This task takes the three suggestions that are desk reads on stored fields and stored instruments (the inertia gate of item 5, the two clock checks of item 6, the moment bound of item 4) and records the other three (a Hamiltonian integrator, a declared second certified action, a converged gap pencil) as rung candidates. The decode of the post is in [`m5_32_convo.md`](m5_32_convo.md).

## TASK PLANNING (2026-11-11; staged, the run waits for the user's go)

**Scope.** A desk run on stored fields and stored instruments, with no release, no download and no code of the author's. R0: the six items routed (§ 1). R1: the isospin-inertia integrand read on the stored census baryon states, the gate the author places before N4 (§ 2). R2: the two clock checks on the electron-ID instrument, unmodified (§ 3). R3: the arithmetic of the moment bound (§ 4). R4: a fresh-agent audit and the reply draft (§ 5). Out of scope: items 1, 2 and 3 of the post (each is a rung of [M5.32](m5_32_task_details.md) or a platform build, picked by the user, § 1); the two gates the post lists for the πN test and the frozen-card offer ([M5.37](m5_37_task_details.md) § 5, held); any relaxation or S-run; any icon or pointer in [`MODELS.md`](../../../../../MODELS.md).

**Definition of done.** § 1 written against the record; three scripts with their JSONs (`m5_38_inertia_gate.py`, `m5_38_clock_checks.py`, `m5_38_post_checks.py`, under `scripts/` and `data/`), every check stating the input that would make it fail; the known-answer controls of § 2 and § 3 green before any read is labeled (cap 3 tries, then stop and report the failing control); one panel plot embedded in the findings; a fresh-agent audit on a different model with its own scripts at 0 REFUTED, or the refutations folded in; the reply draft in the terminal; `check-docs --changed`, `check_roadmaps.py` and `check_no_local.py` clean.

**Gating.** The user's go. Nothing is downloaded and nothing from outside the repo is executed.

**Parallel shape.** Three scripts of seconds each on one core, a documented choice: the reads are array contractions over at most 48³ cells and about a hundred analytic seedings of the electron-ID instrument, so there is no compute to split and no GPU use. One audit agent runs in the background beside the write-up.

**Duration estimate.** Tens of minutes from the go to the review: under a minute of compute, the tool calls of three small scripts and one write-up, and one audit in parallel.

**Consumers.**

| Sub-result | Read by | The instrument check that reader needs |
| --- | --- | --- |
| R1, the count of unbroken generators on the census states | any later plan that stages a rotor or moment read on a baryon state of this column (the N4 reading of [M5.37](m5_37_task_details.md) § 2) | a planted compact record on which the reader returns FLAT, and the closed-form vacuum density on the exterior |
| R2, the centroid and the multipoles over a clock cycle | the electron-ID reading of the magnetic-moment row ([Q66](../m5_question_tracker.md#q66-detail)) and the reply | a planted orbiting charge recovered at its planted radius, and the same read at three resolutions |
| R3, the moment bound | the reply; ask 1 of M5.37 § 4 | routine (arithmetic of printed numbers) |

**Blindspot pass.** Read against the failure families this series has met before. (i) Unvalidated instrument: both new readers get a control on which they return the other answer (a compact record for R1, a planted orbit for R2) before any label. (ii) Artifact read as result: the R1 labels are read on two radial windows and two resolutions, the R2 labels at three resolutions; the active mask of the electron-ID instrument excludes the core, where almost all of the charge sits (0.57 of 12.54 at n 24, 0.13 of 12.56 at n 48 in the M5.34 record), so every R2 number is read on the full interior and the masked number is listed beside it, never alone. (iii) Unfalsifiable check: at δ = 0.3 the vacuum tensor of the census is biaxial, so three broken generators on the pinned exterior follow from the boundary condition; the R1 read is falsifiable only through the closed-form density (a wrong exterior moves it) and through the inner region, and is worded as a measurement of the precondition, not as a discovery. (iv) Resolution mistaken for accuracy: the author's bar for check (b) of § 3 is 2.6 × 10⁻⁹ of an interaction energy; a cubic lattice has its own fourfold anisotropy at order h², so a multipole fraction is quoted with the floor measured on the same grid beside it. (v) Wrong diagnostic for the regime: the stored census fields are 3×3 and pinned, the electron-ID electron is an analytic seed with a pinned clock; neither is a relaxed spinning 4×4 state, and the task says so wherever "spinning core" is read. Unknowns by quadrant: known unknowns, the five author-gated definitions of § 5; unknown known, the user's pick among items 1 to 3; machine-checkable, R1 to R3.

### 1. The six items and their routing

| # | The author's suggestion | What the record holds | Routing |
| --- | --- | --- | --- |
| 1 | A Hamiltonian integrator (symplectic leapfrog, RATTLE where constrained, with energy-drift and charge-conservation gates), so that wall formation (R13-W), the fate of the R29 ghost sector, Floquet stability and the spinning nucleon's radiation stop being numerically unresolved | A certified damped-wave leapfrog with an energy-conservation gate at zero damping ([M5.21.6](m5_21_6_task_details.md)) and its 4×4 lift ([M5.21.9](m5_21_9_task_details.md), cited in [M5.23.1](m5_23_1_task_details.md)); the M5.32 instruments are static (FIRE, L-BFGS, eigen-pencils). No constrained symplectic form and no charge-conservation gate exists on the M5.32 action | rung candidate, the user's pick; not staged |
| 2 | A declared "certified action v2" with the H-adjoint `I1^h` for dressed-core runs, rerunning the R29 ghost read and the R3 pair on it | R15-V of [M5.32](m5_32_task_details.md): the floor witness holds on the vacuum and `−4 I1^h` repairs it there. The same record notes that this repo's registry name `I1_h` denotes the norm-only form, so the object must be pinned before any declaration | a decision on the stack of record: the user's pick, after the definition (§ 5, ask 5); not staged |
| 3 | Converge the gap pencil before more ladders: a preconditioner for LOBPCG (shift-invert or multigrid on the static Hessian) and the norm-only R4 form, ahead of the 96³ rung | R29-4 of M5.32: every ladder value is an upper bound, NOT CONVERGED at every rung; the norm-only R4 ladder was recorded as a later gap rung; the 96³ rung was not run | rung candidate, the user's pick; not staged |
| 4 | Two results of the author's capstone: (a) on a time-even action with an exact isorotation zero mode the first-order response to the rotation is a pure zero mode, so first-order cranking cannot fix a transition-current Ward mismatch; (b) for locks with a minimally gauged rigid-rotor current, (μ_p − μ_n)(Δ − N) ≤ M_N | (a) the held capstone text (v2.8) does not carry the statement; (b) the identity μ_V Δ₀ = M_p is `V84` part (3), rerun in [M5.36](m5_36_task_details.md), and the open moment 1.504 μ_N is row 3 of M5.37 § 3 | (a) recorded, not checked, ask 4; (b) R3 |
| 5 | Cheap gates before N4 and the πN test: the isospin-inertia integral on a box ladder for the census baryon (flat means N4 can be staged); for the πN test, the Δ position from the phase and a two-sided Ward check at the crossing, with the author's scorer offered as a template | the N4 reading of M5.37 § 2, from stored records with nothing computed | the inertia gate: R1; the πN gates: with the held offer of M5.37 § 5 |
| 6 | Two clock checks for the electron-ID instrument: the centroid's second moment against the Casimir radius (⟨δ²⟩ = 3λ_C²/4; the projection gives one third), and no rotating charge multipole (the core's charge density axially symmetric about the clock axis) | the fixed-clock electron of [`m5_8_2r_electron_id.py`](../scripts/m5_8_2r_electron_id.py); R1k (ii) of [M5.34](m5_34_task_details.md) read the dipole and its phase derivative at one phase | R2 |

The post closes with the author's note that posts will be checked for unfilled links and definitions before sending.

### 2. R1: the inertia gate on the stored census states

The read in equations, with `M(x)` the 3×3 order tensor of a stored state, `T_a` the three rotation generators acting by commutator at fixed position, and `h` the grid step:

```text
ι_ab(x) = tr( [T_a, M(x)]ᵀ [T_b, M(x)] ),      (T_a)_bc = −ε_abc
I_ab(R) = h³ Σ_{|x| < R} ι_ab(x)
Σ_a ι_aa(x) = 2 Σ_{i<j} (λ_i(x) − λ_j(x))²                    (λ the eigenvalues of M(x))
vacuum (1, δ, 0):   Σ_a ι_aa = 2 [ (1 − δ)² + 1 + δ² ]  =  3.16, 3.36, 3.64 at δ = 0.3, 0.2, 0.1
Θ_ab(shell) = ⟨ ι_ab ⟩ over r₁ ≤ |x| < r₂       its near-zero eigenvalues count the unbroken generators
```

**The reading, ours and labeled.** The author's statement ([M5.37](m5_37_task_details.md) § 2) is written for a record field that can vanish outside a core. This column has one tensor field, so `N` is read as the order tensor itself and `T` as the internal rotation of that tensor. Whether the inertia should instead be read on a deviation from the vacuum is author-gated (§ 5, ask 1); no such deviation is defined here for a state whose far field winds.

**The objects.** The stored endpoints of the M5.22 census, box side 48, pinned shell: the proton-analog (`P-0.5_plane_sc6`, at n 32 and n 48, the extended n 48 endpoint, scale 4 at n 32, and δ 0.1 and 0.2 at n 32) and the neutron-analog (`P-1_plane_sc6`, at n 32 and n 48, and δ 0.1 and 0.2 at n 32), with the lepton reference (`E+0.5_plane_sc6`, n 32) beside them. The stored box has one side, so the ladder the author asks for is read as the growth of the integral with the radius inside that box, on centered balls of R = 4 to 22 (the pinned shell excluded). No new box is relaxed.

**Controls, before any read.** (a) A uniform vacuum tensor: the summed density equals the closed form at round-off. (b) A random symmetric field: the summed density equals the eigenvalue form at round-off. (c) A planted compact record (a stored field blended to a multiple of the identity outside r = 8): every entry of `I_ab(R)` is flat beyond the blend, so the reader can return FLAT. (d) A uniform uniaxial tensor diag(1, 0, 0): exactly one eigenvalue of Θ vanishes.

**The column pierces every shell.** The census row of the proton-analog lists low-gap cells on the axis out to \|z\| = 20, so the vortex column crosses the outer shells and its cells do not carry the vacuum spectrum. The shell statistics are taken off the column (cylindrical radius above 6) and the on-column cells are listed separately.

**The labels, pre-registered.** On the outer shell 18 ≤ r < 22, and again on 16 ≤ r < 20: an eigen-direction of Θ is UNBROKEN if its eigenvalue is below 10⁻³ of the trace and the integral `I_ab(R)` projected on that direction moves by less than 1 percent from R = 14 to R = 22; otherwise BROKEN, with the growth from R = 14 to R = 22 listed against the pure-volume value 3.88. The state's word is the count of unbroken directions: 3 (the gate is FLAT, N4 can be staged by the author's rule), 1, or 0 (EXTENSIVE). The summed shell density is listed against the closed form; a departure above 2 percent means the exterior is not the vacuum spectrum the reading assumed and the word is withheld.

**Fails if.** The M5.37 reading (at most one unbroken generator on the census proton-analog) fails if two or more directions read UNBROKEN on the n 48 endpoint in both windows.

**What an outcome licenses.** EXTENSIVE turns the N4 reading of M5.37 § 2 from a reading of stored records into a measurement on the stored fields; N4 stays 🚧 and no state of this column meets the precondition. FLAT would make the staging of N4 a decision for the user. Neither outcome moves an icon.

### 3. R2: the two clock checks on the electron-ID instrument

**The instrument.** The fixed-clock electron of [`m5_8_2r_electron_id.py`](../scripts/m5_8_2r_electron_id.py), loaded unmodified through the import shim of [`m5_34_kit_checks.py`](../scripts/m5_34_kit_checks.py): the analytic 3×3 biaxial hedgehog with the clock pinned at phase φ in the tilt plane (the only channel with a moment), tilt 0 and the instrument's `B_STAR`, n = 24, 32, 48 at the fixed box. The charge density is the divergence of the radial 2-form, the reading of M5.34. It is sampled at 16 phases over one clock cycle, on the full interior and on the instrument's active mask.

```text
ρ(x; φ) = div B(x; φ)                    e = Σ ρ h³
c(φ)    = Σ x ρ h³ / e                   split along the moment axis ẑ:  c = c_∥ ẑ + c_⊥
⟨δ²⟩_orbit = ⟨ |c_⊥(φ)|² ⟩_φ            ⟨δ²⟩_axial = ⟨ (c_∥(φ) − ⟨c_∥⟩)² ⟩_φ
Q_ij(φ) = Σ (3 x_i x_j − r² δ_ij) ρ h³
the Darwin term:  (ħ² / 8m²c²) ∇²V = (⟨δ²⟩ / 6) ∇²V   at   ⟨δ²⟩ = (3/4) λ_C²,   λ_C = ħ / mc
a planar orbit of radius R, averaged over orientations:  ⟨δ²⟩ = R²,   so  R = √(s(s+1)) λ_C  at  s = 1/2
```

**Check (a), the centroid.** The orbit and axial second moments of the charge centroid over the cycle, in units of λ_C² under two stated readings of the instrument's clock rate ω (the rest-phase rate, λ_C = c/ω; the interband beat, λ_C = 2c/ω), against 3/4 (the Casimir radius) and 1/4 (the projection). The three symbolic statements of the block above are checked in the script. Words: AT REST (the orbit radius below 10⁻⁹ λ_C at every n), ORBITING (above, converged across the three n, the ratio to 3/4 and to 1/4 listed), FLOOR-LIMITED (above the bar and falling with h at second order or faster).

**Check (b), the multipoles.** The dipole and the traceless quadrupole at each phase; the moving part is the cycle variation over e · r_rms^l, split by harmonic of φ and by azimuthal order about the moment axis. Words: STATIC (every moving fraction below the lattice floor measured on the same grid), MOVING (above the floor and converged across n, with its harmonic), FLOOR-LIMITED. The author's 2.6 × 10⁻⁹ is listed beside the fractions and is not applied as a gate: the energy it is a fraction of is author-gated (§ 5, ask 3).

**The prior of record, not a prediction.** At φ = 0 the dipole vanishes at round-off; its phase derivative on the active mask points along the moment axis (2.43, 1.66, 0.70 at n 24, 32, 48) and falls with the mask's residual charge. A cycle-long read on the full interior was never taken, so neither word is known in advance.

**Controls, before any read.** (a) A planted Gaussian charge on a circle of radius 0.5 about the axis, rotating with φ, added to ρ at a stated amplitude: the orbit radius and the rotating quadrupole come back within 1 percent. (b) The centered hedgehog at one phase read at three resolutions: the fourfold lattice content of the quadrupole, the floor of check (b).

**Fails if.** The centered-charge reading of M5.34 ("the charge is centred, so the source-current moment is zero") fails if the full-interior centroid orbits at a radius converged across the three resolutions.

**The limit of the read.** The author's check concerns a spinning core; the instrument is an analytic seed with a pinned clock, not a relaxed state. The fixed-J states of R31 are the nearest stored spinning objects and have no charge-density reader on most of the ladder; they are outside this task.

### 4. R3: the moment bound

Arithmetic of printed numbers, in `m5_38_post_checks.py`:

| # | Item | Source | The check |
| --- | --- | --- | --- |
| 1 | μ_V Δ₀ = M_p on the author's two cards | `V84` part (3), the shipped JSON as rerun in M5.36 | the two products against 938.2721 at the printed digits |
| 2 | the data against the bound | μ_p − μ_n = 4.70589 μ_N, Δ − N = 293 MeV | the product over M_p, above 1 if the data sit outside the bound |
| 3 | "the missing ~1.5 μN" | the post | 4.70589 − M_p / 293 against 1.504 (row 3 of M5.37 § 3) |
| 4 | the direction of the inequality | the post | not derivable from the held material, which states an identity on two cards: ask 4 |
| 5 | the Darwin statements of § 3 | the post | the three symbolic identities |

Item (a) of the post's point 4 is recorded and not checked: the capstone text held on this column (v2.8) does not carry it.

### 5. R4: the audit and the reply

A fresh agent on a different model, with its own scripts and with the post in hand, attacks the claims of § 2 to § 4 before the review is presented. The reply is drafted in the terminal after the review: the results of R1 to R3, the routing of items 1 to 3 as the user decides it, and the asks below. The two asks of M5.37 § 4 are still held and ride with this reply or not on the user's word.

Asks to the author (author-gated; none blocks the run):

1. The tensor and the generators of the inertia integral on a column with no separate record field: the order tensor, or a deviation from the vacuum.
2. The centroid of the first clock check (of the charge, or of the energy) and the rate that sets λ_C on the instrument (the rest-phase rate or the interband beat).
3. The interaction energy the 2.6 × 10⁻⁹ is a fraction of.
4. The release and section that carry the first-order cranking statement and the inequality (μ_p − μ_n)(Δ − N) ≤ M_N.
5. The object named `I1^h` in the proposed second certified action (the H-adjoint of the witness of 5 September 2026, or the norm-only form).

### 6. The records

The convo row in [`m5_32_convo.md`](m5_32_convo.md); the roadmap row and change-log entry in [`m5_roadmap.md`](../m5_roadmap.md); a pointer from [M5.37](m5_37_task_details.md) § 2 to § 2 of this document.
