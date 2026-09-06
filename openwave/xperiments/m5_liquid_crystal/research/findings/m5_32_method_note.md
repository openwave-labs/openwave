# M5.32 method note: the autonomous Lagrangian hunt (rungs R0 to R10)

> **Status: the task is PAUSED, not finished.** This note is built to the
> [`METHOD_NOTE.md`](../../../../../dev_docs/METHOD_NOTE.md) standard: the reader must be able to
> audit every number below by reading, without trusting the run and without reverse-engineering
> Python. It is written for the model author and for the maintainer re-reading it later, who are the
> same reader. Section 8 carries the open questions.
>
> Task record: [`tasks/m5_32_task_details.md`](../tasks/m5_32_task_details.md) (plan, RUNG LOG, the
> pause records). Machine ledger: [`data/m5_32_ledger.json`](../data/m5_32_ledger.json).
> Code links resolve once the task's branch is merged to `main`.

## 1. The physics, before any result

### 1.1 Field, metric, conventions

```text
M(x)            a real 4x4 matrix field on a periodic cubic lattice
eta             diag(-1, +1, +1, +1); index 0 is time, as a derivative index AND as
                the internal row of M
A_mu            d_mu M, the jets, with RAW CONTRAVARIANT internal entries
Lorentz action  M -> L M L^T   (raw entries contravariant)
                under M -> L^-T M L^-1 the roles swap; the two agree on M_cov = eta M eta,
                so any covariant-metric object must be converted before mixing
vacuum          M_vac = diag(-s g, 1, delta, 0);  toy point s = -1, g = 32, delta = 0.3
lattice         h = L / n, certified symmetric stencil: the density is formed per
                forward and backward branch, then weight-averaged
```

Contraction rule, locked at R0 and audited: a derivative-derivative index pair contracts with
`eta`, an internal-internal pair with `eta`, and a MIXED derivative-internal pair with `delta`.
The all-`eta` reading is not covariant (measured boost drift 32.2, the R0 audit's figure) and is retained only as the
control term `I3_mixed_eta`.

### 1.2 The certified action

```text
F_mu nu      = A_mu eta A_nu  -  A_nu eta A_mu            (curvature, quadratic in the jets)
<F, G>_eta   = tr( eta F eta G^T )                        (the bracket)
I1           = sum_{mu < nu} eta^mu mu eta^nu nu <F_mu nu, F_mu nu>_eta
             = (1/2) F_abcd F^abcd
V4           = w sum_{p = 1..4} ( tr((M eta)^p) - C_p )^2 ,  C_p = (s g)^p + 1 + delta^p
w            = 7.24023879e-4
L_cert       = -4 I1  -  V4                               (CERTIFIED_COEFFS)
```

### 1.3 The clock, and the two energies

The clock is a one-parameter internal rotation applied in the body frame of the hedgehog ansatz.
Its tangent at `t = 0` is the field `a0`, and the time jet is `A_0 = omega a0`:

```text
a0(x)        = Qh(x) ( G1 d4 + d4 G1^T ) Qh(x)^T          (the co-moving flow)
G1           the (2,3)-plane rotation generator: G1[2,3] = -1, G1[3,2] = +1
Qh(x)        = R3(phi) R2(theta), the Euler frame carrying the eigenvalue-1
             eigenvector to n-hat = x / |x|
```

Every term in the registry is a polynomial in `omega`, so the Lagrangian read and the Hamiltonian
(energy) read are related term by term by a Legendre transform:

```text
quadratic terms    I(omega) = A + B omega + C omega^2
                   H_I      = C omega^2 - A
quartic terms      I(omega) = A + C2 omega^2 + C4 omega^4
                   J        = dI/domega = 2 C2 omega + 4 C4 omega^3
                   H_I      = C2 omega^2 + 3 C4 omega^4 - A
lattice energy     E_cert   = 4 (U + omega^2 T) + V4      (the factor 4 is |CERTIFIED_COEFFS[I1]|)
clock inertia      kin      = -4 x (the omega^2 coefficient of I1) = INS4.kin_of(M, a0, cfg)
                            the two measures agree on the rigid ansatz to ten digits
fixed-J energy     E_J      = E_stat + J^2 / (4 kin)  ,  omega* = J / (2 kin)
```

`B` (the `omega`-odd piece) is zero for `I1` and nonzero for the mixed contractions `I2` to `I6`;
it shifts the fixed-J relation to `omega* = (J - B) / (2 C)` but leaves `H` even, so it never
creates a free minimum. That Legendre argument is re-verified symbolically per term.

### 1.4 The candidate families tested

```text
lambda-family (class C2, rung R2)
    L_lambda = -4 [ (1 - lambda) I1 + lambda I1_h ] - V4
    h_cov    = eta + 2 (eta u)(eta u)^T ,  u the timelike unit eigenvector of M eta, u^T eta u = -1
    I1_h     the bracket with eta -> h_cov on the INTERNAL pair only

K_T (class C4, rung R7)
    K_T      = (1/2) sum_mu eta^mu mu [ tr(h A_mu h A_mu) - tr(eta A_mu eta A_mu) ]
             = 2 sum_mu eta^mu mu sum_j (A_mu)_{0j}^2      in the u-frame
    L        = -4 [ (1 - lambda) I1 + lambda I1_h ] - c2 K_T - V4 ,  c2 > 0

quartics (classes C5 and C6, rung R8)
    Q_I1sq   = (I1 density)^2
    Q_I4sq   = (I4 density)^2 ,  I4 = R_ac R^ac ,  R[nu, a] = sum_mu F[mu, nu, a, mu]
    Q_Fpair  = sum_{mu<nu, rho<sigma} eta-weighted <F_mu nu, F_rho sigma>_eta^2
    Q_C6a    = [ sum_mu eta^mu mu tr(A_mu eta A_mu eta) ]^2
    Q_C6b    = sum_{mu nu} eta^mu mu eta^nu nu [ tr(A_mu eta A_nu eta) ]^2
    Q_BI     = b^2 ( sqrt(1 + 2 I1 / b^2) - 1 ) ,  b^2 = 1e4      (not polynomial in omega)
```

### 1.5 The ansatz, the relaxation, and the degree

```text
ansatz       M = Q d4 Q^T ,  Q = Qb Qh ,  d4 = diag(-s g, 1, delta, 0)
             Qb = I + sinh(m) K + (cosh(m) - 1) K2, the boost dressing built from n-hat
             at m = 0 (used throughout R7 to R10) Q = Qh, the Euler frame alone
relaxation   FIRE on E_static only (a0 = None, omega = 0), boundary shell pinned at the ansatz
             by ~pin_shell(n, h) with default depth 1.6
degree       read_charge_from_M on the SPATIAL 3x3 block of M (every caller passes
             M[..., 1:4, 1:4]): eigh, take V[..., -1] (the LEADING eigenvector, n-hat), lift
             its sign field over a centered cube surface, integrate the RP^2 degree
```

The last line is stated here in full because rung R10 turns on it: the instrument reads ONE
eigenvector of the spatial block, not the full order parameter, and the time row is invisible to
it. The sign of that degree is a lift convention, so it is reported as `+-1` throughout.

### 1.6 The gate and class vocabulary used below

The run pre-registered seven gates a candidate had to pass, and worked through term classes in a
fixed order. Both are defined in full in the task record
([`tasks/m5_32_task_details.md`](../tasks/m5_32_task_details.md), sections 4 and 8); the short
form, so the tables below can be read without it:

| Gate | Statement |
| --- | --- |
| G1 | Coulomb kept: like charges repel with a 1/r trend, the static sector unchanged |
| G2 | Newton reversed: two boost-dressed defects attract |
| G3 | Electron clock: a finite nonzero omega* at positive energy, the vacuum preferring omega = 0 |
| G4 | Lorentz covariance of the energy functional, verified numerically |
| G5 | Bounded below with no guard on every runaway family |
| G6 | Collateral: the certified positives survive (census ordering, protection, the fixed-J clock) |
| G7 | Parsimony and robustness: few terms, O(1) coefficients, every sign holding over a factor >= 4 |

| Class | Content |
| --- | --- |
| C0, C1 | the author's two contractions, and the full quadratic basis of `F x F` |
| C2 | field-dependent internal metrics (`h_cov`), `M`-inserted contractions, projector currents |
| C3 | the potential axis: eigenvalue penalties, the LdG lift, dressing-sensitive `V` |
| C4 | the lower-order 2-derivative term (`K_T`) |
| C5, C6 | curvature^4 and saturation; non-commutator quartics |
| C7, C8 | higher-order timelike-current / Skyrme contractions; cross-model imports (never opened) |

## 2. Equation-to-code map

Base: `https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/`

| Object in section 1 | Function | File and lines |
| --- | --- | --- |
| `F_mu nu` from the jets | `F_of_A` | [`scripts/m5_32_lagrangian.py#L134`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L134-L141) |
| the bracket and every contraction pattern | `_K_from_pattern`, `density_from_K` | [`#L142`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L142-L182) |
| `I1` (sympy reference) | `I1_sym` | [`#L313`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L313-L321) |
| `I4 = R_ac R^ac`, the mixed trace | `I4_sym`, `R_readings_np` | [`#L355`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L355-L360), [`#L205`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L205-L219) |
| `V4` and its weight `w` | `V4_sym`, `v4_density_np`, `W1` | [`#L371`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L371-L382), [`#L236`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L236-L242), [`#L109`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L109) |
| `L_cert = -4 I1 - V4` | `CERTIFIED_COEFFS` | [`#L469`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L469) |
| `A_0 = omega a0`, the stencil branches | `lattice_jets` | [`#L479`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L479-L491) |
| `H_I = C omega^2 - A` | `term_hamiltonian` | [`#L500`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L500-L510) |
| `(A, B, C)` per term | `omega_decompose` | [`#L515`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py#L515-L522) |
| `K_T` density, both readings | `kt_density_np`, `kt_density_sym` | [`scripts/m5_32_r7_a_kt_form.py#L122`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r7_a_kt_form.py#L122-L154) |
| the u-frame time row | `uframe_time_row` | [`#L410`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r7_a_kt_form.py#L410-L419) |
| `E_stat` and `kin` under `L_lambda + c2 K_T` | `es_kin` | [`scripts/m5_32_r7_b_kt_lattice.py#L207`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r7_b_kt_lattice.py#L207-L213) |
| `E_J` minimized over the dressing | `min_over_amp`, `scan_R` | [`#L230`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r7_b_kt_lattice.py#L230-L275) |
| the six quartic densities | `d_I1`, `d_I4`, `d_Fpair`, `d_C6a`, `d_C6b`, `d_BI`, `QUARTICS` | [`scripts/m5_32_r8_a_quartics.py#L68`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r8_a_quartics.py#L68-L140) |
| the exact degree-4 `omega` extraction | `omega_poly` | [`#L158`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r8_a_quartics.py#L158-L175) |
| the generator enumeration `[X, M_vac]` | `generators`, `stage_generators` | [`scripts/m5_32_r8_b_ir_theorem.py#L53`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r8_b_ir_theorem.py#L53-L94) |
| the far-field tail measurement | `stage_tail` | [`#L95`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r8_b_ir_theorem.py#L95-L136) |
| the frame-free identity at `delta = 0` | `stage_equivalence` | [`scripts/m5_32_r9_b_string.py#L74`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r9_b_string.py#L74-L109) |
| the continuum ring (the string, measured) | `M_continuum`, `stage_continuum_ring` | [`#L110`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r9_b_string.py#L110-L147) |
| the fixed-physical-radius excision | `rho_of`, `run_box` | [`scripts/m5_32_r9_a_tube.py#L48`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r9_a_tube.py#L48-L82) |
| `kin` on a relaxed field, and its shells | `kin_c2`, `kin_shells` | [`scripts/m5_32_r10_relaxed_ladder.py#L72`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r10_relaxed_ladder.py#L72-L91) |
| the relaxation protocol | `relax` | [`#L92`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r10_relaxed_ladder.py#L92-L125) |
| FIRE, the pinned shell, `kin_of`, `e_parts` | `fire`, `pin_shell`, `kin_of`, `e_parts` | [`scripts/m5_21_3_a_4d.py#L327`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_21_3_a_4d.py#L327), [`#L109`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_21_3_a_4d.py#L109), [`#L274`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_21_3_a_4d.py#L274), [`#L179`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_21_3_a_4d.py#L179) |
| the ansatz and the clock tangent | `dressed`, `a0_unit` | [`scripts/m5_21_8_b_lattice.py#L56`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_21_8_b_lattice.py#L56-L87) |
| **the degree instrument** (the record's) | `read_charge_from_M` on the spatial 3x3 block | [`scripts/m5_22_e_audit.py#L192`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_22_e_audit.py#L192-L200), called through [`m5_32_r6_a_deltaladder.py#L156`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r6_a_deltaladder.py#L156-L169) |
| the R10 audit's own degree reader (a different lift and triangulation) | `directors`, `read_surface` | [`scripts/m5_32_r10_audit.py#L142`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r10_audit.py#L142), [`#L247`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r10_audit.py#L247) |
| the degree-0 vacuum-interior seed (the 78 % figure) | `unwound_seed` | [`scripts/m5_32_r10_audit.py#L260`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r10_audit.py#L260) |
| the melt paths, the clock taper, the `g = 32` probe (the R10 audit's typed results) | the `zero_barrier_robustness`, `clock_taper`, `scope_g32` blocks | [`scripts/m5_32_r10_audit.py#L1036`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r10_audit.py#L1036-L1092) |
| the note audit's independent re-derivation of all of the above | `path_energy`, `stage_barrier`, `stage_boundary`, `stage_taper`, `stage_g32` | [`scripts/m5_32_note_audit.py#L446`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_note_audit.py#L446-L708) |

## 3. The physics module

[`scripts/m5_32_lagrangian.py`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py)
is the single-purpose registry: each term has ONE definition string (hashed, so a term is never
silently re-tried under a different meaning), a sympy implementation on the notebook conventions,
and a numpy implementation on the certified stencil, plus per-term selftests. Every driver imports
it; no driver re-implements physics. Run `python3 scripts/m5_32_lagrangian.py --selftest` for the
17-line check, and `--mutant eta_time_row` for the negative control that must redden it.

## 4. Results, each with its pre-registered gate

| # | Result | Gate it was pre-registered against | Convergence evidence |
| --- | --- | --- | --- |
| R0 | The stack, the record and the author's 2026-08-17 Newton notebook all reproduce | selftests within 1e-3 of the record | 17/17 selftests; 10/10 record items at <= 2.2e-15; notebook fit `A = 863.733`, `B = 167.668`, sign `+`, to six digits |
| R1 | **The whole constant-coefficient current-order class is infeasible**: no coefficients make the energy's `omega^2` form PSD on every time channel with the boost weight reversed | a coefficient region existing under either Coulomb gate | Farkas / LP certificates at `(g, delta)` = (32, 0.3), (8, 0.3), (32, 0.1), with and without the parity-odd terms, on every channel alone, even at `c_I1 = +4` |
| R2 | The covariant flip family `L_lambda` is bounded below for `lambda >= 1/2` by a pointwise theorem wherever `M eta` has a real timelike eigenvector, keeps the static sector exactly, and gives `lambda* = 1/2` on every channel | G4, G5, the sector half of G1 | 0 negative densities in 27,560 random non-Lorentz samples; 36 channel x g cases; lattice probes bounded with no guard |
| R3 | G2 not met on any of three constructions | sign robust across 2 of 3 constructions, 2 boxes, both boundary types | ansatz repulsive at 348 reads; 44 relaxed pair heals; the cross-inertia undecidable at this resolution |
| R4 | The fixed-J minimizer runs to the box wall on every localized dressing family, `omega*` proportional to `1/L` | an interior `R*` with `omega*` stable across the domain ladder | 96/96 producer cases and every audit case; `omega* L` constant at 7.1 |
| R6 | **C3 orbit-blindness theorem**: any Lorentz-invariant derivative-free `V` is constant along a Lorentz dressing | a potential that localizes the dressing | variation <= 2e-7 on 50 dressed points and the whole R4 family up to rapidity 3; Euclidean controls O(1e4) |
| R7 | `K_T` moves the fixed-J minimizer off the box wall, but the audit found the interior minimum is the dressing switching off (0.43 % deep), and the term is exactly inert on the realized clock channel; G7 fails on the drift gate alone | interior `R*` with `omega*` drift <= 10 % over a c2 range >= factor 4 | interior at `c2` = 0.03 and 0.1 in both boxes; drift never below 0.301 against the 0.10 bar; the range half is MET on a dense ladder (factor 4.87) |
| R8 | C6's `omega^4` inertia is exactly VOLUME extensive and h-independent | an IR-convergent `omega^4` inertia | L exponent 3.0000 to 1e-13, ratio 8.000 over a factor 2 in L; h exponent -3.6e-14 |
| R9 | The ansatz carries a topologically protected biaxial disclination on the z axis; at `delta = 0` the field is exactly frame-free and the PERIODIC clock vanishes identically (a smooth radial-boost flow survives). The audit's headline: a relaxation resolves the line into a finite core (radius 3.98 at h = 1.5, 3.58 at h = 0.75) with the clock surviving at an h-convergent inertia 351.17 / 351.14, at `g = 8` | a string-free hedgehog with a nonzero clock | frame-free identity to 2.08e-17 relative; continuum ring spread exactly `delta / 2` and radius-INDEPENDENT over a 1e4 shrink, against a spread proportional to the radius at `delta = 0` |
| R10 | **No energy barrier protects the ansatz's degree, and the extensive inertia is not a property of an object** | the relaxed core-resolved soliton's inertia still extensive (the producer's registered prediction; the unwinding and boundary claims below are the AUDIT's findings and carry no pre-registered gate of their own) | from the UNRELAXED ansatz, a straight-line melt to the degree-0 state never rises above its start (energy 62.852 at the start, 14.794 at the end, \|Q\| 1 -> 0 inside each of five melt windows, 201 points each); the barrier from the RELAXED state was not computed, and the probes run from it rise 0.73 to 4.49; a degree-0 configuration with a vacuum interior out to `r = 15` still carries 78 % of the inertia (272.20 of 351.17, equal-budget unconverged comparison); a linear taper of the clock flow over `r = 12` to `15` leaves 32.9 % and removes the box dependence, as any compactly supported clock on a box-independent interior must |

### 4.1 The two results that reverse earlier readings

**The degree instrument is not an invariant of this order-parameter space.**
`read_charge_from_M` takes `V[..., -1]` of the spatial 3x3 block, the leading eigenvector alone, so
it measures an `RP^2` degree of one eigenvector. The stabilizer of `d4` in `SO(1,3)+` is the Klein
four-group (an `L` with `L^T eta L = eta` and `L d4 L^T = d4` must commute with `d4 eta`, whose
spectrum is distinct, so `L` is a diagonal sign matrix; `det = +1` and `L_00 = +1` leave four), so
`pi_1 = Q8` and `pi_2 = 0`. With `pi_2 = 0` and the eigenvalues frozen on the `SO(1,3)+` orbit of
`d4`, any CONTINUOUS map `S^2 -> OPS` has leading-eigenvector degree 0; the degree `+-1` reading is
possible only because the ansatz is discontinuous on the measurement surface. On the same three
surfaces the instrument calls conflict-free, the MIDDLE eigenvector admits no consistent sign lift
at all. What was measured about protection is narrower than a barrier: from the unrelaxed ansatz a
straight-line melt to degree 0 never rises above its start, while the relaxed state's own barrier
was not computed and FIRE holds `|Q| = 1` on every surface through 12000 iterations; the protection
argument is `pi_2 = 0`, not a measured barrier.

**The extensive clock inertia belongs to the boundary and the frozen-clock convention.**
`kin` is quadratic in `a0`, so the frozen (non-tapering) clock flow is only an UPPER BOUND on the
fixed-J inertia, and an upper bound that grows with `L` cannot establish that the inertia grows with
`L`. A linear taper of the clock flow to zero over `r = 12` to `15` leaves 32.9 % of it, and a
compactly supported clock on a box-independent interior is L-independent by construction.

## 5. Minimal inspection set (physics first, driver last)

| Order | Artifact | Why |
| --- | --- | --- |
| 1 | [`scripts/m5_32_lagrangian.py`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_lagrangian.py) | the functional: every term's definition, sympy and numpy side by side |
| 2 | [`scripts/m5_22_e_audit.py#L192`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_22_e_audit.py#L192-L200) | the degree instrument, because section 4.1 turns on what it measures |
| 3 | [`scripts/m5_32_note_audit.py`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_note_audit.py) | the independent instrument behind section 9: its own ansatz, energy, clock tangent and degree reader, so every load-bearing number here has a second implementation to read against |
| 4 | [`data/m5_32_ledger.json`](../data/m5_32_ledger.json) | every rung's hypothesis, claims and audited verdicts in machine form; the drivers are listed there and are the last thing to read |

## 6. What was NOT computed

| Not computed | Why it matters |
| --- | --- |
| Any candidate carried through the full G1 to G7 battery | no candidate survived far enough; the `lambda`-family died at G2 and G3, `K_T` at G3 and G7 |
| The classes C7 (higher-order timelike-current / Skyrme contractions) and C8 (cross-model imports) | never opened; R10's criterion says they cannot move G3, but that is an argument, not a measurement |
| A relaxed two-box ladder at the toy point `g = 32` | every relaxation here is at `g = 8`, where `V4` is 4096x softer; the `g = 32` probe reached `V4 = 0.00097` with no cell isotropized (the smallest top eigenvalue gap 0.616 against the 0.35 threshold that defines the melt front) but ended unconverged at `fmax` 5.14 |
| Whether a protected object exists in this space at all | `pi_1 = Q8` suggests a disclination LOOP rather than a point hedgehog; not tested |
| A converged relaxation anywhere | every FIRE run stops on `max_iter`; the audited slope decrements per doubling depend on which box pair is read (-0.267 / -0.403 on (24, 36), -0.159 / -0.130 on (36, 48)), so neither convergence nor decay is established |
| The unwinding barrier of the RELAXED state | only the melt from the unrelaxed ansatz was run; the probes from the relaxed state rise 0.73 to 4.49, and FIRE never moves the degree |
| A degree instrument that sees the time row | the instrument reads the spatial 3x3 block only, so anything the relaxation does to the time row is invisible to it |
| Converged comparisons behind the 78 % and 32.9 % figures | both are equal-budget, unconverged comparisons |
| The physical clock localization | which clock flow is the physical one is a convention question the run could not settle from inside |
| `J = hbar / 2` in program units | undefined in the record; never invented, so every fixed-J number is at an arbitrary `J` |
| The Coulomb pair half of G1 on the 4x4 stack | the like-charge static control fails on this stack (the string form), so the instrument could not decide it |

## 7. The adversarial audit record

Every rung was audited by an independent agent instructed to REFUTE, with its own implementation
(different stencil branch order, own amp grid, own minimizer, own densities) and forbidden from
reading the producer's scripts. The audits are the reason several headline claims below R7 no longer
stand as first written.

| Rung | Verdicts | What the audit changed |
| --- | --- | --- |
| R7 | 8 CONFIRMED, 5 QUALIFIED, 0 REFUTED | found that the LP channel list contains no channel built from the clock the model actually runs, so `c2` gives exactly zero help there; found the dressed-pair Coulomb anchors are not `c2`-independent |
| R8 | 4 CONFIRMED, 4 QUALIFIED, 2 REFUTED | found the ansatz's z-axis discontinuity and that 73 to 98 % of every C5 quartic coefficient sits beside it; refuted the producer's `c5` coefficient ladder as an h-artifact (`h^+2.99`) |
| R9 | 5 CONFIRMED, 2 QUALIFIED, 2 REFUTED | refuted the producer's exclusion theorem by RELAXING it: the line resolves into a finite core and the clock survives; established `pi_1 = Q8`, `pi_2 = 0` |
| R10 | 2 CONFIRMED, 3 REFUTED | refuted the degree's topological meaning, measured the unwinding barrier at exactly 0.0, showed the inertia belongs to the boundary, and scoped the whole core-melt effect to `g = 8` |

Producer errors caught and logged rather than buried: an off-center excision mask (built on a
cell-centered grid while the density lives on the certified offset grid); a spherical-shell
integration biased 18.5 % low because it discards the cube corners; a generator table built from
`X M - M X^T`, which is antisymmetric and not a tangent; a "converging to a nonzero value" claim
withdrawn by the producer on its own 12000-iteration point and sent to the auditor as a claim to
refute BEFORE that auditor ruled.

One certified-stack defect was found and is owed as a platform issue: `gen_catalog` normalizes `a0`
by `max(norm, 1e-300)`, so at `delta = 0` it returns a unit-norm noise field and reports a phantom
`kin` of 2.25. Any `delta -> 0` study routed through it sees a clock that is not there.

## 8. Open questions for the author

Each of these is author-gated in the strict sense: the run can measure around it but cannot settle
it from inside, because the answer is a statement of intent or of convention about the model.

| # | Question | Why it is author-gated |
| --- | --- | --- |
| 1 | Which clock localization is physical: the rigid co-moving flow, or one that decays away from the defect? | the run measured that the answer decides whether the clock inertia is extensive at all: tapering the flow at `r = 12` leaves 32.9 % of it and removes the box dependence entirely |
| 2 | Is the electron intended as a point hedgehog, given `pi_2 = 0` and `pi_1 = Q8` in this order-parameter space? | the protected objects here are lines, and a disclination loop is a different object |
| 3 | The intended reading of the mixed trace `R_ac` | exactly one independent mixed trace exists up to sign, and it is not symmetric, so `I4 != I5` |
| 4 | `J = hbar / 2` in program units | undefined in the record; every fixed-J number is at an arbitrary `J` without it |
| 5 | Whether a spectral function of `M` (a projector, `h_cov`) is admissible under the model's own boundary | `h_cov` is undefined past the degeneracy locus `t* = (g + 1) / 2`, where the spectrum of `M eta` goes complex |
| 6 | The earlier open questions carried from the M5.21 series | in the model's question tracker ([`m5_question_tracker.md`](../m5_question_tracker.md)), unchanged |

## 9. The pre-send audit of this note

Per the standard, an independent agent (a different model from the one that produced the run)
audited THIS DOCUMENT before it was sent: it re-derived the six load-bearing claims with its own
ansatz, stencil energy, clock tangent, kin density, degree reader (a different sign lift and
triangulation) and melt paths, traced every number in the note to its artifact, re-walked all rows
of the equation-to-code map on the working tree, and checked every section 1 equation against the
code term by term. Instrument: [`scripts/m5_32_note_audit.py`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_note_audit.py);
record: [`data/m5_32_note_audit.json`](../data/m5_32_note_audit.json).

| Load-bearing claim | Verdict | The auditor's own number |
| --- | --- | --- |
| the degree instrument reads one leading eigenvector | CONFIRMED, with a correction to the note's wording (it acts on the spatial 3x3 block; fed the 4x4 it raises) | `+-1` on three surfaces; the middle eigenvector admits no lift |
| stabilizer = Klein four-group, `pi_1 = Q8`, `pi_2 = 0` | CONFIRMED by its own derivation | stabilizer order 4; the SU(2) lift closes on a non-abelian group of order 8 containing `-1` |
| the unwinding barrier is exactly 0.0 | **QUALIFIED, and the note's lead claim was re-scoped on it**: the number reproduces exactly (five melt windows, 201 points each, endpoint 14.7940) but the path starts at the UNRELAXED ansatz; from the relaxed state the probes rise 0.73 to 4.49 and FIRE holds `\|Q\| = 1` through 12000 iterations | 0.0 from the rigid start; 0.725 / 3.037 / 4.490 from the relaxed states |
| a degree-0 vacuum-interior state carries 78 % of the inertia | CONFIRMED from its own seed | 272.204 / 351.170 = 77.5 % |
| a clock taper at `r = 12` leaves 32.9 %, L-independent | CONFIRMED, with the taper's shape made explicit (a linear ramp to zero over 12 to 15) | 115.385 / 351.170 = 32.86 %; L = 36 / 48 / 60 agree to 4 digits |
| at `g = 32` the relaxation leaves `V4 = 0.00097` and no melt | CONFIRMED | own `V4` 0.000970; smallest top gap 0.616 against the 0.35 threshold |

Beyond the six: all code-map anchors resolved to the named function; every section 1 equation
matched the code (`I1 = (1/2) F_abcd F^abcd` to 8e-15, `E_cert = 4(U + omega^2 T) + V4` to 2e-16,
both Legendre forms, `kin = -4 C(I1)`, the six quartics); the per-rung audit verdict counts matched
the audit records. Six blocking findings were applied before sending (the barrier re-scope above,
the instrument's input block, "monotone" replaced by "never above the start", the gate and class
legend added as section 1.6, two transcription errors corrected: the boost drift figure and a
mid-run `fmax` reported as an endpoint) and the fourteen minor findings were folded into the rows
they concerned. The audit did not edit this note; the producer applied its findings and this
section records them.

## 10. Addendum (2026-08-29): the author's reply and the two rungs it triggered

The author's reply to this note (record: [`m5_32_convo.md`](../tasks/m5_32_convo.md) § 2026-08-29) attached a same-sign version of the Newton notebook and proposed `(F_abcd F^abcd)^2` against the omega divergence; on the object question it named the charged ring as acceptable. Two rungs ran the same day, each with its independent audit; the rung rows and the record section are in the [task doc](../tasks/m5_32_task_details.md#r11--r12-record-2026-08-29-the-staged-rungs-after-the-authors-reply).

| Rung | What was measured | Audited outcome |
| --- | --- | --- |
| R11 | the 08-29 notebook = the 08-17 notebook with the spatial density globally negated (same-sign centers in both); negating the static curvature sector has no floor on exact Lorentz orbits (`E_u[M_s] = s E_u[M]`, V4 ~ s^-3); the same-sign pair stays repulsive under the certified action; `(F_abcd F^abcd)^2` = the C5 quartic of § 4 (R8): its well-opening coefficient grows with L and the clock frequency at fixed coefficient drifts 42 % across the box ladder in both Hamiltonian readings, with the static energy driven negative at threshold | 4 / 4 CONFIRMED; one producer explanation refuted (a grid artifact); qualifiers: the notebook's fit constants scale as 1/sqrt(cutoff) |
| R12 | the charged disclination ring (the M5.21.2 seed) under the § 1.5 relaxation protocol: half-winding q = 1/2 survives 3000 iterations at two radii and two boxes (a protected object exists here); the cord shrinks 6.5 % sub-grid and decelerating (park or collapse undecided); the rigid clock inertia is extensive exactly as for the hedgehog, and the tapered density peaks at r = 9-12 for both objects | instrument CONFIRMED; three producer readings REFUTED (no-shrink, energy lead, seed-level fixed-J minimum) |

What changed in the reading of § 4.1: the extensive inertia has the same radial shape for the ring and the point, so it is a property of the clock CONVENTION (the rigid rotation of the vacuum frame), not of the object; the § 8 question 1 (which clock localization is physical) is now the load-bearing one, and a clock flow that vanishes in the vacuum is the candidate to build. Not computed here: the 12000-iteration ring ladder, relaxed-ring fixed J with a box-scaled taper, the direct dilation test of the (F·F)² class against a flipped static sector.


## 11. Addendum (2026-09-05): R13-W and the R14 ladder (the two-derivative class and the coexistence conjecture)

The coordination thread (record: [`m5_32_convo.md`](../tasks/m5_32_convo.md), 2026-08-29 to 2026-09-05) carried the author's degenerate-wall clock convention (tested as R13-W, [ledger § 6.2](m5_32_candidate_ledger.md)), then the author's 2026-09-03 analysis with three two-derivative candidate terms and a structural conjecture, and the corrections of 2026-09-05; the R14 packet was frozen in [ledger § 6.3](m5_32_candidate_ledger.md) before any number and ran as an autonomous ladder ([task record](../tasks/m5_32_task_details.md), R14 section, with five independent audits). Equations first, the code map, then the results with their gates.

### 11.1 The entrants (equations; the E-orientation: a positive coefficient is the energy-positive sense of the certified `4 I1`)

| Term | Definition | Character |
| --- | --- | --- |
| `K_lambda` | `E = (1/2) sum_a [ sum_i (d_i lambda_a)^2 + omega^2 (d_t lambda_a)^2 ]`, `lambda_a` the eigenvalues of `N = M eta`; on the lattice the static part is the certified finite difference of the sorted spectrum fields; `d_mu lambda_a = (v_a^T eta A_mu eta v_a) / (v_a^T eta v_a)` | zero on every Lorentz-orbit texture and on every generator channel: an eigenvalue-channel stiffness only |
| `R_G` | `sum_{mu nu} G_cd [ (A_mu)^{nu c} (A_nu)^{mu d} - (A_mu)^{mu c} (A_nu)^{nu d} ]`, the derivative index of one jet contracted by delta with a raw internal index of the other, `G` a covariant (0,2) tensor: `eta`, `eta M eta`, `M^-1`, `h_cov = eta + 2 (eta u)(eta u)^T` | no `eta^{mu nu}`, hence no omega^2 content for any `G`; `R_eta` has EL identically zero; `M^-1` is undefined on the certified vacuum (eigenvalue 0) |
| `K_P^h` | `E = (1/2) [ sum_i tr(Om_i^T H Om_i H^-1) + omega^2 tr(Om_0^T H Om_0 H^-1) ]`, `Om_mu = P A_mu eta P`, `P = (N - lambda_t)(N - 1)` with `lambda_t = -g` the vacuum's timelike eigenvalue, `H = eta + 2 (eta u)(eta u)^T`, `H^-1 = eta + 2 u u^T`: the Frobenius norm of the projected jet in the eta-orthonormal eigenbasis of `N` (PSD everywhere) | blind to boosts and tilts at the vacuum; phase stiffness `[f(lambda_2) f(lambda_3)]^2 (lambda_2 - lambda_3)^2` on the (2,3) block, `f(x) = (x + g)(x - 1)`; the plain trace `tr(Om^2)` is indefinite off the vacuum spectrum, and the transposed placement `tr(Om H Om^T H^-1)` is not invariant (R14-0 audit) |
| `T1 .. T4` | `T1 = eta^{mu nu} tr(A_mu eta A_nu eta)`, `T2 = eta^{mu nu} tr(A_mu eta) tr(A_nu eta)`, `T3 = div_b eta^{bd} div_d` with `div^b = sum_mu (A_mu)^{mu b}`, `T4 = sum_{mu nu} (A_mu)^{mu nu} tr(A_nu eta)` | the covariant constant-coefficient quadratic jet forms; `T5 = T3 + R_eta`; the Frank form with zero splay `Q_F = T1 - T3` is in the span |

The modified potential of R14-D: `V' = V4 + mu (m2 - m3)^2` (the (2,3) eigenvalue penalty, class C3).

### 11.2 Equation-to-code map (additions)

| Equation | Code |
| --- | --- |
| the entrants, their selftests (covariance, positivity on the orbit, the perturbation formula against finite differences, complex-step gradient gates) | [`m5_32_r14_terms.py`](../scripts/m5_32_r14_terms.py): `klam_static_fd`, `klam_kin`, `klam_energy_grad`, `rg_density`, `rg_grad`, `rg_hcov_energy_grad`, `kp_static` (plain), `kp_h_static`, `kp_h_kin`, `kp_h_energy_grad` (jets and eigenbasis chained) |
| the R14-0 statements, each with a named mutation | [`m5_32_r14_0_verify.py`](../scripts/m5_32_r14_0_verify.py) |
| the LP: basis, rows, the exact UV quadratic forms, cutting planes, the rational certificate | [`m5_32_r14_a_lp.py`](../scripts/m5_32_r14_a_lp.py): `build_basis`, `build_rows`, `uv_quadratic_forms`, `stage_refine`, `farkas`; the quadratic forms `d_T1 .. d_T4` |
| the fixed-J descent with `K_P^h` | [`m5_32_r14_b_fixedj.py`](../scripts/m5_32_r14_b_fixedj.py): `fire_kph` |
| the Newton arms | [`m5_32_r14_c_newton.py`](../scripts/m5_32_r14_c_newton.py): `wrapped_energy_grad` (over `m5_32_r2_b_bounded.energy_grad`), `stage_klambda` |
| the rotating-frame potential of uniform states, the modified potential, the split line | [`m5_32_r14_d_bridge.py`](../scripts/m5_32_r14_d_bridge.py): `v4`, `iota`, `main` |
| the coexistence wall on the reduced 1D functional and its lattice cross-check | [`m5_32_r14_d2_wall.py`](../scripts/m5_32_r14_d2_wall.py): `F_reduced`, `relax_wall`, `lattice_check` |
| the LP corner under the descent | [`m5_32_r14_b2_vertex.py`](../scripts/m5_32_r14_b2_vertex.py) |
| the term catalog | [`m5_32_term_catalog.md`](m5_32_term_catalog.md) |

### 11.3 Results, each with its pre-registered gate (the numbers and the audit counts in the task record)

| Rung | Gate | Audited outcome |
| --- | --- | --- |
| R13-W (2026-09-02) | the wall convention gives a localized fixed-J clock on `L_cert` (W1 tension, W2 decoupling, W3 bag) | `ESTABLISHED_KINEMATIC` at best: every planar profile has `E_u = 0` (walls tensionless), the phase field on the vacuum has no action, a non-commuting planar twist carries inertia at zero static cost (no fixed-J minimizer on `L_cert`, a theorem), W3 not stationary (eigenvalue-zigzag flank inertia) |
| R14-0 | the author's 09-03 statements, CONFIRMED / QUALIFIED / REFUTED each with a mutation | 10 / 4 / 0 in the audit; the orbit theorem holds on rotation orbits and single-plane boost textures and fails on two-plane boost textures for the three `M`-dependent `G`; the free inertia (S5) survives the whole two-derivative set |
| R14-A | `CLASS_INFEASIBLE` (certificate) or `CONE_FEASIBLE` (vertices) over the frozen basis and rows, exact UV forms included | the two-derivative class, with or without the quartics: `CLASS_INFEASIBLE` with an exact rational certificate on both the producer's and the auditor's assembly (the binding structure: the like-charge 1/d form against the two hedgehog tails, with the zigzag sheet or the relaxed pair); the full basis has no point below coefficient norm 100, and its bounded corners above that (an `I1_h` corner at 675, an `I6` corner at 200 on the auditor's Coulomb block) are outside the linear-response validity of the rows; the certified `4 I1` has a negative omega^2 coefficient on the hedgehog boost tangents, repaired by `K_T >= 0.064` in every feasible point |
| R14-B | `PERIODIC_ORBIT_EXISTS` (ladder convergence) or `CANDIDATE_REFUTED` | `CANDIDATE_REFUTED` at c = 1, 3: no stationary state (logarithmic plateau), the fixed-J term numerically invisible, `omega = J / (2 kin)` a lattice cell count (the exterior ticks; the descent is h-blind); the pre-registered (2,3) closure started and stalled; the boost sector never sampled (a saddle at c = 0.3, undecided there) |
| R14-C | G2-lite per term and sign | `R_G`: the pair slope is `(certified) + c_R (R_G slope)`, `-882 + 2058 c_R` at lambda 0 and `-2316 + 2058 c_R` at lambda 1 on the g = 32 pairs, so the sign follows `c_R` only above 0.43 and never within `\|c_R\| <= 1` at lambda 1; the R_G slope scales with g; the static 3x3 record is changed; on the ansatz the R_G pair energy is a boundary-flux term with no power law. `K_lambda`: no long-range static exchange on `V4` (core overlap, exponent 6); an attractive Yukawa only with an assumed light mass |
| R14-D | `MAXWELL_CROSSING_EXISTS` or `NO_CLOCK_ACTIVE_BULK` | on `L_cert + c K_P^h` the exterior ticks and the fixed-omega functional is unbounded along the split (sealed behind eigenvalue collisions; a far-split pocket is born at omega 1.0e-3): the P250 object (an exterior at rest) does not exist with the certified potential. On `V4 + mu (m2 - m3)^2`, `mu >= 5.6e-4`, a first-order crossing exists at the plane level (audit): an exterior at rest at the diagonal minimum 0.157, a rotating interior off the fixed-sum line, tension 0.63 to 4.0, thin-wall radius 1100 down to 54 |
| R14-D2, R14-B', R14-B2 (overnight, 2026-09-05) | the wall constructed on the reduced 1D functional and cross-checked on a lattice slab; the boost-seeded descents; the `I6` corner under the descent | to be filled from the task record at the close |

### 11.4 Not computed (in addition to § 6)

The author's gates 1 to 3 of 2026-09-05 (an exact Noether clock charge of a cyclic action, the principal symbol and strong hyperbolicity after constraints, the constrained second variation of `E - omega J`); a Hamiltonian time integrator (the Floquet lifetime, wall formation); the other eight directions of the 4x4 field at the D and D2 phases (`m0`, `m1` and the off-diagonals frozen); the R14-B ladder beyond 3000 / 1000 / 600 iterations; the Lovelock class (dropped: every ghost-free epsilon-epsilon structure vanishes on planar profiles).

### 11.5 The adversarial audit record of the ladder

| Rung | Claims | CONFIRMED | QUALIFIED | REFUTED | Applied |
| --- | --- | --- | --- | --- | --- |
| R14-0 | 14 | 10 | 4 | 0 | the V4-type flat count (7), the H-adjoint order, the stencil dependence of the tail exponent |
| R14-A | 9 | 4 | 4 | 1 | the certificate's support reading, the norm ladder above 100, the eps artifact of the stored `K_P^h` forms, the negative certified boost inertia |
| R14-B | 8 | 4 | 3 | 1 | the volume-law reading (an h-blind descent), the iteration-100 start values, the unsampled boost sector |
| R14-C | 9 | 1 | 4 | 4 | the wrapper-stacking defect (heals rerun), the threshold in `c_R`, the pair law, the g = 32 cfg |
| R14-D | 8 | 5 | 1 | 2 | the box artifact of the fixed-omega scan, the plane-level first-order crossing |

## 12. Addendum (2026-09-06): R15, the floor witness, the tilt channel, and the author's projector object

The author's 2026-09-05 reply (record: [`m5_32_convo.md`](../tasks/m5_32_convo.md), the 16:54 UTC entry) accepted the R14 verdicts and pre-registered a new object on the degenerate vacuum, a floor witness for the certified kinetic term, and a tilt-channel claim; the R15 packet was frozen in [ledger § 6.4](m5_32_candidate_ledger.md) before any number and ran 2026-09-05 20:18 UTC to 2026-09-06 (every number and the five audits in the [task record](../tasks/m5_32_task_details.md), R15 section).

### 12.1 The objects (equations; E-orientation as in § 11.1)

| Object | Definition |
| --- | --- |
| the degenerate vacuum | `d = diag(g, 1, delta, delta)`, `N = M eta` with spectrum `(-g, 1, delta, delta)`; `V4^dd = W1 sum_{p=1..4} (tr N^p - C_p)^2`, `C_p = (-g)^p + 1 + 2 delta^p` |
| the split stiffness | `mu (lambda_2 - lambda_3)^2 = mu (s^2 - 4 p)`, `s = tr N - lambda_g - lambda_1`, `p = det N / (lambda_g lambda_1)`, `lambda_g`, `lambda_1` the two isolated eigenvalues (read per cell, Newton-polished on the characteristic polynomial so the map is holomorphic and the complex-step gate is exact) |
| the projector | `P23 = I - P_g - P_1`, `P_g = (N - lambda_1)(N^2 - s N + p) / [(lambda_g - lambda_1)(lambda_g^2 - s lambda_g + p)]`, `P_1` likewise; equals the author's `(N - g)(N - 1) / [(lambda_23 - g)(lambda_23 - 1)]` at `lambda_2 = lambda_3` and is a projector everywhere (the reading asked back to the author) |
| `K_P^23` | `E = (1/2) [ sum_i tr(Om_i^T eta Om_i eta) + omega^2 tr(Om_0^T eta Om_0 eta) ]`, `Om_mu = P23 A_mu eta P23`; THEOREM: `tr(Om^T H Om H^-1) = tr(Om^T eta Om eta)` on the projected block for `H = eta + 2 (eta u)(eta u)^T`, because `P23 u = 0` |
| `L_P` (our reading of the author's object) | `E_stat = E_u + V4^dd + mu SPLIT + c_P K_P^23`, descents under the certified `-4 I1` (`E_u`), `-4 I1^h` read on the end fields (`E = +4 x` the Lagrangian read at omega 0) |
| the floor witness (jets) | `M = L_a(chi) R_12(psi) D R_12^T L_a^T` (twist inside) or `R_12 L_a D L_a^T R_12^T` (after); `F_st = [b d_chi M, k d_psi M]_eta`; `U_G = 4 <F_st, F_st>_G = b^2 k^2 c_G` |
| the tilt channel | `M = R_23(omega t) R_12(theta(t, z)) D_s R_12^T R_23^T`, `D_s = diag(g, 1, delta + s, delta - s)`; `L_2 = alpha theta_t^2 + gamma theta_z^2 + eps theta^2`; regulator `w [tr(A_0 G A_0 G) - tr(A_z G A_z G)]` |
| the reduced planar functional | `F = int dz { (c/2)(m2'^2 + m3'^2) + V4^dd + mu s^2 - omega^2 [c s^2 + 8 s^2 s'^2] }` on `diag(g, 1, m2(z), m3(z))`, `s = m2 - m3`; `V_eff = V4^dd + (mu - omega^2 c) s^2` |
| the fixed-J functional | `E_J = E_stat + J^2 / (4 kin_tot)`, `kin_tot = kin_I1 + c_P kin_KP23`, `a0 = a0_local(M)` refreshed each step and frozen in the gradient |

### 12.2 Equation-to-code map (additions)

| Equation | Code |
| --- | --- |
| the trace targets, the split, the projector, `K_P^23` energy and exact gradient, `L_P`, the fixed-J FIRE, the 19 selftests | [`m5_32_r15_common.py`](../scripts/m5_32_r15_common.py): `cp_dd`, `spectrum_parts`, `projectors`, `kp23_cells`, `kp23_energy_grad`, `split_cells`, `split_energy_grad`, `lp_parts`, `lp_grad`, `lp_kin_grad`, `fire_lp`, `i1h_static`, `selftest` |
| the floor-witness jets and the tilt channel | [`m5_32_r15_vh_symbolic.py`](../scripts/m5_32_r15_vh_symbolic.py): `va_mode`, `h_mode` (`taylor2`) |
| the witness on the lattice | [`m5_32_r15_vb_lattice.py`](../scripts/m5_32_r15_vb_lattice.py): `boost_field`, `twist_field`, `run_grid` |
| the Hessian, the relaxations, the reads, the calibrated verdict | [`m5_32_r15_m_hedgehog.py`](../scripts/m5_32_r15_m_hedgehog.py): `hess_mode`, `relax_mode`, `reads`, `static_density`, `collect_mode` |
| the tails | [`m5_32_r15_p2_tail.py`](../scripts/m5_32_r15_p2_tail.py) |
| the reduced functional, the theorem check, the onset, the diagonal-sector wall, the slab check | [`m5_32_r15_p3_wall.py`](../scripts/m5_32_r15_p3_wall.py): `veff`, `F_reduced`, `theorem_check`, `onset_scan`, `ising_wall`, `slab_check` |
| fixed J and the stationarity test | [`m5_32_r15_p4_fixedj.py`](../scripts/m5_32_r15_p4_fixedj.py): `main`, `stationarity`, `verdict` |

### 12.3 Results, each with its pre-registered gate

| Rung | Gate | Audited outcome |
| --- | --- | --- |
| R15-V | V1 sign eta negative growing `k^2`, V2 sign h positive, V3 twist-after hides, V4 the symbolic coefficient | V1, V2 CONFIRMED (jets: `c_eta = -8 (delta - 1)^2 (g + d_a)^2 = -c_h`, rapidity-independent; lattice n64: `-515 / -1739 / -3583` against `+557 / +1835 / +4002`); V3 QUALIFIED (both positive, unequal, the jet-level equality only at rapidity 0); V4 the author's coefficient is the large-g leading form at ratio exactly 4; on the relaxed hedgehog both forms go negative; the lattice numbers are grid-divergent through the dressing's origin |
| R15-H | H1 no `theta_t^2` from curvature terms, H2 the `omega^2 k^2 theta^2` coefficient, H3 the hyperbolicity inequality, H4 `K_P^23` blind to the (1,2) sheet | all CONFIRMED exactly: `gamma(-4 I1) = 32 omega^2 s^2 (delta + s - 1)^2`, `alpha` only from the regulator, hyperbolic iff `w > 16 omega^2 s^2`, the static `K_P^23` exactly zero on any (1,2) twist sheet (a free direction: no fixed-J minimizer by the R13-W theorem) |
| R15-M | ADMISSIBLE / NOT_LOCALIZED / RUNAWAY | ADMISSIBLE on all eight (calibrated rule; the pre-registered 0.8 fraction fails the certified reference itself); a finite-energy hedgehog with a `1/R` tail, the pair staying degenerate, the exterior the seed's; the Hessian null counts as predicted (7 then 5, split stiffness `4 mu`) |
| R15-P-ii | L-exponent 0 (finite) against 1.34 | TAIL_FINITE: `K_P^23 ~ r^-4.1`, exponent 0.08 to 0.11 (the seed's `1/L`) |
| R15-P-iii | CONTINUOUS_ONSET at `omega_c^2 = mu / (c kappa_P)` or FIRST_ORDER_CROSSING | CONTINUOUS_ONSET on all nine points, a theorem (`V4^dd >= 0`); no coexistence wall; the Ising wall a Goldstone saddle (audit); the decay length `(1/2) sqrt(c/mu)` is not what the hedgehog's split shows (2.1 to 2.4 regardless of `c_P`) |
| R15-P-iv | PERIODIC_ORBIT_EXISTS / CANDIDATE_REFUTED / BLIND_BY_THEOREM | CANDIDATE_REFUTED (no stationary state): the descent inflates the split in the innermost cells to buy inertia (`E_J` 3.7e5 to 89 in 600 iterations, the fixed-J term 43 of 89, the kinetic density lattice-scale) and pins itself on the `lambda_1 = lambda_3` eigenvalue crossing, the branch cut of the ordered-label `P23` and of `a0_local` (the audit: the finite-difference curvature is a kink, 90 percent of the reported inertia a labeling artifact); n48 L72 replicates it (`E_J` 98.72, six cells on the crossing at r 1.3, gap 3.6e-6) |

### 12.4 Not computed (in addition to §§ 6 and 11.4)

A fixed-J descent under `-4 I1^h` (no exact gradient in the registry); a label-free fixed-J functional (a `P23` defined off the crossing, the author's call); the M-b descents to stationarity and n64 L96; the P-iii functional with the off-diagonal (2,3) entry free beyond the audit's Goldstone identification; the diagonal-entry reading of the split term (the author's choice is asked).

### 12.5 The adversarial audit record of the ladder

| Rung | Claims | CONFIRMED | QUALIFIED | REFUTED | Applied |
| --- | --- | --- | --- | --- | --- |
| R15-V | 7 | 5 | 2 | 0 | the hedgehog cross terms relative to the hedgehog's own `E_u`; the grid divergence of the dressed baseline; the stencil attenuation of the k growth; the h-column normalization mismatch |
| R15-H | 5 | 5 | 0 | 0 | the regulator's `s = 0` term (the Coriolis partner); the (2,3) sheet not free for `K_P^23` |
| R15-M | 6 | 2 | 4 | 0 | the tail is the seed's; the certified reference's z-axis line; the split identities; the gradient level; the mu dependence at `c_P 0` |
| R15-P-iii | 5 | 3 | 2 | 0 | the 32-vs-16 criterion; the Ising wall a Goldstone saddle; the decay length `(1/2) sqrt(c/mu)`; the two readings of the split term |
| R15-P-iv | 5 | 3 | 2 | 0 | the branch-cut pinning replaces the stiff-valley reading; the split sits in six cells, not a shell; the label-free inertia |

## 13. Addendum (2026-09-06): R16-0, the author's 2026-09-06 claims verified on our stack

The four comments of 2026-09-06 (the coordination-thread record in the task folder) answered the two R15 definitions, diagnosed the two failed R15 predictions by Coleman's condition, corrected one sentence of our post, and proposed the local-circle object v4 with a pre-registered ladder. R16-0 is the verification rung: every checkable claim through our own scripts before any instrument is built on the new object. Nothing was relaxed.

### 13.1 The objects (equations; E-orientation as in § 11.1)

| Object | Equation |
| --- | --- |
| the two H-adjoint completions | `F^eta_mn = A_m eta A_n - A_n eta A_m`, `F^G_mn = A_m G A_n - A_n G A_m`, `G = eta + 2 (eta u)(eta u)^T`; `I_norm = sum_{m<n} eta^m eta^n tr(G F^eta G F^eta^T)` (the registry's `I1_h`), `I_rebuild = sum_{m<n} eta^m eta^n tr(G F^G G F^G^T)`; static energies `E = +4 x` the read |
| the local circle | `T_alpha M = R_n(alpha / 2) M R_n(alpha / 2)^T`, `R_n` the rotation about the local director `n` (the eigenvector of the isolated eigenvalue 1 of `N`); on the sheet `M = R12(psi) R23(phi) D_s R23^T R12^T` it is `phi -> phi + alpha / 2` |
| the split block | `B = P23 N P23 - (1/2) tr(P23 N) P23`, `rho^2 = (1/2) tr B^2` (`= s^2` on the diagonal sheet); the eigenvalue metric `(d lambda_+)^2 + (d lambda_-)^2 = 2 (a da + b db)^2 / (a^2 + b^2)` for `B = [[a, b], [b, -a]]` |
| the reduced line | `E_J = int 4 pi r^2 [(c / 2) s'^2 + V(s)] dr + J^2 / (4 int 4 pi r^2 c s^2 dr)`, `c` the `K_P^23` inertia of the uniform split (`4 c_P` per `s^2`); Coleman: a crossing needs an interior minimum of `V / s^2`; the sextic `V = mu s^2 - nu s^4 + kappa s^6` has it at `s*^2 = nu / (2 kappa)`, value `mu - nu^2 / (4 kappa)` |
| the weighted condition | `C(s) = c s^2 W(s)`, `W = [w(delta + s) w(delta - s)]^2`, the rational `w = f(lambda) / f(delta)`, `f(x) = (x - g)(x - 1) / ((x - g)^2 + (x - 1)^2)` |
| biaxiality | `beta^2 = 1 - 6 (tr Q^3)^2 / (tr Q^2)^3`, `Q` the traceless spatial triple; the `beta^2`-weighted quadrupole `Q_ij = sum beta^2 (x_i x_j / r^2 - delta_ij / 3) / sum beta^2` (a great-circle ring: `(-1/6, 1/12, 1/12)`) |
| the spin-weight-2 content | `zeta = S_ee - S_ff + 2 i S_ef` in the oriented transverse frame, `c_m = (4 pi / N) sum zeta conj(2Y_2m)`, `P_m = abs(c_m)^2`, `<m> = sum m P_m / sum P_m`; the rotation tangent `[G_z, M] - (x d_y - y d_x) M` |
| the chiral pseudo-scalar | `tau = eps_ijk S_il d_j S_kl`, `T2 = sum tau^2 h^3`; on a uniaxial texture `tau = (1 - delta)^2 n . (curl n)` |

### 13.2 Equation-to-code map (additions)

| Equation | Code |
| --- | --- |
| the completions on jets, the circle on the sheet and on point jets, the lattice invariance defects, the sheet inertias, the tilt substitution, the boost sheets, the eigenvalue metric | [`m5_32_r16_0_symbolic.py`](../scripts/m5_32_r16_0_symbolic.py): `c1`, `c4` (`numzero`, `jets`, `dens`, the lattice block), `c5`, `c6` |
| the sheet `V4^dd`, the inertia coefficient, `V / s^2`, the uniform-limit ladder, the thin-wall estimate, the 1D profiles with the analytic gradient, the weighted condition | [`m5_32_r16_0_reduced.py`](../scripts/m5_32_r16_0_reduced.py): `v4dd`, `a_coef`, `main` (`EJ_grad`, `w_rational`) |
| the completions per cell, the witness and hedgehog reads, the biaxiality and quadrupole, the tangent, the `2Y_2m` builder and shell decomposition, `tau` | [`m5_32_r16_0_fields.py`](../scripts/m5_32_r16_0_fields.py): `completions_density`, `both`, `c1`, `c3`, `spatial_triple`, `biaxiality`, `c7`, `sY2`, `shell_decomp`, `frame_zeta`, `c8`, `c9` |

### 13.3 Results, each against the author's statement

| Claim | The author | Ours |
| --- | --- | --- |
| C1 | two completions, counterexample; 2 to 6 percent apart on the witness | counterexample exact; equal on the witness JET; 24 to 46 percent apart on the witness LATTICE; the author's h column is `I_rebuild / 4`, ours `I_norm` |
| C2 | Coleman: no crossing for `V4 + mu s^2`, the sextic crosses at 9e-3 and `s* = 0.2236`; the rational weight kills it (0.01117), the plateau restores it | all numbers reproduced; `omega_c^2 = mu / (4 c_P)` for `U = mu rho^2`; 0.01117 is the value at `s*`, the minimum is `mu`; the Q-ball needs J above 2.6e4, radius above 69 |
| C3 | `E_h >= 0` pointwise, the cross terms mean non-stationarity | CONFIRMED for both completions; our floor sentence retracted; `I_rebuild` keeps the sign flip on the hedgehog (`+195 / +629 / +1500`), `I_norm` does not |
| C4 | the local circle is an exact symmetry of `L_v4` (potential, projected `K_P` invariant; `I1` averaged) | potential and `K_P^23` invariant (exact on jets, `O(h^2)` on the lattice); `I1` and BOTH completions not; the regulator `E2` NOT invariant either (not averaged in v4 as written) |
| C5 | the sheet inertias, `K_P` blind on the sheet, `rho^2 E2` stiffness, `c_s > 16 omega^2`, the boost-sheet law | all exact; the boost-sheet law reads `(g + delta +- s)^2` |
| C6 | the eigenvalue metric discontinuous, `rho^2` smooth | exact |
| C7 | the P-iv end state is the Landau-de Gennes biaxial-ring core | oblate uniaxial center (= the R15 crossing), a `beta^2 = 1` ring of radius 2.9 with the great-circle quadrupole signature, axis on the lattice body diagonal, identical in cell units on both boxes: lattice-scale until refined |
| C8 | `J_z = 0` on axisymmetric configurations; report the `2Y_2m` content | tangent `O(h^2)` on a smooth axisymmetric split field; builder gates pass; the P-iv split is achiral (`<m> = 0.00`, `P_m` symmetric) |
| C9 | `T2` = 2e-29 / 1e-2 / 91 / 0 | 1.7e-29 / 2.9e-2 / 630 / 0; the identity holds |

### 13.4 Not computed

The h-refinement of the P-iv end state; any relaxation under `I_rebuild` (no gradient); the circle-averaged instrument and the four lattice stages (the second go); the atomic gates, the Q-hopfion, the bend theorem beyond C9, the Longa-Trebin LP.

### 13.5 The adversarial audit record

| Rung | Claims | CONFIRMED | QUALIFIED | REFUTED | Applied |
| --- | --- | --- | --- | --- | --- |
| R16-0 | 17 | 13 | 4 | 0 | the completion ratios not converged in h, the author's-column reading an interpretation; the sextic threshold is a localized-profile statement; the two prolate body-diagonal core cells; the l = 2 fraction and the twist `T2` counting |

Audit: [`m5_32_r16_0_audit.py`](../scripts/m5_32_r16_0_audit.py), [`m5_32_r16_0_audit.json`](../data/m5_32_r16_0_audit.json) (own sympy, own jets by the analytic chain rule, own lattice fields at three resolutions, own reduced-line minimizer with a pinned-edge variant, own frame and least-squares projection).
