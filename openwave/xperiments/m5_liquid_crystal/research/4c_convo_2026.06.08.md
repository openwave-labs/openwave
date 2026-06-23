# Duda thread, June 2026 (4×4 clarification → δ/g calibration → boost/EM-GEM correction)

Working record of the multi-day exchange with Jarek Duda (Manfried Faber added mid-thread) that followed the M5.8 results report (`99_summary_report.md`). Captures the correspondence and the runs it triggered, so the back-and-forth is preserved in one place alongside the findings-focused entry in `0b_M5_roadmap.md` ("DUDA 2026-06-08 FOLLOW-UP" + its 2026-06-09 addendum).

**Status:** the δ/g calibration question is closed at the seed level; gravity-domination retracted (Duda was right); EM/GEM split delivered. Round 3 (2026-06-09 afternoon) converged on the forward program: Duda's `F_μν` decomposition matches our split symbol-for-number, and his 3×3 fixed-clock electron suggestion became the **ElectronID project** (μ + J extraction, roadmap "ELECTRON-ID PROJECT"). Open: the δ (quantum-phase) sector is dynamical and not yet weighed; the absolute unit calibration (Coulomb + LdG-to-rest-energy) is the real axis.

**Cross-refs:** [`0b_M5_roadmap.md`](0b_M5_roadmap.md) (DUDA FOLLOW-UP + 06-09 addendum, NG-12), [`99_summary_report.md`](99_summary_report.md), `sandbox_vn/m5_8_2q_delta_scaling.py` (energy sweep + Phase D/E boost/EM-GEM split), `sandbox_vn/m5_8_2q_omega.py` (ω(δ) scaffold, deferred), `../theory/faber_universe_2025.pdf`.

---

## 1. The 4×4 clarification

Duda read the report and first took it for the 1+1D toy ("the 1+1D is only toy-model ... the real clock rather requires full 3+1D with 4×4 tensor field, including gravity"). Correction sent: the report IS the full 3+1D test, and the field is already the 4×4 tensor `M = O D Oᵀ`, `D = diag(1, δ, 0, g)`, time as the 4th index, Minkowski signature, the clock fuel coming from the `(α,3)` block (Euclidean twin kills it). What is NOT yet there: dynamical gravity. The 4th axis is the `g` slot, currently a constant background dressed by a flat Lorentz boost, not a dynamical metric.

## 2. Duda 2026-06-08: the QED dictionary + directions

He attached the QED Lagrangian as a decoder ring mapping onto our `D`:

| QED term | coefficient | our `D` axis | his prescription |
| --- | --- | --- | --- |
| Dirac kinetic `ψ̄γ∂ψ` (quantum phase) | `ℏc` ("tiny") | `δ` (minor), enters as `δ²` | `δ ~ 10⁻¹⁰`, not 0.3 |
| mass `mc²ψ̄ψ` (→ gravity) | `mc²` | `g` (time axis) | `g ~ 1/δ ~ 10¹⁰`, ideally `g·δ = 1` |
| gauge `F²/4` (EM) | `1/4` | the `1` (major) | order unity |

Plus: our quantities are dimensionless, so fix units by comparing to Coulomb or the clock; tune the LdG potential to the particle rest energies. He found the `5.5×10¹⁹ rad/s` clock "surprisingly close, especially for using much too large δ."

## 3. Our δ/g calibration runs (`2q` energy sweep)

| Finding | Result |
| --- | --- |
| `g=1/δ` (his hierarchy) | H_static diverges `∝ δ^(−6.8)`, so the literal physical scale is numerically impossible (matches his "neglect gravity, put clock by hand") |
| δ-sweep, gravity decoupled (g fixed) | H_static is δ-flat (16.74→20.64 across δ 0.3→0.001) |
| clock ratio | with the N-6a ZBW law (`ω ∝ H_rest`), `R = ω·δ/(2H) ≈ 0.033·δ`, so the δ knob does NOT calibrate the clock; smaller δ moves it away from 1 |
| ⚠️ initial misread | first reported the rest energy as "gravity-axis dominated" (`∝ g⁸`), corrected in §5 |

## 4. Duda 2026-06-09: the correction

| Point | His claim |
| --- | --- |
| Gravity not dominant | for particles the boost perturbation of the time axis is tiny (macroscopic tilt needs a black hole), so he disagrees that the rest energy is gravity-dominated |
| EM dominates | the rest energy is dominated by the dynamics of the length-1 axis (Faber's unit vector), which is the EM sector |
| δ is dynamical | the quantum-phase contribution grows through the fast (~10²¹ Hz) clock changes, not the static seed |
| EM/GEM ratio | the ratio of "curvature of rotations (EM)" vs "curvature of boosts (GEM)" is surprisingly difficult (maybe Manfried has ideas) |
| Mass question | activating the clock should slightly REDUCE the mass; how much would the electron's mass grow if its Zitterbewegung were stopped? |

## 5. Our 2026-06-09 runs (`2q` Phase D/E, the EM/GEM split)

Split the quadratic seed energy into Duda's two sectors via the signed η-blocks: `SP_PAIRS` spatial = EM (curvature of rotations, η-positive), `TM_PAIRS` time-mixing = GEM (curvature of boosts, η-negative). `EM + GEM = H_quad` exactly.

| Finding | Result |
| --- | --- |
| boost = 0 | GEM is exactly 0: gravity contributes nothing without the time-axis tilt. Duda is right, "gravity-dominated" retracted |
| physical knob | gravity enters only as the boost tilt `b·g`; `GEM ∝ (b·g)²` in the small-tilt regime (raising `g` at fixed boost was an unphysically large tilt) |
| EM/GEM ratio | NOT a constant: 210:1 at a physical small boost (`b=0.01`), 2:1 at the clock dressing (`b=0.13`), scaling `1/(b·g)²`. EM dominates the rest energy in every physical case |
| GEM sign | negative (the Minkowski clock-fuel block), so the boost/clock REDUCES the rest energy by `\|GEM\|` |
| mass-reduction | `\|GEM\|/EM ∝ (b·g)²`: ~0.5% at a physical boost, up to ~50% at the large clock dressing. Stopping the Zitterbewegung removes the negative GEM and the mass rises. This is the concrete answer to his "how much" |

## 6. The corrected physical picture

| Sector | `D` axis | role | weight in the rest energy |
| --- | --- | --- | --- |
| EM (curvature of rotations) | the `1` major axis (Faber unit vector) | electromagnetic | DOMINANT (~210:1 over GEM at a physical boost) |
| GEM (curvature of boosts) | the `g` time axis, via the boost tilt `b·g` | gravity / clock fuel | tiny and NEGATIVE; zero without the boost |
| quantum phase | the `δ` minor axis | the de Broglie clock | small in the STATIC seed, but its real weight is DYNAMICAL (the fast ~10²¹ Hz clock), not yet measured |

## 7. Open threads (after round 2)

| Thread | State |
| --- | --- |
| δ (quantum-phase) sector weighed dynamically | open; the static split does not capture it (needs the ω-dynamics on the settled state, NG-12) |
| EM/GEM ratio | delivered as a function of the boost; Duda floated Manfried for the deeper Lagrangian-level ratio |
| absolute unit calibration | the real calibration axis: fix units via Coulomb + tune LdG to rest energy (NG-1/NG-3), not the δ knob |
| repo | scripts, data, and docs are public at github.com/openwave-labs/openwave |

## 8. Duda 2026-06-09 (round 3): the `F_μν` decomposition + the fixed-clock electron

His third email, with a Mathematica-derived `F_μν` figure as the attachment, consolidates the picture and hands over a practical recipe.

| Point | His text |
| --- | --- |
| Parameters to fit | "delta for QM, g for gravity, and probably 1 or 2 parameters of potential", to be found by agreement with physics |
| EM is primary | "the most important is EM, dynamics of length 1 axis, as n vector in Faber's articles" |
| QM + GEM tiny | both should be relatively tiny contributions, "crucial especially for clock propulsion" |
| Clock propulsion terms | the marked NEGATIVE contribution terms coupling spatial rotation `Γ` with boost `Γ̃` (red-boxed in his figure) |
| Electron picture | hedgehog of the unitary axis → effective Coulomb + electron mass as rest energy; then "evolution of Γ¹ low energy twists of the clock, together with gravitational mass" |
| Practical suggestion | "maybe try getting such electron with 3x3 field and fixed clock, to get proper magnetic dipole moment and angular momentum of electron" |

**The attached figure decoded.** `M̄_μ` is the connection at `g ≫ 1 ≫ δ > 0`: time-space entries carry `g·Γ̃ᵃ` (boost curvature), spatial entries carry `Γ³, Γ², δΓ¹` (rotation curvature; `δΓ¹` is the QM twist). The commutator `F_μν` then splits into color-coded sectors:

| His sector (figure) | Curvature content | Our split |
| --- | --- | --- |
| EM: tilt-tilt | `R¹ + g²R̃¹` | inside our spatial/EM block |
| QM: tilt-twist | `δR² − δ²R̃²` | ALSO inside our spatial block (we lumped EM+QM, see caveat) |
| gravity: boosts | the `g²` time-space block | our GEM (time-mixing) block |
| clock propulsion | the red-boxed NEGATIVE rotation×boost couplings (`−R¹ − g²R̃¹`, `−δR³ − δ²R̃³`) | the negative sign of our GEM block, measured `∝ (b·g)²` |

**Alignment with our numbers:** our Phase E found the time-mixing block negative (the clock fuel) with EM dominating, which is exactly his decomposition with magnitudes attached. **Caveat:** our split was 2-way (spatial vs time-mixing), so our "EM" number lumps his EM (tilt-tilt) with his QM (tilt-twist); separating them is a finer index split within the spatial block (the ElectronID Phase B below).

## 9. The forward program: ELECTRON-ID (agreed 2026-06-09)

Why his fixed-clock suggestion is the right route: the model has mass (rest energy) and charge (effective Coulomb) but not yet μ (magnetic dipole moment, target one Bohr magneton / g≈2) or J (spin, target ℏ/2), the remaining two electron identifiers. Our `2p` spin readout on the DYNAMICAL 4D clock came out J-neutral (`J < 1e-4`, swamped by 24³ box torque), so the dynamical route is washed out. A static 3×3 hedgehog with the clock PINNED at definite phase/winding carries a definite circulating current (→ integrate μ) and a definite field angular momentum (→ integrate J), with the divergent boost sector dropped entirely.

| Phase | What | Status |
| --- | --- | --- |
| B: 3-way sector split | refine Phase E to his exact figure: EM (tilt-tilt) vs QM (tilt-twist) vs GEM (boosts), separating the spatial block | ✅ 2026-06-10: EM 16.34 / QM 2.23 / GEM −9.37 at clock dressing (his hierarchy confirmed); label correction: EM = the (1,2) component (R=Γ×Γ points along the major generator) |
| C: the 3×3 fixed-clock electron | static hedgehog + pinned clock; integrate μ and J; compare to Bohr magneton (g≈2) and ℏ/2 | ⚠️ 2026-06-10: μ exists only via the TILT/precession channel (0.221 linear response); the twist clock (his Γ¹) is EM-silent; orbital J = 0 structurally (E∥B dyon, no Thomson term); spin = the Noether clock charge L_int (L/Q=ω family, 61.6 φ-flat); g ≈ 2 needs the Coulomb e_scale unit fix first (his "fix units by Coulomb" is REQUIRED, not optional). Full record: roadmap § EID results |

Roadmap home: `0b_M5_roadmap.md` "ELECTRON-ID PROJECT". Connects to NG-8 (the magnetic-dipole placeholder this would make real) and supersedes the dynamical-route μ/J attempts.

## 10. Duda 2026-06-10 (round 4): the coverage-matrix review

Duda replied to `MODELS.md` (now cc'ing Nielsen, Watford, Werbos, Faber, Hal, Vidvuds + the list) with row-by-row commentary on the M5 column, 4 mechanism images attached. All comments incorporated into the matrix cells; two became immediate runs.

| His comment | Action taken |
| --- | --- |
| Spin-½: no belt trick needed, the apolar ellipsoid returns at π | **RUN + flipped to ✅**: `m5_8_2s_spin_half_apolar.py` verifies M(φ+π) = M(φ) machine-exact (1e-16, both clock planes) while the frame needs 2π = the double cover; same factor 2 as G7 |
| Annihilation natural "starting with sine-Gordon" | **RUN**: `sine_gordon_annihilation.py`, the principle trail: integrable pass-through vs damped capture → breather → vacuum (E 16.3 → 0.00, Q = 0 throughout) |
| DM as thermal noise of non-EM field sectors (with Faber) | cell updated (CMBR analog for weak/strong/grav DOF) |
| Leptons = 3 energy minima for elementary charge; hard parts = Higgs-like regularization + oscillation (gravity propulsion) | cell updated |
| Neutrinos: huge stable quark-string loops (Nature s41586-024-08479-6); oscillation = SO(3) rotation (PLB S0370269326000730); "derivable from LdGS" | cell updated + refs; a parameter-finding target he hands us |
| Cornell linear term from fractional-charge quantization violation (~1 GeV/fm); string-breaking on stretch | quarks + strong cells updated |
| μ + J from the clock's continuous SO(2) twisting | cell updated; consistent with the EID tilt-channel finding |
| Baryons = simplest knots, vortex loop around vortex (Nature Physics s41567-025-03107-0); mesons = Möbius-twisted loops (strangeness = twist) | cells updated + ref |
| Magnetic force needs Γ_i AND Γ_0 (the clock) | cell updated; the EID run computes exactly F_0i from Γ_0, `2r` added to the cell |
| Gravity "appears naturally going from 3×3 to 4×4 adding boosts" | cell updated: direct endorsement of the implemented M5.8.1 route |
| Orbital quantization via pilot waves (hydrodynamic quantum analogs, Perrard ncomms4219) | cell updated + ref |
| Weak: requires the beta-decay study | unchanged (matches the cell) |

**Runnable leads he handed us (future):** neutrino-loop parameters from LdGS (the SO(3) oscillation paper), the fractional-charge pair energy-vs-distance scan (Cornell σ), knot configurations for baryons (with "needs practical approximations" his own caveat).

## 11. Open threads (current)

| Thread | State |
| --- | --- |
| ElectronID B + C | ✅ run 2026-06-10 (`sandbox_vn/m5_8_2r_electron_id.py`); report-back to the thread pending; residuals (box convergence, e_scale) folded into NG-1/NG-3 |
| δ (quantum-phase) sector weighed dynamically | open (NG-12); Phase B's tilt-twist split gives its STATIC weight as a first step |
| absolute unit calibration | Coulomb-units + LdG-to-rest-energy (NG-1/NG-3); Duda round 3 reconfirms "1 or 2 parameters of potential" to fit |
| EM/GEM Lagrangian-level ratio | Duda floated Manfried; our boost-dependent measurement stands as input |
