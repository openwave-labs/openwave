# M5 / Liquid-Crystal — Question Tracker + Hardest Pieces

**Purpose:** single source of truth for **(a) what physical reality the M5 substrate reproduces and how (MODEL EMERGENCE)**, (b) open research questions on M5, (c) resolution chronology, and (d) the long-running **hardest-pieces** status board. Mirrors the M6 tracker pattern. Update this doc when a question opens, gets sent to the group (Duda / Close / Yee), gets answered, demoted, or when new emergent physics is verified. Source material consolidated here from [`1b_topological_defect.md`](1b_topological_defect.md) § STRATEGIC MAPPING + § OUTGOING-WAVE L+T + § Empirical validation, and [`4a_convo_2026.05.12.md`](4a_convo_2026.05.12.md) § 7 Force unification + § 8 Eigenvalue mapping — so this tracker reads stand-alone. The deeper *explanations* (what is a defect, what is a twist, derivation walk-throughs) stay in 1b / 4a / 5a; this doc carries only the emergent-reality catalog + numerical mechanism + status.

**Sister docs:**

| Doc | Purpose |
| --- | --- |
| [`0b_M5_roadmap.md`](0b_M5_roadmap.md) | Phase sequence (M5.0 → M5.9) + current state |
| [`3b_lagrangian_roadblocks.md`](3b_lagrangian_roadblocks.md) | M5.2 negative-result diagnosis + original group questions |
| [`4a_convo_2026.05.12.md`](4a_convo_2026.05.12.md) | Duda thread study — substrate decision, eigenvalue map, slides |
| [`1a_lagrangian_framework.md`](1a_lagrangian_framework.md) | Framework + the full email thread history |
| [`5a_lagrangian_evolution.md § 10e`](5a_lagrangian_evolution.md) | Canonical build-spec: substrate + action + construction recipe + verification gates |

**Last updated:** 2026-06-08 (**THE FULL ZBW PROGRAM LANDED 2026-06-07, N-1…N-6e** — M5.8 sandbox arc complete). Headlines, all machine-checked: the quadratic action has no 3+1D clock saturation (dt-invariant runaway); the signed quartic `u+βu²` saturates it; the saturated state is a self-starting, frequency-RIGID, quasi-periodic breather whose ω₁ is an ATTRACTOR (start- AND dressing-energy-independent — it belongs to the core); it HOLDS resolution-robustly (N-2: the M5.7 dispersal REVERSES under the quartic — defect structure decays slower at 48³ than 24³); it classifies as a MOLTEN CLOCK (low-dim chaos on a persistent comb) that REGULARIZES toward a near-regular cold ground state; the invariant matrix is COMPLETE (signed-u winner / Skyrme damps / A4 pinned / covariant deferred); matter+light share a maximal radial cone ceiling; the first ABSOLUTE ω = 5.5×10¹⁹ rad/s sits ~28× below 2m_ec²/ℏ (a STRUCTURAL gap given rigidity); no intrinsic spin J above the box-torque floor (the clock is J-neutral). The retired claim of record: the earlier "strictly periodic ω₀=0.262 comb" was an FFT-window artifact, caught + retired. M5 now PAUSES for the Duda RESULTS report. Earlier: 2026-06-05 (2c-1 + Option B spike — the full self-consistent 4D evolution is stable under the constrained spectral-projection integrator).

---

## Empirical validation — key findings → runnable reproductions

The model's verified emergent physics, each with a runnable reproduction script — on the production 4×4 engine where the finding is engine-level (charge, Coulomb, Eq.18, LdG), standalone numpy/sympy where the finding is analytic (Maxwell routes, KG-geometric, Faber mass), and Duda's own 1+1D toy for the clock. **All re-verified 2026-06-05** after the M5.8.1 4×4 promotion + ψ-retire; the engine-level checks reproduced their original 3×3 numbers, empirically confirming the promotion is physics-preserving:

| Key finding | Result | Runnable reproduction |
| --- | --- | --- |
| Topological charge is INTEGER + ADDITIVE (Q = director winding) | ✅ Q=±0.996 single hedgehogs; pair: ±1 around each core, 0.000 on the enclosing sphere | `m5_8_1_topo_charge.py` |
| Coulomb: attractive 1/d force between opposite defects | ✅ R²=0.9781, b<0 — matches the M5.1 reference (0.978) digit-for-digit | `m5_4_coulomb_matrix.py` |
| Coulomb — Duda's page-18 Mathematica cross-validation | ✅ R²=0.9959, attractive | `m5_4_coulomb_page18.py` |
| E = ∇·n̂ / B = ∇×n̂ → Maxwell (hydro↔EM dictionary) | ✅ PASS — ∇·B=0, Gauss charge identity, Faraday, Lorentz force | `m5_6_4a_hydro_em.py` |
| EM-from-tilts: Faber R=Γ×Γ → abelian Coulomb far-field | ✅ PASS — closed field strength, ‖R‖~1/r², running-coupling onset at r₀ | `m5_6_4b_faber_curvature_em.py` |
| KG mass is GEOMETRIC (biaxial twist, not an added V) | ✅ PASS — explicit mass cancels exactly; mass from the hedgehog connection Â | `m5_6_1_kg_operator_check.py` |
| The biaxial hedgehog SOURCES its own twist + restoring mass (`C_μν≠0` — the M5.8 clock seed) | ✅ ‖C‖~1/r² mass source confirmed; twist sourced by the background curvature + restoring force — stable, energy-conserving | `m5_6_2a_biaxial_hedgehog.py` + `m5_6_2b_biaxial_evolution.py` |
| Finite defect mass pinned by the core radius, E∝1/r₀ (Faber — the M5.9 lepton calibration handle) | ✅ E·r₀ constant (CV=0.0%); E₀=α_f·ℏc·π/(4r₀); r₀=2.2132 fm → 0.511 MeV e⁻; reproduced on Duda's matrix substrate | `m5_6_3a_faber_regularization.py` + `m5_6_3b_faber_on_M.py` |
| Eq.18 matrix evolution conserves energy | ✅ secular drift 2.15%→1.13%→0.03% as dt→0 (symplectic) | `m5_5_4_matrix_evolution_check.py` |
| LdG well confinement (V(M) Eq.13, b=0 amplitude well) | ✅ confines ~3.3× across k∈[0.5,25], no blow-up | `m5_6_5c_prod_scale.py` |
| Duda's 1+1D time-crystal kink (arXiv:2501.04036) — the clock | ✅ ω* = √(70/61) = 1.0712 reproduced EXACTLY; energy drift ~10⁻¹⁵; clock robust under +5% perturbation | `m5_8_0b_collective_clock.py` |
| Clock propulsion is SPONTANEOUS + bounded ("propelled by mass") | ✅ ω=0 is an unstable maximum → relaxation spins the clock up to ω*; every seed → ω* (no noise needed); E floor 2.1257 > 0 | `m5_8_0c0d_propulsion_robustness.py` |
| M5.7 resonance hunt: free defect disperses (expected — Derrick), DRIVEN defect sustains | ✅ free → dispersal (the empirical push to the 4D clock, M5.8); driven → bounded, frequency-selective (A,ω) excess, ~3× free at resonance | `m5_7_*.py` |
| 3+1D Minkowski clock FUEL on the hedgehog (Eq.42 ℋ, the M5.8 negative-energy mechanism) | ✅ fuel C(b)<0 once the time axis is boost-dressed, exactly 0 undressed (= the M5.7 null); Euclidean flip kills it (the signature IS the mechanism); core-localized; ω_M = 2·ω_clock exact (the 2mc²/ℏ seed) | `m5_8_2a_4d_hamiltonian.py` |
| Boost dressing LOWERS the defect's energy (GEM dip) — the dressed defect is the stable static ground state; the GLOBAL rigid clock mode is ruled out | ✅ dressed center E*=2.61 < bare 6.14, held dynamically (H drift 4.9e-8); ghost branch saddle-only (net global clock inertia vanishes at the window edges) ⇒ the physical clock = the core-localized twist (M5.8.2b-2) | `m5_8_2b_cc_clock.py` |
| The 4D clock is REAL at field level + the Minkowski signature COHERES it | ✅ core-localized twist on the dressed hedgehog: single coherent mode ω=5.86 (fft/zc agree 0.4%) vs Euclidean multi-mode (89% split); best core retention (0.66 Mink > 0.59 undressed > 0.46 Euclid); ghost geography mapped (stable region 77% at b*; fuel shell λ≈15.6/t linear runaway = the propulsion signature, constrained treatment = 2c) | `m5_8_2b2_field_clock.py` |
| The Minkowski signature SUSTAINS the clock at full nonlinearity (saturated regime) — the time-crystal ENGINE | ✅ kill-control twin (M5.8.2e, same seed+kick, 24k steps): Euclid clock DECAYS (1.4e-2 → 6.7e-3) while Minkowski+quartic clock GROWS (7.6e-3 → 2.4e-2). ⚠️ The "STRICTLY PERIODIC ω₀=0.262 + exact comb" half RETIRED 2026-06-07 (m5_8_2h W1: window artifact — the slow peak moves with FFT window length, the comb was bin arithmetic); the valid ω statement is the 2h row ↓ (quasi-periodic, ω₁ ≈ 1.1 + 2ω₁ attractor) | `m5_8_2e_invariant_matrix.py` (euclid mode); `sandbox_vn/m5_8_2h_omega_attractor.py` (ref mode = the W1 scrutiny) |
| The 2d/2e breathing state is NOT a global-clock radial-dressing state — the variational manifold is ANSATZ-SELECTED (Track C C1, the M6 BVP-transfer's first outcome) | ✅ NEGATIVE-DECISIVE (hard gates PASS): in the reduced `H_eq[b(r)] = V + βQ + p_Θ²/4K` (exact s-polynomial tabulation, machine-exact canary, every claim direct-audited) there is NO interior ghost minimum at ANY (β, p_Θ) — V+βQ rises MONOTONE 8.4 → 1.5×10¹⁰ along the ghost branch (the static u is positive-dominated, u_min = −0.034: the quartic taxes the very amplitude the ghost needs) while the fuel term p²/4K diverges only IN the onset layer ⇒ saddle-only — the 2b net-cancellation verdict SURVIVES the quartic, variationally. Byproducts: 2b-1's E*=2.606 refined to direct 2.93 (its Δb=0.1 spline artifact); the β=0 slope-fuel channel opens only at θ≳1.3 and is NOT net-extractable by radial profiles (the 2c FIELD runaway lives outside radial-coherent reductions) | `m5_8_2f_breathing_bvp.py` |
| ...and NOT a statically-twisted rotating state either — the static clock-phase twist is PURE COST (Track C C2-B; the m5_6_2b KG twist-mass, now variational) ⇒ **the breathing state is IRREDUCIBLY TIME-DEPENDENT: `ḃ ≠ 0` IS the bounce** | ✅ NEGATIVE-CLEAN (hard gates PASS): `W = O_hh·B(b(r))·R(Θ+ψ(r))` is exactly (b′,ψ′)-bilinear with ψ′ a FREE function and K twist-independent; the twist-fuel pocket is real but 5 orders too weak (`v₀₂ = −0.0031`/voxel, GEM-dip window θ∈[0.2,0.5] only); NO twisted state below the t≡0 control in any (β, p_Θ); the marginal ΔH=−0.002 candidate REFUTED by the direct twist-differential dial (surrogate dip −0.006 vs direct rise +0.41 — a near-cancelling 3-Gauss stack riding the stencil delta). Required + delivered the EXACT 9-pt phase average (the 2b 4-pt secular set is not shift-invariant under twist; t-canary 7-21% → 0.01%). C3 redefined: PERIODIC-ORBIT search in the reduced CC dynamics (ω free — the M6 eigenvalue transfer for a limit cycle) | `m5_8_2f2_localized_clock.py` |
| **THE UN-SITTABLE MINIMUM — the dressed ground state CANNOT sit still: spontaneity DERIVED (Track C C3)** + the reduced-CC program's boundary located | ✅ all gates PASS: the dressed static minimum (a*=0.1244) has `K_bb(a*) = −67.6 < 0` (DIRECT quadrature) with `U″ = +1723 > 0` — energetically minimal, kinetically GHOST ⇒ compulsory motion = G-2c-3's spontaneity at the CC level. Kinetic structure (table ≡ direct, 9-pt exact): `K_bΘ ≡ 0` (2b's −47.5 cross-term was 4-pt sampling residue — the reduced system is gyroscopically decoupled; 2b B5's "bounded libration" was stabilized by that artifact); `K_bb` ghost below b≈0.127, `K_ΘΘ` ghost above b≈0.25 — complementary windows. At anchored β the minimum is FULLY ghost-kinetic (𝐊 eigs −1140/−26/−1.5); the orbit exits through `det 𝕂 = 0` where any NET-inertia reduction is ill-posed ⇒ **containment is irreducibly many-mode** (the 2c-1 per-voxel eigenprojection's job). Handoff: the unkicked-seed 2d spontaneity test with the amplitude probes | `m5_8_2f3_breather_orbit.py` |
| **SPONTANEITY CONFIRMED AT FIELD LEVEL — the settled field with zero momentum MOVES (the field handoff executed)** | ✅ dt-converged: S4 damps the field to a quasi-settled discrete config (T drained 64×); S5 restarts from it with `P = 0` EXACT — T regrows 0 → 2.59 (τ=2000) → 5.76 (τ=4000), H conserved to 0.5%, dM 6× above the residual-slide line, **and the dt/2 run reproduces it at matched τ to 4 significant digits** (dM 0.2500/0.2501, T 5.761/5.762). The blindness diagnosis unconditional: `\|s\| ~ 1e-18` machine-zero throughout while the amplitude channel carries O(1) dynamics — the 2c-1 "spontaneous start machine-zero" null measured a symmetry-protected probe. Unkicked reaches the SAME breathing state as kicked (attractor-consistent). Honest: past τ≈5000 the deep-floor stepper cascade dominates (late numbers numerical — the stiffness-stepper item); the ω-attractor measurement remains | `m5_8_2g_spontaneity.py` |
| **ω IS AN ATTRACTOR — the breathing fundamental is a property of the STATE (N-1)** | ✅ kicked / EXACTLY-unkicked / jittered starts share ω₁ ≈ 1.09/1.15/1.07 (± one FFT bin) + a 2ω₁ harmonic, detrend-stable (2.5/5/10 τ dial); the state is QUASI-periodic — the 2e "strict ω₀=0.262 comb" RETIRED (window artifact, caught by pre-registered scrutiny); rod readout: the breather is near-isotropic + core-centered (A_z≈1.1 vs static 5.6), NOT a rod mode | `sandbox_vn/m5_8_2h_omega_attractor.py` |
| **The FIRST ZBW ratio measurement** `ω₁/(2H_rest)` (N-3 — feeds N-6, the promoted ZBW program) | ✅ measured, lattice-natural units: ω₁/(2H_static) = 0.0326/0.0343/0.0326 across the 3 arms (5.4% spread — start-independent, the unit-free statement); H_static(dressed seed, T=0) = 16.74; the breather is strongly excited (H_dyn ≈ 2.7–2.8 × H_static). NO pass/fail vs 1 claimed. **N-6b ABSOLUTE (2026-06-07): under explicit postulates (energy↔0.511 MeV; lattice action↔ℏ) ω = 5.51×10¹⁹ rad/s vs the electron ZBW 1.553×10²¹ — a factor ≈28 short, STRUCTURAL given the N-6a rigidity** (not closable by energy bookkeeping; candidates: V-on/Faber-r₀ core, faithful kinetic, or the action postulate — the V=0 frame is scale-free so no length anchor exists) | `sandbox_vn/m5_8_2j_zbw_ratio.py` |
| **The EM-tilt cone check (N-4) — SHARED CEILING, DIFFERENT TANGENTIAL OPTICS** | ✅ all 3 channel generators (twist Gx, tilts Gy/Gz) share the MAXIMAL cone: fast direction RADIAL, identical ceiling c(r̂)/2 = 1.0000 at N=48 & 64, stiff axis exactly tangential — matter (KG) and light (EM) signals have the SAME maximal speed on the defect background (local SR survives as cone-relative kinematics, radially). Tangential shapes DIFFER per channel: twist (0, 0.33, 0.67) mild 1.74 cone (N-stable, reproduces the c-isotropy gate); Gy near-rank-1 (10:1); Gz rank-1 ≤1e-3 (consistent with EXACT — non-propagating along one tangential direction). Static principal-symbol scope; dynamic tilt transit = refinement | `sandbox_vn/m5_8_2k_tilt_cone.py` |
| **The saturated breather is a MOLTEN CLOCK (N-6d) — the DTC classification** | ✅ controls-anchored battery: persistent ω₁ comb + LOW-DIMENSIONAL CHAOTIC dressing (λ_max ≈ +0.4–0.7/τ, D₂ ≈ 2.7–3.0 — dynamical, NOT additive noise; the noisy-clock control keeps λ ≈ +0.01); the 0-1 K-test exposed as unreliable both ways on this data class (false-1 on noise, false-0 on oversampling) — D₂+λ carry the verdict; molten-ness GROWS with excitation (least-excited arm D₂ 1.70) ⇒ the ground clock may be regular (N-6c tests it) | `sandbox_vn/m5_8_2n_chaos_battery.py` |
| **The unit-free ZBW law test (N-6a): ω IS RIGID — the naive law fails, informatively** | ✅ R_W dressing family (knob-gated, 177.8% H spread, identical probe regions): ω₁ = 1.152/1.188/1.191 (within one FFT bin) across H_static 11.0/16.7/29.1 and H_dyn 23.8/43.9/93.7 — **exponent ω ∝ H^0.033 vs the ZBW law's 1**. The breathing frequency does NOT track dressing energy: it is RIGID (the DTC robustness property) and belongs to the defect CORE — which the V=0 stack cannot vary (the bare frame is scale-free) ⇒ the true ZBW mass family requires the V-on/Faber-r₀ core (resumed-track scope). The v1 RC family was an audit-caught no-op (RC never enters the seed; exponent void); v1 bonus: ω is probe-region-structured (outer shell reads 2ω₁ dominant) | `sandbox_vn/m5_8_2m_zbw_law.py` |
| **The excitation floor (N-6c phase 1): a cold defect CANNOT be prepared by kicking** | ✅ weak-kick arms (0.0125/0.025) land at the SAME-or-higher H_dyn as the 0.05 control (49.2/52.4 vs 47.6 — excitation ∝ kick² is swamped): the seed's own spontaneous release floors excitation at ~1.7× H_rest — the un-sittable minimum's preparation-physics face; cooling requires DAMPING (the settle protocol). Within the accessible band ω₁ scatters 1.09–1.60 with no E-trend (mode composition, consistent with the rigidity finding) and molten metrics stay uniformly high (D₂ 2.3–3.0). **THE GROUND-CLOCK READOUT (phase 2, deep-settle route): THE CLOCK REGULARIZES** — T drained 45.8 → 0.21 (217×), restart P=0 exact; the coldest measured state (T = 2.36, third 1 of the regrowth): **λ_max +0.110 (≈ the short-window bias floor — consistent-with-REGULAR; hot arms +0.5–0.75) and D₂ 1.68 (hot: 2.7–3.0; two-tone control 1.15)** ⇒ cold defect = near-regular clock, hot = molten — the melting curve's cold anchor; the molten-clock bridge closes the right way. Full coldness FORBIDDEN by un-sittability (T regrows — the self-start IS the obstruction: a feature). ω along the regrowth: 1.58 cold → 2.73 heated → [13.6 numerical/cascade]; the saturated attractor 1.09–1.15 — ω is state/mode-dependent, not E-monotone. Cold-ω ⇒ N-6b absolute 7.3e19 rad/s (~21× short — the structural gap stands) | `sandbox_vn/m5_8_2o_omega_of_E.py` |
| **The first spin readout (N-6e): a clean NULL + an instrument bound** | ✅ the rotation Noether charge J (internal + orbital, 3 axes) derived + gated (G1 pairing 1.000000 exact). ★ THE CLOCK KICK IS J-NEUTRAL (J < 1e-4 at t→0 despite finite P — the Θ-twist cancels over the hedgehog sphere): the ZBW breathing channel carries NO net frame angular momentum. G2 failed informatively — BOX TORQUE (the 24³ walls break rotation invariance; secular J growth O(1–10) by t=6; the wall-contacting settled arm reads J~10 immediately) ⇒ canonical states carry no intrinsic J above the noise floor; an ℏ/2-class measurement needs torque-free boundaries (production scale). Spin-½ candidates for the resumed track: polarized/spinning seeds, hopfions, far-field tilt J | `sandbox_vn/m5_8_2p_spin_readout.py` |
| **The invariant matrix COMPLETE (N-5)** | ✅ all candidates measured/classified: signed `u+βu²` WINNER (2d) / Skyrme saturates-but-damps-clock-10× (2e) / Euclid kill-control (2e) / **A4 amplitude quartic GEOMETRICALLY PINNED** (⟨M,M⟩_s constant across the field, dev p95 = 0.0000 ⇒ no lever arm — the fuel is in the curvature sector, which is WHY the signed-u form is correct) / **covariant `𝒮+β𝒮²` deferred-with-reason** (sympy: P cubic in Ṁ ⇒ non-quadratic Legendre + state-dependent inertia; static sector ≡ the validated quartic) | `sandbox_vn/m5_8_2l_invariant_completion.py` |
| The FULL self-consistent 4D evolution is STABLE under the constrained integrator — and the kernel runs in Taichi exactly | ⚠️ **HORIZON-LIMITED (2026-06-05 late evening): stable to ~1150 steps ≈ 2 clock periods, then a negative-H ghost runaway opens — IN BOTH f32 AND f64 (collapse curves agree to ~1%: precision exonerated, the runaway is the constrained evolution itself; H → −8.6×10⁹, align → 0.55 by step 6000; **dt-halving verdict: onset at FIXED τ, ratio exactly 2.0 ⇒ TRUE DYNAMICS of the quadratic action — the engine is exonerated entirely; → M5.8.2d quartic test + the Duda question**).** Within the horizon: ✅ 900 steps, no frozen background: dressed defect held (align 1.000→0.983), H rose then PLATEAUED (9.8→17.4→17.1, no λ≈15.6/t catastrophe), v-cap never engaged, signature-cohesion survives backreaction (core-frac loss −0.08 Mink vs −0.26 Euclid). Taichi port machine-exact (field diff 5.5×10⁻¹⁵ @890 vs f64 numpy), f32 benign (10⁻⁵), Metal clean, 14.5 ms/step at 64³ (69 steps/s) | `m5_8_2c1_full_evolution.py` + `m5_8_2cb_taichi_constrained.py` |

Everything above comes straight out of the medium topology. **No particle stability yet** — the 4D mechanism has passed every level tested so far (1D exact → 4D energy functional → CC → linearized field, each with Euclidean kill-controls), but everything to date **freezes the hedgehog background by hand**. What remains open is exactly the following gate list — the first self-consistent test is M5.8.2c:

### Open gates — claimed by the model, NOT yet validated

| Open gate | Honest status | Owner / pass-fail definition |
| --- | --- | --- |
| The defect holds ITSELF together (the M5.7 dispersal question, self-consistent) | ✅ **DECIDED (m5_8_2i, 2026-06-07)**: free dressed defect, full backreaction, quartic stack, 24³ + 48³ — structural alignment decays SLOWER at 48³ at every matched t (0.980/0.951/0.931/0.884 vs 0.962/0.919/0.889/0.842 — the ANTI-M5.7 signature; M5.7's wash-out strengthened with resolution, this WEAKENS), decelerating to a plateau ≈0.8–0.88 ≫ 0.5 random; act-mask-insensitive (core3/rod15 ≡ std); early core-contrast 3–4× volume fraction at both grids, slow closed-box spreading (no resolution wash-out). Honest qualifiers: ω₁ shifts 1.15→0.83 across grids under anchored β (β is cutoff-dependent — scheme systematic recorded for N-6b); 48³ healthy window ends t≈13 (the stepper cascade arrives EARLIER at resolution — the stiffness item quantified) | **G-2c-1** ✅ — the defect holds, resolution-robust; `sandbox_vn/m5_8_2i_dispersal_gate.py` |
| Fuel-shell saturation (propulsion bounded) | ❌ **FAILS for the QUADRATIC action (2026-06-05 long-horizon)**: the 900-step plateau was transient — a negative-H fuel runaway opens at τ ≈ 2 clock periods, dt-invariant, in f64≡f32, V-on/off, 24³/63³ ⇒ compactness + backreaction do NOT saturate; the field-level twin of the 2b-1 no-cap theorem and the 1D article's own βR⁴ requirement | **G-2c-2 — ✅ CLOSED (2026-06-07)**: `V_u = u + βu²` restores BOUNDED dynamics (M5.8.2d), and the closeout items all landed — f64 confirm ✅ (2e), Euclid kill-control ✅ (2e), and the invariant question is now MEASURED not asked (N-5: matrix complete — signed-u winner / Skyrme damps / A4 pinned / covariant deferred-with-reason; reported to Duda as a finding, not a question). Residuals: the stepper stiffness near the deep floor (NG-2 — quantified by N-2: the cascade arrives EARLIER at 48³) + β is CUTOFF-DEPENDENT (u_min deepens with resolution ⇒ the anchored quartic strength runs with the lattice — an EFT-flavored scheme note feeding the N-6b error budget) |
| ω self-selection (spontaneous spin-up — the time-crystal claim proper) | ✅ **THE SPONTANEITY HALF DEMONSTRATED (m5_8_2g, 2026-06-06)**: a damped-SETTLED config restarted with `P = 0` EXACT regrows kinetic energy 0 → 5.76 by τ=4000 with H conserved to 0.5% — **dt/2-converged at matched τ to 4 significant digits** (real dynamics; no transient explanation survives — the settle protocol closed that hole). The old machine-zero null was PROBE BLINDNESS (derived in Track C C3, confirmed in-field: `\|s\| ~ 1e-18` symmetry-protected while the amplitude channel carries O(1) motion). The unkicked seed reaches the SAME breathing state as the kicked one (dM 1.58/1.46, T 80/84 — attractor-consistent). **THE ω-ATTRACTOR HALF CLOSED (m5_8_2h, 2026-06-07)**: dense dt/2 spectra of 4 arms — kicked / EXACTLY-unkicked / jittered share the breathing fundamental **ω₁ ≈ 1.09/1.15/1.07 (± one FFT bin 0.13) + a 2ω₁ harmonic (2.04–2.34)**, detrend-stable across 2.5/5/10 τ filter windows ⇒ ω is a property of the STATE — **G-2c-3 ✅ (both halves)**. Honest content: the state is QUASI-periodic (fundamental + harmonic over broadband drift), NOT strictly periodic — the 2e "ω₀=0.262 exact comb" was an FFT-window artifact (caught by pre-registered W1 scrutiny); the settled-restart arm shows its own stable 1.79 line while spatially still in transit toward the breather (r_mean 6.8→5.2 inward all window) — inconclusive for that arm, not contradicting. Bonus same-run: rod readout (motion near-isotropic A_z≈1.1, frac_rod = volume fraction — the breather is NOT a rod mode) + N-3 step 1 (no 5.86-class line at saturation; the linear clock softens ~5×) | **G-2c-3**: from a non-spinning dressed defect, the dynamics selects a specific ω — ✅ |
| Physical calibration ω = 2mc²/ℏ (the electron Zitterbewegung) | ❌ untouched — sandbox units only | **M5.8.3** (GROUP-HEADLINE gate): `ω·ℏ/(2H_rest) → 1` within 10%; absolute Hz via the Faber `r₀` fix |

---

## Active Count

```text
0 IMMEDIATE   M5.7 COMPLETE (2026-05-28). Free vs driven split: free
              defect DISPERSES (M5.7.1 seeded + M5.7.2 intrinsic, both NULL
              resolution-confirmed) ⇒ particle/clock is 4D (M5.8); DRIVEN
              defect SUSTAINS a bounded frequency-selective (A,ω) excess
              (M5.7.3 preview, EM-lever ~3× free at resonance) ⇒ the
              driven-excess substrate. Root cause of free dispersal:
              V confines amplitude not orientation (M5.6.5c). Carry-overs
              migrated to M5.8. NEXT PHASE = Rodrigo's call (M5.8 4D clock,
              or the driven-excess lever).

1 SELF-       Q11 Close's Eq.23 exact form — NO LONGER a hard gate (PLAN B
DETERMINE         = self-determination, likely Plan A): Eq.23 is in his
(was gate)        published paper (read + test all 3 forms ourselves, the
                  physical one preserves ∇·s=0); seeded-vs-intrinsic is
                  empirical (the sim answers it). Don't bottleneck on email.

6 BACKGROUND  Q4  Liu et al. lab anchor — does it change sim priority?
(long-tail)   Q7  exact V(M) coeffs — NO LONGER gates (M5.6 shipped Eq.13
                  b=0 confinement interim); exact Λ=(1,δ,0) form feeds M5.9
              Q8  Faber exact running-coupling — NO LONGER gates (M5.6
                  ported + validated the mechanism, mass pinned E∝1/r₀);
                  exact scheme refines M5.9 calibration
              Q9  deeper substrate beneath M (anisotropic fluid?)
              Q10 weak-force clean SU(2) mechanism (gap; beta decay as
                  topology reconnection is a partial answer)
              Q12 Bell / Kochen-Specker vs M5's definite-orientation
                  defect (foundational; defense template + MERW bridge)

5 RESOLVED    Q1  initial-condition construction (no static soliton —
              triple-confirmed: Duda Fig.10 + Close + Werbos + M5.2)
              Q2  V(ψ) shape (wrong substrate — matrix M required)
              Q3  connection/curvature A-layer (load-bearing; paper §II)
              Q5  substrate: full M = ODO^T vs Q-tensor (full M)
              Q6  eigenvalue → physics mapping (1=EM, δ=QM, g=gravity)

Total: 12 questions (0 immediate, 7 open, 5 resolved). M5.7 complete; M5.8 (4D clock) next — Rodrigo's call.
```

---

## OPEN QUESTIONS

| ID | Question | Surfaced | Gates | Status |
| --- | --- | --- | --- | --- |
| Q7 | Exact V(M) form — Duda's Eq.13 Landau-de Gennes `V_LG = a Tr(M²) − b Tr(M³) + c (Tr(M²))²`, or "slightly different"? Duda calls this "the most difficult" part and an open research question. | Duda 2026-05-15 (`4a §12`) | ~~M5.6~~ → M5.9 | OPEN — **no longer gating**. M5.6 (✅) shipped the working interim: the 3-term Eq.13 form has **no biaxial minimum** (`∂V/∂λ` collapses nonzero eigenvalues to one root), so a `b≠0` term erodes δ → uniaxial; M5.6.5c uses a **`b=0` amplitude well** `V=a·Tr(M²)+c·(Tr(M²))²` (min at `Tr(M²)=1+δ²`) that confines amplitude while leaving δ exactly flat (live in production, `LDG_STIFFNESS_K`). A fully biaxial-STABLE vacuum needs an extra invariant — that exact `Λ=(1,δ,0)` form is the Duda-open piece, feeds M5.9 lepton calibration. |
| Q8 | Faber regularization — exact form to "activate" V(M) and produce the running-coupling effect. Duda points to Faber's two papers as the starting scheme. | Duda 2026-04-19 / 05-15 | ~~M5.6~~ → M5.9 | OPEN — **no longer gating**. M5.6 (✅) **ported Faber's MTF** (`Universe` 11/2025/113): reproduced `E₀=α_f ℏc·π/(4r₀)`, mapped `Λ=q₀⁶/r₀⁴` onto `M=ODO^T` via spatially-melting eigenvalues, **mass pinned `E∝1/r₀`** (`r₀=2.2132 fm → 0.511 MeV`), and confirmed Faber `R=Γ×Γ` → homogeneous Maxwell + running-coupling onset at `r₀` (M5.6.4). The mass knob is now the physical `r₀` tied to `α_f`. The exact `arctan` profile re-derivation (vs imposed) is the remaining Duda-open piece → M5.9 BVP. |
| Q11 | Close's Eq.23 exact implementation form — direct form vs vector-potential `s = ∇×A` vs divergence-cleaning projection; and should the resonance amplitude sweep span the full `m ∈ {−1, 0, +1}` dipole family? | Rodrigo → Close 2026-04-19 (`1a`) | M5.7 | OPEN, pending Close. **PLAN B = self-determination (and likely Plan A — see below).** M5.7.1 already proceeded on best-current reading. We do NOT need to wait on Close: (1) Eq.23 is in his **published** paper (`../theory/Equation-of-Everything.pdf`) — we can read it directly and **empirically test all three candidate forms ourselves** (the physical one preserves `∇·s = 0` + gives bounded, energy-conserving dynamics — a numerical discriminator, not an opinion); (2) the deeper "seeded resonance vs the defect's intrinsic oscillation" question is **empirical** — the simulation answers it (M5.7.1: seeded l=1 disperses; M5.7.2: does the intrinsic oscillation cohere?), Close's view is confirmation not gate. Close's framework is cited (published); the answers are ours to derive. M5 must not bottleneck on advisor email latency. |
| Q4 | Liu et al. *Nature Physics* 2026 (first direct laser creation of isolated hopfions + skyrmions) — does the lab confirmation change what the framework should simulate first (hedgehog before hopfion? simpler stabilizer before LdG?)? | `3b` (2026-05-12) | — | OPEN. Lab-existence anchor for topology-as-particles. Current plan keeps hedgehog-first; hopfion is post-M5.8 frontier (`project_particle_defect_correspondence`). |
| Q10 | Weak force — is there a clean matrix mechanism (SU(2) chiral) the way EM/gravity/strong have one? | `4a §7` | — | OPEN GAP. Partial answer in slides: beta decay appears as a topology-reconnection event (defect-class transition), not a force in the EM/gravity sense. No clean SU(2) mechanism yet. |
| Q9 | Is there a deeper substrate beneath the matrix field M (an "anisotropic fluid")? Duda hints the matrix may be effective, with something deeper below. | Duda 2026-05-15 (`4a §1, §12`) | — | OPEN / PARKED. Matches OpenWave's existing granule-level picture (matrix effective, granules deeper). Not actionable now. |
| Q12 | How does M5's definite-orientation defect (real spin axis + clock phase, deterministic local PDE) reconcile with Bell / Kochen-Specker contextuality? | Duda↔Hadley thread 2026-05-11; Close/Adenier thread 2026-05 | — | OPEN (foundational). Defense template: field evolution is local-realistic (cannot violate Bell); QM statistics enter at the measurement/collapse layer (Duda's Malus analogy + Feynman-ensemble), bridged by MERW `ρ=\|ψ\|²` (`4a §11b.4`). Close building the parallel Adenier / Eq.10-factoring defense. Full note in [`1b_topological_defect.md` § Foundational stance](1b_topological_defect.md#foundational-stance--m5-as-a-local-realistic-field-bell--kochen-specker). |

---

## RESOLVED QUESTIONS

| ID | Question | Resolution |
| --- | --- | --- |
| Q1 | Initial-condition construction for a stable single particle — relax against full Lagrangian (a), exact-soliton ODE BVP (b), or something else (c)? | RESOLVED — premise was wrong: **there is no static stable solution**. The particle is a **time-periodic resonance**. Triple-confirmed: Duda paper Fig.10 (4D Lorentz negative-energy auto-propels the clock) + Close email ("l=1 amplitudes, I doubt you'll find completely stable non-radiating solutions") + Werbos chaoiton claim + our own M5.2 empirics. See `feedback_no_static_solitons`. |
| Q2 | Is there a V(ψ) admitting a stable 3D hedgehog soliton (we tried φ⁴, biharmonic)? | RESOLVED — wrong substrate. The real potential is Eq.13 LdG `V_LG(M)` on the **matrix field M = ODO^T**, not any V(ψ) on Vector(3). None of our scalar V(ψ) probes were testing the actual proposed potential. Folds into Q5 + Q7. |
| Q3 | Is an explicit connection/curvature `A`-as-primary-field layer load-bearing for hedgehog stability, or the same physics in different clothing? | RESOLVED — load-bearing (paper §II). `A_μ = [M, ∂_μ M]` (Eq.19), `F_μν = ∂_μ A_ν − ∂_ν A_μ` (Eq.20). The Brouwer-degree winding integral we use is the right diagnostic, but the underlying field is the matrix M. |
| Q5 | Substrate: full real-symmetric matrix `M = ODO^T` (6 DoF) vs traceless Q-tensor (5 DoF)? | RESOLVED 2026-05-15 — **full M, no Q-tensor pivot**. Confirmed three ways: Duda's image 2 writes `M = ODO^T` with `D = diag(g,1,δ,0)`; the slides reaffirm it everywhere; Duda's follow-up used the same notation. Refactor green-lit (`4a §2`). |
| Q6 | Eigenvalue → physics mapping — what does each eigenvalue of D tag? | RESOLVED 2026-05-15 (Duda direct quotes, `4a §8`) — `1` = EM tilts (highest Lagrangian contribution); `δ ~ ℏ` = QM twists (quantum phase); `g ≫ 1` = gravity boosts (hedgehog = black hole); `0` (4D only) = null/time axis (clock propulsion). The 3 leptons come from 3D spatial axis-choice, NOT from the g eigenvalue. |

---

## MODEL EMERGENCE — observed reality from the matrix field `M = ODO^T`

What known physical phenomena the M5 substrate reproduces from the Duda Eq.18 action, with the numerical mechanism + status + what's pending. **No model comparisons** (those live in the `1_model_selection.md`). **No conceptual explanations** of substrate / defect / twist / eigenvalues — those stay in `1b` / `4a` / `5a`. Just emergence + status.

**Substrate (recap, see Q6 RESOLVED).** Symmetric 3×3 `M(x) = O(x)·D·O^T(x)` on a 3D lattice; 5 DoF/voxel. Preferred shape `D = diag(g, 1, δ, 0)`. Eigenvalue → physics map (M5.6.4-verified):

| Eigenvalue | Physical mechanism | Phase |
| --- | --- | --- |
| `g ≫ 1` | gravity (boost; hedgehog ↔ black hole) | M5.8 (4D) |
| `1` | EM (tilts → Maxwell) | ✅ M5.6 |
| `δ ~ ℏ` | QM (twist → Klein-Gordon, mass is GEOMETRIC — Fig.9) | ✅ M5.6 |
| `0` (4D only) | time signature → Zitterbewegung clock propulsion (Fig.10) | M5.8 |

### PARTICLES (full spectrum)

| Phenomenon | M5 numerical mechanism | Status |
| --- | --- | --- |
| Quantized electric charge | Brouwer winding (S² → S² topology) on M's principal director; integer, conserved by topology | ✅ M5.1 (`compute_winding_number`) |
| Electron rest mass | Faber regularization `E₀ = α_f · ℏc · π / (4 r₀)`; physical knob `r₀ = 2.2132 fm → 0.511 MeV` | ✅ M5.6.3 (mass pinned `E ∝ 1/r₀`) |
| Lepton families `e / μ / τ` | 3 axis-choices of biaxial `Λ = (g, 1, δ)`; same Q, different masses | 🚧 M5.9 calibration |
| Neutrino | Closed vortex loop (distinct topology class from hedgehog) | 🚧 M5.9 / sub-eV |
| Quarks (u, d, s, …) | String endpoint — 1D topological vortex (not 0D hedgehog); Cornell `V(r) = −α/r + σ·r`, `σ ≈ 1 GeV/fm` | 🚧 M5.9 |
| Nucleons (p, n) | 3-string Y-configuration of quark endpoints | DEFERRED → 15a |
| Atomic orbital binding | EM Coulomb + de Broglie standing-wave quantization (NOT a separate force) | → 15a (cross-mass-class) |
| Composites (nuclei, atoms, hopfions) | Multi-defect / knot configurations | DEFERRED → 15a |
| Zitterbewegung clock `ω = 2mc²/ℏ` | Biaxial hedgehog dynamically sources its own twist (M5.6.2b: `ψ = 0` is NOT static = the clock seed); 4D Lorentz negative-energy from `0`-eigenvalue propels it | 🚧 M5.8 (GROUP HEADLINE; build spec `5a §10d` + reorganized roadmap M5.8 gates/breadth/carry-overs/unblocks; faithful kinetic per M5.6.5d) |
| Spin (real, transportable, conserved) | Director orientation = topological invariant; preserved by dynamics | ✅ inherent; empirical anchor: graphene spin-transport |
| First metastable resonance ("first long-lived M5 particle") | `l = 1` harmonic seed on biaxial substrate; extended-lifetime localization (NOT static stability — Q1 RESOLVED) | ✅ **M5.7 COMPLETE (2026-05-28) — the particle is 4D (M5.8), not 3D.** **M5.7.1 (seeded l=1) + M5.7.2 (intrinsic oscillation) = TWO confirmed NULLS:** both seeded and intrinsic 3D orientation energy disperse (apparent N=32 localization washed out at N=48 in both; intrinsic osc frequency 0.25→0.10/t = resolution-dependent, not a converged clock). **Root cause: V confines amplitude but NOT orientation** (M5.6.5c, rotation-invariant). Consistent with Derrick + Duda's framework: the stable particle is the **4D Zitterbewegung clock** (Fig.10), so 3D dispersing is *expected* — M5.7 empirically motivates M5.8 (4D) as necessary. Nuance: topological defect itself permanent (winding conserved); *driven* oscillation is a separate question — **M5.7.3 (preview, 2026-05-28) confirmed the EM-like lever sustains a bounded, frequency-selective `(A,ω)` excess (~3× free, resonant at the natural mode), resolution-confirmed** → the driven-excess substrate. Energy-conservation bankable (N=48 H-drift 0.01%). (`5a §5h/§5i/§5j`) |
| Lorentz contraction / relativistic kinematics | Wave-equation relativistic invariance of the matrix dynamics | ✅ structural; quantitative test = M5.8 |
| Annihilation (Q=+1 + Q=−1 → vacuum + photons) | Topology cancellation event releases stored field energy as outgoing EM waves | ✅ substrate; quantify (future) |

**Mass → Zitterbewegung target table (M5.8 validation):**

| Particle | Target `ω = 2mc²/ℏ` | Notes |
| --- | --- | --- |
| Electron | 1.55 × 10²¹ rad/s | M5.8 exit gate (within 10%) |
| Muon | 3.21 × 10²³ rad/s | cross-particle test |
| Tau | 5.39 × 10²⁴ rad/s | cross-particle test |
| Neutrino | ~10¹⁵ rad/s | sub-eV (closed vortex loop) |
| Up quark | ~7 × 10²¹ rad/s | string endpoint |
| Down quark | ~1.4 × 10²² rad/s | string endpoint |
| Proton | 2.85 × 10²⁴ rad/s | 3-string Y-config (15a) |

### FORCES

| Phenomenon | M5 numerical mechanism | Status |
| --- | --- | --- |
| Coulomb `1/r` (electrostatic) | Director splay around hedgehog defect; `∇·n̂ ≈ 2/r` far field — **pure topology, no waves needed** | ✅ M5.1 R²=0.978 (1/d relaxation), M5.4 R²=0.9959 (analytical page-18 cross-val), both attractive |
| Maxwell (full EM) | 1-axis tilts of `M`; two independent routes verified — (a) hydro↔EM dictionary `∇×u ↔ B`, `ω×u ↔ E`; (b) Faber `R = Γ × Γ` closed field strength (Maurer–Cartan ⇒ `dR=0` ⇒ homogeneous Maxwell) | ✅ M5.6.4 both routes |
| Strong — confinement (linear) | 1D vortex string, Cornell `V = −α/r + σ·r` (Coulomb + linear); `σ ≈ 1 GeV/fm` per unit length | 🚧 M5.9 (string tension) |
| Strong — short-range (running coupling) | Faber `α_sol(d)`: `R` is non-abelian; Maxwell is long-range abelian limit; short-range = running coupling onset at `r₀` | ✅ M5.6.4 (mechanism); 🚧 M5.9 (calibration) |
| Gravity (linear, GEM) | g-axis boosts on 4D-extended `M`; gravito-EM has same form as Maxwell with `Γ⁰` as boost component | 🚧 M5.8 (4D + SO(1,3) Lorentz) |
| Weak force | PARTIAL — beta decay as topology-reconnection event (defect-class transition: neutron → split → proton + electron + neutrino). No clean SU(2) chiral mechanism yet. | gap → Q10 |
| Magnetism (intrinsic per-defect) | S¹-loop topology (vs S² for electric charge — different topology, same SO(3) parent); the T-component of the defect's outgoing wave (the L+T decomposition) | ✅ mechanism; ⌛ inertial-observability via frequency-downshift = Phase 4 falsifiable target |
| Magnetism (coherent macroscopic) | Coherent T-component summing across aligned defects (permanent magnets) or coherent motion (electromagnets) — reproduces both standard regimes by construction | ✅ structural |
| Atomic orbital "force" | NOT fundamental — EM Coulomb + de Broglie standing-wave quantization | → 15a |

### EM WAVES (photon emergence)

| Phenomenon | M5 numerical mechanism | Status |
| --- | --- | --- |
| Maxwell wave-packet propagation | Tilt modes of `M` propagate at `c` (abelian linear limit of Faber `R`) | ✅ M5.6.4 |
| Outgoing-wave divergence + curl decomposition | Each defect emits a **divergence/radial** part (`∇·n̂` = electric splay — the longitudinal/Coulomb piece in EWT/M3 terms) + a **curl/circulation** part (`∇×n̂` = magnetic — the transverse/radiative piece); the Helmholtz split (M5-native div+curl = EWT's L+T). Shared single `ω = 2mc²/ℏ` Zitterbewegung clock at defect core | ✅ structural; per-defect quantification = future |
| Running coupling vs distance | Faber `α_sol(d)`: `‖R‖ ~ 1/r²` far field; `‖R‖·r²` rolls off within `r₀` = onset of non-abelian short-range | ✅ M5.6.4 |
| Photon (quantized EM excitation) | Tilt-mode standing waves on the matrix substrate; quantization mechanism = a future research program | 🚧 future |
| Annihilation → photon emission | Q=+1 + Q=−1 topological cancellation dumps stored field energy into outgoing EM waves | ✅ structural; quantify (future) |
| Static field vs radiation distinction | Static EM = static topology (Frank elastic, no waves); radiation = dynamic tilt-mode propagation. Different mechanisms in the same field | ✅ structural |

### CURRENT LIMITATIONS — what M5 does NOT yet do

| Limitation | Phase that addresses it |
| --- | --- |
| No FREE metastable resonance in 3D — M5.7.1 (seeded) + M5.7.2 (intrinsic) both disperse; the stable particle/clock is 4D (`5a §5h/§5i`) | M5.8 (4D Zitterbewegung clock — the actual stable particle). Driven oscillation (the driven-excess lever) is the separate near-term test (M5.7.3 preview). |
| No Zitterbewegung frequency measurement | M5.8 (faithful kinetic on `O(x) ∈ SO(3)`, per M5.6.5d) |
| Lepton mass ratios not calibrated to PDG | M5.9 |
| Gravity emergence not run (still 3D) | M5.8 (4D + SO(1,3) Lorentz) |
| Quark string + Cornell `σ ≈ 1 GeV/fm` not validated | M5.9 |
| Photon quantization mechanism not formalized | future (post-M5) |
| Weak force — no clean SU(2) chiral mechanism | Q10 (long-tail open) |
| Inertial observability of magnetism via frequency-downshift | Phase 4 falsifiable target |
| Foundational stance vs Bell / Kochen-Specker not formally settled | Q12 (foundational; defense template in `1b § Foundational stance`) |
| Composites (nuclei, atoms, hopfions) deferred | 15a |
| Exact biaxial-stable `V(M)` coeffs `Λ = (1, δ, 0)` | Q7 → M5.9 (M5.6 ships b=0 confinement interim) |
| Exact Faber running-coupling profile (vs imposed `arctan`) | Q8 → M5.9 (M5.6 ports the validated mechanism) |

---

## HARDEST PIECES — status board

The long-running hardest-pieces tracker for M5 (mirrors M6's). These are the load-bearing unknowns the framework must resolve, distinct from the discrete Q-numbered questions above — a piece can persist across many phases.

| Hardest piece | Gates | Status (2026-05-27) | M6 lend? |
| --- | --- | --- | --- |
| Matrix substrate migration (Vector(3) ψ → `M = ODO^T`) | M5.4 ✅ | ✅ **COMPLETE (2026-05-26).** `ti.Matrix.field(3,3)` triple buffer + `eigen_decompose` lynchpin (director recovery 0.9995) + matrix seeders + `‖M−D‖_F`/`‖Ṁ‖_F` trackers. **M5.1 Coulomb reproduced on M** — R²=0.9704 relaxed + R²=0.9959 analytical page-18 cross-val, both attractive. Operators verified vs analytic. Rendering re-sourced from M (on-screen verified); live wave path retired (ψ engine dormant legacy). | — |
| **V(M) potential form** (Duda's #1 limitation, Q7) | M5.6 ✅ | ✅ **M5.6 (2026-05-27).** Eq.13 LdG implemented + V-on confinement live (`LDG_STIFFNESS_K`). Key finding: 3-term form has **no biaxial minimum** → M5.6.5c uses a `b=0` amplitude well (confines `Tr(M²)`, leaves δ flat). Exact biaxial-stable `Λ=(1,δ,0)` form = Q7, Duda-open → M5.9. | ✅ strong (see below) |
| Regularization (Faber) to activate V(M) + running coupling (Duda #2, Q8) | M5.6 ✅ | ✅ **M5.6 (2026-05-27).** Faber MTF ported: `E₀=α_f ℏc·π/(4r₀)` reproduced, `Λ=q₀⁶/r₀⁴` mapped onto M, **mass pinned `E∝1/r₀`**, `R=Γ×Γ`→Maxwell + running-coupling onset at `r₀`. Mass knob is now physical `r₀`. Exact `arctan`-profile re-derivation → M5.9. | 🔶 partial |
| First metastable resonance (no static soliton) | M5.7 ✅ → M5.8 | ✅ **M5.7 COMPLETE (2026-05-28) — free vs driven split.** Free defect disperses (M5.7.1 seeded + M5.7.2 intrinsic, both NULL; root cause V confines amplitude not orientation, M5.6.5c) ⇒ the stable particle is the **4D Zitterbewegung clock (M5.8)** (3D dispersing is expected per Derrick — M5.7 empirically motivates the 4D extension). **Driven defect SUSTAINS** a bounded, frequency-selective `(A,ω)` excess (M5.7.3 preview, EM-lever ~3× free at the resonant natural mode) ⇒ the driven-excess substrate. Energy-conservation bankable (N=48 H-drift 0.01%). `5a §5h/§5i/§5j`. | ✅ strong (→ M5.8) |
| KG-from-twist emergence | M5.6 ✅ | ✅ **DONE (2026-05-27).** KG mass is **GEOMETRIC** — minimal coupling to the hedgehog connection `Â`, the explicit mass term cancels, core regularization generates `mass²(r)=3r_c²/(2 reg²)` (Fig.9 reproduced, `5a §5a-c`). Biaxial hedgehog port + z-axis disclination handling done; defect dynamically sources its own twist (M5.8 clock seed). | — |
| Zitterbewegung clock / 4D negative-energy propulsion | M5.8 | Mechanism known (Fig.10); toy-model numerics available (slide p.33). **Externally validated 2026-05** as THE hard problem: group consensus that ALL macroscopic analogues fail at intrinsic non-damping oscillation (Couder droplets need a shaker; spinning needles damp like a pendulum) — only the elementary-particle clock is intrinsic + non-damping, so propulsion must be intrinsic. Validates the 4D negative-energy direction. | 🔶 partial |
| Biaxial eigenvalue hierarchy → lepton masses (Duda #3/#9/#10) | M5.9 | Calibration parameters (M5.9). | — |
| Weak-force clean SU(2) mechanism (Duda #7, Q10) | — | GAP. Beta decay as topology reconnection is partial. Out of near-term scope. | — |
| Deeper substrate beneath M (Duda #8, Q9) | — | PARKED. Matches granule-level picture. | — |

### Duda's stated limitations of his own LdGS / M5 model

Physics limitations Duda (or his papers, as quoted in our notes) flags about his own framework. Sourced from the M5 research docs.

| # | Limitation | Source |
| --- | --- | --- |
| 1 | **V(M) potential form unspecified.** *"the Higgs-like potential V(M) — its specific form is not yet determined"* | `1a:308` |
| 2 | **Regularization is the hardest part.** *"LdG regularization is where lepton masses come from, and it is the hardest part to include. The specific regularization form is still an open research question"* | `1a:230, 260` |
| 3 | **Biaxial axis-length parameters need numerical calibration** — *"their exact values require numerical simulation … treat them as calibration parameters"* | `1a:262` |
| 4 | **Charge magnitude + rigorous Coulomb derivation open** | `1a:115` |
| 5 | **Lepton mass ratios deferred, mechanism only** | `_overview:393` |
| 6 | **Static hedgehog doesn't survive dynamic evolution** — Q decays +0.996 → ~0 within 4–5 steps regardless of V(ψ); confirms the triple admission that no static stable solitons exist | `3b:2-9, 103` |
| 7 | **Weak force mechanism unaddressed** — *"topology-changing events … still an open research question"* | `4a:337-354` |
| 8 | **Deeper substrate beneath M hinted but unconfirmed** — *"there might be an even deeper 'anisotropic fluid' beneath"* | `4a:602` |
| 9 | **Extreme biaxial eigenvalue hierarchy unexplained** — ~3477:1 axis-length ratio not derived from first principles | `1a:230-231` |
| 10 | **Three lepton families pinned to "3 axes in 3D" by symmetry, not derived** — mass ratios still need V(ψ) curvature input | `1b:469` |

### Which M5 phase each limitation gates (physics roadmap)

This table keeps the OpenWave-physics phase-gating for each of Duda's stated limitations.

| Limitation | Gates | Reading |
| --- | --- | --- |
| #1 V(M) form | M5.5 | Direct blocker for the paper Lagrangian. Until V(M) is pinned, defect dynamics under perturbation aren't uniquely simulable. |
| #2 regularization | M5.6 ✅ | The hard step — **ported cleanly (2026-05-27)**. Faber's MTF mapped onto M, mass pinned `E∝1/r₀`, `R=Γ×Γ`→Maxwell. M5.6 unblocked; exact running-coupling form → M5.9. |
| #6 no static solitons | (design) | Already a design choice — pursue Close's `l=1` resonance protocol (M5.7), not static seeds. |
| #7 weak force | — | Out of all near-term scope. |
| #9 biaxial hierarchy | M5.9 | Lepton masses are a scientific deliverable (M5.9). |

---

## M6 → M5 cross-pollination — what the closed M6 sandbox can lend M5

**ACTIVATED 2026-06-05 night (M5.8.2e)**: the transfer is now in use for the saturating-invariant program — (i) the v9 "forward-IVP fails / BVP-with-free-eigenvalue works" lesson → the breathing state as a radial BVP with ω free (Track C, the M5.8.3 doorstep); (ii) Pohozaev virial → the saturation self-consistency probe; (iii) the chaoiton's `gβ⁴` quartic + Lean-proved CS/Hopf charge → the invariant candidates list (amplitude-quartic + the Faddeev–Skyrme/Hopf-bound literature anchor); (iv) `l=1` BC + η factor → the BVP setup.

**FIRST OUTCOME (Track C C1, `m5_8_2f_breathing_bvp.py`, 2026-06-05/06 night) — the transfer's own deepest lesson applied**: the BVP attack is ANSATZ-GATED. The rigid-global-clock + radial-boost-dressing manifold provably hosts NO interior minimum even with the 2d quartic on (the 2b net-cancellation, now variational — V+βQ monotone along the ghost branch, fuel only in the onset layer). ⇒ C2 poses the CORE-LOCALIZED clock (the 2b-2/m5_6_2b twist mode, `R(Θ·φ(r))` with its own radial profile) — the M5 analog of M6's `l=1`-BC unlock: get the reduced manifold right FIRST, then free the eigenvalue.

**SECOND OUTCOME (Track C C2-B, `m5_8_2f2_localized_clock.py`, 2026-06-06)**: the phase-twisted eigenframe `R(Θ+ψ(r))` ALSO closes negative-clean — the static twist is pure KG cost (m5_6_2b confirmed variationally; the twist-fuel pocket exists but is 5 orders too weak). **Both static-profile reductions closed decisive ⇒ the breathing state is irreducibly time-dependent (`ḃ≠0` IS the 2d bounce)** — so the transfer evolves: M6's BVP found a STATIC ground state; ours is a time-PERIODIC breather, and C3 poses it as a **periodic-orbit/limit-cycle problem in the reduced CC dynamics with ω as the free eigenvalue** — the same eigenvalue discipline, one derivative up. (Option A, differential rotation, is not a steady ansatz — winding tax grows linearly in Θ; kept as a diagnostic only.)

**THIRD OUTCOME (Track C C3, `m5_8_2f3_breather_orbit.py`, 2026-06-06) — Track C CLOSES with the spontaneity DERIVED**: the reduced CC dynamics finds **the dressed static minimum is UN-SITTABLE** — `K_bb(a*) = −67.6 < 0` (direct-confirmed) with `U″ = +1723 > 0`: energetically minimal, kinetically ghost ⇒ compulsory motion (G-2c-3's spontaneity, derived variationally — and `K_bΘ ≡ 0` under the exact phase average exposes 2b's stabilizing cross-term as sampling residue). But containment is NOT reducible: at anchored β the minimum is fully ghost-kinetic and the orbit exits through the `det 𝕂 = 0` surface where ANY net-inertia reduction is ill-posed — the full field redistributes momentum across modes there (the 2c-1 per-voxel eigenprojection's job). **The M6-transfer's final form: the eigenvalue discipline carried the program exactly as far as a reduction can go — the rest is the field's.** Falsifiable handoff: the unkicked dressed seed in 2d should spontaneously breathe (probe `b_peak`/`u_min`/`H` — the `s(t)` clock probe is blind to the amplitude channel, diagnosing the old machine-zero null).

**TRANSFER VERDICT (2026-06-07): strongly positive, and FINISHED — it paid out differently than designed.** The direct transfer (pose-the-state-as-BVP) closed negative three times in a row — and those negatives were the cheapest, most decisive results of the whole program, because the ladder forced out the real physics. The closing move was the field handoff: 2g confirmed the C3-predicted spontaneity in-field (settled config, `P=0` exact, T regrows 0→5.76, **dt/2-converged to 4 significant digits**) — the M6 methodology delivering M5's missing spontaneity result. The section is CLOSED; M6 stays sandbox-only/repo-as-deliverable.

| Transfer item | What it produced |
| --- | --- |
| "Forward-IVP fails / BVP-with-free-eigenvalue works" (v9) | Track C C1→C2-B→C3: both static reductions closed decisive; the reduced dynamics DERIVED the un-sittable minimum — spontaneity as a theorem, not a hope |
| The chaoiton `gβ⁴` quartic analog | The 2d `u+βu²` saturating quartic — the single term that fixed the 3+1D runaway (≥45 clock periods, f64-anchored) |
| Pohozaev/virial discipline | The self-consistency probes + the auto-grade honesty pattern (`trend_report`) |
| `l=1` BC / ansatz lesson | The deepest one: BVP attacks are ANSATZ-GATED — applied three times to kill wrong manifolds fast |

M6 is closed (DM paper published, sandbox-only, no Taichi port), but its sandbox **methodology** transfers to M5. M6's scipy sandbox is far cheaper to prototype in than M5's Taichi matrix substrate, so several M5 unknowns can be de-risked in a scipy M5-sandbox *before* committing Taichi kernels — exactly the path M6 took. **This is sandbox methodology transfer, NOT an M6 production port.**

### The V(M) lead — M6 settled its potential the hard way, and the method transfers

M6's biggest relevance to M5's #1 hardest piece (V(M) form): M6 **settled its neutral-sector potential + geometry** as a concrete BVP. The neutral chaoiton ground state is the 3D spherical `l=1` p-wave ODE

```text
β'' + (2/r)β' − (2/r²)β − m_J²β + 4gβ³ = 0
```

found in **sandbox v9 phase 2** by treating `m_J` as a free eigenvalue with a proper `l=1` origin BC (`β ~ B0·r`) and `B0` fixed — *after* forward-IVP/shooting had failed. The transferable lesson is methodological, not the equation itself:

| M6 lesson (v9 phase 2 + v10) | M5 application |
| --- | --- |
| Forward-IVP/shooting does NOT find localized soliton ground states; **BVP with the mass parameter as a free eigenvalue does** | M5.6/M5.7 — solve the matrix-field defect radial profile as a BVP-eigenvalue problem, not by guessing V(M) and forward-stepping (note: M5.5.4's *forward* leapfrog confirms this — an unregularized hedgehog sloshes, energy-conserving, but does not settle into a localized profile without the M5.6 regularized BVP) |
| Correct `l=1` p-wave origin BC (`β ~ B0·r`) is what unlocked the ground state | M5's hedgehog is an `l=1`-class structure — the BC handling transfers directly |
| `η` geometry-conversion factor `(∫β²r dr)/(∫β²r² dr)` reconciles cylindrical vs spherical energy integrals | M5 will compare matrix-field energies across defect geometries — same conversion needed |
| Pohozaev-type virial identity as a self-consistency diagnostic (M6's `m_J/η = 1.21` family-invariant, Q47) | A scaling identity is a cheap correctness check on any M5 soliton profile |
| Sandbox-first: prototype in scipy, validate, *then* port to production | Prototype the M5 V(M) + matrix-defect-profile BVP in a scipy M5-sandbox before the Taichi M5.6+ kernels (the path M5.5 followed: `sandbox_v5` scipy/numpy → Taichi port) |

Canonical M6 numerical recipe (charged sector via 2-function IVP, neutral sector via the BVP above) is consolidated in `m6_ouroboros/research/0d_canonical.md` — the reference if we want to mirror the methodology.

### M6 strengths available to M5 regardless of M6's gate outcomes

Imported from `m6_ouroboros/research/0b_model_gates.md` § "What M6 offers regardless of gate outcomes" and `1_model_selection.md`:

| M6 strength | Relevance to M5 |
| --- | --- |
| Neutral chaoiton **ground state reachable** via BVP (v9 phase 2) | Proof-of-method that a topological soliton's ground state IS findable numerically — encouragement + recipe for M5.7's metastable resonance hunt |
| DM mass prediction **published** (DM paper v2, Zenodo 20350105, 2026-05-22): `m_χ = 0.460 MeV`, `m_J = 0.618 MeV`, `C = 770 MeV·fm` (canonical g=1.0, B0=0.5; `0d_canonical.md`). Supersedes the earlier v8 scan numbers (0.998/1.033/6.7e-4) | Independent topology-as-particles result; cross-validates the shared "no static soliton / time-periodic" premise M5 also relies on |
| Lepton spectrum at PDG precision via 2-function reduction | Demonstrates the ansatz-reduction discipline (collapse DoF before solving) that M5.5 can borrow for the matrix field |
| Charge quantization via Chern-Simons / Hopf (Lean-proved) | Different topological flavor (Hopf-linking) from M5's Brouwer-winding — but confirms charge-from-topology is robust across frameworks |
| Methodology cross-pollination (η, `l=1` BC, virial diagnostic) | Listed above — the concrete transferable pieces |

**Caveat (load-bearing):** M6 is **not** the substrate for M5's `(A, ω)` excitation degree of freedom — that lives on M5's matrix field. M6's contribution to M5 is **method + credibility cross-validation**, not physics substrate. A hypothetical M6-substrate cross-check would require a full Taichi production port (ensemble + external-drive kernels), which is NO-GO / deferred indefinitely.

---

## Notes on scope

- This tracker covers **M5 physics/framework questions for the group** (Duda / Close / Yee) plus the hardest-pieces board. M5.4 implementation decisions (Taichi storage layout, eigen-kernel design, granule/glyph UX calls) are tracked as **roadmap tasks** in [`0b_M5_roadmap.md`](0b_M5_roadmap.md) and [`4b_rendering_features.md`](4b_rendering_features.md), not here — same split as M6 (questions here, sandbox tasks in the work log).
- M5 has **no active email round in flight** (unlike M6's intense Werbos/DeepSeek cadence). Duda's last substantive reply was 2026-05-15. **M5.6 (complete 2026-05-27) now gives a strong results bundle to seed a Q7/Q8 outreach**: the Eq.18 action runs in production + is energy-conserving, KG mass is geometric (Fig.9), Maxwell recovered both routes, Faber ported with mass pinned `E∝1/r₀`, and the **V(M)-is-rotation-invariant finding** (V acts only on the eigenvalue/regularization sector, not the twist) + the **3-term-LdG-has-no-biaxial-minimum** result sharpen the Q7 ask — Duda's exact `Λ=(1,δ,0)` Eq.13 coeffs + Faber's exact running-coupling scheme (Q8) are the natural questions, now feeding M5.9 calibration. **M5.7 (complete 2026-05-28) adds a further outreach hook** — the free-disperses/driven-sustains split (the particle is 4D; the EM lever sustains a bounded `(A,ω)` excess) is a clean result to share with Close (Q11) + Duda, no longer a question. None blocks current work (M5.7 complete; next = M5.8, Rodrigo's call).
