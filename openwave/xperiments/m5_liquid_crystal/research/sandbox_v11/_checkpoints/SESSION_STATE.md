# M5.11 session state (run 2 complete , Taichi AD)

Run 1 go: 2026-06-22 20:43 EDT (P0-P2). Run 2 go: 2026-06-22 21:54 EDT (Taichi-AD, P1b+P2).
Plan: [`11a_vortex_loop.md`](../../11a_vortex_loop.md). Findings: [`11b_findings.md`](../../11b_findings.md).

## Done + validated

| Phase | Script | Result |
| --- | --- | --- |
| P0 minimizer + V_M/LdG | `v11_p0_minimizer.py` | all gates pass |
| P1a Faber electron | `v11_p1_faber_electron.py` | 511.00 keV at r0=2.2132 fm, I=π/4 to 6e-6 |
| P1b machinery + α⁻¹ | `v11_p1b_lattice.py`, `v11_p1b_dipole.py` | 3D+axisym Γ/R validated; charge→1e, 1/α_sol→137.03 |
| AD engine (run 2) | `v11_ad_energy.py` | Taichi reverse-mode AD == P0 (E 4e-16, grad 1.8e-13) |
| P2 run1 plain ring | `v11_p2_vortex_loop.py` | dissolves (curvature combs out) |
| P2 run2 Hopfion | `v11_p2_hopfion.py` | smooth knot EXPANDS (curvature 65→0.10); AD-FIRE monotone |

## The sharpened conclusion (run 2)

The M5 functional = 4th-order curvature + potential, NO 2nd-order Frank term.
Derrick: E(λ)=λ⁻¹ E_curv + λ³ E_pot. A SINGULAR core forces a melt (E_pot>0 → λ³ → stable,
the electron). A SMOOTH knot (Hopfion) keeps amplitude at vacuum (E_pot≈0) → expands.
=> the stable NEUTRINO is a KNOTTED/LINKED **singular** disclination (singular core for the
melt/λ³, knotting for protection). A smooth Hopfion is the wrong object.

## Two open items (both AD-unblocked, for an attended run)

1. **The heliknoton (THE neutrino, P2) , DIRECTION RESOLVED.** The Smalyukh 2020 thesis
   (`theory/liquid_crystal_defects/2020 Topological Solitons in Chiral Condensed Matters PhD.pdf`,
   Duda's own referenced lit) shows the stabilizer is the CHIRAL term, not a Skyrme term:
   add `F_chiral = ∫ 2 q0 L ε_ikl Q_ij ∂_k Q_lj` (q0=2π/p) to the functional + a HELICAL far-field
   background, seed an elementary heliknoton (Hopf Q=1), relax with the AD engine. Our Hopfion
   expanded only because the far-field was uniform and there was no chiral term. Full build spec
   (8 steps + biaxial/window caveats) in `11b_findings.md` § "P2 fork RESOLVED". Concrete change:
   one more differentiable Taichi loop in `v11_ad_energy.py` for the chiral term. Diagnostics:
   Hopf index Q=(1/64π²)∫b·A, chirality tensor C_ij=n_k ε_ljk ∂_i n_l, singular vortex lines = χ
   singularities. Caveat (thesis p.132): biaxial-material singular knots are hard → smooth
   heliknoton first.
2. **Running α(d) (P1b refinement).** Build a differentiable Faber-axisym (ρ,z) two-soliton
   kernel (the M5-tensor AD kernel is the template), minimize at each d, fit α_sol(d). The
   charge route already gave the asymptote (α⁻¹→137); this is the short-d QED-running curve.

P3-P6 depend on a stable loop existing (item 1).

## Resume ping

Not armed this run (no reset time supplied). Lossless via these checkpoints.
