# M5.11 session state (run 3 complete , chiral+Frank heliknoton build)

Run 1 go: 2026-06-22 20:43 EDT (P0-P2). Run 2 go: 2026-06-22 21:54 EDT (Taichi-AD, P1b+P2).
Run 3 go: 2026-06-23 12:34 EDT (P2 heliknoton build: chiral Lifshitz + Frank terms).
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
| P2 run3 chiral+Frank | `v11_p2_heliknoton.py` | chiral Lifshitz + Frank terms built + validated (AD==numpy 1e-14, gradcheck 1.4e-8); biaxial obstruction (see below) |
| P1b' running α(d) | `v11_p1b_running.py` | machinery built+validated; α_sol not cleanly extractable in fast setup (honest negative) |

## The conclusion, sharpened across run 2 + run 3

Run 2: the M5 functional = 4th-order curvature + potential, NO 2nd-order Frank term.
Derrick: E(λ)=λ⁻¹ E_curv + λ³ E_pot. A SINGULAR core forces a melt (E_pot>0 → λ³ → stable,
the electron). A SMOOTH knot (Hopfion) keeps amplitude at vacuum (E_pot≈0) → expands.

Run 3 added the chiral term and refined this:
- The 4th-order term VANISHES for 1D-varying textures (Ecurv(helix)=0), so the chiral term needs
  its 2nd-order **Frank partner** K|∇Q|² (the cholesteric (L/2)|∇Q−q0...|²), NOT the 4th-order
  term. Unified functional: U = c²·4·Σ‖[∂M,∂M]‖² + V_M + K|∇M|² + 2 q0 L ε M ∂M.
- In the **biaxial** M5 tensor, the chiral term drives a blue-phase-like 3D modulation; there is
  NO stable simple helix (any q0/Lc/handedness), so the SMOOTH heliknoton does not form. This is
  the thesis's flagged biaxial-hard case (p.132).
- => the M5 neutrino is most likely the **singular chiral disclination loop** (run-2's singular
  route, now armed with the chiral term), NOT a smooth Hopfion. The chiral+Frank machinery is
  built + validated for it.

## Open items

1. **P2 next , the SINGULAR chiral disclination loop (refined target).** The smooth heliknoton
   does not stabilize in biaxial M5 (run 3). Next: seed a SINGULAR `+1/2` (or `+1`) disclination
   LOOP (singular core = the melt λ³, like run-1's ring) in a chiral background and relax under
   the now-validated `U = 4th-order + V_M + K|∇M|² + chiral`; the chiral term should protect the
   twist where run-1's plain ring (no chiral) dissolved. Diagnostics: does the loop hold finite R
   (vs run-1's combing-out)? Hopf/linking on the χ helical-axis field. ALSO worth: a
   uniaxial-director reduction to demonstrate the heliknoton in the model the thesis actually uses,
   then map back. Machinery ready in `v11_p2_heliknoton.py` (add a disclination-loop seed).
2. **Running α(d) (P1b') , honest negative (2026-06-23).** `v11_p1b_running.py` built + validated
   (single soliton gradcheck 2e-6); `α_sol` not cleanly extractable in the fast 2nd-order setup
   (non-uniform FIRE + small-difference-of-large-energies). Needs Faber's high-order method. The
   robust α⁻¹→137 already stands on the charge route (P1b). Not a priority.

P3-P6 depend on a stable loop existing (item 1).

## Resume ping

Run 3: armed `trig_01SWTaaTUywLwKYva3FMJPLU` for reset+5min (4:05pm EDT 2026-06-23); disarmed at
FINISH (task completed well before the cap). Lossless via these checkpoints either way.
