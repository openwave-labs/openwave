# M5.11 session state , ⏸ PARKED 2026-06-23 (5 runs done; resume guide below)

Run 1 go: 2026-06-22 20:43 EDT (P0-P2). Run 2 go: 2026-06-22 21:54 EDT (Taichi-AD, P1b+P2).
Run 3 go: 2026-06-23 12:34 EDT (P2 heliknoton build: chiral Lifshitz + Frank terms).
Run 4 go: 2026-06-23 14:18 EDT (P2 singular disclination loop ± chiral).
Run 5 go: 2026-06-23 14:28 EDT (P2 singular Hopfion: Hopf charge + painted melt).
Plan: [`11a_vortex_loop.md`](../../11a_vortex_loop.md). Findings: [`11b_findings.md`](../../11b_findings.md).

## ⏸ PARKED , pick up here

M5.11 is parked in a **clean** state: every run FINISHED with findings written, nothing is
mid-flight, and re-entry is a fresh decision (not the continuation of a stalled run). The
credibility wins are SECURED; the neutrino-loop construction is the open frontier, now precisely
isolated.

**DONE and not to re-litigate (bankable, answers Duda's "too simple" critique):**
- Faber's **ELECTRON** reproduced , 511.00 keV at `r0=2.2132 fm`, dimensionless `I=π/4`, non-circular (P1a).
- The **fine-structure constant `α⁻¹→137.03`** from charge quantization (P1b).
- The **Taichi-AD gradient engine** == the production functional to 1e-13 (run 2).
- The **chiral Lifshitz term + its Frank partner** added + validated to 1e-14 (run 3).

**The open problem, stated precisely (the 2×2, see Open item 1):** the neutrino needs a
configuration that is BOTH **forced-singular** (a half-integer director singularity = a melt that
cannot heal) AND **knotted/linked** (topological protection). Every shortcut fills a wrong cell ,
smooth knots expand, unknotted singular loops contract, a painted-on melt heals. The one un-filled
cell = a **knotted/linked HALF-INTEGER DISCLINATION LINE**.

**Three ways back in (pick by appetite; B is the lowest-risk re-entry):**

- **A , build the genuine object (direct, HIGH risk).** Get an explicit single-valued director
  ansatz for a Hopf-linked pair of `+1/2` disclination rings (or a `+1/2` trefoil) from the
  disclination-topology literature (Machon & Alexander, PNAS 2013, "Knots and nonorientable
  surfaces in chiral nematics"; Alexander, Chen, Matsumoto, Kamien, RMP 2012 disclination-loops
  Colloquium), seed it, relax under the validated `U`. The crux is single-valuedness (the director
  winding must close consistently) , start from a literature parametrization, do not improvise it.
- **B , uniaxial-director reduction (de-risk, LOW risk, recommended re-entry).** Reduce M5 from the
  biaxial 3×3 tensor to a unit director `n` with Frank `K|∇n|²` + chiral `2q0 n·(∇×n)`, where
  Smalyukh's heliknotons are KNOWN to be stable. Demonstrate the heliknoton there first (seed from
  Ackerman-Smalyukh / the Tai thesis Fig 6.5), THEN map back to biaxial M5 to see what breaks.
  Smaller functional (3 components vs 16), the chiral term is simpler in director form. This both
  validates the chiral machinery on the object it is meant for and tells us whether biaxiality is
  the obstruction.
- **C , accept M5-as-is = electron only (framing call, NO risk).** Treat the electron + `α⁻¹=137`
  as the deliverable to Duda and document the neutrino loop as requiring an added substrate
  ingredient (chiral-in-uniaxial, or a different stabilizer). Decide whether the program needs the
  loop now or whether the electron result is the report.

**Honest negatives , do NOT naively retry:** running `α(d)` in the fast 2nd-order setup (needs
Faber's high-order method, P1b'); the smooth / painted-melt knots and the biaxial cholesteric
helix (11b runs 3-5).

**Machinery is ready:** [`v11_p2_heliknoton.py`](../v11_p2_heliknoton.py) carries the validated
functional (`4th-order + V_M + Frank + chiral`) + seeds (`disclination_loop_tensor`,
`singular_hopfion_tensor`, `heliknoton_director`) + diagnostics (`melt_diag`, `director_ring_R`,
`curvature_only`). The Taichi kernel is offline-cached; the FIRST run after any kernel-source edit
recompiles (~6 min on this machine), driver-only edits are free.

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
| P2 run4 singular disc loop | `v11_p2_heliknoton.py` mode `disc` | melted-core +1/2 loop DISSOLVES under L=0 AND chiral (melt heals, vol→0); unknotted = unprotected |
| P2 run5 singular Hopfion | `v11_p2_heliknoton.py` mode `shopf` | painted melt on a smooth Hopf knot HEALS (vol 904→0 by it 100) → EXPANDS (all melt depths). melt must be FORCED, not painted |
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

1. **P2 next , a FORCED-SINGULAR knotted/linked disclination LINE (the one un-filled cell).**
   Five clean negatives now map onto a 2×2 (smooth-vs-forced-singular director) × (unknotted-vs-
   knotted):
     - smooth + unknotted = run-2 Hopfion → EXPANDS
     - smooth + knotted = run-5 singular Hopfion → painted melt HEALS → EXPANDS
     - forced-singular + unknotted = run-1/4 `+1/2` loop → CONTRACTS/DISSOLVES
     - forced-singular + knotted/linked = **THE TARGET, still unbuilt**
   Run 5's sharp lesson: a melt PAINTED on a smooth field heals (the Hopf director is non-singular,
   so amplitude refills to lower V_M); a melt only stabilizes when a genuine half-integer director
   singularity FORCES it (why the electron hedgehog is stable: hairy-ball forbids a defined
   director at the point). So the shortcut "melt a Hopfion" lands in the smooth-knot cell and fails.
   The target needs the director **genuinely singular (half-integer) ALONG a knotted/linked line**
   , two Hopf-linked `+1/2` disclination rings, or a `+1/2` trefoil. The hard part = a single-valued
   director with a knotted/linked half-integer disclination (cannot be shortcut by painting a melt).
   Decision the next build faces: build that genuine object, OR move to the **uniaxial-director
   reduction** where Smalyukh's chiral heliknotons are known to live (then map back to biaxial M5),
   OR accept that the M5 functional as-is stabilizes only the point hedgehog (electron) and the
   neutrino needs an added substrate ingredient. Machinery (functional + seeds + diagnostics) ready
   in `v11_p2_heliknoton.py`.
2. **Running α(d) (P1b') , honest negative (2026-06-23).** `v11_p1b_running.py` built + validated
   (single soliton gradcheck 2e-6); `α_sol` not cleanly extractable in the fast 2nd-order setup
   (non-uniform FIRE + small-difference-of-large-energies). Needs Faber's high-order method. The
   robust α⁻¹→137 already stands on the charge route (P1b). Not a priority.

P3-P6 depend on a stable loop existing (item 1).

## Resume ping

Runs 3-5: armed `trig_01SWTaaTUywLwKYva3FMJPLU` for reset+5min (4:05pm EDT 2026-06-23), disarmed
at each FINISH (all completed well before the cap). Lossless via these checkpoints either way.
