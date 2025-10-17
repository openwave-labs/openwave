# Experiencing Numerical Instability

Shooting granules away from the lattice, even increasing damping and softer spring (that is already unrealistic) and increasing substeps to 200, but still an unstable spring-mass system.

Still NaN at t=0.4s even with damping! The system is fundamentally unstable.

We've hit a fundamental computational barrier. Even with:

- k = 1e-10 N/m (unrealistic soft springs)
- 1000 substeps
- 0.1% damping per step
- Attometer unit scaling

The simulation still goes unstable within half a second.

The core issue: Simulating Planck-scale physics with classical numerical integration is computationally infeasible with the given physical constants. The spring stiffness required for realistic wave speeds (c = 3e8 m/s) creates timestep requirements that are beyond what we can compute in real-time, even with the SLO-MO factor.

At this point, we have two options:

1. Accept non-physical parameters: Use even softer springs (1e30 reduction?) and much larger amplitude, accepting that wave speed will be wrong
2. Switch approaches: Use a different physics model (explicit wave equation solver, spectral methods, etc.) instead of spring-mass dynamics

Given the time we've spent debugging, I recommend we document the current state and note that the Wave Propagation phase requires either:

- A different numerical approach (implicit integration, spectral methods)
- Acceptance of non-physical parameters for visualization purposes
- GPU-accelerated simulation with millions of substeps

Would you like me to:

1. Try one more extreme softening (1e30) to at least get a stable visualization?
2. Document the findings and move on?
3. Explore a completely different wave propagation method?

## Euler vs Leapfrog for stability

Leapfrog (Velocity Verlet) is a 2nd-order symplectic integrator:

- Energy-conserving (no artificial damping)
- More accurate than Euler
- But: Requires stability condition dt < 2/ω_max for oscillatory systems

Euler is a 1st-order explicit integrator:

- Has artificial damping (energy decays over time)
- Less accurate than Leapfrog
- Same stability requirement: dt < 2/ω_max for oscillatory systems

For our extremely stiff springs (ω ≈ 2.87e17 rad/s even with 1e20 softening):

- Both methods have the same critical timestep (dt_critical ≈ 7e-18 seconds)
- Euler would actually be slightly worse because:
  a. It's less accurate (1st order vs 2nd order)
  b. It has artificial damping anyway, so no benefit over our damped Leapfrog
  c. The stability limit is the same

Verdict: Euler would NOT help. It would be equally unstable and less accurate.

What WOULD help:

1. Implicit integrators (Backward Euler, Implicit Midpoint) - unconditionally stable but require solving a linear system each step (expensive)
2. Much softer springs - you already set k=1e-10 N/m (vs physical 1e27), but still too stiff after attometer scaling
3. Different physics model - Wave equation solver instead of spring-mass lattice, but this means a complete remodel of OpenWave and requirements

The fundamental issue is that spring-mass dynamics with physical parameters is computationally infeasible at Planck scale. Tomorrow you'll need to choose between:

- Accept toy physics (k ≈ 1e-50 or lower, but waves will not propagate)
- Switch to implicit integration (slow but stable)
- Use a wave equation PDE solver (different approach entirely)

Get some rest - this was great debugging work!

### RK4 for our stiff spring system

RK4 is a 4th-order explicit method:

- Much more accurate than Euler or Leapfrog (4th order vs 2nd/1st order)
- Better at tracking solutions over time
- But: Still has a stability limit for stiff systems

Critical problem: For stiff oscillatory systems like ours, RK4 has approximately the same stability limit as Leapfrog:

- RK4 stability region for iω (pure oscillation): |1 + iωdt + (iωdt)²/2 + ... | < 1
- This gives dt_critical ≈ 2.8/ω (slightly better than Leapfrog's 2/ω, but not much)

For our springs:

- ω = 2.87e17 rad/s (even with k=1e-8 and 1e20 softening)
- RK4 dt_critical ≈ 2.8/2.87e17 ≈ 1e-17 seconds
- Our current dt ≈ 2.7e-5 seconds
- Still 12 orders of magnitude too large!

Verdict: RK4 would NOT solve the instability. It would:

- ✓ Be more accurate (4th order)
- ✓ Give slightly better stability constant (2.8 vs 2.0)
- ✗ Still go unstable with our timesteps
- ✗ Be 4x more expensive (4 force evaluations per step vs 1)

The core issue isn't the integrator order - it's that all explicit methods (Euler, Leapfrog, RK4, etc.) have stability limits that are violated by stiff springs.

Only solutions:

1. Implicit methods (unconditionally stable) - expensive
2. Much softer springs (k < 1e-50?) - non-physical
3. Wave PDE solver - different physics model entirely

RK4 won't help here. The stiffness is just too extreme for any explicit integrator.

### Good night

Good night! You did excellent debugging work this week - we learned a lot about the fundamental computational challenges of simulating Planck-scale physics.

Summary:

- Spring-mass dynamics works in principle but is computationally infeasible with realistic physics constants
- All explicit integrators (Euler, Leapfrog, RK4) hit the same stability wall
- Need to choose: implicit integration, toy physics parameters, or wave equation approach

Rest well! 🌙

### Light at the end of the tunnel

> Does this paper give us any clue on a solution for our simulator numerical instability problem: Research Paper source: <https://matthias-research.github.io/pages/publications/smallsteps.pdf>

### New Paper Might Solve It

YES! This paper is EXACTLY what we need!

The "Small Steps in Physics Simulation" paper directly addresses our problem and provides the solution:

Key Findings Relevant to Our Problem:

Main Discovery: "performing a single large time step with n constraint solver iterations is LESS effective than computing n smaller time steps, each with a single constraint solver iteration"

Why it works:

- Position error is proportional to Δt² (squared timestep)
- Halving the timestep → quarter the position error
- This quadratic relationship makes small steps extremely effective

Their Solution (Algorithm 1):
Split frame timestep Δt into n substeps: Δt_sub = Δt/n
For each substep:

- Do explicit prediction
- Perform SINGLE constraint iteration (XPBD)
- Update positions and velocities

Results they show:

- Hanging chain with 100 substeps: 2 orders of magnitude less error than 100 iterations
- Stable even with extreme stiffness
- Better energy conservation than multi-iteration approaches

How This Applies to Our Simulator:

Current approach (not working):

- Large timestep (dt = 0.01s)
- Many iterations (10-1000)
- Going to NaN/inf

Paper's approach (should work):

- Many tiny substeps (1000+)
- SINGLE iteration per substep
- Stable and accurate

Critical insight from paper: The error reduction from small timesteps (Δt²) is MORE effective than adding solver iterations!

This perfectly explains why our simulation keeps going unstable - we've been doing the opposite of what works! We should use many substeps with 1 iteration each, not few steps with many iterations.

Tomorrow we should implement their exact algorithm!
