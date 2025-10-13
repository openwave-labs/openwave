# SUMMARY

## PARTICLE-BASED WAVE DYNAMICS

Limitations:

- MAX PARTICLE COUNT = 1e6  # granularity for GPU optimized computational performance
- MAX UNIVERSE SIZE = 1e-15 # m, Nyquist sampling wave resolution (granules/lambda), for above particle count
  - only up to neutrino scale simulations (5e-17)
  - electron = 5e-15, nuclei = 1e-14, H atom = 1e-10

### SPRING-MASS (Force-Based Integration)

- Ordinary Differential Equation (ODE)

```python
# Compute spring force
F = -k * x
a = F / m
v += a * dt
x += v * dt
```

- NUMERICAL METHODS (computational solvers)
  - 1st-Order Accuracy: EULER INTEGRATION (explicit)
  - 2nd-Order Accuracy: LEAPFROG INTEGRATION (explicit)

- LIMITATIONS
  - Numerical instability (explosion) even with higher-order methods
  - Issue: Planck scale spring high stiffness
    - High stiffness = explosion
    - Lowering stiffness = non-fidelity to physics (toy-physics, "wet noodle" stiffness waves don't propagate)

  **THE IMPOSSIBLE TRIANGLE:**
  Cannot simultaneously achieve all three:
  - Realistic stiffness (for physics fidelity)
  - Numerical stability (no explosions)
  - Human-visible motion (slow-mo factor)

  Frequency mismatch: 360 million : 1 gap (spring natural frequency 380 MHz vs driving frequency ~1 Hz)
  This gap is unbridgeable with explicit integration methods.

```bash
         Realistic Stiffness
                / \
               /   \
              /     \
        Stability --- Human-Visible Motion
```

**Force-Based Thinking:**

> "Forces cause acceleration, which integrates to velocity, which integrates to position"
> (Dynamic → Kinematic)

### XPBD (Constraint-Based Integration)

- Constraint satisfaction method using Lagrange multipliers

```python
# Directly project positions to satisfy constraint
C(x) = distance - rest_length  # Constraint violation
Δx = -C(x) / ||∇C||²  # Position correction
x += Δx  # Move particles to satisfy constraint
```

- NUMERICAL METHODS (computational solvers)
  - XPBD INTEGRATION, Better for Stiff Systems (no springs = no stability limit - works like implicit solver)

- LIMITATIONS
  - Waves don't propagate properly, don't satisfy real physics fidelity
  - Issue: We're not using actual Planck-size granules, but instead a scaled-up version with larger granules and mass to become computationally feasible (max particle count)
    - Trade-off: can't accurately satisfy either wave properties perfectly: wave-speed (c) or wave-length (lambda)

**Constraint-Based Thinking:**

> "Positions must satisfy geometric constraints, velocities are consequence of position changes"
> (Kinematic → Dynamic)

**Both are valid physics!** XPBD just solves it "backwards" - and turns out to be more stable for stiff systems.

### PHASE-SYNCHRONIZED HARMONIC-OSCILLATORS (PSHO)

Then today I had an idea: remove springs and constraints and use synchronized phase between granules (harmonic phi rhythm) in the file radial_wave.py. A radial wave is point-sourced from the lattice center with propagation via synchronized phase shift - not force/constraint driving a position integrator, but instead a simple harmonic oscillation equation defining position over time for each granule.

Result: We got a perfect wave! I can clearly see the wavefronts and it matches both wave speed and lambda parameters.

I'm very happy with this (and my GPU as well, its also easier on computational load) - it's the fourth experiment. All four experiments are now available in OpenWave.

moved from:

- ❌ Force mechanics paradigm: Forces → Accelerations → Velocities → Positions (breaks down at quantum scale)
- ✅ Wave mechanics paradigm: Phase relationships → Direct position calculation (works perfectly!)

```python
All granules oscillate radially along their direction vectors to lattice center.
Phase is determined by radial distance from center, creating outward-propagating
spherical wave fronts. Granules at similar distances form oscillating shell-like
structures, with the wave originating from the lattice center.

# Directly project positions to satisfy constraint
Position: x(t) = x_eq + A·cos(ωt + φ)·direction
Velocity: v(t) = -A·ω·sin(ωt + φ)·direction (derivative of position)
Phase: φ = -kr, where
    k = 2π/λ is the wave number,
    r is the radial distance from center.
```

#### Direct harmonic oscillation - no integration needed

- displacement = amplitude * cos(ωt + φ₀)
- position = equilibrium + displacement * direction

Benefits:

1. ✅ Perfect wave speed - No numerical dispersion from discretization
2. ✅ Perfect wavelength - Phase relationship enforces exact λ
3. ✅ Unconditionally stable - No timestep constraints, no explosions
4. ✅ Computationally efficient - Just trigonometric functions, no constraint solving
5. ✅ Physically accurate - Matches EWT parameters exactly

The Journey / evolution:

1. Spring Forces (Euler) → Explosion (too stiff)
2. Spring Forces (Leapfrog) → Explosion (still too stiff)
3. XPBD Constraints → Stable but slow waves (~8x too slow)
4. Phase-Synchronized Oscillators → ✅ Perfect waves!

This is actually a profound realization: You can't simulate wave phenomena using particle mechanics at quantum scales - you need to simulate them as waves!

This validates EWT's wave-centric view - phase relationships are more fundamental than forces at quantum scale

## STILL NEEDS VALIDATION

Success criteria: Wave speed ≈ c AND wavelength ≈ λ (within 5-10% tolerance), using real physics parameters (k, m, lattice sizes, wave equations) AND medium natural resonant frequency.

- This will validate the entire physics model: correct k, m, lattice spacing, and wave equation

### WAVE INTERACTION NEEDS

- Wave Interference (constructive, destructive)
- Wave Reflection (particles, boundaries)
- MAP (minimum amplitude principle)

We're using classical physics wave equations, force-based integration methods are not feasible to reach numerical stability necessary when simulating extreme physics of small planck scale, high wave speeds (speed of light) in a high density aether medium (force mechanics vs wave mechanics)

>> high frequencies > high stiffness > high iterations needed and extremely low dt

This only confirms the energy contained in the quantum waves is huge, evidenced by
high forces and momentum impossible to compute because the math fails (the integration methods actually), its not even a computational feasibility issue, even if we had computer power to run

## OPTION: PURE WAVE GRID
