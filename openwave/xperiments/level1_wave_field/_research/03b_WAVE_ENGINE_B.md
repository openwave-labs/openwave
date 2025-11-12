# WAVE ENGINE - FORCE CALCULATION & WAVE PROPAGATION

## Notation Clarification: ψ vs A

**Two Distinct Physical Quantities - Both Needed!**

### 1. ψ (psi): Instantaneous Displacement

- **What it is**: The actual wave displacement at each instant in time
  - Oscillates rapidly at wave frequency (~10²⁵ Hz for energy waves)
  - Can be positive or negative
  - Varies: ψ(x,y,z,t)
  - **Propagates via wave equation**: ∂²ψ/∂t² = c²∇²ψ

- **In code**: `self.displacement_am[i,j,k]`
- **Used for**:
  - Wave propagation (PDEs, Laplacian)
  - Wave mode analysis (longitudinal vs transverse)
  - Phase calculations
  - Instantaneous field values

### 2. A: Amplitude Envelope

- **What it is**: The **maximum displacement** at each location (envelope)
  - For sinusoidal wave: ψ(x,t) = A(x) sin(kx - ωt)
  - A is the peak: |ψ|max = A
  - Always positive: A ≥ 0
  - Slowly varying (envelope of high-frequency oscillation)
  - **Tracked as running maximum** of |ψ| over time

- **In code**: `self.amplitude_am[i,j,k]`
- **Used for**:
  - **Energy density**: u = ρ(fA)² (EWT, no ½ factor, frequency-centric)
  - **Force calculation**: F = -2ρVfA×[f∇A + A∇f] or F = -2ρVf²×A∇A (MAP: Minimum **Amplitude** Principle)
  - Energy gradients
  - Pressure-like field that drives particle motion

### Why Two Fields Are Needed

**The High-Frequency Problem**:

- Energy waves oscillate at ~10²⁵ Hz (from EWT)
- Particles have mass/inertia - cannot respond to every oscillation
- Particles respond to **time-averaged** force = force from **envelope** (A)

**Analogy** (Speaker Diaphragm):

- **ψ**: Diaphragm position oscillating at audio frequency
- **A**: "Volume" setting - controls maximum displacement
- You feel air pressure from **A** (volume), not individual oscillations (ψ)

### Implementation Strategy

**Wave Equation** propagates ψ (displacement):

```python
# High-frequency oscillation (updated every timestep)
∂²ψ/∂t² = c²∇²ψ
self.displacement_am[i,j,k]  # Stores current ψ
```

**Amplitude Tracking** extracts envelope A from ψ:

```python
# Track maximum |ψ| over time (envelope extraction)
@ti.kernel
def track_amplitude_envelope(self):
    for i, j, k in self.displacement_am:
        disp_mag = ti.abs(self.displacement_am[i,j,k])
        ti.atomic_max(self.amplitude_am[i,j,k], disp_mag)
```

**Force Calculation** uses A (not ψ):

```python
# Particles respond to amplitude gradient (envelope)
F = -∇A  # Not -∇ψ !
F = -(∂A/∂x, ∂A/∂y, ∂A/∂z)
```

### Summary Table

| Property | ψ (Displacement) | A (Amplitude) |
|----------|------------------|---------------|
| **Field name** | `displacement_am[i,j,k]` | `amplitude_am[i,j,k]` |
| **Physics** | Instantaneous oscillation | Envelope (max \|ψ\|) |
| **Frequency** | High (~10²⁵ Hz) | Slowly varying |
| **Sign** | ± (positive/negative) | + (always positive) |
| **Propagation** | Wave equation (PDE) | Tracked from ψ |
| **Used for** | Wave dynamics, phase, mode | Forces, energy, MAP |
| **Formula** | ∂²ψ/∂t² = c²∇²ψ | A = max(\|ψ\|) over time |

**Critical Point**: Forces use **amplitude gradient** (∇A), not displacement gradient (∇ψ)! This is because MAP = "Minimum **Amplitude** Principle" - particles move toward regions of lower amplitude envelope, not lower instantaneous displacement.

## Table of Contents

1. [Notation Clarification: ψ vs A](#notation-clarification-ψ-vs-a)
1. [Questions](#questions)
   - [Part 1: Force Calculation in Newtons](#part-1-force-calculation-in-newtons)
   - [Part 2: Wave Amplitude Propagation](#part-2-wave-amplitude-propagation)
   - [Part 3: PDE vs Huygens Wavelets](#part-3-pde-vs-huygens-wavelets)
   - [Part 4: Wave Direction in PDE Propagation](#part-4-wave-direction-in-pde-propagation)
1. [Complete Answers](#complete-answers)
   - [Answer 1: Force in Newtons from Amplitude Gradient](#answer-1-force-in-newtons-from-amplitude-gradient)
   - [Answer 2: Wave Propagation Mechanics](#answer-2-wave-propagation-mechanics)
   - [Answer 3: Choosing Between PDE and Huygens](#answer-3-choosing-between-pde-and-huygens)
   - [Answer 4: Wave Direction Computation](#answer-4-wave-direction-computation)
1. [The Complete Picture](#the-complete-picture)
1. [Implementation Summary](#implementation-summary)

---

## Questions

### Part 1: Force Calculation in Newtons

**Context**: Understanding force calculation in LEVEL-1 wave-field simulation.

**Known**:

- Each voxel stores wave displacement ψ in attometers
- Each voxel tracks amplitude envelope A (max|ψ|) in attometers
- Force follows MAP (Minimum Amplitude Principle): `F = -∇A`
- Amplitude gradient can be calculated from neighboring voxels using finite differences

**Question**:

Can we compute force in **Newtons** (kg⋅m/s²) from the amplitude gradient?

**Requirements**:

- Need force in Newtons at each voxel to calculate particle acceleration
- Given particle mass (calculated from standing wave radius), find acceleration: `a = F/m`
- With acceleration, we can integrate motion: velocity update → position update
- Particles initially are single wave centers (fundamental particles like neutrino), later becoming standalone particles (like electron)
- Particle mass likely comes from standing waves reflected around particle radius (related to wavelength λ)

**Resources**: EWT research papers at `/research_requirements/scientific_source/`

---

### Part 2: Wave Amplitude Propagation

**Context**: Understanding how wave properties propagate through the field.

**System Overview**:

1. **Wave Field**: The medium (grid of voxels)
2. **Energy Charge**: Initial energy injected into system
3. **Wave Propagation**: Wave properties propagate in wave-like motion:
   - Amplitude, displacement
   - Wave direction, speed
   - Energy, phase
   - Frequency, Wavelength
4. **Propagation Methods** (unclear which to use):
   - PDEs (Partial Differential Equations)?
   - Laplacian operator?
   - Wave equation?
   - Huygens wavelets?
5. **Conservation**: How does propagation conserve/transfer energy and momentum?

**Wave Interactions**:

- Waves interfere with each other (constructive/destructive)
- Waves reflect from:
  - Universe boundaries (grid boundaries)
  - Wave centers (fundamental particles)
- Reflection creates standing waves (interference of inward/outward waves)
- Standing waves give particles mass (trapped energy)

**Particle Behavior**:

- Particles interact with each other, forming complex structures
- Particles move to minimize amplitude (MAP principle)
- Movement toward lower amplitude regions is the effect of force
- Force must have value in Newtons

**Fundamental Forces**:

All forces in nature emerge from wave interactions:

- Electric force
- Magnetic force
- Gravitational force
- Strong force
- Orbital mechanics

All generated from the fundamental energy wave (EWT).

**Resources**: Documentation files in `/openwave/xperiments/level1_field_based/_docs/`

---

### Part 3: PDE vs Huygens Wavelets

**Question**: Should we choose between wave equation (PDE) and Huygens wavelets, or use both?

**Sub-questions**:

- What are the pros and cons of each approach?
- Which is more computationally performant?
- Can they be combined?
- Which better represents EWT physics?

---

### Part 4: Wave Direction in PDE Propagation

**Question**: How is the direction of wave propagation computed and stored when using PDE-based wave equation propagation?

**Context**:

- Wave equation: `∂²ψ/∂t² = c²∇²ψ` only evolves amplitude
- Need wave direction for:
  - Force calculations
  - Momentum transfer
  - Interference patterns
  - Particle-wave interactions

---

## Complete Answers

### Answer 1: Force in Newtons from Amplitude Gradient

#### YES, You Can Compute Force in Newtons

**Key Formula from EWT (Frequency-Based)**:

```text
Energy density: u = ρ(fA)²        [J/m³]  (EWT, no ½ factor)
Force density:  f = -∇u           [N/m³]
Force on voxel: F = f × dx³       [N]
Final formula:  F = -2ρVfA×[f∇A + A∇f]  (full form with frequency gradients)
Monochromatic:  F = -2ρVf²×A∇A           (when ∇f = 0)
```

Where:

- `ρ` = medium density (3.860×10²² kg/m³ from EWT)
- `f` = frequency (Hz, where f = c/λ)
- `A` = wave amplitude (meters)
- `V = dx³` = voxel volume (cubic meters)
- Note: c = 2.998×10⁸ m/s embedded in f via f = c/λ

#### Physics Derivation

**Energy in wave field (EWT formulation - Frequency-Based)**:

```text
Total energy: E = ∫ u dV
where u = ρ(fA)² is energy density (from EWT)
```

**Note**: EWT energy equation `E = ρV(fA)²` does **not** include the ½ factor found in classical wave mechanics.

**Elegance of Frequency Formulation**:

- **E = ρV(fA)²** is cleaner than E = ρVc²(A/λ)²
- Aligns with **Planck's E = hf** (energy proportional to frequency!)
- **Spacetime coupling**: f (temporal) × A (spatial) = natural pairing
- Human-intuitive: frequency used in radio (98.7 FM), audio (440 Hz), WiFi (2.4 GHz)

**Physical Meaning of EWT Formula**:

In oscillating wave systems, energy alternates between kinetic (medium motion) and potential (compression):

```text
Classical time-averaged: ⟨E⟩ = ½ρV(fA)²   (average over cycle)

EWT total/peak energy:   E = ρV(fA)²      (total energy capacity)
                         E = 2 × ⟨E⟩classical
```

**Why use total instead of average?**

- Frequency f represents **oscillation rate** (temporal character)
- Amplitude A represents **peak displacement** (maximum pressure)
- Forces respond to **total energy gradients** (peak pressure differences)
- `E_EWT = max(KE + PE)` = total energy that sloshes between kinetic and potential
- This is the "energy budget" that creates pressure gradients driving particle motion

Analogy: Sound pressure - objects respond to peak amplitude (loudness) at a given frequency (pitch), not time-averaged sound.

**Force from total energy gradient**:

```text
F = -∇E
  = -∇[ρV(fA)²]
  = -ρV × ∇(A²f²)
  = -ρV × [f²∇(A²) + A²∇(f²)]
  = -ρV × [f² × 2A∇A + A² × 2f∇f]
  = -2ρV × [f²A × ∇A + A²f × ∇f]
  = -2ρVfA × [f∇A + A∇f]
```

Where `V = dx³` is voxel volume.

**Implementation Note**: The factor of 2 comes from chain rule (∇(A²f²) = 2A∇A×f² + 2A²f∇f) and remains in the final force formula.

**EWT vs Classical Comparison**:

```text
Classical wave mechanics:
  u = ½ρ(fA)²  → F = -ρVfA × [f∇A + A∇f]

EWT (used in this simulation):
  u = ρ(fA)²   → F = -2ρVfA × [f∇A + A∇f]

The EWT force constant is 2× classical due to using total energy (no ½ factor).
Use EWT formulation for consistency with energywavetheory.com

Wavelength relation (when needed for spatial design):
  λ = c/f  (c = 2.998×10⁸ m/s constant)
```

#### Implementation

```python
@ti.kernel
def compute_force_field_newtons(self):
    """
    Compute force in Newtons from amplitude gradient (EWT formulation).

    Physics (Frequency-Based):
    - Energy density: u = ρ(fA)² (EWT, no ½ factor)
    - Force: F = -∇E = -∇(u×V) = -2ρVfA × [f∇A + A∇f]
    - Monochromatic: F = -2ρVf² × A∇A (when ∇f = 0)
    where V = dx³ (voxel volume)

    Note: Factor of 2 from chain rule remains in final formula.
    This differs from classical wave mechanics by factor of 2.

    MAP Principle: Force points toward lower amplitude (negative gradient)
    """
    # Physical constants from EWT
    ρ = ti.f32(constants.MEDIUM_DENSITY)  # 3.860e22 kg/m³
    f = ti.f32(constants.EWAVE_FREQUENCY) # 1.050e25 Hz
    dx_m = self.dx_am * constants.ATTOMETER  # voxel size in meters
    V = dx_m**3  # voxel volume in m³

    # Force scaling factor (EWT frequency-based formulation with factor of 2)
    # F = -2ρVf² × A × ∇A  (monochromatic, ∇f = 0)
    # Dimensional analysis: 2 × (kg/m³)(m³)(Hz²) = 2 × (kg)(1/s²) = kg⋅m/s² when multiplied by A∇A
    force_scale = 2.0 * ρ * V * f**2

    for i, j, k in self.amplitude_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Local amplitude in meters
            A_m = self.amplitude_am[i,j,k] * constants.ATTOMETER

            # Amplitude gradient (dimensionless: am/am)
            grad_x = (self.amplitude_am[i+1,j,k] - self.amplitude_am[i-1,j,k]) / (2.0 * self.dx_am)
            grad_y = (self.amplitude_am[i,j+1,k] - self.amplitude_am[i,j-1,k]) / (2.0 * self.dx_am)
            grad_z = (self.amplitude_am[i,j,k+1] - self.amplitude_am[i,j,k-1]) / (2.0 * self.dx_am)

            grad_vector = ti.Vector([grad_x, grad_y, grad_z])

            # Force in Newtons (MAP: toward lower amplitude)
            # F = -force_scale × A × ∇A
            self.force[i,j,k] = -force_scale * A_m * grad_vector  # N = kg⋅m/s²
```

#### Force Direction Vector

**Understanding Force Direction**:

The force is a **vector** quantity with both magnitude and direction:

```text
F = -∇A = -(∂A/∂x, ∂A/∂y, ∂A/∂z)
```

**Direction Meaning**:

- **Negative gradient**: Force points toward **decreasing** amplitude (MAP principle)
- **Each component**:
  - `F_x`: Force component in x-direction
  - `F_y`: Force component in y-direction
  - `F_z`: Force component in z-direction
- **Magnitude**: `|F| = √(F_x² + F_y² + F_z²)` [Newtons]
- **Direction**: `F_hat = F / |F|` (unit vector)

**Physical Interpretation**:

```python
# Example: Force vector at a voxel
F = [-2.5e-30, 1.0e-30, 0.0]  # Newtons

# This means:
# - Force pulls particle in -x direction (2.5e-30 N)
# - Force pushes particle in +y direction (1.0e-30 N)
# - No force in z direction

# Magnitude
|F| = sqrt(2.5² + 1.0²) × 1e-30 = 2.69e-30 N

# Direction (unit vector)
F_hat = [-0.928, 0.371, 0.0]
# Pointing mostly in -x direction, slightly in +y
```

**Why This Matters for Particle Motion**:

When a particle at position `(x, y, z)` experiences this force:

1. **Interpolate force** from nearby voxels to particle position
2. **Vector components** determine motion in 3D space
3. **Acceleration direction**: `a = F/m` (same direction as force)
4. **Velocity change**: Particle accelerates along force direction
5. **Position update**: Particle moves toward lower amplitude (MAP)

**Computing Force Direction (if needed separately)**:

```python
@ti.func
def get_force_direction(i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:
    """Get unit vector in direction of force at voxel [i,j,k]."""
    F = self.force[i,j,k]
    F_mag = F.norm()

    if F_mag > 1e-40:  # Avoid division by zero
        return F / F_mag  # Normalized direction
    else:
        return ti.Vector([0.0, 0.0, 0.0])  # No preferred direction
```

**Key Point**: The gradient operator `∇A` automatically gives you the **direction** (as a vector) pointing toward maximum amplitude increase. The negative sign `-∇A` reverses this to point toward amplitude **decrease** (MAP principle).

#### Dimensional Analysis Verification (Frequency-Based)

```text
ρ:          [kg/m³]
f:          [Hz] = [1/s]
f²:         [1/s²]
V = dx³:    [m³]
A:          [m]
∇A:         [dimensionless] = [m/m] after gradient divided by dx

force_scale = (kg/m³) × (m³) × (1/s²) = kg/s²
F = force_scale × A × ∇A
F = (kg/s²) × (m) × (dimensionless) = kg⋅m/s² = N  ✓✓
```

**This is correct!** Note how f² naturally provides the 1/s² dimension without needing explicit c².

**Relation to wavelength-based form**:

```text
f = c/λ  →  f² = c²/λ²

force_scale (frequency) = 2ρVf² = 2ρV(c²/λ²) = force_scale (wavelength)  ✓
```

#### Particle Mass from Standing Waves

**From EWT Papers**:

- Particle mass comes from **trapped energy** in standing waves
- For electron: `E_electron = (μ₀c²/4π) × (e_e²/r_e)`
- Mass-energy relation: `m = E/c²`

**In Simulation**:

- Particle standing wave radius: `r = n × λ/2` (nodes at half-wavelengths)
- Energy trapped: `E = ∫ u dV` over standing wave volume
- Particle mass: `m = E/c²`

**Standing wave formation**:

1. Wave center reflects incoming waves
2. Reflected waves interfere with incoming waves
3. Constructive interference creates standing wave pattern
4. Energy trapped in standing wave = particle mass
5. Radius of first node determines particle size

#### Particle Acceleration and Motion

```python
# For particle at position pos_am (in attometers)
# 1. Interpolate force from grid to particle position
F_particle = interpolate_force(self.force, pos_am)  # Newtons

# 2. Newton's second law: F = ma
a = F_particle / particle_mass  # m/s²

# 3. Integrate motion using Velocity Verlet method
v_new = v_old + a * dt  # m/s
pos_new = pos_old + v_new * dt  # meters (or attometers)
```

**This IS the origin of all forces!**

- Electric force: Different wave reflection patterns (charge types)
- Magnetic force: Moving wave patterns (velocity-dependent)
- Gravitational force: Amplitude shading from trapped energy (mass)
- All emerge from wave amplitude gradients!

---

### Answer 2: Wave Propagation Mechanics

#### How Waves Propagate Through Voxels

LEVEL-1 uses **PDEs (Partial Differential Equations)** to propagate waves through the field.

#### The Classical Wave Equation

**3D Wave Equation** (fundamental):

```text
∂²ψ/∂t² = c²∇²ψ
or,
ψ" = c²Δψ
```

Where:

- `ψ` = wave displacement field (scalar)
- `c` = wave propagation speed (speed of light, 2.998×10⁸ m/s)
- `∇²ψ` = Laplacian operator (second-order spatial derivative)
- `∂²ψ/∂t²` = second-order time derivative (acceleration of displacement)

**Physical Interpretation**:

- Left side: How fast displacement is accelerating in time
- Right side: How much displacement differs from neighbors (curvature)
- Equation says: "Displacement accelerates toward its neighbors' average"

#### Laplacian Operator (How Voxels Share Displacement)

**Discrete Laplacian** (6-connectivity, face neighbors only):

```python
# Laplacian at voxel [i,j,k]
∇²ψ[i,j,k] = (
    ψ[i+1,j,k] + ψ[i-1,j,k] +  # Left/right neighbors (x-direction)
    ψ[i,j+1,k] + ψ[i,j-1,k] +  # Front/back neighbors (y-direction)
    ψ[i,j,k+1] + ψ[i,j,k-1] -  # Top/bottom neighbors (z-direction)
    6 × ψ[i,j,k]                # Central voxel (6 neighbors)
) / dx²
```

**Physical Meaning**:

- Laplacian measures how much a voxel's displacement differs from its neighbors' average
- Positive Laplacian: voxel lower than average → displacement will increase
- Negative Laplacian: voxel higher than average → displacement will decrease
- This drives wave propagation: differences smooth out over time

#### Boundary Reflection

Boundary walls emulate all matter in the universe reflecting the energy waves.
So, total energy is conserved inside that volume domain.

Boundary handling (Dirichlet boundary conditions):

1. Propagation loop: ti.ndrange((1, nx - 1), (1, ny - 1), (1, nz - 1))

- Only updates interior points (excludes boundaries at i=0, i=max, j=0, j=max, z=0, z=max)
- Boundary values remain at ψ = 0 (from initialization)

2. Laplacian operator:

- Accesses neighbors directly without bounds checking
- When called on interior points, it reads boundary values (which are always ψ = 0)

This creates :

- Fixed displacement ψ = 0 at all boundaries
- Acts like a rigid wall - waves should reflect back

The boundary behavior is NOT in the Laplacian itself - it's implemented through:

1. Keeping boundaries fixed at zero (never updated)
2. Interior points "see" zero at boundaries when computing Laplacian

#### Wave Superposition

Wave superposition after reflection.
Superposition principle.

#### Time Evolution Implementation

```python
@ti.kernel
def propagate_wave_field(dt: ti.f32):
    """
    Propagate wave displacement using wave equation.

    Second-order in time (requires storing two previous timesteps):
    ψ_new = 2ψ_current - ψ_old + (c×dt/dx)² × ∇²ψ

    This is a centered finite difference scheme, second-order accurate.
    """
    # Speed of light and CFL factor
    c = ti.f32(constants.EWAVE_SPEED)
    cfl_factor = (c * dt / self.dx_am)**2

    # CFL stability condition: cfl_factor ≤ 1/3 for 3D (6-connectivity)
    # If violated, solution becomes unstable

    # Update all interior voxels
    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Compute Laplacian (6-connectivity)
            laplacian = (
                self.displacement_am[i+1,j,k] + self.displacement_am[i-1,j,k] +
                self.displacement_am[i,j+1,k] + self.displacement_am[i,j-1,k] +
                self.displacement_am[i,j,k+1] + self.displacement_am[i,j,k-1] -
                6.0 * self.displacement_am[i,j,k]
            ) / (self.dx_am * self.dx_am)

            # Wave equation update (leap-frog scheme)
            self.displacement_new[i,j,k] = (
                2.0 * self.displacement_am[i,j,k]  # Current displacement
                - self.displacement_old[i,j,k]      # Previous displacement
                + cfl_factor * laplacian          # Wave propagation term
            )

    # Swap timesteps for next iteration
    # old ← current ← new
    self.displacement_old, self.displacement_am = self.displacement_am, self.displacement_new
```

**Storage Requirements**:

- Three displacement fields: `displacement_old`, `displacement_am` (current), `displacement_new`
- Needed for second-order time integration

**Stability Condition** (CFL - Courant-Friedrichs-Lewy):

```text
dt ≤ dx / (c√3)  for 3D, 6-connectivity

Example:
dx = 1.25 am = 1.25e-18 m
c = 2.998e8 m/s
dt_max = 1.25e-18 / (2.998e8 × √3) ≈ 2.4e-27 s
```

**This is extremely small!** Timesteps are ~10⁻²⁷ seconds (rontosecond scale).

**Rontosecond Scaling Solution**:

Just as spatial scales use attometer (10⁻¹⁸ m) scaling for numerical precision, **temporal scales use rontosecond (10⁻²⁷ s) scaling**:

```python
# Scaling constants
ATTOMETER = 1e-18     # m, attometer length scale
RONTOSECOND = 1e-27   # s, rontosecond time scale

# Convert timestep to rontoseconds
dt_rs = dt / constants.RONTOSECOND

# Example:
dt = 2.4e-27  # Physical timestep in seconds (SI)
dt_rs = 2.4e-27 / 1e-27 = 2.4  # Scaled to rontoseconds
```

**Benefits**:

- Maintains numerical precision with f32 (6-7 significant digits)
- Prevents catastrophic cancellation in time derivatives
- dt_rs values ~1.0 (optimal range for f32 precision)
- Naming convention: `_am` for spatial (attometers), `_rs` for temporal (rontoseconds)

#### Timestep Strategy: Fixed vs Elapsed Time

**CRITICAL DECISION**: LEVEL-1 must use **fixed timesteps** (unlike LEVEL-0's elapsed time approach).

**Why LEVEL-0 Uses Elapsed Time**:

```python
# LEVEL-0 (particle-based, no CFL constraint)
elapsed_t = time.time() - previous_time  # Variable (0.001-0.1s typical)
update_particles(elapsed_t)  # Particles can handle variable dt
```

**Pros**: Real-time sync, adapts to frame rate, good for interactive visualization

**Cons**: Non-deterministic, timing depends on system performance

**Why LEVEL-1 CANNOT Use Elapsed Time**:

```python
# Wave equation CFL requirement
dt_max = dx / (c√3) ≈ 2.4e-27 s  # MUST NOT EXCEED!

# But elapsed time is typically:
elapsed_t ≈ 0.001 to 0.1 s  # Frame time (milliseconds)

# Ratio: elapsed_t / dt_max ≈ 10^24
# Result: IMMEDIATE NUMERICAL EXPLOSION 💥
```

**The wave equation becomes unstable if dt > dt_max**. Using elapsed time would violate CFL by ~24 orders of magnitude!

**LEVEL-1 Solution: Fixed Timestep + Frame Accumulator**:

```python
# Fixed physics timestep (respects CFL)
dt_physics = 2.0e-27  # seconds (or 2.0 in rontoseconds)
dt_physics_rs = 2.0   # rontoseconds (scaled)

# Hybrid approach: decouple physics from rendering
accumulated_time = 0.0

def main_loop():
    previous_time = time.time()

    while running:
        # Measure elapsed real time
        current_time = time.time()
        elapsed_t = current_time - previous_time
        previous_time = current_time

        # Accumulate time for physics
        accumulated_time += elapsed_t

        # Run fixed timesteps until caught up
        while accumulated_time >= dt_physics:
            update_physics(dt_physics_rs)  # Fixed dt (rontoseconds)
            accumulated_time -= dt_physics

        # Render at variable rate (decoupled from physics)
        render_frame()
```

**Benefits of Fixed Timestep Approach**:

- ✓ **Guarantees CFL stability** (dt always ≤ dt_max)
- ✓ Deterministic results (reproducible simulations)
- ✓ Accurate physics regardless of frame rate
- ✓ Can run faster or slower than real-time
- ✓ Can save/replay exact simulation states

**Comparison Table**:

| Aspect | LEVEL-0 (Elapsed) | LEVEL-1 (Fixed) |
|--------|-------------------|-----------------|
| **Timestep** | Variable (frame-dependent) | Fixed (CFL-limited) |
| **Stability** | Robust to large dt | Requires dt ≤ dt_max |
| **Real-time sync** | Perfect | Approximate (via accumulator) |
| **Deterministic** | No (varies per run) | Yes (reproducible) |
| **Physics accuracy** | Euler integration (acceptable) | PDE solver (requires fixed dt) |
| **Use case** | Interactive particle systems | Scientific wave simulation |

**Recommendation**: LEVEL-1 **MUST** use fixed timesteps for numerical stability. The hybrid accumulator approach allows real-time rendering while maintaining stable physics.

#### Alternative: Huygens Wavelets

**Huygens' Principle**: Every point on a wavefront acts as a source of secondary wavelets.

**Conceptual Implementation**:

```python
@ti.kernel
def propagate_huygens(dt: ti.f32):
    """
    Propagate using Huygens wavelets.
    Each voxel emits wavelets to neighbors.

    Note: This is less commonly used for regular grids because
    the wave equation (PDE) naturally implements Huygens' principle.
    """
    c = ti.f32(constants.EWAVE_SPEED)
    propagation_distance = c * dt

    for i, j, k in self.displacement_am:
        if ti.abs(self.displacement_am[i,j,k]) > threshold:
            # This voxel emits wavelets to neighbors
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        if di == 0 and dj == 0 and dk == 0:
                            continue  # Skip self

                        # Neighbor indices
                        ni, nj, nk = i + di, j + dj, k + dk

                        # Boundary check
                        if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                            # Distance to neighbor
                            distance = ti.sqrt(ti.f32(di*di + dj*dj + dk*dk)) * self.dx_am

                            # Wavelet contribution (inverse distance weighting)
                            contribution = self.displacement_am[i,j,k] / distance

                            # Add to neighbor (superposition)
                            # Note: This is simplified, full implementation needs proper weighting
                            ti.atomic_add(self.amplitude_new[ni,nj,nk], contribution * dt)
```

**Note**: Huygens method is less efficient on regular grids. The wave equation (PDE) implicitly implements Huygens' principle through the Laplacian operator.

#### Initial Energy Charging: Match EWT Energy Equation

**Context**: When initializing the wave field, we need to charge it with the correct amount of energy as specified by the EWT energy wave equation from `equations.py`.

**EWT Energy Wave Equation** (wavelength-based form):

```python
E = ρV(c/λ × A)²
```

**Frequency-centric equivalent**:

```python
E = ρV(fA)²    # Since f = c/λ
```

**Critical Requirements**:

1. **Match Total Energy**: Initial field energy must equal `energy_wave_equation(volume)` from equations.py
2. **Correct Wave Characteristics**: Use proper frequency, amplitude, wavelength from constants
3. **Simple Initial Condition**: DON'T try to create particle standing waves yet - those emerge automatically later
4. **Energy Conservation**: Wave equation will maintain total energy during propagation

Charge Energy Wave

- injection of n (even) pulses with universe energy (eq)
  - phase 0, time t: max positive displacement
  - phase π, time: t + 1 wave period: min negative displacement
- vs. start with a sine-wave of displacement (wave driver)
  - that will be picked up by stencil voxels

- hard coded point source or multiple sources (fast charge)
  - direction = spherical
- UI button, counter til stable

**Standing Wave Particles Come Later**: The IN/OUT wave reflections from wave centers (reflective voxels) will automatically create steady-state standing waves (fundamental particles) when we implement particle wave centers. For now, just inject energy with correct wave properties.

#### Energy Evolution Sequence: From Pulse to Particle

The complete energy evolution follows seven distinct phases:

##### Phase 1: Center-Concentrated Pulse Injection

- Energy concentrated at universe center
- Single pulse (or a few pulses for precision)
- Total energy exactly matches `equations.energy_wave_equation(volume)`
- Uses proper wave characteristics (f, A, λ from constants)
- Implementation: **Option 2 (Spherical Gaussian)** recommended

##### Phase 2: Outward Propagation via Wave Equation

- Wave equation ∂²ψ/∂t² = c²∇²ψ governs propagation
- Energy travels outward at wave speed c
- Spherical wave fronts expand from center
- No manual 1/r falloff needed - emerges naturally from Laplacian operator
- Energy automatically conserved by wave equation physics

##### Phase 3: Boundary Reflections

- Waves reach universe boundaries (walls)
- Boundary conditions enforce ψ = 0 at walls
- Waves reflect back into domain
- Reflections create interference patterns
- Total energy remains constant (no absorption at boundaries)

##### Phase 4: Energy Dilution into Stable Distributed State

- Multiple reflections distribute energy throughout field
- After sufficient time, energy reaches quasi-equilibrium
- Energy density becomes relatively uniform across universe
- Small fluctuations remain (natural wave motion)
- System ready for wave center insertion

##### Phase 5: Wave Center Insertion (Reflective Voxels)

- Insert reflective voxels at specific positions
- Wave centers: ψ = 0 always (never changes)
- Function like internal boundary walls
- Disturb neighboring voxels to invert wave direction
- Create local reflection sites within the field

##### Phase 6: Standing Wave Emergence (IN + OUT Interference)

- Reflected waves (OUT) interfere with incoming waves (IN)
- Constructive/destructive interference creates nodes and antinodes
- Pattern: Φ = Φ₀ e^(iωt) sin(kr)/r (Wolff's solution)
- Steady-state standing wave forms around wave center
- Wave center = particle core

##### Phase 7: Particle Formation (Mass = Trapped Energy)

- Standing wave boundary defines particle extent
- Energy trapped within standing wave pattern
- Steady-state energy density inside boundary
- **Particle mass = total energy in standing wave region**
- Fundamental particle successfully formed

##### Visual Summary

```text
Time 0:      [    •    ]  ← Concentrated pulse at center

Time 1:     [   ◯◯◯   ]  ← Expanding wave front

Time 2:    [  ◯     ◯  ]  ← Reached boundaries, reflecting

Time 3:    [ ░░░░░░░░░ ]  ← Energy distributed (diluted state)

Time 4:    [ ░ ⊕ ░ ⊕ ░ ]  ← Wave centers (⊕) inserted

Time 5:    [ ░ ◉ ░ ◉ ░ ]  ← Standing waves (◉) formed = particles!
```

##### Why This Sequence?

1. **Energy Correctness**: Start with exact EWT energy amount
2. **Natural Evolution**: Let wave equation physics handle distribution
3. **Stable Foundation**: Distributed energy provides stable background field
4. **Clean Particle Formation**: Wave centers create particles in equilibrium field
5. **Physical Realism**: Mimics natural wave behavior and particle emergence

##### Implementation Timeline

- **NOW (LEVEL-1 Phase A)**: Phases 1-4 (energy injection → stable distribution)
- **LATER (LEVEL-1 Phase B)**: Phases 5-7 (wave centers → particle formation)

##### Implementation - Option 1: Uniform Energy Density (Simplest)

```python
@ti.kernel
def charge_uniform_energy(self):
    """
    Initialize field with uniform energy density matching EWT equation.

    This is the simplest initial condition - just charge the field uniformly
    with the correct total energy from equations.energy_wave_equation().

    Wave propagation will then naturally evolve this initial state.
    Standing waves will emerge when wave centers (reflective voxels) are added.
    """
    # EWT constants
    ρ = ti.f32(constants.MEDIUM_DENSITY)
    f = ti.f32(constants.EWAVE_FREQUENCY)
    A = ti.f32(constants.EWAVE_AMPLITUDE)

    # Uniform initial displacement (all voxels same)
    # Use small random perturbation to avoid perfect symmetry
    import random
    base_displacement = A / 2.0  # Half amplitude to start

    for i, j, k in self.displacement_am:
        # Small random perturbation (±10%) to break symmetry
        perturbation = 1.0 + 0.1 * (random.random() - 0.5)
        displacement = base_displacement * perturbation

        self.displacement_am[i, j, k] = displacement / constants.ATTOMETER
        self.amplitude_am[i, j, k] = ti.abs(self.displacement_am[i, j, k])

    # Initialize old displacement (same as current for stationary start)
    for i, j, k in self.displacement_old:
        self.displacement_old[i, j, k] = self.displacement_am[i, j, k]

    # Verify total energy matches equations.energy_wave_equation()
    # E_total = ρV(fA)² where V = nx × ny × nz × dx³
```

##### Implementation - Option 2: Spherical Gaussian Wave Pulse (Recommended)

ψ(x) = e^(-x²)  (Gaussian bump)

```python
@ti.kernel
def charge_spherical_gaussian(
    self,
    center: ti.math.vec3,      # Wave center position (meters)
    total_energy: ti.f32,       # Total energy to inject (Joules)
    width_factor: ti.f32 = 3.0  # Width as multiple of wavelength
):
    """
    Initialize field with center-concentrated spherical Gaussian pulse.

    IMPLEMENTS PHASE 1 OF ENERGY EVOLUTION SEQUENCE:
    - Single smooth pulse concentrated at universe center
    - Total energy exactly matches equations.energy_wave_equation(volume)
    - Will propagate outward, reflect off boundaries, and dilute (Phases 2-4)
    - After stabilization, wave centers can be inserted (Phase 5)

    This does NOT create particle standing waves - those emerge automatically
    later when wave centers (reflective voxels) are inserted.

    Args:
        center: Pulse center position in meters (typically universe center)
        total_energy: Total energy from equations.energy_wave_equation(volume)
        width_factor: Pulse width = width_factor × wavelength (default: 3.0)
                     Smaller width = more concentrated pulse
                     Larger width = smoother, more spread out
    """
    # Convert to scaled units
    center_am = center / constants.ATTOMETER

    # EWT constants
    ρ = ti.f32(constants.MEDIUM_DENSITY)
    f = ti.f32(constants.EWAVE_FREQUENCY)
    λ_am = ti.f32(constants.EWAVE_LENGTH / constants.ATTOMETER)

    # Gaussian width
    σ_am = width_factor * λ_am  # Width in attometers

    # Calculate amplitude to match desired total energy
    # E = ∫ ρ(fA)² dV for Gaussian: E ≈ ρ(fA)² × (π^(3/2) × σ³)
    # Solve for A: A = √(E / (ρf² × π^(3/2) × σ³))
    volume_factor = (ti.math.pi ** 1.5) * (σ_am * constants.ATTOMETER) ** 3
    A_required = ti.sqrt(total_energy / (ρ * f * f * volume_factor))
    A_am = A_required / constants.ATTOMETER

    # Apply Gaussian wave packet
    for i, j, k_idx in self.displacement_am:
        pos_am = self.get_position_am(i, j, k_idx)
        r_vec = pos_am - center_am
        r_squared = r_vec.dot(r_vec)

        # Gaussian envelope: exp(-r²/(2σ²))
        gaussian = ti.exp(-r_squared / (2.0 * σ_am * σ_am))

        # Initial displacement with Gaussian envelope
        displacement = A_am * gaussian

        self.displacement_am[i, j, k_idx] = displacement
        self.amplitude_am[i, j, k_idx] = ti.abs(displacement)

    # Initialize old displacement (same as current for stationary start)
    for i, j, k_idx in self.displacement_old:
        self.displacement_old[i, j, k_idx] = self.displacement_am[i, j, k_idx]
```

##### Implementation - Option 3: Wolff's Spherical Wave (For Future Particle Implementation)

```python
@ti.kernel
def charge_wolff_spherical_wave(
    self,
    center: ti.math.vec3,
    frequency: ti.f32,
    amplitude: ti.f32,
    initial_phase: ti.f32 = 0.0
):
    """
    Initialize using Wolff's analytical solution: Φ = Φ₀ e^(iωt) sin(kr)/r

    USE THIS LATER when implementing wave centers (reflective voxels).
    This creates the sin(kr)/r pattern that will become a standing wave
    when IN and OUT waves interfere.

    For now, use simpler Gaussian (Option 2) for initial charging.
    """
    center_am = center / constants.ATTOMETER
    amplitude_am = amplitude / constants.ATTOMETER
    k = 2.0 * ti.math.pi * frequency / constants.EWAVE_SPEED

    for i, j, k_idx in self.displacement_am:
        pos_am = self.get_position_am(i, j, k_idx)
        r_vec = pos_am - center_am
        r = r_vec.norm()

        # sin(kr)/r pattern (finite at r=0: lim = k)
        if r < 0.01:
            spatial_factor = k
        else:
            kr = k * r * constants.ATTOMETER
            spatial_factor = ti.sin(kr) / (r * constants.ATTOMETER)

        wave_displacement = amplitude_am * ti.cos(initial_phase) * spatial_factor

        self.displacement_am[i, j, k_idx] = wave_displacement
        self.amplitude_am[i, j, k_idx] = ti.abs(wave_displacement)

    for i, j, k_idx in self.displacement_old:
        self.displacement_old[i, j, k_idx] = self.displacement_am[i, j, k_idx]
```

##### Usage Example (Implementing Phase 1: Center-Concentrated Pulse)

```python
from openwave.common import constants, equations
import taichi as ti

# Calculate universe volume (meters³)
universe_volume = wave_field.actual_universe_size[0] * \
                  wave_field.actual_universe_size[1] * \
                  wave_field.actual_universe_size[2]

# Get correct total energy from EWT equation (Phase 1)
total_energy = equations.energy_wave_equation(
    volume=universe_volume,
    density=constants.MEDIUM_DENSITY,
    speed=constants.EWAVE_SPEED,
    wavelength=constants.EWAVE_LENGTH,
    amplitude=constants.EWAVE_AMPLITUDE
)

# Calculate universe center position
side_length = universe_volume ** (1/3)  # Assuming cubic universe
center_position = ti.Vector([side_length / 2.0] * 3)  # meters

# PHASE 1: Inject center-concentrated pulse with exact EWT energy
# This is a single pulse (or few pulses) that will propagate outward
wave_field.charge_spherical_gaussian(
    center=center_position,           # Universe center
    total_energy=total_energy,        # Exact EWT energy amount
    width_factor=3.0                  # Pulse width = 3× wavelength
)

# Verify energy matches EWT equation
measured_energy = wave_field.compute_total_energy()
energy_match_percent = abs(measured_energy - total_energy) / total_energy * 100

print(f"=== Initial Energy Charging (Phase 1) ===")
print(f"Universe volume: {universe_volume:.2e} m³")
print(f"Target energy (EWT): {total_energy:.2e} J")
print(f"Measured energy: {measured_energy:.2e} J")
print(f"Energy match: {energy_match_percent:.2f}%")
print(f"\nPulse centered at: {center_position} m")
print(f"Pulse width: {3.0 * constants.EWAVE_LENGTH:.2e} m")

# PHASES 2-4 will happen automatically during simulation:
# - Wave propagates outward (Phase 2)
# - Reflects off boundaries (Phase 3)
# - Dilutes into stable state (Phase 4)

# Run simulation to allow energy distribution
# After energy stabilizes, we'll implement Phase 5 (wave center insertion)
```

#### Advanced Technique: Multiple Pulses for Precision

For more precise energy control, you can inject a few successive pulses:

```python
# Option A: Single large pulse (simple, recommended)
wave_field.charge_spherical_gaussian(
    center=center_position,
    total_energy=total_energy,
    width_factor=3.0
)

# Option B: Multiple smaller pulses (higher precision)
# Useful if single pulse causes numerical instability
num_pulses = 3
energy_per_pulse = total_energy / num_pulses

for pulse_idx in range(num_pulses):
    # Add each pulse with small time delay
    wave_field.charge_spherical_gaussian(
        center=center_position,
        total_energy=energy_per_pulse,
        width_factor=3.0
    )
    # Run a few timesteps between pulses to let energy spread
    for _ in range(10):
        wave_field.propagate_wave(dt)
```

Multiple pulses can provide:

- Better numerical stability (smaller amplitude changes per timestep)
- More gradual energy injection
- Finer control over energy distribution

However, single pulse is usually sufficient and simpler.

##### Recommendation

- **Now**: Use **Option 2 (Spherical Gaussian)** - simple, smooth, energy-conserving
- **Single vs Multiple Pulses**: Start with single pulse; use multiple only if needed for stability
- **Later**: Use **Option 3 (Wolff's sin(kr)/r)** when implementing wave centers and particle formation
- **Option 1**: Only for testing wave equation stability

#### Energy and Momentum Conservation

**Energy Density at Each Voxel**:

```python
# At each voxel [i,j,k]
E_kinetic = ½ × ρ × (∂ψ/∂t)²  # Oscillation kinetic energy (motion)
E_potential = ½ × ρc² × ψ²/λ²  # Displacement potential energy (compression)
E_total[i,j,k] = E_kinetic + E_potential
```

**Total Energy** (must be conserved):

```python
E_system = Σ(i,j,k) E_total[i,j,k] × dx³ = constant
```

**Verification**:

```python
@ti.kernel
def compute_total_energy() -> ti.f32:
    """Verify energy conservation in wave field."""
    total_energy = 0.0

    for i, j, k in self.displacement_am:
        # Velocity (time derivative of amplitude)
        v = (self.displacement_am[i,j,k] - self.amplitude_old[i,j,k]) / dt

        # Kinetic energy density
        E_k = 0.5 * ρ * v**2

        # Potential energy density
        E_p = 0.5 * ρ * c**2 * (self.displacement_am[i,j,k] / λ)**2

        # Add to total
        total_energy += (E_k + E_p) * dx**3

    return total_energy
```

**Momentum Density**:

```python
# Momentum carried by wave (vector field)
p[i,j,k] = ρ × ψ[i,j,k] × wave_direction[i,j,k]
```

Where `wave_direction` is determined by the gradient of phase (see Answer 4).

#### Wave Mode: Longitudinal vs Transverse

**Fundamental Concept**:

Wave mode is determined by the relationship between **medium displacement direction** and **wave propagation direction**:

- **Longitudinal wave**: Displacement parallel to propagation (compression wave)
- **Transverse wave**: Displacement perpendicular to propagation (shear wave)
- **Mixed mode**: Both components present

**Mathematical Definition**:

```text
Wave propagation direction: k̂ = S / |S|  (from energy flux, see Answer 4)
Medium displacement direction: û = ∇ψ / |∇ψ|  (from displacement gradient)

Dot product: cos(θ) = k̂ · û
- cos(θ) ≈ ±1: Longitudinal (parallel/antiparallel)
- cos(θ) ≈ 0:  Transverse (perpendicular)
- 0 < |cos(θ)| < 1: Mixed mode
```

**Implementation**:

```python
@ti.kernel
def compute_wave_mode(self):
    """
    Compute wave mode at each voxel.

    Returns:
    - wave_mode[i,j,k] = 1.0:  Pure longitudinal
    - wave_mode[i,j,k] = 0.0:  Pure transverse
    - wave_mode[i,j,k] ∈ (0,1): Mixed mode
    """
    c = ti.f32(constants.EWAVE_SPEED)

    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # 1. Compute wave propagation direction (energy flux)
            psi = self.displacement_am[i,j,k]

            grad_x = (self.displacement_am[i+1,j,k] - self.displacement_am[i-1,j,k]) / (2.0 * self.dx_am)
            grad_y = (self.displacement_am[i,j+1,k] - self.displacement_am[i,j-1,k]) / (2.0 * self.dx_am)
            grad_z = (self.displacement_am[i,j,k+1] - self.displacement_am[i,j,k-1]) / (2.0 * self.dx_am)

            grad_psi = ti.Vector([grad_x, grad_y, grad_z])

            # Energy flux (wave propagation direction)
            S = -c**2 * psi * grad_psi
            S_mag = S.norm()

            if S_mag > 1e-12:
                k_hat = S / S_mag  # Wave propagation direction (unit vector)

                # 2. Compute medium displacement direction
                grad_mag = grad_psi.norm()

                if grad_mag > 1e-12:
                    u_hat = grad_psi / grad_mag  # Displacement direction (unit vector)

                    # 3. Compute alignment (dot product)
                    cos_theta = ti.abs(k_hat.dot(u_hat))  # |cos(θ)|, range [0,1]

                    # Store wave mode
                    # cos_theta = 1.0 → longitudinal
                    # cos_theta = 0.0 → transverse
                    self.wave_mode[i,j,k] = cos_theta
                else:
                    self.wave_mode[i,j,k] = 0.0  # No displacement
            else:
                self.wave_mode[i,j,k] = 0.0  # No propagation
```

**Physical Interpretation**:

```python
# Example values:
wave_mode = 0.98  # Mostly longitudinal (compression wave)
wave_mode = 0.05  # Mostly transverse (shear wave)
wave_mode = 0.50  # Mixed mode (45° angle)

# In EWT context:
# - Gravitational waves: Expected to be longitudinal (compression)
# - EM waves from electron: May have transverse components
# - Near particle centers: Complex mix of modes
```

**Storage**:

```python
# In WaveField class __init__
self.wave_mode = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Range [0,1]
```

#### Wave Decomposition: Separating Longitudinal and Transverse Components

**Key Insight**: A single voxel can carry **both** longitudinal and transverse wave components simultaneously!

**Physical Reality**:

In a complex wave field (like near a particle or in interference regions), the displacement at a voxel is typically **not** purely parallel or perpendicular to the propagation direction. Instead, the displacement vector can be decomposed into:

1. **Longitudinal component**: Part parallel to wave propagation
2. **Transverse component**: Part perpendicular to wave propagation

**Vector Decomposition**:

```text
Given:
- k̂: Wave propagation direction (unit vector)
- u: Medium displacement vector (from ∇ψ)

Decompose u into parallel and perpendicular parts:

u_longitudinal = (u · k̂) k̂           (projection onto k̂)
u_transverse = u - u_longitudinal    (rejection from k̂)

Magnitudes:
|u_longitudinal| = |u · k̂|
|u_transverse| = |u| sin(θ)  where θ is angle between u and k̂

Check (should reconstruct original):
u = u_longitudinal + u_transverse ✓
```

**Implementation**:

```python
@ti.kernel
def compute_wave_components(self):
    """
    Decompose wave into longitudinal and transverse components.

    Stores:
    - longitudinal_amplitude[i,j,k]: Magnitude of longitudinal component
    - transverse_amplitude[i,j,k]: Magnitude of transverse component
    - longitudinal_fraction[i,j,k]: Fraction of energy in longitudinal mode
    """
    c = ti.f32(constants.EWAVE_SPEED)

    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # 1. Compute wave propagation direction (energy flux)
            psi = self.displacement_am[i,j,k]

            grad_x = (self.displacement_am[i+1,j,k] - self.displacement_am[i-1,j,k]) / (2.0 * self.dx_am)
            grad_y = (self.displacement_am[i,j+1,k] - self.displacement_am[i,j-1,k]) / (2.0 * self.dx_am)
            grad_z = (self.displacement_am[i,j,k+1] - self.displacement_am[i,j,k-1]) / (2.0 * self.dx_am)

            grad_psi = ti.Vector([grad_x, grad_y, grad_z])

            # Energy flux (wave propagation direction)
            S = -c**2 * psi * grad_psi
            S_mag = S.norm()

            if S_mag > 1e-12:
                k_hat = S / S_mag  # Wave propagation direction (unit vector)

                # 2. Displacement vector (from gradient)
                u = grad_psi  # Displacement direction

                # 3. Decompose into longitudinal and transverse
                # Longitudinal component (parallel to k̂)
                u_parallel_magnitude = u.dot(k_hat)  # Can be positive or negative
                u_longitudinal = u_parallel_magnitude * k_hat

                # Transverse component (perpendicular to k̂)
                u_transverse = u - u_longitudinal

                # 4. Store magnitudes
                self.longitudinal_amplitude[i,j,k] = ti.abs(u_parallel_magnitude)
                self.transverse_amplitude[i,j,k] = u_transverse.norm()

                # 5. Compute energy fractions
                # Energy ∝ amplitude²
                E_long = u_parallel_magnitude**2
                E_trans = u_transverse.norm()**2
                E_total = E_long + E_trans

                if E_total > 1e-20:
                    self.longitudinal_fraction[i,j,k] = E_long / E_total
                    self.transverse_fraction[i,j,k] = E_trans / E_total
                else:
                    self.longitudinal_fraction[i,j,k] = 0.0
                    self.transverse_fraction[i,j,k] = 0.0

                # 6. Store decomposed vectors (optional, for visualization)
                self.u_longitudinal[i,j,k] = u_longitudinal
                self.u_transverse[i,j,k] = u_transverse
            else:
                # No propagation
                self.longitudinal_amplitude[i,j,k] = 0.0
                self.transverse_amplitude[i,j,k] = 0.0
                self.longitudinal_fraction[i,j,k] = 0.0
                self.transverse_fraction[i,j,k] = 0.0
```

**Physical Example**:

```python
# Example: Wave at a voxel near particle
k_hat = [1.0, 0.0, 0.0]           # Propagating in +x direction
u = [0.8, 0.6, 0.0]               # Displacement vector (not aligned!)

# Decompose:
u_long = (0.8)(1.0, 0.0, 0.0) = [0.8, 0.0, 0.0]  # Longitudinal part
u_trans = [0.8, 0.6, 0.0] - [0.8, 0.0, 0.0] = [0.0, 0.6, 0.0]  # Transverse part

# Magnitudes:
|u_long| = 0.8
|u_trans| = 0.6

# Energy fractions:
E_long = 0.8² = 0.64
E_trans = 0.6² = 0.36
E_total = 1.00

longitudinal_fraction = 0.64 (64% longitudinal)
transverse_fraction = 0.36   (36% transverse)

# This voxel carries BOTH modes!
# The wave_mode[i,j,k] value would be:
cos(θ) = 0.8 / 1.0 = 0.8  (mixed mode, mostly longitudinal)
```

**Visualization**:

```python
# For a voxel, you can visualize the decomposition:
def visualize_wave_components(i, j, k):
    """Show how displacement splits into L and T components."""
    k_hat = get_propagation_direction(i, j, k)
    u_long = longitudinal_component(i, j, k)
    u_trans = transverse_component(i, j, k)

    # Draw arrows:
    # - Black arrow: Total displacement (u)
    # - Red arrow: Longitudinal component (u_long) along k_hat
    # - Blue arrow: Transverse component (u_trans) perpendicular to k_hat
```

**When Does This Happen?**

1. **Wave interference**: When multiple waves cross
   - Each wave may have different propagation direction
   - Superposition creates mixed displacement

2. **Near particle boundaries**:
   - Incident wave (traveling)
   - Reflected wave (traveling opposite direction)
   - Interference creates complex displacement patterns

3. **Spherical waves**:
   - Wave propagates radially outward (k̂ = r̂)
   - But field oscillations can have tangential components
   - Creates both radial (long.) and tangential (trans.) parts

4. **EM wave generation** (in EWT):
   - Electron transforms energy waves
   - May create transverse component from originally longitudinal field
   - Transition region has mixed character

**Energy Partition**:

The total wave energy at a voxel splits between modes:

```text
E_total = E_longitudinal + E_transverse

where:
E_long ∝ |u_long|² (energy in compression/rarefaction)
E_trans ∝ |u_trans|² (energy in shear/transverse oscillation)
```

**Storage Requirements**:

```python
# In WaveField class __init__
# Scalar fields (magnitudes)
self.longitudinal_amplitude = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
self.transverse_amplitude = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
self.longitudinal_fraction = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # [0,1]
self.transverse_fraction = ti.field(dtype=ti.f32, shape=(nx, ny, nz))    # [0,1]

# Vector fields (optional, for detailed visualization)
self.u_longitudinal = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
self.u_transverse = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
```

**Key Takeaway**:

- **`wave_mode[i,j,k]`** gives you a **single number** (0-1) summarizing the dominant mode
  - 1.0 = purely longitudinal
  - 0.0 = purely transverse
  - 0.5 = equal mix (45° angle)

- **Component decomposition** gives you **detailed breakdown**:
  - `longitudinal_fraction[i,j,k]` = energy in longitudinal mode
  - `transverse_fraction[i,j,k]` = energy in transverse mode
  - These two always sum to 1.0

Both approaches are valid! Use `wave_mode` for quick classification, use decomposition for detailed analysis.

#### Wave Type: Standing vs Traveling

**Fundamental Concept**:

Wave type is determined by whether the wave **moves through space**:

- **Traveling wave**: Energy moves, nodes move (wave velocity ≠ 0)
- **Standing wave**: Energy stationary, nodes fixed (wave velocity = 0)
- **Quasi-standing**: Slow-moving pattern (wave velocity ≈ 0)

**Mathematical Definition**:

```text
Wave velocity (phase velocity): v_phase = ∂x/∂t (position of constant phase)

For amplitude-based detection:
Temporal derivative: ∂ψ/∂t (how fast amplitude changes at fixed point)
Spatial derivative: ∇ψ (how much amplitude varies in space)

Standing wave criterion:
- |∂ψ/∂t| is large (rapid oscillation in time)
- But spatial pattern is stationary (nodes don't move)
- Ratio: |∂ψ/∂t| / (c|∇ψ|) ≈ 0 for traveling, ≠ 0 for standing
```

**Better Method - Energy Flux Analysis**:

```text
Traveling wave: Net energy flux (S ≠ 0 consistently)
Standing wave: No net energy flux (S ≈ 0 on average, oscillates locally)

Time-averaged energy flux:
<S> = (1/T) ∫ S dt over period T

- <S> ≈ 0: Standing wave (no net energy transport)
- <S> > threshold: Traveling wave (energy transport)
```

**Implementation**:

```python
@ti.kernel
def compute_wave_type(self):
    """
    Compute wave type at each voxel using energy flux analysis.

    Returns:
    - wave_type[i,j,k] = 0.0: Standing wave (no net energy transport)
    - wave_type[i,j,k] = 1.0: Traveling wave (energy moving)
    - wave_type[i,j,k] ∈ (0,1): Quasi-standing (slow energy transport)

    Method: Measure ratio of kinetic to potential energy
    - Pure standing wave: E_k and E_p oscillate 90° out of phase, <E_k> = <E_p>
    - Pure traveling wave: E_k = E_p at all times (in phase)
    """
    c = ti.f32(constants.EWAVE_SPEED)
    ρ = ti.f32(constants.MEDIUM_DENSITY)
    λ_m = self.wavelength_am * constants.ATTOMETER

    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Current displacement
            psi = self.displacement_am[i,j,k] * constants.ATTOMETER  # meters

            # Velocity (time derivative approximation)
            v_wave = (self.displacement_am[i,j,k] - self.amplitude_old[i,j,k]) / dt
            v_wave_m = v_wave * constants.ATTOMETER  # m/s

            # Kinetic energy density
            E_k = 0.5 * ρ * v_wave_m**2

            # Potential energy density
            E_p = 0.5 * ρ * c**2 * (psi / λ_m)**2

            # Total energy
            E_total = E_k + E_p

            if E_total > 1e-20:  # Avoid division by zero
                # Energy ratio
                # Standing wave: E_k and E_p alternate (ratio varies)
                # Traveling wave: E_k = E_p (ratio = 0.5)
                ratio = E_k / E_total

                # Measure deviation from traveling wave (ratio = 0.5)
                deviation = ti.abs(ratio - 0.5) * 2.0  # Range [0,1]

                # wave_type = 0: Standing (deviation = 1, E_k or E_p dominates)
                # wave_type = 1: Traveling (deviation = 0, E_k = E_p)
                self.wave_type[i,j,k] = 1.0 - deviation
            else:
                self.wave_type[i,j,k] = 0.0  # No wave energy
```

**Alternative Method - Node Motion Detection**:

```python
@ti.kernel
def compute_wave_type_node_motion(self, dt: ti.f32):
    """
    Alternative: Detect wave type by tracking node positions over time.

    Standing wave: Nodes (ψ=0) remain at fixed spatial locations
    Traveling wave: Nodes move with wave velocity
    """
    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Check if current voxel is near a node (zero crossing)
            psi_now = self.displacement_am[i,j,k]
            psi_old = self.amplitude_old[i,j,k]

            # Zero crossing detection
            if psi_now * psi_old < 0:  # Sign change (crossed zero)
                # This is a node position
                # Check if node moved in space

                # For standing wave: same spatial location has zero repeatedly
                # For traveling wave: zero location moves

                # Store node position and compare with previous timesteps
                # (Requires additional node tracking data structure)
                pass
```

**Physical Interpretation**:

```python
# Example values:
wave_type = 0.95  # Traveling wave (energy moving through space)
wave_type = 0.05  # Standing wave (energy oscillating in place)
wave_type = 0.50  # Quasi-standing (slow-moving pattern)

# In EWT context:
# - Far from particles: Traveling waves (expanding spherical fronts)
# - Near particle centers: Standing waves (interference patterns)
# - Particle boundaries: Mixed (partial standing, partial traveling)
```

**Storage**:

```python
# In WaveField class __init__
self.wave_type = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Range [0,1]
```

**Combined Analysis**:

```python
@ti.kernel
def analyze_wave_characteristics(self):
    """
    Complete wave analysis: mode + type.

    Produces 4 wave categories:
    1. Longitudinal traveling wave (gravity wave propagating)
    2. Longitudinal standing wave (mass formation, particle interior)
    3. Transverse traveling wave (EM radiation)
    4. Transverse standing wave (electron orbital?)
    """
    self.compute_wave_mode()    # Longitudinal vs Transverse
    self.compute_wave_type()    # Standing vs Traveling

    # Can combine for visualization or analysis
    for i, j, k in self.wave_mode:
        mode = self.wave_mode[i,j,k]    # 0=transverse, 1=longitudinal
        wtype = self.wave_type[i,j,k]   # 0=standing, 1=traveling

        # Classification:
        if mode > 0.7 and wtype > 0.7:
            # Longitudinal traveling wave (gravitational radiation)
            self.wave_class[i,j,k] = 1
        elif mode > 0.7 and wtype < 0.3:
            # Longitudinal standing wave (particle mass)
            self.wave_class[i,j,k] = 2
        elif mode < 0.3 and wtype > 0.7:
            # Transverse traveling wave (EM radiation)
            self.wave_class[i,j,k] = 3
        elif mode < 0.3 and wtype < 0.3:
            # Transverse standing wave (electron shell?)
            self.wave_class[i,j,k] = 4
        else:
            # Mixed/transitional
            self.wave_class[i,j,k] = 0
```

**Summary Table**:

| Property | Longitudinal | Transverse | Standing | Traveling |
|----------|--------------|------------|----------|-----------|
| **Definition** | Displacement ∥ propagation | Displacement ⊥ propagation | Nodes fixed | Nodes moving |
| **Measurement** | k̂ · û ≈ 1 | k̂ · û ≈ 0 | E_k ≠ E_p (alternating) | E_k = E_p (in phase) |
| **Energy flux** | Along k̂ direction | Perpendicular components | <S> = 0 | <S> ≠ 0 |
| **EWT example** | Gravity waves | EM waves (from electron) | Particle interior | Propagating radiation |
| **Field storage** | `wave_mode[i,j,k]` | `wave_mode[i,j,k]` | `wave_type[i,j,k]` | `wave_type[i,j,k]` |

#### Wavelength and Frequency Variation in the Medium

**Fundamental Relationship**:

```text
c = λ × f

where:
c = wave speed (constant in uniform medium)
λ = wavelength (can vary locally)
f = frequency (can vary locally)
```

**Key Insight for EWT**:

In Energy Wave Theory, the medium (spacetime fabric) has **constant properties everywhere**:

- **Wave speed c**: Always 2.998×10⁸ m/s (speed of light)
- **Medium density ρ**: Always 3.860×10²² kg/m³

However, **wavelength λ and frequency f can vary** due to:

1. **Different energy sources** (particles with different energies)
2. **Wave interactions** (constructive/destructive interference)
3. **Doppler effects** (moving sources)
4. **Energy transformations** (electron converting to EM waves)

**Dispersion in EWT Medium**:

```text
Question: Is the medium dispersive or non-dispersive?

Non-dispersive medium: c is constant for all frequencies
- c = λf holds everywhere
- Different frequencies travel at same speed
- No frequency-dependent effects

Answer for EWT: Non-dispersive!
- All energy waves travel at c regardless of frequency
- λf = c always holds
- This is like electromagnetic waves in vacuum
```

**Measuring Local Frequency - Direct Temporal Method**:

**Frequency is the Primary Measured Property** - Wavelength is derived from it.

The most direct way to measure frequency is to **time the oscillations directly**:

**Method**:

1. **Measure dt** between peaks (when |ψ| reaches A)
2. **Compute f = 1/dt** (frequency in Hz) - this is the core measurement
3. **Derive T = dt** (period, same value different name)
4. **Derive λ = c/f** (wavelength, computed from frequency)

**Why Frequency First?**

- **Direct measurement**: dt → f = 1/dt (immediate, no conversion)
- **Frequency-centric**: Aligns with human intuition (radio, audio, WiFi all use f)
- **Planck relation**: E = hf (energy proportional to frequency, not wavelength!)
- **Spacetime coupling**: A (spatial) × f (temporal) = natural pairing
- **Wavelength is derived**: λ = c/f (computed when needed for spatial design)

**Advantages**:

- **Simple implementation**: Reuses existing amplitude tracking (`amplitude_am`)
- **Physical intuition**: Each voxel times its own oscillations
- **Works per voxel**: Independent measurement at each location
- **Handles superposition**: Measures dominant/beating frequency automatically
- **Natural for multi-frequency**: Frequency domain is the natural representation

**Implementation**:

```python
# In WaveField class __init__, add:
self.last_peak_time_rs = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Time of last peak (rontoseconds)
self.frequency_local = ti.field(dtype=ti.f32, shape=(nx, ny, nz))    # f (Hz) - PRIMARY measured property
self.period_rs = ti.field(dtype=ti.f32, shape=(nx, ny, nz))          # T (rontoseconds) - derived from f
self.wavelength_local = ti.field(dtype=ti.f32, shape=(nx, ny, nz))   # λ (attometers) - derived from f

@ti.kernel
def measure_frequency(self, current_time_rs: ti.f32):
    """
    Measure frequency by timing peaks (when |ψ| = A).

    Measurement hierarchy:
    1. Measure dt between peaks (timing measurement)
    2. Compute f = 1/dt (PRIMARY property)
    3. Derive T = dt (period, same as measured dt)
    4. Derive λ = c/f (wavelength from frequency)

    Note: Frequency-centric approach - f is measured, λ is computed.
    """
    c = ti.f32(constants.EWAVE_SPEED)  # m/s

    for i, j, k in self.displacement_am:
        # Check if displacement is at a peak (|ψ| ≈ A)
        disp_mag = ti.abs(self.displacement_am[i,j,k])
        amp = self.amplitude_am[i,j,k]

        # Peak detection: current displacement within 1% of amplitude
        if amp > 1e-12 and disp_mag >= amp * 0.99:
            # This is a peak!

            if self.last_peak_time_rs[i,j,k] > 0:  # Not the first peak
                # Measure time between peaks (dt)
                dt_rs = current_time_rs - self.last_peak_time_rs[i,j,k]

                if dt_rs > 0:
                    # Convert dt to seconds
                    dt_seconds = dt_rs * constants.RONTOSECOND

                    # PRIMARY: Compute frequency f = 1/dt
                    self.frequency_local[i,j,k] = 1.0 / dt_seconds  # Hz

                    # DERIVED: Store period (same as dt, different name)
                    self.period_rs[i,j,k] = dt_rs  # rontoseconds

                    # DERIVED: Compute wavelength λ = c/f
                    f_Hz = self.frequency_local[i,j,k]
                    wavelength_m = c / f_Hz  # meters
                    self.wavelength_local[i,j,k] = wavelength_m / constants.ATTOMETER  # attometers

            # Update last peak time for next measurement
            self.last_peak_time_rs[i,j,k] = current_time_rs

# Usage in main simulation loop:
def update_timestep(self, dt_rs: ti.f32):
    """Complete wave field update for one timestep."""
    self.current_time_rs += dt_rs

    # 1. Propagate wave displacement
    self.propagate_wave_field(dt_rs)

    # 2. Track amplitude envelope
    self.track_amplitude_envelope()

    # 3. Measure frequency from peaks (f = 1/dt, then derive T and λ)
    self.measure_frequency(self.current_time_rs)

    # 4. Compute wave direction
    self.compute_wave_direction()

    # 5. Compute force field
    self.compute_force_field_newtons()
```

**Physical Notes**:

- **First peak**: Just records time, no period computed yet
- **Second peak**: Computes first period measurement (T = t2 - t1)
- **Convergence**: Takes ~2 oscillation periods to get stable measurements
- **Standing waves**: Correctly measures temporal oscillation frequency
- **Traveling waves**: Measures frequency of passing wave crests
- **Superposition**: Measures dominant/beating frequency pattern

**Connection to LEVEL-0**:

This approach mirrors `wave_engine_level0.py`'s amplitude tracking:

```python
# LEVEL-0 tracks amplitude per granule:
amplitude_am[granule_idx] = max(|ψ|)

# LEVEL-1 extends this to track period:
period_rs[i,j,k] = time between peaks when |ψ| = amplitude_am[i,j,k]
```

**Why This Works Better Than Spatial Methods**:

```text
Spatial gradient method (old):
- k ≈ |∇A|/A, then λ = 2π/k
- Requires stable spatial pattern
- Sensitive to noise in gradients
- Complex for superposition

Temporal peak timing (new - frequency-centric):
- dt = time between peaks
- f = 1/dt (PRIMARY measurement)
- T = dt (period, same value)
- λ = c/f (derived from frequency)
- Direct measurement of oscillation
- Robust to spatial noise
- Natural handling of beating/interference
```

**Multi-Frequency Superposition Challenge**:

When multiple waves with different frequencies overlap at a voxel, the displacement becomes:

```text
ψ_total(t) = Σ A_i sin(k_i·r - ω_i·t)  where ω_i = 2πf_i

Problem: What is "the" frequency at this voxel?
Answer: There isn't one unique frequency!
```

**Implications for Force Calculation (Frequency-Based)**:

The full force derivation with variable f includes a frequency gradient term:

```text
F = -∇E = -∇[ρV(fA)²]

Full expansion with product rule:
F = -2ρVfA × [f∇A + A∇f]

Two terms:
1. Amplitude gradient: -2ρVfA × f∇A = -2ρVf² × A∇A    (primary - particles move toward lower A)
2. Frequency gradient: -2ρVfA × A∇f = -2ρVA²f × ∇f    (secondary - particles move toward higher f)
```

**Solutions (Implementation Strategies)**:

1. **Monochromatic Approximation** (initial implementation):
   - Single wave source → uniform f
   - ∇f ≈ 0 → frequency gradient term negligible
   - Use simplified: `F = -2ρVf² × A∇A`
   - **Recommended for first version**
   - Pros: Simple, fast, good for initial development
   - Cons: Doesn't handle multi-frequency scenarios

2. **Dominant Frequency** (intermediate):
   - Measure frequency from temporal peaks (documented above)
   - Gives dominant/beating frequency: `f_local = 1/dt_measured`
   - Include both gradient terms:

     ```python
     # Compute amplitude gradient force (primary term)
     # F_amplitude = -2ρVf² × A × ∇A
     grad_A = compute_gradient(amplitude_am)
     force_scale = 2.0 * ρ * V * frequency_local**2
     F_amplitude = -force_scale * A * grad_A

     # Compute frequency gradient correction (secondary term)
     # F_frequency = -2ρVA²f × ∇f
     grad_f = compute_gradient(frequency_local)
     correction_scale = 2.0 * ρ * V * A**2 * frequency_local
     F_frequency = -correction_scale * grad_f

     # Total force includes both terms
     F_total = F_amplitude + F_frequency
     ```

   - **Good balance of accuracy vs complexity**
   - Pros: Captures beating/Doppler, computationally tractable
   - Cons: Single dominant mode may miss details

3. **Fourier Decomposition** (advanced, future):
   - Track multiple frequency modes per voxel
   - Store: `amplitude_modes[i,j,k,mode]`, `frequency_modes[i,j,k,mode]`
   - Compute force contribution from each mode separately
   - Requires FFT or temporal spectral analysis
   - **Memory intensive**: num_modes × field size
   - **Computationally expensive**: FFT per voxel per timestep
   - Only needed for complex multi-source scenarios
   - Pros: Physically complete, handles arbitrary superposition
   - Cons: Memory × num_modes, computationally expensive

**Recommendation for Initial Implementation**:

Start with **monochromatic approximation** (uniform f, single source). Once wave propagation is working, add **dominant frequency tracking** with the temporal peak method documented above. This captures the most important physics (beating frequencies, Doppler shifts) without the complexity of full Fourier decomposition.

**Wavelength Propagation and Changes**:

**Q: How does wavelength change propagate through the field?**

**A: Wavelength doesn't "propagate" - it's a property of the local wave pattern!**

Here's the key distinction:

1. **Wave amplitude** propagates (governed by wave equation)
2. **Wavelength** is **measured** from the spatial pattern of amplitude

Think of it like this:

```text
Analogy: Water waves

- Wave height (amplitude): Propagates through water
- Distance between crests (wavelength): Measured from pattern
- If you drop a small stone: short wavelength
- If you drop a large stone: long wavelength
- Different wavelengths can coexist in the same water

In EWT:
- Displacement ψ: Propagates via ∂²ψ/∂t² = c²∇²ψ
- Wavelength λ: Measured from spatial pattern
- Different particles create different wavelengths
- λ varies spatially based on energy source
```

**Frequency Changes and Energy**:

In EWT, **frequency is related to energy**:

```text
E = hf  (Planck relation)

where:
E = photon/quantum energy
h = Planck's constant
f = frequency

Higher frequency → Higher energy
Shorter wavelength → Higher energy (since c = λf is constant)
```

**How Frequency Changes Occur**:

1. **Different sources**:

   ```python
   # Neutrino creates waves at frequency f1
   λ1 = c / f1

   # Electron creates waves at frequency f2
   λ2 = c / f2

   # Both propagate through same medium at speed c
   # But with different wavelengths
   ```

2. **Energy transformations** (e.g., electron converting energy wave → EM wave):

   ```python
   # Incoming energy wave: f_in, λ_in
   # Electron oscillates at f_electron
   # Outgoing EM wave: f_out = f_electron, λ_out = c/f_electron

   # Frequency changed by transformation!
   ```

3. **Doppler shift** (moving source):

   ```text
   For source moving with velocity v:

   f_observed = f_source × (c / (c ± v))

   Approaching: f increases (blueshift), λ decreases
   Receding: f decreases (redshift), λ increases
   ```

**Implementing Frequency/Wavelength Tracking**:

```python
# In WaveField class __init__
self.frequency_local = ti.field(dtype=ti.f32, shape=(nx, ny, nz))   # Measured frequency
self.wavelength_local = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Computed wavelength

# Default wavelength (from initial conditions)
self.wavelength_am = wavelength_m / constants.ATTOMETER

# Update cycle:
def update_wave_properties(self):
    """Update derived wave properties."""
    self.compute_local_frequency()    # Measure from pattern
    self.compute_local_wavelength()   # Compute λ = c/f
```

**Superposition of Different Wavelengths**:

**Key Point**: Multiple waves with **different wavelengths** can exist simultaneously in the same region!

```text
Superposition example:

Wave 1: ψ₁ = A₁ sin(k₁·r - ω₁t)  with λ₁ = 2π/k₁
Wave 2: ψ₂ = A₂ sin(k₂·r - ω₂t)  with λ₂ = 2π/k₂

Total: ψ = ψ₁ + ψ₂

Question: What is "the" wavelength at a voxel?

Answer: There isn't a single wavelength!
- Must decompose into frequency components (Fourier analysis)
- Or define "dominant wavelength" from spatial pattern
- Or track each wave component separately
```

**Fourier Decomposition** (advanced):

For complex wave patterns with multiple wavelengths:

```python
@ti.kernel
def measure_dominant_wavelength(self):
    """
    Measure dominant wavelength using local Fourier analysis.

    Samples amplitude along wave propagation direction,
    performs FFT, finds peak frequency component.
    """
    # For each voxel:
    # 1. Sample amplitude along propagation direction
    # 2. FFT to get frequency spectrum
    # 3. Find dominant frequency
    # 4. Convert to wavelength: λ = c/f_dominant

    # (Requires FFT implementation, beyond basic scope)
    pass
```

**Practical Approach for LEVEL-1**:

For initial implementation, **assume single wavelength** (monochromatic):

```python
# Initialize with single wavelength
wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETER

# All voxels start with this wavelength
# Wave equation preserves wavelength for monochromatic source
```

**Future Enhancement** for multiple wavelengths:

1. **Track wave packets** with different λ separately
2. Use **frequency tagging** to follow each component
3. Implement **Fourier analysis** for decomposition

**Summary Table (Frequency-Centric)**:

| Property | Constant? | How Determined? | Propagates? |
|----------|-----------|-----------------|-------------|
| **ρ** (medium density) | ✓ Yes (uniform medium) | From EWT constants | N/A (property of medium) |
| **c** (wave speed) | ✓ Yes (constant everywhere) | From medium properties | N/A (property of medium) |
| **ψ** (displacement) | ✗ No (varies spatially/temporally) | Wave equation evolution | ✓ Yes (via PDE) |
| **A** (amplitude) | ✗ No (varies spatially/temporally) | **Tracked max**(\|ψ\|) | ✓ Yes (via PDE) |
| **f** (frequency) | ✗ No (varies spatially) | **PRIMARY: Measured** f = 1/dt | ✗ No (derived property) |
| **T** (period) | ✗ No (varies spatially) | **Derived** T = dt = 1/f | ✗ No (derived property) |
| **λ** (wavelength) | ✗ No (varies spatially) | **Derived** λ = c/f | ✗ No (derived property) |
| **E** (energy) | ✗ No (varies spatially) | E = ρV(fA)² | ✓ Yes (via wave energy) |

**Key Takeaways (Frequency-Centric Philosophy)**:

1. **c is constant** throughout the EWT medium (non-dispersive)
2. **dt is measured** directly by timing oscillations (when \|ψ\| reaches A)
3. **f = 1/dt is PRIMARY** - frequency computed first (human-intuitive, Planck E=hf)
4. **T = dt and λ = c/f are DERIVED** - period and wavelength computed from frequency
5. **Simple and robust**: Direct temporal measurement, not sensitive to spatial gradients
6. **Reuses amplitude tracking**: Same infrastructure as LEVEL-0's `amplitude_am` approach
7. For **single source**, f is constant; for **multiple sources**, measures dominant frequency
8. **Spacetime coupling**: f (temporal) × A (spatial) = natural pairing in E = ρV(fA)²

---

### Answer 3: Choosing Between PDE and Huygens

#### Should We Choose One or Use Both?

**Recommendation**: Use **PDE (Wave Equation)** as the primary method.

**Reason**: The wave equation naturally implements Huygens' principle through the Laplacian operator. Each voxel effectively becomes a source of secondary wavelets to its neighbors.

#### Comparison Table

| Aspect | Wave Equation (PDE) | Huygens Wavelets (Explicit) |
|--------|---------------------|----------------------------|
| **Physics** | Fundamental equation | Derived principle |
| **Accuracy** | Second-order accurate (with proper scheme) | Depends on implementation |
| **Efficiency** | ✓✓ Highly optimized | ✗ Computationally expensive |
| **Memory** | 3 fields (old, current, new) | 2 fields minimum |
| **Connectivity** | Natural 6/18/26 neighbors | All neighbors (26) typically |
| **Stability** | CFL condition required | Less restrictive |
| **Energy Conservation** | Excellent (with symplectic integrator) | Requires careful normalization |
| **Implementation** | Straightforward | Complex neighbor loops |
| **GPU Performance** | ✓✓ Excellent (simple stencil) | ✗ Many atomic operations |
| **Anisotropy** | Uniform in all directions | Can handle directional weighting |

#### Performance Analysis

**Wave Equation (PDE)**:

```text
Operations per voxel per timestep:
- 6 neighbor reads (6-connectivity)
- 1 Laplacian computation (7 operations)
- 1 update computation (3 operations)
Total: ~10 operations

For 100³ = 1M voxels:
~10M operations per timestep
```

**Huygens Wavelets** (explicit, 26-connectivity):

```text
Operations per voxel per timestep:
- 26 neighbor checks
- 26 distance calculations
- 26 contribution calculations
- 26 atomic additions (slow!)
Total: ~100+ operations

For 100³ = 1M voxels:
~100M+ operations per timestep
```

**Performance Verdict**: **PDE is ~10× faster**

#### Pros & Cons Summary

**Wave Equation (PDE)**:

**Pros**:

- ✓ Fast: Simple stencil operations
- ✓ Accurate: Well-established numerical methods
- ✓ Stable: Known stability conditions (CFL)
- ✓ Conserves energy naturally
- ✓ GPU-friendly: Coalesced memory access
- ✓ Physically fundamental

**Cons**:

- ✗ Requires small timesteps (CFL condition)
- ✗ Needs 3 amplitude arrays in memory
- ✗ Fixed neighbor connectivity

**Huygens Wavelets** (explicit):

**Pros**:

- ✓ Intuitive physical interpretation
- ✓ Flexible directional weighting
- ✓ Can handle complex geometries

**Cons**:

- ✗ Computationally expensive (~10× slower)
- ✗ Many atomic operations (GPU bottleneck)
- ✗ Energy conservation requires careful implementation
- ✗ More complex to code and debug

#### Recommended

**Use PDE (Wave Equation)** for LEVEL-1:

1. Start with 6-connectivity (face neighbors) for speed
2. Upgrade to 18 or 26-connectivity if isotropy issues arise
3. Use second-order finite difference scheme (leap-frog)
4. Enforce CFL stability condition
5. Monitor energy conservation

**When to Consider Huygens**:

- Complex geometries with curved boundaries
- Directional wave sources with specific emission patterns
- Adaptive mesh refinement (not relevant for LEVEL-1)

---

### Answer 4: Wave Direction Computation

#### The Challenge

The wave equation `∂²ψ/∂t² = c²∇²ψ` only evolves the **scalar amplitude** `ψ`. It doesn't directly compute wave direction.

**But we need direction for**:

- Momentum transfer: `p = ρψ × direction`
- Force calculations: Direction of energy flow
- Particle-wave interactions: Reflection angle
- Visualization: Wave propagation arrows

#### Solution: Compute Direction from Phase Gradient

##### Wave Direction = Gradient of Phase

In wave physics, the wave propagation direction is the gradient of the phase field:

```text
wave_direction = ∇φ / |∇φ|
```

Where `φ` is the wave phase at each voxel.

#### Computing Phase from Amplitude

For a traveling wave:

```text
ψ(x,t) = A(x) cos(kx - ωt + φ₀)
```

The phase at position x and time t is:

```text
φ(x,t) = arctan(ψ_imaginary / ψ_real)
```

But for a real-valued field (which we have), we need the **analytic signal** approach.

#### Practical Implementation: Energy Flux Method

**Better approach**: Compute wave direction from **energy flux** (Poynting-like vector for waves).

**Energy flux density** (energy flow direction):

```text
S = -c² × ψ × ∇ψ
```

Where:

- `ψ` = current amplitude
- `∇ψ` = spatial gradient of amplitude
- Direction of S = direction of energy flow = wave direction

**Implementation**:

```python
@ti.kernel
def compute_wave_direction(self):
    """
    Compute wave propagation direction from energy flux.

    Energy flux: S = -c² × ψ × ∇ψ
    Direction: normalized S
    """
    c = ti.f32(constants.EWAVE_SPEED)

    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Current displacement
            psi = self.displacement_am[i,j,k]

            # Amplitude gradient
            grad_x = (self.displacement_am[i+1,j,k] - self.displacement_am[i-1,j,k]) / (2.0 * self.dx_am)
            grad_y = (self.displacement_am[i,j+1,k] - self.displacement_am[i,j-1,k]) / (2.0 * self.dx_am)
            grad_z = (self.displacement_am[i,j,k+1] - self.displacement_am[i,j,k-1]) / (2.0 * self.dx_am)

            grad_psi = ti.Vector([grad_x, grad_y, grad_z])

            # Energy flux vector
            S = -c**2 * psi * grad_psi

            # Normalize to get direction
            S_mag = S.norm()
            if S_mag > 1e-12:  # Avoid division by zero
                self.wave_direction[i,j,k] = S / S_mag
            else:
                self.wave_direction[i,j,k] = ti.Vector([0.0, 0.0, 0.0])
```

#### Alternative: Velocity-Based Direction

For waves, the **time derivative** of amplitude gives wave velocity:

```text
v_wave = ∂ψ/∂t
```

Direction of velocity = direction of wave propagation.

**Implementation**:

```python
@ti.kernel
def compute_wave_direction_velocity(self):
    """
    Compute wave direction from temporal derivative.

    Wave velocity: v = ∂ψ/∂t ≈ (ψ_current - ψ_old) / dt
    Direction: gradient of velocity field
    """
    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Time derivative (wave velocity)
            v_wave = (self.displacement_am[i,j,k] - self.amplitude_old[i,j,k]) / dt

            # Gradient of velocity gives acceleration direction
            # (This is less direct, energy flux method is better)
            # ...
```

**This method is less reliable** - use energy flux method instead.

#### Storage and Update Frequency

**Storage**:

```python
# In WaveField class __init__
self.wave_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
```

**Update Frequency**:

- Compute wave direction **every timestep** after amplitude update
- Or compute **only when needed** (for force calculations, visualization)
- Trade-off: Computation cost vs storage access

**Recommended**: Compute every timestep for consistency.

#### Summary: Wave Direction Pipeline

```text
1. Initialize amplitude field (initial conditions)
   ↓
2. Propagate amplitude using wave equation
   amplitude_new = 2×amplitude - amplitude_old + cfl_factor×Laplacian
   ↓
3. Compute wave direction from energy flux
   S = -c² × ψ × ∇ψ
   wave_direction = S / |S|
   ↓
4. Use direction for:
   - Momentum calculations: p = ρψ × direction
   - Particle reflections: Incident and reflected angles
   - Visualization: Arrow fields
   - Force directionality: Energy flow patterns
```

---

## The Complete Picture

### Wave Field → Forces → Motion

```text
┌─────────────────────────────────────────────────────────┐
│ 1. WAVE FIELD (grid of voxels)                          │
│    ├─ Initial energy charge (point/plane/spherical)     │
│    ├─ Stabilization period (waves propagate/reflect)    │
│    └─ Quasi-steady state (omni-directional field)       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. WAVE PROPAGATION (PDE evolution)                     │
│    ├─ Wave equation: ∂²ψ/∂t² = c²∇²ψ                    │
│    ├─ Laplacian couples neighboring voxels              │
│    ├─ Interference: constructive/destructive            │
│    ├─ Reflection: boundaries + wave centers             │
│    ├─ Standing waves form around particles              │
│    └─ Direction from energy flux: S = -c²ψ∇ψ            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. FORCE GENERATION (amplitude gradients)               │
│    ├─ Force field from energy gradient: F = -∇E [N]     │
│    ├─── Monochromatic: F = -2ρVf²×A×∇A (∇f≈0)           │
│    ├─── Full: F = -2ρVAf×[f∇A + A∇f]                    │
│    ├─ Forces emerge from wave patterns                  │
│    ├─ Electric: wave reflection patterns (charges)      │
│    ├─ Magnetic: moving wave patterns (currents)         │
│    └─ Gravitational: amplitude shading (mass)           │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. PARTICLE MOTION (Newton's laws)                      │
│    ├─ Interpolate force at particle position            │
│    ├─ Acceleration: a = F/m                             │
│    ├─ Update velocity: v_new = v_old + a×dt             │
│    ├─ Update position: x_new = x_old + v×dt             │
│    └─ Particles move toward amplitude minimum (MAP)     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 5. PARTICLE-FIELD INTERACTION (feedback loop)           │
│    ├─ Particles act as wave reflectors                  │
│    ├─ Create standing wave patterns                     │
│    ├─ Trapped energy = particle mass: m = E/c²          │
│    ├─ Standing wave radius: r = n×λ/2                   │
│    └─ Force between particles from wave overlap         │
└─────────────────────────────────────────────────────────┘
```

### The Beautiful Emergence

**All forces unified from wave amplitude gradients**:

- **Electric force**: Different wave reflection patterns (positive/negative charges have different reflection phases)
- **Magnetic force**: Moving wave patterns (velocity-dependent, Lorentz force emerges)
- **Gravitational force**: Amplitude shading from trapped energy (mass creates amplitude minimum)
- **Strong force**: Near-field standing wave coupling between adjacent wave centers
- **Orbital motion**: Balance between kinetic energy and amplitude gradient force

**Everything emerges from two equations**:

```text
∂²ψ/∂t² = c²∇²ψ                             (wave propagation)
F = -∇E = -2ρVAf×[f∇A + A∇f]               (force from energy gradient, EWT frequency-centric)
F = -2ρVf²×A∇A                              (monochromatic, ∇f ≈ 0)
```

Where (frequency-centric formulation):

- First term (f²∇A): Primary force from amplitude gradients
- Second term (Af∇f): Secondary correction for variable frequency
- For monochromatic waves (∇f ≈ 0): F = -2ρVf²×A×∇A
- Equivalence: f² = (c/λ)² ensures dimensional consistency

This is the foundation of reality in Energy Wave Theory.

---

## Implementation Summary

### WaveField Class Methods

```python
@ti.data_oriented
class WaveField:
    """Complete wave field with force calculation."""

    def __init__(self, nx, ny, nz, wavelength_m, points_per_wavelength=40):
        # ... initialization with attometer scaling ...

        # Three amplitude fields for wave equation (leap-frog)
        self.amplitude_old = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.displacement_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.amplitude_new = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

        # Wave direction (computed from energy flux)
        self.wave_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))

        # Force field (in Newtons)
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))

    @ti.kernel
    def propagate_wave_field(self, dt: ti.f32):
        """PDE-based wave propagation."""
        # See Answer 2
        pass

    @ti.kernel
    def compute_wave_direction(self):
        """Compute direction from energy flux."""
        # See Answer 4
        pass

    @ti.kernel
    def compute_force_field_newtons(self):
        """Compute force in Newtons from amplitude gradient (EWT)."""
        # See Answer 1
        pass

    def update_timestep(self, dt):
        """Complete wave field update for one timestep."""
        # 1. Propagate wave amplitude
        self.propagate_wave_field(dt)

        # 2. Compute wave direction
        self.compute_wave_direction()

        # 3. Compute force field
        self.compute_force_field_newtons()

        # 4. Apply boundary conditions
        self.apply_boundary_conditions()
```

### Simulation Loop

```python
# Initialize
wave_field = WaveField(nx=100, ny=100, nz=100,
                       wavelength_m=constants.EWAVE_LENGTH)

# Charge initial energy using Wolff's standing wave solution
wave_field.charge_spherical_standing_wave(
    center=center,
    frequency=constants.EWAVE_FREQUENCY,
    amplitude=constants.EWAVE_AMPLITUDE,
    initial_phase=0.0
)

# Stabilization phase (waves naturally evolve via PDE)
for step in range(stabilization_steps):
    wave_field.update_timestep(dt)

# Main simulation with particles
for step in range(simulation_steps):
    # 1. Update wave field
    wave_field.update_timestep(dt)

    # 2. Update particles (see 05_MATTER.md)
    particles.update_positions(wave_field.force, dt)

    # 3. Apply particle reflections to field
    particles.apply_reflections_to_field(wave_field)
```

---

**Status**: Comprehensive framework defined with force calculation in Newtons and wave propagation via PDE

**Next Steps**: Implement and validate wave equation solver with energy conservation checks

**Related Documentation**:

- [`01_WAVE_FIELD.md`](./01_WAVE_FIELD.md) - Field architecture and indexing
- [`02_WAVE_PROPERTIES.md`](./02_WAVE_PROPERTIES.md) - WaveField class definition
- [`03_WAVE_ENGINE.md`](./03_WAVE_ENGINE.md) - Complete wave engine details
- [`05_MATTER.md`](./05_MATTER.md) - Particle dynamics and interactions
- [`06_FORCE_MOTION.md`](./06_FORCE_MOTION.md) - Force types and motion equations
