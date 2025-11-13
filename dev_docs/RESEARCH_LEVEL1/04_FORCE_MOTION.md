# FORCE & MOTION

## Table of Contents

1. [Overview](#overview)
1. [Fundamental Principle](#fundamental-principle)
   - [All Forces from Waves](#all-forces-from-waves)
   - [Wave Derivations](#wave-derivations)
1. [Motion Equations](#motion-equations)
1. [Force Field Types](#force-field-types)
   - [Gravitational Field](#gravitational-field)
   - [Electric Field](#electric-field)
   - [Magnetic Field](#magnetic-field)
   - [Electromagnetic Waves](#electromagnetic-waves)
1. [Momentum Transfer](#momentum-transfer)
1. [Particle Motion from Forces](#particle-motion-from-forces)
   - [Minimum Amplitude Principle (MAP)](#minimum-amplitude-principle-map)
   - [Force-Driven Motion](#force-driven-motion)
   - [Position Updates](#position-updates)
   - [Particle Dynamics Algorithm](#particle-dynamics-algorithm)
1. [Implementation Strategy](#implementation-strategy)

## Overview

In Energy Wave Theory (EWT) as implemented in LEVEL-1, **all forces emerge from wave interactions**. There are no separate force fields—gravity, electromagnetism, and all other forces are derivations and compositions of the underlying energy wave field.

Electric, magnetic, gravitational, strong forces are disturbances on the energy wave.

**Key Paradigm Shift**:

- Traditional physics: Particles + separate force fields
- EWT/LEVEL-1: **Only waves**, particles and forces both emerge from wave patterns

## Fundamental Principle

### All Forces from Waves

**Core Concept**:

- Electric field = reflected wave patterns from charged particles
- Magnetic field = reflected wave patterns with specific geometry
- Gravitational field = longitudinal wave amplitude loss to spin (shading effect)
- All forces = amplitude gradients in wave field

**Mathematical Expression** (frequency-centric formulation):

```text
F = -∇E = -∇(u×V) = -2ρVfA × [f∇A + A∇f]   (full form, dual-term)
F = -2ρVf² × A∇A                             (monochromatic, ∇f = 0)
```

Where:

- u = ρ(fA)² is energy density (EWT, frequency-based, no ½ factor)
- ρ = medium density (3.860×10²² kg/m³)
- V = dx³ (voxel volume)
- f = frequency (Hz, where f = c/λ)
- A = amplitude (meters)

Force points toward decreasing energy (downhill on energy landscape). Particles move toward regions of lower amplitude (MAP: Minimum Amplitude Principle).

### Wave Derivations

**How Fields Emerge**:

1. **Particle** = wave center that reflects waves
2. **Reflection** creates standing wave pattern around particle
3. **Standing waves** create amplitude gradients
4. **Gradients** = forces experienced by other particles

**Composition**:

- Multiple particles → multiple overlapping wave patterns
- Superposition creates complex force fields
- Force on particle A from particle B = effect of B's reflected waves on A

## Motion Equations

```bash
p = m * v (momentum is conserved in collisions, Newton's cradle)
F = m * a (Newton's 2nd law)

Derivatives
v = dx/dt
a = dv/dt = d2x/dt2
F = dp/dt (force is diff of momentum)
F = -dU/dt (force is diff of potential)

Work / Energy
W = F * d

OSCILLATING MOTION
HARMONIC OSCLLIATOR (equation of motion)
Fs = -k * x = m * a (Hooke's + Newton's)
m * dx/dt = -k * x (differential equation)

solution, function of position over time
x(t) = A * cos(ω * t)
x(t) = A * cos(2pi*f * t)

ω = sqrt(k / m) (angular frequency)
f = ω / 2pi (frequency)
T = 2pi / ω (period)

U = 1/2 * k * x**2
```

## Force Calculation in Newtons

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

### Force in Newtons from Amplitude/Frequency Gradients

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

@ti.kernel
    def compute_force_field_newtons(self):
        """
        Compute force from amplitude gradient (EWT frequency-based formulation).

        Physics (Frequency-Based):
        - Energy density: u = ρ(fA)² (EWT, no ½ factor)
        - Force: F = -∇E = -∇(u×V) = -2ρVfA × [f∇A + A∇f]
        - Monochromatic: F = -2ρVf² × A∇A (when ∇f = 0)

        Force follows MAP (Minimum Amplitude Principle): particles move toward
        regions of lower amplitude (envelope, not instantaneous ψ).
        """
        ρ = ti.f32(constants.MEDIUM_DENSITY)  # 3.860e22 kg/m³
        f = ti.f32(constants.EWAVE_FREQUENCY) # 1.050e25 Hz
        dx_m = self.dx_am * constants.ATTOMETER
        V = dx_m**3

        # Force scaling factor (EWT frequency-based formulation)
        # F = -2ρVf² × A × ∇A  (monochromatic, ∇f = 0)
        force_scale = 2.0 * ρ * V * f**2

        for i, j, k in self.amplitude_am:
            if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
                A_m = self.amplitude_am[i,j,k] * constants.ATTOMETER

                # Gradient in attometer space (better precision)
                grad_x = (self.amplitude_am[i+1,j,k] - self.amplitude_am[i-1,j,k]) / (2.0 * self.dx_am)
                grad_y = (self.amplitude_am[i,j+1,k] - self.amplitude_am[i,j-1,k]) / (2.0 * self.dx_am)
                grad_z = (self.amplitude_am[i,j,k+1] - self.amplitude_am[i,j,k-1]) / (2.0 * self.dx_am)

                grad_vector = ti.Vector([grad_x, grad_y, grad_z])

                # Force in Newtons (frequency-based formulation)
                self.force[i,j,k] = -force_scale * A_m * grad_vector
```

### Force Direction Vector

**Understanding Force Direction**:

The force is a **vector** quantity with both magnitude and direction, derived from the complete energy gradient:

```text
Complete force (dual-term):
F = -∇E = -2ρVfA × [f∇A + A∇f]
F = -2ρVfA × [(f∂A/∂x + A∂f/∂x, f∂A/∂y + A∂f/∂y, f∂A/∂z + A∂f/∂z)]

Monochromatic simplification (∇f ≈ 0):
F = -2ρVf² × A∇A
F = -2ρVf²A × (∂A/∂x, ∂A/∂y, ∂A/∂z)
```

**Two Force Terms**:

1. **Amplitude gradient term** (primary, always present):
   - `F₁ = -2ρVfA × f∇A = -2ρVf² × A∇A`
   - Points toward **lower amplitude** (MAP: Minimum Amplitude Principle)
   - Dominant force in most scenarios

2. **Frequency gradient term** (secondary, when ∇f ≠ 0):
   - `F₂ = -2ρVfA × A∇f = -2ρVA²f × ∇f`
   - Points toward **lower frequency** (due to negative sign)
   - Present when multiple frequencies overlap

**When to Use Each Form**:

- **Monochromatic (single frequency)**: Use `F = -2ρVf² × A∇A` (∇f = 0)
  - Single wave source throughout field
  - Frequency uniform or slowly varying
  - Initial OpenWave implementation

- **Multi-frequency (multiple sources)**: Use `F = -2ρVfA × [f∇A + A∇f]`
  - Multiple particles with different frequencies
  - Wave interference from different sources
  - Advanced particle interactions

**Direction Meaning**:

- **Each component**:
  - `F_x = -2ρVfA × (f∂A/∂x + A∂f/∂x)`: Force component in x-direction
  - `F_y = -2ρVfA × (f∂A/∂y + A∂f/∂y)`: Force component in y-direction
  - `F_z = -2ρVfA × (f∂A/∂z + A∂f/∂z)`: Force component in z-direction
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

**Key Point**:

- The gradient operator `∇A` points toward maximum amplitude increase
- The negative sign in `-f∇A` reverses this to point toward amplitude **decrease** (MAP principle)
- The gradient operator `∇f` points toward maximum frequency increase
- The negative sign in `-A∇f` reverses this to point toward frequency **decrease**
- In multi-frequency scenarios, both gradient terms contribute to the net force direction

### Dimensional Analysis Verification (Frequency-Based)

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

### Particle Mass from Standing Waves

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

### Particle Acceleration and Motion

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

## Force Field Types

### Electric Field

**Electric Force from Waves**:

- Charged particle = specific wave reflection pattern
- Different from uncharged particle (different standing wave configuration)
- Creates different amplitude gradient pattern
- Can be attractive OR repulsive (unlike gravity)

**Charge Types**:

- **Positive charge**: One wave reflection pattern
- **Negative charge**: Different phase (inverted?) wave reflection pattern
- **Opposite charges attract**: Wave patterns create amplitude minimum between them
- **Like charges repel**: Wave patterns create amplitude maximum between them

**Implementation Questions** (to be researched from EWT papers):

- What distinguishes positive from negative charge at wave level?
- How do wave patterns differ between charge types?
- Why is electric force stronger than gravity?

### Magnetic Field

**Magnetic Force from Waves**:

- Moving charge = moving wave pattern
- Motion creates directional wave propagation
- Directional propagation = magnetic field component

**Velocity Dependence**:

- Stationary charge → only electric field (spherical pattern)
- Moving charge → electric + magnetic field (directional pattern)
- Faster motion → stronger magnetic field

**Force on Moving Charge**:

- Moving charge experiences wave pattern from other moving charges
- Force depends on relative velocities (Lorentz force)
- Cross-product nature (v × B) from wave directional effects

### Gravitational Field

**Gravitational Force from Waves**:

- longitudinal wave amplitude loss to spin (shading effect)
- Mass = trapped energy in standing waves around particle
- More mass = more wave energy = stronger wave reflections
- Other particles experience force from shading gradient

**Mechanism**:

1. Massive particle traps waves (standing wave pattern)
2. Trapped waves reflect incoming waves from other sources
3. Reflection creates amplitude minimum near massive particle
4. Other particles attracted toward amplitude minimum (MAP)
5. Result: Gravitational attraction

**Why Always Attractive?**:

- Wave reflection creates amplitude minimum (node region)
- All particles seek amplitude minimum (MAP)
- Therefore all particles attracted to massive objects

**1/r² Law**:

- Amplitude from spherical source/reflector ∝ 1/r
- Force ∝ amplitude gradient ∝ d/dr(1/r) ∝ 1/r²
- Natural consequence of spherical wave geometry

### Electromagnetic Waves

**EM Waves = Special Wave Type**:

- NOT the same as fundamental energy waves
- Created by electron's special transformation
- Electron transforms energy waves into EM waves
- EM waves propagate at speed c (like energy waves)

**Difference from Energy Waves**:

- Energy waves: Fundamental medium oscillations
- EM waves: Transformed waves from electron oscillation
- Both propagate at c, but different properties
- EM waves interact differently with matter

## Momentum Transfer

**Momentum in Wave Field**:

- Momentum density = wave amplitude × propagation direction
- Transfer through wave interactions
- Conservation: total momentum conserved

**Mechanism**:

- Wave carries momentum
- Wave reflects from particle → momentum transfer
- Particle gains/loses momentum from wave
- Net momentum conserved (particle + field)

## Particle Motion from Forces

### Minimum Amplitude Principle (MAP)

**Single Governing Principle**: Particles move to minimize amplitude.

**Physical Motivation**:

- Forces arise from displacement gradients in the wave field
- Greater wave amplitude = higher momentum density
- Force vectors point toward decreasing wave amplitude
- Particles seek lowest energy configuration
- Amplitude represents wave intensity/pressure
- High amplitude = high pressure → repulsive
- Low amplitude = low pressure → attractive

**Mathematical Statement** (frequency-centric):

```text
Energy density: u = ρ(fA)²
Force from energy gradient: F = -∇E = -∇(u×V)
Expanded form: F = -2ρVfA × [f∇A + A∇f]   (dual-term)
Monochromatic: F = -2ρVf² × A∇A           (when ∇f = 0)
```

Force points toward decreasing energy density (downhill on energy landscape).

**Implications**:

- Particles repelled from high-amplitude regions
- Particles attracted to low-amplitude regions (nodes)
- Creates effective "forces" between particles
- Emergent gravity, EM, all forces from MAP

### Force-Driven Motion

**Force Calculation** (frequency-based):

Force computed from amplitude gradient in field (see [`02_WAVE_ENGINE.md` - Force Calculation](./02_WAVE_ENGINE.md#force-calculation))

```python
# Full form with frequency gradients
F[i,j,k] = -2ρVfA × [f∇A + A∇f][i,j,k]

# Monochromatic approximation (single frequency, ∇f = 0)
F[i,j,k] = -2ρVf² × A∇A[i,j,k]

# Where:
# ρ = constants.MEDIUM_DENSITY  # 3.860e22 kg/m³
# f = constants.EWAVE_FREQUENCY # 1.050e25 Hz
# V = dx³ (voxel volume)
```

**Applying Force to Particle**:

```python
# For particle at position (x, y, z)
# Interpolate force from nearby voxels
F_particle = interpolate_field(force_field, x, y, z)

# Update velocity (Newton's second law)
a = F_particle / mass
v_new = v_old + a * dt

# Update position
x_new = x_old + v_new * dt
```

**Newton's Second Law**:

```text
F = ma
a = F / m
```

**Force Acts on Mass**:

- Amplitude gradient creates force
- Force accelerates mass (trapped wave energy)
- Larger mass → smaller acceleration for same force

**Distance and Work**:

- Force acts over distance: `W = F · d`
- Work changes kinetic energy: `ΔKE = W`
- Energy conservation maintained

### Position Updates

**Integration Scheme**:

```python
@ti.kernel
def update_particle_positions(dt: ti.f32):
    for p in particles:
        # Get force at particle position
        F = get_force_at_position(particles.pos[p])

        # Acceleration from force
        a = F / particles.mass[p]

        # Velocity Verlet integration
        particles.vel[p] += a * dt
        particles.pos[p] += particles.vel[p] * dt
```

**Boundary Handling**:

- Reflect particles at universe boundaries
- Elastic collision with walls
- Or periodic boundary conditions

```python
@ti.kernel
def apply_particle_boundaries():
    for p in particles:
        if particles.active[p]:
            # Reflective boundaries
            if particles.pos[p].x < 0:
                particles.pos[p].x = -particles.pos[p].x
                particles.vel[p].x = -particles.vel[p].x
            if particles.pos[p].x > L:
                particles.pos[p].x = 2*L - particles.pos[p].x
                particles.vel[p].x = -particles.vel[p].x
            # Similar for y, z
```

### Particle Dynamics Algorithm

**Complete Update Cycle**:

```python
@ti.kernel
def particle_dynamics_step(dt: ti.f32):
    """Complete particle dynamics for one timestep."""

    # 1. Compute force field from wave amplitude (frequency-based)
    compute_force_field()  # F = -2ρVf² × A∇A (monochromatic)

    # 2. Update each particle
    for p in range(max_particles):
        if particles.active[p]:
            # Get interpolated force at particle position
            F = interpolate_force(particles.pos[p])

            # Newton's second law
            a = F / particles.mass[p]

            # Velocity Verlet integration (half-step)
            particles.vel[p] += 0.5 * a * dt

            # Update position
            particles.pos[p] += particles.vel[p] * dt

            # Apply boundary conditions
            apply_boundary_reflections(p)

            # Recompute force at new position
            F_new = interpolate_force(particles.pos[p])
            a_new = F_new / particles.mass[p]

            # Velocity Verlet (second half-step)
            particles.vel[p] += 0.5 * a_new * dt

    # 3. Apply wave reflections at new particle positions
    for p in range(max_particles):
        if particles.active[p]:
            apply_wave_reflection_at_center(particles.pos[p])
```

**Force Interpolation** (trilinear):

```python
@ti.func
def interpolate_force(pos: ti.math.vec3) -> ti.math.vec3:
    """Interpolate force field at arbitrary position."""
    # Convert to grid coordinates
    x = pos.x / dx
    y = pos.y / dx
    z = pos.z / dx

    # Grid indices
    i = ti.cast(ti.floor(x), ti.i32)
    j = ti.cast(ti.floor(y), ti.i32)
    k = ti.cast(ti.floor(z), ti.i32)

    # Fractional parts
    fx = x - i
    fy = y - j
    fz = z - k

    # Trilinear interpolation
    F000 = force[i,   j,   k  ]
    F100 = force[i+1, j,   k  ]
    F010 = force[i,   j+1, k  ]
    F110 = force[i+1, j+1, k  ]
    F001 = force[i,   j,   k+1]
    F101 = force[i+1, j,   k+1]
    F011 = force[i,   j+1, k+1]
    F111 = force[i+1, j+1, k+1]

    # Interpolate
    F_interp = (
        F000 * (1-fx) * (1-fy) * (1-fz) +
        F100 * fx     * (1-fy) * (1-fz) +
        F010 * (1-fx) * fy     * (1-fz) +
        F110 * fx     * fy     * (1-fz) +
        F001 * (1-fx) * (1-fy) * fz     +
        F101 * fx     * (1-fy) * fz     +
        F011 * (1-fx) * fy     * fz     +
        F111 * fx     * fy     * fz
    )

    return F_interp
```

## Implementation Strategy

### Development Phases

**Phase 1 - Basic Forces**:

1. Implement energy gradient force (F = -2ρVf² × A∇A, frequency-based)
2. Test with simple wave patterns
3. Verify MAP (particles seek amplitude minimum / energy minimum)

**Phase 2 - Electric Analog**:

1. Implement different particle types (charges)
2. Create attractive and repulsive patterns
3. Test like/opposite charge interactions

**Phase 3 - Magnetic Analog**:

1. Add velocity-dependent forces
2. Implement moving particle wave patterns
3. Test Lorentz-like force (v × B analog)

**Phase 4 - Gravitational Analog**:

1. Create particles with mass (trapped wave energy)
2. Observe attraction between particles
3. Verify 1/r² force law

### Research Requirements

**From EWT Papers**:

- Exact wave patterns for different particle types
- Charge mechanism at wave level
- Magnetic field emergence from motion
- Electron transformation properties

**Validation**:

- Compare emergent forces to known physics
- Verify force laws (1/r², Coulomb, Lorentz)
- Test energy/momentum conservation

---

**Status**: Conceptual framework defined, needs EWT paper research for details

**Next Steps**: Study EWT papers for specific wave patterns of charged particles

**Related Documentation**:

- [`02_WAVE_ENGINE.md`](./02_WAVE_ENGINE.md) - Wave propagation creating these fields
- [`05_MATTER.md`](./05_MATTER.md) - How particles respond to emergent forces
- [`01b_WAVE_FIELD_properties.md`](./01b_WAVE_FIELD_properties.md) - Properties that create fields
