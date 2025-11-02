# FORCES & MOTION

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
1. [The Electron's Special Role](#the-electrons-special-role)
   - [Wave Transformation](#wave-transformation)
   - [EM Wave Generation](#em-wave-generation)
1. [Measurable vs Point Properties](#measurable-vs-point-properties)
   - [Point Properties](#point-properties)
   - [Derived Properties](#derived-properties)
   - [Momentum Transfer](#momentum-transfer)
1. [Near-Field vs Far-Field](#near-field-vs-far-field)
   - [Behavior Differences](#behavior-differences)
   - [Wave Formation Zones](#wave-formation-zones)
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
- Gravitational field = reflected wave patterns from mass (trapped waves)
- All forces = amplitude gradients in wave field

**Mathematical Expression**:

```text
F = -∇A
```

Force is the negative gradient of amplitude. Particles move toward regions of lower amplitude (MAP: Minimum Amplitude Principle).

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

## Force Field Types

### Electric Field

**Electric Force from Waves**:

- Charged particle = specific wave reflection pattern
- Different from uncharged particle (different standing wave configuration)
- Creates different amplitude gradient pattern
- Can be attractive OR repulsive (unlike gravity)

**Charge Types**:

- **Positive charge**: One wave reflection pattern
- **Negative charge**: Different (inverted?) wave reflection pattern
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

- Mass = trapped energy in standing waves around particle
- More mass = more wave energy = stronger wave reflections
- Reflected waves create amplitude gradient around massive particles
- Other particles experience force from this gradient

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

## The Electron's Special Role

### Wave Transformation

**Electron as Transformer**:

- Electron has unique wave center configuration
- Acts as special reflector with transformation properties
- Incoming energy waves → reflected as EM waves
- Like a wavelength/frequency converter

**Why Electron is Special**:

- Specific standing wave pattern (two-center structure?)
- Resonance properties different from other particles
- Can oscillate at different frequencies
- Each oscillation frequency → EM wave of that frequency

### EM Wave Generation

**Oscillating Electron**:

1. Electron oscillates (driven by external wave or force)
2. Oscillation modulates wave reflection pattern
3. Reflected waves have EM wave character
4. EM waves propagate outward at speed c

**Frequency Relationship**:

- Electron oscillation frequency = EM wave frequency
- Higher frequency oscillation → higher frequency EM wave
- Energy of EM wave ∝ frequency (E = hf)

**Applications**:

- Accelerating electrons → EM radiation (synchrotron, antenna)
- Electron transitions in atoms → photons (specific frequencies)
- All light/radio/X-rays from electron oscillations

## Measurable vs Point Properties

### Point Properties

**Stored at Each Voxel**:

- Amplitude: Instantaneous value at [i,j,k]
- Density: Local compression/rarefaction
- Speed: Oscillation velocity at point
- Direction: Wave propagation direction at point
- Phase: Position in wave cycle

**Direct Access**:

```python
amp = amplitude[i, j, k]
dir = wave_direction[i, j, k]
```

### Derived Properties

**Computed from Field**:

- **Wavelength λ**: Measured as distance between wave crests
  - Not stored, measured from spatial pattern
  - `λ = distance(amplitude_max[n], amplitude_max[n+1])`

- **Frequency f**: Can be stored OR derived
  - If stored: propagates with wave
  - If derived: `f = c/λ` from measured wavelength

- **Energy**: Integral of energy density
  - `E_total = Σ energy_density[i,j,k] * dx³`

**Measurement Algorithms**:

```python
@ti.kernel
def measure_wavelength() -> ti.f32:
    """Measure wavelength from spatial pattern."""
    # Find two successive amplitude maxima
    max_positions = find_amplitude_maxima()
    wavelength = distance(max_positions[0], max_positions[1])
    return wavelength
```

### Momentum Transfer

**Momentum in Wave Field**:

- Momentum density = wave amplitude × propagation direction
- Transfer through wave interactions
- Conservation: total momentum conserved

**Mechanism**:

- Wave carries momentum
- Wave reflects from particle → momentum transfer
- Particle gains/loses momentum from wave
- Net momentum conserved (particle + field)

## Near-Field vs Far-Field

### Behavior Differences

**Near-Field** (r < few λ):

- Complex interference patterns
- Standing waves dominate
- Strong amplitude gradients (strong forces)
- Multiple wavelength components
- Non-spherical patterns

**Far-Field** (r >> λ):

- Simpler traveling waves
- Spherical wave fronts
- Weak amplitude gradients (weak forces)
- Single wavelength dominant
- 1/r amplitude falloff

**Transition Region** (r ≈ λ):

- Mixed behavior
- Depends on source geometry
- Important for particle interactions

### Wave Formation Zones

**Formation Region**:

- Occurs in near-field around particles
- Standing waves "lock in" particle structure
- Determines particle properties (mass, charge, etc.)
- Where particle identity is established

**Stable Patterns**:

- Standing wave nodes define particle structure
- Node positions at r = nλ/2
- Specific patterns = specific particles
- Changes in pattern = particle transformation

**Examples**:

- Neutrino: Simple spherical standing wave
- Electron: Two-center pattern with specific node structure
- Proton: Complex multi-center pattern

## Particle Motion from Forces

### Minimum Amplitude Principle (MAP)

**Single Governing Principle**: Particles move to minimize amplitude.

**Physical Motivation**:

- Forces arise from amplitude gradients in the wave field
- Greater wave amplitude = higher momentum density
- Force vectors point toward decreasing wave amplitude
- Particles seek lowest energy configuration
- Amplitude represents wave intensity/pressure
- High amplitude = high pressure → repulsive
- Low amplitude = low pressure → attractive

**Mathematical Statement**:

```text
F = -∇A
```

Force points toward decreasing amplitude (downhill on amplitude landscape).

**Implications**:

- Particles repelled from high-amplitude regions
- Particles attracted to low-amplitude regions (nodes)
- Creates effective "forces" between particles
- Emergent gravity, EM, all forces from MAP

### Force-Driven Motion

**Force Calculation**:

Force computed from amplitude gradient in field (see [`03_WAVE_ENGINE.md` - Force Calculation](./03_WAVE_ENGINE.md#force-calculation))

```python
F[i,j,k] = -∇A[i,j,k]
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

    # 1. Compute force field from wave amplitude
    compute_force_field()  # F = -∇A

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

1. Implement amplitude gradient force (F = -∇A)
2. Test with simple wave patterns
3. Verify MAP (particles seek amplitude minimum)

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

- [`03_WAVE_ENGINE.md`](./03_WAVE_ENGINE.md) - Wave propagation creating these fields
- [`05_MATTER.md`](./05_MATTER.md) - How particles respond to emergent forces
- [`02_WAVE_PROPERTIES.md`](./02_WAVE_PROPERTIES.md) - Properties that create fields
