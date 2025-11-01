# MATTER (Particles, Wave Centers)

## Table of Contents

1. [Overview](#overview)
1. [Particle System Architecture](#particle-system-architecture)
   - [What are Wave Centers?](#what-are-wave-centers)
   - [Implementation Approach](#implementation-approach)
   - [Particle Count](#particle-count)
1. [Wave Center Properties](#wave-center-properties)
   - [Reflection Behavior](#reflection-behavior)
   - [Wave Inversion](#wave-inversion)
   - [Near-Field Effects](#near-field-effects)
1. [Particle Motion Rules](#particle-motion-rules)
   - [Minimum Amplitude Principle (MAP)](#minimum-amplitude-principle-map)
   - [Force-Driven Motion](#force-driven-motion)
   - [Position Updates](#position-updates)
1. [Mass and Energy](#mass-and-energy)
   - [Mass Accumulation](#mass-accumulation)
   - [Energy-Mass Relationship](#energy-mass-relationship)
   - [Force and Acceleration](#force-and-acceleration)
1. [Complex Structures](#complex-structures)
   - [Composite Particles](#composite-particles)
   - [Electron Formation](#electron-formation)
   - [Particle Binding](#particle-binding)
1. [Implementation Details](#implementation-details)
   - [Taichi Particle System](#taichi-particle-system)
   - [Field-Particle Interaction](#field-particle-interaction)
   - [Boundary Conditions](#boundary-conditions)
1. [Particle Dynamics Algorithm](#particle-dynamics-algorithm)

## Overview

In LEVEL-1, **particles are wave centers** that reflect waves back into the medium. Unlike LEVEL-0's millions of granules, LEVEL-1 simulates only hundreds of fundamental particles (neutrinos, electrons, etc.) that interact with the wave field.

**Key Principle**: Particles are not point masses but **wave reflection centers** that create standing wave patterns and experience forces from amplitude gradients in the field.

## Particle System Architecture

### What are Wave Centers?

**Wave Centers** are special points in the field that:

- Reflect incoming waves (like boundaries)
- Create standing wave patterns around themselves
- Experience forces from wave amplitude gradients
- Move to minimize amplitude (MAP: Minimum Amplitude Principle)

**Not Physical Objects**: Wave centers are computational markers where wave reflection occurs. The "particle" is actually the entire standing wave pattern around the center.

### Implementation Approach

**Taichi Particle System**:

- Use `ti.field` for particle positions
- Much smaller count than LEVEL-0 granules
- Particles interact with wave field, not with each other directly
- Interactions emerge from wave-mediated forces

**Hybrid Approach**:

- **Field**: Wave propagation (fixed grid)
- **Particles**: Wave centers (mobile points)
- **Coupling**: Particles reflect waves, waves apply forces to particles

### Particle Count

**LEVEL-1 Scaling**:

- **Fundamental particles**: Hundreds to thousands (not millions)
- Example: 1 neutrino = 1 wave center
- Example: 1 electron = 2 wave centers ("click" together)
- Example: 1 proton = complex multi-center structure

**Comparison to LEVEL-0**:

| System | LEVEL-0 | LEVEL-1 |
|--------|---------|---------|
| **Granules** | ~1 million | N/A (grid-based) |
| **Particles** | N/A | Hundreds-thousands |
| **What particles represent** | Medium granules | Fundamental particles |
| **Computational cost** | High (granule interactions) | Moderate (field + few particles) |

## Wave Center Properties

### Reflection Behavior

**Wave Reflection at Centers**:

- Similar to hard boundary reflection
- Incoming wave inverts phase: `ψ_reflected = -ψ_incident`
- Creates spherical reflected waves around center
- Reflection strength can vary (partial reflection possible)

**Effect on Field**:

```python
# Simplified reflection at wave center position (xc, yc, zc)
@ti.func
def apply_wave_center_reflection(xc: ti.f32, yc: ti.f32, zc: ti.f32):
    # Find nearest voxel to particle
    i = ti.cast(xc / dx, ti.i32)
    j = ti.cast(yc / dx, ti.i32)
    k = ti.cast(zc / dx, ti.i32)

    # Invert amplitude (hard reflection)
    amplitude[i,j,k] = -amplitude[i,j,k]
```

### Wave Inversion

**Propagation Direction Inversion**:

- Incoming wavelet from direction `d`
- Reflects back in direction `-d`
- Creates interference patterns (standing waves)

**Standing Wave Formation**:

- Reflected waves interfere with incoming waves
- Forms nodes and antinodes around particle
- Node spacing: `λ/2`
- Creates "shells" of amplitude at specific radii

### Near-Field Effects

**Near-Field vs Far-Field**:

- **Near-field**: Close to wave center (r < few λ)
  - Complex interference patterns
  - Standing wave shells
  - Strong amplitude gradients (forces)

- **Far-field**: Distant from wave center (r >> λ)
  - Simpler traveling waves
  - Weaker amplitude gradients
  - Particle appears as point source/reflector

**Wave Formation Zone**:

- Standing wave patterns form near wave centers
- Determines particle structure (size, shape)
- Different particles have different near-field patterns

## Particle Motion Rules

### Minimum Amplitude Principle (MAP)

**Single Governing Principle**: Particles move to minimize amplitude.

**Physical Motivation**:

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

## Mass and Energy

### Mass Accumulation

**Mass from Wave Reflection**:

- As particles reflect waves, they accumulate mass
- Mass represents accumulated/trapped energy
- More wave reflection → more mass

**Mechanism**:

- Standing waves around particle contain energy
- Energy trapped in near-field = particle mass
- `E = mc²` relationship

**Implementation**:

```python
# Mass can be computed from standing wave energy
@ti.kernel
def compute_particle_mass(p: ti.i32) -> ti.f32:
    total_energy = 0.0
    # Sum energy in near-field region around particle
    for voxels in near_field_region(particles.pos[p]):
        total_energy += energy_density[voxel]
    # Mass from E=mc²
    mass = total_energy / (c * c)
    return mass
```

### Energy-Mass Relationship

**E = mc²**:

- Particle mass is wave energy
- Standing waves store energy
- Different particles have different mass (different wave patterns)

**Conservation**:

- Total energy (field + particle mass) conserved
- Energy can transfer: field ↔ particle mass
- Particle motion carries kinetic energy

### Force and Acceleration

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

## Complex Structures

### Composite Particles

**Multi-Center Particles**:

- Fundamental particles can combine
- Multiple wave centers form bound state
- Standing wave pattern spans multiple centers

**Examples**:

- **Electron**: 2 wave centers in specific configuration
- **Proton**: Complex multi-center structure
- **Neutron**: Different multi-center arrangement

### Electron Formation

**"Click" Event**:

- Two wave centers approach each other
- At critical distance, standing wave pattern locks
- Centers bind together → electron forms
- Transformation visible in simulation

**Conditions**:

- Specific approach velocity
- Specific wave phase relationship
- Specific energy configuration

**Visualization**:

- Show two centers approaching
- Standing wave pattern changes
- Sudden "snap" into bound configuration
- New composite particle with different properties

### Particle Binding

**Binding Mechanism**:

- Wave interference creates attractive/repulsive regions
- Centers settle into stable configuration
- Minimum amplitude principle maintains binding
- Similar to quantum mechanical orbitals

**Stability**:

- Bound state = local energy minimum
- Perturbations cause oscillations, not unbinding
- Different stable configurations = different particles

## Implementation Details

### Taichi Particle System

**Data Structure**:

```python
import taichi as ti

# Particle field
max_particles = 1000
particles = ti.StructField({
    'pos': ti.math.vec3,      # Position (meters)
    'vel': ti.math.vec3,      # Velocity (m/s)
    'mass': ti.f32,           # Mass (kg)
    'active': ti.i32,         # 1 if active, 0 if not
}, shape=max_particles)
```

**Initialization**:

```python
@ti.kernel
def init_particles():
    for p in range(num_particles):
        particles.pos[p] = initial_positions[p]
        particles.vel[p] = ti.Vector([0.0, 0.0, 0.0])
        particles.mass[p] = particle_mass
        particles.active[p] = 1
```

### Field-Particle Interaction

**Two-Way Coupling**:

1. **Field → Particle**: Force from amplitude gradient
2. **Particle → Field**: Wave reflection at particle location

**Algorithm**:

```python
# Each timestep:
# 1. Propagate wave field (see 03_WAVE_ENGINE.md)
propagate_wave_field(dt)

# 2. Apply wave reflections at particle positions
for p in active_particles:
    apply_wave_center_reflection(particles.pos[p])

# 3. Compute forces on particles from field
compute_particle_forces()

# 4. Update particle positions
update_particle_positions(dt)
```

### Boundary Conditions

**Particle Boundaries**:

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

## Particle Dynamics Algorithm

### Complete Update Cycle

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

### Interpolation

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

---

**Status**: Particle system architecture defined

**Next Steps**: Implement particle dynamics with MAP force calculation

**Related Documentation**:

- [`03_WAVE_ENGINE.md`](./03_WAVE_ENGINE.md) - Wave propagation and force calculation
- [`06_FORCE_MOTION.md`](./06_FORCE_MOTION.md) - How forces emerge from waves
- [`04_VISUALIZATION.md`](./04_VISUALIZATION.md) - Visualizing particles and wave centers
