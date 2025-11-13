# MATTER (Wave Centers, Particles & Anti-particles)

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
1. [Mass and Energy](#mass-and-energy)
   - [Mass Accumulation](#mass-accumulation)
   - [Energy-Mass Relationship](#energy-mass-relationship)
1. [Implementation Details](#implementation-details)
   - [Taichi Particle System](#taichi-particle-system)
   - [Field-Particle Interaction](#field-particle-interaction)
   - [Boundary Conditions](#boundary-conditions)

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
- Example: 1 electron = 10 wave centers ("click" together)
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
    displacement[i,j,k] = -displacement[i,j,k]
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

**For particle motion dynamics**, see [`04_FORCE_MOTION.md` - Particle Motion from Forces](./04_FORCE_MOTION.md#particle-motion-from-forces)

### Near-Field vs Far-Field

#### Behavior Differences

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

#### Wave Formation Zones

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
- Electron: 10-center pattern with specific node structure
- Proton: Complex multi-center pattern

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

1. **Field → Particle**: Force from displacement gradient
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

---

**Status**: Matter/particle system architecture defined

**Next Steps**: Implement wave centers with reflection properties

**Related Documentation**:

- [`02_WAVE_ENGINE.md`](./02_WAVE_ENGINE.md) - Wave propagation and reflection at wave centers
- [`04_FORCE_MOTION.md`](./04_FORCE_MOTION.md) - Particle motion dynamics and force calculations
- [`07_VISUALIZATION.md`](./07_VISUALIZATION.md) - Visualizing particles and wave centers
