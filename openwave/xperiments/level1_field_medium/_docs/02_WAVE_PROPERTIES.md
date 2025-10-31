# WAVE PROPERTIES

## Table of Contents

1. [Overview](#overview)
1. [Scalar Properties (Magnitude)](#scalar-properties-magnitude)
   - [Speed (c)](#speed-c)
   - [Amplitude (A)](#amplitude-a)
   - [Wavelength (λ)](#wavelength-λ)
   - [Frequency (f)](#frequency-f)
   - [Density](#density)
   - [Energy](#energy)
   - [Phase (φ)](#phase-φ)
1. [Vector Properties (Direction + Magnitude)](#vector-properties-direction--magnitude)
   - [Wave Propagation Direction](#wave-propagation-direction)
   - [Amplitude Direction (Wave Mode)](#amplitude-direction-wave-mode)
   - [Velocity (Granules/Particles Only)](#velocity-granulesparticles-only)
   - [Force](#force)
1. [Field Storage in Taichi](#field-storage-in-taichi)
   - [Scalar Fields](#scalar-fields)
   - [Vector Fields](#vector-fields)
1. [Property Relationships](#property-relationships)
1. [LEVEL-0 vs LEVEL-1 Properties](#level-0-vs-level-1-properties)

## Overview

Wave field attributes represent physical quantities and wave disturbances stored at each voxel in the field-based medium. Properties are categorized as **scalar** (magnitude only) or **vector** (magnitude + direction).

## Scalar Properties (Magnitude)

### Speed (c)

**Wave Speed** (constant, medium property):

- **For granules** (LEVEL-0): Speed varies as `sin(ωt)` (oscillation)
- **For waves** (LEVEL-1): Constant propagation speed through medium
  - Depends on medium density and properties
  - Different for wave types:
    - Standing waves: nodes fixed
    - Traveling waves: constant `c`
    - Transverse vs longitudinal modes

**Storage**: Typically derived from medium properties, not stored per-voxel

### Amplitude (A)

**Maximum Displacement/Disturbance**:

- **Granule-based** (LEVEL-0):
  - Displacement from equilibrium position
  - Density fluctuation `ρ / ρ_avg`
  - Phase `φ` per granule
- **Field-based** (LEVEL-1):
  - Amplitude at voxel
  - Proportional to density: `A ∝ ρ`
  - Proportional to pressure: `A ∝ P`
  - Represents energy density at that location

**Storage**: `ti.field(dtype=ti.f32)` per voxel

**Physical meaning**:

- **At maximum displacement (amplitude)**: Maximum potential energy, zero velocity
  - Force is maximum (pulling back toward equilibrium)
  - All energy stored as potential energy in displacement/compression
- **At equilibrium position (zero displacement)**: Maximum kinetic energy, maximum velocity
  - Granules/voxels moving fastest through equilibrium
  - All energy is kinetic (motion)
  - Zero restoring force at this instant
- **Energy oscillation**: Energy continuously converts between kinetic ↔ potential
- **Total amplitude** determines total energy in the wave: `E_total ∝ A²`
- Negative amplitude = displacement in opposite direction from positive

### Wavelength (λ)

**Spatial Period of Wave**:

- Distance between successive wave crests/troughs
- **Not stored directly** in fields
- **Derived/measured** from spatial patterns
- Used to calculate voxel resolution: `dx = λ / points_per_wavelength`

**Calculation**: Measure distance between amplitude maxima in field

### Frequency (f)

**Temporal Period of Wave**:

- `f = c / λ` (wave equation)
- **Spatial frequency**: `ξ = 1/λ` (inverse wavelength)
- Can be stored per-voxel if multiple wave sources with different frequencies

**Storage**: Optional `ti.field(dtype=ti.f32)` if needed for multi-frequency waves

### Density

**Medium Density at Voxel**:

- Energy density
- Mass density (for matter simulations)
- Related to amplitude via equation of state

**Storage**: `ti.field(dtype=ti.f32)` per voxel

**Physical meaning**:

- Represents compression/rarefaction of medium
- Higher density = wave compression
- Lower density = wave rarefaction

### Energy

**Energy Density at Voxel**:

- **Kinetic energy** (motion): `E_k ∝ v²`
  - Maximum at equilibrium position (zero displacement)
  - Zero at maximum displacement (turning points)
- **Potential energy** (compression/displacement): `E_p ∝ A²`
  - Maximum at maximum displacement
  - Zero at equilibrium position
- **Total energy**: `E_total = E_kinetic + E_potential = constant`
- **Energy oscillation**: `E_k ↔ E_p` (continuously converts)
- **Amplitude-energy relationship**: `E_total ∝ A²` (energy proportional to amplitude squared)

**Storage**: `ti.field(dtype=ti.f32)` per voxel (optional, can be computed)

**Conservation**: Total energy must be conserved across entire field

**Wave cycle**:

1. At `t=0` (equilibrium): Max velocity, max KE, zero PE
2. At `t=T/4` (max displacement): Zero velocity, zero KE, max PE
3. At `t=T/2` (equilibrium, opposite): Max velocity, max KE, zero PE
4. At `t=3T/4` (max displacement, opposite): Zero velocity, zero KE, max PE

### Phase (φ)

**Wave Phase at Voxel**:

- Position within wave cycle (0 to 2π)
- Critical for interference patterns
- Determines constructive/destructive interference

**Storage**: `ti.field(dtype=ti.f32)` per voxel

**Physical meaning**:

- Phase difference determines interference
- `Δφ = 0, 2π, ...` → constructive
- `Δφ = π, 3π, ...` → destructive

## Vector Properties (Direction + Magnitude)

### Wave Propagation Direction

**Direction of Wave Travel**:

- Unit vector indicating propagation direction
- Can vary spatially (curved wavefronts, reflections)
- For spherical waves: radial from source
- For plane waves: uniform direction

**Storage**: `ti.Vector.field(3, dtype=ti.f32)` per voxel

**Physical meaning**:

- Points toward energy flow direction
- Orthogonal to wavefronts (for isotropic media)

### Amplitude Direction (Wave Mode)

**Direction of Displacement/Oscillation**:

- **Longitudinal waves**: Parallel to propagation direction
  - Compression waves
  - Sound-like waves
- **Transverse waves**: Perpendicular to propagation direction
  - Shear waves
  - EM-like waves (in EWT context)

**Storage**: `ti.Vector.field(3, dtype=ti.f32)` per voxel

**Physical meaning**:

- Defines wave polarization
- Non-linear for multi-source spherical waves
- Can point in all directions depending on wave superposition

### Velocity (Granules/Particles Only)

**Rate of Position Change**:

- **LEVEL-0 only**: Granules have velocity vectors
- **LEVEL-1**: Field voxels don't move, but can store flow velocity
- Can represent momentum density in field

**Storage**: `ti.Vector.field(3, dtype=ti.f32)` per voxel (for momentum)

**Physical meaning**:

- For particles: actual motion velocity
- For fields: local flow/current of energy

### Force

**Force Vector at Voxel**:

- Derived from amplitude gradients: `F ∝ -∇A`
- Points toward minimum amplitude (MAP: Minimum Amplitude Principle)
- Drives particle motion in LEVEL-1

**Storage**: `ti.Vector.field(3, dtype=ti.f32)` per voxel (computed)

**Physical meaning**:

- Gradient of potential (amplitude)
- Determines particle acceleration
- Source of emergent forces (gravity, EM, etc.)

## Field Storage in Taichi

### Scalar Fields

```python
import taichi as ti

# Required scalar fields
amplitude = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
phase = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

# Optional scalar fields
energy = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
frequency = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # If multi-frequency
```

### Vector Fields

```python
# Required vector fields
wave_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
amplitude_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))

# Computed/optional vector fields
velocity = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
force = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
```

## Property Relationships

Key wave equation relationships:

```python
# Wave equation fundamentals
f = c / wavelength              # Frequency from speed and wavelength
omega = 2 * pi * f              # Angular frequency
k = 2 * pi / wavelength         # Wave number
xi = 1 / wavelength             # Spatial frequency

# Energy relationships
E_total = E_kinetic + E_potential   # Total energy (conserved)
E_kinetic ∝ v²                      # Kinetic energy from velocity
E_potential ∝ A²                    # Potential energy from displacement
E_total ∝ A²                        # Total energy proportional to amplitude squared

# Energy oscillation in time
# At equilibrium (A=0): E_kinetic = max, E_potential = 0
# At max displacement (A=max): E_kinetic = 0, E_potential = max

# Force from amplitude gradient
F = -gradient(amplitude)        # MAP: move toward lower amplitude

# Density-amplitude relationship (equation of state)
density ∝ amplitude             # For compression waves
```

## LEVEL-0 vs LEVEL-1 Properties

| Property | LEVEL-0 (Granule) | LEVEL-1 (Field) |
|----------|-------------------|-----------------|
| **Position** | Per-granule vector | Computed from index: `(i+0.5)*dx` |
| **Velocity** | Per-granule oscillation | Optional momentum density |
| **Displacement** | From equilibrium | Amplitude at voxel |
| **Density** | Count granules in region | Direct field value |
| **Phase** | Per-granule phase | Per-voxel phase |
| **Amplitude** | Displacement magnitude | Direct field value |
| **Wave Direction** | Inferred from motion | Stored vector field |
| **Forces** | Inter-granule forces | Computed from gradients |

**Key Difference**: LEVEL-1 stores properties directly at fixed grid locations, while LEVEL-0 computes from moving particles.

---

**Status**: Properties defined, ready for wave engine implementation

**Next Steps**: Implement wave propagation using these field properties
