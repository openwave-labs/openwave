# WAVE ENGINE

## Table of Contents

1. [Overview](#overview)
1. [Energy Wave Inital Charge](#energy-wave-initial-charge)
   - [Initial Energy Requirement](#initial-energy-requirement)
   - [Charging Methods](#charging-methods)
   - [Stabilization Phase](#stabilization-phase)
   - [Wave Evolution](#wave-evolution)
1. [Wave Propagation Engine](#wave-propagation-engine)
   - [Propagation Mechanics](#propagation-mechanics)
   - [Neighbor Connectivity](#neighbor-connectivity)
   - [PDEs and Wave Equations](#pdes-and-wave-equations)
   - [Huygens Wavelets](#huygens-wavelets)
1. [Key Physics Principles](#key-physics-principles)
   - [Energy Conservation](#energy-conservation)
   - [Amplitude Dilution](#amplitude-dilution)
   - [Boundary Reflections](#boundary-reflections)
1. [Wave Interactions](#wave-interactions)
   - [Interference](#interference)
   - [Reflection](#reflection)
   - [Standing Waves](#standing-waves)
   - [Traveling Waves](#traveling-waves)
   - [Multi-Frequency Superposition](#multi-frequency-superposition)
1. [Computational Implementation](#computational-implementation)
   - [Wave Sources](#wave-sources)
   - [Interference Calculations](#interference-calculations)
   - [Reflection Calculations](#reflection-calculations)
   - [Force Calculation](#force-calculation)
   - [Energy Tracking](#energy-tracking)
1. [Implementation Strategy](#implementation-strategy)

## Overview

The **Wave Engine** is the core computational system that propagates wave disturbances through the field-based medium in LEVEL-1. It handles wave propagation, interference, reflection, and all wave interactions governed by partial differential equations (PDEs).

**Key Principle**: Waves propagate through the grid by transferring amplitude, phase, and energy to neighboring voxels according to wave equations and Huygens' principle.

## Energy Wave Initial Charge

### Initial Energy Requirement

Before wave propagation, reflection, interference, and all subsequent dynamics can occur, **energy must be charged** into the simulation domain. This initial energy establishes the wave field that will then evolve according to the wave equations.

**Energy Equation Constraint**:

- Total energy charged must match the **energy equation value** defined in `equations.py`
- Energy density must be appropriate for the simulation domain volume
- Energy is conserved throughout simulation after initial charge

**Formula**:

```python
# From equations.py
E_total = energy_density * volume
volume = nx * ny * nz * dx³

# Example:
energy_density = <from EWT equations>  # Energy per cubic meter
E_total_required = energy_density * (nx * ny * nz * dx**3)
```

**Critical Requirement**: The amount of energy charged determines:

- Wave amplitude levels
- Force magnitudes (F = -∇A)
- Particle dynamics
- Realistic physics behavior

### Charging Methods

**1. Point Source Charge**:

```python
@ti.kernel
def charge_point_source(pos_x: ti.i32, pos_y: ti.i32, pos_z: ti.i32,
                        energy: ti.f32, frequency: ti.f32):
    """Charge energy at a single point with specific frequency."""
    # Initialize amplitude at source point
    omega = 2.0 * pi * frequency
    amplitude[pos_x, pos_y, pos_z] = compute_amplitude_from_energy(energy)

    # Set initial phase
    phase[pos_x, pos_y, pos_z] = 0.0

    # Set propagation direction (outward from source)
    wave_direction[pos_x, pos_y, pos_z] = ti.Vector([0.0, 0.0, 0.0])  # Will be set by propagation
```

**2. Plane Wave Charge**:

```python
@ti.kernel
def charge_plane_wave(direction: ti.math.vec3, energy: ti.f32, wavelength: ti.f32):
    """Charge uniform plane wave across domain."""
    k = 2.0 * pi / wavelength  # Wave number

    for i, j, k_idx in amplitude:
        pos = get_position(i, j, k_idx)

        # Plane wave: A * sin(k·r)
        phase_val = k * direction.dot(pos)
        amplitude[i, j, k_idx] = A_0 * ti.sin(phase_val)
        wave_direction[i, j, k_idx] = direction.normalized()
```

**3. Spherical Wave Charge**:

```python
@ti.kernel
def charge_spherical_wave(center: ti.math.vec3, energy: ti.f32, wavelength: ti.f32):
    """Charge spherical wave from center point."""
    k = 2.0 * pi / wavelength

    for i, j, k_idx in amplitude:
        pos = get_position(i, j, k_idx)
        r_vec = pos - center
        r = r_vec.norm()

        if r > 0:
            # Spherical wave: (A/r) * sin(kr - ωt)
            amplitude[i, j, k_idx] = (A_0 / r) * ti.sin(k * r)
            wave_direction[i, j, k_idx] = r_vec.normalized()  # Radial
```

**4. Multiple Source Charge**:

```python
# Charge energy from multiple points
for source in source_positions:
    charge_point_source(source.x, source.y, source.z,
                       E_total / num_sources, frequency)
```

### Stabilization Phase

After initial energy Charge, the wave field requires a **stabilization period** before reaching steady-state dynamics.

**Stabilization Process**:

1. **Initial State** (t = 0):
   - Energy concentrated at Charge points/regions
   - Clear propagation path/direction from sources
   - No interference patterns yet

2. **Early Propagation** (t = 0 to t_stab):
   - Waves propagate outward from sources at speed c
   - First reflections from boundaries
   - Initial interference patterns begin to form

3. **Stabilization Time**:

   ```python
   # Time for wave to cross domain and reflect back
   domain_size = max(nx*dx, ny*dx, nz*dx)
   t_stabilization = 2 * domain_size / c  # Round trip time

   # Run simulation without particle updates during stabilization
   for step in range(int(t_stabilization / dt)):
       propagate_wave_field(dt)  # No particle forces yet
   ```

4. **Quasi-Equilibrium** (t > t_stab):
   - Wave energy distributed throughout domain
   - Interference patterns established
   - Ready for particle dynamics

**Why Stabilization is Needed**:

- Prevents artificial transients from affecting particle motion
- Allows interference patterns to form naturally
- Ensures energy distribution reaches quasi-equilibrium
- Provides realistic initial conditions for particle dynamics

### Wave Evolution

**Phase 1: Directed Propagation** (Initial)

- Waves have clear propagation direction from sources
- Wavefronts are coherent (plane, spherical, etc.)
- Energy flows outward from Charging points
- Minimal interference

**Phase 2: Boundary Reflections** (Early Time)

- Waves reach simulation boundaries (walls)
- Hard reflection: phase inversion
- Reflected waves propagate back into domain
- Counter-propagating waves begin to interfere

**Phase 3: Interference Buildup** (Stabilization)

- Incident + reflected waves create interference patterns
- Constructive interference (amplitude increases locally)
- Destructive interference (amplitude cancels locally)
- Standing wave patterns begin to form near boundaries

**Phase 4: Wave Dilution** (Quasi-Steady State)

- Energy spreads throughout volume
- Waves propagate in **all directions** (omni-directional field)
- **"Diluted" waves**: Energy distributed, not concentrated
- Waves continuously:
  - Propagate (traveling component)
  - Reflect (from boundaries and wave centers)
  - Interfere (constructive/destructive)
  - Create standing patterns (nodes/antinodes)

**Steady-State Characteristics**:

```python
# After stabilization, the field exhibits:
# - Waves coming from all directions (not just sources)
# - Complex interference patterns throughout domain
# - Energy density distributed (not localized)
# - Dynamic equilibrium (total energy constant)
```

**Visual Description**:

- **Initial**: Coherent waves radiating from sources (like ripples from stones dropped in water)
- **Intermediate**: Waves bouncing off walls, interfering with each other
- **Steady**: Chaotic-looking field with waves going everywhere, but total energy conserved

**Energy Conservation Check**:

```python
@ti.kernel
def verify_energy_conservation() -> ti.f32:
    """Verify total energy remains constant."""
    total = 0.0
    for i, j, k in amplitude:
        # E = kinetic + potential
        E_k = 0.5 * velocity_field[i,j,k].norm_sqr()
        E_p = 0.5 * amplitude[i,j,k]**2
        total += (E_k + E_p) * dx**3  # Energy per voxel
    return total

# Should match E_total_Charged (within numerical tolerance)
assert abs(verify_energy_conservation() - E_total_Charge) < tolerance
```

## Wave Propagation Engine

### Propagation Mechanics

Each voxel propagates its wave properties to neighboring voxels through a weighted coupling scheme:

**Core Mechanism**:

- Each voxel shares amplitude/energy with neighbors
- Direction vector determines weighted distribution
- Maintains equilibrium by exchanging excess amplitude while receiving from neighbors
- Governed by wave equation PDEs

**Time Evolution**:

```python
# Simplified wave equation update (2nd order in time)
# ∂²ψ/∂t² = c² ∇²ψ

amplitude_new[i,j,k] = (
    2 * amplitude[i,j,k]
    - amplitude_old[i,j,k]
    + (c * dt / dx)² * laplacian[i,j,k]
)
```

### Neighbor Connectivity

Voxel `[i,j,k]` couples to neighbors based on distance and connectivity mode.

**For detailed neighbor classification and weighting**, see [`01_WAVE_FIELD.md` - Voxel Neighbor Connectivity](./01_WAVE_FIELD.md#voxel-neighbor-connectivity)

**Summary**:

- **6-connectivity**: Face neighbors only (distance = dx)
- **18-connectivity**: Face + edge neighbors
- **26-connectivity**: All neighbors (maximum accuracy)

**Distance-Based Coupling Weights**:

- Face: `w = 1.0`
- Edge: `w ≈ 0.707` (1/√2)
- Corner: `w ≈ 0.577` (1/√3)

### PDEs and Wave Equations

The wave engine solves partial differential equations that govern wave behavior:

**Classical Wave Equation** (scalar field):

```test
∂²ψ/∂t² = c² ∇²ψ
```

Where:

- `ψ` = wave amplitude field
- `c` = wave propagation speed
- `∇²` = Laplacian operator (spatial derivatives)

**Laplacian in 3D** (6-connectivity):

```python
laplacian[i,j,k] = (
    amplitude[i+1,j,k] + amplitude[i-1,j,k] +
    amplitude[i,j+1,k] + amplitude[i,j-1,k] +
    amplitude[i,j,k+1] + amplitude[i,j,k-1] -
    6.0 * amplitude[i,j,k]
) / (dx * dx)
```

**Extended for Vector Fields** (for transverse waves):

- Separate equations for each component of amplitude direction
- Coupling between components for polarization effects

### Huygens Wavelets

Huygens' principle: Each point on a wavefront acts as a source of secondary wavelets.

**Implementation**:

- Each voxel with non-zero amplitude emits wavelets to neighbors
- Wavelets propagate spherically from each voxel
- Superposition of all wavelets determines new wavefront
- Direction vector determines wavelet weighting to neighbors

**Wavelet Contribution**:

```python
# From voxel [i,j,k] to neighbor [i+di, j+dj, k+dk]
distance = sqrt(di² + dj² + dk²) * dx
weight = 1.0 / distance  # Inverse distance weighting
wavelet_contribution = amplitude[i,j,k] * weight * directional_factor
```

**Directional Factor**:

- Determined by propagation direction vector
- Anisotropic propagation for non-isotropic waves
- For isotropic waves: uniform in all directions

## Key Physics Principles

### Energy Conservation

**Fundamental Constraint**: Total energy in the system remains constant.

```text
E_total = Σ (E_kinetic[i,j,k] + E_potential[i,j,k])
```

**Implementation Requirements**:

- Energy Charged once at initialization
- No energy creation or destruction during propagation
- Energy only redistributes through wave motion
- Numerical scheme must preserve energy (symplectic integrator preferred)

**Verification**:

```python
@ti.kernel
def compute_total_energy() -> ti.f32:
    total = 0.0
    for i, j, k in amplitude:
        # Kinetic energy ∝ (∂ψ/∂t)²
        E_k = 0.5 * velocity_field[i,j,k].norm_sqr()
        # Potential energy ∝ amplitude²
        E_p = 0.5 * amplitude[i,j,k]**2
        total += E_k + E_p
    return total
```

### Amplitude Dilution

**Geometric Dilution**: Amplitude decreases with distance from source due to energy spreading.

**1/r Law** (spherical waves):

- Amplitude ∝ 1/r (inverse distance)
- Energy density ∝ 1/r² (inverse square)
- Total energy constant: `E = ∫ (energy_density) dV = constant`

**Implementation**:

- Natural consequence of wave equation propagation
- No explicit amplitude reduction needed
- Energy spreads over larger surface as wave expands
- Amplitude reduces, but total energy conserved

### Boundary Reflections

**Reflection at Boundaries**: Waves reflect without energy loss.

**Boundary Types**:

1. **Hard boundaries** (universe walls):
   - Perfect reflection: `ψ_reflected = -ψ_incident`
   - Phase inversion for fixed boundaries
   - No phase inversion for free boundaries

2. **Wave centers** (particles):
   - Reflect waves like boundaries
   - Invert wavelet propagation direction
   - Source of emergent forces (MAP)

**Implementation**:

```python
# At boundary (e.g., i=0 face)
if i == 0:
    amplitude[i,j,k] = -amplitude[i+1,j,k]  # Hard reflection
```

## Wave Interactions

### Interference

**Superposition Principle**: Multiple waves combine linearly at each point.

**Types**:

- **Constructive**: Waves in phase → amplitude increases
  - `Δφ = 0, 2π, 4π, ...`
  - `A_total = A₁ + A₂`

- **Destructive**: Waves out of phase → amplitude decreases
  - `Δφ = π, 3π, 5π, ...`
  - `A_total = A₁ - A₂` (can cancel completely)

**Implementation**:

```python
# Multiple waves naturally interfere by summing contributions
for source in wave_sources:
    amplitude[i,j,k] += source.contribution(i, j, k, t)
```

### Reflection

**Wave Reflection** from boundaries and wave centers:

**At Universe Boundaries**:

- Waves reflect back into domain
- Energy conserved
- Creates standing wave patterns near walls

**At Wave Centers** (particles):

- Particles act as reflectors
- Inverts wave propagation direction
- Creates near-field wave patterns around particles
- Source of particle forces (MAP)

### Standing Waves

**Standing Waves**: Form from interference of counter-propagating waves.

**Characteristics**:

- Nodes: Points of zero amplitude (destructive interference)
- Antinodes: Points of maximum amplitude (constructive interference)
- Fixed spatial pattern, oscillates in time
- Forms around wave centers (particles)

**Condition**:

- Two waves with same frequency traveling in opposite directions
- Typical near reflecting boundaries or between wave centers

**Pattern**:

```text
ψ(x,t) = A sin(kx) cos(ωt)
```

Nodes at: `x = nλ/2` (n = 0, 1, 2, ...)

### Traveling Waves

**Traveling Waves**: Propagate through medium with constant speed.

**Characteristics**:

- Move through space: `ψ(x,t) = A sin(kx - ωt)`
- Carry energy from source to distant regions
- Wavelength λ and frequency f related by: `c = fλ`

**Implementation**:

- Natural result of wave equation propagation
- Source Charges energy with specific frequency
- Wave propagates outward at speed c

### Multi-Frequency Superposition

**Multiple Frequencies**: Different wave sources can Charge different frequencies.

**Behavior**:

- Each frequency propagates independently
- Frequencies combine at each point: `ψ_total = Σ ψᵢ`
- Can store frequency per-voxel if needed (optional)
- Creates complex interference patterns (beats, harmonics)

**Beat Frequency**:

When two close frequencies interfere: `f_beat = |f₁ - f₂|`

## Computational Implementation

### Wave Sources

**Energy Charging Points**:

- Initialize amplitude at specific voxels
- Set initial phase and frequency
- Can be continuous or pulsed sources
- Multiple sources create complex wave patterns

**Source Types**:

1. **Point source**: Single voxel, spherical waves
2. **Plane wave source**: Line/plane of voxels, uniform propagation
3. **Pulsed source**: Time-limited energy Charge
4. **Continuous source**: Ongoing oscillation (for testing)

**Implementation**:

```python
@ti.kernel
def charge_wave_source(x: ti.i32, y: ti.i32, z: ti.i32,
                       freq: ti.f32, phase: ti.f32, t: ti.f32):
    """Charge sinusoidal wave at source location."""
    omega = 2.0 * pi * freq
    amplitude[x, y, z] += A_source * ti.sin(omega * t + phase)
```

### Interference Calculations

**Superposition from Multiple Directions**:

- Each voxel receives contributions from all neighbors
- Sum all incoming wavelets
- Natural consequence of field-based approach

**Algorithm**:

```python
@ti.kernel
def compute_interference():
    for i, j, k in amplitude:
        total_contribution = 0.0
        # Sum contributions from all neighbors
        for neighbor in get_neighbors(i, j, k):
            total_contribution += compute_wavelet(neighbor, i, j, k)
        # Update based on superposition
        amplitude_new[i,j,k] = total_contribution
```

### Reflection Calculations

**Boundary Reflection**:

- Detect boundary voxels
- Apply reflection condition (phase inversion)
- Propagate reflected wave back into domain

**Wave Center Reflection**:

- Detect voxels near particles
- Invert propagation direction
- Creates spherical reflected waves around particles

**Implementation**:

```python
@ti.func
def apply_boundary_reflection(i: ti.i32, j: ti.i32, k: ti.i32):
    """Apply hard boundary reflection."""
    if i == 0:  # Left boundary
        amplitude[i,j,k] = -amplitude[i+1,j,k]
    if i == nx-1:  # Right boundary
        amplitude[i,j,k] = -amplitude[i-1,j,k]
    # Similar for other boundaries
```

### Force Calculation

**Force from Amplitude Gradient**: Particles move toward minimum amplitude (MAP).

```python
F = -∇A
```

**Computation**:

```python
@ti.kernel
def compute_force_field():
    for i, j, k in ti.ndrange((1, nx-1), (1, ny-1), (1, nz-1)):
        # Compute gradient using finite differences
        grad_x = (amplitude[i+1,j,k] - amplitude[i-1,j,k]) / (2*dx)
        grad_y = (amplitude[i,j+1,k] - amplitude[i,j-1,k]) / (2*dx)
        grad_z = (amplitude[i,j,k+1] - amplitude[i,j,k-1]) / (2*dx)

        # Force = -gradient (toward minimum amplitude)
        force[i,j,k] = -ti.Vector([grad_x, grad_y, grad_z])
```

**Usage**: Force field drives particle motion (see [`05_MATTER.md`](./05_MATTER.md))

### Energy Tracking

**Monitor Energy Conservation**:

- Compute total energy each timestep
- Verify conservation: `|E(t) - E(0)| < tolerance`
- Track energy distribution (kinetic vs potential)

**Diagnostics**:

```python
@ti.kernel
def track_energy() -> ti.f32:
    kinetic = 0.0
    potential = 0.0
    for i, j, k in amplitude:
        kinetic += 0.5 * velocity_field[i,j,k].norm_sqr()
        potential += 0.5 * amplitude[i,j,k]**2
    total = kinetic + potential
    return total
```

## Implementation Strategy

### Recommended Approach

1. **Start Simple**: Implement 1D wave equation first
2. **Extend to 3D**: Add spatial dimensions incrementally
3. **Add Features**: Interference → Reflection → Sources → Particles
4. **Optimize**: Profile and optimize critical kernels

### Numerical Schemes

**Time Integration**:

- **Leapfrog method**: Energy-conserving, 2nd order accuracy
- **Verlet integration**: Symplectic, preserves phase space
- **RK4** (Runge-Kutta): Higher accuracy, more expensive

**Spatial Discretization**:

- Finite difference for derivatives
- Configurable stencil (6/18/26 neighbors)

### Stability Criteria

**CFL Condition** (Courant-Friedrichs-Lewy):

```text
dt < dx / c
```

Ensures numerical stability for explicit time integration.

**Recommended**:

```python
dt = 0.5 * dx / c  # Safety factor of 0.5
```

---

**Status**: Wave engine architecture defined

**Next Steps**: Implement wave propagation kernel and test with simple point source

**Related Documentation**:

- [`01_WAVE_FIELD.md`](./01_WAVE_FIELD.md) - Grid architecture
- [`02_WAVE_PROPERTIES.md`](./02_WAVE_PROPERTIES.md) - Wave properties
- [`05_MATTER.md`](./05_MATTER.md) - Particle system and MAP
