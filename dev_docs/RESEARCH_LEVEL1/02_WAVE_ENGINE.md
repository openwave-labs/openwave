# WAVE ENGINE

## Table of Contents

1. [Overview](#overview)
1. [Wave Propagation Mechanics](#wave-propagation-mechanics)
1. [Energy Evolution Sequence: From Pulse to Particle](#energy-evolution-sequence-from-pulse-to-particle)
   - [Phase 1: Center-Concentrated Pulse Injection](#phase-1-center-concentrated-pulse-injection)
   - [Phase 2: Outward Propagation via Wave Equation](#phase-2-outward-propagation-via-wave-equation)
   - [Phase 3: Boundary Reflections](#phase-3-boundary-reflections)
   - [Phase 4: Energy Dilution into Stable Distributed State](#phase-4-energy-dilution-into-stable-distributed-state)
   - [Phase 5: Wave Center Insertion (Reflective Voxels)](#phase-5-wave-center-insertion-reflective-voxels)
   - [Phase 6: Standing Wave Emergence (IN + OUT Interference)](#phase-6-standing-wave-emergence-in--out-interference)
   - [Phase 7: Particle Formation (Mass = Trapped Energy)](#phase-7-particle-formation-mass--trapped-energy)
   - [Visual Summary](#visual-summary)
   - [Why This Sequence?](#why-this-sequence)
1. [Wave Direction Computation](#wave-direction-computation)
1. [Energy and Momentum Conservation](#energy-and-momentum-conservation)
1. [Wave Mode: Longitudinal vs Transverse](#wave-mode-longitudinal-vs-transverse)
1. [Wave Decomposition: Separating Longitudinal and Transverse Components](#wave-decomposition-separating-longitudinal-and-transverse-components)
1. [Wave Type: Standing vs Traveling](#wave-type-standing-vs-traveling)
   - [Fundamental Concept](#fundamental-concept)
   - [Wave Patterns](#wave-patterns)
   - [Mathematical Definition](#mathematical-definition)
   - [Better Method - Energy Flux Analysis](#better-method---energy-flux-analysis)
   - [Implementation](#implementation)
   - [Alternative Method - Node Motion Detection](#alternative-method---node-motion-detection)
   - [Physical Interpretation](#physical-interpretation)
   - [Storage](#storage)
   - [Combined Analysis](#combined-analysis)
   - [Summary Table](#summary-table)
1. [Wavelength and Frequency Variation in the Medium](#wavelength-and-frequency-variation-in-the-medium)
1. [The Complete Picture](#the-complete-picture)
1. [Implementation Summary](#implementation-summary)

## Overview

The **Wave Engine** is the core computational system that propagates wave disturbances through the wave-field medium in LEVEL-1. It handles wave propagation, interference, reflection, and all wave interactions governed by partial differential equations (PDEs).

**Key Principle**: Waves propagate through the grid by transferring amplitude, phase, and energy to neighboring voxels according to wave equations and Huygens' principle.

## Wave Propagation Mechanics

**Context**: properties propagate through the field.

**System Overview**:

1. **Wave Field**: The medium (grid of voxels)
2. **Energy Charge**: Initial energy injected into system
3. **Wave Propagation**: Wave properties propagate in wave-like motion:
   - Amplitude, displacement
   - Wave direction, speed
   - Energy, phase
   - Frequency, Wavelength
4. **Propagation Methods**:
   - PDEs (Partial Differential Equations)
   - Wave equation
   - Laplacian operator
   - Huygens wavelets
5. **Conservation**: propagation conserves/transfers energy and momentum

**Wave Interactions**:

- Waves interfere with each other (constructive/destructive)
- Waves reflect from:
  - Universe boundaries (grid boundaries)
  - Wave centers (fundamental particles)
- Reflection creates standing waves (interference of inward/outward waves)
- Standing waves give particles mass (trapped energy)

## Energy Evolution Sequence: From Pulse to Particle

The complete energy evolution follows seven distinct phases:

### Phase 1: Center-Concentrated Pulse Injection

- Energy concentrated at universe center
- Single pulse (or a few pulses for precision)
- Total energy exactly matches `equations.compute_energy_wave_equation(volume)`
- Uses proper wave characteristics (f, A, λ from constants)
- Implementation: **Option 2 (Spherical Gaussian)** recommended

### Phase 2: Outward Propagation via Wave Equation

- Wave equation ∂²ψ/∂t² = c²∇²ψ governs propagation
- Energy travels outward at wave speed c
- Spherical wave fronts expand from center
- No manual 1/r falloff needed - emerges naturally from Laplacian operator
- Energy automatically conserved by wave equation physics

### Phase 3: Boundary Reflections

- Waves reach universe boundaries (walls)
- Boundary conditions enforce ψ = 0 at walls
- Waves reflect back into domain
- Reflections create interference patterns
- Total energy remains constant (no absorption at boundaries)

### Phase 4: Energy Dilution into Stable Distributed State

- Multiple reflections distribute energy throughout field
- After sufficient time, energy reaches quasi-equilibrium
- Energy density becomes relatively uniform across universe
- Small fluctuations remain (natural wave motion)
- System ready for wave center insertion

### Phase 5: Wave Center Insertion (Reflective Voxels)

- Insert reflective voxels at specific positions
- Wave centers: ψ = 0 always (never changes)
- Function like internal boundary walls
- Disturb neighboring voxels to invert wave direction
- Create local reflection sites within the field

### Phase 6: Standing Wave Emergence (IN + OUT Interference)

- Reflected waves (OUT) interfere with incoming waves (IN)
- Constructive/destructive interference creates nodes and antinodes
- Pattern: Φ = Φ₀ e^(iωt) sin(kr)/r (Wolff's solution)
- Steady-state standing wave forms around wave center
- Wave center = particle core

### Phase 7: Particle Formation (Mass = Trapped Energy)

- Standing wave boundary defines particle extent
- Energy trapped within standing wave pattern
- Steady-state energy density inside boundary
- **Particle mass = total energy in standing wave region**
- Fundamental particle successfully formed

## Visual Summary

```text
Time 0:      [    •    ]  ← Concentrated pulse at center

Time 1:     [   ◯◯◯   ]  ← Expanding wave front

Time 2:    [  ◯     ◯  ]  ← Reached boundaries, reflecting

Time 3:    [ ░░░░░░░░░ ]  ← Energy distributed (diluted state)

Time 4:    [ ░ ⊕ ░ ⊕ ░ ]  ← Wave centers (⊕) inserted

Time 5:    [ ░ ◉ ░ ◉ ░ ]  ← Standing waves (◉) formed = particles!
```

### Why This Sequence?

1. **Energy Correctness**: Start with exact EWT energy amount
2. **Natural Evolution**: Let wave equation physics handle distribution
3. **Stable Foundation**: Distributed energy provides stable background field
4. **Clean Particle Formation**: Wave centers create particles in equilibrium field
5. **Physical Realism**: Mimics natural wave behavior and particle emergence

## Wave Direction Computation

### The Challenge

The wave equation `∂²ψ/∂t² = c²∇²ψ` only evolves the **scalar amplitude** `ψ`. It doesn't directly compute wave direction.

**But we need direction for**:

- Momentum transfer: `p = ρψ × direction`
- Force calculations: Direction of energy flow
- Particle-wave interactions: Reflection angle
- Visualization: Wave propagation arrows

### Solution 1: Compute Direction from Phase Gradient

### Wave Direction = Gradient of Phase

In wave physics, the wave propagation direction is the gradient of the phase field:

```text
wave_direction = ∇φ / |∇φ|
```

Where `φ` is the wave phase at each voxel.

### Computing Phase from Amplitude

For a traveling wave:

```text
ψ(x,t) = A(x) cos(kx - ωt + φ₀)
```

The phase at position x and time t is:

```text
φ(x,t) = arctan(ψ_imaginary / ψ_real)
```

But for a real-valued field (which we have), we need the **analytic signal** approach.

### Solution 2 (better approach): Energy Flux Method

**Better approach**: Compute wave direction from **energy flux** (Poynting-like vector for waves).

**Energy flux density** (energy flow direction):

```text
S = -c² × ψ × ∇ψ
```

Where:

- `ψ` = current displacement
- `∇ψ` = spatial gradient of displacement
- Direction of S = direction of energy flow = wave direction

### Solution 3 (alternative): Velocity-Based Direction

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
            v_wave = (self.displacement_am[i,j,k] - self.displacement_old_am[i,j,k]) / dt

            # Gradient of velocity gives acceleration direction
            # (This is less direct, energy flux method is better)
            # ...
```

**This method is less reliable** - use energy flux method instead.

### Storage and Update Frequency

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

### Summary: Wave Direction Pipeline

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

## Energy and Momentum Conservation

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
        v = (self.displacement_am[i,j,k] - self.displacement_old_am[i,j,k]) / dt

        # Kinetic energy density
        E_k = 0.5 * ρ * v**2

        # Potential energy density
        E_p = 0.5 * ρ * (self.frequency * self.displacement_am[i,j,k])**2

        # Add to total
        total_energy += (E_k + E_p) * dx**3

    return total_energy
```

**Momentum Density**:

```python
# Momentum carried by wave (vector field)
p[i,j,k] = ρ × ψ[i,j,k] × wave_direction[i,j,k]
```

Where `wave_direction` is determined by the energy flux.

## Wave Mode: Longitudinal vs Transverse

**Fundamental Concept**:

Wave mode is determined by the relationship between **medium displacement direction** and **wave propagation direction**:

- **Longitudinal wave**: Displacement parallel to propagation (compression wave)
- **Transverse wave**: Displacement perpendicular to propagation (shear wave)
- **Mixed mode**: Both components present

**Mathematical Definition**:

```text
Wave propagation direction: k̂ = S / |S|  (from energy flux, see below)
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

## Wave Decomposition: Separating Longitudinal and Transverse Components

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

## Wave Type: Standing vs Traveling

### Fundamental Concept

Wave type is determined by whether the wave **moves through space**:

- **Traveling wave**: Energy moves, nodes move (wave velocity ≠ 0)
- **Standing wave**: Energy stationary, nodes fixed (wave velocity = 0)
- **Quasi-standing**: Slow-moving pattern (wave velocity ≈ 0)

### Wave Patterns

**Standing Wave Pattern**:

```text
ψ(x,t) = A sin(kx) cos(ωt)

where:
- Spatial pattern sin(kx) is fixed in space
- Temporal oscillation cos(ωt) varies in time
- Nodes at: x = nλ/2 (n = 0, 1, 2, ...)
- Antinodes at: x = (n + 1/2)λ/2
```

**Characteristics**:

- Nodes: Points of zero amplitude (destructive interference)
- Antinodes: Points of maximum amplitude (constructive interference)
- Fixed spatial pattern, oscillates in time
- Forms around wave centers (particles) and between reflecting boundaries
- Requires two waves with same frequency traveling in opposite directions

**Traveling Wave Pattern**:

```text
ψ(x,t) = A sin(kx - ωt)

where:
- Pattern moves through space at velocity v = ω/k = c
- Both spatial and temporal variations coupled
- Wavelength λ and frequency f related by: c = fλ
```

**Characteristics**:

- Moves through space at constant speed c
- Carries energy from source to distant regions
- Natural result of wave equation propagation
- No fixed nodes (nodes move with wave)

### Mathematical Definition

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

### Better Method - Energy Flux Analysis

```text
Traveling wave: Net energy flux (S ≠ 0 consistently)
Standing wave: No net energy flux (S ≈ 0 on average, oscillates locally)

Time-averaged energy flux:
<S> = (1/T) ∫ S dt over period T

- <S> ≈ 0: Standing wave (no net energy transport)
- <S> > threshold: Traveling wave (energy transport)
```

### Implementation

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
    ρ = ti.f32(constants.MEDIUM_DENSITY)
    f = self.frequency
    
    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Current displacement
            psi = self.displacement_am[i,j,k] * constants.ATTOMETER  # meters

            # Velocity (time derivative approximation)
            v_wave_am = (self.displacement_am[i,j,k] - self.displacement_old_am[i,j,k]) / dt
            v_wave = v_wave_am * constants.ATTOMETER  # m/s

            # Kinetic energy density
            E_k = 0.5 * ρ * v_wave**2

            # Potential energy density
            E_p = 0.5 * ρ * (f * psi)**2

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

### Alternative Method - Node Motion Detection

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
            psi_old = self.displacement_old_am[i,j,k]

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

### Physical Interpretation

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

### Storage

```python
# In WaveField class __init__
self.wave_type = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Range [0,1]
```

### Combined Analysis

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

### Summary Table

| Property | Longitudinal | Transverse | Standing | Traveling |
|----------|--------------|------------|----------|-----------|
| **Definition** | Displacement ∥ propagation | Displacement ⊥ propagation | Nodes fixed | Nodes moving |
| **Measurement** | k̂ · û ≈ 1 | k̂ · û ≈ 0 | E_k ≠ E_p (alternating) | E_k = E_p (in phase) |
| **Energy flux** | Along k̂ direction | Perpendicular components | S = 0 | S ≠ 0 |
| **EWT example** | Gravity waves | EM waves (from electron) | Particle interior | Propagating radiation |
| **Field storage** | `wave_mode[i,j,k]` | `wave_mode[i,j,k]` | `wave_type[i,j,k]` | `wave_type[i,j,k]` |

## Wavelength and Frequency Variation in the Medium

### Fundamental Relationship

```text
c = λ × f

where:
c = wave speed (constant in uniform medium)
λ = wavelength (can vary locally)
f = frequency (can vary locally)
```

### Key Insight for EWT

In Energy Wave Theory, the medium (spacetime fabric) has **constant properties everywhere**:

- **Wave speed c**: Always 2.998×10⁸ m/s (speed of light)
- **Medium density ρ**: Always 3.860×10²² kg/m³

However, **wavelength λ and frequency f can vary** due to:

1. **Different energy sources** (particles with different energies)
2. **Wave interactions** (constructive/destructive interference)
3. **Doppler effects** (moving sources)
4. **Energy transformations** (electron converting to EM waves)

### Dispersion in EWT Medium

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

### Measuring Local Frequency - Direct Temporal Method

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
self.last_peak_time = ti.field(dtype=ti.f32, shape=(nx, ny, nz))    # Time of last peak (seconds)
self.frequency_local = ti.field(dtype=ti.f32, shape=(nx, ny, nz))   # f (Hz) - PRIMARY measured property
self.period = ti.field(dtype=ti.f32, shape=(nx, ny, nz))            # T (seconds) - derived from f
self.wavelength_local = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # λ (attometers) - derived from f

@ti.kernel
def measure_frequency(self, current_time: ti.f32):
    """
    Measure frequency by timing peaks (when |ψ| = A).

    Measurement hierarchy:
    1. Measure dt between peaks (timing measurement, in seconds)
    2. Compute f = 1/dt (PRIMARY property)
    3. Derive T = dt (period, same as measured dt)
    4. Derive λ = c_slowed/f (wavelength from frequency, using slowed wave speed)

    Note: Frequency-centric approach - f is measured, λ is computed.
    With SLO_MO: Wave frequencies are slowed by SLO_MO factor.
    """
    c_slowed = ti.f32(constants.EWAVE_SPEED / config.SLO_MO)  # m/s, slowed wave speed

    for i, j, k in self.displacement_am:
        # Check if displacement is at a peak (|ψ| ≈ A)
        disp_mag = ti.abs(self.displacement_am[i,j,k])
        amp = self.amplitude_am[i,j,k]

        # Peak detection: current displacement within 1% of amplitude
        if amp > 1e-12 and disp_mag >= amp * 0.99:
            # This is a peak!

            if self.last_peak_time[i,j,k] > 0:  # Not the first peak
                # Measure time between peaks (dt, in seconds)
                dt = current_time - self.last_peak_time[i,j,k]

                if dt > 0:
                    # PRIMARY: Compute frequency f = 1/dt
                    self.frequency_local[i,j,k] = 1.0 / dt  # Hz

                    # DERIVED: Store period (same as dt, different name)
                    self.period[i,j,k] = dt  # seconds

                    # DERIVED: Compute wavelength λ = c_slowed/f
                    f_Hz = self.frequency_local[i,j,k]
                    wavelength_m = c_slowed / f_Hz  # meters (using slowed wave speed)
                    self.wavelength_local[i,j,k] = wavelength_m / constants.ATTOMETER  # attometers

            # Update last peak time for next measurement
            self.last_peak_time[i,j,k] = current_time

# Usage in main simulation loop:
def update_timestep(self, dt: ti.f32):
    """Complete wave field update for one timestep."""
    self.current_time += dt  # Accumulate time in seconds

    # 1. Propagate wave displacement
    self.propagate_wave_field(dt)

    # 2. Track amplitude envelope
    self.track_amplitude_envelope()

    # 3. Measure frequency from peaks (f = 1/dt, then derive T and λ)
    self.measure_frequency(self.current_time)

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
period[i,j,k] = time between peaks when |ψ| = amplitude_am[i,j,k]  # seconds
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

---

### Multi-Frequency Superposition Challenge

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

### Solutions (Implementation Strategies)

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

### Recommendation for Initial Implementation

Start with **monochromatic approximation** (uniform f, single source). Once wave propagation is working, add **dominant frequency tracking** with the temporal peak method documented above. This captures the most important physics (beating frequencies, Doppler shifts) without the complexity of full Fourier decomposition.

### Wavelength Propagation and Changes

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

### Frequency Changes and Energy

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

### How Frequency Changes Occur

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

### Implementing Frequency/Wavelength Tracking

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

### Superposition of Different Wavelengths

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

### Fourier Decomposition** (advanced)

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

### Practical Approach for LEVEL-1

For initial implementation, **assume single wavelength** (monochromatic):

```python
# Initialize with single wavelength
wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETER

# All voxels start with this wavelength
# Wave equation preserves wavelength for monochromatic source
```

### Future Enhancement for multiple wavelengths

1. **Track wave packets** with different λ separately
2. Use **frequency tagging** to follow each component
3. Implement **Fourier analysis** for decomposition

### Summary Table (Frequency-Centric)

| Property | Constant? | How Determined? | Propagates? |
|----------|-----------|-----------------|-------------|
| **ρ** (medium density) | ✓ Yes (uniform medium) | From EWT constants | N/A (property of medium) |
| **c** (wave speed) | ✓ Yes (constant everywhere) | From medium properties | N/A (property of medium) |
| **ψ** (displacement) | ✗ No (varies spatially/temporally) | Wave equation evolution | ✓ Yes (via PDE) |
| **A** (amplitude) | ✗ No (varies spatially/temporally) | **Tracked max**(\|ψ\|) | ✓ Yes (via envelop) |
| **f** (frequency) | ✗ No (varies spatially) | **Fundamental: Measured** f = 1/dt | ✗ No (derived property) |
| **T** (period) | ✗ No (varies spatially) | **Derived** T = dt = 1/f | ✗ No (derived property) |
| **λ** (wavelength) | ✗ No (varies spatially) | **Derived** λ = c/f | ✗ No (derived property) |
| **E** (energy) | ✗ No (varies spatially) | E = ρV(fA)² | ✓ Yes (via wave energy) |

### Key Takeaways (Frequency-Centric Philosophy)

1. **c is constant** throughout the EWT medium (non-dispersive)
2. **dt is measured** directly by timing oscillations (when \|ψ\| reaches A)
3. **f = 1/dt is PRIMARY** - frequency computed first (human-intuitive, Planck E=hf)
4. **T = dt and λ = c/f are DERIVED** - period and wavelength computed from frequency
5. **Simple and robust**: Direct temporal measurement, not sensitive to spatial gradients
6. **Reuses amplitude tracking**: Same infrastructure as LEVEL-0's `amplitude_am` approach
7. For **single source**, f is constant; for **multiple sources**, measures dominant frequency
8. **Spacetime coupling**: f (temporal) × A (spatial) = natural pairing in E = ρV(fA)²

## The Complete Picture

### Wave Field → Forces → Motion

```text
┌─────────────────────────────────────────────────────────┐
│ 1. WAVE FIELD (grid of voxels)                          │
│    ├─ Initial energy charge (point/plane/spherical)     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. WAVE PROPAGATION (PDE evolution)                     │
│    ├─ Wave equation: ∂²ψ/∂t² = c²∇²ψ                    │
│    ├─ Laplacian couples neighboring voxels              │
│    ├─ Reflection: boundaries + wave centers             │
│    ├─ Interference: constructive/destructive            │
│    ├─ Direction from energy flux: S = -c²ψ∇ψ            │
│    ├─ Stabilization period (waves propagate/reflect)    │
│    └─ Quasi-steady state (omni-directional field)       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. PARTICLE-FIELD INTERACTION (feedback loop)           │
│    ├─ Particles act as wave reflectors                  │
│    ├─ Standing waves form around particles              │
│    ├─ Trapped energy = particle mass: m = E/c²          │
│    ├─ Standing wave radius: r = n×λ/2                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. FORCE GENERATION (amplitude gradients)               │
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
│ 5. PARTICLE MOTION (Newton's laws)                      │
│    ├─ Interpolate force at particle position            │
│    ├─ Acceleration: a = F/m                             │
│    ├─ Update velocity: v_new = v_old + a×dt             │
│    ├─ Update position: x_new = x_old + v×dt             │
│    └─ Particles move toward amplitude minimum (MAP)     │
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

    # 2. Update particles (see 03_FUNDAMENTAL_PARTICLE.md and 05_MATTER.md)
    particles.update_positions(wave_field.force, dt)

    # 3. Apply particle reflections to field
    particles.apply_reflections_to_field(wave_field)
```

---

**Status**: Comprehensive framework defined with force calculation in Newtons and wave propagation via PDE

**Next Steps**: Implement and validate wave equation solver with energy conservation checks

**Related Documentation**:

- [`01a_WAVE_FIELD_grid.md`](./01a_WAVE_FIELD_grid.md) - Field architecture and indexing
- [`01b_WAVE_FIELD_properties.md`](./01b_WAVE_FIELD_properties.md) - WaveField class definition
- [`02a_WAVE_ENGINE_charge.md`](./02a_WAVE_ENGINE_charge.md) - Energy charging methods
- [`02b_WAVE_ENGINE_propagate.md`](./02b_WAVE_ENGINE_propagate.md) - Wave propagation details
- [`02c_WAVE_ENGINE_interact.md`](./02c_WAVE_ENGINE_interact.md) - Wave interactions and boundaries
- [`03_FUNDAMENTAL_PARTICLE.md`](./03_FUNDAMENTAL_PARTICLE.md) - Wave centers and particles
- [`04_FORCE_MOTION.md`](./04_FORCE_MOTION.md) - Force types and motion equations
- [`05_MATTER.md`](./05_MATTER.md) - Composite particles and interactions
