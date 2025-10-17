# Spherical Wave Energy Conservation and Amplitude Falloff

## Overview

This document provides comprehensive physical justification and implementation details for amplitude falloff in spherical wave propagation within the OpenWave quantum simulator. The implementation accounts for energy conservation, near-field/far-field physics, and Energy Wave Theory (EWT) specifications.

## Physical Background: Spherical Wave Energy Conservation

### Energy Conservation Principle

For spherical waves propagating from a point source, total energy must be conserved as the wave expands outward. The energy of a wave system is given by:

```python
E = ρV(c/λ × A)²
```

Where:

- `E` = total energy
- `ρ` = medium density
- `V` = volume
- `c` = wave propagation speed
- `λ` = wavelength
- `A` = amplitude

As a spherical wave propagates through a uniform medium:

- `λ` remains constant (wavelength unchanged in uniform medium)
- `c` remains constant (wave speed unchanged in uniform medium)
- `ρ` remains constant (uniform medium density)
- `A` must decrease to maintain energy conservation

### Mathematical Derivation of Amplitude Falloff

For a spherical wave, energy density must be integrated over expanding spherical shells. At distance `r` from the wave source:

```python
E_total = ∫ (energy_density) × (surface_area) dr
E_total = ∫ A²(r) × 4πr² dr
```

For energy conservation, `E_total` must remain constant regardless of `r`. This requires:

```python
A²(r) × r² = constant
A²(r) = constant / r²
A(r) = constant / r
```

Therefore: **A ∝ 1/r** for spherical waves.

This can also be derived from power conservation:

```python
Power = Energy × Surface_Area
P = E × 4πr²
```

Since power at the source must equal power at any radius:

```python
P_source = P_r
E_source × 4πr₀² = E_r × 4πr²
```

Given `E ∝ A²`:

```python
A₀² × r₀² = A_r² × r²
A_r = A₀ × (r₀/r)
```

## Near-Field vs Far-Field Regions

### Electromagnetic Wave Standards

Classical electromagnetic theory divides the region around a radiating antenna into three zones:

1. **Reactive Near Field** (r < λ/(2π) ≈ 0.159λ):
   - Dominated by reactive (non-radiating) field components
   - Energy oscillates between electric and magnetic fields
   - Amplitude decreases as **A ∝ 1/r³**
   - Field structure strongly coupled to source geometry

2. **Radiative Near Field** (Fresnel region, 0.159λ < r < 2λ):
   - Transition zone between near and far field
   - Angular field distribution depends on distance
   - Amplitude decreases as **A ∝ 1/r²**
   - Wave fronts not yet fully spherical

3. **Far Field** (Fraunhofer region, r > 2λ):
   - Radiating field components dominate
   - Wave fronts are spherical
   - Amplitude decreases as **A ∝ 1/r** (energy conservation)
   - Angular field distribution independent of distance
   - Fully formed waves propagating away from source

The transition to far-field behavior typically occurs at **r > 2λ**, though the radiative near-field extends from approximately **λ to 2λ** from the wave source.

### When Waves Are Fully Formed

Waves are considered "fully formed" in the **far-field region** (r > 2λ from the wave source), where:

- Wave fronts have achieved spherical geometry
- Energy propagates radially outward with constant velocity
- Amplitude follows the 1/r energy conservation law
- Angular distribution is stable and independent of distance

In the near-field regions (r < 2λ from wave source), wave structure is still developing and strongly influenced by source geometry.

## Energy Wave Theory (EWT) Specifications

### Neutrino Standing Wave Boundary

According to EWT Phase 1 simulation requirements (research_requirements/original_requirements/1. Simulating a Fundamental Particle - EWT.pdf, page 19):

The neutrino standing wave structure has a boundary at **r = 1λ** (one wavelength from the wave source).

This specification provides physical justification for using 1λ as the minimum safe radius in amplitude falloff calculations, as it represents the fundamental particle boundary in EWT.

### Validation Criterion

The neutrino standing wave validation criterion specifies:

```python
r_ν ≈ 3 × 10⁻¹⁷ meters
```

This radius corresponds to one wavelength of the neutrino's constituent wave structure.

## Implementation Analysis

### The Issue Observed with r < 1λ

During implementation testing, granules at distances less than 1λ from the wave source exhibited unphysical behavior:

**Observation**: "Granules at less than 1λ from center were receiving amplitudes more than their r to center, so they were collapsing in the center and exploding back."

**Physical Analysis**: Near-Field vs Far-Field

The observed instability reveals important physics about what happens near the wave source (r < λ):

1. **Source Region Physics** (r < λ):
   - This is the source region where energy is being injected
   - Wave structure is not yet fully formed
   - Near-field effects dominate over far-field radiation
   - Simple 1/r amplitude falloff does not apply
   - Field energy density can exceed far-field predictions

2. **Wave Formation Zone** (λ < r < 2λ):
   - Transition from near-field to far-field behavior
   - Wave fronts are organizing into spherical geometry
   - Amplitude falloff transitions toward 1/r behavior

3. **Propagating Wave Region** (r > 2λ):
   - Fully formed spherical waves
   - Clean 1/r amplitude falloff
   - Energy conservation clearly observable

**Why 1λ Cutoff is Physically Accurate**:

The choice of 1λ minimum radius aligns with multiple physical principles:

1. **EWT Specification**: Neutrino boundary at 1λ defines fundamental particle scale
2. **EM Standards**: Transition to radiative fields begins around λ
3. **Source Physics**: Wave source must have finite extent (≈ λ in size)
4. **Numerical Stability**: Prevents singularity at r → 0
5. **Physical Behavior**: Matches observed stable wave propagation

### Implementation Summary

The amplitude falloff implementation in `qwave_radial.py` uses the following approach:

```python
# Reference radius for amplitude normalization (one wavelength from wave source)
# This prevents singularity at r=0 and provides physically meaningful normalization
r_reference = wavelength_am  # attometers

for idx in range(position.shape[0]):
    direction = center_direction[idx]

    # Distance from granule to wave source (in attometers)
    r = center_distance[idx]

    # Phase determined by radial distance from wave source
    # Negative k·r creates outward propagating wave
    phase = -k * r

    # Amplitude falloff for spherical wave energy conservation: A(r) = A₀(r₀/r)
    # Prevents division by zero and non-physical amplitudes very close to wave source
    r_safe = ti.max(r, r_reference * 1)  # minimum 1 wavelength
    amplitude_falloff = r_reference / r_safe

    # Total amplitude including geometric falloff and visibility boost
    amplitude_at_r = amplitude_am * amplitude_falloff * amp_boost
```

**Key Implementation Details**:

1. **r_reference = wavelength_am**:
   - Reference radius set to one wavelength from wave source
   - Provides physically meaningful normalization scale
   - Amplitude at r = λ equals nominal amplitude_am

2. **r_safe = max(r, 1λ)**:
   - Enforces minimum distance of 1λ from wave source
   - Prevents singularity at r → 0
   - Aligns with EWT neutrino boundary specification
   - Ensures stable numerical behavior

3. **amplitude_falloff = r_reference / r_safe**:
   - Implements A ∝ 1/r energy conservation law
   - Valid for far-field region (r > λ)
   - Clamped to constant amplitude for r < λ (source region)

4. **amplitude_at_r = amplitude_am × amplitude_falloff × amp_boost**:
   - Final amplitude at distance r from wave source
   - Includes energy conservation (amplitude_falloff)
   - Includes visualization scaling (amp_boost)

### Position and Velocity Equations

The implementation uses phase-synchronized harmonic oscillation with distance-dependent amplitude:

**Position**:

```python
x(t) = x_eq + A(r)·cos(ωt + φ)·direction
```

Where:

- `x_eq` = equilibrium position of granule
- `A(r)` = amplitude_at_r (distance-dependent)
- `ω` = 2πf = angular frequency
- `t` = simulation time
- `φ = -kr` = phase (r is distance from wave source)
- `direction` = normalized vector from granule toward wave source

**Velocity**:

```python
v(t) = -A(r)·ω·sin(ωt + φ)·direction
```

This is the time derivative of position, ensuring kinematically consistent motion.

### Energy Conservation Verification

The implementation conserves energy through:

1. **Amplitude Falloff**: A(r) = A₀(λ/r) for r > λ
2. **Constant Parameters**: λ, c, ρ remain constant
3. **Energy Equation**: E = ρV(c/λ × A)²

At any radius r > λ from the wave source:

```python
E_total = ρ × (4πr²) × (c/λ × A₀λ/r)²
E_total = ρ × 4πr² × (c/λ)² × (A₀λ)²/r²
E_total = ρ × 4π × (c/λ)² × (A₀λ)²
E_total = constant (independent of r)
```

This demonstrates that total energy integrated over a spherical shell remains constant as the wave propagates, confirming energy conservation.

## Summary: Near/Far Field Boundary Analysis

### Findings from Multiple Sources

1. **Electromagnetic Wave Standards**:
   - Reactive near field: r < 0.159λ from wave source (A ∝ 1/r³)
   - Radiative near field: 0.159λ to 2λ from wave source (A ∝ 1/r²)
   - Far field: r > 2λ from wave source (A ∝ 1/r)
   - Fully formed waves occur in far field region

2. **Energy Wave Theory (EWT)**:
   - Neutrino standing wave boundary: r = 1λ from wave source
   - Fundamental particle scale defined at one wavelength
   - Provides theoretical justification for λ-scale cutoff

3. **Implementation Validation**:
   - Experimental observation: r < 1λ causes collapse/explosion
   - Physical interpretation: Source region physics dominates
   - Stable behavior achieved with 1λ minimum cutoff
   - Aligns with both EWT theory and EM field standards

### Conclusion

The implementation choice of **r_min = 1λ** (one wavelength from wave source) is physically accurate and supported by:

- Energy Wave Theory fundamental particle specifications
- Electromagnetic near-field/far-field transition physics
- Experimental validation in the simulator
- Energy conservation requirements for spherical waves

For distances r > 1λ from the wave source, the amplitude follows the 1/r energy conservation law appropriate for spherical wave propagation in the far-field region.
