# SPHERICAL WAVE ENERGY CONSERVATION & AMPLITUDE FALLOFF

This document provides comprehensive physical justification and implementation details for amplitude falloff in spherical wave propagation within the OpenWave subatomic simulator. The implementation accounts for energy conservation, near-field/far-field physics, and Energy Wave Theory (EWT) specifications.

## Visual Reference

See `01_wave_spherical_plot.py` for the amplitude vs radius visualization showing:

- Near-field behavior (r < λ)
- Transition zone (λ to 2λ)
- Far-field 1/r falloff (r > 2λ)
- Amplitude cap constraint (A ≤ r)
- Singularity prevention at r_min = 1λ

![Amplitude vs Radius](images/wave_amplitude_vs_radius.png)

## Physical Background: Spherical Wave Energy Conservation

### Energy Conservation Principle

For spherical waves propagating from a point source, total energy must be conserved as the wave expands outward. The energy of a wave system is given by (EWT frequency-centric formulation):

```python
E = ρV(fA)²
```

Where:

- `E` = total energy
- `ρ` = medium density
- `V` = volume
- `f` = frequency (Hz, where f = c/λ)
- `A` = amplitude

Equivalent wavelength-based form (historical):

```python
E = ρV(c/λ × A)² = ρV(fA)²  (since f = c/λ)
```

As a spherical wave propagates through a uniform medium:

- `f` remains constant (frequency unchanged in uniform medium, f = c/λ)
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
   - This is the source region where energy is being Charged
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

## Two Separate Physical Constraints

The implementation uses **two independent constraints** to ensure physical accuracy and numerical stability:

### 1. Singularity Prevention (r_safe)

**Purpose**: Prevent mathematical singularity (division by zero)

The amplitude formula `A(r) = A₀·(λ/r)` creates a singularity at r → 0:

- As r approaches zero, A(r) → ∞ (infinite amplitude)
- Division by zero causes numerical instability
- Solution: `r_safe = max(r, 1λ)` enforces minimum distance

**Effect**: For r < 1λ, amplitude calculation uses r = 1λ, keeping amplitude constant at A₀.

### 2. Physical Cap (A ≤ r)

**Purpose**: Prevent granules from crossing through the wave source

For longitudinal waves, if amplitude A exceeds distance r:

- Granule displacement could place it on opposite side of source
- Physically impossible for longitudinal wave propagation
- Violates constraint: |x - x_eq| ≤ |x_eq - x_source|

**Effect**: Caps amplitude to never exceed distance: `A_final = min(A(r), r)`

These constraints work together but solve different problems:

- **Singularity prevention**: Mathematical/numerical issue
- **A ≤ r cap**: Physical realism constraint

### Implementation Summary

The amplitude falloff implementation in `energy_wave_level0.py` (lines 215-229) uses the following approach:

```python
# Reference radius for amplitude normalization (one wavelength from wave source)
# Prevents singularity at r=0 and provides physically meaningful normalization
r_reference = wavelength_am  # attometers

for source_idx in range(num_sources):
    # Get precomputed direction and distance for this granule-source pair
    direction = sources_direction[granule_idx, source_idx]
    r = sources_distance_am[granule_idx, source_idx]  # distance in attometers

    # Spatial phase: φ = -k·r (negative for outward propagation)
    spatial_phase = -k * r
    total_phase = spatial_phase + phase_offset

    # CONSTRAINT 1: Singularity Prevention
    # Amplitude falloff for spherical wave: A(r) = A₀·(r₀/r)
    # Use r_safe to prevent singularity (division by zero) at r → 0
    # Enforces r_min = 1λ based on EWT neutrino boundary and EM near-field physics
    r_safe = ti.max(r, r_reference)  # minimum 1 wavelength from source
    amplitude_falloff = r_reference / r_safe

    # Total amplitude at granule distance from source
    # Step 1: Apply energy conservation (1/r falloff) and visualization scaling
    amplitude_at_r = amp_local_peak_am * amplitude_falloff * amp_boost

    # CONSTRAINT 2: Physical Cap (A ≤ r)
    # Step 2: Cap amplitude to distance from source (A ≤ r)
    # Prevents non-physical behavior: granules crossing through wave source
    # When A > r, displacement could exceed distance to source, placing granule
    # on opposite side of source (physically impossible for longitudinal waves)
    # This constraint ensures: |x - x_eq| ≤ |x_eq - x_source|
    amplitude_at_r_cap = ti.min(amplitude_at_r, r)
```

**Key Implementation Details**:

1. **r_reference = wavelength_am** (energy_wave_level0.py:191):
   - Reference radius set to one wavelength from wave source
   - Provides physically meaningful normalization scale
   - Amplitude at r = λ equals nominal amplitude A₀

2. **r_safe = max(r, r_reference)** (energy_wave_level0.py:217):
   - **CONSTRAINT 1**: Singularity prevention
   - Enforces minimum distance of 1λ from wave source
   - Prevents division by zero when r → 0
   - Aligns with EWT neutrino boundary specification
   - Ensures numerical stability

3. **amplitude_falloff = r_reference / r_safe** (energy_wave_level0.py:218):
   - Implements A ∝ 1/r energy conservation law
   - Valid for far-field region (r > λ)
   - Clamped to constant amplitude for r < λ (source region)

4. **amplitude_at_r = amp_local_peak_am × amplitude_falloff × amp_boost** (energy_wave_level0.py:222):
   - Amplitude after singularity prevention and energy conservation
   - Includes visualization scaling (amp_boost)

5. **amplitude_at_r_cap = min(amplitude_at_r, r)** (energy_wave_level0.py:229):
   - **CONSTRAINT 2**: Physical cap (A ≤ r)
   - Final amplitude used in wave equation
   - Prevents granules from crossing through wave source
   - Ensures physical validity of longitudinal wave displacement

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
2. **Constant Parameters**: f, λ, c, ρ remain constant (f = c/λ)
3. **Energy Equation** (frequency-centric): E = ρV(fA)²

At any radius r > λ from the wave source:

```python
# Frequency-centric form
E_total = ρ × (4πr²) × (A₀λ/r × f)²
E_total = ρ × 4πr² × f² × (A₀λ)²/r²
E_total = ρ × 4π × f² × (A₀λ)²
E_total = constant (independent of r)

# Equivalent wavelength-based form (since f = c/λ)
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

The implementation uses **two independent constraints** for physical accuracy:

1. **Singularity Prevention (r_min = 1λ)**:
   - Prevents division by zero at wave source
   - Supported by EWT neutrino boundary specification
   - Aligns with EM near-field/far-field transition physics
   - Ensures numerical stability

2. **Physical Cap (A ≤ r)**:
   - Prevents granules from crossing through wave source
   - Enforces physical constraint for longitudinal waves
   - Active primarily in near field (r < 0.2λ)

For distances r > 1λ from the wave source, the amplitude follows the 1/r energy conservation law appropriate for spherical wave propagation in the far-field region.

## Visualization

The behavior described above is visualized in `images/wave_amplitude_vs_radius.png`, generated by `01_wave_spherical_plot.py`. The plot shows:

- **Red line**: Actual implemented amplitude with both constraints
- **Blue dashed line**: Theoretical 1/r falloff (uncapped)
- **Black dotted line**: A = r boundary (cap limit)
- **Field regions**: Near field (r < λ), Transition (λ to 2λ), Far field (r > 2λ)
- **Key points**: A(1λ) = 1.0A₀, A(2λ) = 0.5A₀

The plot uses actual EWT constants:

- A₀ = 9.215×10⁻¹⁹ m (0.922 attometers)
- λ = 2.854×10⁻¹⁷ m (28.541 attometers)
