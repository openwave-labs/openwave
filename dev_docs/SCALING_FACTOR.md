# Scale Factor: Computational Tractability for Wave Simulation

This document explains the scale factor implementation in OpenWave LEVEL-1, its physical justification, and why it correctly preserves wave physics while enabling computational tractability.

## The Problem: Computational Resolution Limits

The fundamental energy-wave has extremely small dimensions:

- Wavelength (λ): 2.854 × 10⁻¹⁷ m
- Amplitude (A): 9.215 × 10⁻¹⁹ m
- Frequency (f): 1.050 × 10²⁵ Hz

To accurately simulate wave propagation, we need adequate spatial sampling:

- **Minimum requirement**: >10 voxels per wavelength
- **Target sampling**: 12 voxels per wavelength (adequate for wave equation stability)

With a maximum voxel capacity of ~350 million (on an M4max GPU), simulating even a small universe at full Planck-scale resolution is computationally prohibitive.

## The Solution: Scale Factor

The scale factor (S) increases the effective wavelength and amplitude while decreasing frequency, maintaining physical invariants that preserve wave behavior.

### Scale Factor Computation

```python
# In L1_field_grid.py
min_sampling = 12  # minimum voxels per wavelength for adequate sampling
scale_factor = max(min_sampling / (EWAVE_LENGTH / dx), 1)
```

The scale factor is computed to ensure at least 12 voxels per wavelength given the current voxel size (dx).

## Scaling Relationships

| Quantity | Symbol | Base Value | Scaled Value | Scale Factor |
|----------|--------|------------|--------------|--------------|
| Wavelength | λ | λ₀ | λ₀ × S | × S |
| Amplitude | A | A₀ | A₀ × S | × S |
| Frequency | f | f₀ | f₀ / S | ÷ S |
| Wave Speed | c | c₀ | c₀ | unchanged |
| Wave Steepness | A/λ | A₀/λ₀ | A₀/λ₀ | unchanged |
| Energy Density | ρ(fA)² | ρ(f₀A₀)² | ρ(f₀A₀)² | unchanged |

## Physical Justification

### Preserving Wave Steepness (A/λ)

Wave steepness is a critical dimensionless parameter that characterizes wave behavior:

```text
Steepness = A/λ

With scaling:
A_scaled/λ_scaled = (A₀ × S)/(λ₀ × S) = A₀/λ₀
```

The steepness ratio is **invariant** under our scaling transformation.

### Preserving the Energy-Wave Equation

The fundamental EWT energy equation:

```text
E = ρV(c/λ × A)² = ρV(fA)²
```

With our scaling:

```text
f_scaled × A_scaled = (f₀/S) × (A₀×S) = f₀ × A₀
```

**Energy density is preserved** because the product fA remains constant.

### Preserving Wave Speed

The wave speed c = λf must remain constant for correct physics:

```text
c_scaled = λ_scaled × f_scaled = (λ₀ × S) × (f₀ / S) = λ₀ × f₀ = c
```

**Wave speed is automatically preserved** by scaling λ and f inversely.

## Why NOT Scale Wave Speed (c)?

### Froude Scaling vs. Numerical Scaling

**Froude scaling** (used in wave tank experiments) requires scaling time and velocity:

| Quantity | Froude Scale Factor |
|----------|---------------------|
| Length | λ |
| Time | √λ |
| Velocity | √λ |
| Wave period | √λ |

This is necessary for **physical similitude** of gravity-driven free-surface flows, where the Froude number (Fr = V/√gL) must match between model and prototype.

**Our case is fundamentally different:**

1. We are not building a physical scale model
1. We are performing **numerical scaling** for computational tractability
1. We are simulating the **same physical system**, just with adjusted numerical representation
1. The governing equation (wave equation) does not involve gravity-driven dynamics

### The Key Distinction

- **Froude scaling**: Different physical sizes, same physics → must scale time/velocity
- **Our scaling**: Same physical system, different numerical representation → preserve c

If we scaled c, we would change the fundamental physics. The wave equation ∂²ψ/∂t² = c²∇²ψ requires the correct wave speed to produce physically meaningful results.

## Implementation Details

### In L1_field_grid.py

```python
# Compute scale factor for computational tractability
min_sampling = 12  # minimum voxels per wavelength
self.scale_factor = max(min_sampling / (constants.EWAVE_LENGTH / self.dx), 1)

# Compute scaled resolution
self.ewave_res = constants.EWAVE_LENGTH / self.dx * self.scale_factor  # voxels/λ
```

### In L1_wave_engine.py

```python
# Scaled angular frequency (ω = 2πf)
omega_rs = 2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor

# Scaled wavelength in grid units
wavelength_grid = base_wavelength * wave_field.scale_factor / wave_field.dx

# Scaled wave number (k = 2π/λ)
k_grid = 2.0 * ti.math.pi / wavelength_grid

# Scaled amplitude
amplitude_scaled = base_amplitude_am * wave_field.scale_factor
```

### In L1_launcher.py (Display Conversion)

```python
# Convert scaled values back to physical units for display
sub.text(f"eWAVE Amplitude: {avg_amplitude/scale_factor:.1e} m")
sub.text(f"eWAVE Frequency: {avg_frequency*scale_factor:.1e} Hz")
sub.text(f"eWAVE Wavelength: {avg_wavelength/scale_factor:.1e} m")
```

## CFL Stability Condition

The timestep is computed from the CFL (Courant-Friedrichs-Lewy) condition:

```python
# CFL Condition: dt ≤ dx / (c × √3) for 3D wave equation
dt_rs = dx_am / (c_amrs * sqrt(3))
cfl_factor = (c_amrs * dt_rs / dx_am)²  # must be ≤ 1/3
```

The CFL condition depends on:

- Wave speed c (not scaled)
- Voxel size dx (determined by universe size and voxel count)

It does **not** depend on wavelength or amplitude, so the scale factor does not affect numerical stability.

## Force Scaling Considerations

When particle interactions are implemented (future LEVEL-2+), forces will need careful handling.

### EWT Force Equations Scale as S⁴

From the electric force equation:

```text
F_e ∝ A⁶/λ² × (1/r²)

With scaling:
A⁶/λ² → (A×S)⁶/(λ×S)² = A⁶×S⁶/(λ²×S²) = (A⁶/λ²) × S⁴
```

### Options for Force Handling

1. **Work in scaled units**: Keep all calculations in scaled space, forces scale consistently
1. **Convert for reporting**: Divide computed forces by S⁴ when displaying physical values

Since LEVEL-1 focuses on field propagation (no particle forces yet), this is documented for future implementation.

## Validation Checklist

| Aspect | Status | Verification |
|--------|--------|--------------|
| Scale factor computation | Correct | Ensures ≥12 voxels/λ |
| Wavelength scaling (λ × S) | Correct | Applied in wave_engine.py |
| Amplitude scaling (A × S) | Correct | Applied in wave_engine.py |
| Frequency scaling (f / S) | Correct | Applied in wave_engine.py |
| Wave steepness (A/λ) | Preserved | Dimensionless, unchanged |
| Energy density ρV(fA)² | Preserved | Key invariant maintained |
| Wave speed (c) | Not scaled | Correct for numerical scaling |
| CFL timestep | Independent | Based on c and dx only |

## Summary

The scale factor implementation in OpenWave LEVEL-1 is **physically sound** because it:

1. Preserves wave steepness (A/λ ratio)
1. Preserves energy density (fA product)
1. Preserves wave speed (c = λf)
1. Does not affect numerical stability (CFL condition)
1. Enables adequate spatial sampling for accurate wave propagation

This is **numerical scaling for computational tractability**, not physical similitude scaling (like Froude). The physics remains unchanged; only the numerical representation is adjusted to fit within computational constraints.

## References

- EWT Constants: `openwave/common/constants.py`
- Energy Wave Equation: `openwave/common/equations.py`
- Scale Factor Implementation: `openwave/spacetime/L1_field_grid.py`
- Wave Engine Scaling: `openwave/spacetime/L1_wave_engine.py`
- Launcher Display: `openwave/xperiments/5_L1_field_based/L1_launcher.py`
