# Force & Motion Module

## Research Documentation

This document captures the implementation plan for the Force & Motion module in OpenWave, connecting energy-wave infrastructure with physical reality: **motion of matter**.

---

## Table of Contents

1. [Overview](#overview)
1. [Theoretical Foundation](#theoretical-foundation)
1. [Implementation Plan](#implementation-plan)
1. [Data Structures](#data-structures)
1. [Phase 1: Motion Smoke Test](#phase-1-motion-smoke-test)
1. [Phase 2: Energy Calculation](#phase-2-energy-calculation)
1. [Phase 3: Force from Energy Gradient](#phase-3-force-from-energy-gradient)
1. [Phase 4: Integration and Validation](#phase-4-integration-and-validation)
1. [Future Extensions](#future-extensions)

---

## Overview

### The Goal

To implement force and motion calculations that connect OpenWave's energy-wave simulation with observable physical phenomena. This module represents a **paradigm validation point** where EWT fundamentals meet classical mechanics.

### Key Insight

**Forces emerge from wave interference patterns**:

- **Same phase** wave centers (e.g., both 0 deg) = **constructive interference** between them = higher amplitude = **repulsion**
- **Opposite phase** wave centers (e.g., 0 deg and 180 deg) = **destructive interference** between them = lower amplitude = **attraction**

This directly models **electrostatic charge** as **wave-center phase offset**.

### Why This Matters

If successful, this implementation will:

1. Validate EWT force derivation from energy gradients
1. Reproduce electric field force behavior from wave interference
1. Provide a novel numerical solution independent of Maxwell's equations
1. Open pathways to unified force modeling (EM, gravity, strong)
1. Enable parameter fine-tuning to match Coulomb force behavior
1. Tie EWT fundamentals with electromagnetism theories, leading to deeper understanding of EM phenomena and potential technological advancements from fundamental force field modeling

---

## Theoretical Foundation

### Energy Wave Equation

From EWT, energy at each voxel:

```text
E = rho * V * (f * A)^2
```

Where:

- `rho` = medium density (3.860e22 kg/m3)
- `V` = voxel volume (dx^3)
- `f` = wave frequency (1.050e25 Hz)
- `A` = wave amplitude (meters)

### Force from Energy Gradient

**Fundamental Principle**: Force is the negative gradient of energy.

```text
F = -grad(E)
```

**Full derivation**:

```text
F = -grad[rho * V * (f * A)^2]
  = -rho * V * grad(f^2 * A^2)
  = -rho * V * [f^2 * grad(A^2) + A^2 * grad(f^2)]
  = -rho * V * [2 * f^2 * A * grad(A) + 2 * A^2 * f * grad(f)]
  = -2 * rho * V * f * A * [f * grad(A) + A * grad(f)]
```

**Monochromatic simplification** (when grad(f) approx 0):

```text
F = -2 * rho * V * f^2 * A * grad(A)
```

### Minimum Amplitude Principle (MAP)

Particles (wave centers) move to minimize amplitude:

- Force points toward **decreasing energy** (downhill on energy landscape)
- Wave centers seek regions of **lower amplitude**
- This creates attraction/repulsion based on interference patterns

### Charge as Phase Offset

In EWT/OpenWave, **electrostatic charge** is modeled as the **phase offset** of a wave center:

| Charge | Phase Offset | Wave Pattern |
| ------ | ------------ | ------------ |
| Electron (-) | 0 deg | Reference |
| Positron (+) | 180 deg (pi) | Inverted |

When two wave centers interact:

- **Same phase** (both 0 deg or both 180 deg): Constructive interference -> higher amplitude between them -> energy gradient pushes outward -> **repulsion**
- **Opposite phase** (0 deg and 180 deg): Destructive interference -> **energy valley** between them (lower amplitude) -> energy gradient pulls inward -> **attraction**

---

## Implementation Plan

### Architecture Overview

```text
+-------------------------------------------------------------+
|                    SIMULATION LOOP                          |
+-------------------------------------------------------------+
|  1. propagate_wave()      -> Updates psiL_am, tracks ampL_am|
|  2. compute_force()       -> F = -grad(E) from amplitude    |
|  3. compute_motion()      -> a = F/m, integrate v and pos   |
|  4. render()              -> Visualize particles + waves    |
+-------------------------------------------------------------+
```

### Module Structure

```text
force_motion.py
|-- compute_energy_field()      # E = rho*V*(f*A)^2 per voxel
|-- compute_force_vector()      # F = -grad(E) around each WC
|-- integrate_motion_euler()   # Euler/Verlet integration
```

---

## Data Structures

### Existing Fields (Available)

From `spacetime_medium.py`:

```python
wave_field.psiL_am[i,j,k]    # Instantaneous displacement (am)
wave_field.dx                 # Voxel size (m)
wave_field.nx, ny, nz        # Grid dimensions
```

From `spacetime_ewave.py` (Trackers):

```python
trackers.ampL_am[i,j,k]      # RMS amplitude per voxel (am)
```

From `particle.py`:

```python
wave_center.position_grid[wc]      # Grid indices (i32, shape=num_sources)
wave_center.velocity_amrs[wc]    # Velocity in am/rs (f32, shape=num_sources)
wave_center.offset[wc]             # Phase offset in radians (charge analog)
```

### New Fields Required

Add to `particle.py` WaveCenter class:

```python
# Position as float for smooth motion (grid indices)
self.position_float = ti.Vector.field(3, dtype=ti.f32, shape=num_sources)

# Velocity in attometers per rontosecond (am/rs) for f32 precision
# Sublight speeds: 0.0001 to 0.3 am/rs (c = 0.3 am/rs)
self.velocity_amrs = ti.Vector.field(3, dtype=ti.f32, shape=num_sources)

# Force in Newtons (SI) - kept in SI for physics accuracy
self.force = ti.Vector.field(3, dtype=ti.f32, shape=num_sources)

# Mass in kg (SI) - hardcoded initially, computed later from standing wave energy
self.mass = ti.field(dtype=ti.f32, shape=num_sources)
```

### Unit System

OpenWave uses scaled units for f32 floating-point precision:

| Quantity | Units | Notes |
| -------- | ----- | ----- |
| Position | grid indices (float) | Smooth motion integration |
| Velocity | am/rs | Attometers per rontosecond |
| Acceleration | am/rs² | Converted from m/s² |
| Force | N (SI) | Kept in SI for physics accuracy |
| Mass | kg (SI) | Kept in SI for physics accuracy |
| Time | rs | Rontoseconds (1e-27 s) |

**Key conversions**:

```text
ACCEL_MS2_TO_AMRS2 = 1e-36    # m/s² to am/rs²
c = 0.3 am/rs                  # Speed of light
Sublight range: 0.0001 to 0.3 am/rs
```

### Physical Constants Reference

From `constants.py`:

```python
MEDIUM_DENSITY = 3.859764604e22    # kg/m^3 (rho)
EWAVE_FREQUENCY = 1.050393558e25   # Hz (f)
EWAVE_AMPLITUDE = 9.215405708e-19  # m (A)
ELECTRON_MASS = 9.109384e-31       # kg
ATTOMETER = 1e-18                  # m/am
RONTOSECOND = 1e-27                # s/rs
```

---

## Phase 1: Motion Smoke Test

### Phase 1 Objective

Implement particle motion with **hardcoded arbitrary force** to:

1. Verify motion integration works correctly
1. Test position update visualization
1. Validate unit conversions and coordinate systems

### Phase 1 Status: COMPLETE ✓

### Implementation: `smoketest_particle_motion()`

```python
@ti.kernel
def smoketest_particle_motion(
    wave_field: ti.template(),
    wave_center: ti.template(),
    dt_rs: ti.f32,
):
    """
    Smoke test: Apply computed acceleration for consistent visible motion.
    Bypasses F/m calculation - purely for testing motion integration.
    """
    # Voxel size in attometers for position conversion
    dx_am = wave_field.dx / ATTOMETER

    # ================================================================
    # SMOKE TEST: Compute acceleration for consistent visible motion
    # ================================================================
    # Target: move 1 grid cell after ~N frames (adjustable)
    # Kinematic: x = 0.5 * a * (N * dt)²
    # Solving for a: a = 2 * x / (N * dt)²
    # Where x = dx_am (1 grid cell in attometers)
    target_frames = ti.cast(50.0, ti.f32)  # Move 1 grid cell in ~50 frames
    a_smoke = 2.0 * dx_am / (target_frames * target_frames * dt_rs * dt_rs)

    for wc in range(wave_center.num_sources):
        # Apply computed acceleration in +x direction
        a_x = a_smoke
        a_y = ti.cast(0.0, ti.f32)
        a_z = ti.cast(0.0, ti.f32)

        # Update velocity: v_new = v_old + a * dt (in am/rs)
        wave_center.velocity_amrs[wc][0] += a_x * dt_rs
        wave_center.velocity_amrs[wc][1] += a_y * dt_rs
        wave_center.velocity_amrs[wc][2] += a_z * dt_rs

        # Position change in attometers
        dx_am_step = wave_center.velocity_amrs[wc][0] * dt_rs
        dy_am_step = wave_center.velocity_amrs[wc][1] * dt_rs
        dz_am_step = wave_center.velocity_amrs[wc][2] * dt_rs

        # Convert attometers to grid index change
        di = dx_am_step / dx_am
        dj = dy_am_step / dx_am
        dk = dz_am_step / dx_am

        # Update float position (smooth motion)
        wave_center.position_float[wc][0] += di
        wave_center.position_float[wc][1] += dj
        wave_center.position_float[wc][2] += dk

        # Update integer grid position for wave generation
        wave_center.position_grid[wc][0] = ti.cast(wave_center.position_float[wc][0], ti.i32)
        wave_center.position_grid[wc][1] = ti.cast(wave_center.position_float[wc][1], ti.i32)
        wave_center.position_grid[wc][2] = ti.cast(wave_center.position_float[wc][2], ti.i32)
```

### Key Finding: Dynamic Acceleration Calculation

**Problem**: Hardcoded force values (e.g., 1e-25 N) produce different motion speeds across xperiments with varying universe sizes and timesteps.

**Solution**: Compute acceleration dynamically based on simulation parameters:

```python
a_smoke = 2.0 * dx_am / (target_frames² * dt_rs²)
```

This guarantees consistent motion (1 grid cell in N frames) regardless of xperiment configuration.

### Phase 1 Issues Resolved

#### 1. Smooth Rendering

**Problem**: Particle motion appeared jumpy, moving in discrete steps.

**Cause**: Renderer used `position_grid` (integer) instead of `position_float`.

**Solution**: Update `_launcher.py` to render using `position_float`:

```python
# Use position_float for smooth rendering (position_grid is integer, causes jumpy motion)
wc_pos_screen = ti.Vector([
    state.wave_center.position_float[wc_idx][0] / max_dim,
    state.wave_center.position_float[wc_idx][1] / max_dim,
    state.wave_center.position_float[wc_idx][2] / max_dim,
], dt=ti.f32)
```

#### 2. Asymmetric Universe Support

**Problem**: Particles rendered at wrong positions in asymmetric universes (e.g., z = 1/4 of x,y).

**Cause**: Renderer normalized by individual dimensions (nx, ny, nz) instead of max dimension.

**Solution**: Normalize by `max_grid_size` (like flux_mesh does):

```python
max_dim = float(state.wave_field.max_grid_size)
# All coordinates normalized by max_dim for correct aspect ratio
```

#### 3. Amplitude/Frequency Trail Artifacts

**Problem**: Moving wave centers left trails of amplitude and frequency values in voxels they passed through.

**Cause**: EMA tracking retains historical values. Waves still propagate through old positions, keeping EMA values high.

**Solution**: Unconditional decay + higher alpha for faster response:

```python
# In spacetime_ewave.py tracker updates:
alpha_rms_L = 0.05  # Higher alpha for faster EMA response
decay_factor = ti.cast(0.99, ti.f32)  # Unconditional decay
trackers.ampL_am[i, j, k] = new_ampL * decay_factor
```

| Parameter | Old Value | New Value | Effect |
| --------- | --------- | --------- | ------ |
| alpha_rms | 0.005 | 0.05 | 10x faster EMA adaptation |
| decay_factor | N/A | 0.99 | Clears trails in ~100-200 frames |

**How it works**:

- Active regions (near wave center): EMA update counteracts decay → stable tracking
- Stale regions (source moved away): No reinforcement → gradual decay to zero

---

## Phase 2: Energy Calculation

**Note**: In the current implementation, Phase 2 is merged into Phase 3. Force is computed directly from the amplitude gradient without storing an intermediate energy field. This approach is more efficient and avoids an extra 3D field allocation. The energy calculation is preserved here for reference and potential future use (e.g., energy visualization, debugging).

### Phase 2 Objective

Compute energy per voxel from tracked amplitude.

### Implementation

```python
@ti.func
def compute_voxel_energy(
    amplitude_am: ti.f32,
    voxel_volume: ti.f32,
) -> ti.f32:
    """
    Compute energy at a voxel using EWT energy equation.

    E = rho * V * (f * A)^2

    Args:
        amplitude_am: RMS amplitude in attometers
        voxel_volume: Voxel volume in m^3

    Returns:
        Energy in Joules
    """
    # Convert amplitude from attometers to meters
    A_m = amplitude_am * ATTOMETER

    # EWT energy equation
    E = MEDIUM_DENSITY * voxel_volume * (EWAVE_FREQUENCY * A_m) ** 2

    return E
```

---

## Phase 3: Force from Energy Gradient

### Phase 3 Objective

Compute force vector at each wave center from energy gradient.

### Phase 3 Status: COMPLETE ✓

**BREAKTHROUGH**: Electrical attraction forces successfully demonstrated from wave interference!

### Key Achievement: Opposite Phase Attraction

Test with `0035_waves.py` (two wave centers with phases 0° and 180°):

- Wave centers move **toward each other** (attraction)
- Force computed from amplitude gradient: `F = -2 * rho * V * f² * A * grad(A)`
- This validates the **EWT model for electrostatic attraction**

### Critical Implementation Detail: f32 Precision

The force scale factor `2 * rho * V * f²` involves extreme values that exceed f32 limits:

| Value | Magnitude | f32 Limit | Issue |
| ----- | --------- | --------- | ----- |
| f * f | ~1.5e+40 | 3.4e+38 (max) | **Overflow** |
| V = dx³ | ~1e-53 | 1.2e-38 (min) | **Underflow** |

**Solution**: Interleave large and small values to keep intermediates in f32 range:

```python
# WRONG: V goes below f32 min, f*f exceeds f32 max
V = dx_m * dx_m * dx_m           # ~1e-53 rounds to 0!
force_scale = 2.0 * rho * V * f * f  # Result = 0

# CORRECT: Interleave to stay in range
# rho*dx = 1.9e+7, *f = 2.3e+27, *dx = 5e+9, *f = 6e+29, *dx = 1.3e+12
force_scale = 2.0 * rho * dx_m * f * dx_m * f * dx_m  # ~1e+12 ✓
```

### Implementation: `compute_force_vector()`

```python
@ti.kernel
def compute_force_vector(
    wave_field: ti.template(),
    trackers: ti.template(),
    wave_center: ti.template(),
):
    """
    Compute force on each wave center from energy gradient.

    F = -grad(E) = -2 * rho * V * f^2 * A * grad(A)   (monochromatic)

    Uses central finite differences for gradient calculation.
    Samples amplitude field around wave center position.
    """
    # Physical constants
    rho = ti.cast(MEDIUM_DENSITY, ti.f32)
    f = ti.cast(EWAVE_FREQUENCY, ti.f32)
    dx_m = wave_field.dx  # voxel size in meters

    # Force scale factor: 2 * rho * V * f^2 where V = dx³
    # CRITICAL: Interleave large/small values to avoid f32 under/overflow!
    force_scale = 2.0 * rho * dx_m * f * dx_m * f * dx_m  # ~1e+12 (safe)

    for wc in range(wave_center.num_sources):
        # Get wave center grid position
        i = wave_center.position_grid[wc][0]
        j = wave_center.position_grid[wc][1]
        k = wave_center.position_grid[wc][2]

        # Boundary check
        nx, ny, nz = wave_field.nx, wave_field.ny, wave_field.nz
        if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
            # Sample amplitude at center (convert to meters)
            A_center = trackers.ampL_am[i, j, k] * ATTOMETER

            # Central difference gradient: grad(A) = (A+ - A-) / (2*dx)
            # X gradient
            A_xp = trackers.ampL_am[i+1, j, k] * ATTOMETER
            A_xm = trackers.ampL_am[i-1, j, k] * ATTOMETER
            dA_dx = (A_xp - A_xm) / (2.0 * dx_m)

            # Y gradient
            A_yp = trackers.ampL_am[i, j+1, k] * ATTOMETER
            A_ym = trackers.ampL_am[i, j-1, k] * ATTOMETER
            dA_dy = (A_yp - A_ym) / (2.0 * dx_m)

            # Z gradient
            A_zp = trackers.ampL_am[i, j, k+1] * ATTOMETER
            A_zm = trackers.ampL_am[i, j, k-1] * ATTOMETER
            dA_dz = (A_zp - A_zm) / (2.0 * dx_m)

            # Force: F = -2 * rho * V * f^2 * A * grad(A)
            wave_center.force[wc][0] = -force_scale * A_center * dA_dx
            wave_center.force[wc][1] = -force_scale * A_center * dA_dy
            wave_center.force[wc][2] = -force_scale * A_center * dA_dz
```

### Velocity Clamping (Relativistic Limit)

Forces are so strong (~0.01 N on neutrino mass) that particles instantly exceed light speed. Added velocity clamping:

```python
# Clamp velocity to speed of light (c = 0.3 am/rs)
c_amrs = ti.cast(0.3, ti.f32)
v_mag = ti.sqrt(vx**2 + vy**2 + vz**2)
if v_mag > c_amrs:
    scale = c_amrs / v_mag
    wave_center.velocity_amrs[wc] *= scale
```

**TODO**: Replace with proper relativistic treatment or add force/mass multipliers for tuning.

### Scale Factor Correction

For scaled universes (scale_factor > 1), forces must be corrected:

```python
# Force scales as S⁴ with universe scaling
S = wave_field.scale_factor
S4 = S * S * S * S
force_scale = force_scale / S4
```

**Why S⁴?** Scaling the universe by S affects:

- Voxel volume V → S³
- Amplitude gradients → 1/S
- Combined effect on force: S³ × (1/S) = S² ... but empirically S⁴ correction works

### Force Multiplier for Visualization

Real quantum forces are tiny. A multiplier boosts them for visible motion:

```python
FORCE_MULTIPLIER = ti.cast(2000, ti.f32)  # Tuned for visible motion
a_x = (F_x / m) * accel_conv * FORCE_MULTIPLIER
```

**Tuning notes:**

- Too high (1e8+): Velocity oscillates between ±c every frame
- Too low (1): No visible motion in reasonable time
- Sweet spot (~2000): Gradual velocity buildup, stable motion

### EMA Alpha Tuning for Amplitude Stability

Reduced EMA alpha 10x for more stable amplitude tracking:

```python
# In spacetime_ewave.py
alpha_rms_L = 0.005  # Was 0.05 - now 10x slower adaptation
```

**Trade-off**: Slower response to wave center movement, but more stable gradient calculations.

### Gradient Sampling Radius

The sampling radius for gradient calculation is critical:

```python
# Sample radius affects which interference pattern features are captured
sample_radius = ti.max(min_dim * pct // 100, 1)
```

**Findings**:

| Radius | Effect |
| ------ | ------ |
| 1 voxel | Works well at scale 1.0x, enables particle annihilation |
| 5% | Good for low scale factors |
| 15% | Better for high scale factors (45x+) |

**Known Issue**: Optimal radius depends on scale factor. Scale factor > 1.0x requires larger sampling radius.

### Experimental Results

#### Test Experiments (Scale Factor 1.0x)

**`0035a_waves.py`** - Opposite Phase (Attraction):

- Phases: 0° and 180°
- Result: **Particles attract and approach each other**
- Validates EWT model for electrical attraction

**`0035b_waves.py`** - Same Phase (Repulsion):

- Phases: 0° and 0°
- Result: **Particles repel and move apart**
- Validates EWT model for electrical repulsion

**Particle Annihilation**:

- With sample_radius = 1 voxel, opposite-phase particles annihilate when touching
- This emergent behavior matches electron-positron annihilation

#### Working Configuration

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Scale Factor | 1.0x | Higher scales have gradient sampling issues |
| Force Multiplier | 2000 | Visible motion without oscillation |
| Sample Radius | 1 voxel | Enables annihilation behavior |
| EMA Alpha | 0.005 | 10x reduction for stability |
| Velocity Clamp | 0.3 am/rs | Speed of light limit |

### Known Issues and Future Work

1. **Scale Factor > 1.0x**: Gradient sampling produces incorrect forces at large scale factors. Needs scale-aware sampling radius or different approach.

2. **Force Calibration**: Current force multiplier is arbitrary. Future work should calibrate to match Coulomb force at known distances.

3. **Annihilation Physics**: Particles "annihilate" by stopping when touching, but true annihilation would convert mass to energy (photon emission).

---

## Phase 4: Integration and Validation

### Motion Integration (Euler)

```python
@ti.kernel
def integrate_motion_euler(
    wave_field: ti.template(),
    wave_center: ti.template(),
    dt_rs: ti.f32,
):
    """
    Integrate particle motion using Euler method.

    v_new = v_old + a * dt  (velocity in am/rs)
    x_new = x_old + v_new * dt  (position in grid indices)
    """
    # Conversion factor: m/s² to am/rs²
    accel_conv = ti.cast(ACCEL_MS2_TO_AMRS2, ti.f32)

    # Voxel size in attometers for position conversion
    dx_am = wave_field.dx / ATTOMETER

    for wc in range(wave_center.num_sources):
        # Get force (Newtons) and mass (kg)
        F_x = wave_center.force[wc][0]
        F_y = wave_center.force[wc][1]
        F_z = wave_center.force[wc][2]
        m = wave_center.mass[wc]

        # Acceleration in m/s², then convert to am/rs²
        a_x = (F_x / m) * accel_conv
        a_y = (F_y / m) * accel_conv
        a_z = (F_z / m) * accel_conv

        # Update velocity (am/rs)
        wave_center.velocity_amrs[wc][0] += a_x * dt_rs
        wave_center.velocity_amrs[wc][1] += a_y * dt_rs
        wave_center.velocity_amrs[wc][2] += a_z * dt_rs

        # Position change in attometers
        dx_am_step = wave_center.velocity_amrs[wc][0] * dt_rs
        dy_am_step = wave_center.velocity_amrs[wc][1] * dt_rs
        dz_am_step = wave_center.velocity_amrs[wc][2] * dt_rs

        # Convert to grid index change
        di = dx_am_step / dx_am
        dj = dy_am_step / dx_am
        dk = dz_am_step / dx_am

        wave_center.position_float[wc][0] += di
        wave_center.position_float[wc][1] += dj
        wave_center.position_float[wc][2] += dk

        # Sync integer position for wave generation
        wave_center.position_grid[wc][0] = ti.cast(wave_center.position_float[wc][0], ti.i32)
        wave_center.position_grid[wc][1] = ti.cast(wave_center.position_float[wc][1], ti.i32)
        wave_center.position_grid[wc][2] = ti.cast(wave_center.position_float[wc][2], ti.i32)
```

### Validation Tests

1. **Single WC, no neighbors**: Should remain stationary (no gradient)
1. **Two WCs, same phase**: Should repel (constructive interference)
1. **Two WCs, opposite phase**: Should attract (destructive interference)
1. **Compare force magnitude**: Should approach Coulomb law at macroscopic distances

---

## Future Extensions

### 1. Velocity Verlet Integration

More accurate, energy-conserving integration for production use.

### 2. Magnetic Force (Transverse Waves)

When psiT is implemented, add velocity-dependent magnetic forces.

### 3. Gravitational Force

From amplitude "shading" effect of massive particles.

### 4. Multi-Frequency Forces

Full force equation with frequency gradients for different particle types.

### 5. Strong Force at Short Range

Standing wave node locking at nuclear distances.

---

## Integration with Simulation Loop

### Updated `compute_force_motion()` in `_launcher.py`

```python
def compute_force_motion(state):
    """Compute forces and update particle motion."""

    # TODO: Configuration - set to False after smoke test passes
    USE_SMOKE_TEST = True

    if USE_SMOKE_TEST:
        # PHASE 1: Smoke test with hardcoded force
        force_motion.smoketest_particle_motion(
            state.wave_field,
            state.wave_center,
            state.dt_rs,
        )
    else:
        # PHASE 3+: Compute force from energy gradient, then integrate motion
        force_motion.compute_force_vector(
            state.wave_field,
            state.trackers,
            state.wave_center,
        )
        force_motion.integrate_motion_euler(
            state.wave_field,
            state.wave_center,
            state.dt_rs,
        )
```

---

## References

### Energy Wave Theory

- EWT equations: <https://energywavetheory.com/equations/>
- Research papers: `/research_requirements/scientific_source/`

### Previous Documentation

- [01_wolff_lafreniere.md](./01_wolff_lafreniere.md) - Wave equation foundations
- [04_FORCE_MOTION.md](/openwave/xperiments/b_laplace_propagation/research/04_FORCE_MOTION.md) - Earlier research

### Key Physics

- **MAP (Minimum Amplitude Principle)**: Particles seek lowest amplitude
- **Phase = Charge**: Wave center phase offset models electrostatic charge
- **Energy Gradient = Force**: F = -grad(E) connects waves to motion

---

*Document created for OpenWave Force & Motion module implementation.*
