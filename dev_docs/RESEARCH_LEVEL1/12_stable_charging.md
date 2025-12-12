# Stable Wave Charging Strategy

## Overview

This document analyzes wave charging methods for achieving stable energy levels in the OpenWave simulation. The goal is to reach a homogeneous, isotropic energy distribution that maintains constant amplitude and frequency over time.

## Problem Statement

The simulation operates like a wave tank with reflective boundaries. Energy is injected via "chargers" that create displacement oscillations propagating as waves. The challenge is achieving:

1. Stable total energy (target: 100% nominal, tolerance: +/- 10-20%)
1. Stable RMS amplitude at probe voxels (tolerance: +/- 20%)
1. Stable frequency without spikes
1. No numerical explosions or runaway growth

### Observed Issues (Pre-Optimization)

From instrumentation plots:

- Charge level oscillates +/- 20% around 100% after initial ramp-up
- Large overshoot to ~135% around timestep 1200 before settling
- Persistent oscillations even at timestep 5000+ (never truly stabilizes)
- RMS amplitude oscillates between 0.5 and 2.0 am (factor of 4x variation)
- Frequency spikes to 2.5x nominal at various timesteps
- Beating patterns visible in displacement (interference of multiple frequencies)

## Charging Method Analysis

### Static Methods (One-Time Pulse)

Static methods inject energy once at initialization, then let waves propagate naturally.

| Method | Description | Pros | Cons | Verdict |
|--------|-------------|------|------|---------|
| `charge_full` | Fills entire domain with `A*cos(wt-kr)` spherical wave | Simple, fills volume | Sharp initial condition creates transients | Unstable |
| `charge_gaussian` | Gaussian bump `A*exp(-r^2/2s^2)` | Smooth pulse | Energy overshoot issues | Problematic |
| `charge_falloff` | `A*(l/r)*cos(wt-kr)` with 1/r decay | Physically realistic | Energy spreads too thin | Too weak |
| `charge_1lambda` | Wave only within 1 wavelength radius | Localized | Not enough energy | Too weak |

### Dynamic Methods (Continuous Oscillation)

Dynamic methods inject energy continuously during simulation.

| Method | Description | Pros | Cons | Verdict |
|--------|-------------|------|------|---------|
| `charge_oscillator_sphere` | Sphere at center oscillates | Steady source | Radial bias, not isotropic | Fair |
| `charge_oscillator_falloff` | Entire domain with 1/r decay | Fills volume | Overwrites propagation, fights wave equation | Unstable |
| `charge_oscillator_wall` | 6 boundary walls with source points | Isotropic from all directions | Requires parameter tuning | Best |

### Verdict: Dynamic Wall Charging

Wall charging is optimal for isotropic energy distribution because:

1. Creates waves from all 6 directions (isotropic field)
1. Boundary sources don't interfere with interior wave propagation
1. Reflected waves from opposite walls create standing wave patterns that homogenize energy
1. Avoids center bias inherent in point/sphere sources

**Why NOT sphere oscillator:**

- Single point source creates radial bias
- Center region always has highest energy density
- Interference with reflected waves creates complex beating patterns

## Optimal Charger Parameters

### Source Density

Sources should be spaced less than 1 wavelength apart to act as a coherent wavefront.

**Note:** The number of voxels per wavelength (`ewave_res`) is dynamic and depends on xperiment parameters (universe size, target voxel count). The value ~12 voxels/lambda is a typical minimum for stability, but actual values vary per simulation.

```python
# Ideal: sources spaced by lambda/2 for smooth wavefront
# ewave_res = voxels per wavelength (dynamic, depends on xperiment)
# spacing should be ewave_res / 2 voxels
# Example: for 464^3 grid with ewave_res=12: sources = 464/6 ~ 77 per edge
sources_per_edge = max(wave_field.min_grid_size // (wave_field.ewave_res // 2), 10)
```

Problem with sparse sources (e.g., 7 per edge on 464 grid with ewave_res=12):

- Spacing ~ 66 voxels
- Wavelength ~ 12 voxels
- Sources are 5.5 wavelengths apart leading to interference patterns

### Amplitude Boost

Lower boost with more sources provides smoother, more uniform injection:

```python
# Reduced from 100, compensated by higher source density
boost = 10 to 20
```

## Smooth Envelope Strategy (Trickle Charging)

### The Problem with Sharp Edges

The charger should gradually kick-in, similar to a battery charger or power generator. Without smooth fade-in and fade-out ramping, sharp transitions create their own wave disturbances (transients). This is analogous to trickle chargers in electronics that avoid voltage spikes by slowly ramping current.

Binary on/off charging creates transients:

```python
# Bad: instant ON/OFF creates "clicks"
if state.charging:
    ewave.charge_oscillator_wall(...)
```

Each ON/OFF transition is like flipping a light switch - it creates its own wave disturbance. The sudden change in energy injection rate is itself a perturbation that propagates through the system.

### Smooth Amplitude Envelope

Use a continuous envelope function that varies charging amplitude based on charge level:

```text
Amplitude
    ^
1.0 |         /------\
    |        /        \
0.2 |-------/          \
    |                   \
0.0 |                    \______
    +-----------------------------------> charge_level
        0%      80%    100%
         ^       ^       ^
         |       |       |
       ramp    peak    off
```

### Envelope Function Implementation

Uses `ti.math` (already imported via Taichi) for math operations:

```python
def compute_charge_envelope(charge_level: float) -> float:
    """
    With ADDITIVE chargers, energy accumulates naturally. The envelope
    controls injection rate, and baseline damping provides the energy sink.

    Phases:
    1. Ramp (0% -> 80%): Cosine ramp from 0.2 to 1.0
    2. Taper (80% -> 100%): Smooth cosine fade from 1.0 to 0
    3. Off (>100%): No charging, let damping bring it down
    """
    RAMP_END = 0.80
    TAPER_END = 1.00

    if charge_level < RAMP_END:
        t = charge_level / RAMP_END
        return 0.2 + 0.8 * (0.5 * (1.0 - ti.math.cos(ti.math.pi * t)))
    elif charge_level < TAPER_END:
        t = (charge_level - RAMP_END) / (TAPER_END - RAMP_END)
        return 0.5 * (1.0 + ti.math.cos(ti.math.pi * t))
    else:
        return 0.0
```

### Why No Maintenance with Additive Chargers?

With **additive chargers**, no maintenance is needed:

- **Energy source**: Additive chargers add `ψ += A·cos(ωt)` each timestep
- **Energy sink**: Baseline damping (0.9995 always applied)
- **Equilibrium**: System balances when injection rate equals damping rate

### Why Cosine Taper?

The cosine function has zero derivative at endpoints:

- At `charge_level = 0.90`: slope = 0 (no sudden change)
- At `charge_level = 1.00`: slope = 0 (no sudden stop)

Comparison:

```text
Linear taper:          Cosine taper:
    |\                     /\
    | \                   /  \
    |  \                 /    \___
    |   \               /
    +----\----         +----------
         ^                   ^
    Sharp corner       Smooth (zero slope)
```

## Damping Strategy

### Problem with Binary Damping

```python
# Bad: damping only at 120% creates oscillation
if state.damping:
    ewave.damp_full(state.wave_field, 0.999)
```

This causes:

1. System overshoots to 120% before any correction
1. Damping kicks in suddenly (transient!)
1. Energy drops below 120%
1. Damping stops suddenly (another transient!)
1. System oscillates between 100-120%

### Damping Factor Analysis

With `damping_factor = 0.999`:

- Per timestep energy loss: `1 - 0.999^2 ~ 0.2%` (amplitude squared for energy)
- After 1000 timesteps: `0.999^2000 ~ 13.5%` of original energy remains
- Half-life: ~346 timesteps

This is aggressive for fine-tuning but slow for emergency correction.

### Proportional Damping Implementation

Apply damping proportional to overshoot:

```python
def compute_damping_factor(charge_level, target=1.0, tolerance=0.10):
    """
    Compute proportional damping factor based on charge level.

    - At target (100%): no damping (factor = 1.0)
    - Above target + tolerance (110%): moderate damping
    - At 2x tolerance (120%): stronger damping

    Args:
        charge_level: Current energy as fraction of nominal
        target: Target charge level (default 1.0 = 100%)
        tolerance: Tolerance band width (default 0.10 = 10%)

    Returns:
        float: Damping factor 0.995 to 1.0
    """
    if charge_level <= target:
        return 1.0  # No damping below target

    # Proportional damping above target
    overshoot = (charge_level - target) / tolerance  # 0 at 100%, 1 at 110%, 2 at 120%
    overshoot = min(overshoot, 2.0)  # Cap at 2x for stability

    # Stronger damping: 1.0 -> 0.9975 -> 0.995
    # More aggressive to control overshoot quickly
    return 1.0 - 0.0025 * overshoot
```

Benefits:

1. No discontinuity: damping smoothly increases from 0 to max
1. Self-regulating: stronger overshoot means stronger correction
1. Settles naturally: as energy approaches target, damping reduces automatically

## Hysteresis Band Concept

### Definition

Hysteresis is when a system's response depends on its history, not just its current state. The classic example is a thermostat.

### Without Hysteresis (Problematic)

```python
state.charging = state.charge_level < 0.80  # Single threshold at 80%
```

This causes rapid on/off cycling near 80%:

```text
79.9% -> charging ON -> 80.1% -> charging OFF -> 79.9% -> ON -> 80.1% -> OFF...
```

### With Hysteresis Band

```python
# Two thresholds: turn ON below 75%, turn OFF above 95%
if state.charge_level < 0.75:
    state.charging = True   # Start charging
elif state.charge_level > 0.95:
    state.charging = False  # Stop charging
# else: maintain current state (this is the hysteresis!)
```

```text
Charge Level
    ^
    |   OFF zone          ON zone maintained
95% |----+                    until 75%
    |    |################
    |    |################
    |    |#### DEAD BAND ##
    |    |################
75% |    +----------------
    |          ON zone
    +-------------------------> Time
```

### Why Hysteresis Helps

1. Prevents rapid cycling: system must move significantly before switching
1. Reduces transients: fewer ON/OFF transitions mean fewer "clicks"
1. Allows natural settling: system can drift within band without intervention

### Note on Smooth Envelope

With the smooth envelope approach, hysteresis becomes less critical because:

- There's no binary ON/OFF to cycle
- The envelope smoothly transitions through all charge levels
- The system naturally settles as envelope approaches zero

The smooth envelope provides infinite hysteresis - charging amplitude is a continuous function of charge level, not a binary state.

## Complete Implementation Strategy

### Charging Control Flow

```python
def compute_wave_motion(state):
    """Compute wave propagation with smooth charging and damping."""

    # Compute smooth charging envelope (0.0 to 1.0)
    envelope = compute_charge_envelope(state.charge_level)

    # Apply wall charging with envelope-modulated amplitude
    if envelope > 0.001:  # Small threshold to avoid zero-amplitude calls
        effective_boost = BASE_BOOST * envelope
        sources = max(state.wave_field.min_grid_size // 6, 10)
        ewave.charge_oscillator_wall(
            state.wave_field,
            state.elapsed_t_rs,
            sources,
            effective_boost
        )

    # Propagate waves
    ewave.propagate_ewave(...)

    # Apply proportional damping
    damping = compute_damping_factor(state.charge_level)
    if damping < 0.9999:  # Only apply if damping is significant
        ewave.damp_full(state.wave_field, damping)
```

### Recommended Parameters

```python
# Charging parameters (ADDITIVE MODE)
DYNAMIC_BOOST = 0.5         # Low boost - accumulates each timestep
SOURCES_PER_EDGE = N // 6   # Dense source grid

# Envelope thresholds (no maintenance needed with additive)
RAMP_END = 0.80             # Ramp phase end
TAPER_END = 1.00            # Taper to zero at target

# Damping parameters
BASELINE_DAMPING = 0.9995   # Always applied (~0.05% per step)
DAMP_STRENGTH = 0.002       # Additional damping per 10% overshoot
```

### Expected Behavior

1. **Timesteps 0-N**: Ramp phase, charge builds via additive injection
1. **Near 80%**: Full power, rapid approach to target
1. **80-100%**: Taper phase, charging fades to zero
1. **100%+**: Off phase, baseline damping brings overshoot down

### Success Criteria

- Stable frequency at nominal value (ACHIEVED)
- Charge level stable with minimal oscillation (ACHIEVED)
- Reach 100% target (requires DYNAMIC_BOOST tuning)
- No runaway growth or collapse

## Energy Conservation Analysis

### Leap-Frog Integration

The simulation uses Leap-Frog (Verlet) integration which is symplectic:

```text
ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
```

Symplectic integrators conserve energy **of the discretized system**.

### Boundary Conditions

**Neumann BC (∂ψ/∂n = 0)** - Currently implemented for energy-conserving reflection.

Implementation via ghost cell copy before Laplacian computation:

```python
# X-faces: boundary = adjacent interior value
wave_field.displacement_am[0, j, k] = wave_field.displacement_am[1, j, k]
wave_field.displacement_am[nx-1, j, k] = wave_field.displacement_am[nx-2, j, k]
# Similar for Y and Z faces
```

This ensures waves reflect perfectly without energy loss at boundaries.

**Dirichlet BC (ψ=0 at edges)** - Previously used, absorbs energy on reflection. Not recommended for long simulations.

### Additive vs Overwrite Chargers

**SOLVED: Additive chargers now implemented.**

The original wall chargers used overwrite mode (`ψ = A·cos(ωt)`), which acted as an energy sink when propagated wave amplitude exceeded charger amplitude. This caused instability.

**Additive chargers** (`ψ += A·cos(ωt)`) solve this by:

1. Never removing energy - only adding
2. Working like synchronized pushes to a flywheel
3. Adding timed momentum in the direction of wave motion
4. Allowing natural wave superposition

This is analogous to pushing a child on a swing - you add energy in sync with the motion, not fight against it.

### Wall Charger Placement

Wall chargers are placed 1 voxel interior from boundaries to avoid conflict with Neumann BC ghost cell updates:

- Charger indices: `i=1, nx-2`, `j=1, ny-2`, `k=1, nz-2`
- Boundary indices: `i=0, nx-1`, `j=0, ny-1`, `k=0, nz-1` (ghost cells)

### Remaining Energy Drift Sources

With Neumann BC and additive chargers, remaining drift sources are minor:

1. **Numerical Dispersion**: The discrete 6-point Laplacian stencil is 2nd-order accurate. High-frequency wave components near the Nyquist limit travel at incorrect speeds.

2. **Float32 Precision**: Using `ti.f32` fields provides ~7 significant digits. Accumulated rounding errors can cause slow energy drift.

### Current Status (December 2025 Testing)

**Architecture achieved:**

- Neumann BC for energy-conserving reflection
- Additive chargers (`ψ += delta`) instead of overwrite
- Wall chargers placed 1 voxel interior from boundaries
- Smooth envelope with cosine taper (no sharp transitions)
- Proportional damping above target

**Current envelope implementation:**

```python
TAPER_START = 0.70  # Start reducing power at 70%
TARGET = 1.00       # Target charge level
MIN_POWER = 0.1     # Power at target

if charge_level >= TARGET:
    return 0.0  # At/above target: no charging
elif charge_level >= TAPER_START:
    # Cosine taper from 1.0 to MIN_POWER
    t = (charge_level - TAPER_START) / (TARGET - TAPER_START)
    return MIN_POWER + (1.0 - MIN_POWER) * 0.5 * (1.0 + ti.math.cos(ti.math.pi * t))
else:
    return 1.0  # Full power
```

**Current damping implementation:**

```python
BASELINE = 0.99998  # Very light: ~0.002% per step

if charge_level <= target:
    return BASELINE
else:
    overshoot = (charge_level - target) / tolerance
    overshoot = min(overshoot, 5.0)
    return BASELINE - 0.002 * overshoot
```

### Wall Charger Phase Synchronization (NEW)

**Problem identified:** All 6 walls oscillating with same phase creates DC pumping effect, causing asymmetric displacement (biased positive or negative instead of centered on zero).

**Implemented solution:** Spatial phase shift between opposite walls based on domain size vs wavelength:

```python
# Phase shift = 2π × (domain_voxels / voxels_per_wavelength)
phase_shift_x = 2.0 * ti.math.pi * (wave_field.nx - 2) / wave_field.ewave_res
phase_shift_y = 2.0 * ti.math.pi * (wave_field.ny - 2) / wave_field.ewave_res
phase_shift_z = 2.0 * ti.math.pi * (wave_field.nz - 2) / wave_field.ewave_res

# Low walls: base phase, High walls: shifted phase + inverted sign
osc_x_lo = amp * ti.cos(phase)
osc_x_hi = -amp * ti.cos(phase + phase_shift_x)  # Inverted for push from opposite direction
```

**Rationale:**

- Phase shift ensures waves from opposite walls are coherent (arrive in sync)
- Sign inversion ensures zero net DC bias (when one wall pushes, opposite pulls)
- Together: symmetric oscillation centered on zero displacement

## BREAKTHROUGH: Minimal Intervention Strategy (December 202)

### Key Discovery

After extensive testing of dynamic chargers and damping, the most stable configuration is:

Static single-pulse charge + Dirichlet BC + NO dynamic manipulation

### What Works Best

1. **Static radial pulse** (`charge_full` with boost=1.0) - Fills domain with spherical wave pattern
2. **Dirichlet BC** (ψ=0 at edges) - More stable than Neumann for energy conservation
3. **NO dynamic wall chargers** - They disturb more than stabilize
4. **NO baseline damping** - System naturally conserves energy
5. **Let waves dilute naturally** - Reflections create homogeneous field over time

### Verified Findings

| Component | Finding | Impact |
|-----------|---------|--------|
| Leap-Frog integrator | Energy conserving (symplectic) | No drift |
| 6-point Laplacian (2nd order) | Sufficient accuracy | No need for 4th order |
| Float32 precision | No measurable drift | Memory efficient |
| Dirichlet BC | Better stability than Neumann | Zero energy loss |
| Static pulse only | Best stability | 55-60 FPS (5 FPS gain) |

### Current Performance (December 2025)

- **Charge level**: 80-120% range (±20% from target)
- **Frequency**: Stable at ~0.0105 rHz with periodic spikes
- **Displacement**: Symmetric around zero
- **No energy drift**: System maintains energy indefinitely
- **FPS**: 55-60 (improved from 50 without dynamic overhead)

### Why Dynamic Chargers Failed

1. **Wall chargers create plane waves** - Not natural, causes interference patterns
2. **Phase synchronization issues** - All walls in phase creates DC pumping
3. **Charger vs damping fight** - Competing forces create oscillations
4. **Additional computational cost** - Runtime if statements and function calls

### Why Static Pulse Works

1. **Radial waves are natural** - Like Big Bang, energy expands spherically
2. **Single coherent wavefront** - No phase conflicts
3. **Natural dilution** - Reflections homogenize energy distribution
4. **No runtime overhead** - One-time initialization cost only

### Frequency Spikes Analysis

Frequency spikes correlate with RMS amplitude dips (wave beating). When constructive/destructive interference causes local amplitude minimum, zero-crossing detection becomes noisy.

**Mitigation options:**

1. Increase EMA smoothing factor for frequency tracking
2. Filter out measurements during low-amplitude periods
3. Use longer averaging window

## Next Steps to Test

### Priority 1: Reduce Charge Level Variation (80-120% → 90-110%)

1. **Test wall charger with sparse sources (6-8 per wall)**
   - Creates quasi-spherical waves instead of plane waves
   - May blend better with static radial pulse

2. **Improve `charge_full` function**
   - Test different initial amplitude profiles
   - Gaussian envelope instead of sharp cosine
   - Multiple smaller pulses at different locations

3. **Fractal/chaos-based amplitude variation**
   - Non-periodic amplitude modulation
   - Prevents standing wave resonances
   - Inspired by fractal audio technology

### Priority 2: Frequency Spike Mitigation

1. **Investigate spike timing** - Correlate with wave beating period
2. **Increase frequency EMA alpha** - Slower response, smoother curve
3. **Threshold-based filtering** - Ignore measurements when amplitude < threshold

### Priority 3: Long-term Stability Verification

1. **Run 10,000+ timesteps** - Verify no long-term drift
2. **Run 50,000+ timesteps** - Stress test energy conservation
3. **Monitor memory usage** - Check for leaks over long runs

### Priority 4: Higher-Order Laplacian (TESTED - December 2025)

| Stencil | Points | Accuracy | Memory Reads | Actual FPS | Long-term Stability |
|---------|--------|----------|--------------|------------|---------------------|
| 6-point (current) | 7 | 2nd order | 7 per voxel | 56 FPS | ✅ Excellent |
| 18-point | 19 | 4th order | 19 per voxel | 56 FPS | ❌ Energy drift |

**18-point Laplacian implementation (tested):**

```python
@ti.func
def compute_laplacian18(wave_field, i, j, k):
    """
    18-connectivity: 6 face neighbors + 12 edge neighbors.
    Formula: ∇²ψ ≈ (face_sum + 0.5·edge_sum - 12·center) / (3·dx²)
    Requires 2-cell buffer from boundaries (vs 1-cell for 6-point).
    """
    face_sum = (ψ[i±1,j,k] + ψ[i,j±1,k] + ψ[i,j,k±1])  # 6 neighbors
    edge_sum = (ψ[i±1,j±1,k] + ψ[i±1,j,k±1] + ψ[i,j±1,k±1])  # 12 neighbors
    return (face_sum + 0.5 * edge_sum - 12.0 * center) / (3.0 * dx²)
```

**Test Results (100M and 1M voxel grids):**

| Metric | 6-point | 18-point | Winner |
|--------|---------|----------|--------|
| FPS | 56 | 56 | Tie |
| Charge stability (3000 steps) | 90-115% | 100-140% | 6-point |
| Frequency stability | 0.0105-0.0125 rHz | 0.010-0.016 rHz (drift) | 6-point |
| Displacement range | ±3-4 am | ±6 am (growing) | 6-point |
| Long-term (6000+ steps) | Bounded | Energy accumulation | 6-point |

**Key Findings:**

1. **No performance penalty** - Both stencils run at 56 FPS (GPU not memory-bound at 100M voxels)
2. **18-point amplifies wave interactions** - Better isotropy captures more energy transfer
3. **Energy accumulation** - 18-point charge level climbs to 140% and stays elevated
4. **Frequency drift** - 18-point shows upward frequency drift indicating numerical instability

**Conclusion:** The 6-point Laplacian provides better long-term stability with zero performance cost for using 18-point. The 2nd-order accuracy is sufficient for wave propagation physics. Higher-order stencils are NOT recommended.

### Priority 5: NOT Recommended

1. **Float64 fields** - 2x memory cost, near segfault limit already
2. **Complex dynamic charger logic** - Adds overhead, marginal benefit
3. **Neumann BC** - Less stable than Dirichlet in our tests

## Successful Patterns Summary

| Pattern | Status | Notes |
|---------|--------|-------|
| Static radial pulse | ✅ Works | Best stability |
| Dirichlet BC | ✅ Works | Better than Neumann |
| 6-point Laplacian | ✅ Works | Sufficient accuracy |
| Float32 precision | ✅ Works | No drift detected |
| No dynamic chargers | ✅ Works | Cleaner, faster |
| No runtime damping | ✅ Works | Natural equilibrium |

## Failed Patterns Summary

| Pattern | Status | Notes |
|---------|--------|-------|
| Dynamic wall chargers | ❌ Failed | Creates interference |
| Baseline damping always | ❌ Failed | Drains energy |
| Neumann BC | ⚠️ Marginal | Less stable |
| Phase-shifted wall chargers | ❌ Failed | Complex, no benefit |
| Peak-reached hysteresis | ❌ Failed | Mode switch creates transients |
| 18-point Laplacian | ❌ Failed | Energy accumulation, frequency drift |

## References

- `L1_wave_engine.py`: Charging and propagation implementations
- `L1_launcher.py`: Simulation control loop
- `_L1_instrumentation.py`: Diagnostic plotting
- `10_STABILITY_ANALYSIS.md`: CFL condition and numerical stability
