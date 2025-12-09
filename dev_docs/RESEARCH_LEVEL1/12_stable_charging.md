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
1.0 |        ___________
    |       /           \
    |      /             \
    |     /               \
0.0 |____/                 \____
    +----------------------------> charge_level
        0%   50%   100%  120%
         ^    ^      ^     ^
         |    |      |     |
       start full   fade  stop
```

### Envelope Function Implementation

Uses `ti.math` (already imported via Taichi) for math operations:

```python
def compute_charge_envelope(charge_level: float) -> float:
    """
    Compute smooth charging envelope based on current charge level.

    Phases:
    1. Ramp-up (0% -> 50%): Linear increase from 0 to 1
    2. Full power (50% -> 90%): Constant at 1.0
    3. Taper (90% -> 100%): Smooth cosine fade to 0
    4. Off (>100%): No charging

    Args:
        charge_level: Current energy as fraction of nominal (0.0 to 1.5+)

    Returns:
        float: Envelope value 0.0 to 1.0
    """
    if charge_level < 0.50:
        # Phase 1: Ramp-up - linear from 0 to 1
        return charge_level / 0.50
    elif charge_level < 0.90:
        # Phase 2: Full power
        return 1.0
    elif charge_level < 1.00:
        # Phase 3: Taper - smooth cosine fade
        # Maps 0.90->1.00 to 1.0->0.0 using cosine for smooth derivative
        t = (charge_level - 0.90) / 0.10  # 0 to 1
        return 0.5 * (1.0 + ti.math.cos(ti.math.pi * t))  # 1.0 to 0.0
    else:
        # Phase 4: Off
        return 0.0
```

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
    - Above target + tolerance (110%): mild damping
    - At 2x tolerance (120%): stronger damping

    Args:
        charge_level: Current energy as fraction of nominal
        target: Target charge level (default 1.0 = 100%)
        tolerance: Tolerance band width (default 0.10 = 10%)

    Returns:
        float: Damping factor 0.999 to 1.0
    """
    if charge_level <= target:
        return 1.0  # No damping below target

    # Proportional damping above target
    overshoot = (charge_level - target) / tolerance  # 0 at 100%, 1 at 110%, 2 at 120%
    overshoot = min(overshoot, 2.0)  # Cap at 2x for stability

    # Damping increases with overshoot: 1.0 -> 0.9995 -> 0.999
    return 1.0 - 0.0005 * overshoot
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
# Charging parameters
BASE_BOOST = 15.0           # Moderate amplitude (was 100)
SOURCES_PER_EDGE = N // 6   # Dense source grid (was 7)

# Envelope thresholds
RAMP_END = 0.50             # End of ramp-up phase
TAPER_START = 0.90          # Start of taper phase
CHARGE_TARGET = 1.00        # Full charge target

# Damping parameters
DAMP_TOLERANCE = 0.10       # Start damping at 110%
DAMP_MAX_FACTOR = 0.999     # Maximum damping strength
```

### Expected Behavior

1. **Timesteps 0-500**: Ramp-up phase, charge increases linearly
1. **Timesteps 500-1500**: Full power phase, charge approaches 90%
1. **Timesteps 1500-2000**: Taper phase, charging fades as charge approaches 100%
1. **Timesteps 2000+**: Stable phase, minor damping maintains equilibrium

### Success Criteria

- Reach 100% +/- 10% charge by timestep 2000
- No overshoot beyond 120%
- Probe amplitude variation < 20% of mean
- Probe frequency constant within 5% of nominal, no spikes
- No beating patterns or interference fringes
- Avoid chaos, promote stability

## References

- `L1_wave_engine.py`: Charging and damping implementations
- `L1_launcher.py`: Simulation control loop
- `_L1_instrumentation.py`: Diagnostic plotting
- `10_STABILITY_ANALYSIS.md`: CFL condition and numerical stability
