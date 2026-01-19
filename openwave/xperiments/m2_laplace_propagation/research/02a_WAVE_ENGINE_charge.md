# Initial Energy Charging

## Table of Contents

1. [Match EWT Energy Equation](#match-ewt-energy-equation)
1. [Charge Energy Wave](#charge-energy-wave)
1. [Implementation - Option 1: Uniform Energy Density (Simplest)](#implementation---option-1-uniform-energy-density-simplest)
1. [Implementation - Option 2: Spherical Gaussian Wave Pulse (Recommended)](#implementation---option-2-spherical-gaussian-wave-pulse-recommended)
1. [Implementation - Option 3: Wolff's Spherical Wave (For Future Particle Implementation)](#implementation---option-3-wolffs-spherical-wave-for-future-particle-implementation)
1. [Usage Example (Implementing Phase 1: Center-Concentrated Pulse)](#usage-example-implementing-phase-1-center-concentrated-pulse)
1. [Advanced Technique: Multiple Pulses for Precision](#advanced-technique-multiple-pulses-for-precision)
1. [Recommendation](#recommendation)
1. [Wave Evolution](#wave-evolution)

## Match EWT Energy Equation

**Context**: When initializing the wave field, we need to charge it with the correct amount of energy as specified by the EWT energy wave equation from `equations.py`.

**EWT Energy Wave Equation** (wavelength-based form):

```python
E = ρV(c/λ × A)²
```

**Frequency-centric equivalent**:

```python
E = ρV(fA)²    # Since f = c/λ
```

**Critical Requirements**:

1. **Match Total Energy**: Initial field energy must equal `compute_energy_wave_equation(volume)` from equations.py
2. **Correct Wave Characteristics**: Use proper frequency, amplitude, wavelength from constants
3. **Simple Initial Condition**: DON'T try to create particle standing waves yet - those emerge automatically later
4. **Energy Conservation**: Wave equation will maintain total energy during propagation

## Charge Energy Wave

- injection of n (even) pulses with universe energy (eq)
  - phase 0, time t: max positive displacement
  - phase π, time: t + 1 wave period: min negative displacement
- vs. start with a sine-wave of displacement (wave driver)
  - that will be picked up by stencil voxels

- hard coded point source or multiple sources (fast charge)
  - direction = spherical
- UI button, counter til stable

**Standing Wave Particles Come Later**: The IN/OUT wave reflections from wave centers (reflective voxels) will automatically create steady-state standing waves (fundamental particles) when we implement particle wave centers. For now, just inject energy with correct wave properties.

## Implementation - Option 1: Uniform Energy Density (Simplest)

```python
@ti.kernel
def charge_uniform_energy(self):
    """
    Initialize field with uniform energy density matching EWT equation.

    This is the simplest initial condition - just charge the field uniformly
    with the correct total energy from equations.compute_energy_wave_equation().

    Wave propagation will then naturally evolve this initial state.
    Standing waves will emerge when wave centers (reflective voxels) are added.
    """
    # EWT constants
    ρ = ti.f32(constants.MEDIUM_DENSITY)
    f = ti.f32(constants.EWAVE_FREQUENCY)
    A = ti.f32(constants.EWAVE_AMPLITUDE)

    # Uniform initial displacement (all voxels same)
    # Use small random perturbation to avoid perfect symmetry
    import random
    base_displacement = A / 2.0  # Half amplitude to start

    for i, j, k in self.psiL_am:
        # Small random perturbation (±10%) to break symmetry
        perturbation = 1.0 + 0.1 * (random.random() - 0.5)
        displacement = base_displacement * perturbation

        self.psiL_am[i, j, k] = displacement / constants.ATTOMETER
        self.amp_local_peak_am[i, j, k] = ti.abs(self.psiL_am[i, j, k])

    # Initialize old displacement (same as current for stationary start)
    for i, j, k in self.displacement_old:
        self.displacement_old[i, j, k] = self.psiL_am[i, j, k]

    # Verify total energy matches equations.compute_energy_wave_equation()
    # E_total = ρV(fA)² where V = nx × ny × nz × dx³
```

## Implementation - Option 2: Spherical Gaussian Wave Pulse (Recommended)

```text
ψ(r) = A·e^(-r²)   [Gaussian bump]
```

```python
@ti.kernel
def charge_spherical_gaussian(
    self,
    center: ti.math.vec3,      # Wave center position (meters)
    total_energy: ti.f32,       # Total energy to inject (Joules)
    width_factor: ti.f32 = 3.0  # Width as multiple of wavelength
):
    """
    Initialize field with center-concentrated spherical Gaussian pulse.

    IMPLEMENTS PHASE 1 OF ENERGY EVOLUTION SEQUENCE:
    - Single smooth pulse concentrated at universe center
    - Total energy exactly matches equations.compute_energy_wave_equation(volume)
    - Will propagate outward, reflect off boundaries, and dilute (Phases 2-4)
    - After stabilization, wave centers can be inserted (Phase 5)

    This does NOT create particle standing waves - those emerge automatically
    later when wave centers (reflective voxels) are inserted.

    Args:
        center: Pulse center position in meters (typically universe center)
        total_energy: Total energy from equations.compute_energy_wave_equation(volume)
        width_factor: Pulse width = width_factor × wavelength (default: 3.0)
                     Smaller width = more concentrated pulse
                     Larger width = smoother, more spread out
    """
    # Convert to scaled units
    center_am = center / constants.ATTOMETER

    # EWT constants
    ρ = ti.f32(constants.MEDIUM_DENSITY)
    f = ti.f32(constants.EWAVE_FREQUENCY)
    λ_am = ti.f32(constants.EWAVE_LENGTH / constants.ATTOMETER)

    # Gaussian width
    σ_am = width_factor * λ_am  # Width in attometers

    # Calculate amplitude to match desired total energy
    # E = ∫ ρ(fA)² dV for Gaussian: E ≈ ρ(fA)² × (π^(3/2) × σ³)
    # Solve for A: A = √(E / (ρf² × π^(3/2) × σ³))
    volume_factor = (ti.math.pi ** 1.5) * (σ_am * constants.ATTOMETER) ** 3
    A_required = ti.sqrt(total_energy / (ρ * f * f * volume_factor))
    A_am = A_required / constants.ATTOMETER

    # Apply Gaussian wave packet
    for i, j, k_idx in self.psiL_am:
        pos_am = self.get_position_am(i, j, k_idx)
        r_vec = pos_am - center_am
        r_squared = r_vec.dot(r_vec)

        # Gaussian envelope: exp(-r²/(2σ²))
        gaussian = ti.exp(-r_squared / (2.0 * σ_am * σ_am))

        # Initial displacement with Gaussian envelope
        displacement = A_am * gaussian

        self.psiL_am[i, j, k_idx] = displacement
        self.amp_local_peak_am[i, j, k_idx] = ti.abs(displacement)

    # Initialize old displacement (same as current for stationary start)
    for i, j, k_idx in self.displacement_old:
        self.displacement_old[i, j, k_idx] = self.psiL_am[i, j, k_idx]
```

## Implementation - Option 3: Wolff's Spherical Wave (For Future Particle Implementation)

```text
ψ(r,t) = A·e^(iωt)·sin(kr)/r   [Wolff's bump]
ψ(r,t) = A·e^i(ωt±kr)/r       [DeBroglie Wave]
```

```python
@ti.kernel
def charge_wolff_spherical_wave(
    self,
    center: ti.math.vec3,
    frequency: ti.f32,
    amplitude: ti.f32,
    initial_phase: ti.f32 = 0.0
):
    """
    Initialize using Wolff's analytical solution: Φ = Φ₀ e^(iωt) sin(kr)/r

    USE THIS LATER when implementing wave centers (reflective voxels).
    This creates the sin(kr)/r pattern that will become a standing wave
    when IN and OUT waves interfere.

    For now, use simpler Gaussian (Option 2) for initial charging.
    """
    center_am = center / constants.ATTOMETER
    amp_local_peak_am = amplitude / constants.ATTOMETER
    k = 2.0 * ti.math.pi * frequency / constants.EWAVE_SPEED

    for i, j, k_idx in self.psiL_am:
        pos_am = self.get_position_am(i, j, k_idx)
        r_vec = pos_am - center_am
        r = r_vec.norm()

        # sin(kr)/r pattern (finite at r=0: lim = k)
        if r < 0.01:
            spatial_factor = k
        else:
            kr = k * r * constants.ATTOMETER
            spatial_factor = ti.sin(kr) / (r * constants.ATTOMETER)

        wave_displacement = amp_local_peak_am * ti.cos(initial_phase) * spatial_factor

        self.psiL_am[i, j, k_idx] = wave_displacement
        self.amp_local_peak_am[i, j, k_idx] = ti.abs(wave_displacement)

    for i, j, k_idx in self.displacement_old:
        self.displacement_old[i, j, k_idx] = self.psiL_am[i, j, k_idx]
```

## Usage Example (Implementing Phase 1: Center-Concentrated Pulse)

```python
from openwave.common import constants, equations
import taichi as ti

# Calculate universe volume (meters³)
universe_volume = wave_field.actual_universe_size[0] * \
                  wave_field.actual_universe_size[1] * \
                  wave_field.actual_universe_size[2]

# Get correct total energy from EWT equation (Phase 1)
total_energy = equations.compute_energy_wave_equation(
    volume=universe_volume,
    density=constants.MEDIUM_DENSITY,
    speed=constants.EWAVE_SPEED,
    wavelength=constants.EWAVE_LENGTH,
    amplitude=constants.EWAVE_AMPLITUDE
)

# Calculate universe center position
side_length = universe_volume ** (1/3)  # Assuming cubic universe
center_position = ti.Vector([side_length / 2.0] * 3)  # meters

# PHASE 1: Inject center-concentrated pulse with exact EWT energy
# This is a single pulse (or few pulses) that will propagate outward
wave_field.charge_spherical_gaussian(
    center=center_position,           # Universe center
    total_energy=total_energy,        # Exact EWT energy amount
    width_factor=3.0                  # Pulse width = 3× wavelength
)

# Verify energy matches EWT equation
measured_energy = wave_field.compute_total_energy()
energy_match_percent = abs(measured_energy - total_energy) / total_energy * 100

print(f"=== Initial Energy Charging (Phase 1) ===")
print(f"Universe volume: {universe_volume:.2e} m³")
print(f"Target energy (EWT): {total_energy:.2e} J")
print(f"Measured energy: {measured_energy:.2e} J")
print(f"Energy match: {energy_match_percent:.2f}%")
print(f"\nPulse centered at: {center_position} m")
print(f"Pulse width: {3.0 * constants.EWAVE_LENGTH:.2e} m")

# PHASES 2-4 will happen automatically during simulation:
# - Wave propagates outward (Phase 2)
# - Reflects off boundaries (Phase 3)
# - Dilutes into stable state (Phase 4)

# Run simulation to allow energy distribution
# After energy stabilizes, we'll implement Phase 5 (wave center insertion)
```

## Advanced Technique: Multiple Pulses for Precision

For more precise energy control, you can inject a few successive pulses:

```python
# Option A: Single large pulse (simple, recommended)
wave_field.charge_spherical_gaussian(
    center=center_position,
    total_energy=total_energy,
    width_factor=3.0
)

# Option B: Multiple smaller pulses (higher precision)
# Useful if single pulse causes numerical instability
num_pulses = 3
energy_per_pulse = total_energy / num_pulses

for pulse_idx in range(num_pulses):
    # Add each pulse with small time delay
    wave_field.charge_spherical_gaussian(
        center=center_position,
        total_energy=energy_per_pulse,
        width_factor=3.0
    )
    # Run a few timesteps between pulses to let energy spread
    for _ in range(10):
        wave_field.propagate_wave(dt)
```

Multiple pulses can provide:

- Better numerical stability (smaller amplitude changes per timestep)
- More gradual energy injection
- Finer control over energy distribution

However, single pulse is usually sufficient and simpler.

## Recommendation

- **Now**: Use **Option 2 (Spherical Gaussian)** - simple, smooth, energy-conserving
- **Single vs Multiple Pulses**: Start with single pulse; use multiple only if needed for stability
- **Later**: Use **Option 3 (Wolff's sin(kr)/r)** when implementing wave centers and particle formation
- **Option 1**: Only for testing wave equation stability

## Wave Evolution

After initial energy charge, the wave field requires a **stabilization period** before reaching steady-state dynamics.

**Stabilization Process**:

1. **Initial State** (t = 0):
   - Energy concentrated at charge points/regions
   - Clear propagation direction from sources
   - No interference patterns yet

2. **Early Propagation** (t = 0 to t_stab):
   - Waves propagate outward from sources at speed c
   - First reflections from boundaries
   - Initial interference patterns begin to form

3. **Stabilization Time Calculation**:

   ```python
   # Time for wave to cross domain and reflect back
   domain_size = max(nx*dx, ny*dx, nz*dx)
   t_stabilization = 2 * domain_size / c  # Round trip time

   # Run simulation without particle updates during stabilization
   for step in range(int(t_stabilization / dt)):
       propagate_wave_field(dt)  # No particle forces yet
   ```

4. **Steady-State** (t > t_stab):
   - Waves coming from all directions (not just sources)
   - Complex interference patterns throughout domain
   - Energy density distributed (not localized)
   - Dynamic equilibrium (total energy constant)

**Why Stabilization is Needed**:

- Prevents artificial transients from affecting particle motion
- Allows interference patterns to form naturally
- Ensures energy distribution reaches quasi-equilibrium
- Provides realistic initial conditions for particle dynamics

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
        E_p = 0.5 * displacement[i,j,k]**2
        total += (E_k + E_p) * dx**3  # Energy per voxel
    return total

# Should match E_total_Charged (within numerical tolerance)
assert abs(verify_energy_conservation() - E_total_Charge) < tolerance
```
