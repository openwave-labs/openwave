# The Classical Wave Equation

## Table of Contents

1. [Laplacian Operator (How Voxels Share Displacement)](#laplacian-operator-how-voxels-share-displacement)
1. [Time Evolution Implementation](#time-evolution-implementation)
1. [Timestep Strategy: Fixed vs Elapsed Time](#timestep-strategy-fixed-vs-elapsed-time)
1. [Why Leap-Frog instead of Euler Integration Method](#why-leap-frog-instead-of-euler-integration-method)
1. [Alternative: Huygens Wavelets](#alternative-huygens-wavelets)
1. [Choosing Between PDE and Huygens](#choosing-between-pde-and-huygens)
   - [Should We Choose One or Use Both?](#should-we-choose-one-or-use-both)
   - [Comparison Table](#comparison-table)
   - [Performance Analysis](#performance-analysis)
   - [Pros & Cons Summary](#pros--cons-summary)
   - [Recommended](#recommended)
1. [Key Physics Principles](#key-physics-principles)
   - [Energy Conservation](#energy-conservation)
   - [Amplitude Dilution](#amplitude-dilution)

LEVEL-1 uses **PDEs (Partial Differential Equations)** to propagate waves through the field.

**3D Wave Equation** (fundamental):

```text
∂²ψ/∂t² = c²∇²ψ

or simplified as:
ψ̈ = c²∇²ψ
```

Where:

- `ψ` = wave displacement field (scalar)
- `c` = wave propagation speed (speed of light, 2.998×10⁸ m/s)
- `∇²ψ` = Laplacian operator (second-order spatial derivative, laplacian of psi)
- `∂²ψ/∂t²` = second-order time derivative (acceleration of displacement, psi double-prime)

**Physical Interpretation**:

- Left side: How fast displacement is accelerating in time
- Right side: How much displacement differs from neighbors (curvature)
- Equation says: "Displacement accelerates toward its neighbors' average"

## Laplacian Operator (How Voxels Share Displacement)

**Discrete Laplacian** (6-connectivity, face neighbors only):

```python
# Laplacian at voxel [i,j,k]
∇²ψ[i,j,k] = (
    ψ[i+1,j,k] + ψ[i-1,j,k] +  # Left/right neighbors (x-direction)
    ψ[i,j+1,k] + ψ[i,j-1,k] +  # Front/back neighbors (y-direction)
    ψ[i,j,k+1] + ψ[i,j,k-1] -  # Top/bottom neighbors (z-direction)
    6 × ψ[i,j,k]                # Central voxel (6 neighbors)
) / dx²
```

**Physical Meaning**:

- Laplacian measures how much a voxel's displacement differs from its neighbors' average
- Positive Laplacian: voxel lower than average → displacement will increase
- Negative Laplacian: voxel higher than average → displacement will decrease
- This drives wave propagation: differences smooth out over time

## Time Evolution Implementation

```python
@ti.func
def compute_laplacian_am(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.f32:
    """
    Compute Laplacian ∇²ψ at voxel [i,j,k] (6-connectivity).
    ∇²ψ = (∂²ψ/∂x² + ∂²ψ/∂y² + ∂²ψ/∂z²)

    Discrete Laplacian (second derivative in space):
    ∇²ψ[i,j,k] = (ψ[i±1] + ψ[i,j±1] + ψ[i,j,k±1] - 6ψ[i,j,k]) / dx²

    Args:
        i, j, k: Voxel indices (must be interior: 0 < i,j,k < n-1)

    Returns:
        Laplacian in units [1/am] = [am/am²]
    """
    # 6-connectivity stencil (face neighbors only)
    laplacian_am = (
        self.displacement_am[i+1, j, k] + self.displacement_am[i-1, j, k] +
        self.displacement_am[i, j+1, k] + self.displacement_am[i, j-1, k] +
        self.displacement_am[i, j, k+1] + self.displacement_am[i, j, k-1] -
        6.0 * self.displacement_am[i, j, k]
    ) / (self.dx_am * self.dx_am)

    return laplacian_am
```

```python
@ti.kernel
def propagate_wave(self, dt: ti.f32, SIM_SPEED: ti.f32):
    """
    Propagate wave displacement using wave equation (PDE Solver).

    Wave Equation: ∂²ψ/∂t² = c²∇²ψ

    Discrete Form (Leap-Frog/Verlet):
        ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
        where ∇²ψ = (neighbors_sum - 6·center) / dx²

    Args:
        dt: Timestep in seconds. Typical: ~1/60 s (60 FPS frame time)
        SIM_SPEED: Wave speed multiplier for visualization. Default: 1.0

    CFL Stability:
        Condition: dt ≤ dx / (c·√3) for 3D 6-connectivity

        Problem: Real wave speed c = 3×10⁸ m/s requires dt_max ~ 1.2e-26 s,
        but frame time dt ~ 0.016 s violates CFL by ~10²⁴×.

        Solution: Slow wave speed instead of shrinking timestep.
            c_slo = (c / SLO_MO) × SIM_SPEED
            With SLO_MO = 1.05×10²⁵: dt_critical ≈ 0.121 s > dt_frame ✓ STABLE
    """
    # Speed of light (apply SLO_MO factor, then SIM_SPEED for human-visible waves)
    c_slo = ti.f32(constants.EWAVE_SPEED / config.SLO_MO) * SIM_SPEED  # m/s

    # Convert c to attometers/second for consistent units
    c_am = c_slo / constants.ATTOMETER  # am/s

    # CFL stability condition: cfl_factor ≤ 1/3 for 3D (6-connectivity)
    # cfl_factor = (c_am·dt/dx_am)² [dimensionless]
    # Units: [(am/s)·s / am]² = dimensionless ✓
    cfl_factor = (c_am * dt / dx_am)**2

    # Update all interior voxels (boundaries stay at ψ=0)
    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Compute Laplacian (returns [1/am])
            laplacian_am = self.compute_laplacian_am(i, j, k)

            # Leap-frog update
            # Standard form: ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
            # Units check:
            # ψ: [am]
            # (c_am·dt)²·∇²ψ: [am²]·[1/am] = [am] ✓
            # Result: [am] = [am] - [am] + [am] ✓
            self.displacement_new_am[i, j, k] = (
                2.0 * self.displacement_am[i, j, k]
                - self.displacement_old_am[i, j, k]
                + (c_am * dt)**2 * laplacian_am
            )

    # Swap time levels for next iteration
    # Python tuple swap: (old, current, new) ← (current, new, old)
    self.displacement_old_am, self.displacement_am, self.displacement_new_am = \
        self.displacement_am, self.displacement_new_am, self.displacement_old_am
```

```python
@ti.kernel
def track_amplitude_envelope(self):
    """
    Track amplitude envelope by computing running maximum of |ψ|.

    Amplitude A is the envelope of the high-frequency displacement oscillation.
    Uses ti.atomic_max for thread-safe updates in parallel execution.
    """
    for i, j, k in self.displacement_am:
        disp_mag = ti.abs(self.displacement_am[i,j,k])
        ti.atomic_max(self.amplitude_am[i,j,k], disp_mag)
```

```python
@ti.kernel
def compute_wave_direction(self):
    """
    Compute wave propagation direction from energy flux.

    Energy flux: S = -c² × ψ × ∇ψ
    Direction: normalized S
    """
    c = ti.f32(constants.EWAVE_SPEED)

    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Current displacement
            psi = self.displacement_am[i,j,k]

            # Displacement gradient
            grad_x = (self.displacement_am[i+1,j,k] - self.displacement_am[i-1,j,k]) / (2.0 * self.dx_am)
            grad_y = (self.displacement_am[i,j+1,k] - self.displacement_am[i,j-1,k]) / (2.0 * self.dx_am)
            grad_z = (self.displacement_am[i,j,k+1] - self.displacement_am[i,j,k-1]) / (2.0 * self.dx_am)

            grad_psi = ti.Vector([grad_x, grad_y, grad_z])

            # Energy flux vector
            S = -c**2 * psi * grad_psi

            # Normalize to get direction
            S_mag = S.norm()
            if S_mag > 1e-12:  # Avoid division by zero
                self.wave_direction[i,j,k] = S / S_mag
            else:
                self.wave_direction[i,j,k] = ti.Vector([0.0, 0.0, 0.0])
```

```python
def update_timestep(self, dt: ti.f32, SIM_SPEED: ti.f32):
    """
    Complete wave field update for one timestep.

    Args:
        dt: Timestep in seconds (with SLO_MO factor applied)
        SIM_SPEED: Frequency multiplier (applied after SLO_MO)
    """
    # 1. Propagate wave displacement
    self.propagate_wave(dt, SIM_SPEED)

    # 2. Track amplitude envelope
    self.track_amplitude_envelope()

    # 3. Compute wave direction
    self.compute_wave_direction()

    # 4. Apply boundary conditions (handled by not updating boundaries in propagate)
```

**Summary of Merged Implementation**:

- ✅ **Single `propagate_wave(dt, SIM_SPEED)` function**
- ✅ **Encapsulated `compute_laplacian_am()`** as `@ti.func` returning full laplacian [1/am]
- ✅ **Correct dimensional analysis**: All units in attometers (consistent throughout)
- ✅ **SLO_MO factor** applied to wave speed `c` (slows simulation ~10²⁵× for human visibility)
- ✅ **SIM_SPEED parameter**: Optional frequency multiplier (like LEVEL-0's `oscillate_granules()`)
- ✅ **CFL stability maintained** with effective wave speed c_slo = (c / SLO_MO) × SIM_SPEED
- ✅ **No rontosecond conversion needed**: dt already slowed by SLO_MO
- ✅ **Consistent units**: displacement_am [am], dx_am [am], c_am [am/s], dt [s]
- ✅ **60 FPS timestep**: dt ~ 0.016s (60Hz screen refresh rate)

**Key Formula**:

```python
ψ_new = 2ψ - ψ_old + (c_am·dt)² · ∇²ψ
```

Where:

- `c_slo = (EWAVE_SPEED / SLO_MO) × SIM_SPEED` (m/s, slowed + boosted)
- `c_am = c_slo / ATTOMETER` (wave speed in am/s)
- `dt` ~ 1/60 s (0.016 seconds for 60 FPS)
- `SIM_SPEED` ~ 1.0 (default, no boost) or higher for faster visualization
- `∇²ψ` in [1/am] units (from `compute_laplacian_am()`)
- `dx_am` in [am] units (standard voxel size)
- Result in [am] units ✓

**Dimensional Analysis**:

```text
(c_am·dt)²·∇²ψ = [(am/s)·s]² · [1/am] = [am²]·[1/am] = [am] ✓
```

**Storage Requirements**:

- Three displacement fields: `displacement_old`, `displacement_am` (current), `displacement_new`
- Needed for second-order time integration

**Stability Condition** (CFL - Courant-Friedrichs-Lewy):

```text
dt ≤ dx / (c√3)  for 3D, 6-connectivity

Example (6 fm³ universe, 1B voxels):
dx = 6 am = 6e-18 m
c = 2.998e8 m/s
dt_max = 6e-18 / (2.998e8 × √3) ≈ 1.2e-26 s
```

**This is extremely small!** However, LEVEL-1 doesn't use these tiny timesteps - instead, it slows the wave speed (see SLO_MO approach below).

## Timestep Strategy: Fixed vs Elapsed Time

**CRITICAL DECISION**: LEVEL-1 must use **fixed timesteps** (unlike LEVEL-0's elapsed time approach).

### Benefits of FIXED DT

- ✓ Physics accuracy: Consistent time sampling at regular intervals
- ✓ Smooth animations: Constant dt → no jitter in offline renders
- ✓ Reproducible: Same dt sequence → identical results every run
- ✓ CFL safe: With SLO_MO, dt=0.0167s < dt_critical=0.121s ✓
- ✓ Offline rendering: Perfect for hours-long background simulations
- ✓ Data collection: Regular time intervals for analysis

**Why LEVEL-0 Uses Elapsed Time**:

```python
# LEVEL-0 (particle-based, no CFL constraint)
elapsed_t = time.time() - previous_time  # Variable (0.001-0.1s typical)
update_particles(elapsed_t)  # Particles can handle variable dt
```

**Pros**: Real-time sync, adapts to frame rate, good for interactive visualization

**Cons**: Non-deterministic, timing depends on system performance

**Why LEVEL-1 CANNOT Use Elapsed Time**:

```python
# Wave equation CFL requirement (for dx = 6 am, 1B voxels)
dt_max = dx / (c√3) ≈ 1.2e-26 s  # MUST NOT EXCEED!

# But elapsed time is typically:
elapsed_t ≈ 0.001 to 0.1 s  # Frame time (milliseconds)

# Ratio: elapsed_t / dt_max ≈ 10^24
# Result: IMMEDIATE NUMERICAL EXPLOSION 💥
```

**The wave equation becomes unstable if dt > dt_max**. Using elapsed time would violate CFL by ~24 orders of magnitude!

**LEVEL-1 Solution: Fixed Timestep + Frame Accumulator**:

```python
# Fixed physics timestep (respects CFL)
dt_physics = 1.0e-26  # seconds (or 10 in rontoseconds)
dt_physics_rs = 10    # rontoseconds (scaled)

# Hybrid approach: decouple physics from rendering
accumulated_time = 0.0

def main_loop():
    previous_time = time.time()

    while running:
        # Measure elapsed real time
        current_time = time.time()
        elapsed_t = current_time - previous_time
        previous_time = current_time

        # Accumulate time for physics
        accumulated_time += elapsed_t

        # Run fixed timesteps until caught up
        while accumulated_time >= dt_physics:
            update_physics(dt_physics_rs)  # Fixed dt (rontoseconds)
            accumulated_time -= dt_physics

        # Render at variable rate (decoupled from physics)
        render_frame()
```

**Benefits of Fixed Timestep Approach**:

- ✓ **Guarantees CFL stability** (dt always ≤ dt_max)
- ✓ Deterministic results (reproducible simulations)
- ✓ Accurate physics regardless of frame rate
- ✓ Can run faster or slower than real-time
- ✓ Can save/replay exact simulation states

**Comparison Table**:

| Aspect | LEVEL-0 (Elapsed) | LEVEL-1 (Fixed) |
|--------|-------------------|-----------------|
| **Timestep** | Variable (frame-dependent) | Fixed (CFL-limited) |
| **Stability** | Robust to large dt | Requires dt ≤ dt_max |
| **Real-time sync** | Perfect | Approximate (via accumulator) |
| **Deterministic** | No (varies per run) | Yes (reproducible) |
| **Physics accuracy** | Euler integration (acceptable) | PDE solver (requires fixed dt) |
| **Use case** | Interactive particle systems | Scientific wave simulation |

**Recommendation**: LEVEL-1 **MUST** use fixed timesteps for numerical stability. The hybrid accumulator approach allows real-time rendering while maintaining stable physics.

## Why Leap-Frog instead of Euler Integration Method

why the wave equation uses the leap-frog scheme instead of regular Euler integration.

Wave Equation gives ψ acceleration directly

- Given: ∂²ψ/∂t² = c²∇²ψ

The acceleration of ψ is:

- ψ_acceleration = c² · ∇²ψ

So yes, the wave equation directly gives us the second derivative (acceleration) of displacement!

### Why Leap-Frog instead of Euler?

### Option 1: Euler Method (velocity-position split)

You could do it like particle motion:

```python
# Split into velocity and position
ψ_velocity[i,j,k] += ψ_acceleration * dt  # vel += a·dt
ψ[i,j,k] += ψ_velocity[i,j,k] * dt        # pos += vel·dt
```

Problems:

- ✗ Need to store velocity field (extra memory: 100³ voxels × f32)
- ✗ First-order accurate (error ∝ dt)
- ✗ Energy drift: Euler doesn't conserve energy in oscillating systems
- ✗ Waves gradually lose or gain energy over time (numerical artifact)

### Option 2: Leap-Frog (Verlet) - Direct from acceleration

```python
# Update displacement directly from acceleration (no velocity storage)
ψ_new = 2ψ - ψ_old + (c·dt)² · ∇²ψ
```

Advantages:

- ✓ No velocity field needed (saves memory)
- ✓ Second-order accurate (error ∝ dt²) - 100× more accurate!
- ✓ Energy conserving (symplectic integrator)
- ✓ Perfect for oscillating systems (waves, springs, orbits)
- ✓ One equation instead of two

### How Leap-Frog Works

The leap-frog derives from Taylor expansion of ψ(t):

- Forward:  ψ(t+dt) = ψ(t) + ψ'·dt + ½ψ''·dt² + ...
- Backward: ψ(t-dt) = ψ(t) - ψ'·dt + ½ψ''·dt² + ...

- Add them: ψ(t+dt) + ψ(t-dt) = 2ψ(t) + ψ''·dt² + O(dt⁴)
- Rearrange: ψ(t+dt) = 2ψ(t) - ψ(t-dt) + ψ''·dt²

Since ψ'' = c²∇²ψ:

- ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ

Key insight: The velocity terms ψ'·dt cancel out when you add forward and backward! This is why we don't need to store velocity.

Comparison Table

| Aspect    | Euler (vel+pos)                   | Leap-Frog (direct)         |
|-----------|-----------------------------------|----------------------------|
| Equations | 2 (vel, pos)                      | 1 (pos only)               |
| Memory    | 4 fields (ψ, ψ_old, vel, vel_old) | 3 fields (ψ, ψ_old, ψ_new) |
| Accuracy  | O(dt) first-order                 | O(dt²) second-order        |
| Energy    | Drifts over time                  | Conserved (symplectic)     |
| Best for  | Damped systems                    | Oscillating systems        |
| Speed     | Slower (2 updates)                | Faster (1 update)          |

### Why This Matters for Waves

Waves are oscillatory - they cycle energy between kinetic (∂ψ/∂t)² and potential (∇ψ)². Energy conservation is critical!

With Euler:

- E(t=0) = 100
- E(t=1000 steps) = 95  ❌ Lost 5% energy (numerical damping)

With Leap-Frog:

- E(t=0) = 100
- E(t=1000 steps) = 100.001  ✓ Nearly perfect conservation

### Bottom Line

The leap-frog scheme:

1. Uses the wave equation's acceleration directly: ψ'' = c²∇²ψ
2. Skips velocity storage by using ψ_old instead
3. Is 100× more accurate than Euler (second-order vs first-order)
4. Conserves energy for oscillating systems

That's why wave equations, springs, and orbital mechanics all use leap-frog/Verlet instead of Euler! 🎯

## Alternative: Huygens Wavelets

**Huygens' Principle**: Every point on a wavefront acts as a source of secondary wavelets.

**Conceptual Implementation**:

```python
@ti.kernel
def propagate_huygens(dt: ti.f32):
    """
    Propagate using Huygens wavelets.
    Each voxel emits wavelets to neighbors.

    Note: This is less commonly used for regular grids because
    the wave equation (PDE) naturally implements Huygens' principle.
    """
    c = ti.f32(constants.EWAVE_SPEED)
    propagation_distance = c * dt

    for i, j, k in self.displacement_am:
        if ti.abs(self.displacement_am[i,j,k]) > threshold:
            # This voxel emits wavelets to neighbors
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        if di == 0 and dj == 0 and dk == 0:
                            continue  # Skip self

                        # Neighbor indices
                        ni, nj, nk = i + di, j + dj, k + dk

                        # Boundary check
                        if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                            # Distance to neighbor
                            distance = ti.sqrt(ti.f32(di*di + dj*dj + dk*dk)) * self.dx_am

                            # Wavelet contribution (inverse distance weighting)
                            contribution = self.displacement_am[i,j,k] / distance

                            # Add to neighbor (superposition)
                            # Note: This is simplified, full implementation needs proper weighting
                            ti.atomic_add(self.amplitude_new[ni,nj,nk], contribution * dt)
```

**Note**: Huygens method is less efficient on regular grids. The wave equation (PDE) implicitly implements Huygens' principle through the Laplacian operator.

## Choosing Between PDE and Huygens

### Should We Choose One or Use Both?

**Recommendation**: Use **PDE (Wave Equation)** as the primary method.

**Reason**: The wave equation naturally implements Huygens' principle through the Laplacian operator. Each voxel effectively becomes a source of secondary wavelets to its neighbors.

### Comparison Table

| Aspect | Wave Equation (PDE) | Huygens Wavelets (Explicit) |
|--------|---------------------|----------------------------|
| **Physics** | Fundamental equation | Derived principle |
| **Accuracy** | Second-order accurate (with proper scheme) | Depends on implementation |
| **Efficiency** | ✓✓ Highly optimized | ✗ Computationally expensive |
| **Memory** | 3 fields (old, current, new) | 2 fields minimum |
| **Connectivity** | Natural 6/18/26 neighbors | All neighbors (26) typically |
| **Stability** | CFL condition required | Less restrictive |
| **Energy Conservation** | Excellent (with symplectic integrator) | Requires careful normalization |
| **Implementation** | Straightforward | Complex neighbor loops |
| **GPU Performance** | ✓✓ Excellent (simple stencil) | ✗ Many atomic operations |
| **Anisotropy** | Uniform in all directions | Can handle directional weighting |

### Performance Analysis

**Wave Equation (PDE)**:

```text
Operations per voxel per timestep:
- 6 neighbor reads (6-connectivity)
- 1 Laplacian computation (7 operations)
- 1 update computation (3 operations)
Total: ~10 operations

For 100³ = 1M voxels:
~10M operations per timestep
```

**Huygens Wavelets** (explicit, 26-connectivity):

```text
Operations per voxel per timestep:
- 26 neighbor checks
- 26 distance calculations
- 26 contribution calculations
- 26 atomic additions (slow!)
Total: ~100+ operations

For 100³ = 1M voxels:
~100M+ operations per timestep
```

**Performance Verdict**: **PDE is ~10× faster**

### Pros & Cons Summary

**Wave Equation (PDE)**:

**Pros**:

- ✓ Fast: Simple stencil operations
- ✓ Accurate: Well-established numerical methods
- ✓ Stable: Known stability conditions (CFL)
- ✓ Conserves energy naturally
- ✓ GPU-friendly: Coalesced memory access
- ✓ Physically fundamental

**Cons**:

- ✗ Requires small timesteps (CFL condition)
- ✗ Needs 3 amplitude arrays in memory
- ✗ Fixed neighbor connectivity

**Huygens Wavelets** (explicit):

**Pros**:

- ✓ Intuitive physical interpretation
- ✓ Flexible directional weighting
- ✓ Can handle complex geometries

**Cons**:

- ✗ Computationally expensive (~10× slower)
- ✗ Many atomic operations (GPU bottleneck)
- ✗ Energy conservation requires careful implementation
- ✗ More complex to code and debug

### Recommended

**Use PDE (Wave Equation)** for LEVEL-1:

1. Start with 6-connectivity (face neighbors) for speed
2. Upgrade to 18 or 26-connectivity if isotropy issues arise
3. Use second-order finite difference scheme (leap-frog)
4. Enforce CFL stability condition
5. Monitor energy conservation

**When to Consider Huygens**:

- Complex geometries with curved boundaries
- Directional wave sources with specific emission patterns
- Adaptive mesh refinement (not relevant for LEVEL-1)

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
        E_p = 0.5 * displacement[i,j,k]**2
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
