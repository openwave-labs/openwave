# The Classical Wave Equation

LEVEL-1 uses **PDEs (Partial Differential Equations)** to propagate waves through the field.

**3D Wave Equation** (fundamental):

```text
∂²ψ/∂t² = c²∇²ψ

or simplified as:

ψ" = c²Δψ
```

Where:

- `ψ` = wave displacement field (scalar)
- `c` = wave propagation speed (speed of light, 2.998×10⁸ m/s)
- `∇²ψ` = Laplacian operator (second-order spatial derivative)
- `∂²ψ/∂t²` = second-order time derivative (acceleration of displacement)

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
@ti.kernel
def propagate_wave_field(dt: ti.f32):
    """
    Propagate wave displacement using wave equation.

    Second-order in time (requires storing two previous timesteps):
    ψ_new = 2ψ_current - ψ_old + (c×dt/dx)² × ∇²ψ

    This is a centered finite difference scheme, second-order accurate.
    """
    # Speed of light and CFL factor
    c = ti.f32(constants.EWAVE_SPEED)
    cfl_factor = (c * dt / self.dx_am)**2

    # CFL stability condition: cfl_factor ≤ 1/3 for 3D (6-connectivity)
    # If violated, solution becomes unstable

    # Update all interior voxels
    for i, j, k in self.displacement_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Compute Laplacian (6-connectivity)
            laplacian = (
                self.displacement_am[i+1,j,k] + self.displacement_am[i-1,j,k] +
                self.displacement_am[i,j+1,k] + self.displacement_am[i,j-1,k] +
                self.displacement_am[i,j,k+1] + self.displacement_am[i,j,k-1] -
                6.0 * self.displacement_am[i,j,k]
            ) / (self.dx_am * self.dx_am)

            # Wave equation update (leap-frog scheme)
            self.displacement_new[i,j,k] = (
                2.0 * self.displacement_am[i,j,k]  # Current displacement
                - self.displacement_old[i,j,k]      # Previous displacement
                + cfl_factor * laplacian          # Wave propagation term
            )

    # Swap timesteps for next iteration
    # old ← current ← new
    self.displacement_old, self.displacement_am = self.displacement_am, self.displacement_new
```

**Storage Requirements**:

- Three displacement fields: `displacement_old`, `displacement_am` (current), `displacement_new`
- Needed for second-order time integration

**Stability Condition** (CFL - Courant-Friedrichs-Lewy):

```text
dt ≤ dx / (c√3)  for 3D, 6-connectivity

Example:
dx = 1.25 am = 1.25e-18 m
c = 2.998e8 m/s
dt_max = 1.25e-18 / (2.998e8 × √3) ≈ 2.4e-27 s
```

**This is extremely small!** Timesteps are ~10⁻²⁷ seconds (rontosecond scale).

**Rontosecond Scaling Solution**:

Just as spatial scales use attometer (10⁻¹⁸ m) scaling for numerical precision, **temporal scales use rontosecond (10⁻²⁷ s) scaling**:

```python
# Scaling constants
ATTOMETER = 1e-18     # m, attometer length scale
RONTOSECOND = 1e-27   # s, rontosecond time scale

# Convert timestep to rontoseconds
dt_rs = dt / constants.RONTOSECOND

# Example:
dt = 2.4e-27  # Physical timestep in seconds (SI)
dt_rs = 2.4e-27 / 1e-27 = 2.4  # Scaled to rontoseconds
```

**Benefits**:

- Maintains numerical precision with f32 (6-7 significant digits)
- Prevents catastrophic cancellation in time derivatives
- dt_rs values ~1.0 (optimal range for f32 precision)
- Naming convention: `_am` for spatial (attometers), `_rs` for temporal (rontoseconds)

### Timestep Strategy: Fixed vs Elapsed Time

**CRITICAL DECISION**: LEVEL-1 must use **fixed timesteps** (unlike LEVEL-0's elapsed time approach).

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
# Wave equation CFL requirement
dt_max = dx / (c√3) ≈ 2.4e-27 s  # MUST NOT EXCEED!

# But elapsed time is typically:
elapsed_t ≈ 0.001 to 0.1 s  # Frame time (milliseconds)

# Ratio: elapsed_t / dt_max ≈ 10^24
# Result: IMMEDIATE NUMERICAL EXPLOSION 💥
```

**The wave equation becomes unstable if dt > dt_max**. Using elapsed time would violate CFL by ~24 orders of magnitude!

**LEVEL-1 Solution: Fixed Timestep + Frame Accumulator**:

```python
# Fixed physics timestep (respects CFL)
dt_physics = 2.0e-27  # seconds (or 2.0 in rontoseconds)
dt_physics_rs = 2.0   # rontoseconds (scaled)

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

## CLAUDE REVIEW: Old Propagation Mechanics

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
    2 * displacement[i,j,k]
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
    displacement[i+1,j,k] + displacement[i-1,j,k] +
    displacement[i,j+1,k] + displacement[i,j-1,k] +
    displacement[i,j,k+1] + displacement[i,j,k-1] -
    6.0 * displacement[i,j,k]
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
wavelet_contribution = displacement[i,j,k] * weight * directional_factor
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
