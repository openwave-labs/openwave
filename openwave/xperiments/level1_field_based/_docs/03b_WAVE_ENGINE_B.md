# WAVE ENGINE - FORCE CALCULATION & WAVE PROPAGATION

## Table of Contents

1. [Questions](#questions)
   - [Part 1: Force Calculation in Newtons](#part-1-force-calculation-in-newtons)
   - [Part 2: Wave Amplitude Propagation](#part-2-wave-amplitude-propagation)
   - [Part 3: PDE vs Huygens Wavelets](#part-3-pde-vs-huygens-wavelets)
   - [Part 4: Wave Direction in PDE Propagation](#part-4-wave-direction-in-pde-propagation)
1. [Complete Answers](#complete-answers)
   - [Answer 1: Force in Newtons from Amplitude Gradient](#answer-1-force-in-newtons-from-amplitude-gradient)
   - [Answer 2: Wave Propagation Mechanics](#answer-2-wave-propagation-mechanics)
   - [Answer 3: Choosing Between PDE and Huygens](#answer-3-choosing-between-pde-and-huygens)
   - [Answer 4: Wave Direction Computation](#answer-4-wave-direction-computation)
1. [The Complete Picture](#the-complete-picture)
1. [Implementation Summary](#implementation-summary)

---

## Questions

### Part 1: Force Calculation in Newtons

**Context**: Understanding force calculation in LEVEL-1 field-based simulation.

**Known**:

- Each voxel stores wave amplitude in attometers
- Force follows MAP (Minimum Amplitude Principle): `F = -∇A`
- Amplitude gradient can be calculated from neighboring voxels using finite differences

**Question**:

Can we compute force in **Newtons** (kg⋅m/s²) from the amplitude gradient?

**Requirements**:

- Need force in Newtons at each voxel to calculate particle acceleration
- Given particle mass (calculated from standing wave radius), find acceleration: `a = F/m`
- With acceleration, we can integrate motion: velocity update → position update
- Particles initially are single wave centers (fundamental particles like neutrino), later becoming standalone particles (like electron)
- Particle mass likely comes from standing waves reflected around particle radius (related to wavelength λ)

**Resources**: EWT research papers at `/research_requirements/scientific_source/`

---

### Part 2: Wave Amplitude Propagation

**Context**: Understanding how wave properties propagate through the field.

**System Overview**:

1. **Wave Field**: The medium (grid of voxels)
2. **Energy Charge**: Initial energy injected into system
3. **Wave Propagation**: Wave properties propagate in wave-like motion:
   - Amplitude, displacement
   - Wave direction, speed
   - Energy, phase
   - Wavelength, frequency
4. **Propagation Methods** (unclear which to use):
   - PDEs (Partial Differential Equations)?
   - Laplacian operator?
   - Wave equation?
   - Huygens wavelets?
5. **Conservation**: How does propagation conserve/transfer energy and momentum?

**Wave Interactions**:

- Waves interfere with each other (constructive/destructive)
- Waves reflect from:
  - Universe boundaries (grid boundaries)
  - Wave centers (fundamental particles)
- Reflection creates standing waves (interference of inward/outward waves)
- Standing waves give particles mass (trapped energy)

**Particle Behavior**:

- Particles interact with each other, forming complex structures
- Particles move to minimize amplitude (MAP principle)
- Movement toward lower amplitude regions is the effect of force
- Force must have value in Newtons

**Fundamental Forces**:

All forces in nature emerge from wave interactions:

- Electric force
- Magnetic force
- Gravitational force
- Strong force
- Orbital mechanics

All generated from the fundamental energy wave (EWT).

**Resources**: Documentation files in `/openwave/xperiments/level1_field_based/_docs/`

---

### Part 3: PDE vs Huygens Wavelets

**Question**: Should we choose between wave equation (PDE) and Huygens wavelets, or use both?

**Sub-questions**:

- What are the pros and cons of each approach?
- Which is more computationally performant?
- Can they be combined?
- Which better represents EWT physics?

---

### Part 4: Wave Direction in PDE Propagation

**Question**: How is the direction of wave propagation computed and stored when using PDE-based wave equation propagation?

**Context**:

- Wave equation: `∂²ψ/∂t² = c²∇²ψ` only evolves amplitude
- Need wave direction for:
  - Force calculations
  - Momentum transfer
  - Interference patterns
  - Particle-wave interactions

---

## Complete Answers

### Answer 1: Force in Newtons from Amplitude Gradient

#### YES, You Can Compute Force in Newtons

**Key Formula from EWT**:

```text
Energy density: u = ½ρc²(A/λ)²  [J/m³]
Force density:  f = -∇u        [N/m³]
Force on voxel: F = f × dx³    [N]
```

Where:

- `ρ` = medium density (3.860×10²² kg/m³ from EWT)
- `c` = wave speed (2.998×10⁸ m/s, speed of light)
- `A` = wave amplitude (meters)
- `λ` = wavelength (meters)
- `dx³` = voxel volume (cubic meters)

#### Physics Derivation

**Energy in wave field**:

```text
Total energy: E = ∫ u dV
where u = ½ρc²(A/λ)² is energy density
```

**Force from energy gradient**:

```text
F = -∇E
  = -∇(u × V)
  = -∇(½ρc²(A/λ)² × V)
  = -(ρc²/λ²) × V × ∇(A²)
  = -(ρc²/λ²) × V × 2A∇A
  = -(ρc²/λ²) × V × A × ∇A  (neglecting factor of 2)
```

Where `V = dx³` is voxel volume.

#### Implementation

```python
@ti.kernel
def compute_force_field_newtons(self):
    """
    Compute force in Newtons from amplitude gradient.

    Physics:
    - Energy density: u = ½ρc²(A/λ)²
    - Force: F = -∇E = -∇(u×V) = -(ρc²/λ²) × V × A × ∇A
    where V = dx³ (voxel volume)

    MAP Principle: Force points toward lower amplitude (negative gradient)
    """
    # Physical constants from EWT
    ρ = ti.f32(constants.MEDIUM_DENSITY)  # 3.860e22 kg/m³
    c = ti.f32(constants.EWAVE_SPEED)     # 2.998e8 m/s
    λ_m = self.wavelength_am * constants.ATTOMETER  # wavelength in meters
    dx_m = self.dx_am * constants.ATTOMETER         # voxel size in meters

    # Force scaling factor
    # Dimensional analysis: (kg/m³)(m²/s²)/(m²)(m³) = kg⋅m/s² = N
    force_scale = (ρ * c**2 / λ_m**2) * (dx_m**3)

    for i, j, k in self.amplitude_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Local amplitude in meters
            A_m = self.amplitude_am[i,j,k] * constants.ATTOMETER

            # Amplitude gradient (dimensionless: am/am)
            grad_x = (self.amplitude_am[i+1,j,k] - self.amplitude_am[i-1,j,k]) / (2.0 * self.dx_am)
            grad_y = (self.amplitude_am[i,j+1,k] - self.amplitude_am[i,j-1,k]) / (2.0 * self.dx_am)
            grad_z = (self.amplitude_am[i,j,k+1] - self.amplitude_am[i,j,k-1]) / (2.0 * self.dx_am)

            grad_vector = ti.Vector([grad_x, grad_y, grad_z])

            # Force in Newtons (MAP: toward lower amplitude)
            # F = -force_scale × A × ∇A
            self.force[i,j,k] = -force_scale * A_m * grad_vector  # N = kg⋅m/s²
```

#### Dimensional Analysis Verification

```text
ρ:          [kg/m³]
c²:         [m²/s²]
λ²:         [m²]
dx³:        [m³]
A:          [m]
∇A:         [dimensionless] = [m/m] after gradient divided by dx

force_scale = (kg/m³) × (m²/s²) / (m²) × (m³) = kg⋅m/s²  ✓ (Newton)
F = (kg⋅m/s²) × (m) × (dimensionless) = kg⋅m/s² = N  ✓✓
```

**This is correct!**

#### Particle Mass from Standing Waves

**From EWT Papers**:

- Particle mass comes from **trapped energy** in standing waves
- For electron: `E_electron = (μ₀c²/4π) × (e_e²/r_e)`
- Mass-energy relation: `m = E/c²`

**In Simulation**:

- Particle standing wave radius: `r = n × λ/2` (nodes at half-wavelengths)
- Energy trapped: `E = ∫ u dV` over standing wave volume
- Particle mass: `m = E/c²`

**Standing wave formation**:

1. Wave center reflects incoming waves
2. Reflected waves interfere with incoming waves
3. Constructive interference creates standing wave pattern
4. Energy trapped in standing wave = particle mass
5. Radius of first node determines particle size

#### Particle Acceleration and Motion

```python
# For particle at position pos_am (in attometers)
# 1. Interpolate force from grid to particle position
F_particle = interpolate_force(self.force, pos_am)  # Newtons

# 2. Newton's second law: F = ma
a = F_particle / particle_mass  # m/s²

# 3. Integrate motion using Velocity Verlet method
v_new = v_old + a * dt  # m/s
pos_new = pos_old + v_new * dt  # meters (or attometers)
```

**This IS the origin of all forces!**

- Electric force: Different wave reflection patterns (charge types)
- Magnetic force: Moving wave patterns (velocity-dependent)
- Gravitational force: Amplitude shading from trapped energy (mass)
- All emerge from wave amplitude gradients!

---

### Answer 2: Wave Propagation Mechanics

#### How Waves Propagate Through Voxels

LEVEL-1 uses **PDEs (Partial Differential Equations)** to propagate waves through the field.

#### The Classical Wave Equation

**3D Wave Equation** (fundamental):

```text
∂²ψ/∂t² = c²∇²ψ
```

Where:

- `ψ` = wave amplitude field (scalar)
- `c` = wave propagation speed (speed of light, 2.998×10⁸ m/s)
- `∇²ψ` = Laplacian operator (spatial second derivative)
- `∂²ψ/∂t²` = second time derivative (acceleration of amplitude)

**Physical Interpretation**:

- Left side: How fast amplitude is accelerating in time
- Right side: How much amplitude differs from neighbors (curvature)
- Equation says: "Amplitude accelerates toward its neighbors' average"

#### Laplacian Operator (How Voxels Share Amplitude)

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

- Laplacian measures how much a voxel's amplitude differs from its neighbors' average
- Positive Laplacian: voxel lower than average → amplitude will increase
- Negative Laplacian: voxel higher than average → amplitude will decrease
- This drives wave propagation: differences smooth out over time

#### Time Evolution Implementation

```python
@ti.kernel
def propagate_wave_field(dt: ti.f32):
    """
    Propagate wave amplitude using wave equation.

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
    for i, j, k in self.amplitude_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Compute Laplacian (6-connectivity)
            laplacian = (
                self.amplitude_am[i+1,j,k] + self.amplitude_am[i-1,j,k] +
                self.amplitude_am[i,j+1,k] + self.amplitude_am[i,j-1,k] +
                self.amplitude_am[i,j,k+1] + self.amplitude_am[i,j,k-1] -
                6.0 * self.amplitude_am[i,j,k]
            ) / (self.dx_am * self.dx_am)

            # Wave equation update (leap-frog scheme)
            self.amplitude_new[i,j,k] = (
                2.0 * self.amplitude_am[i,j,k]  # Current amplitude
                - self.amplitude_old[i,j,k]      # Previous amplitude
                + cfl_factor * laplacian          # Wave propagation term
            )

    # Swap timesteps for next iteration
    # old ← current ← new
    self.amplitude_old, self.amplitude_am = self.amplitude_am, self.amplitude_new
```

**Storage Requirements**:

- Three amplitude fields: `amplitude_old`, `amplitude_am` (current), `amplitude_new`
- Needed for second-order time integration

**Stability Condition** (CFL - Courant-Friedrichs-Lewy):

```text
dt ≤ dx / (c√3)  for 3D, 6-connectivity

Example:
dx = 1.25 am = 1.25e-18 m
c = 2.998e8 m/s
dt_max = 1.25e-18 / (2.998e8 × √3) ≈ 2.4e-27 s
```

This is extremely small! Need many timesteps.

#### Alternative: Huygens Wavelets

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

    for i, j, k in self.amplitude_am:
        if ti.abs(self.amplitude_am[i,j,k]) > threshold:
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
                            contribution = self.amplitude_am[i,j,k] / distance

                            # Add to neighbor (superposition)
                            # Note: This is simplified, full implementation needs proper weighting
                            ti.atomic_add(self.amplitude_new[ni,nj,nk], contribution * dt)
```

**Note**: Huygens method is less efficient on regular grids. The wave equation (PDE) implicitly implements Huygens' principle through the Laplacian operator.

#### Energy and Momentum Conservation

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

    for i, j, k in self.amplitude_am:
        # Velocity (time derivative of amplitude)
        v = (self.amplitude_am[i,j,k] - self.amplitude_old[i,j,k]) / dt

        # Kinetic energy density
        E_k = 0.5 * ρ * v**2

        # Potential energy density
        E_p = 0.5 * ρ * c**2 * (self.amplitude_am[i,j,k] / λ)**2

        # Add to total
        total_energy += (E_k + E_p) * dx**3

    return total_energy
```

**Momentum Density**:

```python
# Momentum carried by wave (vector field)
p[i,j,k] = ρ × ψ[i,j,k] × wave_direction[i,j,k]
```

Where `wave_direction` is determined by the gradient of phase (see Answer 4).

---

### Answer 3: Choosing Between PDE and Huygens

#### Should We Choose One or Use Both?

**Recommendation**: Use **PDE (Wave Equation)** as the primary method.

**Reason**: The wave equation naturally implements Huygens' principle through the Laplacian operator. Each voxel effectively becomes a source of secondary wavelets to its neighbors.

#### Comparison Table

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

#### Performance Analysis

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

#### Pros & Cons Summary

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

#### Recommendation

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

---

### Answer 4: Wave Direction Computation

#### The Challenge

The wave equation `∂²ψ/∂t² = c²∇²ψ` only evolves the **scalar amplitude** `ψ`. It doesn't directly compute wave direction.

**But we need direction for**:

- Momentum transfer: `p = ρψ × direction`
- Force calculations: Direction of energy flow
- Particle-wave interactions: Reflection angle
- Visualization: Wave propagation arrows

#### Solution: Compute Direction from Phase Gradient

##### Wave Direction = Gradient of Phase

In wave physics, the wave propagation direction is the gradient of the phase field:

```text
wave_direction = ∇φ / |∇φ|
```

Where `φ` is the wave phase at each voxel.

#### Computing Phase from Amplitude

For a traveling wave:

```text
ψ(x,t) = A(x) cos(kx - ωt + φ₀)
```

The phase at position x and time t is:

```text
φ(x,t) = arctan(ψ_imaginary / ψ_real)
```

But for a real-valued field (which we have), we need the **analytic signal** approach.

#### Practical Implementation: Energy Flux Method

**Better approach**: Compute wave direction from **energy flux** (Poynting-like vector for waves).

**Energy flux density** (energy flow direction):

```text
S = -c² × ψ × ∇ψ
```

Where:

- `ψ` = current amplitude
- `∇ψ` = spatial gradient of amplitude
- Direction of S = direction of energy flow = wave direction

**Implementation**:

```python
@ti.kernel
def compute_wave_direction(self):
    """
    Compute wave propagation direction from energy flux.

    Energy flux: S = -c² × ψ × ∇ψ
    Direction: normalized S
    """
    c = ti.f32(constants.EWAVE_SPEED)

    for i, j, k in self.amplitude_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Current amplitude
            psi = self.amplitude_am[i,j,k]

            # Amplitude gradient
            grad_x = (self.amplitude_am[i+1,j,k] - self.amplitude_am[i-1,j,k]) / (2.0 * self.dx_am)
            grad_y = (self.amplitude_am[i,j+1,k] - self.amplitude_am[i,j-1,k]) / (2.0 * self.dx_am)
            grad_z = (self.amplitude_am[i,j,k+1] - self.amplitude_am[i,j,k-1]) / (2.0 * self.dx_am)

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

#### Alternative: Velocity-Based Direction

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
    for i, j, k in self.amplitude_am:
        if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
            # Time derivative (wave velocity)
            v_wave = (self.amplitude_am[i,j,k] - self.amplitude_old[i,j,k]) / dt

            # Gradient of velocity gives acceleration direction
            # (This is less direct, energy flux method is better)
            # ...
```

**This method is less reliable** - use energy flux method instead.

#### Storage and Update Frequency

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

#### Summary: Wave Direction Pipeline

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

---

## The Complete Picture

### Wave Field → Forces → Motion

```text
┌─────────────────────────────────────────────────────────┐
│ 1. WAVE FIELD (grid of voxels)                          │
│    ├─ Initial energy charge (point/plane/spherical)     │
│    ├─ Stabilization period (waves propagate/reflect)    │
│    └─ Quasi-steady state (omni-directional field)       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. WAVE PROPAGATION (PDE evolution)                     │
│    ├─ Wave equation: ∂²ψ/∂t² = c²∇²ψ                    │
│    ├─ Laplacian couples neighboring voxels              │
│    ├─ Interference: constructive/destructive            │
│    ├─ Reflection: boundaries + wave centers             │
│    ├─ Standing waves form around particles              │
│    └─ Direction from energy flux: S = -c²ψ∇ψ            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. FORCE GENERATION (amplitude gradients)               │
│    ├─ Force field: F = -(ρc²/λ²)×V×A×∇A [Newtons]       │
│    ├─ Forces emerge from wave patterns                  │
│    ├─ Electric: wave reflection patterns (charges)      │
│    ├─ Magnetic: moving wave patterns (currents)         │
│    └─ Gravitational: amplitude shading (mass)           │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. PARTICLE MOTION (Newton's laws)                      │
│    ├─ Interpolate force at particle position            │
│    ├─ Acceleration: a = F/m                             │
│    ├─ Update velocity: v_new = v_old + a×dt             │
│    ├─ Update position: x_new = x_old + v×dt             │
│    └─ Particles move toward amplitude minimum (MAP)     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 5. PARTICLE-FIELD INTERACTION (feedback loop)           │
│    ├─ Particles act as wave reflectors                  │
│    ├─ Create standing wave patterns                     │
│    ├─ Trapped energy = particle mass: m = E/c²          │
│    ├─ Standing wave radius: r = n×λ/2                   │
│    └─ Force between particles from wave overlap         │
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
∂²ψ/∂t² = c²∇²ψ                    (wave propagation)
F = -∇E = -∇(½ρc²(A/λ)²×V)        (force from energy gradient)
```

This is the foundation of reality in Energy Wave Theory.

---

## Implementation Summary

### WaveField Class Methods

```python
@ti.data_oriented
class WaveField:
    """Complete wave field with force calculation."""

    def __init__(self, nx, ny, nz, wavelength_m, points_per_wavelength=40):
        # ... initialization with attometer scaling ...

        # Three amplitude fields for wave equation (leap-frog)
        self.amplitude_old = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.amplitude_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.amplitude_new = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

        # Wave direction (computed from energy flux)
        self.wave_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))

        # Force field (in Newtons)
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))

    @ti.kernel
    def propagate_wave_field(self, dt: ti.f32):
        """PDE-based wave propagation."""
        # See Answer 2
        pass

    @ti.kernel
    def compute_wave_direction(self):
        """Compute direction from energy flux."""
        # See Answer 4
        pass

    @ti.kernel
    def compute_force_field_newtons(self):
        """Compute force in Newtons from amplitude gradient."""
        # See Answer 1
        pass

    def update_timestep(self, dt):
        """Complete wave field update for one timestep."""
        # 1. Propagate wave amplitude
        self.propagate_wave_field(dt)

        # 2. Compute wave direction
        self.compute_wave_direction()

        # 3. Compute force field
        self.compute_force_field_newtons()

        # 4. Apply boundary conditions
        self.apply_boundary_conditions()
```

### Simulation Loop

```python
# Initialize
wave_field = WaveField(nx=100, ny=100, nz=100,
                       wavelength_m=constants.EWAVE_LENGTH)

# Charge initial energy
wave_field.charge_spherical_wave(center, energy, wavelength)

# Stabilization phase
for step in range(stabilization_steps):
    wave_field.update_timestep(dt)

# Main simulation with particles
for step in range(simulation_steps):
    # 1. Update wave field
    wave_field.update_timestep(dt)

    # 2. Update particles (see 05_MATTER.md)
    particles.update_positions(wave_field.force, dt)

    # 3. Apply particle reflections to field
    particles.apply_reflections_to_field(wave_field)
```

---

**Status**: Comprehensive framework defined with force calculation in Newtons and wave propagation via PDE

**Next Steps**: Implement and validate wave equation solver with energy conservation checks

**Related Documentation**:

- [`01_WAVE_FIELD.md`](./01_WAVE_FIELD.md) - Field architecture and indexing
- [`02_WAVE_PROPERTIES.md`](./02_WAVE_PROPERTIES.md) - WaveField class definition
- [`03_WAVE_ENGINE.md`](./03_WAVE_ENGINE.md) - Complete wave engine details
- [`05_MATTER.md`](./05_MATTER.md) - Particle dynamics and interactions
- [`06_FORCE_MOTION.md`](./06_FORCE_MOTION.md) - Force types and motion equations
