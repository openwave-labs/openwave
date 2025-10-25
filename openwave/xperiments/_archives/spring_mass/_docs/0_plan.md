# Harmonic Oscillators Specs (EWave Source)

## Overview

Implement harmonic oscillation for the 8 lattice vertices to inject energy into the spacetime lattice. These vertices will act as "wave makers" that later propagate motion through the spring-mass system to other granules.

**Goal**: Create radially-oscillating boundary conditions that drive energy wave propagation through the BCC lattice.

**Physics**: Each vertex oscillates harmonically along its direction vector to lattice center, with:

- Amplitude: `EWAVE_AMPLITUDE = 9.215e-19 m`
- Frequency: `EWAVE_SPEED / EWAVE_LENGTH ≈ 1.05e25 Hz` (slowed by factor 1e25)
- Motion: `displacement(t) = A·cos(2πft)` along radial direction

**Two-Phase Implementation**:

1. **Phase 1** (Current): Forced harmonic motion of 8 vertices only
2. **Phase 2** (Next): Spring-mass wave propagation to all granules

## Reference Materials

Review papers at `/research_requirements/scientific_source` for more info on the harmonic motion of spacetime granules, mechanics and geometry.

Also check `/research_requirements/original_requirements/1. Simulating a Fundamental Particle - EWT.pdf` that contain more info for this project

## PHASE 1: Vertex Harmonic Oscillation (Current Phase)

### 1. Constants already declared

- `AMPLITUDE` - how far vertices move from equilibrium (units already declared as well)
- `FREQUENCY` - base frequency of oscillation (Hz)
- `SLOW_MO` FACTOR - slow-down multiplier for human-visible motion
- `TARGET_FPS` (optional) - e.g., 60 or 30 to match screen refresh rate, lets skip that for now, if need we implement it later

### 2. Data Storage Needs

- **Equilibrium positions**: Store initial vertex positions in `vertex_equilibrium` field (8-element Vector field)
  - Reason: We NEED this because `self.positions` will be constantly updated each frame
  - Without stored equilibrium, we'd have to recalculate from grid indices every frame (wasteful)
  - Add to medium.py during initialization: `self.vertex_equilibrium = ti.Vector.field(3, dtype=ti.f32, shape=8)`
  - Populate during `build_vertex_index()` by storing `self.positions[vertex_idx]`

- **Time accumulation**: Use accumulated time variable `t` (not just dt)
  - Reason: `cos(2π * f * t)` requires absolute time, not delta
  - Without tracking `t`, the oscillation phase will be wrong
  - Solution: Add `t = 0.0` before while loop, then `t += dt` each frame
  - Pass `t` to the oscillation update function

- **Current positions**: Continue using `self.positions` (no duplication needed)

### 3. Harmonic Oscillator Equation

Vertices should oscillate in the direction found at BCCLattice.vertex_directions from equilibrium (starting position) inwards 1 amplitude back to equilibrium and outwards 1 amplitude, with velocities variations in a sine wave form (harmonic).

**Position equation**:

```python
displacement(t) = AMPLITUDE * cos(2π * f_slowed * t)
where f_slowed = FREQUENCY / SLOW_MO
new_position = equilibrium_pos + displacement * direction_vector
```

**Velocity equation** (derivative of position):

```python
velocity(t) = -AMPLITUDE * (2π * f_slowed) * sin(2π * f_slowed * t) * direction_vector
```

Note: For Phase 1, we can either:

- Option A: Update both position AND velocity (physically consistent)
- Option B: Update only position (simpler, velocity unused until Phase 2)

Recommendation: Update both for physical consistency, prepares for Phase 2.

### 4. Implementation Approach

- Option A: `@ti.kernel` function (fast, GPU-parallel, but only 8 vertices might not benefit much)
- Option B: Regular Python function (simpler, easier to debug, sufficient for 8 vertices)
- Reply: I think using a taichi kernel would be better, maybe we want to make the wave maker bigger with more granules in harmonic oscillation injecting wave energy in the lattice

### 5. Frame Rate & Time Step

- Should we cap frame rate (e.g., 60 FPS)?
- `dt` vs actual elapsed time - use `time.time()` for real-time or fixed `dt`? I was thinking on using dt not tracking current time, specially preparing for the next stage of spring-mass oscillation on other granules for wave propagation, that I'm planning to use a numerical integrator method alike game developers use
- Currently the while loop runs uncapped - should we add frame rate limiting? lets skip FPS cap for now, if need we implement it later

### 6. Function Placement & Implementation Details

**Location**: `energy_wave.py` (not medium.py - keep it modular)

**Function signature**:

```python
@ti.kernel
def oscillate_vertex(
    lattice_positions: ti.template(),
    lattice_velocities: ti.template(),
    vertex_index: ti.template(),
    vertex_equilibrium: ti.template(),
    vertex_directions: ti.template(),
    t: ti.f32,
    amplitude: ti.f32,
    frequency: ti.f32,
    slow_mo: ti.f32
):
    """Update 8 vertex positions and velocities using harmonic oscillation."""
    f_slowed = frequency / slow_mo
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency

    for v in range(8):
        idx = vertex_index[v]
        direction = vertex_directions[v]

        # Position: x(t) = x_eq + A·cos(ωt)·direction
        displacement = amplitude * ti.cos(omega * t)
        lattice_positions[idx] = vertex_equilibrium[v] + displacement * direction

        # Velocity: v(t) = -A·ω·sin(ωt)·direction (derivative of position)
        velocity_magnitude = -amplitude * omega * ti.sin(omega * t)
        lattice_velocities[idx] = velocity_magnitude * direction
```

**Call from ewave.py**:

```python
t = 0.0  # Initialize time before while loop
while window.running:
    t += DT  # Accumulate time
    energy_wave.oscillate_vertex(
        lattice.positions,
        lattice.velocities,
        lattice.vertex_index,
        lattice.vertex_equilibrium,
        lattice.vertex_directions,
        t, AMPLITUDE, FREQUENCY, SLOW_MO
    )
    # ... rest of render loop
```

### 7. Implementation Checklist (Phase 1)

#### Step 1: Modify medium.py

- [ ] Add field: `self.vertex_equilibrium = ti.Vector.field(3, dtype=ti.f32, shape=8)`
- [ ] In `build_vertex_index()`: Store equilibrium positions after computing index
  - `self.vertex_equilibrium[v] = self.positions[self.vertex_index[v]]`

#### Step 2: Create energy_wave.py functions

- [ ] Import taichi and necessary modules
- [ ] Implement `oscillate_vertex()` kernel (see signature above)
- [ ] Test with simple parameters first

#### Step 3: Modify ewave.py

- [ ] Import energy_wave module: `import openwave.spacetime.energy_wave as ewave`
- [ ] Initialize time: `t = 0.0` before while loop
- [ ] Add time accumulation: `t += DT` at start of while loop
- [ ] Call oscillation update before rendering
- [ ] Verify constants are properly defined (already done - AMPLITUDE, FREQUENCY, SLOW_MO)

#### Step 4: Debugging & Validation

- [ ] Print initial equilibrium positions to verify storage
- [ ] Print first few frames of vertex positions to verify oscillation
- [ ] Check that vertices move along correct direction vectors
- [ ] Verify amplitude matches expected displacement
- [ ] Confirm frequency produces visible oscillation with SLOW_MO factor

#### Step 5: Visual Validation

- [ ] Verify vertex granules (white spheres) oscillate visibly
- [ ] Check that motion is radial (toward/away from center)
- [ ] Confirm all 8 vertices oscillate in phase (synchronously)
- [ ] Measure oscillation period visually (should match expected frequency)
- [ ] Ensure no vertices "escape" the lattice bounds or behave erratically

#### Potential Issues to Watch

- **Initial phase**: `cos(0) = 1` means at t=0, displacement = +AMPLITUDE (vertices start OUTWARD)
  - If you want vertices to start INWARD, use `-cos()` or add phase shift `cos(2πft + π)`
  - Recommendation: Use `cos()` as-is for simplicity, document that motion starts outward
- **Direction sign**: `vertex_directions` points FROM vertex TO center (inward)
  - Positive displacement moves INWARD (correct)
  - Negative displacement moves OUTWARD (correct)
- **Numerical precision**: With SLOW_MO, ensure `f_slowed` doesn't underflow to zero
  - `f_slowed = (3e8 / 2.85e-17) / 1e25 ≈ 1.05e0` (safe, no underflow)
- **Performance**: Monitor FPS with uncapped frame rate
- **Coordinate system**: Ensure `vertex_directions` normalized vectors are correct
- **Spring system interaction**: Later, vertex forced motion may need to override spring forces

## Questions to Answer

1. **Amplitude units**: Should amplitude be in Planck lengths, or a fraction of `lattice_size` (e.g., 0.1 means 10% of lattice)?

   **Answer:** above

1. **Equilibrium storage**: Store original positions separately or restore from calculation each frame?

   **Answer:** above

1. **Frame rate**: Cap at 60 FPS? 30 FPS? Or leave uncapped?

   **Answer:** above

1. **Time tracking**: Use `time.time()` for real elapsed time, or accumulate `dt` each frame?

   **Answer:** above

1. **Kernel vs Python**: Prefer `@ti.kernel` (faster) or regular Python (simpler) for 8-vertex update?

   **Answer:** above

## PHASE 2: Mass-Spring System (EWave-Propagation)

### Overview Phase 2

Propagate vertex oscillations through the entire lattice using spring-mass dynamics.

### Key Requirements

- Connect vertices to other granules via spring forces: spring links are already created at medium.py Spring class
- Propagate oscillating motion like a wave through the lattice
- Implement spring constants: k = 5.56081e44 kg/s² (single spring when granule radius is Planck length)
  - Already adjusted for scale-up factor on Spring class: `k = COULOMB_CONSTANT / granule.radius`
  - **BCC Spring Constant Issue**: In BCC, each granule has 8 springs. Question: is the effective stiffness different?
    - Option 1: Use individual spring k directly (8 springs act independently, forces add vectorially)
    - Option 2: Calculate effective k_eff for the lattice (may need tensor analysis)
    - **Recommendation**: Start with Option 1 (individual k), validate wave speed matches theory
    - Yes lets start with option 1
- No damping in our case (energy-conserving system)
- Wave propagation dynamics using numerical integration method detailed below

### Critical Physics Notes

**Wave Speed Validation**:

- Expected wave speed in medium: `v = sqrt(k/m) * lattice_spacing`
- For our system: check if emergent wave speed ≈ speed of light (c = 299,792,458 m/s)
- If too fast/slow: adjust k or validate spring topology

**Wavelength Validation**:

- Driving frequency: f = EWAVE_SPEED / EWAVE_LENGTH ≈ 1.05e25 Hz (slowed by factor 1e25 → ~1 Hz visible)
- Expected wavelength: λ = c / f ≈ 2.854e-17 m (EWAVE_LENGTH constant)
- Measurement: Sample granule positions along propagation axis, find spatial period
- Relationship: λ = v / f, so if v ≈ c and f is correct → λ should match EWAVE_LENGTH
- **This validates the entire physics model**: correct k, m, lattice spacing, and wave equation

**BCC Spring Topology**:

- Each interior granule has 8 neighbors
- Force on granule i: `F_i = sum over j (k * (|r_ij| - L0) * r_ij_hat)`
- Resultant force is vector sum of 8 spring forces
- Should naturally produce isotropic wave propagation in 3D

### Data Structures Needed

- Spring connectivity (already exists in Spring class)
- Per-granule mass (already exists as Granule.mass)
- Per-granule velocity (already exists as self.velocities)
- Per-granule acceleration (compute on-the-fly, don't store)
- Spring rest lengths (already exists as Spring.rest_length)

### Implementation Strategy

Use `energy_wave.py` for all wave dynamics functions:

1. `compute_spring_forces()` - Compute displacement and calculate force on each granule from connected springs
2. `integrate_motion()` - Update velocities and positions using best method below
3. `propagate_ewave()` - Main function that orchestrates the above with substepping

- **ATTENTION**: vertex granules have their own motion (harmonic oscillation) as they inject energy in the lattice (wave makers) so we can't update vertex velocities/position when iterating over all the other granules in the lattice, OR we might find a better way to implement both for better performance

### Phase 2 Implementation Checklist

#### Step 1: Leapfrog Integration Research & Setup

- [ ] Study Leapfrog algorithm (velocity Verlet variant)
- [ ] Implement basic Leapfrog kernel for single particle test
- [ ] Compare energy conservation: Leapfrog vs Euler
- [ ] Determine optimal substep count for stability

#### Step 2: Spring Force Computation Kernel

- [ ] Implement `compute_spring_forces()` kernel in energy_wave.py
- [ ] For each granule: iterate through Spring.links to find neighbors
- [ ] Calculate displacement vector: `d = pos[neighbor] - pos[current]`
- [ ] Calculate distance: `dist = |d|`
- [ ] Calculate spring extension: `x = dist - rest_length`
- [ ] Calculate spring force magnitude: `F_mag = -k * x`
- [ ] Calculate force direction: `F_vec = F_mag * (d / dist)` (unit vector)
- [ ] Accumulate forces from all 8 neighbors into resultant force
- [ ] Return force field (or acceleration field = F/mass)

#### Step 3: Vertex Exclusion Strategy

Two approaches to handle vertices (wave makers) vs propagating granules:

- **Option A: Skip vertices in propagation loop (won't work)**
  - Check `if granule_type[i] != TYPE_VERTEX` before computing spring forces
  - Vertices keep harmonic motion, others use spring-mass dynamics
  - Simple but may lose some coupling at vertex-neighbor interface

- **Option B: Hybrid force model**
  - Vertices: prescribed motion from `oscillate_vertex()` (boundary condition)
  - Non-vertices: spring forces include vertex positions as inputs
  - More physically accurate energy injection from vertices to neighbors
  - Recommended approach
  - I agree to use option B, if we remove vertex there wont be any motion (they ar the wave makers, the triggers)

#### Step 4: Leapfrog Integration Kernel

- [ ] Implement `integrate_motion()` in energy_wave.py
- [ ] Half-step velocity update: `v(t+dt/2) = v(t) + a(t) * dt/2`
- [ ] Position update: `x(t+dt) = x(t) + v(t+dt/2) * dt`
- [ ] Force recompute at new positions
- [ ] Final half-step velocity: `v(t+dt) = v(t+dt/2) + a(t+dt) * dt/2`
- [ ] Exclude vertices from integration (keep their prescribed motion)

#### Step 5: Main Propagation Orchestrator

- [ ] Create `propagate_ewave()` function
- [ ] Call `oscillate_vertex()` first (boundary condition)
- [ ] Loop substeps:
  - Compute spring forces on non-vertex granules
  - Integrate motion using Leapfrog
  - Accumulate substep time
- [ ] Update rendering positions after all substeps

#### Step 6: Validation & Tuning

- [ ] Verify wave propagates from vertices
- [ ] Check energy conservation (should be stable over time)
- [ ] Tune substep count for stability vs performance
- [ ] **Measure wave speed**: Compare emergent propagation velocity to expected `c = EWAVE_SPEED`
- [ ] **Measure wavelength**: Track spatial period of oscillation, compare to `λ = EWAVE_LENGTH`
  - Method: Sample positions along radial line from vertex, measure distance between peaks
  - Expected: λ ≈ 2.854e-17 m (from constants)
  - Validates both spring constant k and lattice discretization
- [ ] Visualize wave patterns (should see spherical/radial propagation)
- [ ] **Success criteria**: Wave speed ≈ c AND wavelength ≈ λ (within 5-10% tolerance)

### Integration Method Review (TODO - Before Phase 2 Implementation)

#### Priority: Evaluate Leapfrog Integration Method

The Leapfrog integrator occupies a sweet spot for our needs:

- **Second-order accuracy** (more accurate than Euler's first-order)
- **Symplectic** (conserves energy better - critical for undamped systems)
- **Computational efficiency** (similar cost to Euler, unlike RK4's 4x overhead)
- **Physics pedigree** (widely used in quantum physics and molecular dynamics)
- **May reduce substep requirements** (better accuracy per step than Euler)

**Evaluation Checklist**:

- [ ] **Leapfrog Method (Primary Candidate)**
  - Second-order symplectic integrator for Newton's equations of motion
  - Energy-conserving (essential for our undamped wave propagation)
  - Minimal performance cost vs Euler
  - Well-suited for oscillatory spring-mass systems
  - May significantly reduce need for multiple substeps

- [ ] **Runge-Kutta 4 (RK4) - Secondary Option**
  - Fourth-order accuracy (very high precision)
  - NOT symplectic (may drift in energy over long simulations)
  - 4x computational cost per step (expensive for large lattice)
  - Consider only if Leapfrog accuracy insufficient

- [ ] **Explicit Euler (Baseline)**
  - First-order accuracy (lowest)
  - Requires many substeps for stability
  - Simple but energy-drifting
  - Keep as fallback/comparison

**Decision Point**: Research and implement Leapfrog before Phase 2 integration. Compare energy conservation and performance against Euler baseline.

### Leapfrog Algorithm (Velocity Verlet) - Detailed

**Standard Leapfrog (Kick-Drift-Kick)**:

```python
# Half-step velocity update (kick)
v_half = v(t) + 0.5 * a(t) * dt

# Full-step position update (drift)
x(t+dt) = x(t) + v_half * dt

# Compute new acceleration from new positions
a(t+dt) = F(x(t+dt)) / m

# Final half-step velocity update (kick)
v(t+dt) = v_half + 0.5 * a(t+dt) * dt
```

**Key Properties**:

- Time-reversible (run backward to get original state)
- Symplectic (preserves phase space volume → energy conservation)
- Second-order accurate (error O(dt²))
- Requires 1 force evaluation per timestep (same as Euler)
- Velocities and positions offset by dt/2 (leapfrog pattern)

**Implementation Strategy for Phase 2**:

1. Store current acceleration field from previous step (or compute initially)
2. Half-kick all non-vertex velocities
3. Drift all non-vertex positions
4. Update vertex positions via `oscillate_vertex()` (boundary condition)
5. Compute new accelerations from spring forces (using updated positions including vertices)
6. Final half-kick non-vertex velocities
7. Repeat for substeps

## Integration method (draft, from game development practices, upgrade to Leapfrog method after research)

compute each granule new position = propagation

- Fs, v, x = zero # initialize local vars for safety

compute 8-way 3D granule distance and direction vectors (A-B), plus forces and acceleration

- x = |B-A| - L0 (displacement from rest L0, already computed in `Spring.rest_length`)
  - use Spring.links to find the up to 8 linked granules to each granule iteration, find distance B-A (and displacement from rest calculation) with direction (for resultant force calculation later)
- Fs = -k * x (computes the spring force from current displacement at that frame)
  - k constant needs to be defined, we have the single spring k = 5.56081e44 (kg/s^2) at planck length and an equation for k related to granule.radius, but the BCC lattice configuration may generate a new k constant, i'm not sure if the linear 1D in series spring k equation works here (1/k_total)=n*(1/k)
- a = Fs / m (compute acceleration from Granule.mass)
- find resultant 3D force vector (from 8-way force vectors)

Perform Integration (LeapFrog)

- vel(i+1/2) = v(i-1/2) + a(i) * dt # compute new velocity
- pos(i+1) = pos(i) + vel(i+1/2) * dt # compute new position

## How to achieve numerical stability in spring-mass systems (best practices & techniques in computational physics simulators)

So the 30 iterations are substeps per frame, not pre-warming! This is a common technique in physics simulations where:

- Each rendered frame runs multiple small physics steps
- This improves stability (smaller time steps = more accurate physics)
- The cloth moves smoothly even with complex spring interactions

Think of it like: for every single frame you see on screen, the physics engine calculates 30 tiny movement updates. This prevents the springs from "exploding" due to large time steps and makes the cloth behave more realistically.

The comment at line 201-202 confirms this: "Run multiple physics substeps per frame for stability" and "30 substeps provides good balance between accuracy and performance."

Here's why multiple small timesteps are crucial:

- Numerical Stability: Spring forces can be very strong (stiffness = 1200). With large timesteps, the spring force calculation can overshoot, causing particles to fly apart explosively. This is called "numerical instability" or "exploding springs."
- Stiff System Problem: The equation `v[i] += force * dt` becomes unstable when `force * dt` is too large. With high stiffness values, even a moderately sized dt can cause:
  - Particles to overshoot their equilibrium positions- Energy to artificially increase each step- The simulation to "blow up"
- CFL Condition: There's a mathematical constraint (Courant-Friedrichs-Lewy condition) that limits timestep size based on:- Spring stiffness (higher stiffness → smaller dt needed)- Particle mass (implicit here, assumed = 1)- Grid spacingWith stiffness = 1200 and cell_size ≈ 0.0078, using dt = 0.015 in a single step would likely cause the cloth to explode or behave erratically.The 30 substeps keep each individual physics calculation stable while still advancing the simulation at a reasonable pace. It's a standard technique in physics simulation called "substepping" - trading computation for stability.You could experiment with fewer substeps and larger dt, but you'd quickly hit the stability limit where the cloth starts behaving unrealistically.
