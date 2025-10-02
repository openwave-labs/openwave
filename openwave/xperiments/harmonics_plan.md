# Vertex Harmonic Oscillators Specs (QWave-Maker)

## Overview

Implement harmonic oscillation for the 8 lattice vertices to inject energy into the spacetime lattice. These vertices will act as "wave makers" that later propagate motion through the spring-mass system to other granules.

**Goal**: Create radially-oscillating boundary conditions that drive quantum wave propagation through the BCC lattice.

**Physics**: Each vertex oscillates harmonically along its direction vector to lattice center, with:

- Amplitude: `QWAVE_AMPLITUDE = 9.215e-19 m`
- Frequency: `QWAVE_SPEED / QWAVE_LENGTH ≈ 1.05e25 Hz` (slowed by factor 1e25)
- Motion: `displacement(t) = A·cos(2πft)` along radial direction

**Two-Phase Implementation**:

1. **Phase 1** (Current): Forced harmonic motion of 8 vertices only
2. **Phase 2** (Future): Spring-mass wave propagation to all granules

## Reference Materials

Review papers at `/research_requirements/scientific_source` for more info on the harmonic motion of spacetime granules, mechanics and geometry.

Also check `/research_requirements/original_requirements/1. Simulating a Fundamental Particle - EWT.pdf` that contain more info for this project

## PHASE 1: Vertex Harmonic Oscillation (Current Phase)

### 1. Constants already declared (at script qwave_render.py beginning)

- `AMPLITUDE` - how far vertices move from equilibrium (units already declared as well)
- `FREQUENCY` - base frequency of oscillation (Hz)
- `SLOW_MO` FACTOR - slow-down multiplier for human-visible motion
- `TARGET_FPS` (optional) - e.g., 60 or 30 to match screen refresh rate, lets skip that for now, if need we implement it later

### 2. Data Storage Needs

- **Equilibrium positions**: Store initial vertex positions in `vertex_equilibrium` field (8-element Vector field)
  - Reason: We NEED this because `self.positions` will be constantly updated each frame
  - Without stored equilibrium, we'd have to recalculate from grid indices every frame (wasteful)
  - Add to spacetime.py during initialization: `self.vertex_equilibrium = ti.Vector.field(3, dtype=ti.f32, shape=8)`
  - Populate during `build_vertex_indices()` by storing `self.positions[vertex_idx]`

- **Time accumulation**: Use accumulated time variable `t` (not just dt)
  - Reason: `cos(2π * f * t)` requires absolute time, not delta
  - Without tracking `t`, the oscillation phase will be wrong
  - Solution: Add `t = 0.0` before while loop, then `t += dt` each frame
  - Pass `t` to the oscillation update function

- **Current positions**: Continue using `self.positions` (no duplication needed)

### 3. Harmonic Oscillator Equation

Vertices should oscillate in the direction found at Lattice.vertex_directions from equilibrium (starting position) inwards 1 amplitude back to equilibrium and outwards 1 amplitude, with velocities variations in a sine wave form (harmonic).

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
- `dt` vs actual elapsed time - use `time.time()` for real-time or fixed `dt`? I was thinking on using dt not tracking current time, specially preparing for the next stage of spring-mass oscillation on other granules for wave propagation, that I'm planning to use an Euler Integrator method alike game developers use
- Currently the while loop runs uncapped - should we add frame rate limiting? lets skip FPS cap for now, if need we implement it later

### 6. Function Placement & Implementation Details

**Location**: `quantum_wave.py` (not spacetime.py - keep it modular)

**Function signature**:

```python
@ti.kernel
def oscillate_vertex(
    lattice_positions: ti.template(),
    lattice_velocities: ti.template(),
    vertex_indices: ti.template(),
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
        idx = vertex_indices[v]
        direction = vertex_directions[v]

        # Position: x(t) = x_eq + A·cos(ωt)·direction
        displacement = amplitude * ti.cos(omega * t)
        lattice_positions[idx] = vertex_equilibrium[v] + displacement * direction

        # Velocity: v(t) = -A·ω·sin(ωt)·direction (derivative of position)
        velocity_magnitude = -amplitude * omega * ti.sin(omega * t)
        lattice_velocities[idx] = velocity_magnitude * direction
```

**Call from qwave_render.py**:

```python
t = 0.0  # Initialize time before while loop
while window.running:
    t += DT  # Accumulate time
    quantum_wave.oscillate_vertex(
        lattice.positions,
        lattice.velocities,
        lattice.vertex_indices,
        lattice.vertex_equilibrium,
        lattice.vertex_directions,
        t, AMPLITUDE, FREQUENCY, SLOW_MO
    )
    # ... rest of render loop
```

### 7. Implementation Checklist (Phase 1)

#### Step 1: Modify spacetime.py

- [ ] Add field: `self.vertex_equilibrium = ti.Vector.field(3, dtype=ti.f32, shape=8)`
- [ ] In `build_vertex_indices()`: Store equilibrium positions after computing index
  - `self.vertex_equilibrium[v] = self.positions[self.vertex_indices[v]]`

#### Step 2: Create quantum_wave.py functions

- [ ] Import taichi and necessary modules
- [ ] Implement `oscillate_vertex()` kernel (see signature above)
- [ ] Test with simple parameters first

#### Step 3: Modify qwave_render.py

- [ ] Import quantum_wave module: `import openwave.source.quantum_wave as qwave`
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
- **Numerical precision**: With SLOW_MO = 1e25, ensure `f_slowed` doesn't underflow to zero
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

## PHASE 2: Mass-Spring System (Future QWave-Propagation)

### Overview Phase 2

Propagate vertex oscillations through the entire lattice using spring-mass dynamics.

### Key Requirements

- Connect vertices to other granules via spring forces: spring links are already created at spacetime.py Spring class
- Propagate oscillating motion like a wave through the lattice
- Implement spring constants: k = 5.56081e44 kg/s² (single spring at Planck scale)
  - Need to adjust for scale-up factor and BCC lattice configuration
  - May need effective k_total calculation for BCC topology
- No damping in our case (energy-conserving system)
- Wave propagation dynamics using Euler integration

### Data Structures Needed

- Spring connectivity (already exists in Spring class)
- Per-granule mass (already exists as Granule.mass)
- Per-granule velocity (already exists as self.velocities)
- Per-granule acceleration (compute on-the-fly, don't store)
- Spring rest lengths (already exists as Spring.rest_length)

### Implementation Strategy

Use `quantum_wave.py` for all wave dynamics functions:

1. `compute_spring_forces()` - Calculate force on each granule from connected springs
2. `integrate_euler()` - Update velocities and positions using explicit Euler
3. `update_wave_propagation()` - Main function that orchestrates the above with substepping

## Explicit Euler integration method (draft, from game development practices)

example file: spring_mass_example.py

compute each granule new position = propagation

- Fs, v, x = zero # initialize local vars for safety

compute 8-way 3D granule distance and direction vectors (A-B)

- x = |B-A| - L0 (displacement from rest L0, already computed in Spring.rest_length)
  - we need to find the up to 8 linked granules to each granule iteration, find distance (for displacement from rest calculation here) and direction (for resultant force calculation later)
- Fs = -k * x (computes the spring force from current displacement at that frame)
  - k constant will be defined later, we need a new way to find it, we have the single spring k = 5.56081e44 (kg/s^2), but this is the individual spring at planck scale, we're using a planck scale-up factor, and also the BCC lattice configuration may generate a new k constant, i'm not sure if the linear 1D in series spring k equation works here (1/k_total)=n*(1/k)
- a = Fs / m (compute acceleration from Granule.mass)

- find resultant 3D force vector (from 8-way force vectors)

Perform Euler Integration

- v += a * dt # compute new velocity
- p += v * dt # compute new position

## How to achieve numerical stability in spring-mass systems (best practices & techniques in computational physics simulators)

Research Paper source: <https://matthias-research.github.io/pages/publications/smallsteps.pdf>

example file: spring_mass_example.py

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
