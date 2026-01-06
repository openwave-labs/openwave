# XPBD vs SPH vs MPM

XPBD (Extended Position-Based Dynamics)

Type: Constraint-based solver
Domain: Solves constraints (geometric relationships)
Method:

- Define geometric constraints (e.g., "distance between particles = L₀")
- Iteratively project positions to satisfy constraints
- Derive velocities from position changes

Example constraint: "Spring connecting particles i,j must have length L₀"

## Move particles to fix constraint violation

C = ||xi - xj|| - L0  # How much constraint is violated
Δx = correction to make C = 0  # Move particles

Used for: Cloth, ropes, rigid bodies, soft bodies, ragdolls
Key feature: Geometric - directly manipulates positions

---
SPH (Smoothed Particle Hydrodynamics)

Type: Lagrangian fluid solver (meshless)
Domain: Solves fluid dynamics (continuum mechanics)
Method:

- Represent fluid as particles with properties (density, pressure, velocity)
- Compute forces from neighboring particles using kernel functions
- Each particle carries fluid properties, kernel smooths them

Core equation:
ρ(x) = Σ m_j · W(x - x_j, h)  # Density from weighted neighbors
P = k(ρ - ρ₀)  # Pressure from density
F = -∇P  # Pressure force

Used for: Water, liquids, gases, explosions
Key feature: Physics-based - models continuum as particles

---
MPM (Material Point Method)

Type: Hybrid Eulerian-Lagrangian (particles + grid)
Domain: Solves continuum mechanics (elastoplasticity)
Method:

- Particles carry material properties (mass, velocity, deformation)
- Transfer to background grid → solve on grid → transfer back to particles
- Grid handles topology changes, particles track material history

Algorithm:

1. Particle-to-Grid: Transfer mass, momentum to grid
2. Grid Update: Solve forces, update velocities on grid
3. Grid-to-Particle: Transfer velocities back to particles
4. Particle Update: Move particles, update deformation

Used for: Snow, sand, foam, elastoplastic materials, fracture
Key feature: Hybrid - combines particle tracking with grid-based solving

---
Key Differences:

| Aspect         | XPBD                       | SPH                     | MPM                       |
|----------------|----------------------------|-------------------------|---------------------------|
| Physics        | Constraints (geometric)    | Fluid dynamics          | Continuum mechanics       |
| Representation | Particles with constraints | Particles with kernels  | Particles + Grid          |
| Solver         | Iterative projection       | Force-based integration | Grid-based PDE solver     |
| Topology       | Fixed connectivity         | Fully meshless          | Particles can split/merge |
| Best for       | Cloth, ropes, bodies       | Liquids, gases          | Snow, sand, elastoplastic |

---
How They Relate to Your Medium lattice:

Your BCC Lattice Structure:

- Fixed connectivity (8 neighbors per granule)
- Spring-like interactions
- Needs to maintain lattice structure

Which Method Fits?

XPBD ✅ BEST MATCH

- Your granules have fixed neighbor relationships (like cloth/rope)
- You want distance constraints between neighbors
- Constraints preserve lattice structure
- Handles stiffness well

SPH ❌ Poor fit

- SPH is for fluids with changing neighbors
- Uses kernel smoothing (assumes continuum)
- Your lattice is discrete structure, not fluid
- Would need to recompute neighbors every frame (expensive)

MPM ❌ Overkill

- Designed for elastoplastic deformation (permanent changes)
- Requires background grid + particle transfer (2x complexity)
- Your lattice is elastic only (springs, no plasticity)
- Would be like using a sledgehammer for a nail

---
Conceptually, XPBD is More Like:

Inverse Kinematics (IK) - solving for positions that satisfy constraints

- Skeleton bones must maintain length → XPBD distance constraints
- Joint angle limits → XPBD angular constraints

Geometric Constraint Solving (CAD software)

- "These two lines must be perpendicular"
- "This circle must have radius R"
- XPBD: "These particles must be distance L apart"

---
For Your Medium lattice:

XPBD is the right choice because:

1. ✅ Your BCC lattice = fixed constraint network (like cloth mesh)
2. ✅ Each spring = distance constraint between neighbors
3. ✅ Maintains lattice topology (doesn't flow like SPH)
4. ✅ Handles extreme stiffness (unlike force-based springs)
5. ✅ Simple to implement (no grid like MPM, no kernels like SPH)

Think of it as: "Cloth simulation, but in 3D with BCC connectivity instead of triangular mesh"

The paper you read (Small Steps) is literally about cloth simulation using XPBD - your lattice is structurally identical (fixed
connectivity, distance constraints), just different topology (BCC vs triangle mesh).

---
Should I help you:

1. Implement XPBD distance constraints for your BCC lattice?
2. Compare code structure: force-based → constraint-based?
3. Document why XPBD fits better than SPH/MPM?

## SOLUTIONS

### Implicit Solvers for Stiff Springs

Backward Euler (Most Common)

- How it works: Solves for next state using forces at the future timestep
- Equation: M(v_{n+1} - v_n)/dt = f(x_{n+1})
- Requires: Solving nonlinear system via Newton-Raphson
- Stability: Unconditionally stable (any timestep size)
- Damping: Introduces artificial damping (energy loss)

BDF2 (2nd Order Backward Difference)

- Better than: Backward Euler (less damping)
- Used by: Baraff & Witkin 1998 cloth simulation
- Trade-off: More complex, needs 2 previous states

Variational/Symplectic Implicit

- Examples: Implicit midpoint, Newmark-β
- Benefit: Better energy conservation than Backward Euler
- Used in: High-quality physics (games usually skip this)

### XPBD Position Constraints (What the Paper Actually Uses)

This is KEY - the paper doesn't use spring forces at all! They use position-based constraints.

The Difference:

```python
Force-Based (What we are doing):
# Compute spring force
F = -k * (distance - rest_length) * direction
a = F / m
v += a * dt
x += v * dt

Position-Based (XPBD - What paper uses):
# Directly project positions to satisfy constraint
C(x) = distance - rest_length  # Constraint violation
Δx = -C(x) / ||∇C||²  # Position correction
x += Δx  # Move particles to satisfy constraint

XPBD Algorithm (from paper page 3):

Looking at Equations 4-7 in the paper:

# For each constraint (e.g., distance constraint between particles i,j):
Δλ = -C(x) / (∇C · M^-1 · ∇C^T + α̃)  # Eq 7
Δx = M^-1 · ∇C^T · Δλ                  # Eq 4
x_new = x + Δx                         # Apply correction

Where:
- C(x) = ||x_i - x_j|| - L_0 (distance constraint)
- ∇C = (x_i - x_j) / ||x_i - x_j|| (constraint gradient)
- α̃ = 1/(k·dt²) (compliance, inverse of stiffness)
- M^-1 = inverse mass matrix
```

Why XPBD is Better for Stiff Systems:

1. No stability limit - works like implicit solver
2. Simpler - no matrix assembly/factorization
3. Local - solves one constraint at a time
4. Handles inequality constraints easily (contacts, friction)

For Your Subatomic Simulation:

You have two viable paths:

#### PATH A: Implement XPBD (Recommended for your use case)

Pros:

- Unconditionally stable (handles any stiffness)
- Simpler than full implicit
- Works with your existing BCC neighbor structure
- Can still use Small Steps strategy

Cons:

- Major code refactor (constraint-based vs force-based)
- Need to understand Lagrange multipliers
- Iterative (but paper shows 1 iteration per substep works!)

Implementation outline:

```python
@ti.kernel
def solve_distance_constraint_xpbd(
    positions, masses, links, rest_length, 
    compliance, dt, granule_type
):
    for i in range(positions.shape[0]):
        if granule_type[i] == 0: continue  # Skip vertices

        for j in range(links_count[i]):
            neighbor = links[i, j]

            # Constraint: C = ||xi - xj|| - L0
            d = positions[neighbor] - positions[i]
            distance = d.norm()
            C = distance - rest_length

            # Gradient: ∇C = d / ||d||
            grad = d / distance

            # Compliance α̃ = 1/(k·dt²)
            alpha_tilde = 1.0 / (stiffness * dt * dt)

            # Lagrange multiplier increment (Eq 7)
            w_sum = (1/masses[i] + 1/masses[neighbor])
            delta_lambda = -C / (w_sum + alpha_tilde)

            # Position correction (Eq 4)
            delta_x_i = -(1/masses[i]) * grad * delta_lambda
            delta_x_j = (1/masses[neighbor]) * grad * delta_lambda

            # Apply corrections
            positions[i] += delta_x_i
            positions[neighbor] += delta_x_j

Then update velocities: v = (x_new - x_old) / dt
```

#### PATH B: Backward Euler (Full Implicit)

Pros:

- Unconditionally stable
- Can use larger timesteps
- Well-documented in literature

Cons:

- Very complex - requires:
  - Assembling global stiffness matrix
  - Newton-Raphson iterations
  - Linear solver (Conjugate Gradient or direct)
- Artificial damping (loses energy)
- Hard to get right

Not recommended for first attempt - XPBD is simpler and works better for your use case.

---
My Recommendation:

Try XPBD approach because:

1. ✅ Handles extreme stiffness (unconditionally stable)
2. ✅ Simpler than full implicit
3. ✅ Works with Small Steps (paper's whole point!)
4. ✅ Local solver (fits your neighbor-based structure)
5. ✅ Can use 1 iteration per substep (paper proves this works)

The paper's entire contribution is showing that XPBD with many substeps (1 iteration each) beats traditional implicit solvers (few
substeps, many iterations).

---
Ready for Tomorrow's Implementation:

Core XPBD algorithm to implement (from paper Eq 4-7):

For each distance constraint:

- C = ||xi - xj|| - L0           # Constraint violation
- α̃ = 1/(k·dt²)                  # Compliance (inverse stiffness)
- ∇C = (xi - xj)/||xi - xj||     # Gradient
- Δλ = -C / (∇C·M⁻¹·∇Cᵀ + α̃)    # Lagrange multiplier
- Δx = M⁻¹·∇Cᵀ·Δλ                # Position correction
- x_new = x + Δx                 # Apply correction

Key advantages over spring-mass:

1. No stability limit (works with ANY stiffness)
2. Simpler per-iteration (just position projection)
3. Can use realistic stiffness values
4. Still benefits from Small Steps (Δt² error reduction)

## XPBD SUMMARY

### How XPBD Propagates Waves WITHOUT Forces

#### The Fundamental Difference

```python
Force-Based (Newtons Laws):
F = ma → a = F/m → v += a·dt → x += v·dt
[Force causes acceleration → velocity → position]

XPBD (Constraint-Based):
C(x) = 0 → Δx to satisfy C → v = Δx/dt
[Constraint violation → position correction → velocity derived]
```

### Wave Propagation in XPBD (Step-by-Step)

#### 1. Constraints Replace Springs

Instead of:

```python
# Force-based:
F = -k × (distance - L₀)  # Spring force
```

You have:

```python
# XPBD:
C = distance - L₀  # Distance constraint
# "These two granules MUST be L₀ apart"
```

#### 2. How Momentum Transfers (The Magic!)

Let's trace a wave through 3 granules:

```python
Time t=0: Vertex oscillates
┌─────┐     ┌─────┐     ┌─────┐
│  V  │────▶│  A  │─────│  B  │
└─────┘     └─────┘     └─────┘
 moves       still      still
```

##### Step 1: Vertex moves (boundary condition)

```python
# Vertex moves outward by distance d
vertex_new = vertex_old + d
```

##### Step 2: Constraint between Vertex-A is violated

```python
C_VA = ||vertex_new - A|| - L₀
# Result: C_VA > 0 (too far apart!)
```

##### Step 3: XPBD corrects positions to satisfy constraint

```python
# Lagrange multiplier tells us "how hard to pull":
Δλ = -C_VA / (1/m_vertex + 1/m_A + α̃)

# Position corrections:
vertex_correction = -(1/m_vertex) × direction × Δλ
A_correction = +(1/m_A) × direction × Δλ

# Result: Vertex pulls back slightly, A moves toward vertex!
A_new = A_old + A_correction  # A moves!
```

##### Step 4: Now constraint A-B is violated

```python
C_AB = ||A_new - B|| - L₀
# Result: C_AB > 0 (A moved, now too far from B!)
```

##### Step 5: Constraint satisfaction propagates

```python
# A-B constraint correction:
B_correction = +(1/m_B) × direction × Δλ_AB
B_new = B_old + B_correction  # B moves!

# Wave has propagated: V → A → B
```

#### 3. Where's the Momentum?

**The brilliant part:** Momentum is **implicit** in position changes!

```python
# After constraint solving, positions changed:
Δx_A = A_new - A_old

# Velocity is derived from position change:
v_A = Δx_A / dt

# Momentum automatically emerges:
p_A = m_A × v_A = m_A × (Δx_A / dt)
```

**Momentum conservation happens through:**

1. **Mass weighting** in position corrections: `w = 1/m`
2. **Newton's 3rd law** is built-in: `Δx_i = -Δx_j` (equal/opposite)
3. **Velocity derived** from position changes

### Visual Analogy

#### Force-Based (Push)

```python
[V pushes A] → [A accelerates] → [A pushes B] → [B accelerates]
    F_VA          a_A = F/m         F_AB         a_B = F/m
```

**Problem:** Accumulates error, requires tiny timesteps for stability

#### XPBD (Constrained Dance)

```python
[V moves] → [A must follow (constraint)] → [B must follow] → ...
   x_V        C_VA forces A to move      C_AB forces B to move
```

**Advantage:** Directly enforces geometric relationships, stable!

### Why This Propagates Waves

#### Local Constraint → Global Wave

```python
# Each granule has 8 distance constraints to neighbors
# When one granule moves, ALL 8 constraints get violated
# Correction propagates to all 8 neighbors
# Each neighbor has 8 more constraints...
# → Wave spreads through lattice!
```

**Example with your vertex:**

```python
Frame 1:
Vertex oscillates → 8 neighbors move (satisfy constraint)

Frame 2:
8 neighbors moved → each has 7 more neighbors affected
                    (56 granules total)

Frame 3:
56 granules moved → each has connections
                    (hundreds affected)

...wave propagates spherically!
```

### The Physics is EQUIVALENT

#### Mathematical Proof

**Force-based spring:**

```python
F = -k(x - L₀)
a = F/m = -k(x - L₀)/m
```

**XPBD constraint (first iteration is equivalent!):**

```python
Δλ = -C / (1/m + α̃) = -(x - L₀) / (1/m + 1/(k·dt²))

# For small dt and large k:
α̃ = 1/(k·dt²) << 1/m

Δλ ≈ -(x - L₀) / (1/m) = -m(x - L₀)
Δx = (1/m)·Δλ ≈ -(x - L₀)

# Velocity from position change:
v = Δx/dt ≈ -(x - L₀)/dt

# This is equivalent to:
v += a·dt = -(k/m)(x - L₀)·dt  (spring-mass!)
```

**From the Small Steps paper:** "the first iteration of XPBD is equivalent to the first step of a Newton solver operating on the backward Euler equations."

**Translation:** XPBD IS doing implicit integration, just without solving the matrix!

### Summary

#### XPBD Still Uses

- ✅ Granules (particles) - same as before
- ✅ Masses - for position correction weighting
- ✅ Connectivity (8 neighbors) - same BCC lattice
- ✅ Stiffness (k) - encoded in compliance α̃ = 1/(k·dt²)

#### XPBD Does NOT Use

- ❌ Spring forces (F = -kx)
- ❌ Force accumulation
- ❌ Explicit velocity integration

#### How Waves Propagate

1. **Vertex moves** (boundary condition)
2. **Constraints violated** (distances too large/small)
3. **XPBD corrects positions** (pull granules together/apart)
4. **Corrections propagate** (neighbors move → their constraints violated → ...)
5. **Velocities derived** (v = Δx/dt)
6. **Momentum emerges** (p = m·v)
7. **Wave spreads** through lattice!

### The Conceptual Shift

**Force-Based Thinking:**

> "Forces cause acceleration, which integrates to velocity, which integrates to position"
> (Dynamic → Kinematic)

**Constraint-Based Thinking:**

> "Positions must satisfy geometric constraints, velocities are consequence of position changes"
> (Kinematic → Dynamic)

**Both are valid physics!** XPBD just solves it "backwards" - and turns out to be more stable for stiff systems.

## Implementation Status

### Files Created

- `xpbd.py` - Main render loop
- `ewave_xpbd.py` - XPBD constraint solver
- `medium_level0.py` - Unchanged (same BCC structure)

### Files Archived (for reference)

- `ewave_springs_euler.py` - Semi-implicit Euler integration
- `ewave_springs_leap.py` - Velocity Verlet attempt

### Implementation Checklist

**Step 1: XPBD Constraint Kernel** (Pending)

- Create `solve_distance_constraint_xpbd()` kernel
- Implement position corrections per Eq 4-7
- Handle vertex exclusion (boundary conditions)
- Use Jacobi iteration (parallel-safe for GPU)

**Step 2: Velocity Update** (Pending)

- Compute velocities from position changes: `v = (x_new - x_old) / dt`
- Store previous positions for velocity calculation

**Step 3: Small Steps Integration** (Pending)

- Split frame into substeps (start with 30-100)
- Single XPBD iteration per substep
- Update vertex boundary conditions per frame

**Step 4: Validation** (Pending)

- Test with realistic stiffness (no reduction needed!)
- Verify no NaN/Inf explosions
- Check wave propagation from vertices
- Measure wave speed vs expected c
- Measure wavelength vs expected λ

### Key Implementation Notes

**Jacobi vs Gauss-Seidel:**

- Paper uses Jacobi (parallel iteration)
- Read positions from previous iteration, write corrections to buffer
- Apply all corrections simultaneously (GPU-friendly)
- Gauss-Seidel (sequential) would be faster but not parallelizable

**Compliance Parameter α̃:**

- α̃ = 1/(k·dt²) where k is spring stiffness
- Lower α̃ = stiffer constraint (less compliance)
- With realistic k and small dt, α̃ will be very small → near-rigid constraints
- This is GOOD - we want stiff springs to maintain lattice structure

**Vertex Handling:**

- Vertices maintain prescribed harmonic motion
- Non-vertices solve XPBD constraints
- Vertices act as moving boundary conditions
- Energy injection happens through vertex-neighbor constraints

**Performance Expectations:**

- XPBD single iteration ≈ same cost as one force evaluation
- With 100 substeps: ~3000 constraint solves/second at 30 FPS
- Should be much faster than 200 substeps with 2 force evals (spring-mass)

## Detailed XPBD Mathematics (From Paper)

### Distance Constraint for BCC Lattice

For each pair of connected granules (i, j):

```python
Constraint function:
C(x_i, x_j) = ||x_i - x_j|| - L₀ = 0

Where:
- x_i, x_j = positions of granules i and j
- L₀ = rest length (BCC neighbor distance)
- ||·|| = Euclidean norm (distance)
```

### Constraint Gradient (∇C)

```python
∂C/∂x_i = (x_i - x_j) / ||x_i - x_j|| = n̂  (unit direction)
∂C/∂x_j = -(x_i - x_j) / ||x_i - x_j|| = -n̂

Where n̂ is the normalized direction vector from j to i
```

### Compliance (α̃)

```python
α̃ = 1/(k·Δt²)

Where:
- k = spring stiffness (N/m)
- Δt = substep timestep
- α̃ → 0 as k → ∞ (rigid constraint)
- α̃ → ∞ as k → 0 (no constraint)
```

### Lagrange Multiplier (Δλ)

From paper Equation 7:

```python
Δλ = -C(x) / (∇C·M⁻¹·∇Cᵀ + α̃)

For two particles:
Δλ = -C / (w_i + w_j + α̃)

Where:
- w_i = 1/m_i (inverse mass of particle i)
- w_j = 1/m_j (inverse mass of particle j)
- C = ||x_i - x_j|| - L₀ (current constraint violation)
```

### Position Corrections (Δx)

From paper Equation 4:

```python
Δx_i = w_i · n̂ · Δλ  (move particle i)
Δx_j = -w_j · n̂ · Δλ (move particle j in opposite direction)

Where:
- Positive Δλ → particles too far apart → pull together
- Negative Δλ → particles too close → push apart
- Heavier particles (larger m) move less (smaller w=1/m)
```

### Full Algorithm Per Constraint

```python
# 1. Current distance and violation
d = x_i - x_j
distance = ||d||
C = distance - L₀

# 2. Normalized direction
n̂ = d / distance

# 3. Inverse masses
w_i = 1 / m_i
w_j = 1 / m_j

# 4. Compliance
α̃ = 1 / (k · Δt²)

# 5. Lagrange multiplier
Δλ = -C / (w_i + w_j + α̃)

# 6. Position corrections
Δx_i = w_i · n̂ · Δλ
Δx_j = -w_j · n̂ · Δλ

# 7. Apply corrections
x_i_new = x_i + Δx_i
x_j_new = x_j - Δx_j

# 8. Update velocities (after all constraints)
v_i = (x_i_new - x_i_old) / Δt
```

### Physical Interpretation

**Why this works:**

1. **C** measures how much the constraint is violated (stretch/compression)
2. **Δλ** is the "force intensity" needed to fix the violation
3. **Δx** moves particles proportionally to their mobility (w=1/m)
4. **α̃** controls stiffness: small α̃ → nearly rigid constraint
5. Process repeats every substep, gradually satisfying all constraints

**Energy conservation:**

- With α̃=0 (infinite stiffness): perfectly rigid, energy-conserving
- With α̃>0 (finite stiffness): slight energy loss (numerical damping)
- Small substeps minimize this damping (paper's key finding!)

### BCC Lattice Application

For a granule with 8 neighbors:

```python
for each neighbor j in [0..7]:
    Apply distance constraint between granule i and neighbor j
    Accumulate position corrections

Final position = initial + sum of all 8 corrections
```

This creates a coupled system where:

- Each granule affected by all 8 neighbors
- Iterative solving propagates constraints through lattice
- Wave behavior emerges from constraint satisfaction dynamics

---

## Additional XPBD Implementation Details (From Unified Particle Physics Paper)

### Parallel Constraint Solving with Jacobi Iteration

**Key Issue**: Gauss-Seidel (sequential) vs Jacobi (parallel)

**From Unified Particle Physics (Section 4.1-4.3):**

- **Gauss-Seidel**: Sequential solving, better convergence, NOT GPU-friendly
- **Jacobi**: Parallel solving, slower convergence, but can diverge for rank-deficient systems
- **Solution**: Constraint averaging (mass-splitting) for stability

#### Constraint Averaging (Section 4.2)

Instead of immediately applying position corrections, accumulate them and average:

```python
# Phase 1: Accumulate deltas (parallel)
@ti.kernel
def solve_constraints_jacobi():
    for i in range(num_granules):
        delta[i] = vec3(0.0)
        count[i] = 0

        for j in range(8):  # 8 neighbors in BCC
            neighbor = links[i, j]
            # Compute constraint correction
            Δλ = compute_lagrange_multiplier(i, neighbor)
            Δx = compute_position_delta(i, Δλ)

            # Accumulate (don't apply yet!)
            delta[i] += Δx
            count[i] += 1

# Phase 2: Apply averaged corrections (parallel)
@ti.kernel
def apply_corrections():
    for i in range(num_granules):
        positions[i] += delta[i] / count[i]  # Average over constraints
```

**Why this works**: Prevents oscillation when multiple identical/near-identical constraints affect same particle.

#### Successive Over-Relaxation (SOR) - Section 4.3

Add user parameter ω to control convergence speed:

```python
# Apply with relaxation factor
positions[i] += (ω / count[i]) * delta[i]

# Typical values:
# ω = 1.0  → Standard averaging (safe)
# ω = 1.5  → Faster convergence (recommended for stiff systems)
# ω = 2.0  → Maximum over-relaxation (may diverge)
```

**For your Medium lattice**: Try ω = 1.5 to accelerate convergence with 8-neighbor connectivity.

---

### Handling Initial Constraint Violations (Section 4.4)

**Problem**: Starting with violated constraints adds artificial energy!

**Example**: Granule starts interpenetrating → XPBD projects it out → velocity derived from projection → false momentum!

**Solution**: Pre-stabilization pass

```python
# Algorithm 1, steps 10-15: Stabilization before main solve
@ti.kernel
def stabilize_initial_positions():
    # Use ORIGINAL positions (not predicted x*)
    for iteration in range(1, 2):  # 1-2 iterations sufficient
        solve_distance_constraints(positions_original)  # Not x*!

    # Apply same corrections to predicted positions
    x_star += (positions - positions_original)
```

**When to use**:

- ✅ Contact constraints (most visible source of error)
- ❌ Not needed for distance constraints in stable lattice (always satisfied)

**For your BCC lattice**: Probably not needed initially, but useful if implementing collision detection later.

---

### Particle Sleeping (Section 4.5)

**Problem**: Positional drift when constraints not fully converged.

**Solution**: Freeze particles below velocity threshold:

```python
@ti.kernel
def apply_sleeping(epsilon: float):
    for i in range(num_granules):
        if velocities[i].norm() < epsilon:
            positions[i] = positions_old[i]  # Freeze in place
        else:
            positions[i] = x_star[i]  # Apply new position
```

**For your Medium lattice**:

- Not needed for active wave region (vertices always moving)
- Could be useful for large lattice to "freeze" distant granules
- Set threshold: `epsilon = 1e-20 m/s` (Planck scale appropriate)

---

### Constraint Grouping for Faster Convergence (Section 4.3)

**Idea**: Process constraint types in groups, apply corrections between groups.

**From Algorithm 1 (steps 17-21)**:

```python
# Instead of single iteration over all constraints:
for iteration in range(solver_iterations):
    for constraint_group in constraint_groups:
        solve_constraint_group(constraint_group)  # Parallel within group
        apply_corrections()  # Apply before next group

# Example groups for BCC lattice:
# Group 1: Distance constraints (8-neighbors)
# Group 2: Contact constraints (if any)
# Group 3: Vertex boundary conditions (prescribed motion)
```

**Benefit**: Propagates corrections faster → fewer iterations needed.

**For your Medium lattice**:

- Start with single group (all distance constraints)
- If convergence slow, split by spatial regions

---

### Damping Formulation for Small Steps (From Small Steps Paper)

**Critical finding**: Many small steps can cause over-stiffness without damping!

**From Small Steps paper Eq. 8**:

```python
# Add explicit damping to velocities
velocities[i] *= damping_factor

# Typical values:
# damping = 0.999  → 0.1% energy loss per step
# damping = 0.99   → 1% energy loss per step
# damping = 1.0    → No damping (conservative, may oscillate)
```

**For your Medium lattice**: Start with `damping = 0.999` (per substep).

---

### Complete XPBD Algorithm (Unified Paper + Small Steps)

**Full algorithm combining both papers**:

```python
def simulate_frame(dt_frame: float, num_substeps: int):
    dt_sub = dt_frame / num_substeps

    # Pre-stabilization (if needed)
    for iteration in range(2):
        solve_constraints(positions)

    # Main substep loop
    for substep in range(num_substeps):
        # 1. Apply external forces (gravity, etc.)
        apply_forces(dt_sub)

        # 2. Predict positions (explicit)
        predict_positions(dt_sub)

        # 3. Update vertex boundary conditions
        update_vertex_motion(t_current)

        # 4. Solve constraints (XPBD core)
        for iteration in range(solver_iterations_per_substep):  # Usually 1!
            delta[:] = 0
            count[:] = 0

            # Solve all constraints in parallel (Jacobi)
            solve_distance_constraints_parallel(delta, count)

            # Apply with SOR
            apply_corrections_with_sor(delta, count, omega=1.5)

        # 5. Update velocities from position changes
        update_velocities(dt_sub, damping=0.999)

        # 6. Apply sleeping (optional)
        apply_sleeping(epsilon=1e-20)

        # 7. Update time
        t_current += dt_sub
```

---

### Performance Optimization Tips (From Unified Paper)

#### Memory Coherence (Section 9)

- Reorder particle data by spatial hash-grid cell index
- Improves cache locality for neighbor lookups
- Can give 2-3x speedup on GPU

```python
# Once per frame (or less frequently):
sort_particles_by_spatial_hash()
```

#### Atomic Operations vs Gather (Section 10, Algorithms 2-3)

**Constraint-Centric (scatter)**:

```python
# Each thread processes ONE constraint
for constraint in constraints:  # Parallel
    compute Δλ, ∇C
    for particle in constraint:
        atomicAdd(delta[particle], correction)  # Atomic write
```

**Particle-Centric (gather)**:

```python
# Each thread processes ONE particle
for particle in particles:  # Parallel
    for constraint in affecting_particle:
        compute correction
        delta += correction  # No atomics needed!
    positions[particle] += delta  # Single write
```

**For BCC lattice**: Use **particle-centric** (gather) approach:

- Each granule processes its 8 distance constraints
- No atomic operations needed
- Better performance on GPU

---

### Expected Performance (From Paper Table 1)

**Unified Particle Physics Examples**:

- 44k particles, 2 substeps, 2 iters → 3.8 ms/frame (263 FPS)
- 50k particles, 2 substeps, 4 iters → 10.1 ms/frame (100 FPS)

**For your 7×7×7 BCC lattice**:

- ~343 granules
- 100 substeps, 1 iter/substep
- Expected: < 1 ms/frame (1000+ FPS)

**Bottleneck will be rendering, not simulation!**

---

## Implementation Roadmap (Updated with Paper Insights)

### Phase 1: Basic XPBD Distance Constraints

- Read both papers
- Implement particle-centric Jacobi solver
- Add constraint averaging
- Test with realistic stiffness (k = 1e7 N/m, no reduction!)

### Phase 2: Optimization

- Add SOR parameter (ω = 1.5)
- Add velocity damping (0.999 per substep)
- Verify no explosions with extreme stiffness

### Phase 3: Validation

- Measure wave speed vs theoretical
- Measure wavelength vs driving frequency
- Compare energy conservation vs force-based

### Phase 4: Advanced Features (Future)

- Particle sleeping for large lattices
- Spatial hash reordering for performance
- Pre-stabilization for contact constraints

---

## Key Takeaways from Papers

### Small Steps Paper

1. **Many substeps + 1 iteration** > Few substeps + many iterations
2. **Explicit damping** needed for small timesteps
3. **Quadratic error reduction**: error ∝ dt²
4. **Jacobi iteration** works with proper averaging

### Unified Particle Physics Paper

1. **Constraint averaging** prevents Jacobi divergence
2. **SOR parameter** accelerates convergence (ω ∈ [1, 2])
3. **Pre-stabilization** prevents artificial energy injection
4. **Particle-centric** gather better than constraint-centric scatter
5. **Real-time capable** even with complex constraint networks

### Application to Medium lattice

**Your BCC lattice is structurally identical to cloth simulation**:

- Fixed connectivity (8 neighbors per granule)
- Distance constraints maintain lattice structure
- Vertices as moving boundary conditions
- XPBD handles extreme stiffness without instability

**Expected outcome**:

- ✅ Stable with realistic stiffness (k ~ 1e7 N/m)
- ✅ Wave propagation at correct speed
- ✅ Real-time performance (30-60 FPS)
- ✅ No frequency mismatch issues (XPBD is unconditionally stable)
