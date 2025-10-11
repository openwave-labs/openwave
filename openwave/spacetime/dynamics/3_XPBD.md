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
How They Relate to Your Quantum Lattice:

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
For Your Quantum Lattice:

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

For Your Quantum Simulation:

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
