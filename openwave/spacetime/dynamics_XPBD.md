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
