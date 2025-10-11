"""
QUANTUM-WAVE DYNAMICS
(AKA: PRANA @yoga, QI @daoism, JEDI FORCE @starwars)

Wave dynamics and motion physics for spacetime using XPBD (Extended Position-Based Dynamics).

XPBD Implementation based on:
1. "Small Steps in Physics Simulation" - Macklin & Müller (2019)
2. "Unified Particle Physics for Real-Time Applications" - Macklin et al. (2014)

Key advantages of XPBD for quantum lattice:
- Unconditionally stable (handles extreme stiffness)
- Correct wave propagation at speed of light
- No frequency mismatch issues
- Real-time performance with 100 substeps
"""

import taichi as ti

from openwave.common import constants

# ================================================================
# Quantum-Wave Oscillation Parameters
# ================================================================
amplitude_am = constants.QWAVE_AMPLITUDE / constants.ATTOMETTER  # am, oscillation amplitude
frequency = constants.QWAVE_SPEED / constants.QWAVE_LENGTH  # Hz, quantum-wave frequency


# ================================================================
# Quantum-Wave Driver Kernel (energy injection, harmonic oscillation, rhythm)
# ================================================================


@ti.kernel
def oscillate_vertex(
    positions: ti.template(),  # type: ignore
    velocities: ti.template(),  # type: ignore
    vertex_indices: ti.template(),  # type: ignore
    vertex_equilibrium: ti.template(),  # type: ignore
    vertex_directions: ti.template(),  # type: ignore
    t: ti.f32,  # type: ignore
    slow_mo: ti.f32,  # type: ignore
):
    """Injects energy into 8 vertices using harmonic oscillation (wave drivers, rhythm).

    Vertices oscillate radially along direction vectors toward/away from lattice center.
    Position: x(t) = x_eq + A·cos(ωt + φ)·direction
    Velocity: v(t) = -A·ω·sin(ωt + φ)·direction (derivative of position)

    Args:
        positions: Position field for all granules
        velocities: Velocity field for all granules
        vertex_indices: Indices of 8 corner vertices
        vertex_equilibrium: Equilibrium positions of 8 vertices
        vertex_directions: Normalized direction vectors from vertices to center
        t: Current simulation time (accumulated)
    """
    f_slowed = frequency / slow_mo
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency

    for v in range(8):
        idx = vertex_indices[v]
        direction = vertex_directions[v]

        # Add phase shift to break symmetry: each vertex oscillates with different phase
        # Phase = v * π/4 (45° increments for 8 vertices)
        phase = float(v) * ti.math.pi / 4.0

        # Position: x(t) = x_eq + A·cos(ωt + φ)·direction
        displacement = amplitude_am * ti.cos(omega * t + phase)
        positions[idx] = vertex_equilibrium[v] + displacement * direction

        # Velocity: v(t) = -A·ω·sin(ωt + φ)·direction (derivative of position)
        velocity_magnitude = -amplitude_am * omega * ti.sin(omega * t + phase)
        velocities[idx] = velocity_magnitude * direction


# ================================================================
# XPBD Constraint Solving (replaces force-based dynamics)
# ================================================================


@ti.kernel
def solve_distance_constraints_xpbd(
    positions: ti.template(),  # type: ignore
    prev_positions: ti.template(),  # type: ignore
    masses: ti.f32,  # type: ignore
    links: ti.template(),  # type: ignore
    links_count: ti.template(),  # type: ignore
    rest_length: ti.f32,  # type: ignore
    stiffness: ti.f32,  # type: ignore
    granule_type: ti.template(),  # type: ignore
    dt: ti.f32,  # type: ignore
    omega: ti.f32,  # type: ignore
    position_delta: ti.template(),  # type: ignore
    constraint_count: ti.template(),  # type: ignore
):
    """XPBD distance constraint solver with Jacobi iteration + constraint averaging.

    Based on "Unified Particle Physics" paper (Section 4.1-4.3):
    - Particle-centric approach (gather): Each granule processes its 8 neighbors
    - Constraint averaging: Accumulate corrections, divide by constraint count
    - SOR (Successive Over-Relaxation): omega parameter for faster convergence

    For each distance constraint between granules i and j:
    1. C(x) = ||xi - xj|| - L0           (constraint violation)
    2. α̃ = 1/(k·dt²)                     (compliance, inverse stiffness)
    3. ∇C = (xi - xj)/||xi - xj||         (constraint gradient)
    4. Δλ = -C / (wi + wj + α̃)           (Lagrange multiplier)
    5. Δxi = wi · ∇C · Δλ                 (position correction for i)
    6. Δxj = -wj · ∇C · Δλ                (position correction for j)

    Args:
        positions: Current positions (predicted x*)
        prev_positions: Previous positions (for velocity computation)
        masses: Granule mass (scalar, same for all)
        links: Connectivity matrix
        links_count: Number of links per granule
        rest_length: BCC neighbor distance L0
        stiffness: Spring constant k (N/m) - now can use REAL physical value!
        granule_type: Skip vertices (prescribed motion)
        dt: Substep timestep
        omega: SOR parameter (1.5 recommended)
        position_delta: Accumulated position corrections (output)
        constraint_count: Number of constraints per granule (output)
    """
    # Phase 1: Accumulate position deltas (PARALLEL - Jacobi iteration)
    for i in range(positions.shape[0]):
        # Skip vertices - they have prescribed motion from oscillate_vertex()
        if granule_type[i] == 0:  # TYPE_VERTEX = 0
            position_delta[i] = ti.Vector([0.0, 0.0, 0.0])
            constraint_count[i] = 0
            continue

        # Initialize accumulation
        delta_sum = ti.Vector([0.0, 0.0, 0.0])
        count = 0

        # Inverse mass (same for all granules)
        wi = 1.0 / masses

        # Compliance parameter: α̃ = 1/(k·dt²)
        alpha_tilde = 1.0 / (stiffness * dt * dt)

        # Process all 8 distance constraints for this granule
        num_links = links_count[i]
        for j in range(num_links):
            neighbor_idx = links[i, j]
            if neighbor_idx >= 0:  # Valid connection
                # Constraint: C = ||xi - xj|| - L0
                d = positions[i] - positions[neighbor_idx]
                distance = d.norm()

                if distance > 1e-12:  # Avoid division by zero
                    # Constraint violation (positive = stretched, negative = compressed)
                    C = distance - rest_length

                    # Constraint gradient: ∇C = d / ||d||
                    grad_C = d / distance  # Unit vector pointing from j to i

                    # Neighbor inverse mass
                    wj = 1.0 / masses

                    # Lagrange multiplier: Δλ = -C / (wi + wj + α̃)
                    delta_lambda = -C / (wi + wj + alpha_tilde)

                    # Position correction for granule i: Δxi = wi · ∇C · Δλ
                    delta_xi = wi * grad_C * delta_lambda

                    # Accumulate correction
                    delta_sum += delta_xi
                    count += 1

        # Store accumulated delta and count
        position_delta[i] = delta_sum
        constraint_count[i] = count


@ti.kernel
def apply_position_corrections(
    positions: ti.template(),  # type: ignore
    position_delta: ti.template(),  # type: ignore
    constraint_count: ti.template(),  # type: ignore
    granule_type: ti.template(),  # type: ignore
    omega: ti.f32,  # type: ignore
):
    """Apply averaged position corrections with SOR.

    Phase 2 of XPBD Jacobi iteration: Apply accumulated corrections
    with constraint averaging and over-relaxation.

    Δx_final = (ω / count) · Σ Δx_constraints

    Args:
        positions: Position field (updated in-place)
        position_delta: Accumulated corrections from solve_distance_constraints_xpbd
        constraint_count: Number of constraints per granule
        granule_type: Skip vertices
        omega: SOR parameter (1.0 = standard averaging, 1.5 = over-relaxation)
    """
    for i in range(positions.shape[0]):
        # Skip vertices
        if granule_type[i] == 0:
            continue

        # Apply averaged correction with SOR
        count = constraint_count[i]
        if count > 0:
            # Constraint averaging with SOR: Δx = (ω/n) · Σ Δx
            positions[i] += (omega / ti.f32(count)) * position_delta[i]


@ti.kernel
def update_velocities_from_positions(
    positions: ti.template(),  # type: ignore
    prev_positions: ti.template(),  # type: ignore
    velocities: ti.template(),  # type: ignore
    granule_type: ti.template(),  # type: ignore
    dt: ti.f32,  # type: ignore
    damping: ti.f32,  # type: ignore
):
    """Derive velocities from position changes with damping.

    In XPBD, velocity is derived from position changes:
    v = (x_new - x_old) / dt

    Explicit damping is added per "Small Steps" paper:
    v_damped = damping · v

    Args:
        positions: New positions (after constraint solve)
        prev_positions: Old positions (before constraint solve)
        velocities: Velocity field (updated in-place)
        granule_type: Skip vertices
        dt: Substep timestep
        damping: Damping coefficient (0.999 recommended)
    """
    for i in range(positions.shape[0]):
        # Skip vertices - they have prescribed velocity from oscillate_vertex()
        if granule_type[i] == 0:
            continue

        # Velocity from position change
        velocity_raw = (positions[i] - prev_positions[i]) / dt

        # Apply damping
        velocities[i] = damping * velocity_raw


# Orchestrator function to run the full XPBD propagation step
def propagate_qwave(
    lattice,
    granule,
    neighbors,
    stiffness,
    t: float,
    dt: float,
    substeps: int = 100,
    slow_mo: float = 1.0,
    damping: float = 0.999,
    omega: float = 1.5,
):
    """Main wave propagation using XPBD with Small Steps strategy.

    Implements XPBD constraint solving from:
    - "Small Steps in Physics Simulation" (substep strategy)
    - "Unified Particle Physics" (Jacobi + averaging + SOR)

    XPBD Algorithm per substep:
    1. Update vertex boundary conditions (once per frame)
    2. Save previous positions (for velocity computation)
    3. Solve distance constraints (Jacobi iteration):
       a. Accumulate position corrections (parallel)
       b. Apply averaged corrections with SOR (parallel)
    4. Update velocities from position changes
    5. Apply damping

    Key advantages:
    - Unconditionally stable (no timestep limit!)
    - Can use REAL physical stiffness (k ~ 2.66e21 N/m for 1M particles)
    - Correct wave speed (speed of light)
    - Real-time performance (100 substeps, 1 iteration each)

    Args:
        lattice: Lattice instance with positions, velocities, granule_type
        granule: Granule instance for mass
        neighbors: BCCNeighbors instance with connectivity
        stiffness: Spring constant k (N/m) - USE REAL PHYSICAL VALUE!
        t: Current simulation time
        dt: Frame timestep (real-time from clock)
        substeps: Number of substeps (100 recommended)
        slow_mo: Slow motion factor for time microscope
        damping: Velocity damping per substep (0.999 recommended)
        omega: SOR parameter for faster convergence (1.5 recommended)
    """
    # Allocate temporary fields for XPBD (if not already created)
    if not hasattr(lattice, "prev_positions_am"):
        lattice.prev_positions_am = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
        lattice.position_delta_am = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
        lattice.constraint_count = ti.field(dtype=ti.i32, shape=lattice.total_granules)

    # Substep timestep (Small Steps strategy)
    dt_sub = dt / substeps

    # Update vertex positions ONCE per frame (boundary condition)
    oscillate_vertex(
        lattice.positions_am,
        lattice.velocities_am,
        lattice.vertex_indices,
        lattice.vertex_equilibrium_am,
        lattice.vertex_directions,
        t,
        slow_mo,
    )

    # XPBD substep loop (following "Small Steps" paper Algorithm 1)
    for step in range(substeps):
        # Save previous positions for velocity computation
        lattice.prev_positions_am.copy_from(lattice.positions_am)

        # XPBD constraint solve (SINGLE iteration per substep)
        # Phase 1: Accumulate position deltas (Jacobi iteration)
        solve_distance_constraints_xpbd(
            lattice.positions_am,
            lattice.prev_positions_am,
            granule.mass,
            neighbors.links,
            neighbors.links_count,
            neighbors.rest_length_am,
            stiffness * constants.ATTOMETTER,  # Convert to N/am for attometer units
            lattice.granule_type,
            dt_sub,
            omega,
            lattice.position_delta_am,  # Output: accumulated deltas
            lattice.constraint_count,  # Output: constraint counts
        )

        # Phase 2: Apply averaged corrections with SOR
        apply_position_corrections(
            lattice.positions_am,
            lattice.position_delta_am,
            lattice.constraint_count,
            lattice.granule_type,
            omega,
        )

        # Update velocities from position changes with damping
        update_velocities_from_positions(
            lattice.positions_am,
            lattice.prev_positions_am,
            lattice.velocities_am,
            lattice.granule_type,
            dt_sub,
            damping,
        )
