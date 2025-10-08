"""
QUANTUM-WAVE
(AKA: PRANA @yoga, QI @daoism, JEDI FORCE @starwars)

Wave dynamics and motion physics for spacetime granules.
OpenWave proprietary physics engine.
"""

import taichi as ti

from openwave.common import constants

# ================================================================
# Quantum-Wave Oscillation Parameters
# ================================================================
UNIT_SCALE = 1e18  # meters to attometers (must match spacetime.py)
amplitude = constants.QWAVE_AMPLITUDE * UNIT_SCALE  # am, oscillation amplitude
frequency = constants.QWAVE_SPEED / constants.QWAVE_LENGTH  # Hz, quantum-wave frequency
# slow-motion factor (divides frequency for human-visible motion, time microscope)
slow_mo = 1e25  # (1 = real-time, 10 = 10x slower, 1e25 = 10 * trillions * trillions FPS)


# ================================================================
# Quantum-Wave Driver Kernel (harmonic oscillations)
# ================================================================


@ti.kernel
def oscillate_vertex(
    positions: ti.template(),  # type: ignore
    velocities: ti.template(),  # type: ignore
    vertex_indices: ti.template(),  # type: ignore
    vertex_equilibrium: ti.template(),  # type: ignore
    vertex_directions: ti.template(),  # type: ignore
    t: ti.f32,  # type: ignore
):
    """Update 8 vertex positions and velocities using harmonic oscillation (wave drivers).

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
        displacement = amplitude * ti.cos(omega * t + phase)
        positions[idx] = vertex_equilibrium[v] + displacement * direction

        # Velocity: v(t) = -A·ω·sin(ωt + φ)·direction (derivative of position)
        velocity_magnitude = -amplitude * omega * ti.sin(omega * t + phase)
        velocities[idx] = velocity_magnitude * direction


# ================================================================
# Quantum-Wave Propagation (spring-mass dynamics)
# ================================================================


@ti.kernel
def compute_spring_forces(
    positions: ti.template(),  # type: ignore
    mass: ti.f32,  # type: ignore
    spring_links: ti.template(),  # type: ignore
    spring_links_count: ti.template(),  # type: ignore
    rest_length: ti.f32,  # type: ignore
    accelerations: ti.template(),  # type: ignore
    stiffness: ti.f32,  # type: ignore
):
    """Compute spring forces and accelerations for all non-vertex granules.

    For each granule, calculates resultant force from connected springs:
    F = sum over neighbors: -k * (distance - L0) * direction_unit_vector

    Args:
        positions: Position field for all granules
        spring_links: Connectivity matrix [granule_idx, neighbor_idx]
        spring_links_count: Number of active links per granule
        accelerations: Output acceleration field (F/m)
        stiffness: Spring constant k (N/am)
        rest_length: Spring rest length (unit_cell_edge * sqrt(3)/2 for BCC)
        mass: Granule mass
    """
    for i in range(positions.shape[0]):
        # Compute forces for ALL granules (including vertices for debug)
        # Note: vertices will have their motion overridden by oscillate_vertex anyway

        # Accumulate spring forces from all neighbors
        total_force = ti.Vector([0.0, 0.0, 0.0])
        num_links = spring_links_count[i]

        for j in range(num_links):
            neighbor_idx = spring_links[i, j]
            if neighbor_idx >= 0:  # Valid connection
                # Displacement vector from current to neighbor
                d = positions[neighbor_idx] - positions[i]
                distance = d.norm()

                if distance > 1e-12:  # Avoid division by zero
                    # Spring extension (positive = stretched, negative = compressed)
                    extension = distance - rest_length

                    # Spring force: F = -k * x * direction
                    force_magnitude = -stiffness * extension
                    force_direction = d / distance  # Unit vector
                    total_force += force_magnitude * force_direction

        # Acceleration = F / m
        acc = total_force / mass
        accelerations[i] = acc


@ti.kernel
def integrate_motion(
    positions: ti.template(),  # type: ignore
    velocities: ti.template(),  # type: ignore
    granule_type: ti.template(),  # type: ignore
    accelerations: ti.template(),  # type: ignore
    dt: ti.f32,  # type: ignore
):
    """Integrate equations of motion using Leapfrog (Velocity Verlet) method.

    Leapfrog is a symplectic integrator that conserves energy in oscillatory systems.
    Uses kick-drift-kick pattern:
    1. Half-step velocity update (kick)
    2. Full-step position update (drift)
    3. Compute new accelerations (done externally)
    4. Final half-step velocity update (kick)

    This function performs steps 1-2, then caller recomputes accelerations, then
    call this again for final velocity update. OR use two-stage approach.

    For simplicity, using velocity Verlet single-call version:
    v(t+dt/2) = v(t) + a(t)*dt/2
    x(t+dt) = x(t) + v(t+dt/2)*dt
    (compute a(t+dt) externally)
    v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2

    Args:
        positions: Position field
        velocities: Velocity field
        accelerations: Acceleration field (from spring forces)
        granule_type: Type classification (skip vertices)
        dt: Timestep
    """
    for i in range(positions.shape[0]):
        # Skip vertices - they have prescribed motion from oscillate_vertex()
        if granule_type[i] == 0:  # TYPE_VERTEX = 0
            continue

        # Damped Leapfrog integration for numerical stability
        # Light damping prevents instability in extremely stiff spring system
        damping = 0.999  # 0.1% energy loss per step
        velocities[i] = damping * (velocities[i] + accelerations[i] * dt)
        positions[i] += velocities[i] * dt


# Global acceleration field for wave propagation (allocated once)
accelerations = None


def initialize_propagation(total_granules: int):
    """Initialize acceleration field for wave propagation."""
    global accelerations
    accelerations = ti.Vector.field(3, dtype=ti.f32, shape=total_granules)


# Orchestrator function to run the full propagation step
def propagate_qwave(
    lattice,
    granule,
    springs,
    stiffness,
    t: float,
    dt: float,
    substeps: int = 1000,
):
    """Main wave propagation orchestrator using spring-mass dynamics.

    Propagates quantum waves through the lattice using:
    - Vertex boundary conditions (harmonic oscillators inject energy)
    - Spring-mass dynamics for non-vertex granules
    - Leapfrog integration for energy conservation

    Args:
        lattice: Lattice instance with positions, velocities, granule_type
        springs: Spring instance with connectivity and stiffness
        granule: Granule instance for mass
        t: Current simulation time
        dt: Frame timestep
        substeps: Number of substeps per frame for stability
    """
    global accelerations

    # Initialize on first call
    if accelerations is None:
        initialize_propagation(lattice.total_granules)
        print(f"[WAVE_PROPAGATION_INITIALIZED] All 8 vertices will oscillate with phase shifts")

    # Substep for numerical stability
    dt_sub = dt / substeps

    # Update vertex positions ONCE per frame (not per substep)
    # Vertices provide boundary condition for entire frame
    oscillate_vertex(
        lattice.positions,
        lattice.velocities,
        lattice.vertex_indices,
        lattice.vertex_equilibrium,
        lattice.vertex_directions,
        t,
    )

    for step in range(substeps):

        # Step 2: Compute spring forces on all granules (vertices get zero acceleration)
        # Scale rest_length to attometers to match position units
        # Scale stiffness to N/am (from N/m) to match extension units
        compute_spring_forces(
            lattice.positions,
            granule.mass,
            springs.links,
            springs.links_count,
            springs.rest_length * UNIT_SCALE,
            accelerations,
            stiffness / UNIT_SCALE,
        )

        # Step 3: Integrate motion for non-vertex granules
        # dt stays in seconds - no scaling needed (positions in am, velocities in am/s)
        integrate_motion(
            lattice.positions,
            lattice.velocities,
            lattice.granule_type,
            accelerations,
            dt_sub,
        )

        # Debug: Check granule 1 (neighbor to vertex 0) periodically
        if step == 0:
            # Print every ~0.01 seconds
            if abs(t - round(t * 100) / 100) < 0.0001:
                g1_pos_eq = ti.Vector([0.0, 0.0, 42.857143])  # Equilibrium for granule 1
                g1_disp = lattice.positions[1] - g1_pos_eq
                disp_norm = g1_disp.norm()
                vel_norm = lattice.velocities[1].norm()
                # Check if values are finite
                is_finite = disp_norm == disp_norm and vel_norm == vel_norm  # NaN check
                status = "OK" if is_finite else "NaN/Inf!"
                print(
                    f"[t={t:.3f}] g1: disp={disp_norm:.2e} am, vel={vel_norm:.2e} am/s [{status}]"
                )
