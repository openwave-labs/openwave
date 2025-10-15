"""
QUANTUM-WAVE
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
import numpy as np

from openwave.common import constants

# ================================================================
# Quantum-Wave Oscillation Parameters
# ================================================================
amplitude_am = constants.QWAVE_AMPLITUDE / constants.ATTOMETTER  # am, oscillation amplitude
frequency = constants.QWAVE_SPEED / constants.QWAVE_LENGTH  # Hz, quantum-wave frequency


# ================================================================
# Quantum-Wave Source Kernel (energy injection, harmonic oscillation, rhythm)
# ================================================================


@ti.kernel
def oscillate_vertex(
    position: ti.template(),  # type: ignore
    velocity: ti.template(),  # type: ignore
    vertex_index: ti.template(),  # type: ignore
    vertex_equilibrium: ti.template(),  # type: ignore
    vertex_center_direction: ti.template(),  # type: ignore
    t: ti.f32,  # type: ignore
    slow_mo: ti.f32,  # type: ignore
    amp_boost: ti.f32,  # type: ignore
):
    """Injects energy into 8 vertices using harmonic oscillation (wave source, rhythm).

    Vertices oscillate radially along direction vectors toward/away from lattice center.
    Position: x(t) = x_eq + A·cos(ωt + φ)·direction
    Velocity: v(t) = -A·ω·sin(ωt + φ)·direction (derivative of position)

    Args:
        position: Position field for all granules
        velocity: Velocity field for all granules
        vertex_index: Indices of 8 corner vertices
        vertex_equilibrium: Equilibrium position of 8 vertices
        vertex_center_direction: Normalized direction vectors from vertices to center
        t: Current simulation time (accumulated)
        slow_mo: Slow motion factor
        amp_boost: Multiplier for oscillation amplitude (for visibility in scaled lattices)
    """
    f_slowed = frequency / slow_mo
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency

    for v in range(8):
        idx = vertex_index[v]
        direction = vertex_center_direction[v]

        # Add phase shift to break symmetry: each vertex oscillates with different phase
        # Phase = v * π/4 (45° increments for 8 vertices)
        phase = float(v) * ti.math.pi / 4.0

        # Position: x(t) = x_eq + A·cos(ωt + φ)·direction
        # Apply amp_boost for visibility in scaled-up lattices
        displacement = amplitude_am * amp_boost * ti.cos(omega * t + phase)
        position[idx] = vertex_equilibrium[v] + displacement * direction

        # Velocity: v(t) = -A·ω·sin(ωt + φ)·direction (derivative of position)
        velocity_magnitude = -amplitude_am * amp_boost * omega * ti.sin(omega * t + phase)
        velocity[idx] = velocity_magnitude * direction


# ================================================================
# XPBD Constraint Solving (replaces force-based dynamics)
# ================================================================


@ti.kernel
def solve_distance_constraints_xpbd(
    position: ti.template(),  # type: ignore
    prev_position: ti.template(),  # type: ignore
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
        position: Current position (predicted x*)
        prev_position: Previous position (for velocity computation)
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
    for i in range(position.shape[0]):
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
                d = position[i] - position[neighbor_idx]
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
    position: ti.template(),  # type: ignore
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
        position: Position field (updated in-place)
        position_delta: Accumulated corrections from solve_distance_constraints_xpbd
        constraint_count: Number of constraints per granule
        granule_type: Skip vertices
        omega: SOR parameter (1.0 = standard averaging, 1.5 = over-relaxation)
    """
    for i in range(position.shape[0]):
        # Skip vertices
        if granule_type[i] == 0:
            continue

        # Apply averaged correction with SOR
        count = constraint_count[i]
        if count > 0:
            # Constraint averaging with SOR: Δx = (ω/n) · Σ Δx
            position[i] += (omega / ti.f32(count)) * position_delta[i]


@ti.kernel
def update_velocity_from_position(
    position: ti.template(),  # type: ignore
    prev_position: ti.template(),  # type: ignore
    velocity: ti.template(),  # type: ignore
    granule_type: ti.template(),  # type: ignore
    dt: ti.f32,  # type: ignore
    damping: ti.f32,  # type: ignore
):
    """Derive velocity from position changes with damping.

    In XPBD, velocity is derived from position changes:
    v = (x_new - x_old) / dt

    Explicit damping is added per "Small Steps" paper:
    v_damped = damping · v

    Args:
        position: New position (after constraint solve)
        prev_position: Old position (before constraint solve)
        velocity: Velocity field (updated in-place)
        granule_type: Skip vertices
        dt: Substep timestep
        damping: Damping coefficient (0.999 recommended)
    """
    for i in range(position.shape[0]):
        # Skip vertices - they have prescribed velocity from oscillate_vertex()
        if granule_type[i] == 0:
            continue

        # Velocity from position change
        velocity_raw = (position[i] - prev_position[i]) / dt

        # Apply damping
        velocity[i] = damping * velocity_raw


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
    amp_boost: float = 1.0,
):
    """Main wave propagation using XPBD with Small Steps strategy.

    Implements XPBD constraint solving from:
    - "Small Steps in Physics Simulation" (substep strategy)
    - "Unified Particle Physics" (Jacobi + averaging + SOR)

    XPBD Algorithm per substep:
    1. Update vertex boundary conditions (once per frame)
    2. Save previous position (for velocity computation)
    3. Solve distance constraints (Jacobi iteration):
       a. Accumulate position corrections (parallel)
       b. Apply averaged corrections with SOR (parallel)
    4. Update velocity from position changes
    5. Apply damping

    Key advantages:
    - Unconditionally stable (no timestep limit!)
    - Can use REAL physical stiffness (k ~ 2.66e21 N/m for 1M particles)
    - Correct wave speed (speed of light)
    - Real-time performance (100 substeps, 1 iteration each)

    Args:
        lattice: Lattice instance with position, velocity, granule_type
        granule: Granule instance for mass
        neighbors: BCCNeighbors instance with connectivity
        stiffness: Spring constant k (N/m) - USE REAL PHYSICAL VALUE!
        t: Current simulation time
        dt: Frame timestep (real-time from clock)
        substeps: Number of substeps (100 recommended)
        slow_mo: Slow motion factor for time microscope
        damping: Velocity damping per substep (0.999 recommended)
        omega: SOR parameter for faster convergence (1.5 recommended)
        amp_boost: Multiplier for oscillation amplitude (1.0 = physical, >1 = visible)
    """
    # Allocate temporary fields for XPBD (if not already created)
    if not hasattr(lattice, "prev_position_am"):
        lattice.prev_position_am = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
        lattice.position_delta_am = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
        lattice.constraint_count = ti.field(dtype=ti.i32, shape=lattice.total_granules)

    # Substep timestep (Small Steps strategy)
    dt_sub = dt / substeps

    # Update vertex positions ONCE per frame (boundary condition)
    oscillate_vertex(
        lattice.position_am,
        lattice.velocity_am,
        lattice.vertex_index,
        lattice.vertex_equilibrium_am,
        lattice.vertex_center_direction,
        t,
        slow_mo,
        amp_boost,
    )

    # XPBD substep loop (following "Small Steps" paper Algorithm 1)
    for step in range(substeps):
        # Save previous position for velocity computation
        lattice.prev_position_am.copy_from(lattice.position_am)

        # XPBD constraint solve (SINGLE iteration per substep)
        # Phase 1: Accumulate position deltas (Jacobi iteration)
        solve_distance_constraints_xpbd(
            lattice.position_am,
            lattice.prev_position_am,
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
            lattice.position_am,
            lattice.position_delta_am,
            lattice.constraint_count,
            lattice.granule_type,
            omega,
        )

        # Update velocity from position changes with damping
        update_velocity_from_position(
            lattice.position_am,
            lattice.prev_position_am,
            lattice.velocity_am,
            lattice.granule_type,
            dt_sub,
            damping,
        )


# ================================================================
# Wave Diagnostics
# ================================================================

# Global diagnostic state
_diagnostic_state = {
    "measurement_interval": 1.0,  # seconds (wallclock time)
    "last_measurement_time": 0.0,
    "first_measurement": None,
    "measurement_count": 0,
}


def init_wave_diagnostics(measurement_interval: float = 1.0):
    """Initialize wave diagnostics system.

    Args:
        measurement_interval: Time between measurements in seconds (wallclock)
    """
    global _diagnostic_state
    _diagnostic_state = {
        "measurement_interval": measurement_interval,
        "last_measurement_time": 0.0,
        "first_measurement": None,
        "measurement_count": 0,
    }


def probe_wave_diagnostics(
    lattice,
    neighbors,
    t: float,
    current_time: float,
    slow_mo: float,
):
    """Probe wave diagnostics and print measurements.

    Call this every frame in the main loop. It will automatically
    measure and print diagnostics at the configured interval.

    Args:
        lattice: Lattice instance
        neighbors: Neighbors instance (for expected wavelength)
        t: Simulation time
        current_time: Wallclock time
        slow_mo: Slow motion factor
    """
    global _diagnostic_state

    # Check if it's time for a measurement
    if (
        current_time - _diagnostic_state["last_measurement_time"]
        < _diagnostic_state["measurement_interval"]
    ):
        return

    _diagnostic_state["last_measurement_time"] = current_time
    _diagnostic_state["measurement_count"] += 1

    # Measure wave speed (track wavefront propagation)
    wave_data = measure_wave_speed(lattice, threshold_fraction=0.1, slow_mo=slow_mo)

    if _diagnostic_state["first_measurement"] is None and wave_data["max_distance_am"] > 0:
        _diagnostic_state["first_measurement"] = {
            "time_sim": t,
            "time_real": current_time,
            "distance_am": wave_data["max_distance_am"],
        }
    elif (
        _diagnostic_state["first_measurement"] is not None
        and wave_data["max_distance_am"] > _diagnostic_state["first_measurement"]["distance_am"]
    ):
        # Calculate wave speed
        dt_sim = t - _diagnostic_state["first_measurement"]["time_sim"]
        distance_traveled_am = (
            wave_data["max_distance_am"] - _diagnostic_state["first_measurement"]["distance_am"]
        )
        distance_traveled_m = distance_traveled_am * constants.ATTOMETTER

        # Convert simulation time to real time (account for SLOW_MO)
        # SLOW_MO slows down time, so to get real time: divide by SLOW_MO
        dt_real = dt_sim / slow_mo
        wave_speed = distance_traveled_m / dt_real if dt_real > 0 else 0

        # Calculate error vs speed of light
        speed_error = abs(wave_speed - constants.QWAVE_SPEED) / constants.QWAVE_SPEED * 100

        print(f"\n{'='*70}")
        print(f"WAVE SPEED MEASUREMENT #{_diagnostic_state['measurement_count']}")
        print(f"{'='*70}")
        print(f"Wave speed: {wave_speed:.3e} m/s")
        print(f"Expected:   {constants.QWAVE_SPEED:.3e} m/s (speed of light)")
        print(f"Error:      {speed_error:.1f}%")
        print(f"Distance traveled: {distance_traveled_m:.3e} m ({distance_traveled_am:.1f} am)")
        print(f"Time elapsed (sim): {dt_sim:.3e} s (real: {dt_real:.3e} s)")
        print(f"{'='*70}\n")

    # Measure wavelength (less frequent - every 5 measurements)
    # DISABLED: Still has O(peaks²) complexity in peak-to-peak distance calculation
    # TODO: Fix peak-to-peak distance to use spatial hashing or k-d tree
    if False and _diagnostic_state["measurement_count"] % 5 == 0:
        wavelength_data = measure_wavelength(lattice, neighbors, num_samples=100)

        # Debug output to understand peak detection
        print(f"\n[DEBUG] Wavelength Measurement Diagnostics (3D peak detection):")
        print(f"  Local maxima detected: {wavelength_data['num_peaks']}")
        print(
            f"  Max displacement: {wavelength_data['max_displacement']:.3f} am "
            f"({wavelength_data['max_displacement']/neighbors.rest_length_am*100:.1f}% of rest_length)"
        )
        print(f"  Peak threshold: {wavelength_data['threshold_displacement']:.3f} am (30% of max)")
        if len(wavelength_data["peak_position"]) > 0:
            print(
                f"  Peak-to-peak distances: min={np.min(wavelength_data['peak_position']):.1f} am, "
                f"avg={np.mean(wavelength_data['peak_position']):.1f} am, "
                f"max={np.max(wavelength_data['peak_position']):.1f} am"
            )

        if wavelength_data["num_peaks"] >= 1:  # Show even single peak
            wavelength_m = wavelength_data["wavelength_am"] * constants.ATTOMETTER
            expected_wavelength_m = 2 * neighbors.rest_length  # λ_lattice = 2L
            wavelength_error = (
                abs(wavelength_m - expected_wavelength_m) / expected_wavelength_m * 100
                if wavelength_data["num_peaks"] >= 2
                else 0
            )

            print(f"\n{'='*70}")
            print(f"WAVELENGTH MEASUREMENT (3D Interference Pattern)")
            print(f"{'='*70}")
            if wavelength_data["num_peaks"] >= 2:
                print(
                    f"Measured wavelength: {wavelength_m:.3e} m ({wavelength_data['wavelength_am']:.1f} am)"
                )
                print(
                    f"Expected (2L):       {expected_wavelength_m:.3e} m ({2*neighbors.rest_length_am:.1f} am)"
                )
                print(f"Error:               {wavelength_error:.1f}%")
                print(f"")
                print(f"Reference lengths:")
                print(f"  Unit cell edge:    {lattice.unit_cell_edge_am:.1f} am")
                print(f"  Rest length (L):   {neighbors.rest_length_am:.1f} am")
                print(f"  QWAVE_LENGTH:      {constants.QWAVE_LENGTH/constants.ATTOMETTER:.1f} am")
                print(f"")
                print(
                    f"Measured / Unit cell: {wavelength_data['wavelength_am']/lattice.unit_cell_edge_am:.2f}x"
                )
                print(
                    f"Measured / 2L:        {wavelength_data['wavelength_am']/(2*neighbors.rest_length_am):.2f}x"
                )
                print(
                    f"Measured / λ_quantum: {wavelength_data['wavelength_am']/(constants.QWAVE_LENGTH/constants.ATTOMETTER):.2f}x"
                )
            else:
                print(
                    f"Only {wavelength_data['num_peaks']} peak detected - need ≥2 for wavelength"
                )
                print(
                    f"Expected wavelength: {expected_wavelength_m:.3e} m ({2*neighbors.rest_length_am:.1f} am)"
                )
            print(f"Number of peaks:     {wavelength_data['num_peaks']}")
            print(f"{'='*70}\n")
        else:
            print(f"  → No peaks detected. Wave may not have propagated far enough yet.\n")


@ti.kernel
def measure_wave_displacement(
    position: ti.template(),  # type: ignore
    equilibrium: ti.template(),  # type: ignore
    displacement_mag: ti.template(),  # type: ignore
):
    """Measure displacement magnitude for each granule.

    Calculates ||x - x_eq|| for detecting wavefront propagation.

    Args:
        position: Current position
        equilibrium: Equilibrium position
        displacement_mag: Output displacement magnitudes (scalar field)
    """
    for i in range(position.shape[0]):
        disp = position[i] - equilibrium[i]
        displacement_mag[i] = disp.norm()


def measure_wave_speed(
    lattice,
    threshold_fraction: float = 0.1,
    slow_mo: float = 1.0,
) -> dict:
    """Measure wave propagation speed from vertex to interior.

    Tracks the wavefront by finding granules with displacement above threshold.
    Measures the maximum distance from vertices where wave has reached.

    Args:
        lattice: Lattice instance
        threshold_fraction: Fraction of amplitude to detect wavefront (0.1 = 10%)
        slow_mo: Slow motion factor (to convert simulation time to real time)

    Returns:
        dict with 'max_distance_am', 'amplitude_am', 'threshold_am'
    """
    # Allocate displacement field if needed
    if not hasattr(lattice, "displacement_mag_am"):
        lattice.displacement_mag_am = ti.field(dtype=ti.f32, shape=lattice.total_granules)

    # Measure displacements
    measure_wave_displacement(
        lattice.position_am,
        lattice.equilibrium_am,
        lattice.displacement_mag_am,
    )

    # Find maximum displacement (amplitude)
    displacements = lattice.displacement_mag_am.to_numpy()
    max_displacement = np.max(displacements)
    threshold = max_displacement * threshold_fraction

    # Find maximum distance from vertices where displacement exceeds threshold
    max_distance = 0.0
    vertex_position = []
    for v in range(8):
        idx = lattice.vertex_index[v]
        pos = lattice.position_am[idx].to_numpy()
        vertex_position.append(pos)

    # Check each granule
    position_np = lattice.position_am.to_numpy()
    for i in range(lattice.total_granules):
        if displacements[i] > threshold:
            # Find minimum distance to any vertex
            pos = position_np[i]
            min_dist = float("inf")
            for v_pos in vertex_position:
                dist = np.linalg.norm(pos - v_pos)
                min_dist = min(min_dist, dist)
            max_distance = max(max_distance, min_dist)

    return {
        "max_distance_am": max_distance,
        "amplitude_am": max_displacement,
        "threshold_am": threshold,
    }


@ti.kernel
def sample_radial_line(
    position: ti.template(),  # type: ignore
    equilibrium: ti.template(),  # type: ignore
    vertex_idx: ti.i32,  # type: ignore
    center_pos: ti.template(),  # type: ignore
    num_samples: ti.i32,  # type: ignore
    sample_position: ti.template(),  # type: ignore
    sample_displacements: ti.template(),  # type: ignore
):
    """Sample displacement along a radial line from vertex to center.

    For wavelength measurement, we need to find the spatial period.
    This samples granules along a line and measures their displacements.

    Args:
        position: Current position
        equilibrium: Equilibrium position
        vertex_idx: Starting vertex index
        center_pos: Center position of lattice
        num_samples: Number of samples to take
        sample_position: Output sample position along line (distances from vertex)
        sample_displacements: Output displacement magnitudes at samples
    """
    vertex_pos = equilibrium[vertex_idx]
    direction = (center_pos - vertex_pos).normalized()
    max_dist = (center_pos - vertex_pos).norm()

    for i in range(num_samples):
        # Sample position along line
        t = ti.f32(i) / ti.f32(num_samples - 1)
        sample_pos = vertex_pos + direction * (t * max_dist)

        # Find nearest granule to sample position
        nearest_idx = 0
        min_dist_sq = 1e30
        for j in range(position.shape[0]):
            dist_sq = (equilibrium[j] - sample_pos).norm_sqr()
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_idx = j

        # Record displacement at this sample
        disp = position[nearest_idx] - equilibrium[nearest_idx]
        sample_position[i] = (sample_pos - vertex_pos).norm()
        sample_displacements[i] = disp.norm()


def measure_wavelength(
    lattice,
    neighbors,
    num_samples: int = 100,
) -> dict:
    """Measure wavelength by finding nearest-neighbor distances between displacement peaks.

    Instead of sampling along a line (which doesn't work for radial waves),
    this finds granules with local maximum displacement and measures the
    average distance between neighboring peaks in 3D space.

    Uses existing BCC neighbor links for O(N) complexity instead of O(N²).

    Args:
        lattice: Lattice instance
        neighbors: BCCNeighbors instance with precomputed links
        num_samples: Number of samples along line (kept for backwards compatibility)

    Returns:
        dict with 'wavelength_am', 'num_peaks', diagnostic data
    """
    # Get all displacements
    if not hasattr(lattice, "displacement_mag_am"):
        lattice.displacement_mag_am = ti.field(dtype=ti.f32, shape=lattice.total_granules)

    measure_wave_displacement(
        lattice.position_am,
        lattice.equilibrium_am,
        lattice.displacement_mag_am,
    )

    displacements = lattice.displacement_mag_am.to_numpy()
    equilibrium_np = lattice.equilibrium_am.to_numpy()
    max_disp = np.max(displacements)
    threshold_disp = 0.3 * max_disp  # Look for peaks >30% of max

    # Find local maxima using BCC neighbor links (O(N) - FAST!)
    # A granule is a local max if its displacement is higher than all its BCC neighbors
    peak_indices = []
    for i in range(lattice.total_granules):
        if lattice.granule_type[i] == 0:  # Skip vertices (they're driven)
            continue

        if displacements[i] < threshold_disp:
            continue

        # Check if this is a local maximum compared to its BCC neighbors
        is_local_max = True
        num_links = neighbors.links_count[i]

        # Check only the actual BCC neighbors (8 or fewer)
        for j in range(num_links):
            neighbor_idx = neighbors.links[i, j]
            if neighbor_idx >= 0 and displacements[neighbor_idx] > displacements[i]:
                is_local_max = False
                break

        if is_local_max:
            peak_indices.append(i)

    # Calculate average nearest-neighbor distance between peaks
    wavelength_am = 0.0
    peak_distances = []

    if len(peak_indices) >= 2:
        # For each peak, find distance to nearest other peak
        for i, idx_i in enumerate(peak_indices):
            pos_i = equilibrium_np[idx_i]
            min_dist = float("inf")

            for j, idx_j in enumerate(peak_indices):
                if i == j:
                    continue
                pos_j = equilibrium_np[idx_j]
                dist = np.linalg.norm(pos_i - pos_j)
                min_dist = min(min_dist, dist)

            if min_dist < float("inf"):
                peak_distances.append(min_dist)

        # Average nearest-neighbor distance
        if len(peak_distances) > 0:
            wavelength_am = np.mean(peak_distances)

    return {
        "wavelength_am": wavelength_am,
        "num_peaks": len(peak_indices),
        "sample_position": np.array([]),  # Not used in new approach
        "sample_displacements": displacements,
        "peak_position": peak_distances,
        "max_displacement": max_disp,
        "threshold_displacement": threshold_disp,
    }
