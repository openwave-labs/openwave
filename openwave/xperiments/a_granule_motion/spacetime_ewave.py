"""
ENERGY-WAVE ENGINE

ON GRANULE METHOD

Wave Physics Engine @spacetime module. Wave dynamics and motion.

Multiple Wave Sources: Models wave interference from multiple harmonic oscillators.
Each source generates spherical longitudinal waves that superpose at each granule.
"""

import taichi as ti

from openwave.common import colormap, constants, equations, utils

# ================================================================
# Energy-Wave Oscillation Parameters
# ================================================================
base_amplitude_am = constants.EWAVE_AMPLITUDE / constants.ATTOMETER  # in am
base_frequency = constants.EWAVE_SPEED / constants.EWAVE_LENGTH  # in Hz
wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETER  # in am

# ================================================================
# Energy-Wave Source Data (Global Fields)
# ================================================================
# These fields are initialized once by build_source_vectors() and used by oscillate_granules()
# Shape: (granule_count, num_sources) for parallel access
sources_direction = None  # Direction vectors from each granule to each wave source
sources_distance_am = None  # Distances from each granule to each wave source (attometers)
sources_phase_offset = None  # Phase offset for each wave source (radians)

# Displacement tracking for energy calculation
peak_amplitude_am = None  # max displacement across all granules
avg_amplitude_am = None  # RMS amplitude (peak × 0.707)
last_amp_boost = None  # for detecting amp_boost changes
last_in_wave_toggle = None  # for detecting in_wave toggle changes
last_out_wave_toggle = None  # for detecting out_wave toggle changes

# Center of sources for signed radial displacement
sources_center_am = None  # geometric center of all wave sources (attometers)


# ================================================================
# Energy-Wave Source Kernel (energy charge, harmonic oscillation, rhythm)
# ================================================================


def build_source_vectors(num_sources, sources_position, sources_offset_deg, lattice):
    """Precompute distance & direction vectors from all granules to multiple wave sources.

    This function is called once during initialization. It computes the geometric
    relationship between every granule and every wave source, storing:
    - Direction vectors (normalized): which way waves propagate from each source
    - Distances (attometers): affects phase and amplitude of waves from each source
    - Phase offsets (radians): initial phase for each source

    This function handles arbitrary source positions that may change between xperiments.

    Args:
        num_sources: Number of wave sources
        sources_position: List of [x,y,z] coordinates (normalized 0-1) for each wave source.
            Uses Z-up coordinate system: X=horizontal, Y=depth, Z=vertical.
        sources_offset_deg: List of phase offsets (degrees) for each wave source
        lattice: BCCLattice instance with granule positions and universe parameters
    """
    global sources_direction, sources_distance_am, sources_phase_offset, sources_pos_field
    global peak_amplitude_am, avg_amplitude_am, sources_center_am
    global last_amp_boost, last_in_wave_toggle, last_out_wave_toggle

    # Convert phase from degrees to radians
    sources_offset_rad = [deg * ti.math.pi / 180 for deg in sources_offset_deg]

    # Allocate Taichi fields for all granules and all wave sources
    # Shape: (granules, sources) allows parallel access in oscillate_granules kernel
    sources_direction = ti.Vector.field(
        3, dtype=ti.f32, shape=(lattice.granule_count, num_sources)
    )
    sources_distance_am = ti.field(dtype=ti.f32, shape=(lattice.granule_count, num_sources))
    sources_phase_offset = ti.field(dtype=ti.f32, shape=num_sources)

    # Convert Python lists to Taichi fields for kernel access
    sources_pos_field = ti.Vector.field(3, dtype=ti.f32, shape=num_sources)

    # Initialize displacement tracking fields
    peak_amplitude_am = ti.field(dtype=ti.f32, shape=())  # max displacement
    avg_amplitude_am = ti.field(dtype=ti.f32, shape=())  # RMS amplitude
    last_amp_boost = ti.field(dtype=ti.f32, shape=())  # for change detection
    last_in_wave_toggle = ti.field(dtype=ti.i32, shape=())  # detects in_wave toggle changes
    last_out_wave_toggle = ti.field(dtype=ti.i32, shape=())  # detects out_wave toggle changes

    # Compute geometric center of all sources (for signed radial displacement)
    sources_center_am = ti.Vector.field(3, dtype=ti.f32, shape=())
    center_sum = [0.0, 0.0, 0.0]
    for i in range(num_sources):
        for j in range(3):
            center_sum[j] += sources_position[i][j]
    # Convert normalized center to attometers
    sources_center_am[None] = ti.Vector(
        [
            center_sum[0] / num_sources * lattice.max_universe_edge_am,
            center_sum[1] / num_sources * lattice.max_universe_edge_am,
            center_sum[2] / num_sources * lattice.max_universe_edge_am,
        ]
    )

    # Copy source data to Taichi fields
    for i in range(num_sources):
        sources_pos_field[i] = ti.Vector(sources_position[i])
        sources_phase_offset[i] = sources_offset_rad[i]

    @ti.kernel
    def compute_vectors(num_active: ti.i32):  # type: ignore
        """Compute direction and distance from each granule to each wave source.

        Parallelized over all granules (outermost loop). Inner loop over sources
        is sequential but short (num_sources).
        """
        for granule_idx in range(lattice.granule_count):
            # Loop through all wave sources
            for source_idx in range(num_active):
                # Convert normalized source position to attometers
                source_pos_am = sources_pos_field[source_idx] * lattice.max_universe_edge_am

                # Vector from source to granule (for outward propagation)
                dir_vec = lattice.position_am[granule_idx] - source_pos_am

                # Distance from source to granule (with epsilon for numerical stability)
                dist = dir_vec.norm() + 1e-10

                # Store normalized direction vector (outward from source)
                sources_direction[granule_idx, source_idx] = dir_vec / dist

                # Store distance in attometers
                sources_distance_am[granule_idx, source_idx] = dist

    # Execute the kernel
    compute_vectors(num_sources)


@ti.kernel
def oscillate_granules(
    position_am: ti.template(),  # type: ignore
    equilibrium_am: ti.template(),  # type: ignore
    amplitude_am: ti.template(),  # type: ignore
    velocity_am: ti.template(),  # type: ignore
    granule_var_color: ti.template(),  # type: ignore
    freq_boost: ti.f32,  # type: ignore
    amp_boost: ti.f32,  # type: ignore
    color_palette: ti.i32,  # type: ignore
    num_sources: ti.i32,  # type: ignore
    in_wave_toggle: ti.i32,  # type: ignore
    out_wave_toggle: ti.i32,  # type: ignore
    elapsed_t: ti.f32,  # type: ignore
):
    """Charges energy into all granules from multiple wave sources using wave superposition.

    Each granule receives wave contributions from all active wave sources. Waves are summed
    (superposed) to create interference patterns - constructive where waves align in phase,
    destructive where they oppose.

    Physics Model:
    - Each wave source generates spherical longitudinal waves
    - Waves propagate radially outward from each wave source
    - At each granule, displacement = sum of displacements from all wave sources
    - Spacial phase determined by distance from each wave source (creates wave fronts)
    - Amplitude decreases as 1/r for energy conservation (spherical waves)

    Wave Superposition Principle:
        x_total(t) = Σ[x_i(t)] for all wave sources i
        x_i(t) = x_eq + A_i(r_i)·cos(ωt + kr_i + φ_source_i)·dir_i

    Where for each wave source i:
        - r_i: distance from wave source i to granule
        - k·r_i: spatial phase (wave propagation)
        - φ_source_i: initial phase offset of wave source i
        - dir_i: direction from wave source i to granule (outward propagation)
        - A_i(r_i) = A₀·(r₀/r_i): amplitude falloff with distance

    Near-Field vs Far-Field (per wave source):
        - Near field (r < λ): Source region, wave structure forming
        - Transition zone (λ < r < 2λ): Wave fronts organizing
        - Far field (r > 2λ): Fully formed spherical waves, 1/r falloff

    Interference Patterns:
        - Constructive: waves from different sources arrive in phase (bright fringes)
        - Destructive: waves arrive π out of phase (dark fringes/nodes)
        - Complex patterns emerge from phase relationships between sources

    Energy Conservation (per source):
        A(r) = A₀·(r₀/r), where r₀ = 1λ (reference radius)
        Ensures total energy flux through spherical shells remains constant

    Implementation uses r_min = 1λ per source based on:
        - EWT neutrino boundary specification at r = 1λ
        - EM theory transition to radiative fields around λ
        - Prevents singularity at r → 0
        - Ensures numerical stability

    Parallelization Strategy:
        - Outermost loop (granules): Fully parallelized on GPU
        - Inner loop (sources): Sequential per granule (num_sources iterations)
        - This maximizes GPU utilization while handling variable source counts

    Displacement Tracking for Numerical Analysis:
        - Tracks maximum displacement per frame using atomic_max (peak amplitude)
        - Converts to RMS amplitude using max * 0.707 (peak / √2) for energy calculation

    Args:
        position_am: Position field for all granules (modified in-place, in attometers)
        equilibrium_am: Equilibrium (rest) positions for all granules (in attometers)
        amplitude_am: Amplitude field for all granules (modified in-place, in attometers)
        velocity_am: Velocity field for all granules (modified in-place, in attometers/second)
        granule_var_color: Color field for displacement/amplitude visualization
        freq_boost: Frequency multiplier (applied after slowed frequency)
        amp_boost: Amplitude multiplier (for visibility in scaled lattices)
        color_palette: Coloring palette selection
        num_sources: Number of wave sources
        in_wave_toggle: Toggle for in_wave (1 = enable, 0 = disable)
        out_wave_toggle: Toggle for out_wave (1 = enable, 0 = disable)
        elapsed_t: Current simulation time (accumulated, seconds)
    """
    # Compute angular frequency (ω = 2πf) for temporal phase variation
    frequency_slo = base_frequency / 1e25 * freq_boost  # slowed frequency (1Hz * boost)
    omega_slo = 2.0 * ti.math.pi * frequency_slo  # angular frequency (rad/s)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    k_am = 2.0 * ti.math.pi / wavelength_am  # radians per attometer

    # Reference radius for amplitude normalization (r₀ = 1λ)
    # Prevents singularity at r=0 and sets 1/r falloff reference point
    r_reference_am = wavelength_am

    # Process all granules in parallel (outermost loop = GPU parallelization)
    for granule_idx in range(position_am.shape[0]):
        # Initialize accumulation variables for wave superposition
        total_displacement_am = ti.Vector([0.0, 0.0, 0.0])  # sum of displacements from all sources
        total_velocity_am = ti.Vector([0.0, 0.0, 0.0])  # sum of velocities from all sources

        # Sum contributions from all sources (wave superposition)
        for source_idx in range(num_sources):
            # Get precomputed direction and distance for this granule-source pair
            direction = sources_direction[granule_idx, source_idx]
            r_am = sources_distance_am[granule_idx, source_idx]

            # Phase shift between in/out waves (at wave-center)
            phase_shift = ti.math.pi / 2

            # Source phase offset: initial phase of this wave-center
            source_offset = sources_phase_offset[source_idx]

            # Temporal phase: φ = ω·t, oscillatory in time
            temporal_phase = omega_slo * elapsed_t

            # Spatial phase: φ = k·r, creates spherical wave fronts
            spatial_phase = k_am * r_am

            # Amplitude falloff for spherical wave: A(r) = A₀/r
            # Clamp to r_min to avoid singularity at r = 0
            r_safe_am = ti.max(r_am, r_reference_am)
            amplitude_falloff = r_reference_am / r_safe_am
            # Total amplitude at this distance (with visualization scaling)
            amplitude_at_r_am = base_amplitude_am * amp_boost * amplitude_falloff
            # Cap amplitude to distance from source (A ≤ r)
            # Prevents granules crossing through wave source
            amplitude_at_r_cap_am = ti.min(amplitude_at_r_am, r_am)

            # MAIN WAVE FUNCTION ========================================
            # IN & OUT Wave displacement from this source
            # A·cos(ωt ± kr + φ)·direction, positive for inward propagation, full amp
            # A(r)·cos(ωt ± kr + φ)·direction, negative for outward propagation, amp falloff
            in_wave_psi = (
                in_wave_toggle
                * base_amplitude_am
                * amp_boost
                / num_sources  # incoming wave do not superpose, split per WC for energy conservation
                * ti.cos(temporal_phase + spatial_phase + source_offset)
            )
            out_wave_psi = (
                out_wave_toggle
                * amplitude_at_r_cap_am
                * ti.cos(temporal_phase - spatial_phase + source_offset + phase_shift)
            )
            source_displacement_am = (in_wave_psi + out_wave_psi) * direction

            # Wave velocity from this source: -A(r)·ω·sin(ωt ± kr + φ)·direction
            in_wave_vel = (
                in_wave_toggle
                * -base_amplitude_am
                * omega_slo
                * ti.sin(temporal_phase + spatial_phase + source_offset)
            ) / num_sources  # incoming wave do not superpose, gets split per WC
            out_wave_vel = (
                out_wave_toggle
                * -amplitude_at_r_cap_am
                * omega_slo
                * ti.sin(temporal_phase - spatial_phase + source_offset + phase_shift)
            )
            source_velocity_am = (in_wave_vel + out_wave_vel) * direction

            # Accumulate this source's contribution (wave superposition)
            total_displacement_am += source_displacement_am
            total_velocity_am += source_velocity_am

        # Rounding to prevent floating-point precision error (at rounding boundaries)
        # Critical for opposing phase sources that should cancel / annihilate
        # Also critical at peak amplitudes (in wave has same amplitude in space)
        # eg. (+1.250001) + (-1.249999) = 0.000002 (not a perfect cancel)
        precision = 1e4  # rounding precision to ensure cancellation cases
        total_displacement_am = ti.round(total_displacement_am * precision) / precision
        total_velocity_am = ti.round(total_velocity_am * precision) / precision

        # Apply superposed wave to granule position and velocity
        position_am[granule_idx] = equilibrium_am[granule_idx] + total_displacement_am
        velocity_am[granule_idx] = total_velocity_am

        # WAVE AMPLITUDE TRACKERS ============================================
        # Initialize before conditional (Taichi scope requirement)
        displacement_am = total_displacement_am.norm()

        # Track per-granule PEAK Amplitude & global PEAK Amplitude
        ti.atomic_max(amplitude_am[granule_idx], displacement_am)
        ti.atomic_max(peak_amplitude_am[None], displacement_am)

        # COLOR CONVERSION OF DISPLACEMENT/AMPLITUDE VALUES
        # Map value to color using selected gradient
        if color_palette == 3:  # orange (magnitude only)
            granule_var_color[granule_idx] = colormap.get_orange_color(
                displacement_am,
                0.0,
                peak_amplitude_am[None],
            )
        elif color_palette == 5:  # ironbow (magnitude only: thermal scale)
            granule_var_color[granule_idx] = colormap.get_ironbow_color(
                amplitude_am[granule_idx],
                0.0,
                peak_amplitude_am[None],
            )

    # Reset amplitude trackers when amp_boost changes
    # Prevents stale high values when amp_boost is reduced
    if (
        last_amp_boost[None] != amp_boost
        or last_in_wave_toggle[None] != in_wave_toggle
        or last_out_wave_toggle[None] != out_wave_toggle
    ):
        peak_amplitude_am[None] = 0.0
        for i in range(amplitude_am.shape[0]):
            amplitude_am[i] = 0.0
        last_amp_boost[None] = amp_boost
        last_in_wave_toggle[None] = in_wave_toggle
        last_out_wave_toggle[None] = out_wave_toggle
    # Convert peak to RMS amplitude: RMS = peak / √2 ≈ peak × 0.707
    avg_amplitude_am[None] = peak_amplitude_am[None] * 0.707


def update_lattice_energy(lattice):
    """Update lattice energy based on RMS amplitude.

    Must be called after oscillate_granules() to compute energy from wave amplitude.
    Cannot be done inside the kernel as lattice.nominal_energy is not a Taichi field.

    Uses RMS amplitude (peak * 0.707) which represents the effective energy content
    of sinusoidal oscillations. The peak is tracked across all granules, and the
    RMS conversion gives the equivalent constant amplitude that would produce the
    same energy. This avoids expensive per-frame averaging while providing physically
    meaningful energy values.

    Args:
        lattice: Lattice instance with universe_volume and energy fields
    """
    lattice.nominal_energy = equations.compute_energy_wave_equation(
        volume=lattice.universe_volume, amplitude=avg_amplitude_am[None] * constants.ATTOMETER
    )
    lattice.nominal_energy_kWh = lattice.nominal_energy * utils.J2KWH
    lattice.nominal_energy_years = lattice.nominal_energy_kWh / (183230 * 1e9)  # years
