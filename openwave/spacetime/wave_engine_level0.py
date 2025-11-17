"""
ENERGY-WAVE ENGINE

LEVEL-0: ON GRANULE-BASED MEDIUM

Wave Physics Engine @spacetime module. Wave dynamics and motion.

Multiple Wave Sources: Models wave interference from multiple harmonic oscillators.
Each source generates spherical longitudinal waves that superpose at each granule.
"""

import taichi as ti

from openwave.common import config, constants, equations

# ================================================================
# Energy-Wave Oscillation Parameters
# ================================================================
base_amplitude_am = constants.EWAVE_AMPLITUDE / constants.ATTOMETER  # am, oscillation amplitude
wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETER  # in attometers
frequency = constants.EWAVE_SPEED / constants.EWAVE_LENGTH  # Hz, energy-wave frequency

# ================================================================
# Energy-Wave Source Data (Global Fields)
# ================================================================
# These fields are initialized once by build_source_vectors() and used by oscillate_granules()
# Shape: (granule_count, num_sources) for parallel access
sources_direction = None  # Direction vectors from each granule to each wave source
sources_distance_am = None  # Distances from each granule to each wave source (attometers)
sources_phase_shift = None  # Phase offset for each wave source (radians)

# Adaptive displacement tracking for numerical analysis
peak_amplitude_am = None  # Peak amplitude (maximum displacement from all granules)
avg_amplitude_am = None  # RMS amplitude for energy calculation (peak * 0.707)
last_amp_boost = None  # Track last amp_boost value for reset


# ================================================================
# Energy-Wave Source Kernel (energy charge, harmonic oscillation, rhythm)
# ================================================================


def build_source_vectors(sources_position, sources_phase_deg, num_sources, lattice):
    """Precompute distance & direction vectors from all granules to multiple wave sources.

    This function is called once during initialization. It computes the geometric
    relationship between every granule and every wave source, storing:
    - Direction vectors (normalized): which way waves propagate from each source
    - Distances (attometers): affects phase and amplitude of waves from each source
    - Phase offsets (radians): initial phase shift for each source

    This function handles arbitrary source positions that may change between xperiments.

    Args:
        sources_position: List of [x,y,z] coordinates (normalized 0-1) for each wave source.
            Uses Z-up coordinate system: X=horizontal, Y=depth, Z=vertical.
        sources_phase_deg: List of phase offsets (degrees) for each wave source
        num_sources: Number of wave sources
        lattice: BCCLattice instance with granule positions and universe parameters
    """
    global sources_direction, sources_distance_am, sources_phase_shift, sources_pos_field
    global peak_amplitude_am, avg_amplitude_am, last_amp_boost

    # Convert phase from degrees to radians for physics calculations
    # Conversion: radians = degrees × π/180
    sources_phase_rad = [deg * ti.math.pi / 180 for deg in sources_phase_deg]

    # Allocate Taichi fields for all granules and all wave sources
    # Shape: (granules, sources) allows parallel access in oscillate_granules kernel
    sources_direction = ti.Vector.field(
        3, dtype=ti.f32, shape=(lattice.granule_count, num_sources)
    )
    sources_distance_am = ti.field(dtype=ti.f32, shape=(lattice.granule_count, num_sources))
    sources_phase_shift = ti.field(dtype=ti.f32, shape=num_sources)

    # Convert Python lists to Taichi fields for kernel access
    sources_pos_field = ti.Vector.field(3, dtype=ti.f32, shape=num_sources)

    # Initialize displacement tracking fields for numerical analysis
    peak_amplitude_am = ti.field(dtype=ti.f32, shape=())  # peak amplitude (max displacement)
    avg_amplitude_am = ti.field(dtype=ti.f32, shape=())  # RMS amplitude
    last_amp_boost = ti.field(dtype=ti.f32, shape=())  # Track last amp_boost value

    # Copy source data to Taichi fields
    for i in range(num_sources):
        sources_pos_field[i] = ti.Vector(sources_position[i])
        sources_phase_shift[i] = sources_phase_rad[i]

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

                # Distance from source to granule
                dist = dir_vec.norm() + 1e-10  # Add epsilon to avoid division by zero

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
    ironbow: ti.i32,  # type: ignore
    var_displacement: ti.i32,  # type: ignore
    num_sources: ti.i32,  # type: ignore
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
    - Phase determined by distance from each wave source (creates wave fronts)
    - Amplitude decreases as 1/r for energy conservation (spherical waves)

    Wave Superposition Principle:
        x_total(t) = Σ[x_i(t)] for all wave sources i
        x_i(t) = x_eq + A_i(r_i)·cos(ωt + φ_i + φ_source_i)·dir_i

    Where for each wave source i:
        - r_i: distance from wave source i to granule
        - φ_i = -k·r_i: spatial phase (wave propagation)
        - φ_source_i: initial phase offset of wave source i
        - dir_i: direction from wave source i to granule (outward propagation)
        - A_i(r_i) = A₀·(r₀/r_i): amplitude falloff with distance

    Near-Field vs Far-Field (per wave source):
        - Near field (r < λ): Source region, wave structure forming
        - Transition zone (λ < r < 2λ): Wave fronts organizing
        - Far field (r > 2λ): Fully formed spherical waves, 1/r falloff

    Interference Patterns:
        - Constructive: waves from different sources arrive in phase (bright fringes)
        - Destructive: waves arrive out of phase (dark fringes/nodes)
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
        ironbow: Ironbow coloring toggle
        var_displacement: Displacement vs amplitude toggle
        num_sources: Number of wave sources
        elapsed_t: Current simulation time (accumulated, seconds)
        freq_boost: Frequency multiplier (applied after slow_mo)
        amp_boost: Amplitude multiplier (for visibility in scaled lattices)
    """
    # Compute temporal parameters (same for all wave sources)
    f_slowed = frequency / config.SLOW_MO * freq_boost  # slowed frequency (Hz)
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency (rad/s)

    # Wave number k = 2π/λ (spatial phase variation)
    k = 2.0 * ti.math.pi / wavelength_am  # wave number (radians per attometer)

    # Reference radius for amplitude normalization (one wavelength)
    # Prevents singularity at r=0 and provides physically meaningful normalization
    r_reference_am = wavelength_am  # attometers

    # Process all granules in parallel (outermost loop = GPU parallelization)
    for granule_idx in range(position_am.shape[0]):
        # Initialize accumulation variables for wave superposition
        total_displacement_am = ti.Vector([0.0, 0.0, 0.0])  # sum of displacements from all sources
        total_velocity_am = ti.Vector([0.0, 0.0, 0.0])  # sum of velocities from all sources

        # Sum contributions from all sources (superposition principle)
        for source_idx in range(num_sources):
            # Get precomputed direction and distance for this granule-source pair
            direction = sources_direction[granule_idx, source_idx]
            r_am = sources_distance_am[granule_idx, source_idx]  # distance in attometers

            # Get source phase offset
            phase_offset = sources_phase_shift[source_idx]

            # Spatial phase: φ = -k·r (negative for outward propagation)
            # Creates spherical wave fronts expanding from source
            spatial_phase = -k * r_am

            # Total phase: includes spatial phase and source's initial offset
            total_phase = spatial_phase + phase_offset

            # Amplitude falloff for spherical wave: A(r) = A₀·(r₀/r)
            # Use r_safe to prevent singularity (division by zero) at r → 0
            # Enforces r_min = 1λ based on EWT neutrino boundary and EM near-field physics
            r_safe_am = ti.max(r_am, r_reference_am)  # minimum 1 wavelength_am from source
            amplitude_falloff = r_reference_am / r_safe_am

            # Total amplitude at granule distance from source
            # Step 1: Apply energy conservation (1/r falloff) and visualization scaling
            amplitude_am_at_r = base_amplitude_am * amplitude_falloff * amp_boost

            # Step 2: Cap amplitude to distance from source (A ≤ r)
            # Prevents non-physical behavior: granules crossing through wave source
            # When A > r, displacement could exceed distance to source, placing granule
            # on opposite side of source (physically impossible for longitudinal waves)
            # This constraint ensures: |x - x_eq| ≤ |x_eq - x_source|
            amplitude_am_at_r_cap = ti.min(amplitude_am_at_r, r_am)

            # MAIN EQUATION OF MOTION
            # Wave displacement from this source: A(r)·cos(ωt + φ)·direction
            source_displacement_am_magnitude = amplitude_am_at_r_cap * ti.cos(
                omega * elapsed_t + total_phase
            )
            source_displacement_am = source_displacement_am_magnitude * direction

            # Wave velocity from this source: -A(r)·ω·sin(ωt + φ)·direction
            velocity_am_magnitude = (
                -amplitude_am_at_r_cap * omega * ti.sin(omega * elapsed_t + total_phase)
            )
            source_velocity_am = velocity_am_magnitude * direction

            # Accumulate this source's contribution (wave superposition)
            total_displacement_am += source_displacement_am
            total_velocity_am += source_velocity_am

        # Apply superposed wave to granule position and velocity
        position_am[granule_idx] = equilibrium_am[granule_idx] + total_displacement_am
        velocity_am[granule_idx] = total_velocity_am

        # ================================================================
        # DISPLACEMENT TRACKING - NUMERICAL ANALYSIS
        # ================================================================
        # Compute displacement magnitude
        displacement_am = total_displacement_am.norm()

        # Track granule amplitude for analysis (max displacement per granule)
        # Thread-safe atomic max for parallel GPU execution
        ti.atomic_max(amplitude_am[granule_idx], displacement_am)

        # Track peak amplitude across all granules (thread-safe atomic max)
        # Used for numerical analysis and energy calculation
        ti.atomic_max(peak_amplitude_am[None], displacement_am)

        # COLOR CONVERSION OF DISPLACEMENT/AMPLITUDE VALUES
        # Map displacement/amplitude to gradient color
        if ironbow:
            granule_var_color[granule_idx] = config.get_ironbow_color(
                displacement_am if var_displacement else amplitude_am[granule_idx],
                0.0,
                peak_amplitude_am[None],
            )
        else:
            granule_var_color[granule_idx] = config.get_blueprint_color(
                displacement_am if var_displacement else amplitude_am[granule_idx],
                0.0,
                peak_amplitude_am[None],
            )

    # Reset amplitude trackers if amplitude boost changed
    # Prevents stale high values when amp_boost is reduced
    if last_amp_boost[None] != amp_boost:
        peak_amplitude_am[None] = 0.0
        for i in range(amplitude_am.shape[0]):
            amplitude_am[i] = 0.0
        last_amp_boost[None] = amp_boost

    # Convert peak amplitude to RMS amplitude for energy calculation
    # RMS amplitude = peak / √2 ≈ peak * 0.707
    # This is the standard conversion for sinusoidal oscillations
    # Energy equation uses amplitude, and RMS gives the effective energy content
    avg_amplitude_am[None] = peak_amplitude_am[None] * 0.707


def update_lattice_energy(lattice):
    """Update lattice energy based on RMS amplitude from max displacement.

    Must be called after oscillate_granules() to compute energy from wave amplitude.
    Cannot be done inside the kernel as lattice.energy is not a Taichi field.

    Uses RMS amplitude (peak * 0.707) which represents the effective energy content
    of sinusoidal oscillations. The peak is tracked across all granules, and the
    RMS conversion gives the equivalent constant amplitude that would produce the
    same energy. This avoids expensive per-frame averaging while providing physically
    meaningful energy values.

    Args:
        lattice: Lattice instance with universe_volume and energy fields
    """
    lattice.energy = equations.energy_wave_equation(
        volume=lattice.universe_volume, amplitude=avg_amplitude_am[None] * constants.ATTOMETER
    )
    lattice.energy_kWh = lattice.energy * constants.J2KWH  # in KWh
    lattice.energy_years = lattice.energy_kWh / (183230 * 1e9)  # global energy use
