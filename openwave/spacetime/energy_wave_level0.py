"""
ENERGY-WAVE

LEVEL-0: ON GRANULE-BASED MEDIUM

Wave Physics Engine @spacetime module.
Wave dynamics and motion.

Multiple Wave Sources: Models wave interference from multiple harmonic oscillators.
Each source generates spherical longitudinal waves that superpose at each granule.
"""

import taichi as ti

from openwave.common import config
from openwave.common import constants

# ================================================================
# Energy-Wave Oscillation Parameters
# ================================================================
amplitude_am = constants.EWAVE_AMPLITUDE / constants.ATTOMETTER  # am, oscillation amplitude
wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETTER  # in attometers
frequency = constants.EWAVE_SPEED / constants.EWAVE_LENGTH  # Hz, energy-wave frequency

# ================================================================
# Energy-Wave Source Data (Global Fields)
# ================================================================
# These fields are initialized once by build_source_vectors() and used by oscillate_granules()
# Shape: (total_granules, num_sources) for parallel access
sources_direction = None  # Direction vectors from each granule to each wave source
sources_distance_am = None  # Distances from each granule to each wave source (attometers)
sources_phase_shift = None  # Phase offset for each wave source (radians)


# ================================================================
# Energy-Wave Source Kernel (energy injection, harmonic oscillation, rhythm)
# ================================================================


def build_source_vectors(sources_position, sources_phase, num_sources, lattice):
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
        sources_phase: List of phase offsets (radians) for each wave source
        num_sources: Number of wave sources
        lattice: BCCLattice instance with granule positions and universe parameters
    """
    global sources_direction, sources_distance_am, sources_phase_shift, sources_pos_field

    # Allocate Taichi fields for all granules and all wave sources
    # Shape: (granules, sources) allows parallel access in oscillate_granules kernel
    sources_direction = ti.Vector.field(
        3, dtype=ti.f32, shape=(lattice.total_granules, num_sources)
    )
    sources_distance_am = ti.field(dtype=ti.f32, shape=(lattice.total_granules, num_sources))
    sources_phase_shift = ti.field(dtype=ti.f32, shape=num_sources)

    # Convert Python lists to Taichi fields for kernel access
    sources_pos_field = ti.Vector.field(3, dtype=ti.f32, shape=num_sources)

    # Copy source data to Taichi fields
    for i in range(num_sources):
        sources_pos_field[i] = ti.Vector(sources_position[i])
        sources_phase_shift[i] = sources_phase[i]

    @ti.kernel
    def compute_vectors(num_active: ti.i32):  # type: ignore
        """Compute direction and distance from each granule to each wave source.

        Parallelized over all granules (outermost loop). Inner loop over sources
        is sequential but short (num_sources).
        """
        for granule_idx in range(lattice.total_granules):
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
    position: ti.template(),  # type: ignore
    equilibrium: ti.template(),  # type: ignore
    velocity: ti.template(),  # type: ignore
    num_sources: ti.i32,  # type: ignore
    t: ti.f32,  # type: ignore
    freq_boost: ti.f32,  # type: ignore
    amp_boost: ti.f32,  # type: ignore
):
    """Injects energy into all granules from multiple wave sources using wave superposition.

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

    Args:
        position: Position field for all granules (modified in-place)
        velocity: Velocity field for all granules (modified in-place)
        equilibrium: Equilibrium (rest) positions for all granules
        num_sources: Number of wave sources
        t: Current simulation time (accumulated, seconds)
        freq_boost: Frequency multiplier (applied after slow_mo)
        amp_boost: Amplitude multiplier (for visibility in scaled lattices)
    """
    # Compute temporal parameters (same for all wave sources)
    f_slowed = frequency / config.SLOW_MO * freq_boost
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency (rad/s)

    # Wave number k = 2π/λ (spatial phase variation)
    k = 2.0 * ti.math.pi / wavelength_am  # wave number (radians per attometer)

    # Reference radius for amplitude normalization (one wavelength)
    # Prevents singularity at r=0 and provides physically meaningful normalization
    r_reference = wavelength_am  # attometers

    # Process all granules in parallel (outermost loop = GPU parallelization)
    for granule_idx in range(position.shape[0]):
        # Initialize accumulation variables for wave superposition
        total_displacement = ti.Vector([0.0, 0.0, 0.0])  # sum of displacements from all sources
        total_velocity = ti.Vector([0.0, 0.0, 0.0])  # sum of velocities from all sources

        # Sum contributions from all sources (sequential loop, num_sources iterations)
        for source_idx in range(num_sources):
            # Get precomputed direction and distance for this granule-source pair
            direction = sources_direction[granule_idx, source_idx]
            r = sources_distance_am[granule_idx, source_idx]  # distance in attometers

            # Get source phase offset
            phase_offset = sources_phase_shift[source_idx]

            # Spatial phase: φ = -k·r (negative for outward propagation)
            # Creates spherical wave fronts expanding from source
            spatial_phase = -k * r

            # Total phase: includes spatial phase and source's initial offset
            total_phase = spatial_phase + phase_offset

            # Amplitude falloff for spherical wave: A(r) = A₀·(r₀/r)
            # Prevents division by zero using r_min = 1λ
            r_safe = ti.max(r, r_reference)  # minimum 1 wavelength from source
            amplitude_falloff = r_reference / r_safe

            # Total amplitude at this distance from source
            # Step 1: Apply energy conservation (1/r falloff) and visualization scaling
            amplitude_uncapped = amplitude_am * amplitude_falloff * amp_boost

            # Step 2: Cap amplitude to distance from source (A ≤ r)
            # Prevents non-physical behavior: granules crossing through wave source
            # When A > r, displacement could exceed distance to source, placing granule
            # on opposite side of source (physically impossible for longitudinal waves)
            # This constraint ensures: |x - x_eq| ≤ |x_eq - x_source|
            amplitude_at_r = ti.min(amplitude_uncapped, r)

            # MAIN EQUATION OF MOTION
            # Wave displacement from this source: A(r)·cos(ωt + φ)·direction
            displacement_magnitude = amplitude_at_r * ti.cos(omega * t + total_phase)
            source_displacement = displacement_magnitude * direction

            # Wave velocity from this source: -A(r)·ω·sin(ωt + φ)·direction
            velocity_magnitude = -amplitude_at_r * omega * ti.sin(omega * t + total_phase)
            source_velocity = velocity_magnitude * direction

            # Accumulate this source's contribution (wave superposition)
            total_displacement += source_displacement
            total_velocity += source_velocity

        # Apply superposed wave to granule position and velocity
        position[granule_idx] = equilibrium[granule_idx] + total_displacement
        velocity[granule_idx] = total_velocity
