"""
ENERGY-WAVE ENGINE

LEVEL-1: ON FIELD-BASED METHOD

Wave Physics Engine @spacetime module. Wave dynamics and motion.
"""

import taichi as ti

from openwave.common import colormap, constants

# ================================================================
# Energy-Wave Oscillation Parameters
# ================================================================
base_amplitude_am = constants.EWAVE_AMPLITUDE / constants.ATTOMETER  # am, oscillation amplitude
base_wavelength = constants.EWAVE_LENGTH  # in meters
base_frequency = constants.EWAVE_FREQUENCY  # in Hz
base_frequency_rHz = base_frequency * constants.RONTOSECOND  # in rHz (cycles per rontosecond)
rho = constants.MEDIUM_DENSITY  # medium density (kg/m³)


@ti.kernel
def charge_1lambda(
    wave_field: ti.template(),  # type: ignore
    dt_rs: ti.f32,  # type: ignore
):
    """
    Initialize a spherical outgoing wave within 1 base_wavelength.
    Similar to initiate_charge() but limits the wave to within 1 base_wavelength

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        dt_rs: Time step size (rs)
    """

    # Find center position (in grid indices)
    center_x = wave_field.nx // 2
    center_y = wave_field.ny // 2
    center_z = wave_field.nz // 2

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = 2.0 * ti.math.pi * base_frequency_rHz  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength / wave_field.dx
    k_grid = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Create radial sinusoidal displacement pattern (interior points only)
    # Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Distance from center (in grid indices)
        r_grid = ti.sqrt((i - center_x) ** 2 + (j - center_y) ** 2 + (k - center_z) ** 2)

        # Amplitude = base if r < λ, 0 otherwise, oscillates radially
        amplitude_am_at_r = base_amplitude_am if r_grid < wavelength_grid else 0.0

        # Simple sinusoidal radial pattern
        # Outward displacement from center source: A(r)·cos(ωt - kr)
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        disp = amplitude_am_at_r * ti.cos(omega_rs * 0 - k_grid * r_grid)  # t0 initial condition
        disp_old = amplitude_am_at_r * ti.cos(omega_rs * -dt_rs - k_grid * r_grid)

        # Apply both displacements (in attometers)
        wave_field.displacement_am[i, j, k] = disp  # at t=0
        wave_field.displacement_old_am[i, j, k] = disp_old  # at t=-dt


@ti.kernel
def charge_falloff(
    wave_field: ti.template(),  # type: ignore
    dt_rs: ti.f32,  # type: ignore
):
    """
    Initialize a spherical outgoing wave with 1/r amplitude falloff.

    Similar to initiate_charge() but includes realistic amplitude decay with
    distance (λ/r falloff). Creates a radial sinusoidal displacement pattern
    where amplitude decreases inversely with distance from the source.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        dt_rs: Time step size (rs)
    """

    # Find center position (in grid indices)
    center_x = wave_field.nx // 2
    center_y = wave_field.ny // 2
    center_z = wave_field.nz // 2

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = 2.0 * ti.math.pi * base_frequency_rHz  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength / wave_field.dx
    k_grid = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Create radial sinusoidal displacement pattern (interior points only)
    # Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Distance from center (in grid indices)
        r_grid = ti.sqrt((i - center_x) ** 2 + (j - center_y) ** 2 + (k - center_z) ** 2)

        # Amplitude decreases with distance (1/r falloff)
        r_safe_am = ti.max(r_grid, wavelength_grid)  # clamp to minimum 1λ to avoid singularity
        amplitude_falloff = wavelength_grid / r_safe_am  # λ/r falloff factor
        amplitude_am_at_r = base_amplitude_am * amplitude_falloff

        # Simple sinusoidal radial pattern
        # Outward displacement from center source: A(r)·cos(ωt - kr)
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        disp = amplitude_am_at_r * ti.cos(omega_rs * 0 - k_grid * r_grid)  # t0 initial condition
        disp_old = amplitude_am_at_r * ti.cos(omega_rs * -dt_rs - k_grid * r_grid)

        # Apply both displacements (in attometers)
        wave_field.displacement_am[i, j, k] = disp  # at t=0
        wave_field.displacement_old_am[i, j, k] = disp_old  # at t=-dt


@ti.kernel
def charge_gaussian(
    wave_field: ti.template(),  # type: ignore
):
    """
    Initialize a stationary Gaussian wave packet centered in the wave field.

    Creates a spherical displacement pattern with Gaussian envelope, normalized
    to contain the total energy of the universe. The wave starts at rest
    (zero initial velocity) by setting displacement_old = displacement.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """

    # Find center position (in grid indices)
    center_x = wave_field.nx // 2
    center_y = wave_field.ny // 2
    center_z = wave_field.nz // 2

    # Gaussian width: σ = λ/2 (half wavelength)
    sigma = base_wavelength / 2  # in meters
    sigma_grid = sigma / wave_field.dx  # in grid indices

    # Calculate amplitude to match total universe energy
    # Energy integral: E = ∫ ρ(fA)² × G² dV, where G = exp(-r²/(2σ²))
    # Squared Gaussian integral: ∫ exp(-r²/σ²) dV = π^(3/2) × σ³
    # Solving for A: A = √(E / (ρf² × π^(3/2) × σ³))
    # Restructured to avoid f32 overflow (ρf² ~ 10^72 exceeds f32 max ~ 10^38):
    #   A = √E / (√ρ × f × π^(3/4) × σ^(3/2))
    sqrt_rho_times_f = ti.f32(rho**0.5 * base_frequency)  # ~6.53e36 (within f32)
    g_vol_sqrt = ti.pow(ti.math.pi, 0.75) * ti.pow(sigma, 1.5)  # π^(3/4) × σ^(3/2)
    A_required = ti.sqrt(wave_field.energy) / (sqrt_rho_times_f * g_vol_sqrt)
    A_am = A_required / ti.f32(constants.ATTOMETER)  # convert to attometers

    # Apply Gaussian displacement (interior points only)
    # Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):

        # Squared Distance from center (in grid indices)
        r_squared = (i - center_x) ** 2 + (j - center_y) ** 2 + (k - center_z) ** 2

        # Gaussian envelope: G(r) = exp(-r²/(2σ²))
        gaussian = ti.exp(-r_squared / (2.0 * sigma_grid * sigma_grid))

        wave_field.displacement_am[i, j, k] = A_am * gaussian

    # Set old displacement equal to current (zero initial velocity: ∂ψ/∂t = 0)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        wave_field.displacement_old_am[i, j, k] = wave_field.displacement_am[i, j, k]


@ti.kernel
def charge_full(
    wave_field: ti.template(),  # type: ignore
    dt_rs: ti.f32,  # type: ignore
):
    """
    Initialize a spherical outgoing wave pattern centered in the wave field.

    Creates a radial sinusoidal displacement pattern emanating from the grid center
    using the wave equation: A·cos(ωt - kr). Sets up both current and previous
    timestep displacements for time-stepping propagation.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        dt_rs: Time step size (rs)
    """

    # Find center position (in grid indices)
    center_x = wave_field.nx // 2
    center_y = wave_field.ny // 2
    center_z = wave_field.nz // 2

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = 2.0 * ti.math.pi * base_frequency_rHz  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength / wave_field.dx
    k_grid = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Create radial sinusoidal displacement pattern (interior points only)
    # Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Distance from center (in grid indices)
        r_grid = ti.sqrt((i - center_x) ** 2 + (j - center_y) ** 2 + (k - center_z) ** 2)

        # Simple sinusoidal radial pattern
        # Outward displacement from center source: A·cos(ωt - kr)
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        disp = base_amplitude_am * ti.cos(omega_rs * 0 - k_grid * r_grid)  # t0 initial condition
        disp_old = base_amplitude_am * ti.cos(omega_rs * -dt_rs - k_grid * r_grid)

        # Apply both displacements (in attometers)
        wave_field.displacement_am[i, j, k] = disp  # at t=0
        wave_field.displacement_old_am[i, j, k] = disp_old  # at t=-dt


@ti.kernel
def charge_oscillator_sphere(
    wave_field: ti.template(),  # type: ignore
    elapsed_t_rs: ti.f32,  # type: ignore
):
    """
    Apply harmonic oscillation to a spherical volume at the grid center.

    Creates a uniform displacement within a spherical region using:
        ψ(t) = A·cos(ωt-kr)

    The oscillator acts as a coherent wave source, with all voxels inside
    the sphere oscillating in phase.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        elapsed_t_rs: Elapsed simulation time (rs)
    """
    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = 2.0 * ti.math.pi * base_frequency_rHz  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength / wave_field.dx
    k_grid = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Find center position (in grid indices)
    center_x = wave_field.nx // 2
    center_y = wave_field.ny // 2
    center_z = wave_field.nz // 2

    # Define oscillator sphere radius
    charge_radius_grid = int(0.05 * wave_field.max_grid_size)  # in grid indices

    # Apply oscillating displacement within source sphere
    # Harmonic motion: A·cos(ωt-kr), positive = expansion, negative = compression
    for i, j, k in ti.ndrange(
        (center_x - charge_radius_grid, center_x + charge_radius_grid + 1),
        (center_y - charge_radius_grid, center_y + charge_radius_grid + 1),
        (center_z - charge_radius_grid, center_z + charge_radius_grid + 1),
    ):
        # Check if voxel is within spherical source region
        r_grid = ti.sqrt((i - center_x) ** 2 + (j - center_y) ** 2 + (k - center_z) ** 2)
        if r_grid <= charge_radius_grid:
            wave_field.displacement_am[i, j, k] = base_amplitude_am * ti.cos(
                omega_rs * elapsed_t_rs - k_grid * r_grid
            )


@ti.kernel
def charge_oscillator_falloff(
    wave_field: ti.template(),  # type: ignore
    elapsed_t_rs: ti.f32,  # type: ignore
):
    """
    Apply harmonic oscillation with 1/r amplitude falloff.

    Similar to charge_oscillator() but includes realistic amplitude decay with
    distance (λ/r falloff). Creates a radial sinusoidal displacement pattern
    where amplitude decreases inversely with distance from the source.

    Creates a uniform displacement using:
        ψ(t) = A(r)·cos(ωt-kr), where A(r) = A·(λ/r)

    The oscillator acts as a coherent wave source, with all voxels inside
    the sphere oscillating in phase.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        elapsed_t_rs: Elapsed simulation time (rs)
    """
    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = 2.0 * ti.math.pi * base_frequency_rHz  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength / wave_field.dx
    k_grid = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Find center position (in grid indices)
    center_x = wave_field.nx // 2
    center_y = wave_field.ny // 2
    center_z = wave_field.nz // 2

    # Apply oscillating displacement
    # Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)
    # Harmonic motion: A·cos(ωt-kr), positive = expansion, negative = compression
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Distance from center (in grid indices)
        r_grid = ti.sqrt((i - center_x) ** 2 + (j - center_y) ** 2 + (k - center_z) ** 2)

        # Amplitude decreases with distance (1/r falloff)
        r_safe_am = ti.max(r_grid, wavelength_grid)  # clamp to minimum 1λ to avoid singularity
        amplitude_falloff = wavelength_grid / r_safe_am  # λ/r falloff factor
        amplitude_am_at_r = base_amplitude_am * amplitude_falloff

        wave_field.displacement_am[i, j, k] = amplitude_am_at_r * ti.cos(
            omega_rs * elapsed_t_rs - k_grid * r_grid
        )


@ti.kernel
def damp_energy_sphere(
    wave_field: ti.template(),  # type: ignore
    decay_factor: ti.f32,  # type: ignore
):
    """
    Gradually absorb wave energy within a spherical volume at the grid center.

    Applies exponential damping to displacement values, simulating energy
    absorption. Each frame, displacement is multiplied by decay_factor,
    causing gradual decay toward zero.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        decay_factor: Damping multiplier per frame (e.g., 0.95 = 5% absorption per frame)
    """

    # Find center position (in grid indices)
    center_x = wave_field.nx // 2
    center_y = wave_field.ny // 2
    center_z = wave_field.nz // 2

    # Define damp sphere radius
    damp_radius_grid = int(0.3 * wave_field.max_grid_size)  # in grid indices

    # Apply damping displacement within sphere
    # NOTE: Must be called AFTER propagate_ewave to damp the propagated values
    for i, j, k in ti.ndrange(
        (center_x - damp_radius_grid, center_x + damp_radius_grid + 1),
        (center_y - damp_radius_grid, center_y + damp_radius_grid + 1),
        (center_z - damp_radius_grid, center_z + damp_radius_grid + 1),
    ):
        # Check if voxel is within spherical drain region
        r_grid = ti.sqrt((i - center_x) ** 2 + (j - center_y) ** 2 + (k - center_z) ** 2)
        if r_grid <= damp_radius_grid:
            wave_field.displacement_am[i, j, k] *= decay_factor


@ti.kernel
def damp_energy_full(
    wave_field: ti.template(),  # type: ignore
    decay_factor: ti.f32,  # type: ignore
):
    """
    Gradually absorb wave energy within the entire grid volume.

    Applies exponential damping to displacement values, simulating energy
    absorption. Each frame, displacement is multiplied by decay_factor,
    causing gradual decay toward zero.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        decay_factor: Damping multiplier per frame (e.g., 0.95 = 5% absorption per frame)
    """

    # Apply damping displacement within entire grid
    # NOTE: Must be called AFTER propagate_ewave to damp the propagated values
    for i, j, k in ti.ndrange(
        (0, wave_field.nx),
        (0, wave_field.ny),
        (0, wave_field.nz),
    ):
        wave_field.displacement_am[i, j, k] *= decay_factor


@ti.func
def compute_laplacian_am(
    wave_field: ti.template(),  # type: ignore
    i: ti.i32,  # type: ignore
    j: ti.i32,  # type: ignore
    k: ti.i32,  # type: ignore
):
    """
    Compute Laplacian ∇²ψ at voxel [i,j,k] (6-connectivity).
    ∇²ψ = (∂²ψ/∂x² + ∂²ψ/∂y² + ∂²ψ/∂z²)

    Discrete Laplacian (second derivative in space):
    ∇²ψ[i,j,k] = (ψ[i±1] + ψ[j±1] + ψ[k±1] - 6ψ[i,j,k]) / dx²

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        i, j, k: Voxel indices (must be interior: 0 < i,j,k < n-1)

    Returns:
        Laplacian in units [1/am]
    """
    # 6-connectivity stencil (face neighbors only)
    laplacian_am = (
        wave_field.displacement_am[i + 1, j, k]
        + wave_field.displacement_am[i - 1, j, k]
        + wave_field.displacement_am[i, j + 1, k]
        + wave_field.displacement_am[i, j - 1, k]
        + wave_field.displacement_am[i, j, k + 1]
        + wave_field.displacement_am[i, j, k - 1]
        - 6.0 * wave_field.displacement_am[i, j, k]
    ) / wave_field.dx_am**2

    return laplacian_am


@ti.kernel
def propagate_ewave(
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    c_amrs: ti.f32,  # type: ignore
    dt_rs: ti.f32,  # type: ignore
    elapsed_t_rs: ti.f32,  # type: ignore
):
    """
    Propagate wave displacement using wave equation (PDE Solver).
    Wave Equation: ∂²ψ/∂t² = c²∇²ψ
    Includes wave propagation, reflection at boundaries, superposition
    of wavefronts, and energy conservation (leap-frog).
    Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)

    Discrete Form (Leap-Frog/Verlet):
        ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
        where ∇²ψ = (neighbors_sum - 6·center) / dx²

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        trackers: WaveTrackers instance for tracking wave properties
        c_amrs: Effective wave speed after slow-motion factor (am/rs)
        dt_rs: Time step size (rs)
        elapsed_t_rs: Elapsed simulation time (rs)

    Note: Metal GPU backend may be unstable for grids >720³.
    """

    # Update all interior voxels only
    # Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)
    # Direct range specification avoids conditional branching on GPU
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Compute spatial Laplacian ∇²ψ
        laplacian_am = compute_laplacian_am(wave_field, i, j, k)

        # Leap-frog update
        # Standard form: ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
        wave_field.displacement_new_am[i, j, k] = (
            2.0 * wave_field.displacement_am[i, j, k]
            - wave_field.displacement_old_am[i, j, k]
            + (c_amrs * dt_rs) ** 2 * laplacian_am
        )

        # WAVE TRACKERS ============================================
        # AMPLITUDE tracking envelope, using exponential moving average (EMA)
        # EMA formula: A_new = α * |ψ| + (1 - α) * A_old
        # α controls adaptation speed: higher = faster response, lower = smoother
        # Asymmetric EMA: fast attack (α=0.3) when rising, slow decay (α=0.02) when falling
        # TODO: 2 polarities tracked: longitudinal & transverse
        disp_mag = ti.abs(wave_field.displacement_am[i, j, k])
        current_amp = trackers.amplitudeL_am[i, j, k]
        alpha = 0.3 if disp_mag > current_amp else 0.02
        trackers.amplitudeL_am[i, j, k] = alpha * disp_mag + (1.0 - alpha) * current_amp

        # FREQUENCY tracking, via zero-crossing detection
        # Detect positive-going zero crossing (negative → positive transition)
        # Period = time between consecutive positive zero crossings
        # More robust than peak detection since it's amplitude-independent
        prev_disp = wave_field.displacement_old_am[i, j, k]
        curr_disp = wave_field.displacement_am[i, j, k]
        if prev_disp < 0.0 and curr_disp >= 0.0:  # Zero crossing detected
            period_rs = elapsed_t_rs - trackers.last_crossing[i, j, k]
            if period_rs > dt_rs * 2:  # Filter out spurious crossings
                trackers.frequency_rHz[i, j, k] = 1.0 / period_rs  # in rHz
            trackers.last_crossing[i, j, k] = elapsed_t_rs

    # Swap time levels: old ← current, current ← new
    for i, j, k in ti.ndrange(wave_field.nx, wave_field.ny, wave_field.nz):
        wave_field.displacement_old_am[i, j, k] = wave_field.displacement_am[i, j, k]
        wave_field.displacement_am[i, j, k] = wave_field.displacement_new_am[i, j, k]


@ti.kernel
def _copy_slice_xy(
    trackers: ti.template(),  # type: ignore
    slice_amp: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    mid_z: ti.i32,  # type: ignore
):
    """Copy center XY slice (fixed z) to 2D buffer."""
    for i, j in slice_amp:
        slice_amp[i, j] = trackers.amplitudeL_am[i, j, mid_z]
        slice_freq[i, j] = trackers.frequency_rHz[i, j, mid_z]


@ti.kernel
def _copy_slice_xz(
    trackers: ti.template(),  # type: ignore
    slice_amp: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    mid_y: ti.i32,  # type: ignore
):
    """Copy center XZ slice (fixed y) to 2D buffer."""
    for i, k in slice_amp:
        slice_amp[i, k] = trackers.amplitudeL_am[i, mid_y, k]
        slice_freq[i, k] = trackers.frequency_rHz[i, mid_y, k]


@ti.kernel
def _copy_slice_yz(
    trackers: ti.template(),  # type: ignore
    slice_amp: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    mid_x: ti.i32,  # type: ignore
):
    """Copy center YZ slice (fixed x) to 2D buffer."""
    for j, k in slice_amp:
        slice_amp[j, k] = trackers.amplitudeL_am[mid_x, j, k]
        slice_freq[j, k] = trackers.frequency_rHz[mid_x, j, k]


# Cached slice buffers (initialized on first call)
_slice_xy_amp = None
_slice_xy_freq = None
_slice_xz_amp = None
_slice_xz_freq = None
_slice_yz_amp = None
_slice_yz_freq = None


def sample_avg_trackers(
    wave_field,
    trackers,
):
    """
    Estimate average amplitude and frequency by sampling 3 orthogonal planes.

    Samples XY, XZ, and YZ center slices to avoid full 3D GPU->CPU transfer.
    Provides a representative estimate without the performance penalty.

    Args:
        wave_field: WaveField instance containing grid dimensions
        trackers: WaveTrackers instance with per-voxel and average fields
    """
    global _slice_xy_amp, _slice_xy_freq
    global _slice_xz_amp, _slice_xz_freq
    global _slice_yz_amp, _slice_yz_freq

    nx, ny, nz = wave_field.nx, wave_field.ny, wave_field.nz

    # Initialize slice buffers once
    if _slice_xy_amp is None:
        _slice_xy_amp = ti.field(dtype=ti.f32, shape=(nx, ny))
        _slice_xy_freq = ti.field(dtype=ti.f32, shape=(nx, ny))
        _slice_xz_amp = ti.field(dtype=ti.f32, shape=(nx, nz))
        _slice_xz_freq = ti.field(dtype=ti.f32, shape=(nx, nz))
        _slice_yz_amp = ti.field(dtype=ti.f32, shape=(ny, nz))
        _slice_yz_freq = ti.field(dtype=ti.f32, shape=(ny, nz))

    # Copy 3 center slices to 2D buffers (parallel kernels)
    mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2
    _copy_slice_xy(trackers, _slice_xy_amp, _slice_xy_freq, mid_z)
    _copy_slice_xz(trackers, _slice_xz_amp, _slice_xz_freq, mid_y)
    _copy_slice_yz(trackers, _slice_yz_amp, _slice_yz_freq, mid_x)

    # Transfer 2D slices to numpy, excluding boundary voxels (always zero)
    xy_amp = _slice_xy_amp.to_numpy()[1:-1, 1:-1]
    xy_freq = _slice_xy_freq.to_numpy()[1:-1, 1:-1]
    xz_amp = _slice_xz_amp.to_numpy()[1:-1, 1:-1]
    xz_freq = _slice_xz_freq.to_numpy()[1:-1, 1:-1]
    yz_amp = _slice_yz_amp.to_numpy()[1:-1, 1:-1]
    yz_freq = _slice_yz_freq.to_numpy()[1:-1, 1:-1]

    # Compute combined average from all 3 planes
    total_amp = xy_amp.sum() + xz_amp.sum() + yz_amp.sum()
    total_freq = xy_freq.sum() + xz_freq.sum() + yz_freq.sum()
    n_samples = xy_amp.size + xz_amp.size + yz_amp.size

    trackers.avg_amplitudeL_am[None] = float(total_amp) / n_samples
    trackers.avg_frequency_rHz[None] = float(total_freq) / n_samples


@ti.kernel
def update_flux_mesh_colors(
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    color_palette: ti.i32,  # type: ignore
):
    """
    Update flux mesh colors by sampling wave properties from voxel grid.

    Samples wave displacement at each plane vertex position and maps it to a color.
    Should be called every frame after wave propagation to update visualization.

    Color palettes:
        1 (redshift): red=negative, gray=zero, blue=positive displacement
        2 (ironbow): black-red-yellow-white heat map for amplitude
        3 (blueprint): blue gradient for frequency visualization

    Args:
        wave_field: WaveField instance containing flux mesh fields and displacement data
        trackers: WaveTrackers instance with amplitude/frequency data for color scaling
        color_palette: Color palette selection (1=redshift, 2=ironbow, 3=blueprint)
    """

    # Get center indices for each Plane
    center_i = wave_field.nx // 2
    center_j = wave_field.ny // 2
    center_k = wave_field.nz // 2

    # ================================================================
    # XY Plane: Sample at z = center_k
    # ================================================================
    # Always update all planes (conditionals cause GPU branch divergence)
    for i, j in ti.ndrange(wave_field.nx, wave_field.ny):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[i, j, center_k]
        amp_value = trackers.amplitudeL_am[i, j, center_k]
        freq_value = trackers.frequency_rHz[i, j, center_k]

        # Map value to color using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_frequency_rHz[None] * 2
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_ironbow_color(
                amp_value, 0, trackers.avg_amplitudeL_am[None] * 2
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_redshift_color(
                disp_value,
                -trackers.avg_amplitudeL_am[None] * 2,
                trackers.avg_amplitudeL_am[None] * 2,
            )

    # ================================================================
    # XZ Plane: Sample at y = center_j
    # ================================================================
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[i, center_j, k]
        amp_value = trackers.amplitudeL_am[i, center_j, k]
        freq_value = trackers.frequency_rHz[i, center_j, k]

        # Map value to color using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_frequency_rHz[None] * 2
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_ironbow_color(
                amp_value, 0, trackers.avg_amplitudeL_am[None] * 2
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_redshift_color(
                disp_value,
                -trackers.avg_amplitudeL_am[None] * 2,
                trackers.avg_amplitudeL_am[None] * 2,
            )

    # ================================================================
    # YZ Plane: Sample at x = center_i
    # ================================================================
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[center_i, j, k]
        amp_value = trackers.amplitudeL_am[center_i, j, k]
        freq_value = trackers.frequency_rHz[center_i, j, k]

        # Map value to color using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_frequency_rHz[None] * 2
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_ironbow_color(
                amp_value, 0, trackers.avg_amplitudeL_am[None] * 2
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_redshift_color(
                disp_value,
                -trackers.avg_amplitudeL_am[None] * 2,
                trackers.avg_amplitudeL_am[None] * 2,
            )


# TODO: Move to dedicated numerical analysis/plotting module
def plot_charge_profile(wave_field):
    """
    Plot the displacement profile along the x-axis through the center of the wave field.

    Args:
        wave_field: WaveField instance containing displacement data
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get center indices
    center_j = wave_field.ny // 2
    center_k = wave_field.nz // 2

    # Extract displacement along x-axis at center (y, z)
    x_indices = np.arange(wave_field.nx)
    displacements_L = np.zeros(wave_field.nx)
    displacements_T = np.zeros(wave_field.nx)

    # Sample longitudinal displacement values
    for i in range(wave_field.nx):
        displacements_L[i] = wave_field.displacement_am[i, center_j, center_k]
        # displacements_L[i] = wave_field.displacement_old_am[i, center_j, center_k]
        displacements_T[i] = 0.0

    # Calculate distance from center in grid indices
    center_x = wave_field.nx // 2
    distances = x_indices - center_x

    # Create the plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 6), facecolor=colormap.DARK_GRAY[1])
    fig.suptitle("OPENWAVE Analytics", fontsize=20, family="Monospace")

    # Plot 1: Longitudinal Displacement vs distance from center
    plt.subplot(1, 2, 1)
    plt.plot(
        distances,
        displacements_L,
        color=colormap.viridis_palette[0][1],
        linewidth=4,
        label="LONGITUDINAL",
    )
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.3)
    plt.ylim(-1.5, 1.5)
    plt.xlabel("Distance from Center (grid indices)", family="Monospace")
    plt.ylabel("Displacement (attometers)", family="Monospace")
    plt.title("INITIAL CHARGE PROFILE", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Transverse Displacement vs distance from center
    plt.subplot(1, 2, 2)
    plt.plot(
        distances,
        displacements_T,
        color=colormap.viridis_palette[4][1],
        linewidth=4,
        label="TRANSVERSE",
    )
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.3)
    plt.ylim(-1.5, 1.5)
    plt.xlabel("Distance from Center (grid indices)", family="Monospace")
    plt.ylabel("Displacement (attometers)", family="Monospace")
    plt.title("INITIAL CHARGE PROFILE", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save to research directory
    from pathlib import Path

    save_path = (
        Path(__file__).parent.parent
        / "xperiments"
        / "5_level1_field_based"
        / "_research"
        / "displacement_profile.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    # plt.show()
