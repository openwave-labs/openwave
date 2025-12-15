"""
ENERGY-WAVE ENGINE

LEVEL-1: ON FIELD-BASED METHOD

Wave Physics Engine @spacetime module. Wave dynamics and motion.
"""

import taichi as ti
import numpy as np

from openwave.common import colormap, constants

# ================================================================
# Energy-Wave Oscillation Parameters
# ================================================================
base_amplitude_am = constants.EWAVE_AMPLITUDE / constants.ATTOMETER  # am, oscillation amplitude
base_wavelength = constants.EWAVE_LENGTH  # in meters
base_frequency = constants.EWAVE_FREQUENCY  # in Hz
base_frequency_rHz = constants.EWAVE_FREQUENCY * constants.RONTOSECOND  # in rHz (1/rontosecond)
rho = constants.MEDIUM_DENSITY  # medium density (kg/m³)


# ================================================================
# STATIC CHARGING methods (one-time pulse pattern)
# ================================================================


@ti.kernel
def charge_full(
    wave_field: ti.template(),  # type: ignore
    dt_rs: ti.f32,  # type: ignore
    boost: ti.f32,  # type: ignore
):
    """
    Initialize a spherical outgoing wave pattern centered in the wave field.

    Creates a radial sinusoidal displacement pattern emanating from the grid center
    using the wave motion: A·cos(ωt - kr). Sets up both current and previous
    timestep displacements for time-stepping propagation.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        dt_rs: Time step size (rs)
        boost: Oscillation amplitude multiplier
    """

    # Find center position (in grid indices)
    center_x = wave_field.nx // 2
    center_y = wave_field.ny // 2
    center_z = wave_field.nz // 2

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = (
        2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor
    )  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength * wave_field.scale_factor / wave_field.dx
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
        disp = (
            base_amplitude_am
            * boost
            * wave_field.scale_factor
            * ti.cos(omega_rs * 0 - k_grid * r_grid)
        )  # t0
        disp_old = (
            base_amplitude_am
            * boost
            * wave_field.scale_factor
            * ti.cos(omega_rs * -dt_rs - k_grid * r_grid)
        )  # t-dt

        # Apply both displacements (in attometers)
        wave_field.psiL_am[i, j, k] = disp  # at t=0
        wave_field.psiL_old_am[i, j, k] = disp_old  # at t=-dt


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
    sigma = base_wavelength * wave_field.scale_factor / 2  # in meters
    sigma_grid = sigma / wave_field.dx  # in grid indices

    # Calculate amplitude to match total universe energy
    # Energy integral: E = ∫ ρ(fA)² × G² dV, where G = exp(-r²/(2σ²))
    # Squared Gaussian integral: ∫ exp(-r²/σ²) dV = π^(3/2) × σ³
    # Solving for A: A = √(E / (ρf² × π^(3/2) × σ³))
    # Restructured to avoid f32 overflow (ρf² ~ 10^72 exceeds f32 max ~ 10^38):
    #   A = √E / (√ρ × f × π^(3/4) × σ^(3/2))
    sqrt_rho_times_f = ti.f32(rho**0.5 * base_frequency)  # ~6.53e36 (within f32)
    g_vol_sqrt = ti.pow(ti.math.pi, 0.75) * ti.pow(sigma, 1.5)  # π^(3/4) × σ^(3/2)
    A_required = ti.sqrt(wave_field.nominal_energy) / (sqrt_rho_times_f * g_vol_sqrt)
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

        wave_field.psiL_am[i, j, k] = A_am * gaussian

    # Set old displacement equal to current (zero initial velocity: ∂ψ/∂t = 0)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        wave_field.psiL_old_am[i, j, k] = wave_field.psiL_am[i, j, k]


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
    omega_rs = (
        2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor
    )  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength * wave_field.scale_factor / wave_field.dx
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
        amplitude_am_at_r = base_amplitude_am * wave_field.scale_factor * amplitude_falloff

        # Simple sinusoidal radial pattern
        # Outward displacement from center source: A(r)·cos(ωt - kr)
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        disp = amplitude_am_at_r * ti.cos(omega_rs * 0 - k_grid * r_grid)  # t0 initial condition
        disp_old = amplitude_am_at_r * ti.cos(omega_rs * -dt_rs - k_grid * r_grid)

        # Apply both displacements (in attometers)
        wave_field.psiL_am[i, j, k] = disp  # at t=0
        wave_field.psiL_old_am[i, j, k] = disp_old  # at t=-dt


@ti.kernel
def charge_1lambda(
    wave_field: ti.template(),  # type: ignore
    dt_rs: ti.f32,  # type: ignore
):
    """
    Initialize a spherical outgoing wave within 1 wavelength.
    Similar to initiate_charge() but limits the wave to within 1 wavelength

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        dt_rs: Time step size (rs)
    """

    # Find center position (in grid indices)
    center_x = wave_field.nx // 2
    center_y = wave_field.ny // 2
    center_z = wave_field.nz // 2

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = (
        2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor
    )  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength * wave_field.scale_factor / wave_field.dx
    k_grid = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Create radial sinusoidal displacement pattern (interior points only)
    # Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Distance from center (in grid indices)
        r_grid = ti.sqrt((i - center_x) ** 2 + (j - center_y) ** 2 + (k - center_z) ** 2)

        # Amplitude = base if r < λ, 0 otherwise, oscillates radially
        amplitude_am_at_r = (
            base_amplitude_am * wave_field.scale_factor if r_grid < wavelength_grid else 0.0
        )

        # Simple sinusoidal radial pattern
        # Outward displacement from center source: A(r)·cos(ωt - kr)
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        disp = amplitude_am_at_r * ti.cos(omega_rs * 0 - k_grid * r_grid)  # t0 initial condition
        disp_old = amplitude_am_at_r * ti.cos(omega_rs * -dt_rs - k_grid * r_grid)

        # Apply both displacements (in attometers)
        wave_field.psiL_am[i, j, k] = disp  # at t=0
        wave_field.psiL_old_am[i, j, k] = disp_old  # at t=-dt


# ================================================================
# CHARGING CONTROL (smooth envelope for stability)
# ================================================================


def compute_charge_envelope(charge_level: float) -> float:
    """
    Compute smooth charging envelope based on current charge level.

    HYBRID STRATEGY: Static pulse provides energy at initialization.
    Dynamic chargers do the top-off from 70% to 100% with very soft landing.

    Envelope curve:
    - Full power (0-70%): Strong charging to reach target zone
    - Soft taper (70-100%): Smooth S-curve for very gentle landing
    - Off (100%+): No charging, let natural equilibrium hold

    Args:
        charge_level: Current energy as fraction of nominal (0.0 to 1.5+)

    Returns:
        float: Envelope value 0.0 to 1.0
    """
    TAPER_START = 0.70  # Start soft landing at 70%
    TARGET = 1.00  # Target charge level

    if charge_level >= TARGET:
        return 0.0  # At/above target: no charging
    elif charge_level >= TAPER_START:
        # SOFT LANDING (70-100%): Smoothstep S-curve for very gentle approach
        # Slower decay near target than cosine
        t = (charge_level - TAPER_START) / (TARGET - TAPER_START)  # 0 at 70%, 1 at 100%
        # Smoothstep: 3t² - 2t³ gives S-curve, invert for decay
        smooth = t * t * (3.0 - 2.0 * t)  # 0→1 S-curve
        return 1.0 - smooth  # 1.0 at 70%, 0.0 at 100%
    else:
        # FULL POWER (0-70%): Keep charging strong
        return 1.0


def compute_damping_factor(
    charge_level: float, target: float = 1.0, tolerance: float = 0.10
) -> float:
    """
    Compute proportional damping factor based on charge level.

    HYBRID STRATEGY: Light baseline damping always to balance charger injection.
    Stronger proportional damping above target to correct overshoots quickly.

    Args:
        charge_level: Current energy as fraction of nominal
        target: Target charge level (default 1.0 = 100%)
        tolerance: Overshoot tolerance before max damping (default 0.10 = 10%)

    Returns:
        float: Damping factor 0.98 to 0.9999
    """
    BASELINE = 0.99998  # Ultra light: ~0.002% per step, minimal drain

    if charge_level <= target:
        return BASELINE  # Light baseline damping
    else:
        # Above target: stronger proportional damping
        overshoot = (charge_level - target) / tolerance
        overshoot = min(overshoot, 5.0)
        # Damping strength: BASELINE at target, 0.98 at target + 5*tolerance
        return BASELINE - 0.004 * overshoot


# ================================================================
# DYNAMIC CHARGING methods (oscillator during simulation)
# ================================================================


@ti.kernel
def charge_oscillator_sphere(
    wave_field: ti.template(),  # type: ignore
    elapsed_t_rs: ti.f32,  # type: ignore
    radius: ti.f32,  # type: ignore
    boost: ti.f32,  # type: ignore
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
        boost: Oscillation amplitude multiplier
    """
    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = (
        2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor
    )  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength * wave_field.scale_factor / wave_field.dx
    k_grid = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Find center position (in grid indices)
    center_x = wave_field.nx // 2
    center_y = wave_field.ny // 2
    center_z = wave_field.nz // 2

    # Define oscillator sphere radius (a fraction of min edge voxels)
    charge_radius_grid = int(radius * wave_field.min_grid_size)  # in grid indices

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
            wave_field.psiL_am[i, j, k] = (
                base_amplitude_am
                * boost
                * wave_field.scale_factor
                * ti.cos(omega_rs * elapsed_t_rs - k_grid * r_grid)
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
    omega_rs = (
        2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor
    )  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength * wave_field.scale_factor / wave_field.dx
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
        amplitude_am_at_r = base_amplitude_am * wave_field.scale_factor * amplitude_falloff

        wave_field.psiL_am[i, j, k] = amplitude_am_at_r * ti.cos(
            omega_rs * elapsed_t_rs - k_grid * r_grid
        )


@ti.kernel
def charge_oscillator_wall(
    wave_field: ti.template(),  # type: ignore
    elapsed_t_rs: ti.f32,  # type: ignore
    sources: ti.i32,  # type: ignore
    boost: ti.f32,  # type: ignore
):
    """
    Apply additive harmonic oscillation near boundary walls from source voxels.

    ADDITIVE MODE: Adds oscillation to existing displacement rather than
    overwriting. This prevents energy loss when propagated wave amplitude
    exceeds charger amplitude, and allows natural wave superposition.

    Formula: ψ_new = ψ_old + A·cos(ωt) * boost

    Sources are placed 1 voxel interior from walls to avoid conflict with
    Neumann BC ghost cell updates.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        elapsed_t_rs: Elapsed simulation time (rs)
        sources: Number of source points per edge
        boost: Oscillation amplitude multiplier (envelope-controlled)
    """
    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = (
        2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor
    )  # angular frequency (rad/rs)

    # Source grid calculation (interior, excluding boundary ghost cells)
    # Range is [1, n-2] to stay 1 voxel away from boundaries
    interior_nx = wave_field.nx - 2
    interior_ny = wave_field.ny - 2
    interior_nz = wave_field.nz - 2
    sources_x = min(interior_nx, int(sources * interior_nx / wave_field.max_grid_size))
    sources_y = min(interior_ny, int(sources * interior_ny / wave_field.max_grid_size))
    sources_z = min(interior_nz, int(sources * interior_nz / wave_field.max_grid_size))
    sources_x = max(sources_x, 1)
    sources_y = max(sources_y, 1)
    sources_z = max(sources_z, 1)
    skip_x = interior_nx / sources_x
    skip_y = interior_ny / sources_y
    skip_z = interior_nz / sources_z
    offset_x = skip_x / 2 + 1  # +1 to start at index 1, not 0
    offset_y = skip_y / 2 + 1
    offset_z = skip_z / 2 + 1

    # Oscillation delta for this timestep (ADDITIVE, not absolute)
    osc_delta = (
        base_amplitude_am * boost * wave_field.scale_factor * ti.cos(omega_rs * elapsed_t_rs)
    )

    # ADD oscillating displacement on 6 near-boundary planes (1 voxel interior)
    # Harmonic motion: ψ += A·cos(ωt), positive = expansion, negative = compression
    # Z-faces: k=1 and k=nz-2
    for i, j in ti.ndrange(sources_x, sources_y):
        idx_x = int(i * skip_x + offset_x)
        idx_y = int(j * skip_y + offset_y)
        wave_field.psiL_am[idx_x, idx_y, 1] += osc_delta
        wave_field.psiL_am[idx_x, idx_y, wave_field.nz - 2] += osc_delta

    # Y-faces: j=1 and j=ny-2
    for i, k in ti.ndrange(sources_x, sources_z):
        idx_x = int(i * skip_x + offset_x)
        idx_z = int(k * skip_z + offset_z)
        wave_field.psiL_am[idx_x, 1, idx_z] += osc_delta
        wave_field.psiL_am[idx_x, wave_field.ny - 2, idx_z] += osc_delta

    # X-faces: i=1 and i=nx-2
    for j, k in ti.ndrange(sources_y, sources_z):
        idx_y = int(j * skip_y + offset_y)
        idx_z = int(k * skip_z + offset_z)
        wave_field.psiL_am[1, idx_y, idx_z] += osc_delta
        wave_field.psiL_am[wave_field.nx - 2, idx_y, idx_z] += osc_delta


# ================================================================
# DYNAMIC DAMPING methods (energy sink during simulation)
# ================================================================


@ti.kernel
def damp_full(
    wave_field: ti.template(),  # type: ignore
    damping_factor: ti.f32,  # type: ignore
):
    """
    Gradually absorb wave energy within the entire grid volume.

    Applies exponential damping to displacement values, simulating energy
    absorption. Each frame, displacement is multiplied by damping_factor,
    causing gradual decay toward zero.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        damping_factor: Damping multiplier per frame (e.g., 0.95 = 5% absorption per frame)
    """

    # Apply damping displacement within entire grid
    # NOTE: Must be called AFTER propagate_ewave to damp the propagated values
    for i, j, k in ti.ndrange(
        (0, wave_field.nx),
        (0, wave_field.ny),
        (0, wave_field.nz),
    ):
        wave_field.psiL_am[i, j, k] *= damping_factor


# ================================================================
# WAVE PROPAGATION ENGINE
# ================================================================


@ti.func
def compute_laplacian6(
    wave_field: ti.template(),  # type: ignore
    i: ti.i32,  # type: ignore
    j: ti.i32,  # type: ignore
    k: ti.i32,  # type: ignore
):
    """
    Compute Laplacian ∇²ψ at voxel [i,j,k]
    Using 6-connectivity stencil, 2nd-order finite difference.
    ∇²ψ = (∂²ψ/∂x² + ∂²ψ/∂y² + ∂²ψ/∂z²)

    Discrete Laplacian (second derivative in space):
    Formula: ∇²ψ ≈ (face_sum - 6·center) / dx²

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        i, j, k: Voxel indices (must be interior: 0 < i,j,k < n-1)

    Returns:
        Laplacian in units [1/am]
    """
    # Face neighbors (6 total): ψ[i±1] + ψ[j±1] + ψ[k±1]
    face_sum = (
        wave_field.psiL_am[i + 1, j, k]
        + wave_field.psiL_am[i - 1, j, k]
        + wave_field.psiL_am[i, j + 1, k]
        + wave_field.psiL_am[i, j - 1, k]
        + wave_field.psiL_am[i, j, k + 1]
        + wave_field.psiL_am[i, j, k - 1]
    )

    center = wave_field.psiL_am[i, j, k]
    laplacian6_am = (face_sum - 6.0 * center) / (wave_field.dx_am**2)

    return laplacian6_am


@ti.func
def compute_laplacian18(
    wave_field: ti.template(),  # type: ignore
    i: ti.i32,  # type: ignore
    j: ti.i32,  # type: ignore
    k: ti.i32,  # type: ignore
):
    """
    Compute Laplacian ∇²ψ at voxel [i,j,k]
    Using 18-connectivity stencil, 4th-order finite difference.
    ∇²ψ = (∂²ψ/∂x² + ∂²ψ/∂y² + ∂²ψ/∂z²)

    18-point stencil: 6 face neighbors + 12 edge neighbors.
    Weights derived from Taylor expansion for isotropic Laplacian:
    - Face neighbors (distance=dx): weight = 1
    - Edge neighbors (distance=√2·dx): weight = 0.5
    - Center: weight = -12 (ensures sum=0 for constant fields)

    Formula: ∇²ψ ≈ (face_sum + 0.5·edge_sum - 12·center) / (3·dx²)

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        i, j, k: Voxel indices (must be interior: 1 < i,j,k < n-2)

    Returns:
        Laplacian in units [1/am]
    """
    # Face neighbors (6 total): ψ[i±1] + ψ[j±1] + ψ[k±1]
    face_sum = (
        wave_field.psiL_am[i + 1, j, k]
        + wave_field.psiL_am[i - 1, j, k]
        + wave_field.psiL_am[i, j + 1, k]
        + wave_field.psiL_am[i, j - 1, k]
        + wave_field.psiL_am[i, j, k + 1]
        + wave_field.psiL_am[i, j, k - 1]
    )

    # Edge neighbors (12 total): diagonal pairs in each plane
    edge_sum = (
        # XY plane edges (4)
        wave_field.psiL_am[i + 1, j + 1, k]
        + wave_field.psiL_am[i + 1, j - 1, k]
        + wave_field.psiL_am[i - 1, j + 1, k]
        + wave_field.psiL_am[i - 1, j - 1, k]
        # XZ plane edges (4)
        + wave_field.psiL_am[i + 1, j, k + 1]
        + wave_field.psiL_am[i + 1, j, k - 1]
        + wave_field.psiL_am[i - 1, j, k + 1]
        + wave_field.psiL_am[i - 1, j, k - 1]
        # YZ plane edges (4)
        + wave_field.psiL_am[i, j + 1, k + 1]
        + wave_field.psiL_am[i, j + 1, k - 1]
        + wave_field.psiL_am[i, j - 1, k + 1]
        + wave_field.psiL_am[i, j - 1, k - 1]
    )

    center = wave_field.psiL_am[i, j, k]
    laplacian18_am = (face_sum + 0.5 * edge_sum - 12.0 * center) / (3.0 * wave_field.dx_am**2)

    return laplacian18_am


@ti.kernel
def propagate_wave(
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

    Boundary Conditions Options:
    - Dirichlet BC (ψ = 0) at edges for fixed boundaries.
    Implemented by skipping boundary voxels in update loop.
    - Neumann BC (∂ψ/∂n = 0) for energy-conserving reflection.
    Implemented via ghost cell copy: boundary = adjacent interior value.

    Discrete Form (Leap-Frog/Verlet):
        ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
        where ∇²ψ = (neighbors_sum - 6·center) / dx²

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        trackers: WaveTrackers instance for tracking wave properties
        c_amrs: Effective wave speed after slow-motion factor (am/rs)
        dt_rs: Time step size (rs)
        elapsed_t_rs: Elapsed simulation time (rs)

    Note: On M4 Max 48GB, wave propagation becomes incorrect above ~350M voxels.
        Cause unknown - may be GPU saturation, thermal throttling, or backend limits.
        Needs testing on other hardware to isolate the issue.
    """
    # Grid dimensions for boundary handling
    nx, ny, nz = wave_field.nx, wave_field.ny, wave_field.nz

    # ================================================================
    # NEUMANN BC: Apply ghost cell copy BEFORE Laplacian computation
    # ∂ψ/∂n = 0 means boundary = adjacent interior value
    # This ensures waves reflect perfectly without energy loss
    # ================================================================

    # # X-faces (i=0 and i=nx-1)
    # for j, k in ti.ndrange(ny, nz):
    #     wave_field.psiL_am[0, j, k] = wave_field.psiL_am[1, j, k]
    #     wave_field.psiL_am[nx - 1, j, k] = wave_field.psiL_am[nx - 2, j, k]

    # # Y-faces (j=0 and j=ny-1)
    # for i, k in ti.ndrange(nx, nz):
    #     wave_field.psiL_am[i, 0, k] = wave_field.psiL_am[i, 1, k]
    #     wave_field.psiL_am[i, ny - 1, k] = wave_field.psiL_am[i, ny - 2, k]

    # # Z-faces (k=0 and k=nz-1)
    # for i, j in ti.ndrange(nx, ny):
    #     wave_field.psiL_am[i, j, 0] = wave_field.psiL_am[i, j, 1]
    #     wave_field.psiL_am[i, j, nz - 1] = wave_field.psiL_am[i, j, nz - 2]

    # ================================================================
    # WAVE PROPAGATION: Update voxels using Leap-Frog
    # ================================================================
    # Update interior voxels only (Dirichlet BC: ψ=0 at edges)
    # 6-point Laplacian: needs 1-cell buffer → range (1, n-1)
    # 18-point Laplacian: needs 2-cell buffer → range (2, n-2)
    for i, j, k in ti.ndrange((1, nx - 1), (1, ny - 1), (1, nz - 1)):  # 6-pt
        # for i, j, k in ti.ndrange((2, nx - 2), (2, ny - 2), (2, nz - 2)):  # 18-pt
        # Compute spatial Laplacian ∇²ψ (toggle between 6-pt and 18-pt)
        laplacian_am = compute_laplacian6(wave_field, i, j, k)
        # laplacian_am = compute_laplacian18(wave_field, i, j, k)  # 3x memory access

        # Leap-Frog update
        # Standard form: ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
        wave_field.psiL_new_am[i, j, k] = (
            2.0 * wave_field.psiL_am[i, j, k]
            - wave_field.psiL_old_am[i, j, k]
            + (c_amrs * dt_rs) ** 2 * laplacian_am
        )

        # WAVE-TRACKERS ============================================
        # RMS AMPLITUDE tracking via EMA on ψ² (squared displacement)
        # Running RMS: tracks √⟨ψ²⟩ - the energy-equivalent amplitude
        # Used for: energy calculation, force gradients, visualization scaling
        # Physics: particles respond to time-averaged energy density, not
        # instantaneous displacement (inertia acts as low-pass filter at ~10²⁵ Hz)
        # EMA on ψ²: rms² = α * ψ² + (1 - α) * rms²_old, then rms = √(rms²)
        # α controls adaptation speed: higher = faster response, lower = smoother
        # TODO: 2 polarities tracked: longitudinal & transverse
        disp_squared = wave_field.psiL_am[i, j, k] ** 2
        current_rms_squared = trackers.ampL_am[i, j, k] ** 2
        alpha_rms = 0.005  # EMA smoothing factor for RMS tracking
        new_rms_squared = alpha_rms * disp_squared + (1.0 - alpha_rms) * current_rms_squared
        trackers.ampL_am[i, j, k] = ti.sqrt(new_rms_squared)

        # FREQUENCY tracking, via zero-crossing detection with EMA smoothing
        # Detect positive-going zero crossing (negative → positive transition)
        # Period = time between consecutive positive zero crossings
        # More robust than peak detection since it's amplitude-independent
        # EMA smoothing: f_new = α * f_measured + (1 - α) * f_old
        # α controls adaptation speed: higher = faster response, lower = smoother
        prev_disp = wave_field.psiL_old_am[i, j, k]
        curr_disp = wave_field.psiL_am[i, j, k]
        if prev_disp < 0.0 and curr_disp >= 0.0:  # Zero crossing detected
            period_rs = elapsed_t_rs - trackers.last_crossing[i, j, k]
            if period_rs > dt_rs * 2:  # Filter out spurious crossings
                measured_freq = 1.0 / period_rs  # in rHz
                current_freq = trackers.freq_rHz[i, j, k]
                alpha_freq = 0.05  # EMA smoothing factor for frequency
                trackers.freq_rHz[i, j, k] = (
                    alpha_freq * measured_freq + (1.0 - alpha_freq) * current_freq
                )
            trackers.last_crossing[i, j, k] = elapsed_t_rs

    # Swap time levels: old ← current, current ← new
    for i, j, k in ti.ndrange(nx, ny, nz):
        wave_field.psiL_old_am[i, j, k] = wave_field.psiL_am[i, j, k]
        wave_field.psiL_am[i, j, k] = wave_field.psiL_new_am[i, j, k]

    # TODO: Testing Wave Center Interaction with Energy Waves
    # WCs modify Energy Wave character (amplitude/phase/lambda/mode) as they pass through
    # Standing Waves should form around WCs as visual artifacts of interaction
    # Energy Waves are Isotropic (omnidirectional) so reflection gets canceled out
    # interact_wc_lens(wave_field)
    # interact_wc_newmann(wave_field)
    # interact_wc_dirichlet(wave_field)
    # interact_wc_signal(wave_field)


# ================================================================
# WAVE CENTER INTERACTIONS
# ================================================================


@ti.func
def interact_wc_lens(wave_field: ti.template()):  # type: ignore
    """
    TEST: Wave center as LENS - clamp-based amplification
    Amplifies/focuses waves rather than blocking
    Allow normal wave propagation but CLAMP the minimum amplitude at WC
    WC is a point-like region where wave amplitude is amplified
    This creates a phase singularity that surrounding waves conform to
    If |ψ| < threshold, boost it to threshold (preserving sign)
    This creates an amplitude floor, not ceiling — no runaway growth

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 5 // 6, wave_field.ny * 5 // 6, wave_field.nz // 2
    wc2x, wc2y, wc2z = wave_field.nx * 19 // 24, wave_field.ny * 19 // 24, wave_field.nz // 2
    amplification = 10.0  # how much the WC amplifies local wave amplitude

    # Reference amplitude from wave field (the base amplitude used for charging)
    ref_amplitude = base_amplitude_am * wave_field.scale_factor
    min_amplitude = ref_amplitude * amplification

    # If amplitude is below minimum, boost it (preserve sign/phase)
    current_val_old = wave_field.psiL_old_am[wc1x, wc1y, wc1z]
    if ti.abs(current_val_old) < min_amplitude:
        phase_sign_old = 1.0 if current_val_old >= 0.0 else -1.0
        wave_field.psiL_old_am[wc1x, wc1y, wc1z] = phase_sign_old * min_amplitude

    current_val = wave_field.psiL_am[wc1x, wc1y, wc1z]
    if ti.abs(current_val) < min_amplitude:
        phase_sign = 1.0 if current_val >= 0.0 else -1.0
        wave_field.psiL_am[wc1x, wc1y, wc1z] = phase_sign * min_amplitude

    # If amplitude is below minimum, boost it (preserve sign/phase)
    current_val_old = wave_field.psiL_old_am[wc2x, wc2y, wc2z]
    if ti.abs(current_val_old) < min_amplitude:
        phase_sign_old = 1.0 if current_val_old >= 0.0 else -1.0
        wave_field.psiL_old_am[wc2x, wc2y, wc2z] = phase_sign_old * min_amplitude

    current_val = wave_field.psiL_am[wc2x, wc2y, wc2z]
    if ti.abs(current_val) < min_amplitude:
        phase_sign = 1.0 if current_val >= 0.0 else -1.0
        wave_field.psiL_am[wc2x, wc2y, wc2z] = phase_sign * min_amplitude


@ti.func
def interact_wc_newmann(wave_field: ti.template()):  # type: ignore
    """
    TEST: Neumann boundary condition on wave center sphere surface
    ∂ψ/∂n = 0 → copy outer values to inner surface (zero gradient)
    This reflects waves WITHOUT phase inversion (soft/free-end reflection)

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 5 // 6, wave_field.ny * 5 // 6, wave_field.nz // 2
    wc_radius = 16  # radius in voxels
    wc_radius_sq = wc_radius**2  # radius² = 8² voxels (≈ λ/4 for λ=30dx)
    wc_radius_inner_sq = (wc_radius - 1) ** 2

    for i, j, k in ti.ndrange(
        (wc1x - wc_radius - 1, wc1x + wc_radius + 2),
        (wc1y - wc_radius - 1, wc1y + wc_radius + 2),
        (wc1z - wc_radius - 1, wc1z + wc_radius + 2),
    ):
        dist_sq = (i - wc1x) ** 2 + (j - wc1y) ** 2 + (k - wc1z) ** 2
        # Only process voxels on inner surface of sphere (r = radius-1)
        if wc_radius_inner_sq <= dist_sq <= wc_radius_sq:
            # Find direction toward center and copy from outer neighbor
            di = ti.cast(ti.math.sign(wc1x - i), ti.i32)
            dj = ti.cast(ti.math.sign(wc1y - j), ti.i32)
            dk = ti.cast(ti.math.sign(wc1z - k), ti.i32)
            # Copy from the voxel just outside the sphere (Neumann: ∂ψ/∂n = 0)
            outer_val = wave_field.psiL_am[i - di, j - dj, k - dk]
            wave_field.psiL_old_am[i, j, k] = outer_val
            wave_field.psiL_am[i, j, k] = outer_val
            wave_field.psiL_new_am[i, j, k] = outer_val


@ti.func
def interact_wc_dirichlet(wave_field: ti.template()):  # type: ignore
    """
    TEST: Dirichlet boundary condition on wave center sphere surface
    ψ = 0 →
    This reflects waves WITH phase inversion (hard/fixed-end reflection)

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 5 // 6, wave_field.ny * 5 // 6, wave_field.nz // 2
    wc2x, wc2y, wc2z = wave_field.nx * 4 // 6, wave_field.ny * 4 // 6, wave_field.nz // 2
    wc_radius = 16  # radius in voxels
    wc_radius_sq = wc_radius**2  # radius² = 8² voxels (≈ λ/4 for λ=30dx)

    for i, j, k in ti.ndrange(
        (wc1x - wc_radius - 1, wc1x + wc_radius + 2),
        (wc1y - wc_radius - 1, wc1y + wc_radius + 2),
        (wc1z - wc_radius - 1, wc1z + wc_radius + 2),
    ):
        dist_sq = (i - wc1x) ** 2 + (j - wc1y) ** 2 + (k - wc1z) ** 2
        # Only process voxels on inner surface of sphere (r = radius-1)
        if dist_sq <= wc_radius_sq:
            wave_field.psiL_old_am[i, j, k] = 0.0
            wave_field.psiL_am[i, j, k] = 0.0
            wave_field.psiL_new_am[i, j, k] = 0.0

    for i, j, k in ti.ndrange(
        (wc2x - wc_radius - 1, wc2x + wc_radius + 2),
        (wc2y - wc_radius - 1, wc2y + wc_radius + 2),
        (wc2z - wc_radius - 1, wc2z + wc_radius + 2),
    ):
        dist_sq = (i - wc2x) ** 2 + (j - wc2y) ** 2 + (k - wc2z) ** 2
        # Only process voxels on inner surface of sphere (r = radius-1)
        if dist_sq <= wc_radius_sq:
            wave_field.psiL_old_am[i, j, k] = 0.0
            wave_field.psiL_am[i, j, k] = 0.0
            wave_field.psiL_new_am[i, j, k] = 0.0


@ti.func
def interact_wc_signal(wave_field: ti.template()):  # type: ignore
    """
    TEST: Dirichlet boundary condition on wave center sphere surface
    ψ = 0 →
    This reflects waves WITH phase inversion (hard/fixed-end reflection)

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 5 // 6, wave_field.ny * 5 // 6, wave_field.nz // 2
    wc_radius = 1  # radius in voxels
    wc_radius_sq = wc_radius**2  # radius² = 8² voxels (≈ λ/4 for λ=30dx)

    for i, j, k in ti.ndrange(
        (wc1x - wc_radius - 1, wc1x + wc_radius + 2),
        (wc1y - wc_radius - 1, wc1y + wc_radius + 2),
        (wc1z - wc_radius - 1, wc1z + wc_radius + 2),
    ):
        dist_sq = (i - wc1x) ** 2 + (j - wc1y) ** 2 + (k - wc1z) ** 2
        # Only process voxels on inner surface of sphere (r = radius-1)
        if dist_sq <= wc_radius_sq:
            wave_field.psiL_old_am[i, j, k] = -wave_field.psiL_old_am[i, j, k]
            wave_field.psiL_am[i, j, k] = -wave_field.psiL_am[i, j, k]
            wave_field.psiL_new_am[i, j, k] = -wave_field.psiL_new_am[i, j, k]


# ================================================================
# 3-PLANE SAMPLING FOR AVERAGE TRACKERS
# ================================================================
# PERFORMANCE NOTE: Full GPU reduction (atomic_add over all voxels) causes
# severe performance issues due to atomic contention with millions of voxels.
# The 3-plane sampling is a deliberate compromise:
# - Samples ~3N² voxels instead of N³ (e.g., 3% for 100³ grid)
# - Assumes isotropic field distribution (valid for most wave scenarios)
# - Acceptable accuracy vs massive performance gain
# ================================================================

# Cached slice buffers (initialized on first call)
_slice_xy_amp = None
_slice_xy_freq = None
_slice_xz_amp = None
_slice_xz_freq = None
_slice_yz_amp = None
_slice_yz_freq = None


@ti.kernel
def _copy_slice_xy(
    trackers: ti.template(),  # type: ignore
    slice_amp: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    mid_z: ti.i32,  # type: ignore
):
    """Copy center XY slice (fixed z) to 2D buffer."""
    for i, j in slice_amp:
        slice_amp[i, j] = trackers.ampL_am[i, j, mid_z]
        slice_freq[i, j] = trackers.freq_rHz[i, j, mid_z]


@ti.kernel
def _copy_slice_xz(
    trackers: ti.template(),  # type: ignore
    slice_amp: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    mid_y: ti.i32,  # type: ignore
):
    """Copy center XZ slice (fixed y) to 2D buffer."""
    for i, k in slice_amp:
        slice_amp[i, k] = trackers.ampL_am[i, mid_y, k]
        slice_freq[i, k] = trackers.freq_rHz[i, mid_y, k]


@ti.kernel
def _copy_slice_yz(
    trackers: ti.template(),  # type: ignore
    slice_amp: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    mid_x: ti.i32,  # type: ignore
):
    """Copy center YZ slice (fixed x) to 2D buffer."""
    for j, k in slice_amp:
        slice_amp[j, k] = trackers.ampL_am[mid_x, j, k]
        slice_freq[j, k] = trackers.freq_rHz[mid_x, j, k]


def sample_avg_trackers(
    wave_field,
    trackers,
):
    """
    Estimate RMS amplitude and average frequency by sampling 3 orthogonal planes.

    Samples XY, XZ, and YZ center slices to avoid full 3D reduction.
    This is a deliberate performance compromise - full GPU reduction with
    atomic operations causes severe contention with millions of voxels.

    For isotropic fields, center-plane sampling provides representative estimates.

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

    # Transfer 2D slices to CPU for numpy operations
    # Exclude boundary voxels
    xy_amp = _slice_xy_amp.to_numpy()[1:-1, 1:-1]
    xy_freq = _slice_xy_freq.to_numpy()[1:-1, 1:-1]
    xz_amp = _slice_xz_amp.to_numpy()[1:-1, 1:-1]
    xz_freq = _slice_xz_freq.to_numpy()[1:-1, 1:-1]
    yz_amp = _slice_yz_amp.to_numpy()[1:-1, 1:-1]
    yz_freq = _slice_yz_freq.to_numpy()[1:-1, 1:-1]

    # Compute RMS amplitude: √(⟨A²⟩) for correct energy weighting
    # ampL_am contains per-voxel RMS values, square them for energy
    total_amp_squared = (xy_amp**2).sum() + (xz_amp**2).sum() + (yz_amp**2).sum()
    total_freq = xy_freq.sum() + xz_freq.sum() + yz_freq.sum()
    n_samples = xy_amp.size + xz_amp.size + yz_amp.size

    trackers.rms_ampL_am[None] = float(np.sqrt(total_amp_squared / n_samples))
    trackers.avg_freq_rHz[None] = float(total_freq / n_samples)


# ================================================================
# FLUX MESH COLOR UPDATING
# ================================================================


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

    # ================================================================
    # XY Plane: Sample at z = fm_plane_z_idx
    # ================================================================
    # Always update all planes (conditionals cause GPU branch divergence)
    for i, j in ti.ndrange(wave_field.nx, wave_field.ny):
        # Sample longitudinal displacement at this voxel
        psi_value = wave_field.psiL_am[i, j, wave_field.fm_plane_z_idx]
        amp_value = trackers.ampL_am[i, j, wave_field.fm_plane_z_idx]
        freq_value = trackers.freq_rHz[i, j, wave_field.fm_plane_z_idx]

        # Map value to color using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_freq_rHz[None] * 2
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_ironbow_color(
                amp_value, 0, trackers.rms_ampL_am[None] * 2
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_redshift_color(
                psi_value,
                -trackers.rms_ampL_am[None] * 2,
                trackers.rms_ampL_am[None] * 2,
            )

    # ================================================================
    # XZ Plane: Sample at y = fm_plane_y_idx
    # ================================================================
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        psi_value = wave_field.psiL_am[i, wave_field.fm_plane_y_idx, k]
        amp_value = trackers.ampL_am[i, wave_field.fm_plane_y_idx, k]
        freq_value = trackers.freq_rHz[i, wave_field.fm_plane_y_idx, k]

        # Map value to color using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_freq_rHz[None] * 2
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_ironbow_color(
                amp_value, 0, trackers.rms_ampL_am[None] * 2
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_redshift_color(
                psi_value,
                -trackers.rms_ampL_am[None] * 2,
                trackers.rms_ampL_am[None] * 2,
            )

    # ================================================================
    # YZ Plane: Sample at x = fm_plane_x_idx
    # ================================================================
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        psi_value = wave_field.psiL_am[wave_field.fm_plane_x_idx, j, k]
        amp_value = trackers.ampL_am[wave_field.fm_plane_x_idx, j, k]
        freq_value = trackers.freq_rHz[wave_field.fm_plane_x_idx, j, k]

        # Map value to color using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_freq_rHz[None] * 2
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_ironbow_color(
                amp_value, 0, trackers.rms_ampL_am[None] * 2
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_redshift_color(
                psi_value,
                -trackers.rms_ampL_am[None] * 2,
                trackers.rms_ampL_am[None] * 2,
            )
