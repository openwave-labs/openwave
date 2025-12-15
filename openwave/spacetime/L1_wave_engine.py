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
# STATIC CHARGING methods (single radial pulse pattern)
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

        wave_field.displacement_am[i, j, k] = A_am * gaussian

    # Set old displacement equal to current (zero initial velocity: ∂ψ/∂t = 0)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        wave_field.displacement_old_am[i, j, k] = wave_field.displacement_am[i, j, k]


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
        wave_field.displacement_am[i + 1, j, k]
        + wave_field.displacement_am[i - 1, j, k]
        + wave_field.displacement_am[i, j + 1, k]
        + wave_field.displacement_am[i, j - 1, k]
        + wave_field.displacement_am[i, j, k + 1]
        + wave_field.displacement_am[i, j, k - 1]
    )

    center = wave_field.displacement_am[i, j, k]
    laplacian6_am = (face_sum - 6.0 * center) / (wave_field.dx_am**2)

    return laplacian6_am


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

    Discrete Form (Leap-Frog/Verlet):
        ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
        where ∇²ψ = (neighbors_sum - 6·center) / dx²

    Boundary Conditions:
    - Dirichlet BC (ψ = 0) at edges for fixed boundaries.
    Implemented by skipping boundary voxels in update loop.

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
    # WAVE PROPAGATION: Update voxels using Leap-Frog
    # ================================================================
    # Update interior voxels only (Dirichlet BC: ψ=0 at edges)
    # 6-point Laplacian: needs 1-cell buffer → range (1, n-1)
    for i, j, k in ti.ndrange((1, nx - 1), (1, ny - 1), (1, nz - 1)):  # 6-pt
        # Compute spatial Laplacian ∇²ψ (toggle between 6-pt and 18-pt)
        laplacian_am = compute_laplacian6(wave_field, i, j, k)

        # Leap-Frog update
        # Standard form: ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
        wave_field.displacement_new_am[i, j, k] = (
            2.0 * wave_field.displacement_am[i, j, k]
            - wave_field.displacement_old_am[i, j, k]
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
        disp_squared = wave_field.displacement_am[i, j, k] ** 2
        current_rms_squared = trackers.amplitudeL_am[i, j, k] ** 2
        alpha_rms = 0.005  # EMA smoothing factor for RMS tracking
        new_rms_squared = alpha_rms * disp_squared + (1.0 - alpha_rms) * current_rms_squared
        trackers.amplitudeL_am[i, j, k] = ti.sqrt(new_rms_squared)

        # FREQUENCY tracking, via zero-crossing detection with EMA smoothing
        # Detect positive-going zero crossing (negative → positive transition)
        # Period = time between consecutive positive zero crossings
        # More robust than peak detection since it's amplitude-independent
        # EMA smoothing: f_new = α * f_measured + (1 - α) * f_old
        # α controls adaptation speed: higher = faster response, lower = smoother
        prev_disp = wave_field.displacement_old_am[i, j, k]
        curr_disp = wave_field.displacement_am[i, j, k]
        if prev_disp < 0.0 and curr_disp >= 0.0:  # Zero crossing detected
            period_rs = elapsed_t_rs - trackers.last_crossing[i, j, k]
            if period_rs > dt_rs * 2:  # Filter out spurious crossings
                measured_freq = 1.0 / period_rs  # in rHz
                current_freq = trackers.frequency_rHz[i, j, k]
                alpha_freq = 0.05  # EMA smoothing factor for frequency
                trackers.frequency_rHz[i, j, k] = (
                    alpha_freq * measured_freq + (1.0 - alpha_freq) * current_freq
                )
            trackers.last_crossing[i, j, k] = elapsed_t_rs

    # Swap time levels: old ← current, current ← new
    for i, j, k in ti.ndrange(nx, ny, nz):
        wave_field.displacement_old_am[i, j, k] = wave_field.displacement_am[i, j, k]
        wave_field.displacement_am[i, j, k] = wave_field.displacement_new_am[i, j, k]

    # TODO: Testing Wave Center Interaction with Energy Waves
    # WCs modify Energy Wave character (amplitude/phase/lambda/mode) as they pass through
    # Standing Waves should form around WCs as visual artifacts of interaction
    # Energy Waves are Isotropic (omnidirectional) so reflection gets canceled out

    # interact_wc_swap(wave_field)
    # interact_wc_pulse(wave_field, elapsed_t_rs)  # forces amp, but no standing wave formation
    # interact_wc_lens(wave_field)  # amplify waves at WC, but no standing wave formation

    # interact_wc_min(wave_field) # no interaction on isotropic waves
    # interact_wc_drain(wave_field) # no interaction on isotropic waves
    # interact_wc_newmann(wave_field) # no interaction on isotropic waves
    # interact_wc_dirichlet(wave_field) # no interaction on isotropic waves
    # interact_wc_signal(wave_field) # just invert oscillation, no wave interaction


# ================================================================
# WAVE CENTER INTERACTIONS
# ================================================================


@ti.func
def interact_wc_swap(wave_field: ti.template()):  # type: ignore
    """
    TEST: Wave center swaps neighbor displacements.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 4 // 6, wave_field.ny * 4 // 6, wave_field.nz // 2

    # Swap displacement at X axis
    left = wave_field.displacement_am[wc1x + 1, wc1y, wc1z]
    right = wave_field.displacement_am[wc1x - 1, wc1y, wc1z]
    wave_field.displacement_am[wc1x + 1, wc1y, wc1z] = right
    wave_field.displacement_am[wc1x - 1, wc1y, wc1z] = left

    # Swap displacement at Y axis
    front = wave_field.displacement_am[wc1x, wc1y + 1, wc1z]
    back = wave_field.displacement_am[wc1x, wc1y - 1, wc1z]
    wave_field.displacement_am[wc1x, wc1y + 1, wc1z] = back
    wave_field.displacement_am[wc1x, wc1y - 1, wc1z] = front

    # Swap displacement at Z axis
    top = wave_field.displacement_am[wc1x, wc1y, wc1z + 1]
    bottom = wave_field.displacement_am[wc1x, wc1y, wc1z - 1]
    wave_field.displacement_am[wc1x, wc1y, wc1z + 1] = bottom
    wave_field.displacement_am[wc1x, wc1y, wc1z - 1] = top


@ti.func
def interact_wc_pulse(wave_field: ti.template(), elapsed_t_rs):  # type: ignore
    """
    TEST: Wave center as PULSE - injects oscillation at WC sphere surface

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 4 // 6, wave_field.ny * 4 // 6, wave_field.nz // 2
    wc_radius = 0  # radius in voxels
    wc_radius_sq = wc_radius**2  # radius² = 8² voxels (≈ λ/4 for λ=30dx)
    boost = 2.0  # amplitude boost factor

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = (
        (1 / boost) * 2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor
    )  # angular frequency (rad/rs)

    for i, j, k in ti.ndrange(
        (wc1x - wc_radius - 1, wc1x + wc_radius + 2),
        (wc1y - wc_radius - 1, wc1y + wc_radius + 2),
        (wc1z - wc_radius - 1, wc1z + wc_radius + 2),
    ):
        dist_sq = (i - wc1x) ** 2 + (j - wc1y) ** 2 + (k - wc1z) ** 2
        # Only process voxels on inner surface of sphere (r = radius-1)
        if dist_sq <= wc_radius_sq:
            wave_field.displacement_am[i, j, k] = (
                base_amplitude_am
                * boost
                * wave_field.scale_factor
                * ti.cos(omega_rs * elapsed_t_rs)
            )


@ti.func
def interact_wc_lens(wave_field: ti.template()):  # type: ignore
    """
    TEST: Wave center as LENS - amplitude increase
    Absorbs wave energy at the wave center (WC) sphere surface

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 4 // 6, wave_field.ny * 4 // 6, wave_field.nz // 2
    wc_radius = 1  # radius in voxels
    wc_radius_sq = wc_radius**2  # radius² = 8² voxels (≈ λ/4 for λ=30dx)
    l = 2  # lens amplification factor

    for i, j, k in ti.ndrange(
        (wc1x - wc_radius - 1, wc1x + wc_radius + 2),
        (wc1y - wc_radius - 1, wc1y + wc_radius + 2),
        (wc1z - wc_radius - 1, wc1z + wc_radius + 2),
    ):
        dist_sq = (i - wc1x) ** 2 + (j - wc1y) ** 2 + (k - wc1z) ** 2
        # Only process voxels on inner surface of sphere (r = radius-1)
        if dist_sq <= wc_radius_sq:
            wave_field.displacement_am[i, j, k] = l * wave_field.displacement_am[i, j, k]


@ti.func
def interact_wc_min(wave_field: ti.template()):  # type: ignore
    """
    TEST: Wave center as amplification
    Amplifies/focuses waves rather than blocking
    Allow normal wave propagation but clamp the minimum amplitude at WC
    WC is a point-like region where wave amplitude is amplified
    This creates a phase singularity that surrounding waves conform to
    If |ψ| < threshold, boost it to threshold (preserving sign)
    This creates an amplitude floor, not ceiling — no runaway growth

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 4 // 6, wave_field.ny * 4 // 6, wave_field.nz // 2
    amplification = 2.0  # how much the WC amplifies local wave amplitude

    # Reference amplitude from wave field (the base amplitude used for charging)
    ref_amplitude = base_amplitude_am * wave_field.scale_factor
    min_amplitude = ref_amplitude * amplification

    # If amplitude is below minimum, boost it (preserve sign/phase)
    current_val = wave_field.displacement_am[wc1x, wc1y, wc1z]
    if ti.abs(current_val) < min_amplitude:
        phase_sign = 1.0 if current_val >= 0.0 else -1.0
        wave_field.displacement_am[wc1x, wc1y, wc1z] = phase_sign * min_amplitude


@ti.func
def interact_wc_drain(wave_field: ti.template()):  # type: ignore
    """
    TEST: Wave center as DRAIN - amplitude reduction
    Absorbs wave energy at the wave center (WC) sphere surface

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 4 // 6, wave_field.ny * 4 // 6, wave_field.nz // 2
    wc_radius = 8  # radius in voxels
    wc_radius_sq = wc_radius**2  # radius² = 8² voxels (≈ λ/4 for λ=30dx)

    for i, j, k in ti.ndrange(
        (wc1x - wc_radius - 1, wc1x + wc_radius + 2),
        (wc1y - wc_radius - 1, wc1y + wc_radius + 2),
        (wc1z - wc_radius - 1, wc1z + wc_radius + 2),
    ):
        dist_sq = (i - wc1x) ** 2 + (j - wc1y) ** 2 + (k - wc1z) ** 2
        # Only process voxels on inner surface of sphere (r = radius-1)
        if dist_sq <= wc_radius_sq:
            wave_field.displacement_am[i, j, k] = 0.5 * wave_field.displacement_am[i, j, k]


@ti.func
def interact_wc_newmann(wave_field: ti.template()):  # type: ignore
    """
    TEST: Neumann boundary condition on wave center sphere surface
    ∂ψ/∂n = 0 → copy outer values to inner surface (zero gradient)
    This reflects waves WITHOUT phase inversion (soft/free-end reflection)

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 4 // 6, wave_field.ny * 4 // 6, wave_field.nz // 2
    wc_radius = 1  # radius in voxels
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
            outer_val = wave_field.displacement_am[i - di, j - dj, k - dk]
            wave_field.displacement_am[i, j, k] = outer_val


@ti.func
def interact_wc_dirichlet(wave_field: ti.template()):  # type: ignore
    """
    TEST: Dirichlet boundary condition on wave center sphere surface
    ψ = 0 →
    This reflects waves WITH phase inversion (hard/fixed-end reflection)

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 4 // 6, wave_field.ny * 4 // 6, wave_field.nz // 2
    wc_radius = 8  # radius in voxels
    wc_radius_sq = wc_radius**2  # radius² = 8² voxels (≈ λ/4 for λ=30dx)

    for i, j, k in ti.ndrange(
        (wc1x - wc_radius - 1, wc1x + wc_radius + 2),
        (wc1y - wc_radius - 1, wc1y + wc_radius + 2),
        (wc1z - wc_radius - 1, wc1z + wc_radius + 2),
    ):
        dist_sq = (i - wc1x) ** 2 + (j - wc1y) ** 2 + (k - wc1z) ** 2
        # Only process voxels on inner surface of sphere (r = radius-1)
        if dist_sq <= wc_radius_sq:
            wave_field.displacement_old_am[i, j, k] = 0.0
            wave_field.displacement_am[i, j, k] = 0.0
            wave_field.displacement_new_am[i, j, k] = 0.0


@ti.func
def interact_wc_signal(wave_field: ti.template()):  # type: ignore
    """
    TEST: Wave center as SIGNAL INVERTER - phase inversion at WC sphere surface
    Inverts wave phase at the wave center (WC) sphere surface

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
    """
    wc1x, wc1y, wc1z = wave_field.nx * 4 // 6, wave_field.ny * 4 // 6, wave_field.nz // 2
    wc_radius = 0  # radius in voxels
    wc_radius_sq = wc_radius**2  # radius² = 8² voxels (≈ λ/4 for λ=30dx)

    for i, j, k in ti.ndrange(
        (wc1x - wc_radius - 1, wc1x + wc_radius + 2),
        (wc1y - wc_radius - 1, wc1y + wc_radius + 2),
        (wc1z - wc_radius - 1, wc1z + wc_radius + 2),
    ):
        dist_sq = (i - wc1x) ** 2 + (j - wc1y) ** 2 + (k - wc1z) ** 2
        # Only process voxels on inner surface of sphere (r = radius-1)
        if dist_sq <= wc_radius_sq:
            wave_field.displacement_am[i, j, k] = -wave_field.displacement_am[i, j, k]


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
    # amplitudeL_am contains per-voxel RMS values, square them for energy
    total_amp_squared = (xy_amp**2).sum() + (xz_amp**2).sum() + (yz_amp**2).sum()
    total_freq = xy_freq.sum() + xz_freq.sum() + yz_freq.sum()
    n_samples = xy_amp.size + xz_amp.size + yz_amp.size

    trackers.rms_amplitudeL_am[None] = float(np.sqrt(total_amp_squared / n_samples))
    trackers.avg_frequency_rHz[None] = float(total_freq / n_samples)


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
        disp_value = wave_field.displacement_am[i, j, wave_field.fm_plane_z_idx]
        amp_value = trackers.amplitudeL_am[i, j, wave_field.fm_plane_z_idx]
        freq_value = trackers.frequency_rHz[i, j, wave_field.fm_plane_z_idx]

        # Map value to color using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_frequency_rHz[None] * 2
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_ironbow_color(
                amp_value, 0, trackers.rms_amplitudeL_am[None] * 2
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_redshift_color(
                disp_value,
                -trackers.rms_amplitudeL_am[None] * 2,
                trackers.rms_amplitudeL_am[None] * 2,
            )

    # ================================================================
    # XZ Plane: Sample at y = fm_plane_y_idx
    # ================================================================
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[i, wave_field.fm_plane_y_idx, k]
        amp_value = trackers.amplitudeL_am[i, wave_field.fm_plane_y_idx, k]
        freq_value = trackers.frequency_rHz[i, wave_field.fm_plane_y_idx, k]

        # Map value to color using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_frequency_rHz[None] * 2
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_ironbow_color(
                amp_value, 0, trackers.rms_amplitudeL_am[None] * 2
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_redshift_color(
                disp_value,
                -trackers.rms_amplitudeL_am[None] * 2,
                trackers.rms_amplitudeL_am[None] * 2,
            )

    # ================================================================
    # YZ Plane: Sample at x = fm_plane_x_idx
    # ================================================================
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[wave_field.fm_plane_x_idx, j, k]
        amp_value = trackers.amplitudeL_am[wave_field.fm_plane_x_idx, j, k]
        freq_value = trackers.frequency_rHz[wave_field.fm_plane_x_idx, j, k]

        # Map value to color using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_frequency_rHz[None] * 2
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_ironbow_color(
                amp_value, 0, trackers.rms_amplitudeL_am[None] * 2
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_redshift_color(
                disp_value,
                -trackers.rms_amplitudeL_am[None] * 2,
                trackers.rms_amplitudeL_am[None] * 2,
            )
