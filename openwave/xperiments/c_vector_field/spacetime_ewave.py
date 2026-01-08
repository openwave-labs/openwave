"""
ENERGY-WAVE ENGINE

ON SCALAR-FIELD METHOD

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
base_wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETER  # in attometers
base_frequency = constants.EWAVE_FREQUENCY  # in Hz
base_frequency_rHz = constants.EWAVE_FREQUENCY * constants.RONTOSECOND  # in rHz (1/rontosecond)
rho = constants.MEDIUM_DENSITY  # medium density for Gaussian energy calc (kg/m³)


# ================================================================
# WAVE PROPAGATION ENGINE
# ================================================================


@ti.kernel
def propagate_wave(
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
    dt_rs: ti.f32,  # type: ignore
    elapsed_t_rs: ti.f32,  # type: ignore
    boost: ti.f32,  # type: ignore
    sim_speed: ti.f32,  # type: ignore
):
    """
    Propagate wave displacement using wave equation (PDE Solver).
    Wave Equation: ∂²ψ/∂t² = c²∇²ψ

    Includes wave propagation, reflection at boundaries & superposition of wavefronts.
    Energy conservation from leap-frog method.

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
    """
    # Grid dimensions for boundary handling
    nx, ny, nz = wave_field.nx, wave_field.ny, wave_field.nz

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = (
        2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor * sim_speed
    )  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength * wave_field.scale_factor / wave_field.dx
    k_grid = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Temporal phase: φ = ω·t, oscillatory in time
    temporal_phase = omega_rs * elapsed_t_rs

    # ================================================================
    # WAVE PROPAGATION: Update voxels using wave functions
    # ================================================================
    # Update all voxels
    for i, j, k in ti.ndrange(nx, ny, nz):
        prev_disp = wave_field.psiL_am[i, j, k]
        wave_field.psiL_am[i, j, k] = 0.0  # reset before accumulation

        # loop over wave-centers (wave superposition principle)
        for wc_idx in ti.ndrange(wave_center.num_sources):
            # Compute radial distance from wave source (in grid indices)
            r_grid = ti.sqrt(
                (i - wave_center.position_grid[wc_idx][0]) ** 2
                + (j - wave_center.position_grid[wc_idx][1]) ** 2
                + (k - wave_center.position_grid[wc_idx][2]) ** 2
            )

            # Cache source-specific phase offset
            source_offset = wave_center.offset[wc_idx]
            # Spatial phase: φ = k·r, creates spherical wave fronts, dimensionless, in radians
            spatial_phase = k_grid * r_grid

            # Combined and Adjusted Wolff-LaFreniere wave-equation form:
            # Expanded form: -cos(ωt)·sin(kr)/r - sin(ωt)·(1-cos(kr))/r
            # Phase term: sin(kr)/r → k as r→0 (physical units)
            phase_term = ti.select(
                r_grid < 0.5,  # threshold in grid units (catches center voxel only)
                k_grid,  # analytical limit
                ti.sin(spatial_phase) / r_grid,
            )

            # Quadrature term: (1-cos(kr))/r → 0 as r→0
            quadrature_term = ti.select(
                r_grid < 0.5,  # threshold in grid units (catches center voxel only)
                0.0,  # analytical limit
                (1 - ti.cos(spatial_phase)) / r_grid,
            )

            oscillator = (
                -ti.cos(temporal_phase + source_offset) * phase_term
                - ti.sin(temporal_phase + source_offset) * quadrature_term
            )

            wave_field.psiL_am[i, j, k] += (
                base_amplitude_am * boost * wave_field.scale_factor * oscillator
            )

        curr_disp = wave_field.psiL_am[i, j, k]

        # # Amplitude falloff for spherical wave: A(r) = A₀/r
        # # Clamp to r_min to avoid singularity at r = 0
        # r_safe_am = ti.max(r_grid, wavelength_grid)
        # amplitude_falloff = wavelength_grid / r_safe_am
        # # Total amplitude at this distance (with visualization scaling)
        # amplitude_at_r_am = base_amplitude_am * amplitude_falloff

        # # Apply spherical wave oscillating displacements
        # # Standing Wave: A·cos(ωt)·cos(kr)
        # prev_disp = wave_field.psiL_am[i, j, k]
        # wave_field.psiL_am[i, j, k] = (
        #     base_amplitude_am
        #     * boost
        #     * wave_field.scale_factor
        #     * ti.cos(temporal_phase)
        #     * ti.sin(spatial_phase)
        # )
        # curr_disp = wave_field.psiL_am[i, j, k]

        # # Traveling Wave: A(r)·cos(ωt-kr), positive = expansion, negative = compression
        # wave_field.psiL_am[i, j, k] += (
        #     amplitude_at_r_am
        #     * boost
        #     * wave_field.scale_factor
        #     * ti.cos(temporal_phase - spatial_phase)
        # )

        # WAVE-TRACKERS ============================================
        # RMS AMPLITUDE tracking via EMA on ψ² (squared displacement)
        # Running RMS: tracks √⟨ψ²⟩ - the energy-equivalent amplitude (Energy ∝ ψ²)
        # Used for: energy calculation, force gradients, visualization scaling
        # Physics: particles respond to time-averaged energy density, not
        # instantaneous displacement (inertia acts as low-pass filter at ~10²⁵ Hz)
        # EMA on ψ²: rms² = α * ψ² + (1 - α) * rms²_old, then rms = √(rms²)
        # α controls adaptation speed: higher = faster response, lower = smoother
        # 2 polarities tracked: longitudinal & transverse
        # Longitudinal RMS amplitude
        disp2_L = wave_field.psiL_am[i, j, k] ** 2
        current_rms2_L = trackers.ampL_am[i, j, k] ** 2
        alpha_rms_L = 0.05  # EMA smoothing factor for RMS tracking
        new_rms2_L = alpha_rms_L * disp2_L + (1.0 - alpha_rms_L) * current_rms2_L
        new_ampL = ti.sqrt(new_rms2_L)
        # Unconditional decay clears trails from moving sources
        # Active regions counteract decay via EMA update from strong displacement
        # Stale regions (waves propagated away) decay to zero over time
        decay_factor = ti.cast(0.99, ti.f32)  # ~100 frames to ~37%, ~230 to ~10%
        trackers.ampL_am[i, j, k] = new_ampL * decay_factor

        # Transverse RMS amplitude
        disp2_T = wave_field.psiT_am[i, j, k] ** 2
        current_rms2_T = trackers.ampT_am[i, j, k] ** 2
        alpha_rms_T = 0.05  # EMA smoothing factor for RMS tracking
        new_rms2_T = alpha_rms_T * disp2_T + (1.0 - alpha_rms_T) * current_rms2_T
        new_ampT = ti.sqrt(new_rms2_T)
        trackers.ampT_am[i, j, k] = new_ampT * decay_factor

        # TODO: review new frequency tracking method
        # FREQUENCY tracking, via zero-crossing detection with EMA smoothing
        # Detect positive-going zero crossing (negative → positive transition)
        # Period = time between consecutive positive zero crossings
        # More robust than peak detection since it's amplitude-independent
        # EMA smoothing: f_new = α * f_measured + (1 - α) * f_old
        # α controls adaptation speed: higher = faster response, lower = smoother
        if prev_disp < 0.0 and curr_disp >= 0.0:  # Zero crossing detected
            period_rs = elapsed_t_rs - trackers.last_crossing[i, j, k]
            if period_rs > dt_rs * 2:  # Filter out spurious crossings
                measured_freq = 1.0 / period_rs  # in rHz
                current_freq = trackers.freq_rHz[i, j, k]
                alpha_freq = 0.5  # EMA smoothing factor for frequency
                trackers.freq_rHz[i, j, k] = (
                    alpha_freq * measured_freq + (1.0 - alpha_freq) * current_freq
                )
            trackers.last_crossing[i, j, k] = elapsed_t_rs

        # Unconditional frequency decay (counteracted by zero-crossing updates in active regions)
        trackers.freq_rHz[i, j, k] *= decay_factor

    # TODO: Testing Wave Center Interaction with Energy Waves
    # WCs modify Energy Wave character (amplitude/phase/lambda/mode) as they pass through
    # Standing Waves should form around WCs as visual artifacts of interaction

    # interact_wc_spinUP(wave_field, dt_rs)  # never worked correctly
    # interact_wc_spinDOWN(wave_field, dt_rs)  # never worked correctly


# ================================================================
# WAVE CENTER INTERACTIONS
# ================================================================


@ti.func
def interact_wc_spinUP(wave_field: ti.template(), dt_rs: ti.f32):  # type: ignore
    """
    Wave center spin-UP interaction: phase-shifts psiL by +90° and creates psiT.

    WAVE CENTER SPIN MECHANISM:
    1. Incoming longitudinal wave psiL contacts wave center
    2. WC spin creates transverse component: psiT = α × psiL (fine structure ratio)
    3. Outgoing psiL is PHASE-SHIFTED by +90° (counterclockwise/leading)

    PHASE SHIFT VIA VELOCITY:
    For sinusoidal wave psiL = A·cos(ωt):
        - Velocity: ∂psiL/∂t = -A·ω·sin(ωt)
        - Normalized: velocity/ω = -A·sin(ωt) = A·cos(ωt + 90°)  ← +90° shifted!

    So (psiL - psiL_old)/(ω·dt) gives the +90° phase-shifted wave.
    This creates a DISTURBANCE in the longitudinal wave from the spin interaction.

    ENERGY CONSERVATION:
        psiT = α × psiL          (transverse component created)
        psiL_out² + psiT² = psiL_in²  (total energy conserved)
        psiL_out = ±√(psiL² - psiT²)  with sign from phase-shifted wave

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        dt_rs: Time step size (rs) for velocity calculation
    """
    wc1x, wc1y, wc1z = wave_field.nx * 4 // 6, wave_field.ny * 4 // 6, wave_field.nz // 2
    alpha = constants.FINE_STRUCTURE  # L→T conversion ratio

    # Angular frequency (scaled for simulation)
    omega_rs = 2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor

    # Current and previous longitudinal displacement at WC
    psiL = wave_field.psiL_am[wc1x, wc1y, wc1z]
    psiL_old = wave_field.psiL_old_am[wc1x, wc1y, wc1z]

    # ================================================================
    # STEP 1: Compute phase-shifted psiL (+90° leading)
    # ================================================================
    # Velocity via finite difference
    delta_psiL = psiL - psiL_old

    # Phase-shifted psiL: -velocity/ω = -delta_psiL / (ω·dt)
    # delta_psiL ≈ -ω·dt·sin(ωt) for cos input, so:
    # psiL_shifted = -(-ω·dt·sin)/(ω·dt) = +sin(ωt)
    # This gives +90° shift: cos → sin
    psiL_shifted = -delta_psiL / (omega_rs * dt_rs)

    # ================================================================
    # STEP 2: Create transverse component (90° from psiL_out)
    # ================================================================
    # We've tried:
    #   psiT = alpha * psiL → gives 180°
    #   psiT = alpha * psiL_shifted → gives 0°
    # For 90°, try negating one: psiT = -alpha * psiL
    psiT = -alpha * psiL  # NEGATED to flip 180° → hopefully 90°

    # Safety clamp: ensure psiT² < psiL² to prevent NaN from sqrt
    max_psiT = 0.99 * ti.abs(psiL)
    psiT = ti.math.clamp(psiT, -max_psiT, max_psiT)

    # ================================================================
    # STEP 3: Output psiL as actual phase-shifted wave
    # ================================================================
    # Energy available for psiL_out after psiT extraction
    psiL_energy = psiL**2 - psiT**2

    # Scale psiL_shifted to have correct energy while preserving its phase
    psiL_shifted_sq = psiL_shifted**2

    # Initialize psiL_out (required for Taichi scoping)
    psiL_out = 0.0

    # Avoid division by zero: if psiL_shifted is tiny, fall back to original
    if psiL_shifted_sq > 1e-20:
        scaling = ti.sqrt(psiL_energy / psiL_shifted_sq)
        psiL_out = psiL_shifted * scaling
    else:
        # At zero crossing of shifted wave, preserve sign of original
        phase_sign = 1.0 if psiL >= 0.0 else -1.0
        psiL_out = phase_sign * ti.sqrt(psiL_energy)

    wave_field.psiL_am[wc1x, wc1y, wc1z] = psiL_out
    wave_field.psiT_am[wc1x, wc1y, wc1z] = psiT


@ti.func
def interact_wc_spinDOWN(wave_field: ti.template(), dt_rs: ti.f32):  # type: ignore
    """
    Wave center spin-DOWN interaction: phase-shifts psiL by -90° and creates psiT.

    WAVE CENTER SPIN MECHANISM (opposite direction):
    1. Incoming longitudinal wave psiL contacts wave center
    2. WC spin creates transverse component: psiT = α × psiL (fine structure ratio)
    3. Outgoing psiL is PHASE-SHIFTED by -90° (clockwise/lagging)

    PHASE SHIFT VIA NEGATIVE VELOCITY:
    For sinusoidal wave psiL = A·cos(ωt):
        - Velocity: ∂psiL/∂t = -A·ω·sin(ωt)
        - Negative normalized: -velocity/ω = A·sin(ωt) = A·cos(ωt - 90°)  ← -90° shifted!

    So -(psiL - psiL_old)/(ω·dt) gives the -90° phase-shifted wave.
    This creates a DISTURBANCE in the longitudinal wave (opposite to spinUP).

    ENERGY CONSERVATION:
        psiT = α × psiL          (transverse component created)
        psiL_out² + psiT² = psiL_in²  (total energy conserved)
        psiL_out = ±√(psiL² - psiT²)  with sign from phase-shifted wave

    COMPARISON:
        spinUP:   psiL phase +90° (leading),  counterclockwise
        spinDOWN: psiL phase -90° (lagging),  clockwise

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        dt_rs: Time step size (rs) for velocity calculation
    """
    wc2x, wc2y, wc2z = wave_field.nx * 9 // 12, wave_field.ny * 9 // 12, wave_field.nz // 2
    alpha = constants.FINE_STRUCTURE  # L→T conversion ratio

    # Angular frequency (scaled for simulation)
    omega_rs = 2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor

    # Current and previous longitudinal displacement at WC
    psiL = wave_field.psiL_am[wc2x, wc2y, wc2z]
    psiL_old = wave_field.psiL_old_am[wc2x, wc2y, wc2z]

    # ================================================================
    # STEP 1: Compute phase-shifted psiL (-90° lagging)
    # ================================================================
    # Velocity via finite difference
    delta_psiL = psiL - psiL_old

    # Phase-shifted psiL: -velocity/ω = -delta_psiL / (ω·dt)
    # Transforms A·cos(ωt) → A·sin(ωt) = A·cos(ωt - 90°)
    psiL_shifted = -delta_psiL / (omega_rs * dt_rs)

    # ================================================================
    # STEP 2: Create transverse component (90° from psiL_out)
    # ================================================================
    # Negated to achieve 90° phase relationship
    psiT = -alpha * psiL  # NEGATED

    # Safety clamp: ensure psiT² < psiL² to prevent NaN from sqrt
    max_psiT = 0.99 * ti.abs(psiL)
    psiT = ti.math.clamp(psiT, -max_psiT, max_psiT)

    # ================================================================
    # STEP 3: Output psiL as actual phase-shifted wave
    # ================================================================
    # Energy available for psiL_out after psiT extraction
    psiL_energy = psiL**2 - psiT**2

    # Scale psiL_shifted to have correct energy while preserving its phase
    psiL_shifted_sq = psiL_shifted**2

    # Initialize psiL_out (required for Taichi scoping)
    psiL_out = 0.0

    # Avoid division by zero: if psiL_shifted is tiny, fall back to original
    if psiL_shifted_sq > 1e-20:
        scaling = ti.sqrt(psiL_energy / psiL_shifted_sq)
        psiL_out = psiL_shifted * scaling
    else:
        # At zero crossing of shifted wave, preserve sign of original
        phase_sign = 1.0 if psiL >= 0.0 else -1.0
        psiL_out = phase_sign * ti.sqrt(psiL_energy)

    wave_field.psiL_am[wc2x, wc2y, wc2z] = psiL_out
    wave_field.psiT_am[wc2x, wc2y, wc2z] = psiT


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
_slice_xy_ampL = None
_slice_xy_ampT = None
_slice_xy_freq = None
_slice_xz_ampL = None
_slice_xz_ampT = None
_slice_xz_freq = None
_slice_yz_ampL = None
_slice_yz_ampT = None
_slice_yz_freq = None


@ti.kernel
def _copy_slice_xy(
    trackers: ti.template(),  # type: ignore
    slice_ampL: ti.template(),  # type: ignore
    slice_ampT: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    mid_z: ti.i32,  # type: ignore
):
    """Copy center XY slice (fixed z) to 2D buffer."""
    for i, j in slice_ampL:
        slice_ampL[i, j] = trackers.ampL_am[i, j, mid_z]
        slice_ampT[i, j] = trackers.ampT_am[i, j, mid_z]
        slice_freq[i, j] = trackers.freq_rHz[i, j, mid_z]


@ti.kernel
def _copy_slice_xz(
    trackers: ti.template(),  # type: ignore
    slice_ampL: ti.template(),  # type: ignore
    slice_ampT: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    mid_y: ti.i32,  # type: ignore
):
    """Copy center XZ slice (fixed y) to 2D buffer."""
    for i, k in slice_ampL:
        slice_ampL[i, k] = trackers.ampL_am[i, mid_y, k]
        slice_ampT[i, k] = trackers.ampT_am[i, mid_y, k]
        slice_freq[i, k] = trackers.freq_rHz[i, mid_y, k]


@ti.kernel
def _copy_slice_yz(
    trackers: ti.template(),  # type: ignore
    slice_ampL: ti.template(),  # type: ignore
    slice_ampT: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    mid_x: ti.i32,  # type: ignore
):
    """Copy center YZ slice (fixed x) to 2D buffer."""
    for j, k in slice_ampL:
        slice_ampL[j, k] = trackers.ampL_am[mid_x, j, k]
        slice_ampT[j, k] = trackers.ampT_am[mid_x, j, k]
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
    global _slice_xy_ampL, _slice_xy_ampT, _slice_xy_freq
    global _slice_xz_ampL, _slice_xz_ampT, _slice_xz_freq
    global _slice_yz_ampL, _slice_yz_ampT, _slice_yz_freq

    nx, ny, nz = wave_field.nx, wave_field.ny, wave_field.nz

    # Initialize slice buffers once
    if _slice_xy_ampL is None:
        _slice_xy_ampL = ti.field(dtype=ti.f32, shape=(nx, ny))
        _slice_xy_ampT = ti.field(dtype=ti.f32, shape=(nx, ny))
        _slice_xy_freq = ti.field(dtype=ti.f32, shape=(nx, ny))
        _slice_xz_ampL = ti.field(dtype=ti.f32, shape=(nx, nz))
        _slice_xz_ampT = ti.field(dtype=ti.f32, shape=(nx, nz))
        _slice_xz_freq = ti.field(dtype=ti.f32, shape=(nx, nz))
        _slice_yz_ampL = ti.field(dtype=ti.f32, shape=(ny, nz))
        _slice_yz_ampT = ti.field(dtype=ti.f32, shape=(ny, nz))
        _slice_yz_freq = ti.field(dtype=ti.f32, shape=(ny, nz))

    # Copy 3 center slices to 2D buffers (parallel kernels)
    mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2
    _copy_slice_xy(trackers, _slice_xy_ampL, _slice_xy_ampT, _slice_xy_freq, mid_z)
    _copy_slice_xz(trackers, _slice_xz_ampL, _slice_xz_ampT, _slice_xz_freq, mid_y)
    _copy_slice_yz(trackers, _slice_yz_ampL, _slice_yz_ampT, _slice_yz_freq, mid_x)

    # Transfer 2D slices to CPU for numpy operations
    # Exclude boundary voxels
    xy_ampL = _slice_xy_ampL.to_numpy()[1:-1, 1:-1]
    xy_ampT = _slice_xy_ampT.to_numpy()[1:-1, 1:-1]
    xy_freq = _slice_xy_freq.to_numpy()[1:-1, 1:-1]
    xz_ampL = _slice_xz_ampL.to_numpy()[1:-1, 1:-1]
    xz_ampT = _slice_xz_ampT.to_numpy()[1:-1, 1:-1]
    xz_freq = _slice_xz_freq.to_numpy()[1:-1, 1:-1]
    yz_ampL = _slice_yz_ampL.to_numpy()[1:-1, 1:-1]
    yz_ampT = _slice_yz_ampT.to_numpy()[1:-1, 1:-1]
    yz_freq = _slice_yz_freq.to_numpy()[1:-1, 1:-1]

    # Compute RMS amplitude: √(⟨A²⟩) for correct energy weighting
    # ampL_am contains per-voxel RMS values, square them for energy
    total_ampL_squared = (xy_ampL**2).sum() + (xz_ampL**2).sum() + (yz_ampL**2).sum()
    total_ampT_squared = (xy_ampT**2).sum() + (xz_ampT**2).sum() + (yz_ampT**2).sum()
    total_freq = xy_freq.sum() + xz_freq.sum() + yz_freq.sum()
    n_samples = xy_ampL.size + xz_ampL.size + yz_ampL.size

    trackers.rms_ampL_am[None] = float(np.sqrt(total_ampL_squared / n_samples))
    trackers.rms_ampT_am[None] = float(np.sqrt(total_ampT_squared / n_samples))
    trackers.avg_freq_rHz[None] = float(total_freq / n_samples)


# ================================================================
# FLUX MESH VALUES UPDATING
# ================================================================


@ti.kernel
def update_flux_mesh_values(
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    color_palette: ti.i32,  # type: ignore
    warp_mesh: ti.i32,  # type: ignore
):
    """
    Update flux mesh colors and vertices by sampling wave properties from voxel grid.

    Samples wave displacement at each plane vertex position and maps it to a color.
    Should be called every frame after wave propagation to update visualization.

    Color palettes:
        1 (redblue): red=negative, gray=zero, blue=positive displacement
        2 (ironbow): black-red-yellow-white heat map for amplitude
        3 (blueprint): blue gradient for frequency visualization

    Args:
        wave_field: WaveField instance containing flux mesh fields and displacement data
        trackers: WaveTrackers instance with amplitude/frequency data for color scaling
        color_palette: Color palette selection (1=redblue, 2=ironbow, 3=blueprint)
    """

    # ================================================================
    # XY Plane: Sample at z = fm_plane_z_idx
    # ================================================================
    # Always update all planes (conditionals cause GPU branch divergence)
    for i, j in ti.ndrange(wave_field.nx, wave_field.ny):
        # Sample longitudinal displacement at this voxel
        psiL_value = wave_field.psiL_am[i, j, wave_field.fm_plane_z_idx]
        psiT_value = wave_field.psiT_am[i, j, wave_field.fm_plane_z_idx]
        ampL_value = trackers.ampL_am[i, j, wave_field.fm_plane_z_idx]
        ampT_value = trackers.ampT_am[i, j, wave_field.fm_plane_z_idx]
        freq_value = trackers.freq_rHz[i, j, wave_field.fm_plane_z_idx]

        # Map value to color/vertex using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 5:  # blueprint
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_freq_rHz[None] * 2
            )
            warp = freq_value / trackers.avg_freq_rHz[None] / 100 + wave_field.flux_mesh_planes[
                2
            ] * (wave_field.nz / wave_field.max_grid_size)
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )
        elif color_palette == 4:  # viridis
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_viridis_color(
                ampL_value, 0, trackers.rms_ampL_am[None] * 2
            )
            warp = ampL_value / trackers.rms_ampL_am[None] / 100 + wave_field.flux_mesh_planes[
                2
            ] * (wave_field.nz / wave_field.max_grid_size)
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )
        elif color_palette == 3:  # ironbow
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_ironbow_color(
                ampT_value, 0, trackers.rms_ampT_am[None] * 2
            )
            warp = ampT_value / trackers.rms_ampT_am[None] / 100 + wave_field.flux_mesh_planes[
                2
            ] * (wave_field.nz / wave_field.max_grid_size)
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )
        elif color_palette == 2:  # yellowgreen
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_yellowgreen_color(
                psiL_value,
                -trackers.rms_ampL_am[None] * 2,
                trackers.rms_ampL_am[None] * 2,
            )
            warp = psiL_value / trackers.rms_ampL_am[None] / 100 + wave_field.flux_mesh_planes[
                2
            ] * (wave_field.nz / wave_field.max_grid_size)
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )
        else:  # default to redblue (palette 1)
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_redblue_color(
                psiT_value,
                -trackers.rms_ampT_am[None] * 2,
                trackers.rms_ampT_am[None] * 2,
            )
            warp = psiT_value / trackers.rms_ampT_am[None] / 100 + wave_field.flux_mesh_planes[
                2
            ] * (wave_field.nz / wave_field.max_grid_size)
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )

    # ================================================================
    # XZ Plane: Sample at y = fm_plane_y_idx
    # ================================================================
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        psiL_value = wave_field.psiL_am[i, wave_field.fm_plane_y_idx, k]
        psiT_value = wave_field.psiT_am[i, wave_field.fm_plane_y_idx, k]
        ampL_value = trackers.ampL_am[i, wave_field.fm_plane_y_idx, k]
        ampT_value = trackers.ampT_am[i, wave_field.fm_plane_y_idx, k]
        freq_value = trackers.freq_rHz[i, wave_field.fm_plane_y_idx, k]

        # Map value to color/vertex using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 5:  # blueprint
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_freq_rHz[None] * 2
            )
            warp = freq_value / trackers.avg_freq_rHz[None] / 100 + wave_field.flux_mesh_planes[
                1
            ] * (wave_field.ny / wave_field.max_grid_size)
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )
        elif color_palette == 4:  # viridis
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_viridis_color(
                ampL_value, 0, trackers.rms_ampL_am[None] * 2
            )
            warp = ampL_value / trackers.rms_ampL_am[None] / 100 + wave_field.flux_mesh_planes[
                1
            ] * (wave_field.ny / wave_field.max_grid_size)
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )
        elif color_palette == 3:  # ironbow
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_ironbow_color(
                ampT_value, 0, trackers.rms_ampT_am[None] * 2
            )
            warp = ampT_value / trackers.rms_ampT_am[None] / 100 + wave_field.flux_mesh_planes[
                1
            ] * (wave_field.ny / wave_field.max_grid_size)
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )
        elif color_palette == 2:  # yellowgreen
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_yellowgreen_color(
                psiL_value,
                -trackers.rms_ampL_am[None] * 2,
                trackers.rms_ampL_am[None] * 2,
            )
            warp = psiL_value / trackers.rms_ampL_am[None] / 100 + wave_field.flux_mesh_planes[
                1
            ] * (wave_field.ny / wave_field.max_grid_size)
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )
        else:  # default to redblue (palette 1)
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_redblue_color(
                psiT_value,
                -trackers.rms_ampT_am[None] * 2,
                trackers.rms_ampT_am[None] * 2,
            )
            warp = psiT_value / trackers.rms_ampT_am[None] / 100 + wave_field.flux_mesh_planes[
                1
            ] * (wave_field.ny / wave_field.max_grid_size)
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )

    # ================================================================
    # YZ Plane: Sample at x = fm_plane_x_idx
    # ================================================================
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        psiL_value = wave_field.psiL_am[wave_field.fm_plane_x_idx, j, k]
        psiT_value = wave_field.psiT_am[wave_field.fm_plane_x_idx, j, k]
        ampL_value = trackers.ampL_am[wave_field.fm_plane_x_idx, j, k]
        ampT_value = trackers.ampT_am[wave_field.fm_plane_x_idx, j, k]
        freq_value = trackers.freq_rHz[wave_field.fm_plane_x_idx, j, k]

        # Map value to color/vertex using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if color_palette == 5:  # blueprint
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_freq_rHz[None] * 2
            )
            warp = freq_value / trackers.avg_freq_rHz[None] / 100 + wave_field.flux_mesh_planes[
                0
            ] * (wave_field.nx / wave_field.max_grid_size)
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )
        elif color_palette == 4:  # viridis
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_viridis_color(
                ampL_value, 0, trackers.rms_ampL_am[None] * 2
            )
            warp = ampL_value / trackers.rms_ampL_am[None] / 100 + wave_field.flux_mesh_planes[
                0
            ] * (wave_field.nx / wave_field.max_grid_size)
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )
        elif color_palette == 3:  # ironbow
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_ironbow_color(
                ampT_value, 0, trackers.rms_ampT_am[None] * 2
            )
            warp = ampT_value / trackers.rms_ampT_am[None] / 100 + wave_field.flux_mesh_planes[
                0
            ] * (wave_field.nx / wave_field.max_grid_size)
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )
        elif color_palette == 2:  # yellowgreen
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_yellowgreen_color(
                psiL_value,
                -trackers.rms_ampL_am[None] * 2,
                trackers.rms_ampL_am[None] * 2,
            )
            warp = psiL_value / trackers.rms_ampL_am[None] / 100 + wave_field.flux_mesh_planes[
                0
            ] * (wave_field.nx / wave_field.max_grid_size)
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )
        else:  # default to redblue (palette 1)
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_redblue_color(
                psiT_value,
                -trackers.rms_ampT_am[None] * 2,
                trackers.rms_ampT_am[None] * 2,
            )
            warp = psiT_value / trackers.rms_ampT_am[None] / 100 + wave_field.flux_mesh_planes[
                0
            ] * (wave_field.nx / wave_field.max_grid_size)
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                warp
                if warp_mesh
                else wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )
