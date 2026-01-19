"""
ENERGY-WAVE ENGINE

ON WOLFF-LAFRENIERE METHOD

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
):
    """
    Compute wave displacement using Wolff-LaFreniere analytical wave equation.

    Wolff-LaFreniere Combined Form:
        ψ(r,t) = A · [sin(ωt - kr) - sin(ωt)] / r

    Expanded Form (used in implementation):
        ψ(r,t) = A · [-cos(ωt)·sin(kr)/r - sin(ωt)·(1-cos(kr))/r]
               = A · [-cos(ωt)·Phase(r) - sin(ωt)·Quadrature(r)]

    Components:
        Phase:      sin(kr)/r  → k as r→0  (standing wave envelope)
        Quadrature: (1-cos(kr))/r → 0 as r→0  (traveling wave component)

    Physical Properties:
        - Near center: standing wave behavior (finite amplitude at r=0)
        - Far from center: transitions to traveling wave
        - 1/r amplitude falloff (energy conserving, from Wolff)
        - Electron core diameter = λ (full wavelength, from LaFreniere)
        - Superposition of multiple wave centers supported

    Wave Trackers Updated:
        - ampL_local_rms_am: RMS amplitude via EMA on ψ² (for energy/force gradients)
        - freq_local_cross_rHz: Frequency via zero-crossing detection with EMA smoothing

    See research/01_wolff_lafreniere.md for full derivation and theory.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        trackers: WaveTrackers instance for tracking wave properties
        wave_center: WaveCenter instance with source positions and phase offsets
        dt_rs: Timestep size (rs)
        elapsed_t_rs: Elapsed simulation time (rs)
    """
    # Grid dimensions for boundary handling
    nx, ny, nz = wave_field.nx, wave_field.ny, wave_field.nz

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = (
        2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor
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
        trackers.ampL_local_envelope_am[i, j, k] = 0.0  # reset envelope before accumulation

        # loop over wave-centers (wave superposition principle causing interference)
        for wc_idx in ti.ndrange(wave_center.num_sources):
            # Skip inactive (annihilated) WCs
            if wave_center.active[wc_idx] == 0:
                continue

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

            # ================================================================
            # Combined and Adjusted WOLFF-LAFRENIERE canonical form:
            #   ψ(r,t) = A · [sin(ωt - kr) - sin(ωt)] / r
            # Expanded form:
            #   ψ(r,t) = A · [-cos(ωt) · sin(kr)/r - sin(ωt) · (1 - cos(kr))/r]
            #   ψ(r,t) = A · [-cos(ωt) · Phase(r) - sin(ωt) · Quadrature(r)]
            # ================================================================
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

            # Oscillator with source_offset phase shift
            oscillator = (
                -ti.cos(temporal_phase + source_offset) * phase_term
                - ti.sin(temporal_phase + source_offset) * quadrature_term
            )

            wave_field.psiL_am[i, j, k] += base_amplitude_am * wave_field.scale_factor * oscillator

            # ================================================================
            # ANALYTICAL SIGNED AMPLITUDE ENVELOPE
            # TODO: Refine envelope model for force calculations
            # TODO: Archive previous envelope models in research/
            # Particles don't respond to 10²⁵ Hz oscillation frequencies.
            # Particle's mass (inertia) acts as a low-pass filter, averaging out the rapid
            # oscillations and responding only to the time-averaged energy-density (envelope).
            # This envelope drives the force calculations, computed directly from wave functions.
            # Applies superposition principle for multiple wave-centers, with signed charge sign.
            # Avoids computationally expensive real-time tracking methods (RMS, zero-crossing).
            # Also avoids instability from real-time EMS calculations of moving wave-centers.
            # ================================================================
            # Charge sign: cos(0)=+1 (eg: positron), cos(π)=-1 (eg: electron)
            charge_sign = ti.cos(source_offset)

            # TODO: Investigate why these constants work well for envelope
            golden_ratio = (1 + ti.sqrt(5)) / 2  # ~1.6180339887
            weight_factor = 2.0 * ti.math.pi**2  # ~19.7392088, decay, damping
            offset_factor = 1 / (wavelength_grid * golden_ratio)

            # # SPIKED 1/r ==================================
            # if r_grid < 0.5:  # CENTER VOXEL only, avoids singularity
            #     trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #         base_amplitude_am * wave_field.scale_factor * k_grid * 2
            #     )  # finite value at center, k_grid / constant + offset
            # else:  # FAR-FIELD: smooth 1/r decay
            #     trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #         base_amplitude_am * wave_field.scale_factor * 1.0 / r_grid
            #     )  # spiked 1/r decay

            # # SMOOTHED 1/r ==================================
            # trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #     base_amplitude_am
            #     * wave_field.scale_factor
            #     * k_grid
            #     / ti.sqrt((k_grid * r_grid) ** 2 + 1)
            # )  # smoothed 1/r decay

            # DAMPED SMOOTHED 1/r ==================================
            trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
                base_amplitude_am
                * wave_field.scale_factor
                * k_grid
                / ti.sqrt((k_grid * r_grid) ** 2 + (2 * ti.math.pi) ** 2)
            )  # smoothed 1/r decay

            # # WOLFF-ORIGINAL ENVELOPE ==================================
            # if r_grid < 0.5:  # CENTER VOXEL only, avoids singularity
            #     trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #         base_amplitude_am * wave_field.scale_factor * k_grid
            #     )  # finite value at center, k_grid
            # else:
            #     trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #         base_amplitude_am * wave_field.scale_factor * ti.sin(k_grid * r_grid) / r_grid
            #     )  # standing-smooth sin(kr)/r decay

            # # WOLFF ONLY AT NEAR-FIELD ==================================
            # if r_grid < 0.5:  # CENTER VOXEL only, avoids singularity
            #     trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #         base_amplitude_am * wave_field.scale_factor * k_grid
            #     )  # finite value at center, k_grid
            # else:
            #     if r_grid <= (2.5 * ti.math.pi / k_grid):  # NEAR-FIELD: time-dilated-1.25λ
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_amq
            #             * wave_field.scale_factor
            #             * ti.sin(k_grid * r_grid)
            #             / r_grid
            #         )  # standing-smooth sin(kr)/r decay
            #     else:  # FAR-FIELD: smooth 1/r decay
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am * wave_field.scale_factor * 1.0 / r_grid
            #         )  # smooth 1/r decay

            # # ABS WOLFF ONLY NEAR-FIELD ==================================
            # if r_grid < 0.5:  # CENTER VOXEL only, avoids singularity
            #     trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #         base_amplitude_am * wave_field.scale_factor * k_grid
            #     )  # finite value at center, k_grid
            # else:
            #     if r_grid <= (2.5 * ti.math.pi / k_grid):  # NEAR-FIELD: time-dilated-1.25λ
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am
            #             * wave_field.scale_factor
            #             * ti.abs(ti.sin(k_grid * r_grid))
            #             / r_grid
            #         )  # standing-smooth sin(kr)/r decay
            #     else:  # FAR-FIELD: smooth 1/r decay
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am * wave_field.scale_factor * 1.0 / r_grid
            #         )  # smooth 1/r decay

            # # DAMPED+OFFSET WOLFF NEAR-FIELD ==================================
            # if r_grid < 0.5:  # CENTER VOXEL only, avoids singularity
            #     trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #         base_amplitude_am * wave_field.scale_factor * k_grid / (2 * ti.math.pi)
            #         + 1 / (wavelength_grid * golden_ratio)
            #     )  # finite value at center, k_grid / constant + offset
            # else:
            #     if r_grid <= (2.5 * ti.math.pi / k_grid):  # NEAR-FIELD: time-dilated-1.25λ
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am
            #             * wave_field.scale_factor
            #             * ti.sin(k_grid * r_grid)
            #             / (r_grid * 2 * ti.math.pi)
            #             + 1 / (wavelength_grid * golden_ratio)
            #         )  # standing-smooth sin(kr)/(r·constant) + offset decay
            #     else:  # FAR-FIELD: smooth 1/r decay
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am * wave_field.scale_factor * 1.0 / r_grid
            #         )  # smooth 1/r decay

            # # ABS DAMPED+OFFSET WOLFF NEAR-FIELD ==================================
            # if r_grid < 0.5:  # CENTER VOXEL only, avoids singularity
            #     trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #         base_amplitude_am * wave_field.scale_factor * k_grid / (2)
            #         + 1 / (wavelength_grid * golden_ratio)
            #     )  # finite value at center, k_grid / constant + offset
            # else:
            #     if r_grid <= (2.5 * ti.math.pi / k_grid):  # NEAR-FIELD: time-dilated-1.5λ
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am
            #             * wave_field.scale_factor
            #             * ti.abs(ti.sin(k_grid * r_grid))
            #             / (r_grid * 2)
            #             + 1 / (wavelength_grid * golden_ratio)
            #         )  # standing-smooth sin(kr)/(r·constant) + offset decay
            #     else:  # FAR-FIELD: smooth 1/r decay
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am * wave_field.scale_factor * 1.0 / r_grid
            #         )  # smooth 1/r decay

            # # DAMPED + WOLFF ==================================
            # if r_grid < 0.5:  # CENTER VOXEL only, avoids singularity
            #     trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #         base_amplitude_am * wave_field.scale_factor * (k_grid / (2 * ti.math.pi))
            #     )  # finite value at center, k_grid / constant
            # else:
            #     if r_grid <= (2.5 * ti.math.pi / k_grid):  # NEAR-FIELD: time-dilated-1.25λ
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am
            #             * wave_field.scale_factor
            #             * (
            #                 k_grid / ti.sqrt((k_grid * r_grid) ** 2 + (48))
            #                 + ti.sin(k_grid * r_grid) / (r_grid * 4)
            #             )
            #         )  # smoothed 1/r decay
            #     else:  # FAR-FIELD: smooth 1/r decay
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am * wave_field.scale_factor * 1.0 / r_grid
            #         )  # smooth 1/r decay

            # # FLAT NEAR-FIELD ==================================
            # if r_grid < 0.5:  # CENTER VOXEL only, avoids singularity
            #     trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #         base_amplitude_am * wave_field.scale_factor * k_grid / (2 * ti.math.pi * 1.25)
            #     )  # finite value at center, k_grid
            # else:
            #     if r_grid <= (2.5 * ti.math.pi / k_grid):  # NEAR-FIELD: time-dilated-1.25λ
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am
            #             * wave_field.scale_factor
            #             * k_grid
            #             / (2 * ti.math.pi * 1.25)
            #         )  # standing-smooth sin(kr)/r decay
            #     else:  # FAR-FIELD: smooth 1/r decay
            #         trackers.ampL_local_envelope_am[i, j, k] += charge_sign * (
            #             base_amplitude_am * wave_field.scale_factor * 1.0 / r_grid
            #         )  # smooth 1/r decay

        # Precision rounding to ensure wave cancellation
        # Critical for opposing phase sources (180°) that should annihilate
        # Floating-point: (+1.250001) + (-1.249999) = 0.000002 (imperfect cancel)
        # With rounding: (+1.2500) + (-1.2500) = 0.0 (perfect cancel)
        precision = ti.cast(1e4, ti.f32)  # round to 4 decimal places
        wave_field.psiL_am[i, j, k] = ti.round(wave_field.psiL_am[i, j, k] * precision) / precision

        curr_disp = wave_field.psiL_am[i, j, k]

        # WAVE-TRACKERS ============================================
        # # PEAK AMPLITUDE tracking
        # ti.atomic_max(trackers.ampL_local_peak_am[i, j, k], curr_disp)
        # decay_factor_peak = ti.cast(0.999, ti.f32)  # ~100 frames to ~37%, ~230 to ~10%
        # trackers.ampL_local_peak_am[i, j, k] = (
        #     trackers.ampL_local_peak_am[i, j, k] * decay_factor_peak
        # )

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
        current_rms2_L = trackers.ampL_local_rms_am[i, j, k] ** 2
        alpha_rms_L = 0.005  # EMA smoothing factor for RMS tracking
        new_rms2_L = alpha_rms_L * disp2_L + (1.0 - alpha_rms_L) * current_rms2_L
        new_ampL = ti.sqrt(new_rms2_L)
        # Unconditional decay clears trails from moving sources
        # Active regions counteract decay via EMA update from strong displacement
        # Stale regions (waves propagated away) decay to zero over time
        decay_factor = ti.cast(0.99, ti.f32)  # ~100 frames to ~37%, ~230 to ~10%
        trackers.ampL_local_rms_am[i, j, k] = new_ampL * decay_factor

        # Transverse RMS amplitude
        disp2_T = wave_field.psiT_am[i, j, k] ** 2
        current_rms2_T = trackers.ampT_local_rms_am[i, j, k] ** 2
        alpha_rms_T = 0.005  # EMA smoothing factor for RMS tracking
        new_rms2_T = alpha_rms_T * disp2_T + (1.0 - alpha_rms_T) * current_rms2_T
        new_ampT = ti.sqrt(new_rms2_T)
        trackers.ampT_local_rms_am[i, j, k] = new_ampT * decay_factor

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
                current_freq = trackers.freq_local_cross_rHz[i, j, k]
                alpha_freq = 0.05  # EMA smoothing factor for frequency
                trackers.freq_local_cross_rHz[i, j, k] = (
                    alpha_freq * measured_freq + (1.0 - alpha_freq) * current_freq
                )
            trackers.last_crossing[i, j, k] = elapsed_t_rs

        # Unconditional frequency decay (counteracted by zero-crossing updates in active regions)
        trackers.freq_local_cross_rHz[i, j, k] *= decay_factor


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
        slice_ampL[i, j] = trackers.ampL_local_rms_am[i, j, mid_z]
        slice_ampT[i, j] = trackers.ampT_local_rms_am[i, j, mid_z]
        slice_freq[i, j] = trackers.freq_local_cross_rHz[i, j, mid_z]


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
        slice_ampL[i, k] = trackers.ampL_local_rms_am[i, mid_y, k]
        slice_ampT[i, k] = trackers.ampT_local_rms_am[i, mid_y, k]
        slice_freq[i, k] = trackers.freq_local_cross_rHz[i, mid_y, k]


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
        slice_ampL[j, k] = trackers.ampL_local_rms_am[mid_x, j, k]
        slice_ampT[j, k] = trackers.ampT_local_rms_am[mid_x, j, k]
        slice_freq[j, k] = trackers.freq_local_cross_rHz[mid_x, j, k]


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
    # ampL_local_rms_am contains per-voxel RMS values, square them for energy
    total_ampL_squared = (xy_ampL**2).sum() + (xz_ampL**2).sum() + (yz_ampL**2).sum()
    total_ampT_squared = (xy_ampT**2).sum() + (xz_ampT**2).sum() + (yz_ampT**2).sum()
    total_freq = xy_freq.sum() + xz_freq.sum() + yz_freq.sum()
    n_samples = xy_ampL.size + xz_ampL.size + yz_ampL.size

    trackers.ampL_global_rms_am[None] = float(np.sqrt(total_ampL_squared / n_samples))
    trackers.ampT_global_rms_am[None] = float(np.sqrt(total_ampT_squared / n_samples))
    trackers.freq_global_avg_rHz[None] = float(total_freq / n_samples)


# ================================================================
# FLUX MESH VALUES UPDATING
# ================================================================


@ti.kernel
def update_flux_mesh_values(
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    wave_menu: ti.i32,  # type: ignore
    warp_mesh: ti.i32,  # type: ignore
):
    """
    Update flux mesh colors and vertices by sampling wave properties from voxel grid.

    Samples wave displacement at each plane vertex position and maps it to a color.
    Should be called every frame after wave propagation to update visualization.

    Args:
        wave_field: WaveField instance containing flux mesh fields and displacement data
        trackers: WaveTrackers instance with amplitude/frequency data for color scaling
        wave_menu: Selected Wave displayed with color palette
    """

    # ================================================================
    # XY Plane: Sample at z = fm_plane_z_idx
    # ================================================================
    # Always update all planes (conditionals cause GPU branch divergence)
    for i, j in ti.ndrange(wave_field.nx, wave_field.ny):
        # Sample longitudinal displacement at this voxel
        psiL_value = wave_field.psiL_am[i, j, wave_field.fm_plane_z_idx]
        ampLr_value = trackers.ampL_local_rms_am[i, j, wave_field.fm_plane_z_idx]
        ampLe_value = trackers.ampL_local_envelope_am[i, j, wave_field.fm_plane_z_idx]
        univ_edge_z = wave_field.universe_size_am[2]

        # Map value to color/vertex using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if wave_menu == 4:  # greenyellow
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_greenyellow_color(
                ampLe_value,
                -trackers.ampL_global_rms_am[None] * 2,
                trackers.ampL_global_rms_am[None] * 2,
            )
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                ampLe_value / univ_edge_z * warp_mesh
                + wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )
        elif wave_menu == 3:  # viridis
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_viridis_color(
                ampLr_value, 0, trackers.ampL_global_rms_am[None] * 2
            )
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                ampLr_value / univ_edge_z * warp_mesh
                + wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )
        else:  # default to greenyellow (wave_menu == 1)
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_greenyellow_color(
                psiL_value,
                -trackers.ampL_global_rms_am[None] * 2,
                trackers.ampL_global_rms_am[None] * 2,
            )
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                psiL_value / univ_edge_z * warp_mesh
                + wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )

    # ================================================================
    # XZ Plane: Sample at y = fm_plane_y_idx
    # ================================================================
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        psiL_value = wave_field.psiL_am[i, wave_field.fm_plane_y_idx, k]
        ampLr_value = trackers.ampL_local_rms_am[i, wave_field.fm_plane_y_idx, k]
        ampLe_value = trackers.ampL_local_envelope_am[i, wave_field.fm_plane_y_idx, k]
        univ_edge_y = wave_field.universe_size_am[1]

        # Map value to color/vertex using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if wave_menu == 4:  # greenyellow
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_greenyellow_color(
                ampLe_value,
                -trackers.ampL_global_rms_am[None] * 2,
                trackers.ampL_global_rms_am[None] * 2,
            )
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                ampLe_value / univ_edge_y * warp_mesh
                + wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )
        elif wave_menu == 3:  # viridis
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_viridis_color(
                ampLr_value, 0, trackers.ampL_global_rms_am[None] * 2
            )
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                ampLr_value / univ_edge_y * warp_mesh
                + wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )
        else:  # default to greenyellow (wave_menu == 1)
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_greenyellow_color(
                psiL_value,
                -trackers.ampL_global_rms_am[None] * 2,
                trackers.ampL_global_rms_am[None] * 2,
            )
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                psiL_value / univ_edge_y * warp_mesh
                + wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )

    # ================================================================
    # YZ Plane: Sample at x = fm_plane_x_idx
    # ================================================================
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        psiL_value = wave_field.psiL_am[wave_field.fm_plane_x_idx, j, k]
        ampLr_value = trackers.ampL_local_rms_am[wave_field.fm_plane_x_idx, j, k]
        ampLe_value = trackers.ampL_local_envelope_am[wave_field.fm_plane_x_idx, j, k]
        univ_edge_x = wave_field.universe_size_am[0]

        # Map value to color/vertex using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if wave_menu == 4:  # greenyellow
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_greenyellow_color(
                ampLe_value,
                -trackers.ampL_global_rms_am[None] * 2,
                trackers.ampL_global_rms_am[None] * 2,
            )
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                ampLe_value / univ_edge_x * warp_mesh
                + wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )
        elif wave_menu == 3:  # viridis
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_viridis_color(
                ampLr_value, 0, trackers.ampL_global_rms_am[None] * 2
            )
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                ampLr_value / univ_edge_x * warp_mesh
                + wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )
        else:  # default to greenyellow (wave_menu == 1)
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_greenyellow_color(
                psiL_value,
                -trackers.ampL_global_rms_am[None] * 2,
                trackers.ampL_global_rms_am[None] * 2,
            )
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                psiL_value / univ_edge_x * warp_mesh
                + wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )
