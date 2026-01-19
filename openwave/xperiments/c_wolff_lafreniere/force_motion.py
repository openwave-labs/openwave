"""
FORCE & MOTION MODULE

Implements force calculation from energy gradients and particle motion integration.

Physics Foundation:
- Force = -grad(E) where E = rho * V * (f * A)^2 (EWT energy equation)
- Monochromatic: F = -2 * rho * V * f^2 * A * grad(A)
- Motion: Euler integration of F = m * a

Units:
- Force: Newtons (SI)
- Mass: qg (quectograms, 1 qg = 1e-33 kg, for f32 precision on GPU)
- Velocity: am/rs (OpenWave scaled units for f32 precision)
- Position: grid indices (float)
- Time: rontoseconds (rs)

Conversion factors:
- v_amrs = v_ms * 1e-9           (m/s to am/rs)
- a_amrs2 = a_ms2 * 1e-36        (m/s² to am/rs²)
- a_amrs2 = (F/m_qg) * 1e-3      (N/qg to am/rs², for GPU f32 precision)
- c = 0.3 am/rs                  (speed of light)

See research/02_force_motion.md for detailed documentation.

This package contains the fundamental components:
- electric: Electric force calculations and representations
- magnetic: Magnetic force calculations and representations
- gravitational: Gravitational force calculations and representations
- strong: Strong force calculations and representations
- orbital: Orbital force calculations and representations
"""

import taichi as ti

import numpy as np

from openwave.common import constants

# ================================================================
# Physical Constants (cached for kernel access, float precision)
# ================================================================
MEDIUM_DENSITY = constants.MEDIUM_DENSITY  # 3.85e22 kg/m^3
MEDIUM_DENSITY_QGAM = constants.MEDIUM_DENSITY_QGAM  # 38.5 qg/am^3
EWAVE_FREQUENCY = constants.EWAVE_FREQUENCY  # 1.05e25 Hz
EWAVE_FREQUENCY_RHZ = constants.EWAVE_FREQUENCY_RHZ  # 0.0105 rHz
EWAVE_AMPLITUDE = constants.EWAVE_AMPLITUDE  # 9.21e-19 m
EWAVE_AMPLITUDE_AM = constants.EWAVE_AMPLITUDE_AM  # 0.92 am
EWAVE_SPEED = constants.EWAVE_SPEED  # 3.00e8 m/s (c)
EWAVE_SPEED_AMRS = constants.EWAVE_SPEED_AMRS  # 0.3 am/rs (c)
EWAVE_LENGTH = constants.EWAVE_LENGTH  # 2.85e-17 m (λ)
EWAVE_LENGTH_AM = constants.EWAVE_LENGTH_AM  # 28.5 am (λ)

ATTOMETER = constants.ATTOMETER  # m/am = 1e-18
RONTOSECOND = constants.RONTOSECOND  # s/rs = 1e-27
QUECTOGRAM = constants.QUECTOGRAM  # kg/qg = 1e-33, for GPU f32 precision

# Coulomb force constants (for reference)
COULOMB_CONSTANT = constants.COULOMB_CONSTANT  # N·m²/C², k = 8.99e9
ELEMENTARY_CHARGE = constants.ELEMENTARY_CHARGE  # C, e = 1.60e-19

# EWT particle constants
NEUTRINO_K = constants.NEUTRINO_K  # K=1 for neutrino
ELECTRON_K = constants.ELECTRON_K  # K=10 for electron
ELECTRON_OUTER_SHELL = constants.ELECTRON_OUTER_SHELL  # Oe for electron
ELECTRON_ORBITAL_G = constants.ELECTRON_ORBITAL_G  # gλ for electron

# Unit conversion factors
# From m/s² to am/rs²: a_amrs2 = a_ms2 / ATTOMETER * RONTOSECOND²
# = a_ms2 * (1/1e-18) * (1e-27)² = a_ms2 * 1e18 * 1e-54 = a_ms2 * 1e-36

# Gradient sampling radius in voxels
# Must be large enough to sample beyond own wave into interference region
# Using ~15% of smallest grid dimension - compromise between:
#   - Reaching interference region (needs large radius)
#   - Staying within bounds for particles at 25%/75% positions (limits radius)
# TODO: Smarter approach - sample toward other particles, or clamp to bounds
# min_dim = ti.min(nx, ti.min(ny, nz))
# sample_radius = ti.max(min_dim * 15 // 100, 10)  # At least 10, ~15% of grid
GRADIENT_SAMPLE_RADIUS = 1  # voxels, for gradient sampling in force calculation


def compute_ewt_electric_force(
    r: float, K: int = 1, Oe: float = 1.0, glambda: float = 1.0
) -> float:
    """
    Compute EWT electric force between two particles.

    F_e = (4πρ K^7 A^6 c² Oe / 3λ²) × gλ × (Q1×Q2 / r²)

    For like particles (Q1=Q2=1), this becomes:
    F_e = (4πρ K^7 A^6 c² Oe gλ / 3λ²) / r²

    Args:
        r: Distance between particles in meters
        K: Wave center count (1 for neutrino, 10 for electron)
        Oe: Outer shell multiplier (1.0 for K=1, ~2.14 for electron)
        glambda: Orbital g-factor (1.0 for K=1, ~0.99 for electron)

    Returns:
        Force in Newtons
    """
    coefficient = (
        4.0
        * np.pi
        * MEDIUM_DENSITY
        * (K**7)
        * (EWAVE_AMPLITUDE**6)
        * (EWAVE_SPEED**2)
        * Oe
        * glambda
    ) / (3.0 * (EWAVE_LENGTH**2))

    return coefficient / (r**2)


# ================================================================
# Force from Energy Gradient
# ================================================================


@ti.kernel
def compute_force_vector(
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
):
    """
    Compute force on each wave center from energy gradient.

    F = -∇(E) = -2 * rho * V * f^2 * A * grad(A)   (monochromatic)

    Uses central finite differences for gradient calculation.
    Samples amplitude field around wave center position.

    Args:
        wave_field: WaveField instance containing grid info
        trackers: Trackers instance with amplitude fields
        wave_center: WaveCenter instance to store computed forces
    """
    # Physical constants
    rho_qgam = ti.cast(MEDIUM_DENSITY_QGAM, ti.f32)  # in qg/am³
    f_rHz = ti.cast(EWAVE_FREQUENCY_RHZ, ti.f32)  # in rHz
    dx_am = wave_field.dx_am  # voxel size in attometers

    # Force scale factor: 2 * rho * V * f^2 where V = dx³
    # CRITICAL: Scaled SI units for f32 precision
    force_scale_qgrs = 2.0 * rho_qgam * dx_am**3 * f_rHz**2  # in qg/rs²

    # Scale factor correction: Force scales as S⁴ with universe scaling
    # F_real = F_scaled / S⁴
    # This converts forces back to physically correct units
    S = wave_field.scale_factor
    S4 = S * S * S * S  # S⁴, dimensionless
    force_scale_qgrs = force_scale_qgrs / S4

    for wc_idx in range(wave_center.num_sources):
        # Skip inactive (annihilated) WCs
        if wave_center.active[wc_idx] == 0:
            continue

        # Get wave center grid position
        i = wave_center.position_grid[wc_idx][0]
        j = wave_center.position_grid[wc_idx][1]
        k = wave_center.position_grid[wc_idx][2]

        # Initialize force to zero
        F_x = ti.cast(0.0, ti.f32)
        F_y = ti.cast(0.0, ti.f32)
        F_z = ti.cast(0.0, ti.f32)

        # Grid dimensions
        nx = wave_field.nx
        ny = wave_field.ny
        nz = wave_field.nz

        # Gradient sampling radius (uses module constant GRADIENT_SAMPLE_RADIUS)
        sample_radius = GRADIENT_SAMPLE_RADIUS

        # Boundary check (need neighbors for gradient at sample_radius distance)
        if (
            i > sample_radius
            and i < nx - sample_radius
            and j > sample_radius
            and j < ny - sample_radius
            and k > sample_radius
            and k < nz - sample_radius
        ):
            # ================================================================
            # Use ENVELOPE field for force calculation (smooth 1/r, no oscillations)
            # Envelope = sum of (charge_sign * A₀/r) from each source
            # Gives EWT-predicted 1/r² force law
            # ================================================================
            # Sample envelope at center
            A_center_am = trackers.ampL_local_envelope_am[i, j, k]

            # Central difference gradient with larger sampling radius:
            # grad(A) = (A[+R] - A[-R]) / (2*R*dx)
            # This averages the gradient over a larger region
            sample_dist_am = 2.0 * sample_radius * dx_am

            # X gradient from envelope field
            A_xp_am = trackers.ampL_local_envelope_am[i + sample_radius, j, k]
            A_xm_am = trackers.ampL_local_envelope_am[i - sample_radius, j, k]
            dA_dx = (A_xp_am - A_xm_am) / sample_dist_am

            # Y gradient from envelope field
            A_yp_am = trackers.ampL_local_envelope_am[i, j + sample_radius, k]
            A_ym_am = trackers.ampL_local_envelope_am[i, j - sample_radius, k]
            dA_dy = (A_yp_am - A_ym_am) / sample_dist_am

            # Z gradient from envelope field
            A_zp_am = trackers.ampL_local_envelope_am[i, j, k + sample_radius]
            A_zm_am = trackers.ampL_local_envelope_am[i, j, k - sample_radius]
            dA_dz = (A_zp_am - A_zm_am) / sample_dist_am

            # Force: F = -2 * rho * V * f^2 * A * grad(A), in N, converted from qg/rs² * am
            F_x = -force_scale_qgrs * A_center_am * dA_dx * 1000  # qg·am/rs² to N (kg·m/s²)
            F_y = -force_scale_qgrs * A_center_am * dA_dy * 1000  # qg·am/rs² to N (kg·m/s²)
            F_z = -force_scale_qgrs * A_center_am * dA_dz * 1000  # qg·am/rs² to N (kg·m/s²)

        # Store computed force (in Newtons)
        wave_center.force[wc_idx][0] = F_x
        wave_center.force[wc_idx][1] = F_y
        wave_center.force[wc_idx][2] = F_z


# ================================================================
# Motion Integration (Euler)
# ================================================================


@ti.kernel
def integrate_motion_euler(
    wave_field: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
    dt_rs: ti.f32,  # type: ignore
):
    """
    Integrate particle motion using Euler method.

    v_new = v_old + a * dt  (velocity in am/rs)
    x_new = x_old + v_new * dt  (position in grid indices)

    Args:
        wave_field: WaveField instance (for dx voxel size)
        wave_center: WaveCenter instance with force/velocity/position fields
        dt_rs: Timestep in rontoseconds
    """
    # Conversion factor: (N / qg) to am/rs²
    # Using quectograms (qg) instead of kg for f32 precision on GPU
    # Division by small kg values (e.g., 4.26e-36) underflows on GPU f32
    # With qg: m_qg = 4.26e-3 (f32-friendly), conversion factor = 1e-3
    accel_conv_qg = ti.cast(1e-3, ti.f32)  # (F_N / m_qg) * 1e-3 -> am/rs²

    # Voxel size in attometers for position conversion
    dx_am = wave_field.dx / ti.cast(ATTOMETER, ti.f32)

    for wc_idx in range(wave_center.num_sources):
        # Skip inactive (annihilated) WCs
        if wave_center.active[wc_idx] == 0:
            continue

        # Get force (Newtons) and mass (qg - quectograms for GPU precision)
        F_x = wave_center.force[wc_idx][0]
        F_y = wave_center.force[wc_idx][1]
        F_z = wave_center.force[wc_idx][2]
        m_qg = wave_center.mass_qg[wc_idx]  # mass in quectograms

        # Acceleration: a = F/m, then convert (N/qg) to am/rs²
        a_x_amrs = (F_x / m_qg) * accel_conv_qg
        a_y_amrs = (F_y / m_qg) * accel_conv_qg
        a_z_amrs = (F_z / m_qg) * accel_conv_qg

        # Update velocity: v_new = v_old + a * dt (in am/rs)
        wave_center.velocity_amrs[wc_idx][0] += a_x_amrs * dt_rs
        wave_center.velocity_amrs[wc_idx][1] += a_y_amrs * dt_rs
        wave_center.velocity_amrs[wc_idx][2] += a_z_amrs * dt_rs

        # Position change in attometers
        dx_am_step = wave_center.velocity_amrs[wc_idx][0] * dt_rs
        dy_am_step = wave_center.velocity_amrs[wc_idx][1] * dt_rs
        dz_am_step = wave_center.velocity_amrs[wc_idx][2] * dt_rs

        # Convert to grid index change
        di = dx_am_step / dx_am
        dj = dy_am_step / dx_am
        dk = dz_am_step / dx_am

        wave_center.position_float[wc_idx][0] += di
        wave_center.position_float[wc_idx][1] += dj
        wave_center.position_float[wc_idx][2] += dk

        # # Clamp position to grid boundaries (with margin for gradient sampling)
        # margin = ti.cast(2, ti.f32)  # Keep 2 voxels from edge
        # nx_f = ti.cast(wave_field.nx, ti.f32)
        # ny_f = ti.cast(wave_field.ny, ti.f32)
        # nz_f = ti.cast(wave_field.nz, ti.f32)

        # wave_center.position_float[wc_idx][0] = ti.max(
        #     margin, ti.min(nx_f - margin, wave_center.position_float[wc_idx][0])
        # )
        # wave_center.position_float[wc_idx][1] = ti.max(
        #     margin, ti.min(ny_f - margin, wave_center.position_float[wc_idx][1])
        # )
        # wave_center.position_float[wc_idx][2] = ti.max(
        #     margin, ti.min(nz_f - margin, wave_center.position_float[wc_idx][2])
        # )

        # Sync integer position for wave generation
        wave_center.position_grid[wc_idx][0] = ti.cast(
            wave_center.position_float[wc_idx][0], ti.i32
        )
        wave_center.position_grid[wc_idx][1] = ti.cast(
            wave_center.position_float[wc_idx][1], ti.i32
        )
        wave_center.position_grid[wc_idx][2] = ti.cast(
            wave_center.position_float[wc_idx][2], ti.i32
        )


# ================================================================
# Annihilation Detection
# ================================================================


@ti.kernel
def detect_annihilation(
    wave_center: ti.template(),  # type: ignore
    annihilation_threshold: ti.f32,  # type: ignore  # Distance threshold in grid units
):
    """
    Annihilation naturally occurs from wave physics, but needs numerical precision check.
    Detect and handle particle annihilation when WCs converge to same position.

    When two wave centers with opposite phase (180°) attract and meet:
    1. Their waves should cancel perfectly (handled by wave precision rounding)
    2. Snap both WCs to exact same position (prevent drift/oscillation)
    3. Zero velocities to prevent separation

    This ensures annihilation is permanent - no wave reappearance from micro-motion.
    Numerical precision limits may cause slight separation otherwise.

    Checks all pairs of WCs - only annihilates pairs with opposite phases (~180° apart).

    Args:
        wave_center: WaveCenter instance with position/velocity fields
        annihilation_threshold: Distance in grid units to trigger annihilation
    """
    # Phase threshold for opposite phases: |phase_diff - π| < tolerance
    # Using ~10° tolerance (0.17 rad) for numerical stability
    phase_tolerance = ti.cast(0.17, ti.f32)
    pi = ti.cast(3.14159265359, ti.f32)

    # Check all pairs (i, j) where i < j
    for i in range(wave_center.num_sources):
        for j in range(i + 1, wave_center.num_sources):
            # Skip if either WC is already inactive
            if wave_center.active[i] == 0 or wave_center.active[j] == 0:
                continue

            # Check if phases are opposite (differ by ~π)
            phase_diff = ti.abs(wave_center.offset[i] - wave_center.offset[j])
            # Normalize to [0, 2π] range and check if near π
            phase_diff_normalized = ti.abs(phase_diff - pi)
            if phase_diff_normalized > phase_tolerance:
                continue  # Not opposite phases, skip

            # Calculate distance between WCs (in grid units)
            dx = wave_center.position_float[i][0] - wave_center.position_float[j][0]
            dy = wave_center.position_float[i][1] - wave_center.position_float[j][1]
            dz = wave_center.position_float[i][2] - wave_center.position_float[j][2]
            distance = ti.sqrt(dx * dx + dy * dy + dz * dz)

            # Check if WCs are within annihilation threshold
            if distance < annihilation_threshold:
                # Snap both WCs to their midpoint (exact same position)
                mid_x = (wave_center.position_float[i][0] + wave_center.position_float[j][0]) / 2.0
                mid_y = (wave_center.position_float[i][1] + wave_center.position_float[j][1]) / 2.0
                mid_z = (wave_center.position_float[i][2] + wave_center.position_float[j][2]) / 2.0

                # Round to exact grid position (integer) to ensure perfect alignment
                mid_x_int = ti.round(mid_x)
                mid_y_int = ti.round(mid_y)
                mid_z_int = ti.round(mid_z)

                # Set both WCs to exact same position
                wave_center.position_float[i][0] = mid_x_int
                wave_center.position_float[i][1] = mid_y_int
                wave_center.position_float[i][2] = mid_z_int
                wave_center.position_float[j][0] = mid_x_int
                wave_center.position_float[j][1] = mid_y_int
                wave_center.position_float[j][2] = mid_z_int

                # Zero velocities to prevent separation
                wave_center.velocity_amrs[i][0] = 0.0
                wave_center.velocity_amrs[i][1] = 0.0
                wave_center.velocity_amrs[i][2] = 0.0
                wave_center.velocity_amrs[j][0] = 0.0
                wave_center.velocity_amrs[j][1] = 0.0
                wave_center.velocity_amrs[j][2] = 0.0

                # Sync integer grid positions
                wave_center.position_grid[i][0] = ti.cast(mid_x_int, ti.i32)
                wave_center.position_grid[i][1] = ti.cast(mid_y_int, ti.i32)
                wave_center.position_grid[i][2] = ti.cast(mid_z_int, ti.i32)
                wave_center.position_grid[j][0] = ti.cast(mid_x_int, ti.i32)
                wave_center.position_grid[j][1] = ti.cast(mid_y_int, ti.i32)
                wave_center.position_grid[j][2] = ti.cast(mid_z_int, ti.i32)

                # Mark both WCs as inactive (annihilated)
                wave_center.active[i] = 0
                wave_center.active[j] = 0
