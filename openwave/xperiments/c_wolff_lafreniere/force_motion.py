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
# PHASE 1: SMOKE TEST - Hardcoded Force Motion
# ================================================================


@ti.kernel
def smoketest_particle_motion(
    wave_field: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
    dt_rs: ti.f32,  # type: ignore
):
    """
    SMOKE TEST: Apply hardcoded force to debug motion integration.
    Removed when validations passed. Replaced with force from energy gradients.

    Applies a constant force in +x direction to all wave centers.
    Used to verify that:
    1. Position updates work correctly
    2. Unit conversions are correct
    3. Visualization shows particle motion

    Args:
        wave_field: WaveField instance (for dx voxel size)
        wave_center: WaveCenter instance with position/velocity fields
        dt_rs: Time step in rontoseconds
    """
    # Voxel size in attometers for position conversion
    dx_am = wave_field.dx / ATTOMETER

    # ================================================================
    # SMOKE TEST: Compute acceleration for consistent visible motion
    # ================================================================
    # Target: move 1 grid cell after ~500 frames
    # Kinematic: x = 0.5 * a * (N * dt)²
    # Solving for a: a = 2 * x / (N * dt)²
    # Where x = dx_am (1 grid cell in attometers)
    #
    # a_amrs = 2 * dx_am / (N² * dt_rs²)
    target_frames = ti.cast(100.0, ti.f32)  # Move 1 grid cell in n frames
    a_smoke = 2.0 * dx_am / (target_frames * target_frames * dt_rs * dt_rs)

    for wc_idx in range(wave_center.num_sources):
        # Apply computed acceleration in +x direction
        # NOTE: NOT physically realistic - purely for testing motion integration.
        a_x = a_smoke
        a_y = ti.cast(0.0, ti.f32)
        a_z = ti.cast(0.0, ti.f32)

        # ================================================================
        # Update velocity: v_new = v_old + a * dt (in am/rs)
        # ================================================================
        wave_center.velocity_amrs[wc_idx][0] += a_x * dt_rs
        wave_center.velocity_amrs[wc_idx][1] += a_y * dt_rs
        wave_center.velocity_amrs[wc_idx][2] += a_z * dt_rs

        # ================================================================
        # Update position: x_new = x_old + v * dt
        # ================================================================
        # Position change in attometers
        dx_am_step = wave_center.velocity_amrs[wc_idx][0] * dt_rs
        dy_am_step = wave_center.velocity_amrs[wc_idx][1] * dt_rs
        dz_am_step = wave_center.velocity_amrs[wc_idx][2] * dt_rs

        # Convert attometers to grid index change: di = dx_am / voxel_size_am
        di = dx_am_step / dx_am
        dj = dy_am_step / dx_am
        dk = dz_am_step / dx_am

        # Update float position (smooth motion)
        wave_center.position_float[wc_idx][0] += di
        wave_center.position_float[wc_idx][1] += dj
        wave_center.position_float[wc_idx][2] += dk

        # Update integer grid position for wave generation
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
# PHASE 3: Force from Energy Gradient
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

        # Gradient sampling radius in voxels
        # Must be large enough to sample beyond own wave into interference region
        # Using ~15% of smallest grid dimension - compromise between:
        #   - Reaching interference region (needs large radius)
        #   - Staying within bounds for particles at 25%/75% positions (limits radius)
        # TODO: Smarter approach - sample toward other particles, or clamp to bounds
        min_dim = ti.min(nx, ti.min(ny, nz))
        sample_radius = 1  # ti.max(min_dim * 1 // 100, 10)  # At least 10, ~15% of grid

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
            # Gives EWT-predicted 1/r² force law independent of initial separation
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

            # DEBUG: Store intermediate values
            wave_center.debug_A_center[wc_idx] = A_center_am * ATTOMETER
            wave_center.debug_dA_dx[wc_idx] = dA_dx
            wave_center.debug_force_scale[wc_idx] = force_scale_qgrs * 1e21  # qg/rs² to kg/s²

            # Force: F = -2 * rho * V * f^2 * A * grad(A), in N, converted from qg/rs² * am
            F_x = -force_scale_qgrs * A_center_am * dA_dx * 1000  # qg·am/rs² to N (kg·m/s²)
            F_y = -force_scale_qgrs * A_center_am * dA_dy * 1000  # qg·am/rs² to N (kg·m/s²)
            F_z = -force_scale_qgrs * A_center_am * dA_dz * 1000  # qg·am/rs² to N (kg·m/s²)

        # Store computed force (in Newtons)
        wave_center.force[wc_idx][0] = F_x
        wave_center.force[wc_idx][1] = F_y
        wave_center.force[wc_idx][2] = F_z


# ================================================================
# PHASE 4: Motion Integration (Euler)
# ================================================================


@ti.kernel
def integrate_motion_euler(
    wave_field: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
    dt_rs: ti.f32,  # type: ignore
    sim_speed: ti.f32,  # type: ignore
):
    """
    Integrate particle motion using Euler method.

    v_new = v_old + a * dt  (velocity in am/rs)
    x_new = x_old + v_new * dt  (position in grid indices)

    Args:
        wave_field: WaveField instance (for dx voxel size)
        wave_center: WaveCenter instance with force/velocity/position fields
        dt_rs: Time step in rontoseconds
        sim_speed: Simulation speed multiplier for particle motion
    """
    # Force multiplier for visualization (physically unrealistic but useful for testing)
    # Real forces are tiny at quantum scale - this boosts them for visible motion
    # Too high = oscillation (velocity flips every frame), too low = no visible motion
    # TODO: Make configurable via xparameters or UI
    FORCE_MULTIPLIER = ti.cast(1, ti.f32)  # Reduced to allow gradual velocity buildup

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
        # Apply force multiplier for visualization
        a_x_amrs = (F_x / m_qg) * accel_conv_qg * FORCE_MULTIPLIER
        a_y_amrs = (F_y / m_qg) * accel_conv_qg * FORCE_MULTIPLIER
        a_z_amrs = (F_z / m_qg) * accel_conv_qg * FORCE_MULTIPLIER

        # Update velocity: v_new = v_old + a * dt (in am/rs)
        wave_center.velocity_amrs[wc_idx][0] += a_x_amrs * dt_rs
        wave_center.velocity_amrs[wc_idx][1] += a_y_amrs * dt_rs
        wave_center.velocity_amrs[wc_idx][2] += a_z_amrs * dt_rs

        # Clamp velocity to speed of light (c = 0.3 am/rs)
        # velocity clamp to prevent superluminal speeds
        c_amrs = ti.cast(0.3, ti.f32)
        v_mag = (
            ti.sqrt(
                wave_center.velocity_amrs[wc_idx][0] ** 2
                + wave_center.velocity_amrs[wc_idx][1] ** 2
                + wave_center.velocity_amrs[wc_idx][2] ** 2
            )
            / sim_speed  # Scale velocity by sim speed for consistent motion
        )
        if v_mag > c_amrs:
            scale = c_amrs / v_mag
            wave_center.velocity_amrs[wc_idx][0] *= scale
            wave_center.velocity_amrs[wc_idx][1] *= scale
            wave_center.velocity_amrs[wc_idx][2] *= scale

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
# PHASE 5: Annihilation Detection
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


# ================================================================
# DEBUG: Force Analysis
# ================================================================


def debug_force_analysis(wave_field, trackers, wave_center, frame: int = 0):
    """
    Debug function to analyze force calculation values.

    Prints amplitude, gradient, and force values for each wave center.
    Includes Coulomb force comparison for model calibration.
    Call periodically (e.g., every 100 frames) to monitor force evolution.
    """
    if frame % 100 != 0:
        return

    print(f"\n{'='*60}")
    print(f"FORCE ANALYSIS - Frame {frame}")
    print(f"{'='*60}")

    # ================================================================
    # EWT FORCE COMPARISON (Primary validation)
    # ================================================================
    # EWT Electric Force: F_e = (4πρ K^7 A^6 c² Oe / 3λ²) × gλ × (1/r²)
    #
    # This is the CORRECT comparison for OpenWave validation:
    # - Derived from wave physics (same as our simulation)
    # - Uses K (wave center count) as parameter
    # - For K=1: Oe=1.0, gλ=1.0 (no outer shell complexity)
    # - For K=10 (electron): Oe=2.14, gλ=0.99
    #
    # If simulated force matches EWT force, OpenWave is validated!
    # ================================================================
    positions_grid = wave_center.position_grid.to_numpy()
    dx = wave_field.dx  # voxel size in meters

    if wave_center.num_sources == 2:
        # Calculate separation distance in meters
        pos0 = positions_grid[0] * dx
        pos1 = positions_grid[1] * dx
        separation = np.linalg.norm(pos1 - pos0)

        if separation > 0:
            # EWT force for K=1 (fundamental particle)
            F_ewt_k1 = compute_ewt_electric_force(separation, K=1, Oe=1.0, glambda=1.0)

            # EWT force for K=10 (electron) - should match Coulomb
            F_ewt_k10 = compute_ewt_electric_force(
                separation, K=10, Oe=ELECTRON_OUTER_SHELL, glambda=ELECTRON_ORBITAL_G
            )

            # Coulomb force for reference
            F_coulomb = COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / separation**2

            # Direction vector (from 0 to 1)
            direction = (pos1 - pos0) / separation

            print(f"\n{'─'*60}")
            print("EWT FORCE PREDICTIONS (Target for validation)")
            print(f"{'─'*60}")
            print(f"  Separation: {separation:.3e} m ({separation/ATTOMETER:.3f} am)")
            print(f"  F_ewt (K=1):  {F_ewt_k1:.3e} N  ← TARGET for simulation")
            print(f"  F_ewt (K=10): {F_ewt_k10:.3e} N  (electron)")
            print(f"  F_coulomb:    {F_coulomb:.3e} N  (reference)")
            print(f"  EWT K=10 / Coulomb ratio: {F_ewt_k10/F_coulomb:.6f}")
            print(
                f"  Direction (WC0→WC1): [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]"
            )
            print(f"{'─'*60}")

    # Get numpy arrays from taichi fields
    # Using envelope field (smooth 1/r) for force calculation instead of RMS
    envelope = trackers.ampL_local_envelope_am.to_numpy()
    positions = wave_center.position_grid.to_numpy()
    forces = wave_center.force.to_numpy()
    velocities = wave_center.velocity_amrs.to_numpy()

    dx = wave_field.dx
    dx_am = wave_field.dx / ATTOMETER

    print(f"Voxel size: {dx:.3e} m ({dx_am:.3f} am)")
    # Compute force_scale same way as kernel (avoid overflow)
    rhoV = MEDIUM_DENSITY * dx**3
    force_scale = 2.0 * rhoV * EWAVE_FREQUENCY * EWAVE_FREQUENCY
    # Scale factor correction
    S = wave_field.scale_factor
    S4 = S * S * S * S
    force_scale_corrected = force_scale / S4

    print(f"Force scale (raw): 2*rho*V*f² = {force_scale:.3e}")
    print(f"  (rho*V = {rhoV:.3e}, f = {EWAVE_FREQUENCY:.3e})")
    print(f"Scale factor: {S:.1f}x (S⁴ = {S4:.3e})")
    print(f"Force scale (corrected): {force_scale_corrected:.3e}")

    # Show sampling radius
    min_dim = min(wave_field.nx, wave_field.ny, wave_field.nz)
    sample_radius = max(min_dim * 15 // 100, 10)
    print(f"Gradient sampling radius: {sample_radius} voxels (~15% of min grid dim {min_dim})")

    # Read kernel debug values
    kernel_A_center = wave_center.debug_A_center.to_numpy()
    kernel_dA_dx = wave_center.debug_dA_dx.to_numpy()
    kernel_force_scale = wave_center.debug_force_scale.to_numpy()

    for wc_idx in range(wave_center.num_sources):
        i, j, k = positions[wc_idx]
        print(f"\n--- Wave Center {wc_idx} at grid [{i}, {j}, {k}] ---")
        print(f"  KERNEL DEBUG VALUES:")
        print(f"    A_center (kernel): {kernel_A_center[wc_idx]:.3e}")
        print(f"    dA_dx (kernel): {kernel_dA_dx[wc_idx]:.3e}")
        print(f"    force_scale (kernel): {kernel_force_scale[wc_idx]:.3e}")
        if kernel_force_scale[wc_idx] != 0:
            expected_Fx = (
                -kernel_force_scale[wc_idx] * kernel_A_center[wc_idx] * kernel_dA_dx[wc_idx]
            )
            print(f"    Expected F_x: {expected_Fx:.3e}")

        # Sampling radius (must match kernel logic)
        min_dim = min(wave_field.nx, wave_field.ny, wave_field.nz)
        sample_radius = max(min_dim * 15 // 100, 10)

        # Boundary check
        if i <= sample_radius or i >= wave_field.nx - sample_radius:
            print(f"  WARNING: Near x boundary!")
            continue
        if j <= sample_radius or j >= wave_field.ny - sample_radius:
            print(f"  WARNING: Near y boundary!")
            continue
        if k <= sample_radius or k >= wave_field.nz - sample_radius:
            print(f"  WARNING: Near z boundary!")
            continue

        # Envelope values at sampling radius distance (smooth 1/r field)
        A_center = envelope[i, j, k] * ATTOMETER
        A_xp = envelope[i + sample_radius, j, k] * ATTOMETER
        A_xm = envelope[i - sample_radius, j, k] * ATTOMETER
        A_yp = envelope[i, j + sample_radius, k] * ATTOMETER
        A_ym = envelope[i, j - sample_radius, k] * ATTOMETER
        A_zp = envelope[i, j, k + sample_radius] * ATTOMETER
        A_zm = envelope[i, j, k - sample_radius] * ATTOMETER

        print(f"  Envelope at center: {A_center:.3e} m ({envelope[i,j,k]:.3f} am)")
        print(f"  Amplitude x±{sample_radius}: [{A_xm:.3e}, {A_xp:.3e}] m")

        # Gradients with larger sampling distance
        sample_dist = 2.0 * sample_radius * dx
        dA_dx = (A_xp - A_xm) / sample_dist
        dA_dy = (A_yp - A_ym) / sample_dist
        dA_dz = (A_zp - A_zm) / sample_dist

        print(f"  Gradient dA/dx: {dA_dx:.3e}")
        print(f"  Gradient dA/dy: {dA_dy:.3e}")
        print(f"  Gradient dA/dz: {dA_dz:.3e}")

        # Force
        F = forces[wc_idx]
        print(f"  Force: [{F[0]:.3e}, {F[1]:.3e}, {F[2]:.3e}] N")

        # Expected acceleration (using actual mass from wave_center in qg)
        masses = wave_center.mass_qg.to_numpy()
        m_qg = masses[wc_idx]  # mass in quectograms
        m_kg = m_qg * QUECTOGRAM  # convert to kg for display
        FORCE_MULTIPLIER = 1  # Must match value in integrate_motion_euler
        # Acceleration using qg conversion: (F/m_qg) * 1e-3 -> am/rs²
        a_amrs2 = (F[0] / m_qg) * 1e-3 * FORCE_MULTIPLIER if m_qg > 0 else 0
        a_ms2 = F[0] / m_kg if m_kg > 0 else 0  # for display
        print(f"  Mass: {m_qg:.3e} qg ({m_kg:.3e} kg)")
        print(f"  Acceleration: {a_ms2:.3e} m/s² ({a_amrs2:.3e} am/rs²)")

        # Velocity
        v = velocities[wc_idx]
        print(f"  Velocity: [{v[0]:.6e}, {v[1]:.6e}, {v[2]:.6e}] am/rs")

    # ================================================================
    # VALIDATION SUMMARY: Simulated vs EWT Prediction
    # ================================================================
    if wave_center.num_sources == 2:
        pos0 = positions_grid[0] * dx
        pos1 = positions_grid[1] * dx
        separation = np.linalg.norm(pos1 - pos0)

        if separation > 0:
            # EWT predicted force for K=1
            F_ewt_k1 = compute_ewt_electric_force(separation, K=1, Oe=1.0, glambda=1.0)

            # Get simulated force magnitudes
            forces = wave_center.force.to_numpy()
            F_sim_0 = np.linalg.norm(forces[0])
            F_sim_1 = np.linalg.norm(forces[1])
            F_sim_avg = (F_sim_0 + F_sim_1) / 2

            # Calculate ratio to EWT K=1 prediction (THIS IS THE KEY METRIC)
            ratio_ewt = F_sim_avg / F_ewt_k1 if F_ewt_k1 > 0 else 0

            # Also show Coulomb comparison for reference
            F_coulomb = COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / separation**2
            ratio_coulomb = F_sim_avg / F_coulomb if F_coulomb > 0 else 0

            print(f"\n{'─'*60}")
            print("VALIDATION: Simulated vs EWT Prediction")
            print(f"{'─'*60}")
            print(f"  EWT F (K=1 target):     {F_ewt_k1:.3e} N")
            print(f"  Simulated F (WC0):      {F_sim_0:.3e} N")
            print(f"  Simulated F (WC1):      {F_sim_1:.3e} N")
            print(f"  Simulated F (average):  {F_sim_avg:.3e} N")
            print(f"{'─'*60}")
            print(f"  RATIO (sim/EWT K=1):    {ratio_ewt:.6f}")
            if ratio_ewt > 0:
                print(f"  Scale needed:           {1/ratio_ewt:.3e}x")
            print(f"{'─'*60}")
            if ratio_ewt > 0.9 and ratio_ewt < 1.1:
                print("  ✓ VALIDATED: Within 10% of EWT prediction!")
            elif ratio_ewt > 0.5 and ratio_ewt < 2.0:
                print("  ~ CLOSE: Within 2x of EWT prediction")
            else:
                print("  ✗ Calibration needed")
            print(f"{'─'*60}")
            print(f"  (Coulomb ref: {F_coulomb:.3e} N, ratio: {ratio_coulomb:.3e})")
            print(f"{'─'*60}")

            # Direction check
            direction = (pos1 - pos0) / separation
            F0_dir = forces[0] / F_sim_0 if F_sim_0 > 0 else np.zeros(3)
            dot_product = np.dot(F0_dir, direction)
            force_type = "ATTRACTION" if dot_product > 0 else "REPULSION"
            print(f"  Force direction: {force_type} (dot={dot_product:.3f})")
            print(f"{'─'*60}")

    print(f"{'='*60}\n")
