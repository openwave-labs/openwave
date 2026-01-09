"""
FORCE & MOTION MODULE

Implements force calculation from energy gradients and particle motion integration.

Physics Foundation:
- Force = -grad(E) where E = rho * V * (f * A)^2 (EWT energy equation)
- Monochromatic: F = -2 * rho * V * f^2 * A * grad(A)
- Motion: Euler integration of F = m * a

Units:
- Force: Newtons (SI)
- Mass: kg (SI)
- Velocity: am/rs (OpenWave scaled units for f32 precision)
- Position: grid indices (float)
- Time: rontoseconds (rs)

Conversion factors:
- v_amrs = v_ms * 1e-9           (m/s to am/rs)
- a_amrs2 = a_ms2 * 1e-36        (m/s² to am/rs²)
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
# Physical Constants (cached for kernel access)
# ================================================================
MEDIUM_DENSITY = constants.MEDIUM_DENSITY  # kg/m^3
EWAVE_FREQUENCY = constants.EWAVE_FREQUENCY  # Hz
ELECTRON_MASS = constants.ELECTRON_MASS  # kg
ATTOMETER = constants.ATTOMETER  # m/am = 1e-18
RONTOSECOND = constants.RONTOSECOND  # s/rs = 1e-27

# Coulomb force constants (for comparison with simulated force)
COULOMB_CONSTANT = constants.COULOMB_CONSTANT  # N·m²/C², k = 8.99e9
ELEMENTARY_CHARGE = constants.ELEMENTARY_CHARGE  # C, e = 1.60e-19

# Unit conversion factors
# From m/s² to am/rs²: a_amrs2 = a_ms2 / ATTOMETER * RONTOSECOND²
# = a_ms2 * (1/1e-18) * (1e-27)² = a_ms2 * 1e18 * 1e-54 = a_ms2 * 1e-36
ACCEL_MS2_TO_AMRS2 = 1.0 / ATTOMETER * RONTOSECOND * RONTOSECOND  # = 1e-36


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
    SMOKE TEST: Apply hardcoded force to verify motion integration.

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

    for wc in range(wave_center.num_sources):
        # Apply computed acceleration in +x direction
        # NOTE: NOT physically realistic - purely for testing motion integration.
        a_x = a_smoke
        a_y = ti.cast(0.0, ti.f32)
        a_z = ti.cast(0.0, ti.f32)

        # ================================================================
        # Update velocity: v_new = v_old + a * dt (in am/rs)
        # ================================================================
        wave_center.velocity_amrs[wc][0] += a_x * dt_rs
        wave_center.velocity_amrs[wc][1] += a_y * dt_rs
        wave_center.velocity_amrs[wc][2] += a_z * dt_rs

        # ================================================================
        # Update position: x_new = x_old + v * dt
        # ================================================================
        # Position change in attometers
        dx_am_step = wave_center.velocity_amrs[wc][0] * dt_rs
        dy_am_step = wave_center.velocity_amrs[wc][1] * dt_rs
        dz_am_step = wave_center.velocity_amrs[wc][2] * dt_rs

        # Convert attometers to grid index change: di = dx_am / voxel_size_am
        di = dx_am_step / dx_am
        dj = dy_am_step / dx_am
        dk = dz_am_step / dx_am

        # Update float position (smooth motion)
        wave_center.position_float[wc][0] += di
        wave_center.position_float[wc][1] += dj
        wave_center.position_float[wc][2] += dk

        # Update integer grid position for wave generation
        wave_center.position_grid[wc][0] = ti.cast(wave_center.position_float[wc][0], ti.i32)
        wave_center.position_grid[wc][1] = ti.cast(wave_center.position_float[wc][1], ti.i32)
        wave_center.position_grid[wc][2] = ti.cast(wave_center.position_float[wc][2], ti.i32)


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

    F = -grad(E) = -2 * rho * V * f^2 * A * grad(A)   (monochromatic)

    Uses central finite differences for gradient calculation.
    Samples amplitude field around wave center position.

    Args:
        wave_field: WaveField instance containing grid info
        trackers: Trackers instance with ampL_am field
        wave_center: WaveCenter instance to store computed forces
    """
    # Physical constants
    rho = ti.cast(MEDIUM_DENSITY, ti.f32)
    f = ti.cast(EWAVE_FREQUENCY, ti.f32)
    dx_m = wave_field.dx  # voxel size in meters

    # Force scale factor: 2 * rho * V * f^2 where V = dx³
    # CRITICAL: Interleave large/small values to avoid f32 under/overflow!
    #
    # Problem 1: V = dx³ can underflow (f32 min ~1.2e-38)
    # Problem 2: f*f ≈ 1.5e+40 OVERFLOWS (f32 max ~3.4e+38)
    # Problem 3: Order matters! rho*dx*f*dx*f overflows for large dx
    #
    # Solution: Compute as rho*dx*dx*f*dx*f (group dx² early):
    #   For dx=1e-16 (large voxels):
    #     rho*dx*dx = 1e+25 * 1e-32 = 1e-7 ✓
    #     *f = 1e-7 * 1e+25 = 1e+18 ✓
    #     *dx = 1e+18 * 1e-16 = 1e+2 ✓
    #     *f = 1e+2 * 1e+25 = 1e+27 ✓
    #   For dx=2e-18 (small voxels):
    #     rho*dx*dx = 1e+25 * 4e-36 = 4e-11 ✓
    #     *f = 4e-11 * 1e+25 = 4e+14 ✓
    #     *dx = 4e+14 * 2e-18 = 8e-4 ✓
    #     *f = 8e-4 * 1e+25 = 8e+21 ✓
    force_scale = 2.0 * rho * dx_m * dx_m * f * dx_m * f  # Safe for all dx

    # Scale factor correction: Force scales as S⁴ with universe scaling
    # F_real = F_scaled / S⁴
    # This converts forces back to physically correct units
    S = wave_field.scale_factor
    S4 = S * S * S * S  # S⁴
    force_scale = force_scale / S4

    for wc in range(wave_center.num_sources):
        # Get wave center grid position
        i = wave_center.position_grid[wc][0]
        j = wave_center.position_grid[wc][1]
        k = wave_center.position_grid[wc][2]

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
            # Sample amplitude at center (convert am to meters)
            A_center = trackers.ampL_am[i, j, k] * ATTOMETER

            # Central difference gradient with larger sampling radius:
            # grad(A) = (A[+R] - A[-R]) / (2*R*dx)
            # This averages the gradient over a larger region, capturing interference patterns
            sample_dist = 2.0 * sample_radius * dx_m

            # X gradient
            A_xp = trackers.ampL_am[i + sample_radius, j, k] * ATTOMETER
            A_xm = trackers.ampL_am[i - sample_radius, j, k] * ATTOMETER
            dA_dx = (A_xp - A_xm) / sample_dist

            # Y gradient
            A_yp = trackers.ampL_am[i, j + sample_radius, k] * ATTOMETER
            A_ym = trackers.ampL_am[i, j - sample_radius, k] * ATTOMETER
            dA_dy = (A_yp - A_ym) / sample_dist

            # Z gradient
            A_zp = trackers.ampL_am[i, j, k + sample_radius] * ATTOMETER
            A_zm = trackers.ampL_am[i, j, k - sample_radius] * ATTOMETER
            dA_dz = (A_zp - A_zm) / sample_dist

            # DEBUG: Store intermediate values
            wave_center.debug_A_center[wc] = A_center
            wave_center.debug_dA_dx[wc] = dA_dx
            wave_center.debug_force_scale[wc] = force_scale

            # Force: F = -2 * rho * V * f^2 * A * grad(A)
            F_x = -force_scale * A_center * dA_dx
            F_y = -force_scale * A_center * dA_dy
            F_z = -force_scale * A_center * dA_dz

        # Store computed force (in Newtons)
        wave_center.force[wc][0] = F_x
        wave_center.force[wc][1] = F_y
        wave_center.force[wc][2] = F_z


# ================================================================
# PHASE 4: Motion Integration (Euler)
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
        dt_rs: Time step in rontoseconds
    """
    # Force multiplier for visualization (physically unrealistic but useful for testing)
    # Real forces are tiny at quantum scale - this boosts them for visible motion
    # Too high = oscillation (velocity flips every frame), too low = no visible motion
    # TODO: Make configurable via xparameters or UI
    FORCE_MULTIPLIER = ti.cast(2000, ti.f32)  # Reduced to allow gradual velocity buildup

    # Conversion factor: m/s² to am/rs²
    accel_conv = ti.cast(ACCEL_MS2_TO_AMRS2, ti.f32)

    # Voxel size in attometers for position conversion
    dx_am = wave_field.dx / ATTOMETER

    for wc in range(wave_center.num_sources):
        # Get force (Newtons) and mass (kg)
        F_x = wave_center.force[wc][0]
        F_y = wave_center.force[wc][1]
        F_z = wave_center.force[wc][2]
        m = wave_center.mass[wc]

        # Acceleration in m/s², then convert to am/rs²
        # Apply force multiplier for visualization
        a_x = (F_x / m) * accel_conv * FORCE_MULTIPLIER
        a_y = (F_y / m) * accel_conv * FORCE_MULTIPLIER
        a_z = (F_z / m) * accel_conv * FORCE_MULTIPLIER

        # Update velocity (am/rs)
        wave_center.velocity_amrs[wc][0] += a_x * dt_rs
        wave_center.velocity_amrs[wc][1] += a_y * dt_rs
        wave_center.velocity_amrs[wc][2] += a_z * dt_rs

        # Clamp velocity to speed of light (c = 0.3 am/rs)
        # velocity clamp to prevent superluminal speeds
        # TODO: Replace with proper relativistic treatment
        c_amrs = ti.cast(0.3, ti.f32)
        v_mag = ti.sqrt(
            wave_center.velocity_amrs[wc][0] ** 2
            + wave_center.velocity_amrs[wc][1] ** 2
            + wave_center.velocity_amrs[wc][2] ** 2
        )
        if v_mag > c_amrs:
            scale = c_amrs / v_mag
            wave_center.velocity_amrs[wc][0] *= scale
            wave_center.velocity_amrs[wc][1] *= scale
            wave_center.velocity_amrs[wc][2] *= scale

        # Position change in attometers
        dx_am_step = wave_center.velocity_amrs[wc][0] * dt_rs
        dy_am_step = wave_center.velocity_amrs[wc][1] * dt_rs
        dz_am_step = wave_center.velocity_amrs[wc][2] * dt_rs

        # Convert to grid index change
        di = dx_am_step / dx_am
        dj = dy_am_step / dx_am
        dk = dz_am_step / dx_am

        wave_center.position_float[wc][0] += di
        wave_center.position_float[wc][1] += dj
        wave_center.position_float[wc][2] += dk

        # Clamp position to grid boundaries (with margin for gradient sampling)
        # margin = ti.cast(2, ti.f32)  # Keep 2 voxels from edge
        # nx_f = ti.cast(wave_field.nx, ti.f32)
        # ny_f = ti.cast(wave_field.ny, ti.f32)
        # nz_f = ti.cast(wave_field.nz, ti.f32)

        # wave_center.position_float[wc][0] = ti.max(
        #     margin, ti.min(nx_f - margin, wave_center.position_float[wc][0])
        # )
        # wave_center.position_float[wc][1] = ti.max(
        #     margin, ti.min(ny_f - margin, wave_center.position_float[wc][1])
        # )
        # wave_center.position_float[wc][2] = ti.max(
        #     margin, ti.min(nz_f - margin, wave_center.position_float[wc][2])
        # )

        # Sync integer position for wave generation
        wave_center.position_grid[wc][0] = ti.cast(wave_center.position_float[wc][0], ti.i32)
        wave_center.position_grid[wc][1] = ti.cast(wave_center.position_float[wc][1], ti.i32)
        wave_center.position_grid[wc][2] = ti.cast(wave_center.position_float[wc][2], ti.i32)


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
    # COULOMB FORCE COMPARISON
    # ================================================================
    # For two-particle systems, calculate expected Coulomb force
    # F_coulomb = k * e² / r²  where k = 8.99e9 N·m²/C², e = 1.60e-19 C
    #
    # This is the BENCHMARK: our wave-based force should match Coulomb
    # force once properly calibrated.
    # ================================================================
    positions_grid = wave_center.position_grid.to_numpy()
    dx_m = wave_field.dx  # voxel size in meters

    if wave_center.num_sources == 2:
        # Calculate separation distance in meters
        pos0 = positions_grid[0] * dx_m
        pos1 = positions_grid[1] * dx_m
        separation = np.linalg.norm(pos1 - pos0)

        if separation > 0:
            # Coulomb force magnitude: F = k * e² / r²
            F_coulomb = COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / separation**2

            # Direction vector (from 0 to 1)
            direction = (pos1 - pos0) / separation

            print(f"\n{'─'*60}")
            print("COULOMB FORCE BENCHMARK (Target for calibration)")
            print(f"{'─'*60}")
            print(f"  Separation: {separation:.3e} m ({separation/ATTOMETER:.3f} am)")
            print(f"  k (Coulomb constant): {COULOMB_CONSTANT:.3e} N·m²/C²")
            print(f"  e (elementary charge): {ELEMENTARY_CHARGE:.3e} C")
            print(f"  F_coulomb = k·e²/r² = {F_coulomb:.3e} N")
            print(
                f"  Direction (WC0→WC1): [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]"
            )
            print(f"{'─'*60}")

    # Get numpy arrays from taichi fields
    ampL = trackers.ampL_am.to_numpy()
    positions = wave_center.position_grid.to_numpy()
    forces = wave_center.force.to_numpy()
    velocities = wave_center.velocity_amrs.to_numpy()

    dx_m = wave_field.dx
    dx_am = wave_field.dx / ATTOMETER

    print(f"Voxel size: {dx_m:.3e} m ({dx_am:.3f} am)")
    # Compute force_scale same way as kernel (avoid overflow)
    rhoV = MEDIUM_DENSITY * dx_m**3
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

    for wc in range(wave_center.num_sources):
        i, j, k = positions[wc]
        print(f"\n--- Wave Center {wc} at grid [{i}, {j}, {k}] ---")
        print(f"  KERNEL DEBUG VALUES:")
        print(f"    A_center (kernel): {kernel_A_center[wc]:.3e}")
        print(f"    dA_dx (kernel): {kernel_dA_dx[wc]:.3e}")
        print(f"    force_scale (kernel): {kernel_force_scale[wc]:.3e}")
        if kernel_force_scale[wc] != 0:
            expected_Fx = -kernel_force_scale[wc] * kernel_A_center[wc] * kernel_dA_dx[wc]
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

        # Amplitude values at sampling radius distance
        A_center = ampL[i, j, k] * ATTOMETER
        A_xp = ampL[i + sample_radius, j, k] * ATTOMETER
        A_xm = ampL[i - sample_radius, j, k] * ATTOMETER
        A_yp = ampL[i, j + sample_radius, k] * ATTOMETER
        A_ym = ampL[i, j - sample_radius, k] * ATTOMETER
        A_zp = ampL[i, j, k + sample_radius] * ATTOMETER
        A_zm = ampL[i, j, k - sample_radius] * ATTOMETER

        print(f"  Amplitude at center: {A_center:.3e} m ({ampL[i,j,k]:.3f} am)")
        print(f"  Amplitude x±{sample_radius}: [{A_xm:.3e}, {A_xp:.3e}] m")

        # Gradients with larger sampling distance
        sample_dist = 2.0 * sample_radius * dx_m
        dA_dx = (A_xp - A_xm) / sample_dist
        dA_dy = (A_yp - A_ym) / sample_dist
        dA_dz = (A_zp - A_zm) / sample_dist

        print(f"  Gradient dA/dx: {dA_dx:.3e}")
        print(f"  Gradient dA/dy: {dA_dy:.3e}")
        print(f"  Gradient dA/dz: {dA_dz:.3e}")

        # Force
        F = forces[wc]
        print(f"  Force: [{F[0]:.3e}, {F[1]:.3e}, {F[2]:.3e}] N")

        # Expected acceleration (using actual mass from wave_center, not ELECTRON_MASS)
        masses = wave_center.mass.to_numpy()
        m = masses[wc]
        FORCE_MULTIPLIER = 2000  # Must match value in integrate_motion_euler
        a_ms2 = F[0] / m if m > 0 else 0
        a_amrs2 = a_ms2 * ACCEL_MS2_TO_AMRS2 * FORCE_MULTIPLIER
        print(f"  Mass: {m:.3e} kg")
        print(f"  Acceleration: {a_ms2:.3e} m/s² (with 2000x multiplier: {a_amrs2:.3e} am/rs²)")

        # Velocity
        v = velocities[wc]
        print(f"  Velocity: [{v[0]:.3e}, {v[1]:.3e}, {v[2]:.3e}] am/rs")

    # ================================================================
    # FINAL COMPARISON SUMMARY (for 2-particle systems)
    # ================================================================
    if wave_center.num_sources == 2:
        pos0 = positions_grid[0] * dx_m
        pos1 = positions_grid[1] * dx_m
        separation = np.linalg.norm(pos1 - pos0)

        if separation > 0:
            F_coulomb = COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / separation**2

            # Get simulated force magnitudes
            forces = wave_center.force.to_numpy()
            F_sim_0 = np.linalg.norm(forces[0])
            F_sim_1 = np.linalg.norm(forces[1])
            F_sim_avg = (F_sim_0 + F_sim_1) / 2

            # Calculate calibration ratio
            ratio = F_sim_avg / F_coulomb if F_coulomb > 0 else 0

            print(f"\n{'─'*60}")
            print("CALIBRATION SUMMARY")
            print(f"{'─'*60}")
            print(f"  Coulomb F (expected):   {F_coulomb:.3e} N")
            print(f"  Simulated F (WC0):      {F_sim_0:.3e} N")
            print(f"  Simulated F (WC1):      {F_sim_1:.3e} N")
            print(f"  Simulated F (average):  {F_sim_avg:.3e} N")
            print(f"  Ratio (sim/coulomb):    {ratio:.3e}")
            if ratio > 0:
                print(f"  Scale needed:           {1/ratio:.3e}x")
            print(f"{'─'*60}")

            # Direction check
            direction = (pos1 - pos0) / separation
            F0_dir = forces[0] / F_sim_0 if F_sim_0 > 0 else np.zeros(3)
            dot_product = np.dot(F0_dir, direction)
            force_type = "ATTRACTION" if dot_product > 0 else "REPULSION"
            print(f"  Force direction: {force_type} (dot={dot_product:.3f})")
            print(f"{'─'*60}")

    print(f"{'='*60}\n")
