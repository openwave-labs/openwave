"""
FORCE & MOTION MODULE

Implements force calculation from energy gradients and particle motion integration.

Physics Foundation:
- Energy per voxel: E = ρ · V · (f · A)²  (EWT energy equation)
- Force: F = -∇E  (negative gradient of energy density)
- Motion: Euler integration of F = m · a

Units:
- Energy: aJ (attojoules, 1 aJ = 1e-18 J)
- Force: Newtons (1 aJ/am = 1 N, no conversion needed)
- Mass: qg (quectograms, 1 qg = 1e-33 kg, for f32 precision on GPU)
- Velocity: am/rs (OpenWave scaled units for f32 precision)
- Position: grid indices (float)
- Time: rontoseconds (rs)

Conversion factors:
- 1 aJ / 1 am = 1e-18 J / 1e-18 m = 1 N  (energy gradient → force)
- a_amrs2 = (F_N / m_qg) * 1e-3            (N/qg → am/rs²)
- c = 0.3 am/rs                             (speed of light)

See research/02_force_motion.md for detailed documentation.
"""

import taichi as ti

import numpy as np

from openwave.common import constants

# ================================================================
# Physical Constants (cached for kernel access)
# ================================================================
ATTOMETER = constants.ATTOMETER  # m/am = 1e-18
RONTOSECOND = constants.RONTOSECOND  # s/rs = 1e-27
QUECTOGRAM = constants.QUECTOGRAM  # kg/qg = 1e-33

# Coulomb force constants (for reference comparisons)
COULOMB_CONSTANT = constants.COULOMB_CONSTANT  # N·m²/C², k = 8.99e9
ELEMENTARY_CHARGE = constants.ELEMENTARY_CHARGE  # C, e = 1.60e-19

# EWT particle constants (for EWT force reference)
MEDIUM_DENSITY = constants.MEDIUM_DENSITY  # kg/m³
EWAVE_AMPLITUDE = constants.EWAVE_AMPLITUDE  # m
WAVE_SPEED = constants.WAVE_SPEED  # m/s
EWAVE_LENGTH = constants.EWAVE_LENGTH  # m
ELECTRON_K = constants.ELECTRON_K
ELECTRON_OUTER_SHELL = constants.ELECTRON_OUTER_SHELL
ELECTRON_ORBITAL_G = constants.ELECTRON_ORBITAL_G


def compute_ewt_electric_force(
    r: float, K: int = 1, Oe: float = 1.0, glambda: float = 1.0
) -> float:
    """
    Compute EWT electric force between two particles (reference/validation).

    F_e = (4πρ K^7 A^6 c² Oe / 3λ²) × gλ × (Q1×Q2 / r²)

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
        * (WAVE_SPEED**2)
        * Oe
        * glambda
    ) / (3.0 * (EWAVE_LENGTH**2))

    return coefficient / (r**2)


# ================================================================
# Force from Energy Gradient: F = -∇E
# ================================================================

# Gradient sampling: number of voxel shells and weight exponent
# GRADIENT_SAMPLE_RADIUS = 1: standard central difference (single shell)
# GRADIENT_SAMPLE_RADIUS > 1: weighted multi-shell gradient (better well resolution)
# GRADIENT_WEIGHT_FALLOFF = 2: weights as 1/d² (particle energy density ∝ A² ∝ 1/r²)
GRADIENT_SAMPLE_RADIUS = 3  # voxels (increased from 1 for better lock-in well resolution)
GRADIENT_WEIGHT_FALLOFF = 2  # exponent for 1/d^n weighting

# Velocity damping: fraction of velocity retained per timestep
# 1.0 = no damping, 0.99 = light damping, 0.95 = moderate damping
# Physically: models energy dissipation via radiation (photon emission)
VELOCITY_DAMPING = 0.999


@ti.kernel
def compute_force_vector(
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
):
    """
    Compute force on each wave center from energy gradient.

    F = -∇E where E is the local energy field (energy_local_aJ).
    Uses weighted central differences: multiple sample shells within
    GRADIENT_SAMPLE_RADIUS, weighted by 1/d² (particle energy density
    falloff ∝ A² ∝ 1/r²). Closer voxels contribute more, matching how
    a particle's own wave structure "feels" the surrounding energy landscape.
    For R=1: identical to standard central difference (single shell).
    Units: aJ/am = N (no conversion needed).

    ┌───────┬────────┬────────────┐
    │ Shell │ Weight │ Percentage │
    ├───────┼────────┼────────────┤
    │ d=1   │ 1.0    │ 73.5%      │
    ├───────┼────────┼────────────┤
    │ d=2   │ 0.25   │ 18.4%      │
    ├───────┼────────┼────────────┤
    │ d=3   │ 0.111  │ 8.2%       │
    └───────┴────────┴────────────┘

    Scale correction: amplitude boost from scale_factor makes energy S² too
    large, so F_physical = F_computed / S².

    Args:
        wave_field: WaveField instance containing grid info
        trackers: Trackers instance with energy_local_aJ field
        wave_center: WaveCenter instance to store computed forces
    """
    dx_am = wave_field.dx_am

    # Scale factor correction: energy scales as S² from amplitude boost
    S = wave_field.scale_factor
    S2 = S * S

    # Precompute weights: 1/d^GRADIENT_WEIGHT_FALLOFF for each shell d = 1..R
    # Physical basis: particle energy density ∝ |ψ|² ∝ 1/r²
    w_sum = ti.cast(0.0, ti.f32)
    for d in ti.static(range(1, GRADIENT_SAMPLE_RADIUS + 1)):
        w_sum += 1.0 / ti.cast(d**GRADIENT_WEIGHT_FALLOFF, ti.f32)

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

        # Boundary check
        if (
            i > GRADIENT_SAMPLE_RADIUS
            and i < nx - GRADIENT_SAMPLE_RADIUS
            and j > GRADIENT_SAMPLE_RADIUS
            and j < ny - GRADIENT_SAMPLE_RADIUS
            and k > GRADIENT_SAMPLE_RADIUS
            and k < nz - GRADIENT_SAMPLE_RADIUS
        ):
            # Weighted gradient: Σ w(d) · (E[+d] - E[-d]) / (2·d·dx) / Σ w(d)
            # w(d) = 1/d^GRADIENT_WEIGHT_FALLOFF (energy density weighting)
            grad_x = ti.cast(0.0, ti.f32)
            grad_y = ti.cast(0.0, ti.f32)
            grad_z = ti.cast(0.0, ti.f32)

            for d in ti.static(range(1, GRADIENT_SAMPLE_RADIUS + 1)):
                w = 1.0 / ti.cast(d**GRADIENT_WEIGHT_FALLOFF, ti.f32)
                dist = 2.0 * d * dx_am

                grad_x += (
                    w
                    * (
                        trackers.energy_local_aJ[i + d, j, k]
                        - trackers.energy_local_aJ[i - d, j, k]
                    )
                    / dist
                )
                grad_y += (
                    w
                    * (
                        trackers.energy_local_aJ[i, j + d, k]
                        - trackers.energy_local_aJ[i, j - d, k]
                    )
                    / dist
                )
                grad_z += (
                    w
                    * (
                        trackers.energy_local_aJ[i, j, k + d]
                        - trackers.energy_local_aJ[i, j, k - d]
                    )
                    / dist
                )

            # F = -∇E / S² (aJ/am = N, with scale correction)
            F_x = -grad_x / (w_sum * S2)
            F_y = -grad_y / (w_sum * S2)
            F_z = -grad_z / (w_sum * S2)

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
        m_qg = wave_center.mass_qg[wc_idx]

        # Acceleration: a = F/m, then convert (N/qg) to am/rs²
        a_x_amrs = (F_x / m_qg) * accel_conv_qg
        a_y_amrs = (F_y / m_qg) * accel_conv_qg
        a_z_amrs = (F_z / m_qg) * accel_conv_qg

        # Update velocity: v_new = v_old + a * dt (in am/rs)
        wave_center.velocity_amrs[wc_idx][0] += a_x_amrs * dt_rs
        wave_center.velocity_amrs[wc_idx][1] += a_y_amrs * dt_rs
        wave_center.velocity_amrs[wc_idx][2] += a_z_amrs * dt_rs

        # Clamp velocity to speed of light (c = 0.3 am/rs)
        # velocity clamp to prevent superluminal speeds
        c_amrs = ti.cast(0.3, ti.f32)
        v_mag = ti.sqrt(
            wave_center.velocity_amrs[wc_idx][0] ** 2
            + wave_center.velocity_amrs[wc_idx][1] ** 2
            + wave_center.velocity_amrs[wc_idx][2] ** 2
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
# Motion Integration (Leapfrog / Velocity Verlet)
# ================================================================


@ti.kernel
def integrate_motion_leapfrog(
    wave_field: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
    dt_rs: ti.f32,  # type: ignore
):
    """
    Integrate particle motion using Velocity Verlet (leapfrog) method.

    Unlike Euler, leapfrog is symplectic — it conserves energy in oscillatory
    systems, preventing numerical drift that causes particles to escape
    lock-in wells. The method uses half-step velocity updates:

    1. v(t + dt/2) = v(t) + a(t) * dt/2       (half-step kick)
    2. x(t + dt)   = x(t) + v(t + dt/2) * dt   (full-step drift)
    3. compute a(t + dt) from new positions      (done externally)
    4. v(t + dt)   = v(t + dt/2) + a(t + dt) * dt/2  (half-step kick)

    Steps 1+2 are done here. Step 3 is the force computation (external).
    Step 4 is done on the NEXT call (the first half-kick uses the NEW force).

    In practice, we store v at half-steps and do:
    v += a * dt   (full kick, combining two half-kicks across calls)
    x += v * dt   (drift with updated velocity)

    This is equivalent to standard leapfrog and is symplectic.

    Args:
        wave_field: WaveField instance (for dx voxel size)
        wave_center: WaveCenter instance with force/velocity/position fields
        dt_rs: Timestep in rontoseconds
    """
    accel_conv_qg = ti.cast(1e-3, ti.f32)  # (F_N / m_qg) * 1e-3 -> am/rs²
    dx_am = wave_field.dx / ti.cast(ATTOMETER, ti.f32)
    damping = ti.cast(VELOCITY_DAMPING, ti.f32)

    for wc_idx in range(wave_center.num_sources):
        if wave_center.active[wc_idx] == 0:
            continue

        F_x = wave_center.force[wc_idx][0]
        F_y = wave_center.force[wc_idx][1]
        F_z = wave_center.force[wc_idx][2]
        m_qg = wave_center.mass_qg[wc_idx]

        # Acceleration from current force
        a_x = (F_x / m_qg) * accel_conv_qg
        a_y = (F_y / m_qg) * accel_conv_qg
        a_z = (F_z / m_qg) * accel_conv_qg

        # Full velocity kick: v += a * dt (leapfrog: combines two half-kicks)
        wave_center.velocity_amrs[wc_idx][0] += a_x * dt_rs
        wave_center.velocity_amrs[wc_idx][1] += a_y * dt_rs
        wave_center.velocity_amrs[wc_idx][2] += a_z * dt_rs

        # Apply damping (models radiation energy loss)
        wave_center.velocity_amrs[wc_idx][0] *= damping
        wave_center.velocity_amrs[wc_idx][1] *= damping
        wave_center.velocity_amrs[wc_idx][2] *= damping

        # Clamp velocity to speed of light (c = 0.3 am/rs)
        c_amrs = ti.cast(0.3, ti.f32)
        v_mag = ti.sqrt(
            wave_center.velocity_amrs[wc_idx][0] ** 2
            + wave_center.velocity_amrs[wc_idx][1] ** 2
            + wave_center.velocity_amrs[wc_idx][2] ** 2
        )
        if v_mag > c_amrs:
            scale = c_amrs / v_mag
            wave_center.velocity_amrs[wc_idx][0] *= scale
            wave_center.velocity_amrs[wc_idx][1] *= scale
            wave_center.velocity_amrs[wc_idx][2] *= scale

        # Drift: x += v * dt (position update with kicked velocity)
        dx_am_step = wave_center.velocity_amrs[wc_idx][0] * dt_rs
        dy_am_step = wave_center.velocity_amrs[wc_idx][1] * dt_rs
        dz_am_step = wave_center.velocity_amrs[wc_idx][2] * dt_rs

        di = dx_am_step / dx_am
        dj = dy_am_step / dx_am
        dk = dz_am_step / dx_am

        wave_center.position_float[wc_idx][0] += di
        wave_center.position_float[wc_idx][1] += dj
        wave_center.position_float[wc_idx][2] += dk

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
    annihilation_threshold: ti.f32,  # type: ignore
):
    """
    Annihilation naturally occurs from wave physics, but needs numerical precision check.
    Detect and handle particle annihilation when WCs converge to same position.

    When two wave centers with opposite phase (180°) attract and meet:
    1. Their waves cancel perfectly (handled by wave precision rounding)
    2. Snap both WCs to exact same position
    3. Zero velocities and mark inactive

    This ensures annihilation is permanent - no wave reappearance from micro-motion.
    Numerical precision limits may cause slight separation otherwise.

    Args:
        wave_center: WaveCenter instance with position/velocity fields
        annihilation_threshold: Distance in grid units to trigger annihilation
    """
    phase_tolerance = ti.cast(0.17, ti.f32)  # ~10° tolerance
    pi = ti.cast(3.14159265359, ti.f32)

    # Check all pairs (i, j) where i < j
    for i in range(wave_center.num_sources):
        for j in range(i + 1, wave_center.num_sources):
            # Skip if either WC is already inactive
            if wave_center.active[i] == 0 or wave_center.active[j] == 0:
                continue

            # Check if phases are opposite (differ by ~π)
            phase_diff = ti.abs(wave_center.offset[i] - wave_center.offset[j])
            phase_diff_normalized = ti.abs(phase_diff - pi)
            if phase_diff_normalized > phase_tolerance:
                continue

            # Calculate distance between WCs (grid units)
            dx = wave_center.position_float[i][0] - wave_center.position_float[j][0]
            dy = wave_center.position_float[i][1] - wave_center.position_float[j][1]
            dz = wave_center.position_float[i][2] - wave_center.position_float[j][2]
            distance = ti.sqrt(dx * dx + dy * dy + dz * dz)

            if distance < annihilation_threshold:
                # Snap both WCs to midpoint
                mid_x = ti.round(
                    (wave_center.position_float[i][0] + wave_center.position_float[j][0]) / 2.0
                )
                mid_y = ti.round(
                    (wave_center.position_float[i][1] + wave_center.position_float[j][1]) / 2.0
                )
                mid_z = ti.round(
                    (wave_center.position_float[i][2] + wave_center.position_float[j][2]) / 2.0
                )

                for idx in ti.static(range(2)):
                    wc = i if idx == 0 else j
                    wave_center.position_float[wc][0] = mid_x
                    wave_center.position_float[wc][1] = mid_y
                    wave_center.position_float[wc][2] = mid_z
                    wave_center.velocity_amrs[wc][0] = 0.0
                    wave_center.velocity_amrs[wc][1] = 0.0
                    wave_center.velocity_amrs[wc][2] = 0.0
                    wave_center.position_grid[wc][0] = ti.cast(mid_x, ti.i32)
                    wave_center.position_grid[wc][1] = ti.cast(mid_y, ti.i32)
                    wave_center.position_grid[wc][2] = ti.cast(mid_z, ti.i32)
                    wave_center.active[wc] = 0
