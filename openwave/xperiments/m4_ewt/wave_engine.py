"""
ENERGY-WAVE ENGINE

ON VECTOR-WAVE MODEL

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
rho_qgam = constants.MEDIUM_DENSITY_QGAM  # qg/am³, for energy computation in scaled units


# ================================================================
# WAVE PROPAGATION ENGINE
# ================================================================


@ti.kernel
def select_voxels(
    wave_field: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
):
    """
    Select voxels for optimized wave computation (neighbor-only mode).

    Populates wave_field.selected_voxels with the center + 6 neighbors
    of each active wave center. Used for force gradient calculations
    when flux mesh visualization is disabled.

    Args:
        wave_field: WaveField instance with selected_voxels buffer
        wave_center: WaveCenter instance with source positions
    """
    # Reset counter
    wave_field.num_selected_voxels[None] = 0

    # 6-neighbor offsets: +x, -x, +y, -y, +z, -z (plus center = 0,0,0)
    # Using static unrolling for performance
    for wc_idx in range(wave_center.num_sources):
        if wave_center.active[wc_idx] == 0:
            continue

        # Get wave center grid position
        cx = wave_center.position_grid[wc_idx][0]
        cy = wave_center.position_grid[wc_idx][1]
        cz = wave_center.position_grid[wc_idx][2]

        # Add center voxel (offset 0)
        idx = ti.atomic_add(wave_field.num_selected_voxels[None], 1)
        if idx < wave_field.max_selected_voxels:
            wave_field.selected_voxels[idx] = ti.Vector([cx, cy, cz])

        # Add 6 neighbors with boundary clamping
        # +x neighbor
        idx = ti.atomic_add(wave_field.num_selected_voxels[None], 1)
        if idx < wave_field.max_selected_voxels:
            wave_field.selected_voxels[idx] = ti.Vector(
                [ti.min(cx + 1, wave_field.nx - 1), cy, cz]
            )

        # -x neighbor
        idx = ti.atomic_add(wave_field.num_selected_voxels[None], 1)
        if idx < wave_field.max_selected_voxels:
            wave_field.selected_voxels[idx] = ti.Vector([ti.max(cx - 1, 0), cy, cz])

        # +y neighbor
        idx = ti.atomic_add(wave_field.num_selected_voxels[None], 1)
        if idx < wave_field.max_selected_voxels:
            wave_field.selected_voxels[idx] = ti.Vector(
                [cx, ti.min(cy + 1, wave_field.ny - 1), cz]
            )

        # -y neighbor
        idx = ti.atomic_add(wave_field.num_selected_voxels[None], 1)
        if idx < wave_field.max_selected_voxels:
            wave_field.selected_voxels[idx] = ti.Vector([cx, ti.max(cy - 1, 0), cz])

        # +z neighbor
        idx = ti.atomic_add(wave_field.num_selected_voxels[None], 1)
        if idx < wave_field.max_selected_voxels:
            wave_field.selected_voxels[idx] = ti.Vector(
                [cx, cy, ti.min(cz + 1, wave_field.nz - 1)]
            )

        # -z neighbor
        idx = ti.atomic_add(wave_field.num_selected_voxels[None], 1)
        if idx < wave_field.max_selected_voxels:
            wave_field.selected_voxels[idx] = ti.Vector([cx, cy, ti.max(cz - 1, 0)])


@ti.func
def compute_voxel_wave(
    i: ti.i32,  # type: ignore
    j: ti.i32,  # type: ignore
    k: ti.i32,  # type: ignore
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
    k_grid: ti.f32,  # type: ignore
    temporal_phase: ti.f32,  # type: ignore
):
    """
    Compute wave displacement and envelope for a single voxel.

    This function contains the core wave physics computation shared by both
    full-grid and neighbor-only iteration modes.

    Weighted Partial Standing Wave:
        ψ(r,t) = A · [w·sin(kr+ωt+φ) + sin(kr-ωt-φ)] / kr · r̂

    Components:
        Out-wave: sin(kr - ωt - φ) / kr  (nodes move outward, always present)
        In-wave:  w(r) · sin(kr + ωt + φ) / kr  (nodes move inward, fades with distance)

    Weight function w(r,λ):
        w = 1 / (1 + (r / (1.25λ))⁸)  — sharp Lorentzian rolloff

    Physical Properties:
        - Center (kr ≈ 0): pure standing wave, sinc → 2·cos(ωt+φ), finite at r=0
        - Transition zone: partially standing, nodes drift outward
        - Far field (r >> 1.25λ): pure traveling wave, 1/kr falloff
        - Vector displacement along radial direction r̂ (longitudinal wave)
        - Superposition of multiple wave centers supported

    Wave Trackers Updated:
        - amp_local_emarms_am: RMS amplitude via EMA on ψ² (for energy/force gradients)
        - freq_local_cross_rHz: Frequency via zero-crossing detection with EMA smoothing

    Args:
        i, j, k: Voxel grid indices
        wave_field: WaveField instance
        trackers: WaveTrackers instance
        wave_center: WaveCenter instance
        k_grid: Angular wave number (radians per grid index)
        temporal_phase: Current temporal phase (ω·t)
    """
    # Reset before accumulation
    prev_disp = wave_field.displacement_am[i, j, k]
    wave_field.displacement_am[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    # Wavelength in grid units (for standing wave transition zone)
    wavelength_grid = 2.0 * ti.math.pi / k_grid

    # ================================================================
    # WAVE PROPAGATION: Update voxels using wave functions
    # ================================================================
    # Loop over wave-centers (wave superposition principle causing interference)
    for wc_idx in range(wave_center.num_sources):
        # Skip inactive (annihilated) WCs
        if wave_center.active[wc_idx] == 0:
            continue

        # Get direction from voxel to wave center (normalized vector)
        dir_vec = [i, j, k] - wave_center.position_grid[wc_idx]
        r_grid = dir_vec.norm() + 1e-10
        direction = dir_vec / r_grid
        # Center voxel: direction is zero (undefined for point source),
        # use unit-z fallback so scalar oscillator reaches amplitude tracker
        direction += ti.cast(r_grid < 0.5, ti.f32) * ti.Vector([0.0, 0.0, 1.0])

        # Spatial phase: φ = k·r, creates spherical wave fronts, dimensionless, in radians
        spatial_phase = k_grid * r_grid

        # Cache source-specific phase offset
        source_offset = wave_center.offset[wc_idx]

        # ================================================================
        # WEIGHTED PARTIAL STANDING WAVE
        # ================================================================
        # Superposition of in-wave + out-wave, where the in-wave fades with distance
        # Near the source: standing wave (in-wave + out-wave interfere)
        # Far from source: pure traveling wave (in-wave fades)
        #
        # ψ(r,t) = A · [w·sin(kr+ωt+φ) + sin(kr-ωt-φ)] / kr
        #
        # Weight(r,λ): smooth decay from 1 → 0, controls standing → traveling transition
        # Standing limit (w=1, r→0): 2·cos(ωt + φ) via sinc normalization
        # Traveling limit (w=0): sin(kr - ωt - φ) / kr
        # ================================================================
        # In-wave weight: sharp Lorentzian rolloff at 1.25λ
        transition = 1 + 1 / 4  # number of wavelengths (λ)
        weight = 1.0 / (1.0 + (r_grid / (transition * wavelength_grid)) ** 8)

        # Weighted partial standing wave oscillator
        oscillator = ti.select(
            r_grid < 0.5,  # center voxel: analytical limit
            2.0 * ti.cos(temporal_phase + source_offset),  # standing wave limit: 2·cos(ωt + φ)
            (
                weight * ti.sin(spatial_phase + (temporal_phase + source_offset))  # in-wave
                + ti.sin(spatial_phase - (temporal_phase + source_offset))  # out-wave
            )
            / spatial_phase,  # 1/kr decay (sinc envelope)
        )

        # Accumulate this source's contribution (wave superposition)
        wave_field.displacement_am[i, j, k] += (
            base_amplitude_am * wave_field.scale_factor * oscillator
        ) * direction

    # TODO: consider precision rounding to ensure perfect cancellation
    # Precision rounding to ensure wave cancellation
    # Critical for opposing phase sources (180°) that should annihilate
    # Floating-point: (+1.250001) + (-1.249999) = 0.000002 (imperfect cancel)
    # With rounding: (+1.2500) + (-1.2500) = 0.0 (perfect cancel)
    # precision = ti.cast(1e4, ti.f32)  # round to 4 decimal places
    # wave_field.displacement_am[i, j, k] = (
    #     ti.round(wave_field.displacement_am[i, j, k] * precision) / precision
    # )

    curr_disp = wave_field.displacement_am[i, j, k]

    # ================================================================
    # WAVE-TRACKERS: Update amplitude and frequency trackers for visualization and forces
    # ================================================================
    # # PEAK AMPLITUDE tracking
    # ti.atomic_max(trackers.ampL_local_peak_am[i, j, k], curr_disp)
    # decay_factor_peak = ti.cast(0.999, ti.f32)  # ~100 frames to ~37%, ~230 to ~10%
    # trackers.ampL_local_peak_am[i, j, k] = (
    #     trackers.ampL_local_peak_am[i, j, k] * decay_factor_peak
    # )

    # TODO: review EMS method for amplitude tracking (values compared to peak amp for force)
    # TODO: review how vector displacement (has direction component) is handled in RMS calculation
    # RMS AMPLITUDE tracking via EMA on ψ² (squared displacement)
    # Running RMS: tracks √⟨ψ²⟩ - the energy-equivalent amplitude (Energy ∝ ψ²)
    # Used for: energy calculation, force gradients, visualization scaling
    # Physics: particles respond to time-averaged energy density, not
    # instantaneous displacement (inertia acts as low-pass filter at ~10²⁵ Hz)
    # EMA on ψ²: rms² = α * ψ² + (1 - α) * rms²_old, then rms = √(rms²)
    # α controls adaptation speed: higher = faster response, lower = smoother
    # RMS amplitude
    disp2 = wave_field.displacement_am[i, j, k].norm() ** 2
    current_rms2 = trackers.amp_local_emarms_am[i, j, k] ** 2
    alpha_rms = 0.005  # EMA smoothing factor for RMS tracking
    new_rms2 = alpha_rms * disp2 + (1.0 - alpha_rms) * current_rms2
    new_amp = ti.sqrt(new_rms2)

    # Unconditional decay clears trails from moving sources
    # Active regions counteract decay via EMA update from strong displacement
    # Stale regions (waves propagated away) decay to zero over time
    decay_factor = ti.cast(0.99, ti.f32)  # ~100 frames to ~37%, ~230 to ~10%
    trackers.amp_local_emarms_am[i, j, k] = new_amp * decay_factor

    # TODO: review new frequency tracking method
    # FREQUENCY tracking, via zero-crossing detection with EMA smoothing
    # Detect positive-going zero crossing (negative → positive transition)
    # Period = time between consecutive positive zero crossings
    # More robust than peak detection since it's amplitude-independent
    # EMA smoothing: f_new = α * f_measured + (1 - α) * f_old
    # α controls adaptation speed: higher = faster response, lower = smoother

    # if prev_disp < 0.0 and curr_disp >= 0.0:  # Zero crossing detected
    #     period_rs = elapsed_t_rs - trackers.last_crossing[i, j, k]
    #     if period_rs > dt_rs * 2:  # Filter out spurious crossings
    #         measured_freq = 1.0 / period_rs  # in rHz
    #         current_freq = trackers.freq_local_cross_rHz[i, j, k]
    #         alpha_freq = 0.05  # EMA smoothing factor for frequency
    #         trackers.freq_local_cross_rHz[i, j, k] = (
    #             alpha_freq * measured_freq + (1.0 - alpha_freq) * current_freq
    #         )
    #     trackers.last_crossing[i, j, k] = elapsed_t_rs

    # # Unconditional frequency decay (counteracted by zero-crossing updates in active regions)
    # trackers.freq_local_cross_rHz[i, j, k] *= decay_factor

    trackers.freq_local_cross_rHz[i, j, k] = (
        base_frequency_rHz  # TODO: fixed frequency for testing, replace with above method for dynamic frequency
    )

    # ================================================================
    # LOCAL ENERGY PER VOXEL: E = ρ · V · (f · A)² in aJ (attojoules)
    # F = -∇E (force = negative energy gradient, computed in force_motion)
    # ================================================================
    # rho_qgam (qg/am³), dx_am³ (am³), f_rHz (1/rs), rms_am (am)
    # Internal units: qg·am²/rs² × 1000 → aJ
    dx_am = wave_field.dx_am
    amp_am = trackers.amp_local_emarms_am[i, j, k]
    trackers.energy_local_aJ[i, j, k] = (
        rho_qgam * dx_am**3 * (base_frequency_rHz * amp_am) ** 2 * constants.INTERNAL_ENERGY_TO_AJ
    )


@ti.kernel
def propagate_wave_neighbors(
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
    elapsed_t_rs: ti.f32,  # type: ignore
    num_selected: ti.i32,  # type: ignore
):
    """
    Compute wave displacement for selected neighbor voxels only (optimized mode).

    Processes only the voxels selected by select_voxels() - the center + 6 neighbors
    of each active wave center. Used for force gradient calculations when flux mesh
    visualization is disabled.

    Performance: ~7*N voxels instead of nx*ny*nz (massive gain for large grids).

    Args:
        wave_field: WaveField instance with selected_voxels populated
        trackers: WaveTrackers instance for amplitude envelope
        wave_center: WaveCenter instance with source positions
        num_selected: Number of selected voxels to process
    """
    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega_rs = (
        2.0 * ti.math.pi * base_frequency_rHz / wave_field.scale_factor
    )  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength * wave_field.scale_factor / wave_field.dx
    k_grid = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Temporal phase: φ = ω·t, oscillatory in time
    temporal_phase = omega_rs * elapsed_t_rs

    # Iterate only over selected voxels (neighbors of wave centers)
    for sel_idx in range(num_selected):
        i = wave_field.selected_voxels[sel_idx][0]
        j = wave_field.selected_voxels[sel_idx][1]
        k = wave_field.selected_voxels[sel_idx][2]
        compute_voxel_wave(i, j, k, wave_field, trackers, wave_center, k_grid, temporal_phase)


@ti.kernel
def propagate_wave_full(
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    wave_center: ti.template(),  # type: ignore
    elapsed_t_rs: ti.f32,  # type: ignore
):
    """
    Compute wave displacement using Wolff-LaFreniere analytical wave equation (full grid).

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        trackers: WaveTrackers instance for tracking wave properties
        wave_center: WaveCenter instance with source positions and phase offsets
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

    # FULL GRID MODE: Compute all voxels (needed for flux mesh visualization)
    for i, j, k in ti.ndrange(nx, ny, nz):
        compute_voxel_wave(i, j, k, wave_field, trackers, wave_center, k_grid, temporal_phase)


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
_slice_xy_energy = None
_slice_xz_amp = None
_slice_xz_freq = None
_slice_xz_energy = None
_slice_yz_amp = None
_slice_yz_freq = None
_slice_yz_energy = None


@ti.kernel
def _copy_slice_xy(
    trackers: ti.template(),  # type: ignore
    slice_amp: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    slice_energy: ti.template(),  # type: ignore
    mid_z: ti.i32,  # type: ignore
):
    """Copy center XY slice (fixed z) to 2D buffer."""
    for i, j in slice_amp:
        slice_amp[i, j] = trackers.amp_local_emarms_am[i, j, mid_z]
        slice_freq[i, j] = trackers.freq_local_cross_rHz[i, j, mid_z]
        slice_energy[i, j] = trackers.energy_local_aJ[i, j, mid_z]


@ti.kernel
def _copy_slice_xz(
    trackers: ti.template(),  # type: ignore
    slice_amp: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    slice_energy: ti.template(),  # type: ignore
    mid_y: ti.i32,  # type: ignore
):
    """Copy center XZ slice (fixed y) to 2D buffer."""
    for i, k in slice_amp:
        slice_amp[i, k] = trackers.amp_local_emarms_am[i, mid_y, k]
        slice_freq[i, k] = trackers.freq_local_cross_rHz[i, mid_y, k]
        slice_energy[i, k] = trackers.energy_local_aJ[i, mid_y, k]


@ti.kernel
def _copy_slice_yz(
    trackers: ti.template(),  # type: ignore
    slice_amp: ti.template(),  # type: ignore
    slice_freq: ti.template(),  # type: ignore
    slice_energy: ti.template(),  # type: ignore
    mid_x: ti.i32,  # type: ignore
):
    """Copy center YZ slice (fixed x) to 2D buffer."""
    for j, k in slice_amp:
        slice_amp[j, k] = trackers.amp_local_emarms_am[mid_x, j, k]
        slice_freq[j, k] = trackers.freq_local_cross_rHz[mid_x, j, k]
        slice_energy[j, k] = trackers.energy_local_aJ[mid_x, j, k]


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
    global _slice_xy_amp, _slice_xy_freq, _slice_xy_energy
    global _slice_xz_amp, _slice_xz_freq, _slice_xz_energy
    global _slice_yz_amp, _slice_yz_freq, _slice_yz_energy

    nx, ny, nz = wave_field.nx, wave_field.ny, wave_field.nz

    # Initialize slice buffers once
    if _slice_xy_amp is None:
        _slice_xy_amp = ti.field(dtype=ti.f32, shape=(nx, ny))
        _slice_xy_freq = ti.field(dtype=ti.f32, shape=(nx, ny))
        _slice_xy_energy = ti.field(dtype=ti.f32, shape=(nx, ny))
        _slice_xz_amp = ti.field(dtype=ti.f32, shape=(nx, nz))
        _slice_xz_freq = ti.field(dtype=ti.f32, shape=(nx, nz))
        _slice_xz_energy = ti.field(dtype=ti.f32, shape=(nx, nz))
        _slice_yz_amp = ti.field(dtype=ti.f32, shape=(ny, nz))
        _slice_yz_freq = ti.field(dtype=ti.f32, shape=(ny, nz))
        _slice_yz_energy = ti.field(dtype=ti.f32, shape=(ny, nz))

    # Copy 3 center slices to 2D buffers (parallel kernels)
    mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2
    _copy_slice_xy(trackers, _slice_xy_amp, _slice_xy_freq, _slice_xy_energy, mid_z)
    _copy_slice_xz(trackers, _slice_xz_amp, _slice_xz_freq, _slice_xz_energy, mid_y)
    _copy_slice_yz(trackers, _slice_yz_amp, _slice_yz_freq, _slice_yz_energy, mid_x)

    # Transfer 2D slices to CPU for numpy operations
    # Exclude boundary voxels
    xy_amp = _slice_xy_amp.to_numpy()[1:-1, 1:-1]
    xy_freq = _slice_xy_freq.to_numpy()[1:-1, 1:-1]
    xy_energy = _slice_xy_energy.to_numpy()[1:-1, 1:-1]
    xz_amp = _slice_xz_amp.to_numpy()[1:-1, 1:-1]
    xz_freq = _slice_xz_freq.to_numpy()[1:-1, 1:-1]
    xz_energy = _slice_xz_energy.to_numpy()[1:-1, 1:-1]
    yz_amp = _slice_yz_amp.to_numpy()[1:-1, 1:-1]
    yz_freq = _slice_yz_freq.to_numpy()[1:-1, 1:-1]
    yz_energy = _slice_yz_energy.to_numpy()[1:-1, 1:-1]

    # Compute RMS amplitude: √(⟨A²⟩) for correct energy weighting
    # amp_local_emarms_am contains per-voxel RMS values, square them for energy
    total_amp_squared = (xy_amp**2).sum() + (xz_amp**2).sum() + (yz_amp**2).sum()
    total_freq = xy_freq.sum() + xz_freq.sum() + yz_freq.sum()
    total_energy = xy_energy.sum() + xz_energy.sum() + yz_energy.sum()
    n_samples = xy_amp.size + xz_amp.size + yz_amp.size

    trackers.amp_global_emarms_am[None] = float(np.sqrt(total_amp_squared / n_samples))
    trackers.freq_global_avg_rHz[None] = float(total_freq / n_samples)
    trackers.energy_global_avg_aJ[None] = float(total_energy / n_samples)


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
        disp_value = wave_field.displacement_am[i, j, wave_field.fm_plane_z_idx].norm()
        amp_value = trackers.amp_local_emarms_am[i, j, wave_field.fm_plane_z_idx]
        freq_value = trackers.freq_local_cross_rHz[i, j, wave_field.fm_plane_z_idx]
        energy_value = trackers.energy_local_aJ[i, j, wave_field.fm_plane_z_idx]
        univ_edge_z = wave_field.universe_size_am[2]

        # Map value to color/vertex using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if wave_menu == 1:  # orange
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_orange_color(
                disp_value, 0, trackers.amp_global_emarms_am[None] * 4
            )
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                disp_value / univ_edge_z * warp_mesh
                + wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )
        elif wave_menu == 2:  # ironbow
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_ironbow_color(
                amp_value, 0, trackers.amp_global_emarms_am[None] * 2
            )
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                amp_value / univ_edge_z * warp_mesh
                + wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )
        elif wave_menu == 3:  # blueprint
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.freq_global_avg_rHz[None] * 2
            )
            wave_field.fluxmesh_xy_vertices[i, j][2] = freq_value / trackers.freq_global_avg_rHz[
                None
            ] / 3000 * warp_mesh + wave_field.flux_mesh_planes[2] * (
                wave_field.nz / wave_field.max_grid_size
            )
        elif wave_menu == 4:  # ironbow
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_ironbow_color(
                energy_value, 0.0, trackers.energy_global_avg_aJ[None] * 2
            )
            wave_field.fluxmesh_xy_vertices[i, j][2] = (
                energy_value / univ_edge_z * warp_mesh
                + wave_field.flux_mesh_planes[2] * (wave_field.nz / wave_field.max_grid_size)
            )

    # ================================================================
    # XZ Plane: Sample at y = fm_plane_y_idx
    # ================================================================
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[i, wave_field.fm_plane_y_idx, k].norm()
        amp_value = trackers.amp_local_emarms_am[i, wave_field.fm_plane_y_idx, k]
        freq_value = trackers.freq_local_cross_rHz[i, wave_field.fm_plane_y_idx, k]
        energy_value = trackers.energy_local_aJ[i, wave_field.fm_plane_y_idx, k]
        univ_edge_y = wave_field.universe_size_am[1]

        # Map value to color/vertex using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if wave_menu == 1:  # orange
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_orange_color(
                disp_value, 0, trackers.amp_global_emarms_am[None] * 4
            )
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                disp_value / univ_edge_y * warp_mesh
                + wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )
        elif wave_menu == 2:  # ironbow
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_ironbow_color(
                amp_value, 0, trackers.amp_global_emarms_am[None] * 2
            )
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                amp_value / univ_edge_y * warp_mesh
                + wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )
        elif wave_menu == 3:  # blueprint
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.freq_global_avg_rHz[None] * 2
            )
            wave_field.fluxmesh_xz_vertices[i, k][1] = freq_value / trackers.freq_global_avg_rHz[
                None
            ] / 3000 * warp_mesh + wave_field.flux_mesh_planes[1] * (
                wave_field.ny / wave_field.max_grid_size
            )
        elif wave_menu == 4:  # ironbow
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_ironbow_color(
                energy_value, 0.0, trackers.energy_global_avg_aJ[None] * 2
            )
            wave_field.fluxmesh_xz_vertices[i, k][1] = (
                energy_value / univ_edge_y * warp_mesh
                + wave_field.flux_mesh_planes[1] * (wave_field.ny / wave_field.max_grid_size)
            )

    # ================================================================
    # YZ Plane: Sample at x = fm_plane_x_idx
    # ================================================================
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[wave_field.fm_plane_x_idx, j, k].norm()
        amp_value = trackers.amp_local_emarms_am[wave_field.fm_plane_x_idx, j, k]
        freq_value = trackers.freq_local_cross_rHz[wave_field.fm_plane_x_idx, j, k]
        energy_value = trackers.energy_local_aJ[wave_field.fm_plane_x_idx, j, k]
        univ_edge_x = wave_field.universe_size_am[0]

        # Map value to color/vertex using selected gradient
        # Scale range to 2× average for headroom without saturation (allows peak visualization)
        if wave_menu == 1:  # orange
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_orange_color(
                disp_value, 0, trackers.amp_global_emarms_am[None] * 4
            )
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                disp_value / univ_edge_x * warp_mesh
                + wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )
        elif wave_menu == 2:  # ironbow
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_ironbow_color(
                amp_value, 0, trackers.amp_global_emarms_am[None] * 2
            )
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                amp_value / univ_edge_x * warp_mesh
                + wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )
        elif wave_menu == 3:  # blueprint
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.freq_global_avg_rHz[None] * 2
            )
            wave_field.fluxmesh_yz_vertices[j, k][0] = freq_value / trackers.freq_global_avg_rHz[
                None
            ] / 3000 * warp_mesh + wave_field.flux_mesh_planes[0] * (
                wave_field.nx / wave_field.max_grid_size
            )
        elif wave_menu == 4:  # ironbow
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_ironbow_color(
                energy_value, 0.0, trackers.energy_global_avg_aJ[None] * 2
            )
            wave_field.fluxmesh_yz_vertices[j, k][0] = (
                energy_value / univ_edge_x * warp_mesh
                + wave_field.flux_mesh_planes[0] * (wave_field.nx / wave_field.max_grid_size)
            )


# ================================================================
# POSITION RENDER
# ================================================================


@ti.kernel
def sample_position_to_render(
    wave_field: ti.template(),  # type: ignore
    amp_boost: ti.f32,  # type: ignore
    stride: ti.i32,  # type: ignore
    num_render: ti.i32,  # type: ignore
):
    """Sample granule positions from z flux_mesh plane with stride, writing to 1D position_render.

    Samples every `stride`-th voxel from the XY plane at the z flux mesh index,
    capping output at `num_render` particles for performance.
    """
    k = int(wave_field.flux_mesh_planes[2] * wave_field.grid_size[2])
    max_dim = ti.cast(wave_field.max_grid_size, ti.f32)
    sampled_ny = (wave_field.ny + stride - 1) // stride  # cols per sampled row

    for render_idx in range(num_render):
        si = render_idx // sampled_ny  # sampled row index
        sj = render_idx % sampled_ny  # sampled col index
        i = si * stride
        j = sj * stride
        displaced = amp_boost * wave_field.displacement_am[i, j, k] / wave_field.dx_am + ti.Vector(
            [ti.cast(i, ti.f32), ti.cast(j, ti.f32), ti.cast(k, ti.f32)]
        )
        wave_field.position_render[render_idx] = displaced / max_dim
