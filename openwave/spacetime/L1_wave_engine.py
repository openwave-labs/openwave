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
base_frequency_rHz = base_frequency * constants.RONTOSECOND  # in rHz (1e-27 Hz)
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
    center_x = ti.cast(wave_field.nx, ti.f32) / 2.0
    center_y = ti.cast(wave_field.ny, ti.f32) / 2.0
    center_z = ti.cast(wave_field.nz, ti.f32) / 2.0

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega = 2.0 * ti.math.pi * base_frequency_rHz  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength / wave_field.dx
    wave_number = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Create radial sinusoidal displacement pattern (interior points only)
    # Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Distance from center (in grid indices)
        dx = ti.cast(i, ti.f32) - center_x
        dy = ti.cast(j, ti.f32) - center_y
        dz = ti.cast(k, ti.f32) - center_z
        r_grid = ti.sqrt(dx * dx + dy * dy + dz * dz)  # in grid indices

        # Amplitude = base if r < λ, 0 otherwise, oscillates radially
        amplitude_am_at_r = base_amplitude_am if r_grid < wavelength_grid else 0.0

        # Simple sinusoidal radial pattern
        # Outward displacement from center source: A(r)·cos(ωt - kr)
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        disp = amplitude_am_at_r * ti.cos(omega * 0 - wave_number * r_grid)  # t0 initial condition
        disp_old = amplitude_am_at_r * ti.cos(omega * -dt_rs - wave_number * r_grid)

        # Apply both displacements (in attometers)
        wave_field.displacement_am[i, j, k] = disp  # at time t=0
        wave_field.displacement_old_am[i, j, k] = disp_old  # at time t=-1


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
    center_x = ti.cast(wave_field.nx, ti.f32) / 2.0
    center_y = ti.cast(wave_field.ny, ti.f32) / 2.0
    center_z = ti.cast(wave_field.nz, ti.f32) / 2.0

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega = 2.0 * ti.math.pi * base_frequency_rHz  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength / wave_field.dx
    wave_number = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Create radial sinusoidal displacement pattern (interior points only)
    # Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Distance from center (in grid indices)
        dx = ti.cast(i, ti.f32) - center_x
        dy = ti.cast(j, ti.f32) - center_y
        dz = ti.cast(k, ti.f32) - center_z
        r_grid = ti.sqrt(dx * dx + dy * dy + dz * dz)  # in grid indices

        # Amplitude decreases with distance, oscillates radially
        r_safe_am = ti.max(r_grid, wavelength_grid)  # minimum 1 λ from source
        amplitude_falloff = wavelength_grid / r_safe_am  # Avoids singularity at r=0
        amplitude_am_at_r = base_amplitude_am * amplitude_falloff

        # Simple sinusoidal radial pattern
        # Outward displacement from center source: A(r)·cos(ωt - kr)
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        disp = amplitude_am_at_r * ti.cos(omega * 0 - wave_number * r_grid)  # t0 initial condition
        disp_old = amplitude_am_at_r * ti.cos(omega * -dt_rs - wave_number * r_grid)

        # Apply both displacements (in attometers)
        wave_field.displacement_am[i, j, k] = disp  # at time t=0
        wave_field.displacement_old_am[i, j, k] = disp_old  # at time t=-1


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
    center_x = ti.cast(wave_field.nx, ti.f32) / 2.0
    center_y = ti.cast(wave_field.ny, ti.f32) / 2.0
    center_z = ti.cast(wave_field.nz, ti.f32) / 2.0

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega = 2.0 * ti.math.pi * base_frequency_rHz  # angular frequency (rad/rs)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = base_wavelength / wave_field.dx
    wave_number = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Create radial sinusoidal displacement pattern (interior points only)
    # Skip boundaries to enforce Dirichlet boundary conditions (ψ=0 at edges)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Distance from center (in grid indices)
        dx = ti.cast(i, ti.f32) - center_x
        dy = ti.cast(j, ti.f32) - center_y
        dz = ti.cast(k, ti.f32) - center_z
        r_grid = ti.sqrt(dx * dx + dy * dy + dz * dz)  # in grid indices

        # Simple sinusoidal radial pattern
        # Outward displacement from center source: A(r)·cos(ωt - kr)
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        disp = base_amplitude_am * ti.cos(omega * 0 - wave_number * r_grid)  # t0 initial condition
        disp_old = base_amplitude_am * ti.cos(omega * -dt_rs - wave_number * r_grid)

        # Apply both displacements (in attometers)
        wave_field.displacement_am[i, j, k] = disp  # at time t=0
        wave_field.displacement_old_am[i, j, k] = disp_old  # at time t=-1


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

    # Grid center position (in grid indices)
    center_x = ti.cast(wave_field.nx, ti.f32) / 2.0
    center_y = ti.cast(wave_field.ny, ti.f32) / 2.0
    center_z = ti.cast(wave_field.nz, ti.f32) / 2.0

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

    # Apply Gaussian displacement (interior points only, boundaries stay at ψ=0)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Squared distance from center (in grid indices)
        dx = ti.cast(i, ti.f32) - center_x
        dy = ti.cast(j, ti.f32) - center_y
        dz = ti.cast(k, ti.f32) - center_z
        r_squared = dx * dx + dy * dy + dz * dz

        # Gaussian envelope: G(r) = exp(-r²/(2σ²))
        gaussian = ti.exp(-r_squared / (2.0 * sigma_grid * sigma_grid))

        wave_field.displacement_am[i, j, k] = A_am * gaussian

    # Set old displacement equal to current (zero initial velocity: ∂ψ/∂t = 0)
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        wave_field.displacement_old_am[i, j, k] = wave_field.displacement_am[i, j, k]


@ti.kernel
def charge_oscillator(
    wave_field: ti.template(),  # type: ignore
    elapsed_t_rs: ti.f32,  # type: ignore
):
    """
    Apply harmonic oscillation to a spherical volume at the grid center.

    Creates a uniform displacement within a spherical region using:
        ψ(t) = A·cos(ωt)

    The oscillator acts as a coherent wave source, with all voxels inside
    the sphere oscillating in phase.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        elapsed_t_rs: Elapsed simulation time (rs)
    """
    # Compute angular frequency (ω = 2πf) for temporal phase variation
    omega = 2.0 * ti.math.pi * base_frequency_rHz  # angular frequency (rad/rs)

    # Define center oscillator radius
    source_radius = int(0.05 * wave_field.max_grid_size)  # fraction of max grid size

    # Find center position (in grid indices)
    cx = wave_field.nx // 2
    cy = wave_field.ny // 2
    cz = wave_field.nz // 2

    # Apply oscillating displacement at center source_radius only
    # Harmonic motion: A·cos(ωt), positive = expansion, negative = compression
    for i, j, k in ti.ndrange(
        (cx - source_radius, cx + source_radius + 1),
        (cy - source_radius, cy + source_radius + 1),
        (cz - source_radius, cz + source_radius + 1),
    ):
        # Compute distance squared inline (Taichi kernels need direct computation)
        dist_sq = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
        if dist_sq <= source_radius**2:
            wave_field.displacement_am[i, j, k] = base_amplitude_am * ti.cos(omega * elapsed_t_rs)


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
    ∇²ψ[i,j,k] = (ψ[i±1] + ψ[i,j±1] + ψ[i,j,k±1] - 6ψ[i,j,k]) / dx²

    Args:
        i, j, k: Voxel indices (must be interior: 0 < i,j,k < n-1)

    Returns:
        Laplacian in units [1/am] = [am/am²]
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
    of wavefronts, and energy conservation.

    Discrete Form (Leap-Frog/Verlet):
        ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
        where ∇²ψ = (neighbors_sum - 6·center) / dx²

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        trackers: WaveTrackers instance for tracking wave properties
        c_amrs: Effective wave speed after slow-motion factor (am/rs)
        dt_rs: Time step size (rs)
        elapsed_t_rs: Elapsed simulation time (rs)

    Known issue: Metal GPU backend is unstable for grids >720³. Use CPU for larger grids.
    """

    # Update all interior voxels only (boundaries stay at ψ=0)
    # Direct range specification avoids conditional branching on GPU
    for i, j, k in ti.ndrange(
        (1, wave_field.nx - 1), (1, wave_field.ny - 1), (1, wave_field.nz - 1)
    ):
        # Compute Laplacian (returns [1/am])
        laplacian_am = compute_laplacian_am(wave_field, i, j, k)

        # Leap-frog update
        # Standard form: ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
        wave_field.displacement_new_am[i, j, k] = (
            2.0 * wave_field.displacement_am[i, j, k]
            - wave_field.displacement_old_am[i, j, k]
            + (c_amrs * dt_rs) ** 2 * laplacian_am
        )

        # WAVE TRACKERS: compute tracked properties during propagation (A & f)
        # Track AMPLITUDE envelope using exponential moving average (EMA)
        # EMA formula: A_new = α * |ψ| + (1 - α) * A_old
        # α controls adaptation speed: higher = faster response, lower = smoother
        # Asymmetric EMA: fast attack (α=0.3) when rising, slow decay (α=0.02) when falling
        # This captures peaks quickly but smooths out the decay
        # TODO: 2 polarities tracked: longitudinal & transverse
        disp_mag = ti.abs(wave_field.displacement_am[i, j, k])
        current_amp = trackers.amplitude_am[i, j, k][0]
        alpha = 0.3 if disp_mag > current_amp else 0.02
        trackers.amplitude_am[i, j, k][0] = alpha * disp_mag + (1.0 - alpha) * current_amp

        # Track FREQUENCY via zero-crossing detection
        # Detect positive-going zero crossing (negative → positive transition)
        # One full period_rs = time between consecutive positive zero crossings
        # More robust than peak detection since it doesn't depend on amplitude envelope
        prev_disp = wave_field.displacement_old_am[i, j, k]
        curr_disp = wave_field.displacement_am[i, j, k]
        if prev_disp < 0.0 and curr_disp >= 0.0:  # Zero crossing detected
            period_rs = elapsed_t_rs - trackers.last_crossing[i, j, k]
            if period_rs > dt_rs * 2:  # Noise filter: ignore if too soon
                trackers.frequency_rHz[i, j, k] = 1.0 / period_rs  # in rHz
            trackers.last_crossing[i, j, k] = elapsed_t_rs

        # Track avg amplitude/frequency across all voxels
        # TODO: compute this as avg from total voxels for energy monitoring
        # Here we use a simple max envelope for visualization scaling
        trackers.avg_amplitude_am[None] = 2 * base_amplitude_am
        trackers.avg_frequency_rHz[None] = 2 * base_frequency_rHz  # in rHz

    # Swap time levels: old ← current, current ← new
    for i, j, k in ti.ndrange(wave_field.nx, wave_field.ny, wave_field.nz):
        wave_field.displacement_old_am[i, j, k] = wave_field.displacement_am[i, j, k]
        wave_field.displacement_am[i, j, k] = wave_field.displacement_new_am[i, j, k]


@ti.kernel
def update_flux_mesh_colors(
    wave_field: ti.template(),  # type: ignore
    trackers: ti.template(),  # type: ignore
    color_palette: ti.i32,  # type: ignore
    var_amp: ti.i32,  # type: ignore
):
    """
    Update flux mesh colors by sampling wave properties from voxel grid.

    Samples wave displacement at each Plane vertex position and maps it to a color
    using the redshift gradient (red=negative, gray=zero, blue=positive).

    This function should be called every frame after wave propagation to update
    the visualization based on current wave field state.

    Color mapping:
    - Uses get_redshift_color() from config module
    - Samples displacement at voxel centers corresponding to vertex positions
    - Maps signed displacement values to red-gray-blue gradient

    Args:
        wave_field: WaveField instance containing flux mesh fields and displacement data
        color_palette: Integer code for color palette selection
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
        amp_value = trackers.amplitude_am[i, j, center_k][0]
        value = amp_value if var_amp else disp_value
        freq_value = trackers.frequency_rHz[i, j, center_k]

        # Map displacement/ amplitude to color using selected gradient
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_frequency_rHz[None]
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_ironbow_color(
                value, -trackers.avg_amplitude_am[None], trackers.avg_amplitude_am[None]
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_redshift_color(
                value, -trackers.avg_amplitude_am[None], trackers.avg_amplitude_am[None]
            )

    # ================================================================
    # XZ Plane: Sample at y = center_j
    # ================================================================
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[i, center_j, k]
        amp_value = trackers.amplitude_am[i, center_j, k][0]
        value = amp_value if var_amp else disp_value
        freq_value = trackers.frequency_rHz[i, center_j, k]

        # Map displacement/ amplitude to color using selected gradient
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_frequency_rHz[None]
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_ironbow_color(
                value, -trackers.avg_amplitude_am[None], trackers.avg_amplitude_am[None]
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_redshift_color(
                value, -trackers.avg_amplitude_am[None], trackers.avg_amplitude_am[None]
            )

    # ================================================================
    # YZ Plane: Sample at x = center_i
    # ================================================================
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[center_i, j, k]
        amp_value = trackers.amplitude_am[center_i, j, k][0]
        value = amp_value if var_amp else disp_value
        freq_value = trackers.frequency_rHz[center_i, j, k]

        # Map displacement/ amplitude to color using selected gradient
        if color_palette == 3:  # blueprint
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_blueprint_color(
                freq_value, 0.0, trackers.avg_frequency_rHz[None]
            )
        elif color_palette == 2:  # ironbow
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_ironbow_color(
                value, -trackers.avg_amplitude_am[None], trackers.avg_amplitude_am[None]
            )
        else:  # default to redshift (palette 1)
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_redshift_color(
                value, -trackers.avg_amplitude_am[None], trackers.avg_amplitude_am[None]
            )


# TODO: migrate to numerical analysis module
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
    nx = wave_field.nx
    x_indices = np.arange(nx)
    displacements_L = np.zeros(nx)
    displacements_T = np.zeros(nx)

    # Sample longitudinal displacement values
    for i in range(nx):
        displacements_L[i] = wave_field.displacement_am[i, center_j, center_k]
        # displacements_L[i] = wave_field.displacement_old_am[i, center_j, center_k]
        displacements_T[i] = 0.0

    # Calculate distance from center in grid indices
    center_x = nx / 2.0
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
