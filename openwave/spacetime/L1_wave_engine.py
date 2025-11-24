"""
ENERGY-WAVE ENGINE

LEVEL-1: ON FIELD-BASED METHOD

Wave Physics Engine @spacetime module. Wave dynamics and motion.
"""

import taichi as ti

from openwave.common import colormap, constants, equations, utils

# ================================================================
# Energy-Wave Oscillation Parameters
# ================================================================
base_amplitude_am = constants.EWAVE_AMPLITUDE / constants.ATTOMETER  # am, oscillation amplitude
frequency = constants.EWAVE_SPEED / constants.EWAVE_LENGTH  # Hz, energy-wave frequency
wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETER  # in attometers


@ti.kernel
def initiate_charge(
    wave_field: ti.template(),  # type: ignore
    slow_mo: ti.f32,  # type: ignore
    freq_boost: ti.f32,  # type: ignore
    dt_frame: ti.f32,  # type: ignore
):
    """
    Initialize a spherical outgoing wave pattern centered in the wave field.

    Creates a radial sinusoidal displacement pattern emanating from the grid center
    using the wave equation: A·cos(ωt - kr). Sets up both current and previous
    timestep displacements for time-stepping propagation.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        slow_mo: Slow-motion factor to reduce effective frequency (higher = slower)
        freq_boost: Frequency multiplier applied after slow-mo division
        dt_frame: Time step between frames (seconds), used to compute t=-1 state
    """

    # Find center position (in grid indices)
    center_x = ti.cast(wave_field.nx, ti.f32) / 2.0
    center_y = ti.cast(wave_field.ny, ti.f32) / 2.0
    center_z = ti.cast(wave_field.nz, ti.f32) / 2.0

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    f_slowed = frequency / slow_mo * freq_boost  # slowed frequency (1Hz * boost)
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency (rad/s)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = wavelength_am / wave_field.dx_am
    wave_number = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Create radial sinusoidal displacement pattern
    for i, j, k in ti.ndrange(wave_field.nx, wave_field.ny, wave_field.nz):
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
        disp_old = base_amplitude_am * ti.cos(omega * -dt_frame - wave_number * r_grid)

        # Apply both longitudinal and transverse displacement (in attometers)
        wave_field.displacement_am[i, j, k] = disp  # at time t=0
        wave_field.displacement_old_am[i, j, k] = disp_old  # at time t=-1


# TODO: remove amplitude falloff post propagation implementation
@ti.kernel
def initiate_falloff(
    wave_field: ti.template(),  # type: ignore
    slow_mo: ti.f32,  # type: ignore
    freq_boost: ti.f32,  # type: ignore
    dt_frame: ti.f32,  # type: ignore
):
    """
    Initialize a spherical outgoing wave with 1/r amplitude falloff.

    Similar to initiate_charge() but includes realistic amplitude decay with
    distance (λ/r falloff). Creates a radial sinusoidal displacement pattern
    where amplitude decreases inversely with distance from the source.

    Note:
        Temporary function for testing amplitude falloff visualization before
        wave propagation is implemented. See TODO marker.

    Args:
        wave_field: WaveField instance containing displacement arrays and grid info
        slow_mo: Slow-motion factor to reduce effective frequency (higher = slower)
        freq_boost: Frequency multiplier applied after slow-mo division
        dt_frame: Time step between frames (seconds), used to compute t=-1 state
    """

    # Find center position (in grid indices)
    center_x = ti.cast(wave_field.nx, ti.f32) / 2.0
    center_y = ti.cast(wave_field.ny, ti.f32) / 2.0
    center_z = ti.cast(wave_field.nz, ti.f32) / 2.0

    # Compute angular frequency (ω = 2πf) for temporal phase variation
    f_slowed = frequency / slow_mo * freq_boost  # slowed frequency (1Hz * boost)
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency (rad/s)

    # Compute angular wave number (k = 2π/λ) for spatial phase variation
    wavelength_grid = wavelength_am / wave_field.dx_am
    wave_number = 2.0 * ti.math.pi / wavelength_grid  # radians per grid index

    # Create radial sinusoidal displacement pattern
    for i, j, k in ti.ndrange(wave_field.nx, wave_field.ny, wave_field.nz):
        # Distance from center (in grid indices)
        dx = ti.cast(i, ti.f32) - center_x
        dy = ti.cast(j, ti.f32) - center_y
        dz = ti.cast(k, ti.f32) - center_z
        r_grid = ti.sqrt(dx * dx + dy * dy + dz * dz)  # in grid indices

        # Amplitude decreases with distance, oscillates radially
        r_safe_am = ti.max(r_grid, wavelength_grid)  # minimum 1 λ from source
        amplitude_falloff = wavelength_grid / r_safe_am
        amplitude_am_at_r = base_amplitude_am * amplitude_falloff

        # Simple sinusoidal radial pattern
        # Outward displacement from center source: A(r)·cos(ωt - kr)
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        disp = amplitude_am_at_r * ti.cos(omega * 0 - wave_number * r_grid)  # t0 initial condition
        disp_old = amplitude_am_at_r * ti.cos(omega * -dt_frame - wave_number * r_grid)

        # Apply both longitudinal and transverse displacement (in attometers)
        wave_field.displacement_am[i, j, k] = disp  # at time t=0
        wave_field.displacement_old_am[i, j, k] = disp_old  # at time t=-1


@ti.kernel
def update_flux_mesh_colors(
    wave_field: ti.template(),  # type: ignore
    color_palette: ti.i32,  # type: ignore
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

    # Displacement range for color scaling (in attometers)
    # TODO: In future, use exponential moving average tracker like LEVEL-0 ironbow
    # For now, use fixed range based on test pattern amplitude
    peak_amplitude_am = base_amplitude_am  # attometers (positive displacement/expansion)

    # ================================================================
    # XY Plane: Sample at z = center_k
    # ================================================================
    # Always update all planes (conditionals cause GPU branch divergence)
    for i, j in ti.ndrange(wave_field.nx, wave_field.ny):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[i, j, center_k]

        # Map displacement to color using selected gradient
        if color_palette == 2:  # blueprint
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_blueprint_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        elif color_palette == 3:  # redshift
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_redshift_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        else:  # default to ironbow (palette 1)
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_ironbow_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )

    # ================================================================
    # XZ Plane: Sample at y = center_j
    # ================================================================
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[i, center_j, k]

        # Map displacement to color using selected gradient
        if color_palette == 2:  # blueprint
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_blueprint_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        elif color_palette == 3:  # redshift
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_redshift_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        else:  # default to ironbow (palette 1)
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_ironbow_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )

    # ================================================================
    # YZ Plane: Sample at x = center_i
    # ================================================================
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_am[center_i, j, k]

        # Map displacement to color using selected gradient
        if color_palette == 2:  # blueprint
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_blueprint_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        elif color_palette == 3:  # redshift
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_redshift_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        else:  # default to ironbow (palette 1)
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_ironbow_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )


@ti.kernel
def update_flux_mesh_colors_tminus1(
    wave_field: ti.template(),  # type: ignore
    color_palette: ti.i32,  # type: ignore
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

    # Displacement range for color scaling (in attometers)
    # TODO: In future, use exponential moving average tracker like LEVEL-0 ironbow
    # For now, use fixed range based on test pattern amplitude
    peak_amplitude_am = base_amplitude_am  # attometers (positive displacement/expansion)

    # ================================================================
    # XY Plane: Sample at z = center_k
    # ================================================================
    # Always update all planes (conditionals cause GPU branch divergence)
    for i, j in ti.ndrange(wave_field.nx, wave_field.ny):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_old_am[i, j, center_k]

        # Map displacement to color using selected gradient
        if color_palette == 2:  # blueprint
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_blueprint_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        elif color_palette == 3:  # redshift
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_redshift_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        else:  # default to ironbow (palette 1)
            wave_field.fluxmesh_xy_colors[i, j] = colormap.get_ironbow_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )

    # ================================================================
    # XZ Plane: Sample at y = center_j
    # ================================================================
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_old_am[i, center_j, k]

        # Map displacement to color using selected gradient
        if color_palette == 2:  # blueprint
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_blueprint_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        elif color_palette == 3:  # redshift
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_redshift_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        else:  # default to ironbow (palette 1)
            wave_field.fluxmesh_xz_colors[i, k] = colormap.get_ironbow_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )

    # ================================================================
    # YZ Plane: Sample at x = center_i
    # ================================================================
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        # Sample longitudinal displacement at this voxel
        disp_value = wave_field.displacement_old_am[center_i, j, k]

        # Map displacement to color using selected gradient
        if color_palette == 2:  # blueprint
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_blueprint_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        elif color_palette == 3:  # redshift
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_redshift_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )
        else:  # default to ironbow (palette 1)
            wave_field.fluxmesh_yz_colors[j, k] = colormap.get_ironbow_color(
                disp_value, -peak_amplitude_am, peak_amplitude_am
            )


# TODO: migrate to numerical analysis module
def plot_displacement_profile(wave_field):
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
    plt.title("LONGITUDINAL Displacement Profile", family="Monospace")
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
    plt.title("TRANSVERSE Displacement Profile", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save to research directory
    from pathlib import Path

    save_path = (
        Path(__file__).parent.parent
        / "xperiments"
        / "5_level1_wave_field"
        / "_research"
        / "displacement_profile.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    # plt.show()
