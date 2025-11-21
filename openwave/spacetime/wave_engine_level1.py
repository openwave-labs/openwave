"""
ENERGY-WAVE ENGINE

LEVEL-1: ON WAVE-FIELD MEDIUM

Wave Physics Engine @spacetime module. Wave dynamics and motion.
"""

import taichi as ti

from openwave.common import colormap, constants, equations, utils

# ================================================================
# Energy-Wave Oscillation Parameters
# ================================================================
base_amplitude_am = constants.EWAVE_AMPLITUDE / constants.ATTOMETER  # am, oscillation amplitude
wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETER  # in attometers
frequency = constants.EWAVE_SPEED / constants.EWAVE_LENGTH  # Hz, energy-wave frequency


@ti.kernel
def initiate_charge(wave_field: ti.template()):  # type: ignore
    """
    Create a simple static displacement pattern for testing flux mesh.
    Define wave character in the wave field.

    Generates a radial wave pattern emanating from the universe center.
    This is temporary test code until wave propagation is implemented.

    Args:
        wave_field: WaveField instance containing displacement field
    """

    # Center position (in voxel indices)
    center_x = ti.cast(wave_field.nx, ti.f32) / 2.0
    center_y = ti.cast(wave_field.ny, ti.f32) / 2.0
    center_z = ti.cast(wave_field.nz, ti.f32) / 2.0

    # Wave number k = 2π/λ (spatial phase variation)
    wave_number = 2.0 * ti.math.pi / (wavelength_am / wave_field.dx_am)  # radians per grid index

    # Create radial displacement pattern
    for i, j, k in ti.ndrange(wave_field.nx, wave_field.ny, wave_field.nz):
        # Distance from center
        dx = ti.cast(i, ti.f32) - center_x
        dy = ti.cast(j, ti.f32) - center_y
        dz = ti.cast(k, ti.f32) - center_z
        r = ti.sqrt(dx * dx + dy * dy + dz * dz)

        # Simple sinusoidal radial pattern
        # Amplitude decreases with distance, oscillates radially
        r_safe_am = ti.max(
            r, wavelength_am / wave_field.dx_am
        )  # minimum 1 wavelength_am from source
        amplitude_falloff = (wavelength_am / wave_field.dx_am) / r_safe_am
        amplitude_am_at_r = base_amplitude_am * amplitude_falloff

        # Displacement magnitude (in attometers for scalar field)
        # displacement from center source: A(r)·cos(ωt - kr), where ωt=0 at t=0
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        disp = amplitude_am_at_r * ti.cos(-wave_number * r)

        # Apply both longitudinal and transverse displacement (in attometers)
        wave_field.displacement_am[i, j, k][0] = disp
        wave_field.displacement_am[i, j, k][1] = 0.0


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
        disp_value = wave_field.displacement_am[i, j, center_k][0]

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
        disp_value = wave_field.displacement_am[i, center_j, k][0]

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
        disp_value = wave_field.displacement_am[center_i, j, k][0]

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
        displacements_L[i] = wave_field.displacement_am[i, center_j, center_k][0]
        displacements_T[i] = wave_field.displacement_am[i, center_j, center_k][1]

    # Calculate distance from center in grid indices
    center_x = nx / 2.0
    distances = x_indices - center_x

    # Create the plot
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(12, 6), facecolor=colormap.WHITE[1])
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
