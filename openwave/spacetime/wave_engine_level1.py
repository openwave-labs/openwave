"""
ENERGY-WAVE ENGINE

LEVEL-1: ON WAVE-FIELD MEDIUM

Wave Physics Engine @spacetime module. Wave dynamics and motion.
"""

import taichi as ti

from openwave.common import colormap


@ti.kernel
def create_test_displacement_pattern(wave_field: ti.template()):  # type: ignore
    """
    Create a simple static displacement pattern for testing flux films.

    Generates a radial wave pattern emanating from the universe center.
    This is temporary test code until wave propagation is implemented.

    Args:
        wave_field: WaveField instance containing displacement field
    """

    # Center position (in voxel indices)
    center_x = ti.cast(wave_field.nx, ti.f32) / 2.0
    center_y = ti.cast(wave_field.ny, ti.f32) / 2.0
    center_z = ti.cast(wave_field.nz, ti.f32) / 2.0
    # Create radial displacement pattern
    for i, j, k in ti.ndrange(wave_field.nx, wave_field.ny, wave_field.nz):
        # Distance from center
        dx = ti.cast(i, ti.f32) - center_x
        dy = ti.cast(j, ti.f32) - center_y
        dz = ti.cast(k, ti.f32) - center_z
        r = ti.sqrt(dx * dx + dy * dy + dz * dz)

        # Simple sinusoidal radial pattern
        # Amplitude decreases with distance, oscillates radially
        max_r = ti.sqrt(center_x * center_x + center_y * center_y + center_z * center_z)
        normalized_r = r / max_r

        # Displacement magnitude (in attometers for scalar field)
        # Creates rings of positive/negative displacement
        # Signed value: positive = expansion, negative = compression
        amplitude = 1000.0 * ti.cos(normalized_r * 10.0 * ti.math.pi)

        # Apply scalar displacement (amplitude in attometers)
        wave_field.displacement_am[i, j, k] = amplitude


@ti.kernel
def update_flux_film_colors(wave_field: ti.template()):  # type: ignore
    """
    Update flux film colors by sampling wave properties from voxel grid.

    Samples wave displacement at each film vertex position and maps it to a color
    using the redshift gradient (red=negative, gray=zero, blue=positive).

    This function should be called every frame after wave propagation to update
    the visualization based on current wave field state.

    Color mapping:
    - Uses get_redshift_color() from config module
    - Samples displacement at voxel centers corresponding to vertex positions
    - Maps signed displacement values to red-gray-blue gradient

    Args:
        wave_field: WaveField instance containing flux film fields and displacement data
    """

    # Get center indices for each film
    center_i = wave_field.nx // 2
    center_j = wave_field.ny // 2
    center_k = wave_field.nz // 2

    # Displacement range for color scaling (in attometers)
    # TODO: In future, use exponential moving average tracker like LEVEL-0 ironbow
    # For now, use fixed range based on test pattern amplitude
    min_disp = -1000.0  # -1000 attometers (negative displacement/compression)
    max_disp = 1000.0  # +1000 attometers (positive displacement/expansion)

    # ================================================================
    # XY Film: Sample at z = center_k
    # ================================================================
    for i, j in ti.ndrange(wave_field.nx, wave_field.ny):
        # Sample scalar displacement at this voxel
        disp_value = wave_field.displacement_am[i, j, center_k]

        # Map displacement to color using redshift gradient
        color = colormap.get_redshift_color(disp_value, min_disp, max_disp)
        wave_field.film_xy_colors[i, j] = color

    # ================================================================
    # XZ Film: Sample at y = center_j
    # ================================================================
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        # Sample scalar displacement at this voxel
        disp_value = wave_field.displacement_am[i, center_j, k]

        # Map to color
        color = colormap.get_redshift_color(disp_value, min_disp, max_disp)
        wave_field.film_xz_colors[i, k] = color

    # ================================================================
    # YZ Film: Sample at x = center_i
    # ================================================================
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        # Sample scalar displacement at this voxel
        disp_value = wave_field.displacement_am[center_i, j, k]

        # Map to color
        color = colormap.get_redshift_color(disp_value, min_disp, max_disp)
        wave_field.film_yz_colors[j, k] = color
