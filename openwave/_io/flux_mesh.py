"""
Flux Mesh Rendering Module

Provides rendering functions for flux mesh visualization in LEVEL-1 wave field simulations.

Flux Mesh is a 2D cross-sectional detector surface used to visualize wave properties
through color gradients. This module handles mesh rendering for three orthogonal
planes (XY, XZ, YZ) intersecting at the universe center.

Part of the I/O module group for reusable visualization components.
"""

import taichi as ti

# ================================================================
# Module-level flattened fields (initialized once, reused every frame)
# ================================================================
_flux_mesh_fields_initialized = False
_xy_vertices_flat = None
_xy_colors_flat = None
_xy_indices_flat = None
_xz_vertices_flat = None
_xz_colors_flat = None
_xz_indices_flat = None
_yz_vertices_flat = None
_yz_colors_flat = None
_yz_indices_flat = None


def initialize_flux_mesh_fields(wave_field: ti.template()):  # type: ignore
    """
    Initialize flattened Taichi fields for flux mesh rendering.

    This should be called once after the wave_field is created. The fields are reused
    every frame to avoid expensive field allocations.

    Args:
        wave_field: WaveField instance containing flux mesh data
    """
    global _flux_mesh_fields_initialized
    global _xy_vertices_flat, _xy_colors_flat, _xy_indices_flat
    global _xz_vertices_flat, _xz_colors_flat, _xz_indices_flat
    global _yz_vertices_flat, _yz_colors_flat, _yz_indices_flat

    if _flux_mesh_fields_initialized:
        return  # Already initialized

    # XY Plane flattened fields
    xy_vertex_count = wave_field.nx * wave_field.ny
    _xy_vertices_flat = ti.Vector.field(3, dtype=ti.f32, shape=xy_vertex_count)
    _xy_colors_flat = ti.Vector.field(3, dtype=ti.f32, shape=xy_vertex_count)
    xy_triangle_count = (wave_field.nx - 1) * (wave_field.ny - 1) * 2
    _xy_indices_flat = ti.field(dtype=ti.i32, shape=xy_triangle_count * 3)

    # XZ Plane flattened fields
    xz_vertex_count = wave_field.nx * wave_field.nz
    _xz_vertices_flat = ti.Vector.field(3, dtype=ti.f32, shape=xz_vertex_count)
    _xz_colors_flat = ti.Vector.field(3, dtype=ti.f32, shape=xz_vertex_count)
    xz_triangle_count = (wave_field.nx - 1) * (wave_field.nz - 1) * 2
    _xz_indices_flat = ti.field(dtype=ti.i32, shape=xz_triangle_count * 3)

    # YZ Plane flattened fields
    yz_vertex_count = wave_field.ny * wave_field.nz
    _yz_vertices_flat = ti.Vector.field(3, dtype=ti.f32, shape=yz_vertex_count)
    _yz_colors_flat = ti.Vector.field(3, dtype=ti.f32, shape=yz_vertex_count)
    yz_triangle_count = (wave_field.ny - 1) * (wave_field.nz - 1) * 2
    _yz_indices_flat = ti.field(dtype=ti.i32, shape=yz_triangle_count * 3)

    # Flatten indices once (they never change)
    flatten_xy_indices(wave_field, _xy_indices_flat)
    flatten_xz_indices(wave_field, _xz_indices_flat)
    flatten_yz_indices(wave_field, _yz_indices_flat)

    _flux_mesh_fields_initialized = True


def render_flux_mesh(scene, wave_field, show_flux_mesh):
    """
    Render flux mesh planes to the scene with two-sided rendering.

    Displays XY, XZ, and YZ flux mesh as colored meshes visualizing wave
    displacement through the universe domain. Uses two-sided rendering to
    ensure visibility from all camera angles.

    Args:
        scene: Taichi GGUI scene object
        wave_field: WaveField instance containing flux mesh data

    Usage:
        from openwave._io.flux_mesh import render_flux_mesh, initialize_flux_mesh_fields

        # Initialize once after wave_field creation
        initialize_flux_mesh_fields(wave_field)

        # Render every frame
        if state.flux_mesh:
            render_flux_mesh(scene, wave_field)
    """
    # Ensure fields are initialized (returns early if already done)
    initialize_flux_mesh_fields(wave_field)

    # ================================================================
    # Update vertices and colors every frame (vertices change with warping)
    # Single kernel call for all three planes to reduce launch overhead
    # ================================================================
    flatten_all_vertices_and_colors(
        wave_field,
        _xy_vertices_flat,
        _xy_colors_flat,
        _xz_vertices_flat,
        _xz_colors_flat,
        _yz_vertices_flat,
        _yz_colors_flat,
    )

    # ================================================================
    # Render flux mesh planes
    # ================================================================
    if show_flux_mesh in (1, 2, 3):
        scene.mesh(
            _xy_vertices_flat,
            indices=_xy_indices_flat,
            per_vertex_color=_xy_colors_flat,
            two_sided=True,
            show_wireframe=True,
        )

    if show_flux_mesh in (2, 3):
        scene.mesh(
            _xz_vertices_flat,
            indices=_xz_indices_flat,
            per_vertex_color=_xz_colors_flat,
            two_sided=True,
            show_wireframe=True,
        )

    if show_flux_mesh == 3:
        scene.mesh(
            _yz_vertices_flat,
            indices=_yz_indices_flat,
            per_vertex_color=_yz_colors_flat,
            two_sided=True,
            show_wireframe=True,
        )


# ================================================================
# Flatten XY Plane (flatten 2D mesh data to 1D arrays for rendering)
# ================================================================


@ti.kernel
def flatten_xy_vertices(
    wave_field: ti.template(),  # type: ignore
    vertices_flat: ti.template(),  # type: ignore
):
    """Flatten XY Plane vertices to 1D array (called once during init)."""
    for i, j in ti.ndrange(wave_field.nx, wave_field.ny):
        idx = i * wave_field.ny + j
        vertices_flat[idx] = wave_field.fluxmesh_xy_vertices[i, j]


@ti.kernel
def flatten_xy_indices(wave_field: ti.template(), indices_flat: ti.template()):  # type: ignore
    """Flatten XY Plane triangle indices to 1D array for rendering."""
    for i, j in ti.ndrange(wave_field.nx - 1, wave_field.ny - 1):
        base_idx = (i * (wave_field.ny - 1) + j) * 6
        for k in ti.static(range(6)):
            indices_flat[base_idx + k] = wave_field.fluxmesh_xy_indices[i, j, k]


# ================================================================
# Flatten XZ Plane (flatten 2D mesh data to 1D arrays for rendering)
# ================================================================


@ti.kernel
def flatten_xz_vertices(
    wave_field: ti.template(),  # type: ignore
    vertices_flat: ti.template(),  # type: ignore
):
    """Flatten XZ Plane vertices to 1D array (called once during init)."""
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        idx = i * wave_field.nz + k
        vertices_flat[idx] = wave_field.fluxmesh_xz_vertices[i, k]


@ti.kernel
def flatten_xz_indices(wave_field: ti.template(), indices_flat: ti.template()):  # type: ignore
    """Flatten XZ Plane triangle indices to 1D array for rendering."""
    for i, k in ti.ndrange(wave_field.nx - 1, wave_field.nz - 1):
        base_idx = (i * (wave_field.nz - 1) + k) * 6
        for m in ti.static(range(6)):
            indices_flat[base_idx + m] = wave_field.fluxmesh_xz_indices[i, k, m]


# ================================================================
# Flatten YZ Plane (flatten 2D mesh data to 1D arrays for rendering)
# ================================================================


@ti.kernel
def flatten_yz_vertices(
    wave_field: ti.template(),  # type: ignore
    vertices_flat: ti.template(),  # type: ignore
):
    """Flatten YZ Plane vertices to 1D array (called once during init)."""
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        idx = j * wave_field.nz + k
        vertices_flat[idx] = wave_field.fluxmesh_yz_vertices[j, k]


@ti.kernel
def flatten_yz_indices(wave_field: ti.template(), indices_flat: ti.template()):  # type: ignore
    """Flatten YZ Plane triangle indices to 1D array for rendering."""
    for j, k in ti.ndrange(wave_field.ny - 1, wave_field.nz - 1):
        base_idx = (j * (wave_field.nz - 1) + k) * 6
        for m in ti.static(range(6)):
            indices_flat[base_idx + m] = wave_field.fluxmesh_yz_indices[j, k, m]


# ================================================================
# Flatten Vertices and Colors (combined for single kernel launch)
# ================================================================


@ti.kernel
def flatten_all_vertices_and_colors(
    wave_field: ti.template(),  # type: ignore
    xy_vertices_flat: ti.template(),  # type: ignore
    xy_colors_flat: ti.template(),  # type: ignore
    xz_vertices_flat: ti.template(),  # type: ignore
    xz_colors_flat: ti.template(),  # type: ignore
    yz_vertices_flat: ti.template(),  # type: ignore
    yz_colors_flat: ti.template(),  # type: ignore
):
    """Flatten all vertices and colors in a single kernel (called every frame).

    Vertices must be flattened every frame since warping modifies Z coordinates.
    Combining with colors reduces kernel launch overhead.
    """
    # XY Plane
    for i, j in ti.ndrange(wave_field.nx, wave_field.ny):
        idx = i * wave_field.ny + j
        xy_vertices_flat[idx] = wave_field.fluxmesh_xy_vertices[i, j]
        xy_colors_flat[idx] = wave_field.fluxmesh_xy_colors[i, j]

    # XZ Plane
    for i, k in ti.ndrange(wave_field.nx, wave_field.nz):
        idx = i * wave_field.nz + k
        xz_vertices_flat[idx] = wave_field.fluxmesh_xz_vertices[i, k]
        xz_colors_flat[idx] = wave_field.fluxmesh_xz_colors[i, k]

    # YZ Plane
    for j, k in ti.ndrange(wave_field.ny, wave_field.nz):
        idx = j * wave_field.nz + k
        yz_vertices_flat[idx] = wave_field.fluxmesh_yz_vertices[j, k]
        yz_colors_flat[idx] = wave_field.fluxmesh_yz_colors[j, k]
