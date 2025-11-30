"""
Mesh Performance Test - Cube vs Plane Rendering

Tests scene.mesh() performance with different geometry complexity:
- Cube: 24 vertices (4 per face × 6 faces), 12 triangles, closed mesh
- Plane: 4 vertices, 2 triangles, thin surface (two_sided rendering)
- Baseline: 1.2M particles with physics simulation

Demonstrates per-vertex color interpolation, proper normals, and two_sided parameter.
"""

import time
import taichi as ti

ti.init(arch=ti.gpu)

# Mesh geometry fields - shared by both cube and plane
# Cube uses all 24 vertices (4 per face × 6 faces) for proper per-face normals
# Plane uses only top face vertices (indices 16-19) with upward normals
mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=24)
# Normals: per-vertex vectors perpendicular to surface, used for lighting calculations
# Each face has 4 vertices with the same normal pointing outward from that face
mesh_normals = ti.Vector.field(3, dtype=ti.f32, shape=24)
# Colors: per-vertex colors for dynamic wave effect in orange gradient
mesh_colors = ti.Vector.field(3, dtype=ti.f32, shape=24)

# Triangle indices
cube_indices = ti.field(ti.i32, shape=36)  # 12 triangles × 3 indices
plane_indices = ti.field(ti.i32, shape=6)  # 2 triangles × 3 indices


@ti.kernel
def populate_mesh_vertices():
    """
    Create mesh geometry with proper per-face normals.

    For sharp edges (cube), each face needs its own vertices with consistent normals.
    Vertices are duplicated at edges/corners where surface orientation changes.
    """
    size = 1.0

    # Front face (Z = 0) - Normal pointing in -Z direction
    mesh_vertices[0] = ti.Vector([0.0, 0.0, 0.0])
    mesh_vertices[1] = ti.Vector([size, 0.0, 0.0])
    mesh_vertices[2] = ti.Vector([size, size, 0.0])
    mesh_vertices[3] = ti.Vector([0.0, size, 0.0])
    mesh_normals[0] = ti.Vector([0.0, 0.0, -1.0])
    mesh_normals[1] = ti.Vector([0.0, 0.0, -1.0])
    mesh_normals[2] = ti.Vector([0.0, 0.0, -1.0])
    mesh_normals[3] = ti.Vector([0.0, 0.0, -1.0])

    # Back face (Z = size) - Normal pointing in +Z direction
    mesh_vertices[4] = ti.Vector([0.0, 0.0, size])
    mesh_vertices[5] = ti.Vector([0.0, size, size])
    mesh_vertices[6] = ti.Vector([size, size, size])
    mesh_vertices[7] = ti.Vector([size, 0.0, size])
    mesh_normals[4] = ti.Vector([0.0, 0.0, 1.0])
    mesh_normals[5] = ti.Vector([0.0, 0.0, 1.0])
    mesh_normals[6] = ti.Vector([0.0, 0.0, 1.0])
    mesh_normals[7] = ti.Vector([0.0, 0.0, 1.0])

    # Right face (X = size) - Normal pointing in +X direction
    mesh_vertices[8] = ti.Vector([size, 0.0, 0.0])
    mesh_vertices[9] = ti.Vector([size, 0.0, size])
    mesh_vertices[10] = ti.Vector([size, size, size])
    mesh_vertices[11] = ti.Vector([size, size, 0.0])
    mesh_normals[8] = ti.Vector([1.0, 0.0, 0.0])
    mesh_normals[9] = ti.Vector([1.0, 0.0, 0.0])
    mesh_normals[10] = ti.Vector([1.0, 0.0, 0.0])
    mesh_normals[11] = ti.Vector([1.0, 0.0, 0.0])

    # Left face (X = 0) - Normal pointing in -X direction
    mesh_vertices[12] = ti.Vector([0.0, 0.0, 0.0])
    mesh_vertices[13] = ti.Vector([0.0, size, 0.0])
    mesh_vertices[14] = ti.Vector([0.0, size, size])
    mesh_vertices[15] = ti.Vector([0.0, 0.0, size])
    mesh_normals[12] = ti.Vector([-1.0, 0.0, 0.0])
    mesh_normals[13] = ti.Vector([-1.0, 0.0, 0.0])
    mesh_normals[14] = ti.Vector([-1.0, 0.0, 0.0])
    mesh_normals[15] = ti.Vector([-1.0, 0.0, 0.0])

    # Top face (Y = size) - Normal pointing in +Y direction
    # Vertices 16-19 are used by the plane option
    mesh_vertices[16] = ti.Vector([0.0, size, 0.0])
    mesh_vertices[17] = ti.Vector([size, size, 0.0])
    mesh_vertices[18] = ti.Vector([size, size, size])
    mesh_vertices[19] = ti.Vector([0.0, size, size])
    mesh_normals[16] = ti.Vector([0.0, 1.0, 0.0])
    mesh_normals[17] = ti.Vector([0.0, 1.0, 0.0])
    mesh_normals[18] = ti.Vector([0.0, 1.0, 0.0])
    mesh_normals[19] = ti.Vector([0.0, 1.0, 0.0])

    # Bottom face (Y = 0) - Normal pointing in -Y direction
    mesh_vertices[20] = ti.Vector([0.0, 0.0, 0.0])
    mesh_vertices[21] = ti.Vector([0.0, 0.0, size])
    mesh_vertices[22] = ti.Vector([size, 0.0, size])
    mesh_vertices[23] = ti.Vector([size, 0.0, 0.0])
    mesh_normals[20] = ti.Vector([0.0, -1.0, 0.0])
    mesh_normals[21] = ti.Vector([0.0, -1.0, 0.0])
    mesh_normals[22] = ti.Vector([0.0, -1.0, 0.0])
    mesh_normals[23] = ti.Vector([0.0, -1.0, 0.0])


@ti.kernel
def populate_cube_indices():
    """
    Define triangles for cube (12 triangles, 2 per face).

    Each face is composed of 2 triangles with counter-clockwise winding
    (when viewed from outside) for proper backface culling.
    """
    # Front face (vertices 0-3)
    cube_indices[0] = 0
    cube_indices[1] = 1
    cube_indices[2] = 2
    cube_indices[3] = 0
    cube_indices[4] = 2
    cube_indices[5] = 3
    # Back face (vertices 4-7)
    cube_indices[6] = 4
    cube_indices[7] = 5
    cube_indices[8] = 6
    cube_indices[9] = 4
    cube_indices[10] = 6
    cube_indices[11] = 7
    # Right face (vertices 8-11)
    cube_indices[12] = 8
    cube_indices[13] = 9
    cube_indices[14] = 10
    cube_indices[15] = 8
    cube_indices[16] = 10
    cube_indices[17] = 11
    # Left face (vertices 12-15)
    cube_indices[18] = 12
    cube_indices[19] = 13
    cube_indices[20] = 14
    cube_indices[21] = 12
    cube_indices[22] = 14
    cube_indices[23] = 15
    # Top face (vertices 16-19)
    cube_indices[24] = 16
    cube_indices[25] = 17
    cube_indices[26] = 18
    cube_indices[27] = 16
    cube_indices[28] = 18
    cube_indices[29] = 19
    # Bottom face (vertices 20-23)
    cube_indices[30] = 20
    cube_indices[31] = 21
    cube_indices[32] = 22
    cube_indices[33] = 20
    cube_indices[34] = 22
    cube_indices[35] = 23


@ti.kernel
def populate_plane_indices():
    """
    Define triangles for plane (2 triangles - top face only).

    Plane uses only the top face vertices (16-19) with upward normals.
    """
    # Top face only (vertices 16-19)
    plane_indices[0] = 16
    plane_indices[1] = 17
    plane_indices[2] = 18
    plane_indices[3] = 16
    plane_indices[4] = 18
    plane_indices[5] = 19


@ti.kernel
def update_mesh_colors(t: ti.f32):  # type: ignore
    """
    Update mesh vertex colors with wave effect in orange gradient.

    GPU performs barycentric interpolation across triangle surfaces,
    creating smooth color gradients from per-vertex colors.
    """
    for i in range(24):
        pos = mesh_vertices[i]
        # Create wave based on position and time
        wave = ti.sin(t * 3.0 + pos[0] * 5.0 + pos[1] * 5.0 + pos[2] * 5.0)
        # Map wave [-1, 1] to brightness [0.3, 1.0]
        brightness = 0.3 + (wave + 1.0) * 0.35
        # Orange gradient: (R=1.0, G=0.5, B=0.0) scaled by brightness
        mesh_colors[i] = ti.Vector([1.0 * brightness, 0.5 * brightness, 0.0])


populate_mesh_vertices()
populate_cube_indices()
populate_plane_indices()

# ============================================================================
# Baseline Workload: 1.2M Particles Simulation
# ============================================================================
# Simulates OpenWave-like workload to test mesh rendering overhead

num_granules = 1200000  # 1.2M particles
granule_positions = ti.Vector.field(3, dtype=ti.f32, shape=num_granules)
granule_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_granules)
equilibrium = ti.Vector.field(3, dtype=ti.f32, shape=num_granules)


@ti.kernel
def init_granules():
    """Initialize particle positions randomly in unit cube."""
    for i in range(num_granules):
        pos = ti.Vector([ti.random() * 1.0, ti.random() * 1.0, ti.random() * 1.0])
        granule_positions[i] = pos
        equilibrium[i] = pos
        granule_colors[i] = ti.Vector([0.1, 0.6, 0.9])


@ti.kernel
def simulate_physics(t: ti.f32):  # type: ignore
    """Simulate wave oscillation and dynamic per-vertex coloring."""
    for i in range(num_granules):
        # Wave oscillation in Z direction
        offset = ti.cos(t * 5.0 + equilibrium[i][0] * 10.0) * 0.02
        granule_positions[i] = equilibrium[i] + ti.Vector([0.0, 0.0, offset])
        # Dynamic color based on displacement (ironbow effect)
        displacement = ti.abs(offset)
        granule_colors[i] = ti.Vector([0.1 + displacement * 10.0, 0.6, 0.9])


init_granules()

# ============================================================================
# UI Setup
# ============================================================================

window = ti.ui.Window("Mesh Performance Test", (1200, 900), vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()
scene = window.get_scene()
camera = ti.ui.Camera()

# UI state - mutually exclusive mesh options
show_cube = True
show_plane = False
frames = 0
last_time = time.time()
elapsed_time = 0.0

canvas.set_background_color((0.1, 0.1, 0.15))

# ============================================================================
# Render Loop
# ============================================================================

while window.running:
    # Time tracking
    current_time = time.time()
    dt_real = current_time - last_time
    last_time = current_time
    elapsed_time += dt_real
    fps = 0 if elapsed_time == 0 else frames / elapsed_time

    # Camera setup
    camera.position(2, 2, 2)
    camera.lookat(0, 0, 0)
    camera.up(0, 0, 1)
    scene.set_camera(camera)

    # Lighting (must be set each frame in GGUI)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0, 0, 2), color=(1, 1, 1))

    # Update physics and colors
    simulate_physics(elapsed_time)
    update_mesh_colors(elapsed_time)

    # Render baseline particle simulation
    scene.particles(granule_positions, radius=0.002, per_vertex_color=granule_colors)

    # Control panel UI
    with gui.sub_window("CONTROL", 0.78, 0.00, 0.22, 0.33) as sub:
        sub.text("MESH OPTIONS")
        sub.text("(mutually exclusive)")
        sub.text("")

        # Mutually exclusive checkboxes
        new_cube = sub.checkbox("Cube Mesh", show_cube)
        new_plane = sub.checkbox("Plane Mesh", show_plane)

        # Handle mutual exclusivity logic
        if new_cube != show_cube and new_cube:
            show_cube = True
            show_plane = False
        elif new_plane != show_plane and new_plane:
            show_plane = True
            show_cube = False
        elif not new_cube and not new_plane:
            show_cube = new_cube
            show_plane = new_plane
        else:
            show_cube = new_cube
            show_plane = new_plane

        # Performance metrics
        sub.text("")
        sub.text("PERFORMANCE")
        sub.text(f"Frames Rendered: {frames}")
        sub.text(f"Elapsed Time: {elapsed_time:.2f}s")
        sub.text(f"Avg Frame-Rate: {fps:.0f} FPS")
        sub.text("")

        # Current mesh info
        sub.text("CURRENT MESH")
        if show_cube:
            sub.text("Cube: 24 vertices, 12 triangles")
            sub.text("two_sided: False (closed mesh)")
        elif show_plane:
            sub.text("Plane: 4 vertices, 2 triangles")
            sub.text("two_sided: True (thin surface)")
        else:
            sub.text("None - baseline test")
            sub.text("Expected: ~48 FPS baseline")

    # Render selected mesh
    if show_cube:
        scene.mesh(
            mesh_vertices,
            indices=cube_indices,
            normals=mesh_normals,
            per_vertex_color=mesh_colors,
            show_wireframe=False,
            two_sided=False,  # Closed mesh - backface culling enabled
        )
    elif show_plane:
        scene.mesh(
            mesh_vertices,
            indices=plane_indices,
            normals=mesh_normals,
            per_vertex_color=mesh_colors,
            show_wireframe=False,
            two_sided=True,  # Thin surface - visible from both sides
        )

    canvas.scene(scene)
    window.show()
    frames += 1
