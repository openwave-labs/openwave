"""
Run sample xperiments shipped with the OpenWave package, tweak them, or create your own

eg. Tweak this xperiment script changing universe_edge = 0.1 m at __main__ entry point
(the approximate size of a tesseract) and simulate this artifact energy content,
sourced from the element aether.
"""

import numpy as np
import taichi as ti

import openwave.common.config as config
import openwave.source.spacetime as spacetime

ti.init(arch=ti.gpu)


# ================================================================
# Rendering Engine
# ================================================================

# Create GGUI window with 3D scene
window = ti.ui.Window(
    "Spacetime (Topology: 3D BCC Lattice)",
    (config.SCREEN_RES[0], config.SCREEN_RES[1]),
    vsync=True,
)
camera = ti.ui.Camera()  # Camera object for 3D view control
canvas = window.get_canvas()  # Canvas for rendering the scene
gui = window.get_gui()  # GUI manager for overlay UI elements
scene = window.get_scene()  # 3D scene for particle rendering


def initialize_scene():
    """Initialize scene settings."""
    canvas.set_background_color(config.COLOR_SPACE[1])


def setup_scene_lighting():
    """Set up scene lighting - must be called every frame in GGUI."""
    scene.ambient_light((0.1, 0.1, 0.15))  # Slight blue ambient
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1.0, 1.0, 1.0))  # White light from above center
    scene.point_light(pos=(1.0, 0.5, 1.0), color=(0.5, 0.5, 0.5))  # Dimmed white light from corner


def initialize_camera():
    """Initialize camera parameters for orbit & zoom controls."""
    global orbit_center, orbit_radius, orbit_theta, orbit_phi, mouse_sensitivity, last_mouse_pos

    # Camera orbit parameters - initial position looking at center
    orbit_center = [0.5, 0.5, 0.5]  # Center of the lattice
    cam_position = [2.0, 1.5, 2.0]  # Camera starting position

    # Calculate initial angles from the desired initial position
    initial_rel_x = cam_position[0] - orbit_center[0]
    initial_rel_y = cam_position[1] - orbit_center[1]
    initial_rel_z = cam_position[2] - orbit_center[2]

    # Calculate initial orbit parameters
    orbit_radius = np.sqrt(initial_rel_x**2 + initial_rel_y**2 + initial_rel_z**2)  # ~1.5
    orbit_theta = np.arctan2(initial_rel_z, initial_rel_x)  # 45 degrees
    orbit_phi = np.arccos(initial_rel_y / orbit_radius)  # angle from vertical

    mouse_sensitivity = 0.5
    last_mouse_pos = None


def handle_camera():
    """Handle mouse and keyboard input for camera controls."""
    global orbit_theta, orbit_phi, orbit_radius, last_mouse_pos

    # Handle mouse input for orbiting
    mouse_pos = window.get_cursor_pos()

    # Check for right mouse button drag
    if window.is_pressed(ti.ui.RMB):
        if last_mouse_pos is not None:
            # Calculate mouse delta
            dx = mouse_pos[0] - last_mouse_pos[0]
            dy = mouse_pos[1] - last_mouse_pos[1]

            # Update orbit angles
            orbit_theta += dx * mouse_sensitivity * 2 * np.pi
            # Allow nearly full vertical rotation (almost 180 degrees, small margin to prevent gimbal lock)
            orbit_phi = np.clip(orbit_phi + dy * mouse_sensitivity * np.pi, 0.01, np.pi - 0.01)

        last_mouse_pos = mouse_pos
    else:
        last_mouse_pos = None

    # Handle keyboard input for zoom
    if window.is_pressed("q"):  # Zoom in
        orbit_radius *= 0.98
        orbit_radius = np.clip(orbit_radius, 0.5, 5.0)
    if window.is_pressed("z"):  # Zoom out
        orbit_radius *= 1.02
        orbit_radius = np.clip(orbit_radius, 0.5, 5.0)

    # Handle keyboard input for panning
    if window.is_pressed(ti.ui.DOWN):  # Tilt up
        orbit_center[1] += 0.01 * orbit_radius
    if window.is_pressed(ti.ui.UP):  # Tilt down
        orbit_center[1] -= 0.01 * orbit_radius
    if window.is_pressed(ti.ui.RIGHT):  # Pan left
        orbit_center[0] -= 0.01 * orbit_radius
    if window.is_pressed(ti.ui.LEFT):  # Pan right
        orbit_center[0] += 0.01 * orbit_radius

    """Now update camera position based on current orbit parameters.
    Converts spherical coordinates (orbit_radius, orbit_theta, orbit_phi) to
    Cartesian coordinates and updates the camera position and orientation.
    Camera always looks at orbit_center with Y-axis up.
    """

    # Calculate camera position from spherical coordinates
    cam_x = orbit_center[0] + orbit_radius * np.sin(orbit_phi) * np.cos(orbit_theta)
    cam_y = orbit_center[1] + orbit_radius * np.cos(orbit_phi)
    cam_z = orbit_center[2] + orbit_radius * np.sin(orbit_phi) * np.sin(orbit_theta)

    # Update camera
    camera.position(cam_x, cam_y, cam_z)
    camera.lookat(orbit_center[0], orbit_center[1], orbit_center[2])
    camera.up(0, 1, 0)
    scene.set_camera(camera)


def render_controls():
    """Render the controls UI overlay."""
    global block_slice, granule_type, show_springs, radius_factor

    # Create overlay windows for stats & controls
    with gui.sub_window("CONTROLS", 0.01, 0.50, 0.20, 0.23) as sub:
        sub.text("Cam Orbit: right-click + drag")
        sub.text("Zoom: Q/Z keys")
        sub.text("Pan: Arrow keys")
        block_slice = sub.checkbox("Block Slice", block_slice)
        granule_type = sub.checkbox("Granule Type Color", granule_type)
        show_springs = sub.checkbox("Show Springs (if <1k granules)", show_springs)
        radius_factor = sub.slider_float("Granule", radius_factor, 0.0, 2.0)
        if sub.button("Reset Granule"):
            radius_factor = 1.0


def render_data_dashboard():
    """Display simulation data dashboard."""
    with gui.sub_window("DATA-DASHBOARD", 0.01, 0.01, 0.24, 0.45) as sub:
        sub.text("--- SPACETIME ---")
        sub.text("Topology: 3D BCC lattice")
        sub.text(f"Total Granules: {lattice.total_granules:,} (config.py)")
        sub.text(f"  - Corner granules: {(lattice.grid_size + 1) ** 3:,}")
        sub.text(f"  - Center granules: {lattice.grid_size ** 3:,}")
        sub.text(f"Universe Lattice Edge: {lattice.universe_edge:.1e} m")
        sub.text(f"Unit-Cells per Lattice Edge: {lattice.grid_size:,}")

        sub.text("")
        sub.text("--- Scaling-Up (for computation) ---")
        sub.text(f"Factor: {lattice.scale_factor:.1e} x Planck Scale (magnified)")
        sub.text(f"BCC Unit-Cell Edge: {lattice.unit_cell_edge:.2e} m")
        sub.text(f"Granule Radius: {granule.radius:.2e} m")
        sub.text(f"Granule Mass: {granule.mass:.2e} kg")

        sub.text("")
        sub.text("--- Simulation Resolution (linear) ---")
        sub.text(f"QWave: {lattice.qwave_res:.0f} granules/qwavelength (min 2)")
        if lattice.qwave_res < 2:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Universe: {lattice.uni_res:.1f} qwaves/universe-edge")

        sub.text("")
        sub.text("--- Universe Lattice Wave Energy ---")
        sub.text(f"Energy: {lattice.energy:.1e} J ({lattice.energy_kWh:.1e} KWh)")
        sub.text(f"{lattice.energy_years:,.1e} Years of global energy use")


def render_lattice(lattice, granule, springs=None):
    """
    Render 3D BCC lattice using GGUI's 3D scene.

    Args:
        lattice: Lattice instance containing positions and universe parameters.
                 Expected to have attributes: positions, total_granules, universe_edge
        granule: Granule instance for size reference.
                 Expected to have attribute: radius
        springs: Spring instance containing connectivity information (optional)


    """
    global orbit_center, orbit_radius, orbit_theta, orbit_phi, mouse_sensitivity, last_mouse_pos
    global block_slice, granule_type, show_springs, radius_factor

    # Normalize granule positions for rendering (0-1 range for GGUI) & block-slicing
    # block-slicing: hide front 1/8th of the lattice for see-through effect
    normalized_positions = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
    normalized_positions_sliced = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
    block_slice = False  # Initialize Block-slicing toggle
    granule_type = False  # Initialize Granule type coloring toggle
    show_springs = False  # Initialize spring visualization toggle

    # Prepare spring line endpoints if springs provided
    max_connections = 0
    spring_lines = None
    spring_color = config.COLOR_INFRA[1]
    if springs is not None:
        # Count total connections for line buffer
        for i in range(lattice.total_granules):
            max_connections += springs.links_count[i]

        if max_connections > 0:
            # Allocate line endpoint buffer (2 points per line)
            spring_lines = ti.Vector.field(3, dtype=ti.f32, shape=max_connections * 2)

    @ti.kernel
    def normalize_positions():
        """Normalize lattice positions to 0-1 range for GGUI rendering."""
        for i in range(lattice.total_granules):
            # Normalize to 0-1 range
            normalized_positions[i] = lattice.positions[i] / lattice.universe_edge

    @ti.kernel
    def normalize_positions_sliced():
        """Normalize lattice positions to 0-1 and apply block-slicing."""
        for i in range(lattice.total_granules):
            # Normalize to 0-1 range
            # And hide front 1/8th of the lattice for see-through effect (block-slicing)
            # 0 = not in front octant (render it), 1 = in front octant (skip it)
            if lattice.front_octant[i] == 0:
                normalized_positions_sliced[i] = lattice.positions[i] / lattice.universe_edge

    # Create a field to track line index atomically
    line_counter = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def build_spring_lines():
        """Build line endpoints for spring connections."""
        # Reset counter
        line_counter[None] = 0

        # Build lines with atomic indexing to ensure correct ordering
        for i in range(lattice.total_granules):
            num_links = springs.links_count[i]
            if num_links > 0:
                pos_i = lattice.positions[i] / lattice.universe_edge  # Normalized position

                for j in range(num_links):
                    neighbor_idx = springs.links[i, j]
                    if neighbor_idx >= 0:  # Valid connection
                        pos_j = lattice.positions[neighbor_idx] / lattice.universe_edge

                        # Get current line index atomically
                        line_idx = ti.atomic_add(line_counter[None], 1)

                        # Add line endpoints (from i to j)
                        if line_idx < max_connections:  # Safety check
                            spring_lines[line_idx * 2] = pos_i
                            spring_lines[line_idx * 2 + 1] = pos_j

    # Normalize positions once before render loop
    normalize_positions()
    normalize_positions_sliced()

    # Build spring lines if springs provided
    if springs is not None and max_connections > 0:
        build_spring_lines()

    # Normalize granule radius to 0-1 range for GGUI rendering
    normalized_radius = max(
        granule.radius / lattice.universe_edge, 0.0001
    )  # Ensure minimum 0.01% of screen radius for visibility
    radius_factor = 1.0  # Initialize granule size factor

    initialize_scene()  # Set up background once
    initialize_camera()  # Set initial camera parameters

    while window.running:
        setup_scene_lighting()  # Lighting must be set each frame in GGUI
        handle_camera()  # Handle camera input and update position

        # Render UI overlay windows
        render_controls()
        render_data_dashboard()

        # Render granules with optional block-slicing and type-coloring
        centers = normalized_positions_sliced if block_slice else normalized_positions

        if granule_type:
            scene.particles(
                centers,
                radius=normalized_radius * radius_factor,
                per_vertex_color=lattice.granule_color,
            )
        else:
            scene.particles(
                centers,
                radius=normalized_radius * radius_factor,
                color=config.COLOR_GRANULE[1],
            )

        # Render springs if enabled and available
        if show_springs and springs is not None and spring_lines is not None:
            scene.lines(spring_lines, width=5, color=spring_color)

        # Render the scene to canvas
        canvas.scene(scene)
        window.show()


# ================================================================
# Main calls
# ================================================================
if __name__ == "__main__":

    # Quantum objects instantiation
    universe_edge = 3e-16  # m (default 300 attometers, contains ~10 qwaves per linear edge)
    lattice = spacetime.Lattice(universe_edge)
    granule = spacetime.Granule(lattice.unit_cell_edge)
    if config.SPACETIME_RES <= 1000:
        springs = spacetime.Spring(lattice)  # Create spring links between granules
    else:
        springs = None  # Skip springs for very high resolutions to save memory

    # Render the 3D lattice
    render_lattice(lattice, granule, springs)
