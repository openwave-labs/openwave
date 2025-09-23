"""
Run sample xperiments shipped with the OpenWave package, tweak them, or create your own

eg. Tweak this xperiment script changing universe_edge = 0.1 m at __main__ entry point
(the approximate size of a tesseract) and simulate this artifact energy content,
sourced from the element aether.
"""

import numpy as np
import taichi as ti

import openwave.core.config as config
import openwave.core.constants as constants
import openwave.spacetime.quantum_space as quantum_space

ti.init(arch=ti.gpu)


# ==================================================================
# Rendering Engine
# ==================================================================

# Create GGUI window with 3D scene
window = ti.ui.Window(
    "Quantum-Space (Topology: 3D BCC Lattice)",
    (config.SCREEN_RES[0], config.SCREEN_RES[1]),
    vsync=True,
)
canvas = window.get_canvas()  # Canvas for rendering the scene
gui = window.get_gui()  # GUI manager for overlay UI elements
scene = window.get_scene()  # 3D scene for particle rendering
camera = ti.ui.Camera()  # Camera object for 3D view control
camera.up(0, 1, 0)  # Y-axis up


def initialize_scene():
    """Initialize scene settings that only need to be set once."""
    # Set background color once
    canvas.set_background_color(config.COLOR_SPACE[2])


def setup_scene_lighting():
    """Set up scene lighting - must be called every frame in GGUI."""
    scene.ambient_light((0.1, 0.1, 0.15))  # Slight blue ambient
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1.0, 1.0, 1.0))  # White light from above center
    scene.point_light(pos=(1.0, 0.5, 1.0), color=(0.5, 0.5, 0.5))  # Dimmed white light from corner


def initialize_camera():
    """Initialize camera parameters for orbit & zoom controls."""
    global orbit_center, orbit_radius, orbit_theta, orbit_phi, mouse_sensitivity, last_mouse_pos

    # Camera orbit parameters - matching initial position looking at center
    orbit_center = [0.5, 0.5, 0.5]  # Center of the lattice

    # Calculate initial angles from the desired initial position
    # close-up start (1.5, 1.2, 1.5) >> relative to center: (1.0, 0.7, 1.0)
    # far-away start (3.67, 2.72, 3.67) >> relative to center: (3.17, 2.22, 3.17)
    initial_rel_x = 3.67 - orbit_center[0]
    initial_rel_y = 2.72 - orbit_center[1]
    initial_rel_z = 3.67 - orbit_center[2]

    # Calculate initial orbit parameters
    orbit_radius = np.sqrt(initial_rel_x**2 + initial_rel_y**2 + initial_rel_z**2)  # ~1.5
    orbit_theta = np.arctan2(initial_rel_z, initial_rel_x)  # 45 degrees
    orbit_phi = np.arccos(initial_rel_y / orbit_radius)  # angle from vertical

    mouse_sensitivity = 0.5
    last_mouse_pos = None


def handle_camera_input():
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
    if window.is_pressed("a"):  # Zoom out
        orbit_radius *= 1.02
        orbit_radius = np.clip(orbit_radius, 0.5, 5.0)


def update_camera():
    """Update camera position based on current orbit parameters.

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
    global block_slice, radius_factor

    # Create overlay windows for stats & controls
    with gui.sub_window("CONTROLS", 0.01, 0.45, 0.20, 0.15) as sub:
        sub.text("Cam Orbit: right-click + drag")
        sub.text("Zoom: Q/A keys")
        block_slice = sub.checkbox("Block Slice", block_slice)
        radius_factor = sub.slider_float("Granule", radius_factor, 0.0, 2.0)
        if sub.button("Reset Granule"):
            radius_factor = 1.0


def render_data_dashboard():
    """Display simulation data dashboard."""
    with gui.sub_window("DATA-DASHBOARD", 0.01, 0.01, 0.24, 0.4) as sub:
        sub.text("--- QUANTUM SPACE (aka: The Aether) ---")
        sub.text("Topology: 3D BCC lattice")
        sub.text(f"Total Granules: {lattice.total_granules:,} (config.py)")
        sub.text(f"Universe Cube Edge: {lattice.universe_edge * constants.ATTO_PREFIX:.1e} m")

        sub.text("")
        sub.text("--- Dynamic Scaling (for computation) ---")
        sub.text(f"Factor: {lattice.scale_factor*constants.ATTO_PREFIX:.1e} x Planck Length")
        sub.text(f"BCC Unit-Cell Edge: {lattice.unit_cell_edge * constants.ATTO_PREFIX:.2e} m")
        sub.text(f"Granule Radius: {granule.radius * constants.ATTO_PREFIX:.2e} m")
        sub.text(f"Granule Mass: {granule.mass * constants.ATTO_PREFIX**3:.2e} kg")

        sub.text("")
        sub.text("--- Simulation Resolution (linear) ---")
        sub.text(f"QWave: {lattice.qwave_res:.0f} granules/qwavelength (min 2)")
        if lattice.qwave_res < 2:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Universe: {lattice.uni_res:.1f} qwaves/universe-edge")

        sub.text("")
        sub.text("--- Cube Wave Energy ---")
        sub.text(f"Energy: {lattice.energy:.1e} J ({lattice.energy_kWh:.1e} KWh)")
        sub.text(f"{lattice.energy_years:,.1e} Years of global energy use")


def render_lattice(lattice, granule):
    """
    Render 3D BCC lattice using GGUI's 3D scene.

    Args:
        lattice: Lattice instance containing positions and universe parameters.
                 Expected to have attributes: positions, total_granules, universe_edge
        granule: Granule instance for size reference.
                 Expected to have attribute: radius (in attometers)


    """
    global orbit_center, orbit_radius, orbit_theta, orbit_phi, mouse_sensitivity, last_mouse_pos
    global block_slice, radius_factor, prev_radius_factor
    global normalized_positions, normalized_radius, normalized_radius_sliced
    global lattice_ref, granule_ref  # Store references for kernel access

    # Store references for kernel access
    lattice_ref = lattice
    granule_ref = granule

    # Normalize granule positions for rendering (0-1 range for GGUI) & block-slicing
    # block-slicing: hide front 1/8th of the lattice for see-through effect
    normalized_positions = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
    normalized_radius = ti.field(dtype=ti.f32, shape=lattice.total_granules)
    normalized_radius_sliced = ti.field(dtype=ti.f32, shape=lattice.total_granules)
    block_slice = True  # Initialize Block-slicing toggle
    radius_factor = 1.0  # Initialize granule size factor
    prev_radius_factor = 1.0  # Track previous value to detect changes

    @ti.kernel
    def normalize_positions():
        """Normalize lattice positions to 0-1 range for GGUI rendering."""
        for i in range(lattice_ref.total_granules):
            # Normalize from attometer scale to 0-1 range
            normalized_positions[i] = lattice_ref.positions[i] / lattice_ref.universe_edge

    @ti.kernel
    def update_radius(factor: ti.f32):
        """Update both radius fields with the given factor."""
        for i in range(lattice_ref.total_granules):
            # granule radius to 0-1 range for GGUI rendering
            base_radius = max(
                granule_ref.radius / lattice_ref.universe_edge * factor, 0.0001
            )  # Show granule with minimum 0.01% of screen radius for visibility

            normalized_radius[i] = base_radius

            # Apply block-slicing for the sliced version
            # Checking if granule is in the front 1/8th block, > halfway on all axes
            if (
                lattice_ref.positions[i][0] > lattice_ref.universe_edge / 2
                and lattice_ref.positions[i][1] > lattice_ref.universe_edge / 2
                and lattice_ref.positions[i][2] > lattice_ref.universe_edge / 2
            ):
                normalized_radius_sliced[i] = 0.0  # Hide granule
            else:
                normalized_radius_sliced[i] = base_radius

    # Normalize positions once before render loop
    print("Normalizing 3D lattice positions to screen...")
    normalize_positions()
    update_radius(radius_factor)

    initialize_scene()  # Set up background once
    initialize_camera()  # Set initial camera parameters

    print("Starting 3D render loop...")

    while window.running:
        setup_scene_lighting()  # Lighting must be set each frame in GGUI

        # Handle input and update camera
        handle_camera_input()
        update_camera()

        # Render UI overlays
        render_controls()
        render_data_dashboard()

        # Update radius fields if factor changed
        if radius_factor != prev_radius_factor:
            update_radius(radius_factor)
            prev_radius_factor = radius_factor

        # Render granules as taichi particles, with block-slicing option
        # Use a base radius value (will be overridden by per_vertex_radius)
        base_radius = granule.radius / lattice.universe_edge * radius_factor
        if block_slice:
            scene.particles(
                normalized_positions,
                radius=base_radius,
                per_vertex_radius=normalized_radius_sliced,
                color=config.COLOR_GRANULE[2],
            )
        else:
            scene.particles(
                normalized_positions,
                radius=base_radius,
                per_vertex_radius=normalized_radius,
                color=config.COLOR_GRANULE[2],
            )

        # Render the scene to canvas
        canvas.scene(scene)
        window.show()


# ==================================================================
# Main calls
# ==================================================================
if __name__ == "__main__":

    # Quantum objects instantiation
    print("\n===============================")
    print("SIMULATION START")
    print("===============================")
    print("Creating quantum objects: lattice and granule...")
    universe_edge = 3e-16  # m (default 300 attometers, contains ~10 qwaves per linear edge)
    lattice = quantum_space.Lattice(universe_edge)
    granule = quantum_space.Granule(lattice.unit_cell_edge)  # already in attometers

    print("\n--- ADDITIONAL-DATA ---")
    print(f"Grid size: {lattice.grid_size} x {lattice.grid_size} x {lattice.grid_size}")
    print(f"  - Corner granules: {(lattice.grid_size + 1) ** 3:,}")
    print(f"  - Center granules: {lattice.grid_size ** 3:,}")

    # Render the 3D lattice
    print("\n--- 3D LATTICE RENDERING ---")
    render_lattice(lattice, granule)  # Pass the already created instances
