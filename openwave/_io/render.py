"""
Rendering engine for OPENWAVE using Taichi GGUI.
"""

import pyautogui

import taichi as ti
import numpy as np

from openwave.common import colormap


def init_UI(universe_size=[1.0, 1.0, 1.0], tick_spacing=0.25, cam_init_pos=[2.0, 2.0, 1.5]):
    """Initialize and open the GGUI window with 3D scene."""
    global window, camera, canvas, gui, scene
    global orbit_theta, orbit_phi, orbit_radius, last_mouse_pos, orbit_center
    global mouse_sensitivity
    global axis_field

    # Get package name and version from source (works with editable installs)
    try:
        from openwave import __version__

        pkg_version = __version__
    except ImportError:
        # Fallback to metadata if __version__ not available
        from importlib.metadata import version

        pkg_version = version("OPENWAVE")

    pkg_name = "OPENWAVE"
    title = pkg_name + " (v" + pkg_version + ")"
    width, height = pyautogui.size()
    # width, height = 1470, 884  # uncomment to test on min supported resolution

    window = ti.ui.Window(title, (width, height), vsync=True)
    camera = ti.ui.Camera()  # Camera object for 3D view control
    canvas = window.get_canvas()  # Canvas for rendering the scene
    gui = window.get_gui()  # GUI manager for overlay UI elements
    scene = window.get_scene()  # 3D scene for rendering

    # Set initial camera parameters & background color
    # Camera orbit parameters - initial position looking at center
    # Compute orbit center proportional to asymmetric universe (normalized by max dimension)
    max_dim = max(universe_size[0], universe_size[1], universe_size[2])
    orbit_center = [
        0.5 * (universe_size[0] / max_dim),  # X center
        0.5 * (universe_size[1] / max_dim),  # Y center
        0.5 * (universe_size[2] / max_dim),  # Z center
    ]

    # Calculate initial angles from the desired initial position
    init_rel_x = cam_init_pos[0] - orbit_center[0]
    init_rel_y = cam_init_pos[1] - orbit_center[1]
    init_rel_z = cam_init_pos[2] - orbit_center[2]

    # Calculate initial orbit parameters
    orbit_radius = np.sqrt(init_rel_x**2 + init_rel_y**2 + init_rel_z**2)  # ~1.5

    # Prevent division by zero and handle edge cases
    if orbit_radius < 1e-6:
        # Camera at exact center - use default position
        orbit_radius = 1.0
        orbit_theta = 0.0
        orbit_phi = np.pi / 4  # 45 degrees default
    else:
        # Handle arctan2(0, 0) case when camera is directly above/below center (Z-up system)
        # Also handle gimbal lock when phi is too close to 0 or pi
        if abs(init_rel_x) < 1e-6 and abs(init_rel_y) < 1e-6:
            # Camera directly above/below - add small offset to avoid gimbal lock
            orbit_theta = 0.0
            orbit_phi = np.arccos(np.clip(init_rel_z / orbit_radius, -1.0, 1.0))
            # If phi is too close to 0 (top) or pi (bottom), add small offset
            if orbit_phi < 0.01:
                orbit_phi = 0.01  # Minimum angle from vertical
            elif orbit_phi > np.pi - 0.01:
                orbit_phi = np.pi - 0.01  # Maximum angle from vertical
        else:
            orbit_theta = np.arctan2(init_rel_y, init_rel_x)
            orbit_phi = np.arccos(np.clip(init_rel_z / orbit_radius, -1.0, 1.0))

    mouse_sensitivity = 0.5
    last_mouse_pos = None
    canvas.set_background_color(colormap.COLOR_SPACE[1])

    # Initialize axis field only once (tick_spacing is constant per session)
    tick_width = 0.01
    num_ticks = int(2.0 / tick_spacing) + 1
    # Calculate total number of points: 6 for axis lines + (num_ticks-1) * 3 axes * 2 points per tick
    total_points = 6 + (num_ticks - 1) * 3 * 2
    axis_field = ti.Vector.field(3, dtype=ti.f32, shape=total_points)
    populate_axis_field(tick_spacing, tick_width, num_ticks)


@ti.kernel
def populate_axis_field(
    tick_spacing: ti.f32, tick_width: ti.f32, num_ticks: ti.i32  # type: ignore
):
    """Populate axis field using Taichi kernel for performance."""
    # Axis lines (6 points = 3 axes * 2 endpoints each)
    axis_field[0] = ti.Vector([0.0, 0.0, 0.0])  # X-axis start
    axis_field[1] = ti.Vector([2.0, 0.0, 0.0])  # X-axis end
    axis_field[2] = ti.Vector([0.0, 0.0, 0.0])  # Y-axis start
    axis_field[3] = ti.Vector([0.0, 2.0, 0.0])  # Y-axis end
    axis_field[4] = ti.Vector([0.0, 0.0, 0.0])  # Z-axis start
    axis_field[5] = ti.Vector([0.0, 0.0, 2.0])  # Z-axis end

    # Tick marks (parallel loop for performance)
    for t in range(1, num_ticks):
        offset = t * tick_spacing
        idx_base = 6 + (t - 1) * 6  # Starting index for this tick's 6 points

        # X-axis ticks (2 points)
        axis_field[idx_base + 0] = ti.Vector([offset, -tick_width, 0.0])
        axis_field[idx_base + 1] = ti.Vector([offset, 0.0, 0.0])

        # Y-axis ticks (2 points)
        axis_field[idx_base + 2] = ti.Vector([-tick_width, offset, 0.0])
        axis_field[idx_base + 3] = ti.Vector([0.0, offset, 0.0])

        # Z-axis ticks (2 points)
        axis_field[idx_base + 4] = ti.Vector([-tick_width, 0.0, offset])
        axis_field[idx_base + 5] = ti.Vector([0.0, 0.0, offset])


def setup_scene_lighting():
    """Set up scene lighting - must be called every frame in GGUI."""
    scene.ambient_light((0.1, 0.1, 0.1))  # Slight blue ambient
    scene.point_light(pos=(1.1, 1.1, 1.1), color=(1.0, 1.0, 1.0))  # White light
    scene.point_light(pos=(0.5, 1.1, 0.5), color=(0.5, 0.5, 0.5))  # Dimmed white light


def handle_camera():
    """Handle mouse and keyboard input for camera controls."""
    global orbit_theta, orbit_phi, orbit_radius, last_mouse_pos, orbit_center
    global cam_x, cam_y, cam_z

    # Handle mouse input for orbiting
    mouse_pos = window.get_cursor_pos()

    # Check for orbit controls: right-click drag OR shift+left-click drag (for trackpads)
    is_orbiting = window.is_pressed(ti.ui.RMB) or (
        window.is_pressed(ti.ui.LMB) and window.is_pressed(ti.ui.SHIFT)
    )

    if is_orbiting:
        if last_mouse_pos is not None:
            # Calculate mouse delta
            dx = mouse_pos[0] - last_mouse_pos[0]
            dy = mouse_pos[1] - last_mouse_pos[1]

            # Update orbit angles (negative sign makes object rotate in same direction as mouse movement)
            orbit_theta += -dx * mouse_sensitivity * 2 * np.pi  # Horizontal orbit
            # Allow nearly full vertical rotation (almost 180 degrees, small margin to prevent gimbal lock)
            orbit_phi = np.clip(
                orbit_phi + dy * mouse_sensitivity * np.pi, 0.01, np.pi - 0.01
            )  # Vertical orbit

        last_mouse_pos = mouse_pos
    else:
        last_mouse_pos = None

    # Handle keyboard input for zoom
    if window.is_pressed("q"):  # Zoom in
        orbit_radius *= 0.98
        orbit_radius = np.clip(orbit_radius, 0.1, 5.0)
    if window.is_pressed("z"):  # Zoom out
        orbit_radius *= 1.02
        orbit_radius = np.clip(orbit_radius, 0.1, 5.0)

    # Handle keyboard input for panning
    # Pan amount scales with orbit radius for consistent feel at different zoom levels
    pan_speed = 0.01 * orbit_radius

    # UP/DOWN arrows: Pan vertically (Z-axis)
    if window.is_pressed(ti.ui.UP):  # Pan up (Z+)
        orbit_center[2] += pan_speed
    if window.is_pressed(ti.ui.DOWN):  # Pan down (Z-)
        orbit_center[2] -= pan_speed

    # LEFT/RIGHT arrows: Pan sideways relative to camera view (screen-space horizontal)
    # Compute camera's "right" vector in XY plane based on current theta angle
    if window.is_pressed(ti.ui.LEFT):  # Pan left (screen-space)
        # Right vector is perpendicular to camera direction in XY plane
        right_x = -np.sin(orbit_theta)  # Perpendicular to camera direction
        right_y = np.cos(orbit_theta)
        orbit_center[0] -= right_x * pan_speed
        orbit_center[1] -= right_y * pan_speed
    if window.is_pressed(ti.ui.RIGHT):  # Pan right (screen-space)
        right_x = -np.sin(orbit_theta)
        right_y = np.cos(orbit_theta)
        orbit_center[0] += right_x * pan_speed
        orbit_center[1] += right_y * pan_speed

    # Now update camera position based on current orbit parameters.
    # Converts spherical coordinates (orbit_radius, orbit_theta, orbit_phi) to
    # Cartesian coordinates and updates the camera position and orientation.
    # Camera always looks at orbit_center with Z-axis up.
    cam_x = orbit_center[0] + orbit_radius * np.sin(orbit_phi) * np.cos(orbit_theta)
    cam_y = orbit_center[1] + orbit_radius * np.sin(orbit_phi) * np.sin(orbit_theta)
    cam_z = orbit_center[2] + orbit_radius * np.cos(orbit_phi)

    camera.position(cam_x, cam_y, cam_z)
    camera.lookat(orbit_center[0], orbit_center[1], orbit_center[2])
    camera.up(0, 0, 1)
    scene.set_camera(camera)


def display_cam_instructions():
    """Display camera movement instructions."""
    global cam_x, cam_y, cam_z
    with gui.sub_window("CAMERA MOVEMENT", 0.00, 0.88, 0.14, 0.12) as sub:
        sub.text("Orbit: RMB or Shift+LMB")
        sub.text("Pan: Arrow keys")
        sub.text("Zoom: Q/Z keys")
        sub.text("Cam Pos: %.2f, %.2f, %.2f" % (cam_x, cam_y, cam_z), color=colormap.LIGHT_BLUE[1])


def init_scene(show_axis=True):
    """Initialize the 3D scene with lighting and camera controls."""
    global axis_field

    setup_scene_lighting()  # Lighting must be set each frame in GGUI
    handle_camera()  # Handle camera input and update position
    display_cam_instructions()  # Display camera movement instructions
    if show_axis:
        # Render the pre-populated axis field (very fast, no data transfer)
        scene.lines(axis_field, color=colormap.COLOR_INFRA[1], width=2)


def show_scene():
    """Render the 3D scene."""
    canvas.scene(scene)
    window.show()
