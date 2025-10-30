"""
Rendering engine for OPENWAVE using Taichi GGUI.
"""

from importlib.metadata import version, metadata

import taichi as ti
import numpy as np
import pyautogui

from openwave.common import config


def init_UI(universe_size=[1.0, 1.0, 1.0], tick_spacing=0.25, cam_init_pos=[2.0, 2.0, 1.5]):
    """Initialize and open the GGUI window with 3D scene."""
    global window, camera, canvas, gui, scene
    global orbit_theta, orbit_phi, orbit_radius, last_mouse_pos, orbit_center
    global mouse_sensitivity
    global axis_field

    # Get package name and version from metadata
    pkg_name = metadata("OPENWAVE")["Name"]
    pkg_version = version("OPENWAVE")
    title = pkg_name + " (v" + pkg_version + ")"
    width, height = pyautogui.size()

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
    canvas.set_background_color(config.COLOR_SPACE[1])

    # Initialize axis field only once (tick_spacing is constant per session)
    # Use particles instead of lines for better performance (scene.particles() is much faster than scene.lines())
    tick_width = 0.02  # Length of tick marks
    num_ticks = int(2.0 / tick_spacing)

    # Total: 3 axes * points_per_axis + 3 axes * num_ticks * points_per_tick
    points_per_axis = 1000  # Dense point sampling to make solid-looking lines
    points_per_tick = 50  # Points per tick mark
    total_points = 3 * points_per_axis + 3 * num_ticks * points_per_tick

    axis_field = ti.Vector.field(3, dtype=ti.f32, shape=total_points)
    populate_axis_field(tick_spacing, tick_width, num_ticks, points_per_axis, points_per_tick)


@ti.kernel
def populate_axis_field(
    tick_spacing: ti.f32,  # type: ignore
    tick_width: ti.f32,  # type: ignore
    num_ticks: ti.i32,  # type: ignore
    points_per_axis: ti.i32,  # type: ignore
    points_per_tick: ti.i32,  # type: ignore
):
    """Populate axis field as particles instead of lines for better performance."""
    axis_length = 2.0

    # X-axis particles (dense points from 0 to 2.0)
    for i in range(points_per_axis):
        t = i / (points_per_axis - 1)
        axis_field[i] = ti.Vector([t * axis_length, 0.0, 0.0])

    # Y-axis particles
    offset = points_per_axis
    for i in range(points_per_axis):
        t = i / (points_per_axis - 1)
        axis_field[offset + i] = ti.Vector([0.0, t * axis_length, 0.0])

    # Z-axis particles
    offset = 2 * points_per_axis
    for i in range(points_per_axis):
        t = i / (points_per_axis - 1)
        axis_field[offset + i] = ti.Vector([0.0, 0.0, t * axis_length])

    # Tick marks as particles (from -tick_width to 0.0 for each axis)
    offset = 3 * points_per_axis
    for tick_idx in range(num_ticks):
        tick_pos = (tick_idx + 1) * tick_spacing
        base_idx = offset + tick_idx * 3 * points_per_tick

        # X-axis tick (varies in Y from -tick_width to 0.0)
        for i in range(points_per_tick):
            t = i / (points_per_tick - 1) if points_per_tick > 1 else 0.5
            y = -tick_width + t * tick_width  # From -tick_width to 0.0
            axis_field[base_idx + i] = ti.Vector([tick_pos, y, 0.0])

        # Y-axis tick (varies in X from -tick_width to 0.0)
        for i in range(points_per_tick):
            t = i / (points_per_tick - 1) if points_per_tick > 1 else 0.5
            x = -tick_width + t * tick_width  # From -tick_width to 0.0
            axis_field[base_idx + points_per_tick + i] = ti.Vector([x, tick_pos, 0.0])

        # Z-axis tick (varies in X from -tick_width to 0.0)
        for i in range(points_per_tick):
            t = i / (points_per_tick - 1) if points_per_tick > 1 else 0.5
            x = -tick_width + t * tick_width  # From -tick_width to 0.0
            axis_field[base_idx + 2 * points_per_tick + i] = ti.Vector([x, 0.0, tick_pos])


def scene_lighting():
    """Set up scene lighting - must be called every frame in GGUI."""
    scene.ambient_light((0.1, 0.1, 0.15))  # Slight blue ambient
    scene.point_light(pos=(0.5, 0.5, 1.5), color=(1.0, 1.0, 1.0))  # White light from above center
    scene.point_light(pos=(1.2, 1.2, 1.2), color=(0.5, 0.5, 0.5))  # Dimmed white light from corner


def handle_camera():
    """Handle mouse and keyboard input for camera controls."""
    global orbit_theta, orbit_phi, orbit_radius, last_mouse_pos, orbit_center
    global cam_x, cam_y, cam_z

    # Handle mouse input for orbiting
    mouse_pos = window.get_cursor_pos()

    # Check for right mouse button drag
    if window.is_pressed(ti.ui.RMB):
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


def cam_instructions():
    """Overlay camera movement instructions."""
    global cam_x, cam_y, cam_z
    with gui.sub_window("CAMERA MOVEMENT", 0.87, 0.88, 0.13, 0.12) as sub:
        sub.text("Orbit: right-click + drag")
        sub.text("Zoom: Q/Z keys")
        sub.text("Pan: Arrow keys")
        sub.text("Cam Pos: %.2f, %.2f, %.2f" % (cam_x, cam_y, cam_z))


def init_scene(show_axis=True):
    """Initialize the 3D scene with lighting and camera controls."""
    global axis_field

    scene_lighting()  # Lighting must be set each frame in GGUI
    handle_camera()  # Handle camera input and update position
    cam_instructions()  # Overlay camera instructions
    if show_axis:
        # Render axis as particles instead of lines (much faster, uses same efficient path as granules)
        scene.particles(axis_field, radius=0.001, color=config.COLOR_INFRA[1])


def show_scene():
    """Render the 3D scene."""
    canvas.scene(scene)
    window.show()
