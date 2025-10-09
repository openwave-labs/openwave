"""
Rendering engine for OPENWAVE using Taichi GGUI.
"""

import numpy as np
import taichi as ti
import pyautogui
import tomli

from openwave.common import config


def init_UI():
    """Initialize and open the GGUI window with 3D scene."""
    global window, camera, canvas, gui, scene
    global orbit_theta, orbit_phi, orbit_radius, last_mouse_pos, orbit_center
    global mouse_sensitivity

    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)

    title = pyproject["project"]["name"] + " (v" + pyproject["project"]["version"] + ")"
    width, height = pyautogui.size()

    window = ti.ui.Window(title, (width, height), vsync=True, fps_limit=120)
    camera = ti.ui.Camera()  # Camera object for 3D view control
    canvas = window.get_canvas()  # Canvas for rendering the scene
    gui = window.get_gui()  # GUI manager for overlay UI elements
    scene = window.get_scene()  # 3D scene for particle rendering

    # Set initial camera parameters & background color
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
    canvas.set_background_color(config.COLOR_SPACE[1])


def scene_lighting():
    """Set up scene lighting - must be called every frame in GGUI."""
    scene.ambient_light((0.1, 0.1, 0.15))  # Slight blue ambient
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1.0, 1.0, 1.0))  # White light from above center
    scene.point_light(pos=(1.0, 0.5, 1.0), color=(0.5, 0.5, 0.5))  # Dimmed white light from corner


def handle_camera():
    """Handle mouse and keyboard input for camera controls."""
    global orbit_theta, orbit_phi, orbit_radius, last_mouse_pos, orbit_center

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
    if window.is_pressed(ti.ui.UP):  # Tilt up
        orbit_center[1] += 0.01 * orbit_radius
    if window.is_pressed(ti.ui.DOWN):  # Tilt down
        orbit_center[1] -= 0.01 * orbit_radius
    if window.is_pressed(ti.ui.LEFT):  # Pan left
        orbit_center[0] -= 0.01 * orbit_radius
    if window.is_pressed(ti.ui.RIGHT):  # Pan right
        orbit_center[0] += 0.01 * orbit_radius

    # Now update camera position based on current orbit parameters.
    # Converts spherical coordinates (orbit_radius, orbit_theta, orbit_phi) to
    # Cartesian coordinates and updates the camera position and orientation.
    # Camera always looks at orbit_center with Y-axis up.
    cam_x = orbit_center[0] + orbit_radius * np.sin(orbit_phi) * np.cos(orbit_theta)
    cam_y = orbit_center[1] + orbit_radius * np.cos(orbit_phi)
    cam_z = orbit_center[2] + orbit_radius * np.sin(orbit_phi) * np.sin(orbit_theta)
    camera.position(cam_x, cam_y, cam_z)
    camera.lookat(orbit_center[0], orbit_center[1], orbit_center[2])
    camera.up(0, 1, 0)
    scene.set_camera(camera)


def cam_instructions():
    """Overlay camera movement instructions."""
    with gui.sub_window("CAMERA MOVEMENT", 0.01, 0.90, 0.13, 0.10) as sub:
        sub.text("Orbit: right-click + drag")
        sub.text("Zoom: Q/Z keys")
        sub.text("Pan/Tilt: Arrow keys")


def show_scene():
    scene_lighting()  # Lighting must be set each frame in GGUI
    handle_camera()  # Handle camera input and update position
    canvas.scene(scene)
    window.show()
