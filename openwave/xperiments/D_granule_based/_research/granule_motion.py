import taichi as ti
import numpy as np

from openwave.common import colormap, constants
from openwave._io import render


# Initialize Taichi
ti.init(arch=ti.gpu)  # GPU preferred

# Energy_wave physical parameters (in attometers and Hz)
amplitude = 0.1  # normalized to screen amplitude [0,1]
frequency = 1  # in Hz
omega = 2 * np.pi * frequency  # angular frequency
wavelength = 0.3  # normalized to screen wavelength [0,1]


# Initialize sample granule positions
num_granules = 2
position = ti.Vector.field(3, dtype=ti.f32, shape=num_granules)
color = ti.Vector.field(3, dtype=ti.f32, shape=num_granules)
num_trails = 12
trails = ti.Vector.field(3, dtype=ti.f32, shape=(num_trails))  # motion trails for granule1

# Initialize time variables
dt = 0.01  # time step
elapsed_time = 0.0  # time accumulator
frame = 1  # frame counter


def compute_temporal_motion():
    """Compute temporal motion of granules."""
    # Compute harmonic motion for granule1
    position[0][0] = 0.5 + amplitude * np.cos(omega * elapsed_time)
    position[0][1] = 0.5 + amplitude * np.cos(omega * elapsed_time + np.pi / 3)
    position[0][2] = 0.5 + amplitude * np.cos(omega * elapsed_time + np.pi / 3 * 2)
    color[0] = colormap.COLOR_PARTICLE[1]

    # Compute harmonic motion for granule2
    position[1][0] = 0.5 + amplitude * np.cos(omega * elapsed_time)
    position[1][1] = 0.7 + amplitude * np.cos(omega * elapsed_time + np.pi / 2)  # Phase shift
    position[1][2] = 0.5 + amplitude * np.cos(omega * elapsed_time + np.pi / 2)
    color[1] = colormap.COLOR_ANTI[1]


def compute_motion_trail():
    """Compute motion trail of granules."""
    for i in range(num_trails):
        t = elapsed_time - i * dt * 6  # Trail time offset
        trails[i][0] = amplitude * np.cos(omega * t)
        trails[i][1] = amplitude * np.cos(omega * t + np.pi / 3)
        trails[i][2] = amplitude / 2 * np.cos(omega * t + np.pi / 3 * 2)


def main():
    global elapsed_time, frame

    # Initialize GGUI rendering
    render.init_UI(cam_init_pos=[0.3, 0.3, 0.3], cam_look_center=[0.0, 0.0, 0.0])

    # Main rendering loop
    while render.window.running:
        render.init_scene()  # Initialize scene with lighting and camera

        # Handle ESC key for window close
        if render.window.is_pressed(ti.ui.ESCAPE):
            render.window.running = False
            break

        # Update granule positions
        # compute_temporal_motion()
        compute_motion_trail()

        # Update time variables
        elapsed_time += dt  # Accumulate simulation time
        frame += 1

        # Render scene elements
        # render.scene.particles(position, radius=0.01, per_vertex_color=color)
        render.scene.particles(trails, radius=0.01, color=colormap.COLOR_PARTICLE[1])

        # Display scene
        render.show_scene()


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()
