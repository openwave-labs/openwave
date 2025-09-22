"""
QUANTUM SPACE
(AKA: AKASHA @yoga, WUJI @daoism, AETHER @classical)

QUANTUM SPACE is a Wave-Medium and Propagates Wave-Motions (QUANTUM WAVE).
Modeled as a particle-based elastic ideal fluid (plasma like),
that allows energy to transfer from one point to the next.

"Aether" can refer to the personification of the bright upper sky in Greek mythology,
the classical fifth element or quintessence filling the universe,
or a hypothetical substance once thought to carry light and other electromagnetic waves.
"""

import numpy as np
import taichi as ti

import openwave.core.config as config
import openwave.core.constants as constants

ti.init(arch=ti.gpu)

UNIVERSE_RADIUS = 1e-16  # m, spherical universe radius
SCREEN_WIDTH = 900  # pixels
SCREEN_HEIGHT = 900  # pixels

# INCLUDE: knob, remove granule scale @config.py, limit scale_factor ranges some place
# granule_scale = 1e-18


class Granule:
    # Granule Model: The aether consists of "granules".
    # Fundamental units that vibrate and create wave patterns.
    # Their collective motion at Planck scale creates all observable phenomena.
    def __init__(self, scale_factor):
        self.radius = constants.PLANCK_LENGTH * scale_factor  # m


@ti.data_oriented
class Lattice2D:
    # Granule Count on Lattice: Potentially trillions of granules requiring
    # spring constant calculations, harmonic motion, and wave propagation
    def __init__(self, scale_factor):
        self.size = UNIVERSE_RADIUS
        self.spacing = 2 * constants.PLANCK_LENGTH * scale_factor * np.e
        self.count = int(self.size / self.spacing)

    def granule_positions(self):
        # taichi: use taichi primitive types
        self.grid = ti.Vector.field(2, dtype=float, shape=(self.count, self.count))
        self._populate_grid()
        return self.grid

    @ti.kernel
    def _populate_grid(self):
        # taichi: parallelized for loop
        for i, j in self.grid:
            self.grid[i, j] = ti.Vector([i * self.spacing, j * self.spacing])


# INCLUDE: Implement RENDER module or class


def universe_to_screen(universe_pos, universe_size):
    """Convert universe coordinates to normalized screen coordinates [0,1]"""
    return universe_pos / universe_size


def render_lattice():
    """Render the granule lattice in 2D GUI"""
    # Create GUI
    gui = ti.GUI("Quantum Granule Lattice", (SCREEN_WIDTH, SCREEN_HEIGHT))
    scale = gui.slider("Granule Scale", -19, -17, step=1)

    # Track previous scale to detect changes
    previous_scale = scale.value

    # Initialize with default scale
    scale_factor = 10**scale.value / 10**-35
    granule = Granule(scale_factor)
    lattice = Lattice2D(scale_factor)
    positions = lattice.granule_positions()

    while gui.running:
        # Check if scale changed - update objects if needed
        if scale.value != previous_scale:
            scale_factor = 10**scale.value / 10**-35
            granule = Granule(scale_factor)
            lattice = Lattice2D(scale_factor)
            positions = lattice.granule_positions()
            previous_scale = scale.value

        # Calculate screen radius for current granule size
        universe_to_screen_ratio = min(SCREEN_WIDTH, SCREEN_HEIGHT) / lattice.size
        screen_radius = granule.radius * universe_to_screen_ratio

        # Ensure minimum visible radius
        min_radius = 1  # pixels
        screen_radius = max(screen_radius, min_radius)

        # Create display offset to center the lattice
        offset = (lattice.size - lattice.spacing * (lattice.count - 1)) / 2

        # Clear to black background
        gui.clear(0x000000)

        # Draw granules
        # INCLUDE: Implement taichi decorator for performance boost
        # INCLUDE: Optimize with Taichi loop root level parallelization (for i, j in range)
        for i in range(lattice.count):
            for j in range(lattice.count):
                universe_pos = positions[i, j]
                # Convert to normalized screen coordinates and apply offset
                screen_x = (universe_pos[0] + offset) / lattice.size
                screen_y = (universe_pos[1] + offset) / lattice.size

                # Draw granule as white circle
                gui.circle([screen_x, screen_y], color=0xFFFFFF, radius=int(screen_radius))

        gui.show()


# Render the lattice
render_lattice()

# Print info, only used when no slider widget
# print(f"Universe size: {lattice.size:.2e} m")
# print(f"Granule radius: {granule.radius:.2e} m")
# print(f"Lattice spacing: {lattice.spacing:.2e} m")
# print(f"Lattice count: {lattice.count}x{lattice.count}")
