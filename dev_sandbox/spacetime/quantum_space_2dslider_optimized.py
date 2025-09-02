"""
QUANTUM SPACE - OPTIMIZED VERSION
(AKA: AKASHA @yoga, WUJI @taoism, AETHER @ancient)
Modeled as an elastic fluid structure (compressible), that allows energy to transfer from one point to the next.
"""

import numpy as np
import taichi as ti

import openwave.core.config as config
import openwave.core.constants as constants

ti.init(arch=ti.gpu)


class Granule:
    def __init__(self, scale_factor):
        self.radius = constants.PLANCK_LENGTH * scale_factor  # m

    def update_scale(self, scale_factor):
        self.radius = constants.PLANCK_LENGTH * scale_factor


@ti.data_oriented
class Lattice2D:
    def __init__(self, scale_factor):
        self.size = config.UNIVERSE_SIZE
        self.max_count = 1000  # Pre-allocate max size
        self.grid = ti.Vector.field(
            2, dtype=ti.f32, shape=(self.max_count, self.max_count)
        )
        self.screen_pos = ti.Vector.field(
            2, dtype=ti.f32, shape=(self.max_count, self.max_count)
        )
        self.update_scale(scale_factor)

    def update_scale(self, scale_factor):
        self.spacing = 2 * constants.PLANCK_LENGTH * scale_factor * np.e
        self.count = min(int(self.size / self.spacing), self.max_count)
        self._populate_grid()

    @ti.kernel
    def _populate_grid(self):
        for i, j in ti.ndrange(self.count, self.count):
            self.grid[i, j] = ti.Vector([i * self.spacing, j * self.spacing])

    @ti.kernel
    def compute_screen_positions(self, offset: float, size: float):
        for i, j in ti.ndrange(self.count, self.count):
            universe_pos = self.grid[i, j]
            self.screen_pos[i, j][0] = (universe_pos[0] + offset) / size
            self.screen_pos[i, j][1] = (universe_pos[1] + offset) / size

    def get_screen_positions_numpy(self):
        return self.screen_pos.to_numpy()[: self.count, : self.count]


def render_lattice_optimized():
    """Optimized render of the granule lattice in 2D GUI"""
    gui = ti.GUI(
        "Quantum Granule Lattice - Optimized",
        (config.SCREEN_WIDTH, config.SCREEN_HEIGHT),
    )
    scale = gui.slider("Granule Scale", -19, -17, step=1)

    # Track previous scale to detect changes
    previous_scale = scale.value

    # Initialize with default scale
    scale_factor = 10**scale.value / 10**-35
    granule = Granule(scale_factor)
    lattice = Lattice2D(scale_factor)

    # Pre-compute constants
    universe_to_screen_ratio = (
        min(config.SCREEN_WIDTH, config.SCREEN_HEIGHT) / lattice.size
    )
    screen_radius = max(granule.radius * universe_to_screen_ratio, 1)
    offset = (lattice.size - lattice.spacing * (lattice.count - 1)) / 2

    while gui.running:
        # Check if scale changed - update objects if needed
        if scale.value != previous_scale:
            scale_factor = 10**scale.value / 10**-35

            # Update existing objects instead of recreating
            granule.update_scale(scale_factor)
            lattice.update_scale(scale_factor)

            # Recalculate cached values
            universe_to_screen_ratio = (
                min(config.SCREEN_WIDTH, config.SCREEN_HEIGHT) / lattice.size
            )
            screen_radius = max(granule.radius * universe_to_screen_ratio, 1)
            offset = (lattice.size - lattice.spacing * (lattice.count - 1)) / 2

            previous_scale = scale.value

        # Compute screen positions in parallel on GPU
        lattice.compute_screen_positions(offset, lattice.size)

        # Clear to black background
        gui.clear(0x000000)

        # Get all positions at once and draw in batch
        screen_positions = lattice.get_screen_positions_numpy()

        # Batch draw all circles
        for i in range(lattice.count):
            for j in range(lattice.count):
                gui.circle(
                    screen_positions[i, j], color=0xFFFFFF, radius=int(screen_radius)
                )

        gui.show()


if __name__ == "__main__":
    render_lattice_optimized()
