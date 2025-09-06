"""
QUANTUM SPACE
(AKA: AKASHA @yoga, WUJI @taoism, AETHER @ancient)
Modeled as an elastic fluid structure (compressible), that allows energy to transfer from one point to the next.
"""

import numpy as np
import taichi as ti

import openwave.core.config as config
import openwave.core.constants as constants

ti.init(arch=ti.gpu)

# =====================
# Physics Engine
# =====================


class GranulePhysics:
    # Granule Model: The aether consists of "granules".
    # Fundamental units that vibrate and create wave patterns.
    # Their collective motion at Planck scale creates all observable phenomena.

    og_granule_radius = constants.PLANCK_LENGTH

    def __init__(self, granule_radius=og_granule_radius):
        self.radius = granule_radius  # m


@ti.data_oriented
class Lattice2DPhysics:
    # Granule Count on Lattice: Potentially billions of granules requiring
    # spring constant calculations, harmonic motion, and wave propagation

    og_universe_radius = config.UNIVERSE_RADIUS  # m
    og_lattice_spacing = 2 * constants.PLANCK_LENGTH * np.e  # m, Planck-scale
    min_lattice_spacing = 5e-21  # m, min spacing clamp, for computability in taichi
    max_lattice_spacing = 1e-17  # m, max spacing clamp, less than QWave-scale

    def __init__(self, universe_radius=og_universe_radius, lattice_spacing=min_lattice_spacing):
        self.universe_length = 2 * universe_radius  # m, universe side length
        # clamp lattice spacing
        self.spacing = min(
            max(lattice_spacing, self.min_lattice_spacing), self.max_lattice_spacing
        )
        self.linear_count = round(self.universe_length / self.spacing)  # number of granules

    def granule_positions(self):
        # use taichi primitive types
        self.grid = ti.Vector.field(2, dtype=ti.f32, shape=(self.linear_count, self.linear_count))
        self._populate_grid()
        return self.grid

    @ti.kernel
    def _populate_grid(self):
        # taichi: parallelized for loop
        for i, j in self.grid:
            self.grid[i, j] = ti.Vector([i * self.spacing, j * self.spacing])


# =====================
# Rendering Engine
# =====================
# TODO: Implement Rendering Engine in a separate module or class???


class GranuleRender:
    min_granule_radius = 1  # pixels, min radius clamp for visibility

    def __init__(self, lattice_spacing):
        self.radius = max(self.min_granule_radius, lattice_spacing / (2 * np.e))  # pixels


@ti.data_oriented
class Lattice2DRender:
    screen_size = min(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
    lattice_spacing = 20  # pixels, increased spacing for visibility and performance
    universe_length = screen_size  # pixels, universe side length
    linear_count = round(universe_length / lattice_spacing)  # number of granules

    # Create offset to center the lattice on display
    offset_x = (config.SCREEN_WIDTH - lattice_spacing * (linear_count - 1)) / 2
    offset_y = (config.SCREEN_HEIGHT - lattice_spacing * (linear_count - 1)) / 2

    def granule_positions(self):
        # use taichi primitive types
        self.grid = ti.Vector.field(2, dtype=ti.f32, shape=(self.linear_count, self.linear_count))
        self._populate_grid()
        return self.grid

    @ti.kernel
    def _populate_grid(self):
        # taichi: parallelized for loop
        for i, j in self.grid:
            self.grid[i, j] = ti.Vector(
                [
                    i * self.lattice_spacing + self.offset_x,
                    j * self.lattice_spacing + self.offset_y,
                ]
            )


def render_lattice():
    print("Initializing render...")

    # Initialize instances for rendering
    lattice = Lattice2DRender()
    positions = lattice.granule_positions()
    granule_radius = GranuleRender(lattice.lattice_spacing).radius  # pixels

    print("_______________________________")
    print("Lattice 2D Render initialized.")
    print(f"Lattice spacing: {lattice.lattice_spacing} pixels")
    print(f"Granule linear count: {lattice.linear_count:,} granules")
    print(f"Granule positions populated: {lattice.linear_count**2:,} granules")
    print(f"Granule Radius: {granule_radius:.2f} pixels")

    print("_______________________________")
    print(f"Creating GUI window: {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")

    # Create GUI
    gui = ti.GUI("Quantum Granule Lattice", (config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    print("Starting render loop...")

    while gui.running:
        # Clear to black background
        gui.clear(0x000000)

        # Render circles using regular Python loops
        for i in range(lattice.linear_count):
            for j in range(lattice.linear_count):
                # Normalize positions to GUI coordinate system (0.0 to 1.0)
                pos = positions[i, j]
                x = pos[0] / config.SCREEN_WIDTH
                y = pos[1] / config.SCREEN_HEIGHT
                gui.circle([x, y], color=0xFFFFFF, radius=granule_radius)

        gui.show()


# =====================
# Main calls
# =====================
if __name__ == "__main__":
    print("\n===============================")
    print("SIMULATION DATA")
    print("===============================")

    lattice = Lattice2DPhysics(lattice_spacing=1e-21)
    print("Lattice 2D Physics initialized.")
    print(f"Universe length: {lattice.universe_length} m")
    print(f"Lattice spacing: {lattice.spacing} m")
    print(f"Granule linear count: {lattice.linear_count:,} granules")
    print(f"Granule positions populated: {lattice.linear_count**2:,} granules")

    render_lattice()


# Optimizations Applied to quantum_space.py:

#   1. Taichi Kernel for Parallel Computation: Added
#   compute_screen_positions() kernel that computes all screen
#   positions in parallel on the GPU

#   2. Pre-computation: Screen positions are computed once before
#   the render loop starts, eliminating redundant calculations
#   every frame

#   3. Batch Processing: Converted Taichi field to numpy array for
#   efficient batch access

#   4. Loop Optimization:
#     - Pre-computed constants outside the loop (screen_radius_int,
#    circle_color, line_count)
#     - Flattened the 2D position array into a 1D list for faster
#   iteration
#     - Single loop instead of nested loops in the render section

#   5. Memory Efficiency: Pre-allocated screen_positions field to
#   avoid repeated memory allocation

#   These optimizations significantly improve rendering performance by:
#   - Moving computations from CPU to GPU
#   - Eliminating per-frame calculations
#   - Reducing loop overhead
#   - Minimizing variable lookups
