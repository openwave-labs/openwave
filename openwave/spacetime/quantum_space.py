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
# Universe Data Section
# =====================

granule_scale_factor = 1e-20 / 1e-35
spacing_scale_factor = 1e-20 / 1e-35
# TODO: implement scale factor clamping
GRANULE_SCALE_MIN = 1e-35  # min granule scale (Planck scale)
GRANULE_SCALE_MAX = 1e-17  # max granule scale (QWave scale)


class Granule:
    # Granule Model: The aether consists of "granules".
    # Fundamental units that vibrate and create wave patterns.
    # Their collective motion at Planck scale creates all observable phenomena.
    def __init__(self, granule_scale_factor):
        self.radius = constants.PLANCK_LENGTH * granule_scale_factor  # m


@ti.data_oriented
class Lattice2D:
    # Granule Count on Lattice: Potentially billions of granules requiring
    # spring constant calculations, harmonic motion, and wave propagation
    def __init__(self, spacing_scale_factor):
        self.line_size = config.UNIVERSE_RADIUS
        self.spacing = 2 * constants.PLANCK_LENGTH * np.e * spacing_scale_factor
        self.line_count = int(self.line_size / self.spacing)

    def granule_positions(self):
        # taichi: use taichi primitive types
        self.grid = ti.Vector.field(
            2, dtype=ti.f32, shape=(self.line_count, self.line_count)
        )
        self._populate_grid()
        return self.grid

    @ti.kernel
    def _populate_grid(self):
        # taichi: parallelized for loop
        for i, j in self.grid:
            self.grid[i, j] = ti.Vector([i * self.spacing, j * self.spacing])


# Initialize instances
granule = Granule(granule_scale_factor)
lattice = Lattice2D(spacing_scale_factor)
positions = lattice.granule_positions()


# =====================
# Screen Render Section
# =====================

# TODO: Implement RENDER in a separate module or class

# Pre-allocate Taichi fields for screen positions (optimized rendering)
screen_positions = ti.Vector.field(
    2, dtype=ti.f32, shape=(lattice.line_count, lattice.line_count)
)


@ti.kernel
def compute_screen_positions(offset: float, line_size: float):
    """Compute screen positions in parallel using Taichi kernel"""
    for i, j in ti.ndrange(lattice.line_count, lattice.line_count):
        universe_pos = positions[i, j]
        # Convert to normalized screen coordinates and apply offset
        screen_positions[i, j][0] = (universe_pos[0] + offset) / line_size
        screen_positions[i, j][1] = (universe_pos[1] + offset) / line_size


def render_lattice():
    """Render the granule lattice in 2D GUI with optimized performance"""
    # Create GUI
    gui = ti.GUI("Quantum Granule Lattice", (config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    # Calculate screen size for granule size
    universe_to_screen_ratio = (
        min(config.SCREEN_WIDTH, config.SCREEN_HEIGHT) / lattice.line_size
    )
    screen_radius = granule.radius * universe_to_screen_ratio

    # Ensure minimum visible radius
    min_radius = 1  # pixels
    screen_radius = max(screen_radius, min_radius)

    # Create display offset to center the lattice
    offset = (lattice.line_size - lattice.spacing * (lattice.line_count - 1)) / 2

    # Pre-compute screen positions once before the loop
    compute_screen_positions(offset, lattice.line_size)

    # Convert to numpy for batch processing
    screen_pos_numpy = screen_positions.to_numpy()

    # Pre-compute constants for the render loop
    screen_radius_int = int(screen_radius)
    circle_color = 0xFFFFFF
    line_count = lattice.line_count

    # Create list of circle positions for batch rendering (further optimization)
    circles_data = []
    for i in range(line_count):
        for j in range(line_count):
            circles_data.append(screen_pos_numpy[i, j])

    while gui.running:

        # Clear to black background
        gui.clear(0x000000)

        # Draw granules using pre-computed positions (optimized)
        # Use pre-computed list for fastest iteration
        for pos in circles_data:
            gui.circle(pos, color=circle_color, radius=screen_radius_int)

        gui.show()


# =====================
# Data Print Section
# =====================
print("\n===============================")
print("SIMULATION DATA")
print("===============================")

print(f"Planck length: {constants.PLANCK_LENGTH:.2e} m")
print(f"Granule Scale factor: {granule_scale_factor:.2e} m")
print(f"Spacing Scale factor: {spacing_scale_factor:.2e} m")

print("_______________________________")
print(f"Universe line size: {lattice.line_size:.2e} m")
print(f"Granule radius: {granule.radius:.2e} m")
print(f"Lattice spacing: {lattice.spacing:.2e} m")
print(f"Lattice line count: {lattice.line_count} x {lattice.line_count}")
print(f"Lattice count: {lattice.line_count**2:,}")

# Render the lattice
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
