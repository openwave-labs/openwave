"""
QUANTUM SPACE
(AKA: AKASHA @yoga, WUJI @taoism, AETHER @classical)

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

# ==================================================================
# Physics Engine
# ==================================================================


class GranulePhysics:
    # Granule Model: The aether consists of "granules".
    # Fundamental units that vibrate and create wave patterns.
    # Their collective motion at Planck scale creates all observable phenomena.

    og_granule_radius = constants.PLANCK_LENGTH

    def __init__(self, granule_radius=og_granule_radius):
        self.radius = granule_radius  # m


@ti.data_oriented
class Lattice2DPhysics:
    # Granule Count on Lattice: Potentially trillions of granules requiring
    # spring constant calculations, harmonic motion, and wave propagation

    og_universe_radius = UNIVERSE_RADIUS  # m
    og_lattice_spacing = 2 * constants.PLANCK_LENGTH * np.e  # m, Planck-scale
    min_lattice_spacing = 5e-21  # m, min spacing clamp, for computability
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


# ==================================================================
# Rendering Engine
# ==================================================================


class GranuleRender:
    min_granule_radius = 1  # pixels, min radius clamp for visibility

    def __init__(self, lattice_spacing):
        # Calculate radius in pixels, then convert to normalized (0.0-1.0) range
        radius = max(self.min_granule_radius, lattice_spacing / (2 * np.e))
        self.normalized_radius = radius / min(SCREEN_WIDTH, SCREEN_HEIGHT)


@ti.data_oriented
class Lattice2DRender:
    screen_size = min(SCREEN_WIDTH, SCREEN_HEIGHT)
    lattice_spacing = 6  # pixels, increased spacing for visibility and performance
    universe_length = screen_size  # pixels, universe side length
    linear_count = round(universe_length / lattice_spacing)  # number of granules

    # Convert pixel values to normalized (0.0-1.0) range for GGUI
    normalized_spacing_x = lattice_spacing / SCREEN_WIDTH
    normalized_spacing_y = lattice_spacing / SCREEN_HEIGHT

    # Create normalized offset to center the lattice on display
    offset_x_pixels = (SCREEN_WIDTH - lattice_spacing * (linear_count - 1)) / 2
    offset_y_pixels = (SCREEN_HEIGHT - lattice_spacing * (linear_count - 1)) / 2
    normalized_offset_x = offset_x_pixels / SCREEN_WIDTH
    normalized_offset_y = offset_y_pixels / SCREEN_HEIGHT

    def granule_positions(self):
        # use taichi primitive types - now storing normalized positions directly
        self.grid = ti.Vector.field(2, dtype=ti.f32, shape=(self.linear_count, self.linear_count))
        self._populate_grid()
        return self.grid

    @ti.kernel
    def _populate_grid(self):
        # taichi: parallelized for loop - now computing normalized positions directly
        for i, j in self.grid:
            # Positions are already in normalized (0.0-1.0) range for GGUI
            self.grid[i, j] = ti.Vector(
                [
                    i * self.normalized_spacing_x + self.normalized_offset_x,
                    j * self.normalized_spacing_y + self.normalized_offset_y,
                ]
            )


def render_lattice():
    print("Initializing render...")

    # Initialize instances for rendering
    lattice = Lattice2DRender()
    positions = lattice.granule_positions()  # Already normalized for GGUI
    granule_radius = GranuleRender(lattice.lattice_spacing).normalized_radius  # Already normalized
    bkg_color = config.COLOR_SPACE[2]  # background
    circle_color = config.COLOR_GRANULE[2]  # granules

    print("_______________________________")
    print("Lattice 2D Render initialized.")
    print(f"Lattice spacing: {lattice.lattice_spacing} pixels")
    print(f"Granule linear count: {lattice.linear_count:,} granules")
    print(f"Granule positions populated: {lattice.linear_count**2:,} granules")
    print(f"Granule Radius (normalized): {granule_radius:.6f}")

    print("_______________________________")
    print(f"Creating GGUI window: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

    # Create GGUI Window
    window = ti.ui.Window(
        "Quantum Granule Lattice (GGUI)", (SCREEN_WIDTH, SCREEN_HEIGHT), vsync=True
    )
    canvas = window.get_canvas()

    # Flatten the 2D grid into a 1D field for batch rendering
    total_points = lattice.linear_count * lattice.linear_count
    vertices = ti.Vector.field(2, dtype=ti.f32, shape=total_points)

    # Copy positions from 2D grid to 1D field (positions are already normalized)
    @ti.kernel
    def flatten_positions():
        for i in range(lattice.linear_count):
            for j in range(lattice.linear_count):
                idx = i * lattice.linear_count + j
                vertices[idx] = positions[i, j]

    flatten_positions()

    print(f"Prepared {total_points:,} normalized positions")
    print("Starting render loop...")

    while window.running:
        # Set background color
        canvas.set_background_color(bkg_color)

        # Draw circles using Canvas - positions and radius are already normalized
        canvas.circles(vertices, radius=granule_radius, color=circle_color)

        # Show the window
        window.show()


# ==================================================================
# Main calls
# ==================================================================
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


# ================================================================
# Optimizations Applied to quantum_space.py:
# ================================================================
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


# Optimizations to the render_lattice() function for small-pixel spacing:
#   1. Batch rendering: renders all points in a single draw call
#   Replaces per-frame, per-point gui.circle() calls with
#   a single batch gui.circles() call for improved performance.

#   2. Pre-computation: Normalizes all positions once before the render
#   loop starts

#   3. Array conversion: Converts Taichi field to numpy array once,
#   outside the loop

#   significantly speeding up rendering, especially with smaller lattice spacing.


# ================================================================
# Conversion from GUI to GGUI
# ================================================================
# Converted the rendering system from ti.GUI to ti.ui.Window with
#   Canvas. Key changes:

#   1. Window creation: Using ti.ui.Window() instead of ti.GUI()
#   2. Canvas rendering: Using window.get_canvas() for 2D drawing
#   3. Taichi fields: Created a vertices field instead of numpy array
#   (GGUI prefers fields)
#   4. Normalized coordinates: All positions and radius normalized to
#   0.0-1.0 range
#   5. Color format: Using normalized RGB tuples instead of hex values
#   6. VSync enabled: Added vsync=True for smoother rendering

#   The Canvas feature provides efficient 2D rendering while maintaining
#    compatibility with Taichi's GPU acceleration.
