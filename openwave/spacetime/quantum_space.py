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
import openwave.core.equations as equations

ti.init(arch=ti.gpu)


# ==================================================================
# Physics Engine
# ==================================================================


class Granule:
    """
    Granule Model: The aether consists of "granules".
    Fundamental units that vibrate and create wave patterns.
    Their collective motion at Planck scale creates all observable phenomena.
    Each granule has a defined radius and mass.
    """

    def __init__(self, unit_cell_edge: float):  # in meters
        self.radius = unit_cell_edge / (2 * np.e)  # m, radius = unit cell edge / (2e)
        self.mass = (
            constants.QSPACE_DENSITY * unit_cell_edge**3 / 2
        )  # kg, mass = density * unit cell volume / 2 granules per cell
        self.scale_factor = (
            self.radius / constants.PLANCK_LENGTH
        )  # linear scale factor from Planck length


@ti.data_oriented
class Lattice:
    """
    3D Body-Centered Cubic (BCC) lattice for quantum space simulation.
    BCC topology: cubic unit cells with an additional granule at the center.
    More efficient packing than simple cubic (68% vs 52% space filling).

    Performance Design: 1D Arrays with 3D Vectors
    - Memory: Contiguous layout, perfect cache line utilization (64-byte alignment)
    - Compute: Single loop parallelization, no index arithmetic (vs i*dim²+j*dim+k)
    - GPU: Direct thread mapping (thread_i→granule_i), coalesced memory access
    - BCC Lattice: Uniform treatment of corner+center granules in single array
    Benefits:
    - Simpler updates: One kernel updates all particles
    - Cleaner code: No need to manage multiple arrays
    - Movement-Ready: Velocity field prepared for dynamics,
    particles can move freely without grid remapping constraints

    This is why high-performance physics engines (molecular dynamics, N-body simulations)
    universally use 1D arrays for particle data, regardless of spatial dimensionality.
    """

    def __init__(self, universe_edge: float):
        """
        Initialize BCC lattice with computed unit-cell spacing.

        Args:
            universe_edge: Edge length of the cubic universe in meters
        """
        # Compute lattice total energy from quantum wave equation
        self.lattice_energy = equations.energy_wave_equation(universe_edge**3)  # in Joules
        self.lattice_energy_kWh = equations.J_to_kWh(self.lattice_energy)  # in KWh
        self.lattice_energy_years = self.lattice_energy_kWh / (183230 * 1e9)  # global energy use

        # Compute total volume and granule count from resolution
        total_granules = config.QSPACE_RES

        # Scale to attometers to avoid float32 precision issues
        # This keeps position values in a reasonable range (e.g., 1000 instead of 1e-15)
        universe_edge = universe_edge / constants.ATTO_PREFIX  # Convert meters to attometers
        universe_volume = universe_edge**3

        # BCC has 2 granules per unit cell (8 corners shared + 1 center)
        # Volume per unit cell = universe_volume / (total_granules / 2), in attometers^3
        unit_cell_volume = universe_volume / (total_granules / 2)

        # Unit cell edge length in attometers (a^3 = volume)
        self.unit_cell_edge = unit_cell_volume ** (1 / 3)
        self.universe_edge = universe_edge

        # Compute quantum wave linear resolution, sampling rate
        # granules per wavelength, should be >2 for Nyquist
        self.qwave_res = constants.QWAVE_LENGTH / (self.unit_cell_edge * constants.ATTO_PREFIX) * 2
        # Compute universe linear resolution, qwavelengths per universe edge
        self.uni_res = universe_edge * constants.ATTO_PREFIX / constants.QWAVE_LENGTH

        # Calculate grid dimensions (number of unit cells per dimension)
        self.grid_size = int(universe_edge / self.unit_cell_edge)

        # Total granules: corners + centers
        # Corners: (grid_size + 1)^3, Centers: grid_size^3
        corner_count = (self.grid_size + 1) ** 3
        center_count = self.grid_size**3
        self.total_granules = corner_count + center_count

        # Initialize position and velocity 1D fields
        # Single field design: Better memory locality, simpler kernels, future-ready for dynamics
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)

        # Populate the lattice
        self._populate_bcc_lattice()

    @ti.kernel
    def _populate_bcc_lattice(self):
        """Populate BCC lattice positions in a 1D field.
        Kernel is properly optimized for Taichi's parallel execution:
        1. Single outermost loop - for idx in range() allows full GPU parallelization
        2. Index decoding - Converts linear index to 3D coordinates using integer division/modulo
        3. No nested loops - All granules computed in parallel across GPU threads
        4. Efficient branching - Simple if/else to determine corner vs center granules
        This structure ensures maximum parallelization on GPU, as each thread independently
        computes one granule's position without synchronization overhead.
        """
        # Parallelize over all granules using single outermost loop
        for idx in range(self.total_granules):
            # Determine if this is a corner or center granule
            corner_count = (self.grid_size + 1) ** 3

            if idx < corner_count:
                # Corner granule: decode 3D position from linear index
                grid_dim = self.grid_size + 1
                i = idx // (grid_dim * grid_dim)
                j = (idx % (grid_dim * grid_dim)) // grid_dim
                k = idx % grid_dim

                self.positions[idx] = ti.Vector(
                    [i * self.unit_cell_edge, j * self.unit_cell_edge, k * self.unit_cell_edge]
                )
            else:
                # Center granule: decode position with offset
                center_idx = idx - corner_count
                i = center_idx // (self.grid_size * self.grid_size)
                j = (center_idx % (self.grid_size * self.grid_size)) // self.grid_size
                k = center_idx % self.grid_size

                offset = self.unit_cell_edge / 2.0
                self.positions[idx] = ti.Vector(
                    [
                        i * self.unit_cell_edge + offset,
                        j * self.unit_cell_edge + offset,
                        k * self.unit_cell_edge + offset,
                    ]
                )

            # Initialize velocity to zero for all granules
            self.velocities[idx] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def update_positions(self, dt: ti.f32):  # type: ignore
        """Update granule positions based on velocities (for future dynamics)."""
        for i in self.positions:
            self.positions[i] += self.velocities[i] * dt


# ==================================================================
# Rendering Engine
# ==================================================================
# TODO: Implement Rendering Engine in a separate module?


def render_lattice(lattice_instance):
    """
    Render 3D BCC lattice using GGUI's 3D scene.

    Args:
        lattice_instance: Lattice instance to render
    """

    lattice = lattice_instance

    # Create GGUI window with 3D scene
    window = ti.ui.Window(
        "Quantum-Space Lattice (3D BCC Topology)",
        (config.SCREEN_RES[0], config.SCREEN_RES[1]),
        vsync=True,
    )
    canvas = window.get_canvas()
    gui = window.get_gui()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Initial camera setup (will be overridden by orbit parameters below)
    camera.up(0, 1, 0)  # Y-axis up

    # Normalize positions for rendering (0-1 range for GGUI)
    # Create a field for normalized positions
    normalized_positions = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)

    @ti.kernel
    def normalize_positions():
        """Normalize lattice positions to 0-1 range for GGUI rendering."""
        for i in range(lattice.total_granules):
            # Normalize from attometer scale to 0-1 range
            normalized_positions[i] = lattice.positions[i] / lattice.universe_edge

    # Prepare colors
    granule_color = config.COLOR_GRANULE[2]  # Blue color for granules
    bkg_color = config.COLOR_SPACE[2]  # Black background

    # Granule radius scaled for visibility
    granule = Granule(lattice.unit_cell_edge)  # in attometers
    # Normalize radius to 0-1 range for GGUI rendering
    normalized_radius = (granule.radius) / lattice.universe_edge
    # Ensure minimum radius for visibility
    min_radius = 0.0001  # Minimum 0.01% of screen
    normalized_radius = max(normalized_radius, min_radius)

    # Normalize positions once before render loop
    print("Normalizing 3D lattice positions to screen...")
    normalize_positions()

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

    print("Starting 3D render loop...")
    print("Controls: Right-click drag to rotate, Q/A keys to zoom in/out")

    while window.running:
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

        # Calculate camera position from spherical coordinates
        cam_x = orbit_center[0] + orbit_radius * np.sin(orbit_phi) * np.cos(orbit_theta)
        cam_y = orbit_center[1] + orbit_radius * np.cos(orbit_phi)
        cam_z = orbit_center[2] + orbit_radius * np.sin(orbit_phi) * np.sin(orbit_theta)

        # Update camera
        camera.position(cam_x, cam_y, cam_z)
        camera.lookat(orbit_center[0], orbit_center[1], orbit_center[2])
        camera.up(0, 1, 0)
        scene.set_camera(camera)

        # Set background color
        canvas.set_background_color(bkg_color)

        # Add ambient and directional lighting
        scene.ambient_light((0.1, 0.1, 0.15))  # Slight blue ambient
        scene.point_light(
            pos=(0.5, 1.5, 0.5), color=(1.0, 1.0, 1.0)  # Light from above center  # White light
        )
        scene.point_light(
            pos=(1.0, 0.5, 1.0),  # Secondary light from corner
            color=(0.5, 0.5, 0.5),  # Dimmer white light
        )

        # Render granules as particles (spheres)
        scene.particles(normalized_positions, radius=normalized_radius, color=granule_color)

        # Create sub-window for stats overlay
        with gui.sub_window("DATA-DASHBOARD", 0.01, 0.01, 0.24, 0.37) as sub:
            sub.text(f"Total Granules: {lattice.total_granules:,} (config.py)")
            sub.text(f"Universe Cube Edge: {lattice.universe_edge * constants.ATTO_PREFIX:.2e} m")

            sub.text(f"")
            sub.text(f"--- Linear Resolutions ---")
            sub.text(f"Universe: {lattice.uni_res:.1f} qwaves/universe-edge")
            sub.text(f"Wave: {lattice.qwave_res:.0f} granules/qwavelength (min 2)")
            sub.text(f"Scale-up: {granule.scale_factor*constants.ATTO_PREFIX:.1e} x Planck Length")

            sub.text(f"")
            sub.text(f"--- Scaled Granule Data ---")
            sub.text(f"BCC Unit-Cell Edge: {lattice.unit_cell_edge * constants.ATTO_PREFIX:.2e} m")
            sub.text(f"Granule Radius: {granule.radius * constants.ATTO_PREFIX:.2e} m")
            sub.text(f"Granule Mass: {granule.mass * constants.ATTO_PREFIX**3:.2e} kg")

            sub.text(f"")
            sub.text(f"--- Universe Energy Data ---")
            sub.text(f"Total Energy: {lattice.lattice_energy:.2e} J")
            sub.text(f"Total Energy: {lattice.lattice_energy_kWh:.2e} KWh")
            sub.text(f"{lattice.lattice_energy_years:,.1e} Years of global energy use")

        # Render the scene to canvas
        canvas.scene(scene)
        window.show()


# ==================================================================
# Main calls
# ==================================================================
if __name__ == "__main__":

    # Quantum-Space Lattice (3D BCC Topology) Stats
    universe_edge = 1e-16  # m
    lattice = Lattice(universe_edge)

    print("\n===============================")
    print("SIMULATION DETAILED-DATA")
    print("===============================")
    print(f"Grid size: {lattice.grid_size} x {lattice.grid_size} x {lattice.grid_size}")
    print(f"  - Corner granules: {(lattice.grid_size + 1) ** 3:,}")
    print(f"  - Center granules: {lattice.grid_size ** 3:,}")

    # Render the 3D lattice
    print("\n--- 3D Lattice Rendering ---")
    render_lattice(lattice)  # Pass the already created lattice instance
