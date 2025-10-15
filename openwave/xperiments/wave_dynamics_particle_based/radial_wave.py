"""
XPERIMENT: Phase Sync Harmonic Oscillation
Run sample XPERIMENTS shipped with the OpenWave package or create your own
Tweak universe_edge and other parameters to explore different scales.

Demonstrates radial harmonic oscillation of all granules in the BCC lattice.
All granules oscillate toward/away from the lattice center along their
individual direction vectors, creating spherical wave interference patterns.

This XPERIMENT showcases:
- Radial wave propagation from lattice center
- Phase-shifted oscillations creating wave interference
- Uniform energy injection across all granules
- No spring coupling (pure oscillation demonstration)
"""

import taichi as ti
import time

from openwave.common import config
from openwave.common import constants
from openwave._io import render

import openwave.spacetime.qmedium_particles as qmedium
import openwave.spacetime.qwave_radial as qwave
import openwave.validations.wave_diagnostics as diagnostics

# Define the architecture to be used by Taichi (GPU vs CPU)
ti.init(arch=ti.gpu)  # Use GPU if available, else fallback to CPU

# ================================================================
# Xperiment Parameters & Quantum Objects Instantiation
# ================================================================

UNIVERSE_EDGE = 1e-16  # m simulation domain size, edge length of cubic universe
TARGET_PARTICLES = 1e6  # target particle count, granularity (impacts performance)

# slow-motion (divides frequency for human-visible motion, time microscope)
SLOW_MO = 1e25  # (1 = real-time, 10 = 10x slower, 1e25 = 10 * trillion * trillions FPS)

lattice = qmedium.BCCLattice(UNIVERSE_EDGE, TARGET_PARTICLES)
granule = qmedium.Granule(lattice.unit_cell_edge)

WAVE_DIAGNOSTICS = True  # Toggle wave diagnostics (speed & wavelength measurements)

# ================================================================
# Xperiment UI and overlay windows
# ================================================================

render.init_UI()  # Initialize the GGUI window


def xperiment_specs():
    """Display xperiment definitions & specs."""
    with render.gui.sub_window("XPERIMENT: Radial Wave PSHO", 0.00, 0.00, 0.19, 0.14) as sub:
        sub.text("QMedium: Particles in BCC lattice")
        sub.text("Granule Type: Point Mass")
        sub.text("Coupling: Phase Sync")
        sub.text("QWave Source: Harmonic Oscillators")
        sub.text("QWave Propagation: Radial from Center")


def data_dashboard():
    """Display simulation data dashboard."""
    with render.gui.sub_window("DATA-DASHBOARD", 0.00, 0.50, 0.19, 0.50) as sub:
        sub.text("--- QUANTUM-MEDIUM ---")
        sub.text(f"Sim Universe Size: {lattice.universe_edge:.1e} m (edge)")
        sub.text(f"Granule Count: {lattice.total_granules:,} particles")
        sub.text(f"QMedium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")

        sub.text("")
        sub.text("--- Scaling-Up (for computation) ---")
        sub.text(f"Factor: {lattice.scale_factor:.1e} x Planck Scale")
        sub.text(f"Unit-Cells per Lattice Edge: {lattice.grid_size:,}")
        sub.text(f"BCC Unit-Cell Edge: {lattice.unit_cell_edge:.2e} m")
        sub.text(f"Granule Radius: {granule.radius:.2e} m")
        sub.text(f"Granule Mass: {granule.mass:.2e} kg")

        sub.text("")
        sub.text("--- Sim Resolution (linear) ---")
        sub.text(f"QWave: {lattice.qwave_res:.0f} granules/qwave (>10)")
        if lattice.qwave_res < 10:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Universe: {lattice.uni_res:.1f} qwaves/universe-edge")

        sub.text("")
        sub.text("--- QUANTUM-WAVE ---")
        sub.text(f"QWAVE Speed (c): {constants.QWAVE_SPEED:.1e} m/s")
        sub.text(f"QWAVE Wavelength (lambda): {constants.QWAVE_LENGTH:.1e} m")
        sub.text(f"QWAVE Frequency (f): {constants.QWAVE_FREQUENCY:.1e} Hz")
        sub.text(f"QWAVE Amplitude (A): {constants.QWAVE_AMPLITUDE:.1e} m")

        sub.text("")
        sub.text("--- Sim Universe Wave Energy ---")
        sub.text(f"Energy: {lattice.energy:.1e} J ({lattice.energy_kWh:.1e} KWh)")
        sub.text(f"{lattice.energy_years:,.1e} Years of global energy use")


def controls():
    """Render the controls UI overlay."""
    global show_axis, block_slice, granule_type, radius_factor, freq_boost, amp_boost, paused

    # Create overlay windows for controls
    with render.gui.sub_window("CONTROLS", 0.85, 0.00, 0.15, 0.21) as sub:
        show_axis = sub.checkbox("Axis", show_axis)
        block_slice = sub.checkbox("Block Slice", block_slice)
        granule_type = sub.checkbox("Granule Type Color", granule_type)
        radius_factor = sub.slider_float("Granule", radius_factor, 0.01, 2.0)
        freq_boost = sub.slider_float("f Boost", freq_boost, 0.1, 10.0)
        amp_boost = sub.slider_float("Amp Boost", amp_boost, 1.0, 5.0)
        if paused:
            if sub.button("Continue"):
                paused = False
        else:
            if sub.button("Pause"):
                paused = True


# ================================================================
# Xperiment Normalization to Screen Coordinates
# ================================================================


@ti.kernel
def normalize_lattice(enable_slice: ti.i32):  # type: ignore
    """Normalize lattice positions to 0-1 range for GGUI rendering."""
    for i in range(lattice.total_granules):
        # Normalize to 0-1 range (positions are in attometers, scale them back)
        if enable_slice == 1 and lattice.front_octant[i] == 1:
            # Block-slicing enabled: hide front octant granules by moving to origin
            normalized_position[i] = ti.Vector([0.0, 0.0, 0.0])
        else:
            # Normal rendering: normalize to 0-1 range
            normalized_position[i] = lattice.positions_am[i] / lattice.universe_edge_am


def normalize_granule():
    """Normalize granule radius to 0-1 range for GGUI rendering"""

    global normalized_radius

    normalized_radius = max(
        granule.radius / lattice.universe_edge, 0.0001
    )  # Ensure minimum 0.01% of screen radius for visibility


# ================================================================
# Xperiment Rendering
# ================================================================


def render_xperiment(lattice, granule):
    """Render 3D BCC lattice with radial harmonic oscillation using GGUI's 3D scene.

    Visualizes all granules oscillating radially from the lattice center,
    creating spherical wave interference patterns through phase-shifted oscillations.

    Args:
        lattice: Lattice instance with positions, directions, and universe parameters
        granule: Granule instance for size reference
        neighbors: BCCNeighbors instance for optional link visualization
    """
    global show_axis, block_slice, granule_type, radius_factor, freq_boost, amp_boost, paused
    global normalized_position

    # Initialize variables
    show_axis = False  # Toggle to show/hide axis lines
    block_slice = True  # Block-slicing toggle
    granule_type = True  # Granule type coloring toggle
    radius_factor = 1.0  # Initialize granule size factor
    freq_boost = 1.0  # Initialize frequency boost
    amp_boost = 5.0  # Initialize amplitude boost
    paused = False  # Pause toggle

    # Time tracking for radial harmonic oscillation of all granules
    t = 0.0
    last_time = time.time()
    frame = 0  # Frame counter for diagnostics

    # Initialize normalized positions (0-1 range for GGUI) & block-slicing
    # block-slicing: hide front 1/8th of the lattice for see-through effect
    normalized_position = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
    normalize_granule()

    # Print diagnostics header if enabled
    if WAVE_DIAGNOSTICS:
        diagnostics.print_initial_parameters(slow_mo=SLOW_MO)

    while render.window.running:
        # Render UI overlay windows
        render.cam_instructions()
        controls()
        data_dashboard()
        xperiment_specs()

        if not paused:
            # Calculate actual elapsed time (real-time tracking)
            current_time = time.time()
            dt_real = current_time - last_time
            last_time = current_time
            t += dt_real  # Use real elapsed time instead of fixed DT

            # Apply radial harmonic oscillation to all granules
            # All granules oscillate toward/away from lattice center along their direction vectors
            # Phase is determined by radial distance, creating outward-propagating spherical waves
            qwave.oscillate_granules(
                lattice.positions_am,  # Granule positions in attometers
                lattice.velocity_am,  # Granule velocity in am/s
                lattice.equilibrium_am,  # Rest positions for all granules
                lattice.center_direction,  # Direction vectors to center for all granules
                lattice.center_distance_am,  # Radial distance from each granule to center
                t,
                SLOW_MO / freq_boost,
                amp_boost,  # Visibility boost for scaled lattices
            )

            # Update normalized positions for rendering (must happen after position updates)
            # with optional block-slicing (see-through effect)
            normalize_lattice(1 if block_slice else 0)

            # Wave diagnostics (minimal footprint)
            if WAVE_DIAGNOSTICS:
                diagnostics.print_wave_diagnostics(
                    t,
                    frame,
                    print_interval=100,  # Print every 100 frames
                )

            frame += 1  # Increment frame counter
        else:
            # Update last_time during pause to prevent time jump on resume
            last_time = time.time()

        # Render granules with optional type-coloring
        if granule_type:
            render.scene.particles(
                normalized_position,
                radius=normalized_radius * radius_factor,
                per_vertex_color=lattice.granule_color,
            )
        else:
            render.scene.particles(
                normalized_position,
                radius=normalized_radius * radius_factor,
                color=config.COLOR_MEDIUM[1],
            )

        # Render the scene to canvas
        render.show_scene(show_axis)


# ================================================================
# Main calls
# ================================================================
if __name__ == "__main__":

    # Render the 3D lattice
    render_xperiment(lattice, granule)
