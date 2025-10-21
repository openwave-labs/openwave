"""
XPERIMENT: Waves Superposition Exploration

Run sample XPERIMENTS shipped with the OpenWave package or create your own
Tweak universe_edge and other parameters to explore different scales.

Demonstrates wave interference from multiple sources in the BCC lattice.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns.

This XPERIMENT showcases:
- Multiple wave sources (up to 8, configurable positions)
- Wave superposition and interference patterns
- Phase control between sources (constructive/destructive interference)
- No spring coupling (pure wave propagation)
"""

import taichi as ti
import time

from openwave.common import config
from openwave.common import constants
from openwave._io import render

import openwave.spacetime.qmedium_granule as qmedium
import openwave.spacetime.qwave_granule as qwave
import openwave.validations.wave_diagnostics as diagnostics

# Define the architecture to be used by Taichi (GPU vs CPU)
ti.init(arch=ti.gpu)  # Use GPU if available, else fallback to CPU

# ================================================================
# Xperiment Parameters & Quantum Objects Instantiation
# ================================================================

UNIVERSE_EDGE = 6 * constants.QWAVE_LENGTH  # m, simulation domain, edge length of cubic universe

# Number of wave sources for this xperiment
NUM_SOURCES = 2

# Wave Source positions: normalized coordinates (0-1 range, relative to universe edge)
# Each row represents [x, y, z] coordinates for one source (Z-up coordinate system)
# Only provide NUM_SOURCES entries (only active sources needed)
sources_position = [
    # [0.5, 0.5, 0.5],  # Wave Source 0 - Center (commented out)
    [0, 1, 1.0],  # Wave Source 1 - Top plane (Z=1), center-back
    [1, 1, 1.0],  # Wave Source 2 - Top plane (Z=1), center-front
    [0.0, 1.0, 0.0],  # Wave Source 3 - Bottom-back-left corner
    [1.0, 0.0, 0.0],  # Wave Source 4 - Bottom-front-right corner
    [0.0, 0.0, 1.0],  # Wave Source 5 - Top-front-left corner
    [1.0, 1.0, 1.0],  # Wave Source 6 - Top-back-right corner
    [0.0, 0.0, 0.0],  # Wave Source 7 - Bottom-front-left corner
    [1.0, 1.0, 0.0],  # Wave Source 8 - Bottom-back-right corner
]

# Phase offsets for each source (integer degrees, converted to radians internally)
# Allows creating constructive/destructive interference patterns
# Only provide NUM_SOURCES entries (only active sources needed)
# Common patterns: 0° = in phase, 180° = opposite phase, 90° = quarter-cycle offset
sources_phase_deg = [
    0,  # Wave Source 0 (eg. 0 = in phase)
    0,  # Wave Source 1 (eg. 180 = opposite phase, creates destructive interference nodes)
    0,  # Wave Source 2
    0,  # Wave Source 3
    0,  # Wave Source 4
    0,  # Wave Source 5
    0,  # Wave Source 6
    0,  # Wave Source 7
    0,  # Wave Source 8
    0,  # Wave Source 9
]

lattice = qmedium.BCCLattice(UNIVERSE_EDGE)
granule = qmedium.BCCGranule(lattice.unit_cell_edge)

WAVE_DIAGNOSTICS = False  # Toggle wave diagnostics (speed & wavelength measurements)

# ================================================================
# Xperiment UI and overlay windows
# ================================================================

render.init_UI(cam_init_pos=[0.50, 2.00, 2.40])  # Initialize the GGUI window


def xperiment_specs():
    """Display xperiment definitions & specs."""
    with render.gui.sub_window("XPERIMENT: Superposing Wave", 0.00, 0.00, 0.19, 0.14) as sub:
        sub.text("QMedium: Granules in BCC lattice")
        sub.text("Granule Type: Point Mass")
        sub.text("Coupling: Phase Sync")
        sub.text(f"QWave Sources: {NUM_SOURCES} Harmonic Oscillators")
        sub.text("QWave Propagation: Radial from Source")


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


def controls():
    """Render the controls UI overlay."""
    global show_axis, block_slice, granule_type, show_sources
    global radius_factor, freq_boost, amp_boost, paused

    # Create overlay windows for controls
    with render.gui.sub_window("CONTROLS", 0.85, 0.00, 0.15, 0.24) as sub:
        show_axis = sub.checkbox("Axis", show_axis)
        block_slice = sub.checkbox("Block Slice", block_slice)
        granule_type = sub.checkbox("Granule Type Color", granule_type)
        show_sources = sub.checkbox("Show Wave Sources", show_sources)
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
            normalized_position[i] = lattice.position_am[i] / lattice.universe_edge_am


def normalize_granule():
    """Normalize granule radius to 0-1 range for GGUI rendering"""

    global normalized_radius

    normalized_radius = max(
        granule.radius / lattice.universe_edge, 0.0001
    )  # Ensure minimum 0.01% of screen radius for visibility


# ================================================================
# Xperiment Rendering
# ================================================================


def render_xperiment(lattice):
    """Render 3D BCC lattice with multiple wave sources using GGUI's 3D scene.

    Visualizes wave superposition from multiple sources, creating interference patterns
    where waves constructively and destructively combine.

    Args:
        lattice: Lattice instance with positions, directions, and universe parameters
    """
    global show_axis, block_slice, granule_type, show_sources
    global radius_factor, freq_boost, amp_boost, paused
    global normalized_position

    # Initialize variables
    show_axis = False  # Toggle to show/hide axis lines
    block_slice = False  # Block-slicing toggle
    granule_type = True  # Granule type coloring toggle
    show_sources = True  # Show wave sources toggle
    radius_factor = 1.0  # Initialize granule size factor
    freq_boost = 1.0  # Initialize frequency boost
    amp_boost = 5.0  # Initialize amplitude boost
    paused = False  # Pause toggle

    # Time tracking for radial harmonic oscillation of all granules
    t = 0.0
    last_time = time.time()
    frame = 0  # Frame counter for diagnostics

    # Initialize normalized position (0-1 range for GGUI) & block-slicing
    # block-slicing: hide front 1/8th of the lattice for see-through effect
    normalized_position = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
    normalize_granule()

    # Convert phase from degrees to radians for physics calculations
    # Conversion: radians = degrees × π/180
    sources_phase_rad = [deg * ti.math.pi / 180 for deg in sources_phase_deg]

    qwave.build_source_vectors(
        sources_position, sources_phase_rad, NUM_SOURCES, lattice
    )  # compute distance & direction vectors to all sources

    # Print diagnostics header if enabled
    if WAVE_DIAGNOSTICS:
        diagnostics.print_initial_parameters()

    while render.window.running:
        # Render UI overlay windows
        render.init_scene(show_axis)  # Initialize scene with lighting and camera
        controls()
        data_dashboard()
        xperiment_specs()

        if not paused:
            # Calculate actual elapsed time (real-time tracking)
            current_time = time.time()
            dt_real = current_time - last_time
            last_time = current_time
            t += dt_real  # Use real elapsed time instead of fixed DT

            # Apply radial harmonic oscillation to all granules from multiple wave sources
            # Each granule receives wave contributions from all active sources
            # Waves superpose creating interference patterns (constructive/destructive)
            qwave.oscillate_granules(
                lattice.position_am,  # Granule positions in attometers
                lattice.equilibrium_am,  # Rest positions for all granules
                lattice.velocity_am,  # Granule velocity in am/s
                NUM_SOURCES,  # Number of active wave sources
                t,
                freq_boost,  # Frequency visibility boost (will be applied over the slow-motion factor)
                amp_boost,  # Amplitude visibility boost for scaled lattices
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

        # Render the wave sources
        if show_sources:
            render.scene.particles(
                centers=qwave.sources_pos_field,
                radius=normalized_radius * 2,
                color=config.COLOR_SOURCE[1],
            )

        # Render the scene to canvas
        render.show_scene()


# ================================================================
# Main calls
# ================================================================
if __name__ == "__main__":

    # Render the 3D lattice
    render_xperiment(lattice)
