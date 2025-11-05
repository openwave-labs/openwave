"""
XPERIMENT: Wave Superposition Exploration

Run sample XPERIMENTS shipped with the OpenWave package or create your own
Tweak universe size and other parameters to explore different scales.

Demonstrates wave interference from multiple sources in the lattice.
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

import openwave.spacetime.medium_level0 as medium
import openwave.spacetime.wave_engine_level0 as ewave
import openwave.validations.wave_diagnostics as diagnostics

# Define the architecture to be used by Taichi (GPU vs CPU)
ti.init(arch=ti.gpu)  # Use GPU if available, else fallback to CPU

# ================================================================
# Xperiment Parameters & Subatomic Objects Instantiation
# ================================================================

UNIVERSE_SIZE = [
    8 * constants.EWAVE_LENGTH,
    6 * constants.EWAVE_LENGTH,
    1 * constants.EWAVE_LENGTH,
]  # m, simulation domain [x, y, z] dimensions (can be asymmetric)
TICK_SPACING = 0.25  # Axis tick marks spacing for position reference

# Number of wave sources for this xperiment
NUM_SOURCES = 3  # This xperiment has pre-populated list for 6 sources, change to test

# Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
# Each row represents [x, y, z] coordinates for one source (Z-up coordinate system)
# Only provide NUM_SOURCES entries (only active sources needed)
z_position = [0.12]  # Initialize Z positions
equilateral = ti.sqrt(0.5**2 - 0.25**2)  # Factor for perfect triangle arrangement
sources_position = [
    [0.25, 0.15 + equilateral, z_position[0]],  # Wave Source 0
    [0.75, 0.15 + equilateral, z_position[0]],  # Wave Source 1
    [0.5, 0.15, z_position[0]],  # Wave Source 2
    [0.0, 0.0, z_position[0]],  # Wave Source 3
    [1.0, 0.0, z_position[0]],  # Wave Source 4
    [1.0, 0.75, z_position[0]],  # Wave Source 5
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
]

# Choose color theme for rendering (OCEAN, DESERT, FOREST)
COLOR_THEME = "OCEAN"

# Instantiate the lattice and granule objects (chose BCC or SC Lattice type)
lattice = medium.BCCLattice(UNIVERSE_SIZE, theme=COLOR_THEME)
granule = medium.BCCGranule(lattice.unit_cell_edge)

# DATA SAMPLING & DIAGNOSTICS ==================================
peak_amplitude = 0.0  # Initialize granule max displacement (data sampling variable)
WAVE_DIAGNOSTICS = False  # Toggle wave diagnostics (speed & wavelength measurements)

# ================================================================
# Xperiment UI and overlay windows
# ================================================================

render.init_UI(
    UNIVERSE_SIZE, TICK_SPACING, cam_init_pos=[0.25, 0.91, 1.00]
)  # Initialize the GGUI window


def xperiment_specs():
    """Display xperiment definitions & specs."""
    with render.gui.sub_window("XPERIMENT: Wave Interference", 0.00, 0.00, 0.19, 0.14) as sub:
        sub.text("Medium: Granules in BCC lattice")
        sub.text("Granule Type: Point Mass")
        sub.text("Coupling: Phase Sync")
        sub.text(f"EWave Sources: {NUM_SOURCES} Harmonic Oscillators")
        sub.text("EWave Propagation: Radial from Source")


def data_dashboard():
    """Display simulation data dashboard."""

    with render.gui.sub_window("DATA-DASHBOARD", 0.00, 0.41, 0.19, 0.59) as sub:
        sub.text("--- WAVE-MEDIUM ---", color=config.LIGHT_BLUE[1])
        sub.text(f"Universe Size: {lattice.max_universe_edge:.1e} m (max edge)")
        sub.text(f"Granule Count: {lattice.total_granules:,} particles")
        sub.text(f"Medium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")

        sub.text("\n--- Scaling-Up (for computation) ---", color=config.LIGHT_BLUE[1])
        sub.text(f"Factor: {lattice.scale_factor:.1e} x Planck Scale")
        sub.text(f"Unit-Cells per Max Edge: {lattice.max_grid_size:,}")
        sub.text(f"Unit-Cell Edge: {lattice.unit_cell_edge:.2e} m")
        sub.text(f"Granule Radius: {granule.radius * radius_factor:.2e} m")
        sub.text(f"Granule Mass: {granule.mass:.2e} kg")

        sub.text("\n--- Sim Resolution (linear) ---", color=config.LIGHT_BLUE[1])
        sub.text(f"EWave: {lattice.ewave_res:.0f} granules/ewave (>10)")
        if lattice.ewave_res < 10:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Universe: {lattice.max_uni_res:.1f} ewaves/universe-edge")

        sub.text("\n--- ENERGY-WAVE ---", color=config.LIGHT_BLUE[1])
        sub.text(f"EWAVE Speed (c): {constants.EWAVE_SPEED:.1e} m/s")
        sub.text(f"EWAVE Wavelength (lambda): {constants.EWAVE_LENGTH:.1e} m")
        sub.text(f"EWAVE Frequency (f): {constants.EWAVE_FREQUENCY:.1e} Hz")
        sub.text(f"EWAVE Amplitude (A): {constants.EWAVE_AMPLITUDE:.1e} m")

        sub.text("\n--- Sim Universe Wave Energy ---", color=config.LIGHT_BLUE[1])
        sub.text(f"Energy: {lattice.energy:.1e} J ({lattice.energy_kWh:.1e} KWh)")

        sub.text("\n--- TIME MICROSCOPE ---", color=config.LIGHT_BLUE[1])
        slowed_mo = config.SLOW_MO / freq_boost
        fps = 0 if elapsed_t == 0 else frame / elapsed_t
        sub.text(f"Frames Rendered: {frame}")
        sub.text(f"Real Time: {elapsed_t / slowed_mo:.2e}s ({fps * slowed_mo:.0e} FPS)")
        sub.text(f"(1 real second = {slowed_mo / (60*60*24*365):.0e}y of sim time)")
        sub.text(f"Sim Time (slow-mo): {elapsed_t:.2f}s ({fps:.0f} FPS)")


def controls():
    """Render the controls UI overlay."""
    global show_axis, block_slice, show_sources
    global radius_factor, freq_boost, amp_boost, paused

    # Create overlay windows for controls
    with render.gui.sub_window("CONTROLS", 0.85, 0.00, 0.15, 0.22) as sub:
        show_axis = sub.checkbox(f"Axis (tick marks: {TICK_SPACING})", show_axis)
        block_slice = sub.checkbox("Block Slice", block_slice)
        show_sources = sub.checkbox("Show Wave Sources", show_sources)
        radius_factor = sub.slider_float("Granule", radius_factor, 0.1, 2.0)
        freq_boost = sub.slider_float("f Boost", freq_boost, 0.1, 10.0)
        amp_boost = sub.slider_float("Amp Boost", amp_boost, 1.0, 5.0)
        if paused:
            if sub.button("Continue"):
                paused = False
        else:
            if sub.button("Pause"):
                paused = True


# Initialize ironbow palette vertices & colors for gradient rendering
palette_vertices, palette_colors = config.ironbow_palette(0.92, 0.69, 0.079, 0.01)


def color_menu():
    """Render color selection menu."""
    global granule_type, ironbow

    with render.gui.sub_window("COLOR MENU", 0.87, 0.76, 0.13, 0.12) as sub:
        if sub.checkbox("Displacement (Ironbow)", ironbow):
            ironbow = True
            granule_type = False
        if sub.checkbox("Granule Type (Color)", granule_type):
            granule_type = True
            ironbow = False
        if sub.checkbox("Medium Default (Color)", not (granule_type or ironbow)):
            granule_type = False
            ironbow = False
        if ironbow:  # Display ironbow gradient palette
            # ironbow5: black -> dark blue -> magenta -> red-orange -> yellow-white
            render.canvas.triangles(palette_vertices, per_vertex_color=palette_colors)
            with render.gui.sub_window("displacement", 0.92, 0.70, 0.08, 0.06) as sub:
                sub.text(f"0       {peak_amplitude:.0e}m")


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
            normalized_position[i] = lattice.position_am[i] / lattice.max_universe_edge_am


def normalize_granule():
    """Normalize granule radius to 0-1 range for GGUI rendering"""
    global normalized_radius

    normalized_radius = max(
        granule.radius / lattice.max_universe_edge, 0.0001
    )  # Ensure minimum 0.01% of screen radius for visibility


# ================================================================
# Xperiment Rendering
# ================================================================


def render_xperiment(lattice):
    """Render 3D lattice with multiple wave sources using GGUI's 3D scene.

    Visualizes wave superposition from multiple sources, creating interference patterns
    where waves constructively and destructively combine.

    Args:
        lattice: Lattice instance with positions, directions, and universe parameters
    """
    global show_axis, block_slice, show_sources
    global radius_factor, freq_boost, amp_boost, paused
    global granule_type, ironbow
    global elapsed_t, frame
    global normalized_position
    global peak_amplitude

    # Initialize UI control variables
    show_axis = False  # Toggle to show/hide axis lines
    block_slice = False  # Block-slicing toggle
    show_sources = True  # Show wave sources toggle
    radius_factor = 1.0  # Initialize granule size factor
    freq_boost = 1.0  # Initialize frequency boost
    amp_boost = 1.0  # Initialize amplitude boost
    paused = False  # Pause toggle
    granule_type = False  # Granule type coloring toggle
    ironbow = True  # Ironbow (displacement) coloring toggle

    # Time tracking for radial harmonic oscillation of all granules
    elapsed_t = 0.0
    last_time = time.time()
    frame = 0  # Frame counter for diagnostics

    # Initialize normalized position (0-1 range for GGUI) & block-slicing
    # block-slicing: hide front 1/8th of the lattice for see-through effect
    normalized_position = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
    normalize_granule()

    # Convert phase from degrees to radians for physics calculations
    # Conversion: radians = degrees × π/180
    sources_phase_rad = [deg * ti.math.pi / 180 for deg in sources_phase_deg]

    ewave.build_source_vectors(
        sources_position, sources_phase_rad, NUM_SOURCES, lattice
    )  # compute distance & direction vectors to all sources

    # Print diagnostics header if enabled
    if WAVE_DIAGNOSTICS:
        diagnostics.print_initial_parameters()

    while render.window.running:
        render.init_scene(show_axis)  # Initialize scene with lighting and camera
        # Render UI overlay windows
        data_dashboard()
        controls()

        if not paused:
            # Calculate actual elapsed time (real-time tracking)
            current_time = time.time()
            dt_real = current_time - last_time
            last_time = current_time
            elapsed_t += dt_real  # Use real elapsed time instead of fixed DT

            # Apply radial harmonic oscillation to all granules from multiple wave sources
            # Each granule receives wave contributions from all active sources
            # Waves superpose creating interference patterns (constructive/destructive)
            ewave.oscillate_granules(
                lattice.position_am,  # Granule positions in attometers
                lattice.equilibrium_am,  # Rest positions for all granules
                lattice.amplitude_am,  # Granule amplitude in am
                lattice.velocity_am,  # Granule velocity in am/s
                lattice.granule_var_color,  # Granule color variations
                NUM_SOURCES,  # Number of active wave sources
                elapsed_t,
                freq_boost,  # Frequency visibility boost (will be applied over the slow-motion factor)
                amp_boost,  # Amplitude visibility boost for scaled lattices
            )

            # Update normalized positions for rendering (must happen after position updates)
            # with optional block-slicing (see-through effect)
            normalize_lattice(1 if block_slice else 0)

            # IN-FRAME DATA SAMPLING & DIAGNOSTICS ==================================
            # Update data sampling every 30 frames to reduce overhead
            if frame % 30 == 0:
                ewave.update_lattice_energy(lattice)  # Update energy based on wave amplitude
                peak_amplitude = ewave.peak_amplitude_am[None] * constants.ATTOMETER

            # Wave diagnostics (minimal footprint)
            if WAVE_DIAGNOSTICS:
                diagnostics.print_wave_diagnostics(
                    elapsed_t,
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
                per_vertex_color=lattice.granule_type_color,
            )
        elif ironbow:
            render.scene.particles(
                normalized_position,
                radius=normalized_radius * radius_factor,
                per_vertex_color=lattice.granule_var_color,
            )
            # Display ironbow gradient palette
            # ironbow5: black -> dark blue -> magenta -> red-orange -> yellow-white
            render.canvas.triangles(palette_vertices, per_vertex_color=palette_colors)
        else:
            render.scene.particles(
                normalized_position,
                radius=normalized_radius * radius_factor,
                color=config.COLOR_MEDIUM[1],
            )

        # Render the wave sources
        if show_sources:
            render.scene.particles(
                centers=ewave.sources_pos_field,
                radius=normalized_radius * 2,
                color=config.COLOR_SOURCE[1],
            )

        # Render final UI overlay and scene
        color_menu()
        xperiment_specs()
        render.show_scene()


# ================================================================
# Main calls
# ================================================================
if __name__ == "__main__":

    # Render the 3D lattice
    render_xperiment(lattice)
