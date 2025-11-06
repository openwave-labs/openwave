"""
XPERIMENT: Harmonic Oscillations

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

from openwave.common import config, constants
from openwave._io import render, video

import openwave.spacetime.medium_level0 as medium
import openwave.spacetime.wave_engine_level0 as ewave
import openwave.validations.wave_diagnostics as diagnostics

# Define the architecture to be used by Taichi (GPU vs CPU)
ti.init(arch=ti.gpu, log_level=ti.WARN)  # Use GPU if available, suppress info logs

# ================================================================
# Xperiment Parameters & Subatomic Objects Instantiation
# ================================================================

UNIVERSE_SIZE = [
    4 * constants.EWAVE_LENGTH,
    4 * constants.EWAVE_LENGTH,
    4 * constants.EWAVE_LENGTH,
]  # m, simulation domain [x, y, z] dimensions (can be asymmetric)
TICK_SPACING = 0.25  # Axis tick marks spacing for position reference

# Number of wave sources for this xperiment
NUM_SOURCES = 9

# Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
# Each row represents [x, y, z] coordinates for one source (Z-up coordinate system)
# Only provide NUM_SOURCES entries (only active sources needed)
sources_position = [
    [0.5, 0.5, 0.5],  # Wave Source 0 - Center
    [0.0, 1.0, 1.0],  # Wave Source 1 - Back-top-left corner
    [1.0, 0.0, 1.0],  # Wave Source 2 - Front-top-right corner
    [0.0, 1.0, 0.0],  # Wave Source 3 - Back-bottom-left corner
    [1.0, 0.0, 0.0],  # Wave Source 4 - Front-bottom-right corner
    [0.0, 0.0, 1.0],  # Wave Source 5 - Front-top-left corner
    [1.0, 1.0, 1.0],  # Wave Source 6 - Back-top-right corner
    [0.0, 0.0, 0.0],  # Wave Source 7 - Front-bottom-left corner
    [1.0, 1.0, 0.0],  # Wave Source 8 - Back-bottom-right corner
]

# Phase offsets for each source (integer degrees, converted to radians internally)
# Allows creating constructive/destructive interference patterns
# Only provide NUM_SOURCES entries (only active sources needed)
# Common patterns: 0° = in phase, 180° = opposite phase, 90° = quarter-cycle offset
sources_phase_deg = [
    180,  # Wave Source 0 (eg. 0 = in phase)
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

# Choose color theme for rendering (OCEAN, DESERT, FOREST)
COLOR_THEME = "OCEAN"

# Instantiate the lattice and granule objects (chose BCC or SC Lattice type)
lattice = medium.BCCLattice(UNIVERSE_SIZE, theme=COLOR_THEME)
granule = medium.BCCGranule(lattice.unit_cell_edge, lattice.max_universe_edge)

# Initialize UI control variables
show_axis = False  # Toggle to show/hide axis lines
block_slice = False  # Block-slicing toggle
show_sources = False  # Show wave sources toggle
radius_factor = 0.5  # Initialize granule size factor
freq_boost = 10.0  # Initialize frequency boost
amp_boost = 1.0  # Initialize amplitude boost
paused = False  # Pause toggle
granule_type = True  # Granule type coloring toggle
ironbow = False  # Ironbow coloring toggle
color_disp = True  # Ironbow displacement vs amplitude toggle

# Initialize time tracking for harmonic oscillations
elapsed_t = 0.0
last_time = time.time()
frame = 0  # Frame counter for diagnostics

# DATA SAMPLING & DIAGNOSTICS --------------------------------------------
max_displacement = 0.0  # Initialize granule max displacement (data sampling variable)
peak_amplitude = 0.0  # Initialize granule peak amplitude (data sampling variable)
WAVE_DIAGNOSTICS = False  # Toggle wave diagnostics (speed & wavelength measurements)
EXPORT_VIDEO = False  # Toggle frame image export to video directory
VIDEO_FRAMES = 24  # The target frame number to stop recording and finalize video export

# ================================================================
# Xperiment UI and overlay windows
# ================================================================

render.init_UI(
    UNIVERSE_SIZE, TICK_SPACING, cam_init_pos=[2.00, 1.50, 1.75]
)  # Initialize the GGUI window


def xperiment_specs():
    """Display xperiment definitions & specs."""
    with render.gui.sub_window("XPERIMENT: Spacetime Vibration", 0.00, 0.00, 0.19, 0.14) as sub:
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
        freq_boost = sub.slider_float("f Boost", freq_boost, 0.5, 10.0)
        amp_boost = sub.slider_float("Amp Boost", amp_boost, 1.0, 5.0)
        if paused:
            if sub.button("Continue"):
                paused = False
        else:
            if sub.button("Pause"):
                paused = True


# Initialize ironbow palette vertices & colors for gradient rendering
palette_vertices, palette_colors = config.ironbow_palette(0.92, 0.67, 0.079, 0.01)


def color_menu():
    """Render color selection menu."""
    global granule_type, ironbow, color_disp

    with render.gui.sub_window("COLOR MENU", 0.87, 0.74, 0.13, 0.14) as sub:
        if sub.checkbox("Displacement (ironbow)", ironbow and color_disp):
            ironbow = True
            color_disp = True
            granule_type = False
        if sub.checkbox("Amplitude (ironbow)", ironbow and not color_disp):
            ironbow = True
            color_disp = False
            granule_type = False
        if sub.checkbox("Granule Type (color)", granule_type):
            granule_type = True
            ironbow = False
        if sub.checkbox("Medium Default (color)", not (granule_type or ironbow)):
            granule_type = False
            ironbow = False
        if ironbow:  # Display ironbow gradient palette
            # ironbow5: black -> dark blue -> magenta -> red-orange -> yellow-white
            render.canvas.triangles(palette_vertices, per_vertex_color=palette_colors)
            with render.gui.sub_window(
                "displacement" if color_disp else "amplitude", 0.92, 0.68, 0.08, 0.06
            ) as sub:
                sub.text(f"0       {max_displacement if color_disp else peak_amplitude:.0e}m")


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
    global elapsed_t, last_time, frame, max_displacement, peak_amplitude

    ewave.build_source_vectors(
        sources_position, sources_phase_deg, NUM_SOURCES, lattice
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
            elapsed_t += dt_real  # Use real elapsed time instead of fixed DT
            last_time = current_time

            # Apply radial harmonic oscillation to all granules from multiple wave sources
            # Each granule receives wave contributions from all active sources
            # Waves superpose creating interference patterns (constructive/destructive)
            ewave.oscillate_granules(
                lattice.position_am,  # Granule positions in attometers
                lattice.equilibrium_am,  # Rest positions for all granules
                lattice.amplitude_am,  # Granule amplitude in am
                lattice.velocity_am,  # Granule velocity in am/s
                lattice.granule_var_color,  # Granule color variations
                freq_boost,  # Frequency visibility boost (will be applied over the slow-motion factor)
                amp_boost,  # Amplitude visibility boost for scaled lattices
                color_disp,  # Ironbow displacement vs amplitude toggle
                NUM_SOURCES,  # Number of active wave sources
                elapsed_t,
            )

            # Update normalized positions for rendering (must happen after position updates)
            # with optional block-slicing (see-through effect)
            lattice.normalize_to_screen(1 if block_slice else 0)

            # IN-FRAME DATA SAMPLING & DIAGNOSTICS ==================================
            # Update data sampling every 30 frames to reduce overhead
            if frame % 30 == 0:
                max_displacement = ewave.max_displacement_am[None] * constants.ATTOMETER
                peak_amplitude = ewave.peak_amplitude_am[None] * constants.ATTOMETER
                ewave.update_lattice_energy(lattice)  # Update energy based on wave amplitude

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

        # Render granules with optional coloring
        if granule_type:
            render.scene.particles(
                lattice.position_screen,
                radius=granule.radius_screen * radius_factor,
                per_vertex_color=lattice.granule_type_color,
            )
        elif ironbow:
            render.scene.particles(
                lattice.position_screen,
                radius=granule.radius_screen * radius_factor,
                per_vertex_color=lattice.granule_var_color,
            )
            # Display ironbow gradient palette
            # ironbow5: black -> dark blue -> magenta -> red-orange -> yellow-white
            render.canvas.triangles(palette_vertices, per_vertex_color=palette_colors)
        else:
            render.scene.particles(
                lattice.position_screen,
                radius=granule.radius_screen * radius_factor,
                color=config.COLOR_MEDIUM[1],
            )

        # Render the wave sources
        if show_sources:
            render.scene.particles(
                centers=ewave.sources_pos_field,
                radius=granule.radius_screen * 2,
                color=config.COLOR_SOURCE[1],
            )

        # Render final UI overlay and scene
        color_menu()
        xperiment_specs()
        render.show_scene()

        # Capture frame for video export (finalizes and stops at set VIDEO_FRAMES)
        if EXPORT_VIDEO:
            video.export(frame, VIDEO_FRAMES)


# ================================================================
# Main calls
# ================================================================
if __name__ == "__main__":

    # Render the 3D lattice
    render_xperiment(lattice)
