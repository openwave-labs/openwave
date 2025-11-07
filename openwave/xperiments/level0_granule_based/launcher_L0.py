"""
XPERIMENT LAUNCHER: Granule-Based Simulations

This unified xperiment launcher allows you to:
- Select and run any xperiment from the UI
- Switch between xperiments without restarting
- Maintain single source of truth for UI and rendering code

All xperiment-specific parameters are defined in separate parameter files
located in the _parameters/ directory.
"""

import taichi as ti
import time
import importlib
import sys
from pathlib import Path

from openwave.common import config, constants
from openwave._io import render, video

import openwave.spacetime.medium_level0 as medium
import openwave.spacetime.wave_engine_level0 as ewave
import openwave.validations.wave_diagnostics as diagnostics

# ================================================================
# XPERIMENT PARAMETERS MANAGEMENT
# ================================================================


class XperimentManager:
    """Manages loading and switching between xperiment parameters."""

    def __init__(self):
        self.available_xperiments = self._discover_xperiments()
        self.current_xperiment = None
        self.current_parameters = None

    def _discover_xperiments(self):
        """Discover all available xperiment parameters in the _parameters/ directory."""
        parameters_dir = Path(__file__).parent / "_parameters"
        xperiment_files = []

        if parameters_dir.exists():
            for file in parameters_dir.glob("*.py"):
                if file.name != "__init__.py":
                    xperiment_files.append(file.stem)

        # Sort alphabetically for consistent ordering
        return sorted(xperiment_files)

    def load_xperiment(self, xperiment_name):
        """Load an xperiment parameters by name.

        Args:
            xperiment_name: Name of the xperiment parameter file (without .py extension)

        Returns:
            dict: Parameters dictionary
        """
        try:
            # Import the parameters module dynamically
            module_path = f"openwave.xperiments.level0_granule_based._parameters.{xperiment_name}"
            parameters_module = importlib.import_module(module_path)

            # Reload the module to ensure fresh parameters (useful during development)
            importlib.reload(parameters_module)

            self.current_xperiment = xperiment_name
            self.current_parameters = parameters_module.PARAMETERS

            return self.current_parameters

        except Exception as e:
            print(f"Error loading xperiment '{xperiment_name}': {e}")
            return None

    def get_xperiment_display_name(self, xperiment_name):
        """Get the display name for an xperiment."""
        # Convert snake_case to Title Case
        return " ".join(word.capitalize() for word in xperiment_name.split("_"))


# ================================================================
# SIMULATION STATE
# ================================================================


class SimulationState:
    """Manages the state of the simulation."""

    def __init__(self):
        self.lattice = None
        self.granule = None
        self.elapsed_t = 0.0
        self.last_time = time.time()
        self.frame = 0
        self.max_displacement = 0.0
        self.peak_amplitude = 0.0

        # Current xperiment parameters
        self.X_NAME = ""
        self.CAM_INIT = [2.00, 1.50, 1.75]
        self.UNIVERSE_SIZE = []
        self.TICK_SPACING = 0.25
        self.NUM_SOURCES = 1
        self.SOURCES_POSITION = []
        self.SOURCES_PHASE_DEG = []
        self.COLOR_THEME = "OCEAN"

        # UI control variables
        self.show_axis = False
        self.block_slice = False
        self.show_sources = False
        self.radius_factor = 0.5
        self.freq_boost = 10.0
        self.amp_boost = 1.0
        self.paused = False
        self.granule_type = True
        self.ironbow = False
        self.blueprint = False
        self.var_displacement = True

        # Diagnostics & video export toggles
        self.WAVE_DIAGNOSTICS = False
        self.EXPORT_VIDEO = False
        self.VIDEO_FRAMES = 24

    def reset(self):
        """Reset simulation state for a new xperiment."""
        self.elapsed_t = 0.0
        self.last_time = time.time()
        self.frame = 0
        self.max_displacement = 0.0
        self.peak_amplitude = 0.0

    def apply_parameters(self, parameters_dict):
        """Apply parameters from dictionary."""
        # Meta
        self.X_NAME = parameters_dict["meta"]["name"]

        # Camera
        self.CAM_INIT = parameters_dict["camera"]["initial_position"]

        # Universe
        universe_cfg = parameters_dict["universe"]
        size_multipliers = universe_cfg["size_multipliers"]
        self.UNIVERSE_SIZE = [
            size_multipliers[0] * constants.EWAVE_LENGTH,
            size_multipliers[1] * constants.EWAVE_LENGTH,
            size_multipliers[2] * constants.EWAVE_LENGTH,
        ]
        self.TICK_SPACING = universe_cfg["tick_spacing"]
        self.COLOR_THEME = universe_cfg["color_theme"]

        # Wave sources
        sources_cfg = parameters_dict["wave_sources"]
        self.NUM_SOURCES = sources_cfg["count"]
        self.SOURCES_POSITION = sources_cfg["positions"]
        self.SOURCES_PHASE_DEG = sources_cfg["phase_offsets_deg"]

        # UI defaults
        ui_cfg = parameters_dict["ui_defaults"]
        self.show_axis = ui_cfg["show_axis"]
        self.block_slice = ui_cfg["block_slice"]
        self.show_sources = ui_cfg["show_sources"]
        self.radius_factor = ui_cfg["radius_factor"]
        self.freq_boost = ui_cfg["freq_boost"]
        self.amp_boost = ui_cfg["amp_boost"]
        self.paused = ui_cfg["paused"]
        self.granule_type = ui_cfg["granule_type"]
        self.ironbow = ui_cfg["ironbow"]
        self.blueprint = ui_cfg["blueprint"]
        self.var_displacement = ui_cfg["var_displacement"]

        # Diagnostics
        diag_cfg = parameters_dict["diagnostics"]
        self.WAVE_DIAGNOSTICS = diag_cfg["wave_diagnostics"]
        self.EXPORT_VIDEO = diag_cfg["export_video"]
        self.VIDEO_FRAMES = diag_cfg["video_frames"]

    def initialize_lattice(self):
        """Initialize or reinitialize the lattice and granule objects."""
        self.lattice = medium.BCCLattice(self.UNIVERSE_SIZE, theme=self.COLOR_THEME)
        self.granule = medium.BCCGranule(
            self.lattice.unit_cell_edge, self.lattice.max_universe_edge
        )


# ================================================================
# UI OVERLAY WINDOWS
# ================================================================


def xperiment_specs(state):
    """Display xperiment definitions & specs."""
    with render.gui.sub_window(f"LEVEL-0: Granule-Based Medium", 0.00, 0.00, 0.19, 0.10) as sub:
        sub.text(f"Wave Source: {state.NUM_SOURCES} Harmonic Oscillators")
        sub.text("Coupling: Phase Sync")
        sub.text("Propagation: Radial from Source")


def xperiment_launcher(xperiment_mgr, state):
    """Display xperiment launcher overlay at top-left.

    Args:
        xperiment_mgr: XperimentManager instance
        state: SimulationState instance

    Returns:
        str or None: Name of xperiment to switch to, or None
    """
    selected_xperiment = None

    with render.gui.sub_window("XPERIMENT LAUNCHER", 0.00, 0.10, 0.13, 0.23) as sub:
        # Display all available xperiments as selectable options
        for xp_name in xperiment_mgr.available_xperiments:
            display_name = xperiment_mgr.get_xperiment_display_name(xp_name)
            is_current = xp_name == xperiment_mgr.current_xperiment

            # Use checkbox to show selection (radio button style)
            if sub.checkbox(f"{display_name}", is_current):
                if not is_current:
                    selected_xperiment = xp_name
        sub.text("(needs window restart)", color=config.LIGHT_BLUE[1])

    return selected_xperiment


def data_dashboard(state):
    """Display simulation data dashboard."""
    with render.gui.sub_window("DATA-DASHBOARD", 0.00, 0.41, 0.19, 0.59) as sub:
        sub.text("--- WAVE-MEDIUM ---", color=config.LIGHT_BLUE[1])
        sub.text(f"Universe Size: {state.lattice.max_universe_edge:.1e} m (max edge)")
        sub.text(f"Granule Count: {state.lattice.total_granules:,} particles")
        sub.text(f"Medium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")

        sub.text("\n--- Scaling-Up (for computation) ---", color=config.LIGHT_BLUE[1])
        sub.text(f"Factor: {state.lattice.scale_factor:.1e} x Planck Scale")
        sub.text(f"Unit-Cells per Max Edge: {state.lattice.max_grid_size:,}")
        sub.text(f"Unit-Cell Edge: {state.lattice.unit_cell_edge:.2e} m")
        sub.text(f"Granule Radius: {state.granule.radius * state.radius_factor:.2e} m")
        sub.text(f"Granule Mass: {state.granule.mass:.2e} kg")

        sub.text("\n--- Sim Resolution (linear) ---", color=config.LIGHT_BLUE[1])
        sub.text(f"EWave: {state.lattice.ewave_res:.0f} granules/ewave (>10)")
        if state.lattice.ewave_res < 10:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Universe: {state.lattice.max_uni_res:.1f} ewaves/universe-edge")

        sub.text("\n--- ENERGY-WAVE ---", color=config.LIGHT_BLUE[1])
        sub.text(f"EWAVE Speed (c): {constants.EWAVE_SPEED:.1e} m/s")
        sub.text(f"EWAVE Wavelength (lambda): {constants.EWAVE_LENGTH:.1e} m")
        sub.text(f"EWAVE Frequency (f): {constants.EWAVE_FREQUENCY:.1e} Hz")
        sub.text(f"EWAVE Amplitude (A): {constants.EWAVE_AMPLITUDE:.1e} m")

        sub.text("\n--- Sim Universe Wave Energy ---", color=config.LIGHT_BLUE[1])
        sub.text(f"Energy: {state.lattice.energy:.1e} J ({state.lattice.energy_kWh:.1e} KWh)")

        sub.text("\n--- TIME MICROSCOPE ---", color=config.LIGHT_BLUE[1])
        slowed_mo = config.SLOW_MO / state.freq_boost
        fps = 0 if state.elapsed_t == 0 else state.frame / state.elapsed_t
        sub.text(f"Frames Rendered: {state.frame}")
        sub.text(f"Real Time: {state.elapsed_t / slowed_mo:.2e}s ({fps * slowed_mo:.0e} FPS)")
        sub.text(f"(1 real second = {slowed_mo / (60*60*24*365):.0e}y of sim time)")
        sub.text(f"Sim Time (slow-mo): {state.elapsed_t:.2f}s ({fps:.0f} FPS)")


def controls(state):
    """Render the controls UI overlay."""
    # Create overlay windows for controls
    with render.gui.sub_window("CONTROLS", 0.85, 0.00, 0.15, 0.22) as sub:
        state.show_axis = sub.checkbox(f"Axis (tick marks: {state.TICK_SPACING})", state.show_axis)
        state.block_slice = sub.checkbox("Block Slice", state.block_slice)
        state.show_sources = sub.checkbox("Show Wave Sources", state.show_sources)
        state.radius_factor = sub.slider_float("Granule", state.radius_factor, 0.1, 2.0)
        state.freq_boost = sub.slider_float("f Boost", state.freq_boost, 0.1, 10.0)
        state.amp_boost = sub.slider_float("Amp Boost", state.amp_boost, 0.1, 5.0)
        if state.paused:
            if sub.button("Continue"):
                state.paused = False
        else:
            if sub.button("Pause"):
                state.paused = True


def color_menu(
    state, ib_palette_vertices, ib_palette_colors, bp_palette_vertices, bp_palette_colors
):
    """Render color selection menu."""
    with render.gui.sub_window("COLOR MENU", 0.87, 0.72, 0.13, 0.16) as sub:
        if sub.checkbox("Displacement (ironbow)", state.ironbow and state.var_displacement):
            state.granule_type = False
            state.ironbow = True
            state.blueprint = False
            state.var_displacement = True
        if sub.checkbox("Amplitude (ironbow)", state.ironbow and not state.var_displacement):
            state.granule_type = False
            state.ironbow = True
            state.blueprint = False
            state.var_displacement = False
        if sub.checkbox("Amplitude (blueprint)", state.blueprint and not state.var_displacement):
            state.granule_type = False
            state.ironbow = False
            state.blueprint = True
            state.var_displacement = False
        if sub.checkbox("Granule Type Color", state.granule_type):
            state.granule_type = True
            state.ironbow = False
            state.blueprint = False
            state.var_displacement = False
        if sub.checkbox(
            "Medium Default Color",
            not (state.granule_type or state.ironbow or state.blueprint),
        ):
            state.granule_type = False
            state.ironbow = False
            state.blueprint = False
            state.var_displacement = False
        if state.ironbow:  # Display ironbow gradient palette
            # ironbow5: black -> dark blue -> magenta -> red-orange -> yellow-white
            render.canvas.triangles(ib_palette_vertices, per_vertex_color=ib_palette_colors)
            with render.gui.sub_window(
                "displacement" if state.var_displacement else "amplitude",
                0.92,
                0.66,
                0.08,
                0.06,
            ) as sub:
                sub.text(
                    f"0       {state.max_displacement if state.var_displacement else state.peak_amplitude:.0e}m"
                )
        if state.blueprint:  # Display blueprint gradient palette
            # blueprint4: dark blue -> medium blue -> light blue -> extra-light blue
            render.canvas.triangles(bp_palette_vertices, per_vertex_color=bp_palette_colors)
            with render.gui.sub_window(
                "displacement" if state.var_displacement else "amplitude",
                0.92,
                0.66,
                0.08,
                0.06,
            ) as sub:
                sub.text(
                    f"0       {state.max_displacement if state.var_displacement else state.peak_amplitude:.0e}m"
                )


# ================================================================
# XPERIMENT RENDERING
# ================================================================


def initialize_xperiment(state):
    """Initialize xperiment sources and diagnostics (called once after lattice init).

    Args:
        state: SimulationState instance containing all xperiment parameters and state
    """
    # Build source vectors (distance & direction to all sources) - only once!
    ewave.build_source_vectors(
        state.SOURCES_POSITION, state.SOURCES_PHASE_DEG, state.NUM_SOURCES, state.lattice
    )

    # Print diagnostics header if enabled
    if state.WAVE_DIAGNOSTICS:
        diagnostics.print_initial_parameters()


def compute_motion(state):
    """Compute 3D lattice motion with multiple wave sources.

    Visualizes wave superposition from multiple sources, creating interference patterns
    where waves constructively and destructively combine.

    Args:
        state: SimulationState instance containing all xperiment parameters and state
    """
    # Apply radial harmonic oscillation to all granules from multiple wave sources
    # Each granule receives wave contributions from all active sources
    # Waves superpose creating interference patterns (constructive/destructive)
    ewave.oscillate_granules(
        state.lattice.position_am,  # Granule positions in attometers
        state.lattice.equilibrium_am,  # Rest positions for all granules
        state.lattice.amplitude_am,  # Granule amplitude in am
        state.lattice.velocity_am,  # Granule velocity in am/s
        state.lattice.granule_var_color,  # Granule color variations
        state.freq_boost,  # Frequency visibility boost (applied over slow-motion factor)
        state.amp_boost,  # Amplitude visibility boost for scaled lattices
        state.ironbow,  # Ironbow coloring toggle
        state.var_displacement,  # Displacement vs amplitude toggle
        state.NUM_SOURCES,  # Number of active wave sources
        state.elapsed_t,
    )

    # Update normalized positions for rendering (must happen after position updates)
    # with optional block-slicing (see-through effect)
    state.lattice.normalize_to_screen(1 if state.block_slice else 0)

    # IN-FRAME DATA SAMPLING & DIAGNOSTICS ==================================
    # Update data sampling every 30 frames to reduce overhead
    if state.frame % 30 == 0:
        state.max_displacement = ewave.max_displacement_am[None] * constants.ATTOMETER
        state.peak_amplitude = ewave.peak_amplitude_am[None] * constants.ATTOMETER
        ewave.update_lattice_energy(state.lattice)  # Update energy based on wave amplitude

    # Wave diagnostics (minimal footprint)
    if state.WAVE_DIAGNOSTICS:
        diagnostics.print_wave_diagnostics(
            state.elapsed_t,
            state.frame,
            print_interval=100,  # Print every 100 frames
        )


def render_elements(state):
    """Render the 3D scene elements with granules and wave sources."""
    # Render granules with optional coloring
    if state.granule_type:
        render.scene.particles(
            state.lattice.position_screen,
            radius=state.granule.radius_screen * state.radius_factor,
            per_vertex_color=state.lattice.granule_type_color,
        )
    elif state.ironbow or state.blueprint:
        render.scene.particles(
            state.lattice.position_screen,
            radius=state.granule.radius_screen * state.radius_factor,
            per_vertex_color=state.lattice.granule_var_color,
        )
    else:
        render.scene.particles(
            state.lattice.position_screen,
            radius=state.granule.radius_screen * state.radius_factor,
            color=config.COLOR_MEDIUM[1],
        )

    # Render the wave sources
    if state.show_sources:
        render.scene.particles(
            centers=ewave.sources_pos_field,
            radius=state.granule.radius_screen * 2,
            color=config.COLOR_SOURCE[1],
        )


# ================================================================
# MAIN LOOP
# ================================================================


def main():
    """Main entry point for the xperiment launcher."""
    # Parse command-line arguments for xperiment selection, CLI
    selected_xperiment_arg = None
    if len(sys.argv) > 1:
        selected_xperiment_arg = sys.argv[1]

    # Initialize Taichi
    ti.init(arch=ti.gpu, log_level=ti.WARN)  # Use GPU if available, suppress info logs

    # Initialize palette vertices & colors for gradient rendering (after ti.init)
    ib_palette_vertices, ib_palette_colors = config.ironbow_palette(0.92, 0.65, 0.079, 0.01)
    bp_palette_vertices, bp_palette_colors = config.blueprint_palette(0.92, 0.65, 0.079, 0.01)

    # Initialize xperiment manager and state
    xperiment_mgr = XperimentManager()
    state = SimulationState()

    # Load xperiment (from command-line arg or default, CLI)
    default_xperiment = selected_xperiment_arg or "spacetime_vibration"
    if default_xperiment in xperiment_mgr.available_xperiments:
        parameters_dict = xperiment_mgr.load_xperiment(default_xperiment)
        if parameters_dict:
            state.apply_parameters(parameters_dict)
            state.initialize_lattice()
            initialize_xperiment(state)  # Initialize sources and diagnostics (once)
    else:
        print(f"Error: Xperiment '{default_xperiment}' not found!")
        return

    # Initialize GGUI UI with initial xperiment parameters
    render.init_UI(state.UNIVERSE_SIZE, state.TICK_SPACING, state.CAM_INIT)

    # Main rendering loop
    while render.window.running:
        render.init_scene(state.show_axis)  # Initialize scene with lighting and camera

        # Handle keyboard shortcuts for window close
        if render.window.is_pressed(ti.ui.ESCAPE):
            render.window.running = False
            break

        # Render UI overlay windows
        new_xperiment = xperiment_launcher(xperiment_mgr, state)
        data_dashboard(state)
        controls(state)

        # Handle xperiment switching - restart program with new xperiment
        if new_xperiment:
            print("\n================================================================")
            print("XPERIMENT LAUNCH")
            print(f"Now running: {new_xperiment}\n")

            # Flush output before restart
            sys.stdout.flush()
            sys.stderr.flush()

            # Launch new instance using nohup (detaches from terminal)
            python = sys.executable
            script = __file__

            import subprocess
            import shutil

            # Check if nohup is available, otherwise fallback to direct execution
            nohup_path = shutil.which("nohup")

            # Launch detached process (logs will be visible)
            subprocess.Popen(
                [python, script, new_xperiment],
                start_new_session=True,
            )

            # Give subprocess a moment to start
            time.sleep(0.1)

            # Close current window and exit
            render.window.running = False
            break  # Exit render loop immediately

        if not state.paused:
            # Calculate actual elapsed time (real-time tracking)
            current_time = time.time()
            dt_real = current_time - state.last_time
            state.elapsed_t += dt_real  # Use real elapsed time instead of fixed DT
            state.last_time = current_time

            # Run xperiment simulation step
            compute_motion(state)
            state.frame += 1  # Increment frame counter

        else:
            # Update last_time during pause to prevent time jump on resume
            state.last_time = time.time()

        # Render 3D scene elements
        render_elements(state)

        # Render final UI overlay and scene
        color_menu(
            state, ib_palette_vertices, ib_palette_colors, bp_palette_vertices, bp_palette_colors
        )
        xperiment_specs(state)
        render.show_scene()

        # Capture frame for video export (finalizes and stops at set VIDEO_FRAMES)
        if state.EXPORT_VIDEO:
            video.export(state.frame, state.VIDEO_FRAMES)


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()
