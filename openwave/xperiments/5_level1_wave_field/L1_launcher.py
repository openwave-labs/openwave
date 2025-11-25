"""
[WIP] L1 XPERIMENT LAUNCHER

Unified launcher for Level-1 wave-field xperiments featuring:
- UI-based xperiment selection and switching
- Single source of truth for rendering and UI code
- Xperiment-specific parameters in /_xparameters directory
"""

import webbrowser
import importlib
import sys
import os
from pathlib import Path

import numpy as np
import taichi as ti

from openwave.common import colormap, constants
from openwave._io import flux_mesh, render, video

import openwave.spacetime.L1_field_grid as data_grid
import openwave.spacetime.L1_wave_engine as ewave
import openwave.validations.wave_diagnostics as diagnostics

# ================================================================
# XPERIMENT PARAMETERS MANAGEMENT
# ================================================================


class XperimentManager:
    """Manages loading and switching between xperiment parameters."""

    def __init__(self):
        self.available_xperiments = self._discover_xperiments()
        self.xperiment_display_names = {}  # Cache display names from meta
        self.current_xperiment = None
        self.current_xparameters = None

    def _discover_xperiments(self):
        """Discover all available xperiment parameters in /_xparameters directory."""
        parameters_dir = Path(__file__).parent / "_xparameters"

        if not parameters_dir.exists():
            return []

        xperiment_files = [
            file.stem for file in parameters_dir.glob("*.py") if file.name != "__init__.py"
        ]

        return sorted(xperiment_files)

    def load_xperiment(self, xperiment_name):
        """Load xperiment parameters by name.

        Args:
            xperiment_name: Parameter file name without .py extension

        Returns:
            dict: Parameters dictionary or None if loading fails
        """
        try:
            module_path = f"openwave.xperiments.5_level1_wave_field._xparameters.{xperiment_name}"
            parameters_module = importlib.import_module(module_path)
            importlib.reload(parameters_module)  # Reload for fresh parameters

            self.current_xperiment = xperiment_name
            self.current_xparameters = parameters_module.XPARAMETERS

            # Cache display name from meta
            self.xperiment_display_names[xperiment_name] = parameters_module.XPARAMETERS["meta"][
                "X_NAME"
            ]

            return self.current_xparameters

        except Exception as e:
            print(f"Error loading xperiment '{xperiment_name}': {e}")
            return None

    def get_xperiment_display_name(self, xperiment_name):
        """Get display name from parameter meta or fallback to filename conversion."""
        if xperiment_name in self.xperiment_display_names:
            return self.xperiment_display_names[xperiment_name]

        # Fallback: try to load just for the name
        try:
            module_path = f"openwave.xperiments.5_level1_wave_field._xparameters.{xperiment_name}"
            parameters_module = importlib.import_module(module_path)
            display_name = parameters_module.XPARAMETERS["meta"]["X_NAME"]
            self.xperiment_display_names[xperiment_name] = display_name
            return display_name
        except:
            # Final fallback: convert filename
            return " ".join(word.capitalize() for word in xperiment_name.split("_"))


# ================================================================
# SIMULATION STATE
# ================================================================


class SimulationState:
    """Manages the state of the simulation."""

    def __init__(self):
        self.wave_field = None
        self.c_slowed = 0.0
        self.dt = 0.0
        self.cfl_factor = 0.0
        self.elapsed_t = 0.0
        self.frame = 0
        self.peak_amplitude = 0.0

        # Current xperiment parameters
        self.X_NAME = ""
        self.CAM_INIT = [2.00, 1.50, 1.75]
        self.UNIVERSE_SIZE = []
        self.TARGET_VOXELS = 1e8
        self.SLO_MO = None

        # UI control variables
        self.SHOW_AXIS = False
        self.TICK_SPACING = 0.25
        self.SHOW_GRID = False
        self.FLUX_MESH_OPTION = 0
        self.SIM_SPEED = 1.0
        self.PAUSED = False

        # Color control variables
        self.COLOR_THEME = "OCEAN"
        self.COLOR_PALETTE = 1  # Color palette index
        self.VAR_AMP = False

        # Diagnostics & video export toggles
        self.WAVE_DIAGNOSTICS = False
        self.EXPORT_VIDEO = False
        self.VIDEO_FRAMES = 24

    def apply_xparameters(self, params):
        """Apply parameters from xperiment parameter dictionary."""
        # Meta
        self.X_NAME = params["meta"]["X_NAME"]

        # Camera
        self.CAM_INIT = params["camera"]["INITIAL_POSITION"]

        # Universe
        universe = params["universe"]
        self.UNIVERSE_SIZE = list(universe["SIZE"])
        self.TARGET_VOXELS = universe["TARGET_VOXELS"]
        self.SLO_MO = universe["SLO_MO"]

        # UI defaults
        ui = params["ui_defaults"]
        self.SHOW_AXIS = ui["SHOW_AXIS"]
        self.TICK_SPACING = ui["TICK_SPACING"]
        self.SHOW_GRID = ui["SHOW_GRID"]
        self.FLUX_MESH_OPTION = ui["FLUX_MESH_OPTION"]
        self.SIM_SPEED = ui["SIM_SPEED"]
        self.PAUSED = ui["PAUSED"]

        # Color defaults
        color = params["color_defaults"]
        self.COLOR_THEME = color["COLOR_THEME"]
        self.COLOR_PALETTE = color["COLOR_PALETTE"]
        self.VAR_AMP = color["VAR_AMP"]

        # Diagnostics
        diag = params["diagnostics"]
        self.WAVE_DIAGNOSTICS = diag["WAVE_DIAGNOSTICS"]
        self.EXPORT_VIDEO = diag["EXPORT_VIDEO"]
        self.VIDEO_FRAMES = diag["VIDEO_FRAMES"]

    def initialize_grid(self):
        """Initialize or reinitialize the wave field grid."""
        self.wave_field = data_grid.WaveField(self.UNIVERSE_SIZE, self.TARGET_VOXELS)

    def compute_timestep(self):
        # Compute maximum safe timestep from CFL condition with safety margin.
        # CFL Condition: dt ≤ dx / (c × √3)
        # With SLO_MO applied: dt_critical = dx / (c_slowed × √3)
        # SIM_SPEED affects perceived speed
        self.c_slowed = constants.EWAVE_SPEED / self.SLO_MO * self.SIM_SPEED  # m/s
        self.dt = self.wave_field.dx / (constants.EWAVE_SPEED / self.SLO_MO * (3**0.5))  # s
        self.cfl_factor = round((self.c_slowed * self.dt / self.wave_field.dx) ** 2, 7)

    def reset_sim(self):
        """Reset simulation state."""
        self.wave_field = None
        self.c_slowed = 0.0
        self.dt = 0.0
        self.cfl_factor = 0.0
        self.elapsed_t = 0.0
        self.frame = 0
        self.peak_amplitude = 0.0
        self.initialize_grid()
        self.compute_timestep()
        initialize_xperiment(self)


# ================================================================
# UI OVERLAY WINDOWS
# ================================================================


def display_xperiment_launcher(xperiment_mgr, state):
    """Display xperiment launcher UI with selectable xperiments.

    Args:
        xperiment_mgr: XperimentManager instance
        state: SimulationState instance

    Returns:
        str or None: Selected xperiment name or None
    """
    selected_xperiment = None

    with render.gui.sub_window("XPERIMENT LAUNCHER (L1)", 0.00, 0.00, 0.13, 0.33) as sub:
        sub.text("(needs window reload)", color=colormap.LIGHT_BLUE[1])
        for xp_name in xperiment_mgr.available_xperiments:
            display_name = xperiment_mgr.get_xperiment_display_name(xp_name)
            is_current = xp_name == xperiment_mgr.current_xperiment

            if sub.checkbox(display_name, is_current) and not is_current:
                selected_xperiment = xp_name

        if sub.button("Close Launcher (esc)"):
            render.window.running = False

    return selected_xperiment


def display_controls(state):
    """Display the controls UI overlay."""
    with render.gui.sub_window("CONTROLS", 0.00, 0.34, 0.16, 0.17) as sub:
        state.SHOW_AXIS = sub.checkbox(f"Axis (ticks: {state.TICK_SPACING})", state.SHOW_AXIS)
        state.FLUX_MESH_OPTION = sub.slider_int("Flux Mesh", state.FLUX_MESH_OPTION, 0, 3)
        state.SIM_SPEED = sub.slider_float("Speed", state.SIM_SPEED, 0.5, 1.0)
        if state.PAUSED:
            if sub.button("Propagate eWave"):
                state.PAUSED = False
        else:
            if sub.button("Pause"):
                state.PAUSED = True
        if sub.button("Reset Sim"):
            state.reset_sim()


def display_color_menu(state):
    """Display color selection menu."""
    tracker = "amplitude" if state.VAR_AMP else "displacement"
    with render.gui.sub_window("COLOR MENU", 0.00, 0.70, 0.14, 0.17) as sub:
        if sub.checkbox(
            "Displacement (blueprint)", state.COLOR_PALETTE == 2 and not state.VAR_AMP
        ):
            state.COLOR_PALETTE = 2
            state.VAR_AMP = False
        if sub.checkbox("Displacement (redshift)", state.COLOR_PALETTE == 3 and not state.VAR_AMP):
            state.COLOR_PALETTE = 3
            state.VAR_AMP = False
        if sub.checkbox("Amplitude (ironbow)", state.COLOR_PALETTE == 1 and state.VAR_AMP):
            state.COLOR_PALETTE = 1
            state.VAR_AMP = True
        if sub.checkbox("Amplitude (blueprint)", state.COLOR_PALETTE == 2 and state.VAR_AMP):
            state.COLOR_PALETTE = 2
            state.VAR_AMP = True
        if state.COLOR_PALETTE == 1:  # Display ironbow gradient palette
            render.canvas.triangles(ib_palette_vertices, per_vertex_color=ib_palette_colors)
            with render.gui.sub_window(tracker, 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(f"0       {state.peak_amplitude:.0e}m")
        if state.COLOR_PALETTE == 2:  # Display blueprint gradient palette
            render.canvas.triangles(bp_palette_vertices, per_vertex_color=bp_palette_colors)
            with render.gui.sub_window(tracker, 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(f"0       {state.peak_amplitude:.0e}m")
        if state.COLOR_PALETTE == 3:  # Display redshift gradient palette
            render.canvas.triangles(rs_palette_vertices, per_vertex_color=rs_palette_colors)
            with render.gui.sub_window(tracker, 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(f"0       {state.peak_amplitude:.0e}m")


def display_level_specs(state, level_bar_vertices):
    """Display OpenWave level specifications overlay."""
    render.canvas.triangles(level_bar_vertices, color=colormap.LIGHT_BLUE[1])
    with render.gui.sub_window("LEVEL-1: FIELD-BASED METHOD", 0.82, 0.01, 0.18, 0.10) as sub:
        sub.text("Coupling: Phase Sync")
        sub.text("Propagation: Radial from Source")
        if sub.button("Wave Notation Guide"):
            webbrowser.open(
                "https://github.com/openwave-labs/openwave/blob/main/openwave/wave_notation.md"
            )


def display_data_dashboard(state):
    """Display simulation data dashboard."""
    with render.gui.sub_window("DATA-DASHBOARD", 0.82, 0.41, 0.18, 0.59) as sub:
        sub.text("--- SPACETIME ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Universe Size: {state.wave_field.max_universe_edge:.1e} m (max edge)")
        sub.text(f"Medium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")
        sub.text(f"eWAVE Speed (c): {constants.EWAVE_SPEED:.1e} m/s")

        sub.text("\n--- eWAVE-FIELD ---", color=colormap.LIGHT_BLUE[1])
        sub.text(
            f"Grid Size: {state.wave_field.nx} x {state.wave_field.ny} x {state.wave_field.nz} voxels"
        )
        sub.text(f"Voxel Count: {state.wave_field.voxel_count:,} voxels")
        sub.text(f"Voxel Edge: {state.wave_field.dx:.2e} m")

        sub.text("\n--- Sim Resolution (linear) ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"eWave: {state.wave_field.ewave_res:.1f} voxels/lambda (>10)")
        if state.wave_field.ewave_res < 10:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Universe: {state.wave_field.max_uni_res:.1f} lambda/universe-edge")

        sub.text("\n--- eWAVE-PROFILING ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"eWAVE Frequency (f): {constants.EWAVE_FREQUENCY:.1e} Hz")
        sub.text(f"eWAVE Amplitude (A): {constants.EWAVE_AMPLITUDE:.1e} m")
        sub.text(f"eWAVE Wavelength (lambda): {constants.EWAVE_LENGTH:.1e} m")
        sub.text(
            f"Energy: {state.wave_field.energy:.1e} J ({state.wave_field.energy_kWh:.1e} KWh)"
        )

        sub.text("\n--- TIME MICROSCOPE ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Frames Rendered: {state.frame}")
        sub.text(f"Sim Elapsed Time: {state.elapsed_t / state.SLO_MO:.2e}s")
        sub.text(f"(1 real second = {state.SLO_MO / (60*60*24*365):.0e}y of sim time)")

        sub.text("\n--- TIMESTEP ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"c_slowed: {state.c_slowed:.2e} m/s")
        sub.text(
            f"dt: {state.dt:.3f}s (needs {1/state.dt:.0f} FPS)",
            color=(1.0, 1.0, 1.0) if state.cfl_factor <= (1 / 3) else (1.0, 0.0, 0.0),
        )
        sub.text(
            f"CFL Factor: {state.cfl_factor:.3f} (target < 1/3)",
            color=((1.0, 1.0, 1.0) if state.cfl_factor <= (1 / 3) else (1.0, 0.0, 0.0)),
        )


# ================================================================
# XPERIMENT RENDERING
# ================================================================


def initialize_xperiment(state):
    """Initialize color palettes, test patterns and diagnostics.

    Args:
        state: SimulationState instance with xperiment parameters
    """
    global ib_palette_vertices, ib_palette_colors
    global rs_palette_vertices, rs_palette_colors
    global bp_palette_vertices, bp_palette_colors
    global level_bar_vertices

    # Initialize color palette scales for gradient rendering and level indicator
    ib_palette_vertices, ib_palette_colors = colormap.get_palette_scale(
        colormap.ironbow, 0.00, 0.63, 0.079, 0.01
    )
    bp_palette_vertices, bp_palette_colors = colormap.get_palette_scale(
        colormap.blueprint, 0.00, 0.63, 0.079, 0.01
    )
    rs_palette_vertices, rs_palette_colors = colormap.get_palette_scale(
        colormap.redshift, 0.00, 0.63, 0.079, 0.01
    )
    level_bar_vertices = colormap.get_level_bar_geometry(0.82, 0.00, 0.179, 0.01)

    # Initialize test displacement pattern for flux mesh visualization
    # TODO: remove multiple charge post-propagation implementation
    ewave.charge_gaussian(state.wave_field, state.c_slowed, state.dt)
    # ewave.charge_1lambda(state.wave_field, state.c_slowed, state.dt)
    # ewave.charge_falloff(state.wave_field, state.c_slowed, state.dt)
    # ewave.charge_full(state.wave_field, state.c_slowed, state.dt)
    # TODO: code toggle to plot initial displacement profile
    ewave.plot_charge_profile(state.wave_field)

    if state.WAVE_DIAGNOSTICS:
        diagnostics.print_initial_parameters()


def compute_wave_motion(state):
    """Compute wave propagation, reflection and superposition and update visualization.

    Args:
        state: SimulationState instance with xperiment parameters
    """

    ewave.propagate_ewave(state.wave_field, state.c_slowed, state.dt)
    # TODO: Implement IN-FRAME DATA SAMPLING & DIAGNOSTICS


def render_elements(state):
    """Render grid, flux mesh and test particles."""
    if state.SHOW_GRID:
        render.scene.lines(state.wave_field.grid_lines, width=1, color=colormap.COLOR_MEDIUM[1])

    if state.FLUX_MESH_OPTION > 0:
        ewave.update_flux_mesh_colors(state.wave_field, state.COLOR_PALETTE)
        flux_mesh.render_flux_mesh(render.scene, state.wave_field, state.FLUX_MESH_OPTION)

    # TODO: remove test particles for visual reference
    # position1 = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    # render.scene.particles(position1, radius=0.01, color=colormap.COLOR_PARTICLE[1])
    # y_pos = 0.5 + (
    #     (round(constants.EWAVE_LENGTH / state.wave_field.dx)) / state.wave_field.max_grid_size
    # )
    # position2 = np.array([[0.5, y_pos, 0.5]], dtype=np.float32)
    # render.scene.particles(position2, radius=0.01, color=colormap.COLOR_ANTI[1])


# ================================================================
# MAIN LOOP
# ================================================================


def main():
    """Main entry point for xperiment launcher."""
    selected_xperiment_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Initialize Taichi
    ti.init(arch=ti.gpu, log_level=ti.WARN)  # Use GPU if available, suppress info logs

    # Initialize xperiment manager and state
    xperiment_mgr = XperimentManager()
    state = SimulationState()

    # Load xperiment from CLI argument or default
    default_xperiment = selected_xperiment_arg or "energy_wave"
    if default_xperiment not in xperiment_mgr.available_xperiments:
        print(f"Error: Xperiment '{default_xperiment}' not found!")
        return

    params = xperiment_mgr.load_xperiment(default_xperiment)
    if not params:
        return

    state.apply_xparameters(params)
    state.initialize_grid()
    state.compute_timestep()
    initialize_xperiment(state)

    # Initialize GGUI rendering
    render.init_UI(state.UNIVERSE_SIZE, state.TICK_SPACING, state.CAM_INIT)

    # Main rendering loop
    while render.window.running:
        render.init_scene(state.SHOW_AXIS)  # Initialize scene with lighting and camera

        # Handle ESC key for window close
        if render.window.is_pressed(ti.ui.ESCAPE):
            render.window.running = False
            break

        # Display UI overlays
        new_xperiment = display_xperiment_launcher(xperiment_mgr, state)
        display_controls(state)

        # Handle xperiment switching via process replacement
        if new_xperiment:
            print("\n================================================================")
            print("XPERIMENT LAUNCH")
            print(f"Now running: {new_xperiment}\n")

            sys.stdout.flush()
            sys.stderr.flush()
            render.window.running = False

            # os.execv replaces current process (macOS shows harmless warning, Cmd+Q broken)
            os.execv(sys.executable, [sys.executable, __file__, new_xperiment])

        if not state.PAUSED:
            # Run simulation step and update time
            state.compute_timestep()
            compute_wave_motion(state)
            state.elapsed_t += state.dt  # Elapsed time accumulation
            state.frame += 1

        # Render scene elements
        render_elements(state)

        # Display additional UI elements and scene
        display_color_menu(state)
        display_data_dashboard(state)
        display_level_specs(state, level_bar_vertices)
        render.show_scene()

        # Capture frame for video export (finalizes and stops at set VIDEO_FRAMES)
        if state.EXPORT_VIDEO:
            video.export_frame(state.frame, state.VIDEO_FRAMES)


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()
