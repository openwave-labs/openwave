"""
Xperiment Launcher

Unified launcher for wave-field xperiments featuring:
- UI-based xperiment selection and switching
- Single source of truth for rendering and UI code
- Xperiment-specific parameters in /xparameters directory
"""

import webbrowser
import importlib
import sys
import os
import time
from pathlib import Path

import taichi as ti
import numpy as np

from openwave.common import colormap, constants
from openwave._io import flux_mesh, render, video

import openwave.xperiments.b_scalar_field.spacetime_medium as medium
import openwave.xperiments.b_scalar_field.spacetime_ewave as ewave
import openwave.xperiments.b_scalar_field.instrumentation as instrument

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
        """Discover all available xperiment parameters in /xparameters directory."""
        parameters_dir = Path(__file__).parent / "xparameters"

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
            module_path = f"openwave.xperiments.b_scalar_field.xparameters.{xperiment_name}"
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
            module_path = f"openwave.xperiments.b_scalar_field.xparameters.{xperiment_name}"
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
        self.trackers = None
        self.c_amrs = 0.0
        self.dt_rs = 0.0
        self.cfl_factor = 0.0
        self.elapsed_t_rs = 0.0
        self.clock_start_time = time.time()
        self.frame = 1
        self.rms_ampL = constants.EWAVE_AMPLITUDE
        self.rms_ampT = 0.0
        self.avg_freq = constants.EWAVE_FREQUENCY
        self.avg_wavelength = constants.EWAVE_LENGTH
        self.total_energy = 0.0
        self.charge_level = 0.0
        self.charging = True
        self.damping = False

        # Current xperiment parameters
        self.X_NAME = ""
        self.CAM_INIT = [2.00, 1.50, 1.75]
        self.UNIVERSE_SIZE = []
        self.TARGET_VOXELS = 1e8
        self.STATIC_BOOST = 1.0

        # UI control variables
        self.SHOW_AXIS = False
        self.TICK_SPACING = 0.25
        self.SHOW_GRID = False
        self.SHOW_EDGES = False
        self.FLUX_MESH_PLANES = [0.5, 0.5, 0.5]
        self.SHOW_FLUX_MESH = 0
        self.WARP_MESH = False
        self.SIM_SPEED = 1.0
        self.PAUSED = False

        # Color control variables
        self.COLOR_THEME = "OCEAN"
        self.COLOR_PALETTE = 1  # Color palette index

        # Data Analytics & video export toggles
        self.INSTRUMENTATION = False
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

        # Charging
        charging = params["charging"]
        self.STATIC_BOOST = charging["STATIC_BOOST"]

        # UI defaults
        ui = params["ui_defaults"]
        self.SHOW_AXIS = ui["SHOW_AXIS"]
        self.TICK_SPACING = ui["TICK_SPACING"]
        self.SHOW_GRID = ui["SHOW_GRID"]
        self.SHOW_EDGES = ui["SHOW_EDGES"]
        self.FLUX_MESH_PLANES = ui["FLUX_MESH_PLANES"]
        self.SHOW_FLUX_MESH = ui["SHOW_FLUX_MESH"]
        self.WARP_MESH = ui["WARP_MESH"]
        self.SIM_SPEED = ui["SIM_SPEED"]
        self.PAUSED = ui["PAUSED"]

        # Color defaults
        color = params["color_defaults"]
        self.COLOR_THEME = color["COLOR_THEME"]
        self.COLOR_PALETTE = color["COLOR_PALETTE"]

        # Data Analytics & video export toggles
        diag = params["analytics"]
        self.INSTRUMENTATION = diag["INSTRUMENTATION"]
        self.EXPORT_VIDEO = diag["EXPORT_VIDEO"]
        self.VIDEO_FRAMES = diag["VIDEO_FRAMES"]

    def initialize_grid(self):
        """Initialize or reinitialize the wave-field grid."""
        self.wave_field = medium.WaveField(
            self.UNIVERSE_SIZE, self.TARGET_VOXELS, self.FLUX_MESH_PLANES
        )
        self.trackers = medium.Trackers(self.wave_field.grid_size, self.wave_field.scale_factor)

    def compute_timestep(self):
        """Compute timestep from CFL stability condition.

        CFL Condition: dt ≤ dx / (c × √3) for 3D wave equation.
        SIM_SPEED scales wave velocity for visualization control.
        """
        self.c_amrs = (
            constants.EWAVE_SPEED / constants.ATTOMETER * constants.RONTOSECOND * self.SIM_SPEED
        )  # am/rs
        self.dt_rs = self.wave_field.dx_am / (self.c_amrs / self.SIM_SPEED * (3**0.5))  # rs
        self.cfl_factor = round((self.c_amrs * self.dt_rs / self.wave_field.dx_am) ** 2, 7)

    def restart_sim(self):
        """Restart simulation state."""
        self.wave_field = None
        self.trackers = None
        self.c_amrs = 0.0
        self.dt_rs = 0.0
        self.cfl_factor = 0.0
        self.elapsed_t_rs = 0.0
        self.clock_start_time = time.time()
        self.frame = 1
        self.rms_ampL = constants.EWAVE_AMPLITUDE
        self.rms_ampT = 0.0
        self.avg_freq = constants.EWAVE_FREQUENCY
        self.avg_wavelength = constants.EWAVE_LENGTH
        self.total_energy = 0.0
        self.charge_level = 0.0
        self.charging = True
        self.damping = False
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

    with render.gui.sub_window("XPERIMENT LAUNCHER", 0.00, 0.00, 0.14, 0.33) as sub:
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
    with render.gui.sub_window("CONTROLS", 0.00, 0.34, 0.16, 0.25) as sub:
        state.SHOW_AXIS = sub.checkbox(f"Axis (ticks: {state.TICK_SPACING})", state.SHOW_AXIS)
        state.SHOW_EDGES = sub.checkbox("Sim Universe Edges", state.SHOW_EDGES)
        state.INSTRUMENTATION = sub.checkbox("Instrumentation", state.INSTRUMENTATION)
        state.SHOW_FLUX_MESH = sub.slider_int("Flux Mesh", state.SHOW_FLUX_MESH, 0, 3)
        state.WARP_MESH = sub.checkbox("Warp Mesh", state.WARP_MESH)
        state.SIM_SPEED = sub.slider_float("Speed", state.SIM_SPEED, 0.5, 1.0)
        if state.PAUSED:
            if sub.button(">> PROPAGATE EWAVE >>"):
                state.PAUSED = False
        else:
            if sub.button("Pause"):
                state.PAUSED = True
        if sub.button("Restart Simulation"):
            state.restart_sim()


def display_wave_menu(state):
    """Display wave properties selection menu."""
    with render.gui.sub_window("WAVE MENU", 0.00, 0.70, 0.15, 0.17) as sub:
        if sub.checkbox("Displacement (Longitudinal)", state.COLOR_PALETTE == 1):
            state.COLOR_PALETTE = 1
        if sub.checkbox("Displacement (Transverse)", state.COLOR_PALETTE == 2):
            state.COLOR_PALETTE = 2
        if sub.checkbox("Amplitude (Longitudinal)", state.COLOR_PALETTE == 4):
            state.COLOR_PALETTE = 4
        if sub.checkbox("Amplitude (Transverse)", state.COLOR_PALETTE == 5):
            state.COLOR_PALETTE = 5
        if sub.checkbox("Frequency (L&T)", state.COLOR_PALETTE == 6):
            state.COLOR_PALETTE = 6
        # Display gradient palette with 2× average range for headroom (allows peak visualization)
        if state.COLOR_PALETTE == 1:  # Display yellowgreen gradient palette
            render.canvas.triangles(yg_palette_vertices, per_vertex_color=yg_palette_colors)
            with render.gui.sub_window("displacement", 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(
                    f"{-state.rms_ampL*2/state.wave_field.scale_factor:.0e}  {state.rms_ampL*2/state.wave_field.scale_factor:.0e}m"
                )
        if state.COLOR_PALETTE == 2:  # Display redblue gradient palette
            render.canvas.triangles(rb_palette_vertices, per_vertex_color=rb_palette_colors)
            with render.gui.sub_window("displacement", 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(
                    f"{-state.rms_ampT*2/state.wave_field.scale_factor:.0e}  {state.rms_ampT*2/state.wave_field.scale_factor:.0e}m"
                )
        if state.COLOR_PALETTE == 4:  # Display viridis gradient palette
            render.canvas.triangles(vr_palette_vertices, per_vertex_color=vr_palette_colors)
            with render.gui.sub_window("amplitude", 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(f"0       {state.rms_ampL*2/state.wave_field.scale_factor:.0e}m")
        if state.COLOR_PALETTE == 5:  # Display ironbow gradient palette
            render.canvas.triangles(ib_palette_vertices, per_vertex_color=ib_palette_colors)
            with render.gui.sub_window("amplitude", 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(f"0       {state.rms_ampT*2/state.wave_field.scale_factor:.0e}m")
        if state.COLOR_PALETTE == 6:  # Display blueprint gradient palette
            render.canvas.triangles(bp_palette_vertices, per_vertex_color=bp_palette_colors)
            with render.gui.sub_window("frequency", 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(f"0       {state.avg_freq*2*state.wave_field.scale_factor:.0e}Hz")


def display_level_specs(state, level_bar_vertices):
    """Display OpenWave level specifications overlay."""
    render.canvas.triangles(level_bar_vertices, color=colormap.LIGHT_BLUE[1])
    with render.gui.sub_window("SCALAR-FIELD METHOD", 0.84, 0.01, 0.16, 0.12) as sub:
        sub.text("Coupling: Laplacian Operator")
        sub.text("Propagation: Wave Equation (PDE)")
        sub.text("Boundary: Dirichlet Condition")
        if sub.button("Wave Notation Guide"):
            webbrowser.open(
                "https://github.com/openwave-labs/openwave/blob/main/openwave/common/wave_notation.md"
            )


def display_data_dashboard(state):
    """Display simulation data dashboard."""
    clock_time = time.time() - state.clock_start_time
    sim_time_years = clock_time / (state.elapsed_t_rs * constants.RONTOSECOND or 1) / 31_536_000

    with render.gui.sub_window("DATA-DASHBOARD", 0.84, 0.37, 0.16, 0.63) as sub:
        sub.text("--- SPACETIME ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Medium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")
        sub.text(f"eWAVE Speed (c): {constants.EWAVE_SPEED:.1e} m/s")

        sub.text("\n--- SIMULATION DOMAIN ---", color=colormap.LIGHT_BLUE[1])
        sub.text(
            f"Universe: {state.wave_field.max_universe_edge:.1e} m ({state.wave_field.max_universe_edge_lambda:.0f} waves)"
        )
        sub.text(f"Voxel Count: {state.wave_field.voxel_count:,}")
        sub.text(
            f"Grid Size: {state.wave_field.nx} x {state.wave_field.ny} x {state.wave_field.nz}"
        )
        sub.text(f"Voxel Edge: {state.wave_field.dx:.2e} m")

        sub.text("\n--- RESOLUTION (scaled-up) ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Scale-up Factor: {state.wave_field.scale_factor:.1f}x")
        sub.text(f"eWave: {state.wave_field.ewave_res:.1f} voxels/wave (~12)")
        if state.wave_field.ewave_res < 10:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))

        sub.text("\n--- ENERGY-WAVE ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Amp Longitudinal: {state.rms_ampL/state.wave_field.scale_factor:.1e} m")
        sub.text(f"Amp Transverse: {state.rms_ampT/state.wave_field.scale_factor:.1e} m")
        sub.text(f"Frequency: {state.avg_freq*state.wave_field.scale_factor:.1e} Hz")
        sub.text(f"Wavelength: {state.avg_wavelength/state.wave_field.scale_factor:.1e} m")
        sub.text(
            f"TOTAL ENERGY: {state.total_energy:.1e} J",
            color=(
                colormap.ORANGE[1]
                if state.charging
                else colormap.LIGHT_BLUE[1] if state.damping else colormap.GREEN[1]
            ),
        )
        sub.text(
            f"Charge Level: {state.charge_level:.0%} {"...CHARGING..." if state.charging else "...DAMPING..." if state.damping else "(target)"}",
            color=(
                colormap.ORANGE[1]
                if state.charging
                else colormap.LIGHT_BLUE[1] if state.damping else colormap.GREEN[1]
            ),
        )

        sub.text("\n--- TIME MICROSCOPE ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Timesteps (frames): {state.frame}")
        sub.text(f"Simulation Time: {state.elapsed_t_rs:.2e} rs")
        sub.text(f"Clock Time: {clock_time:.2f} s")
        sub.text(f"(1s sim time takes {sim_time_years:.0e}y)")

        sub.text("\n--- TIMESTEP ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"c_amrs: {state.c_amrs:.3f} am/rs")
        sub.text(
            f"dt_rs: {state.dt_rs:.3f} rs",
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
    """Initialize color palettes, wave charger and instrumentation.

    Args:
        state: SimulationState instance with xperiment parameters
    """
    global yg_palette_vertices, yg_palette_colors
    global rb_palette_vertices, rb_palette_colors
    global vr_palette_vertices, vr_palette_colors
    global ib_palette_vertices, ib_palette_colors
    global bp_palette_vertices, bp_palette_colors
    global level_bar_vertices

    # Initialize color palette scales for gradient rendering and level indicator
    yg_palette_vertices, yg_palette_colors = colormap.get_palette_scale(
        colormap.yellowgreen, 0.00, 0.63, 0.079, 0.01
    )
    rb_palette_vertices, rb_palette_colors = colormap.get_palette_scale(
        colormap.redblue, 0.00, 0.63, 0.079, 0.01
    )
    vr_palette_vertices, vr_palette_colors = colormap.get_palette_scale(
        colormap.viridis, 0.00, 0.63, 0.079, 0.01
    )
    ib_palette_vertices, ib_palette_colors = colormap.get_palette_scale(
        colormap.ironbow, 0.00, 0.63, 0.079, 0.01
    )
    bp_palette_vertices, bp_palette_colors = colormap.get_palette_scale(
        colormap.blueprint, 0.00, 0.63, 0.079, 0.01
    )
    level_bar_vertices = colormap.get_level_bar_geometry(0.84, 0.00, 0.159, 0.01)

    # STATIC CHARGING methods (single radial pulse pattern, fast charge) ==================================
    # Provides initial energy source, dynamic chargers do maintenance around 100%
    ewave.charge_full(state.wave_field, state.dt_rs, state.STATIC_BOOST)
    ewave.charge_gaussian(state.wave_field)
    # NOTE: (too-light) ewave.charge_falloff(state.wave_field, state.dt_rs)
    # NOTE: (too-light) ewave.charge_1lambda(state.wave_field, state.dt_rs)

    if state.INSTRUMENTATION:
        print("\n" + "=" * 64)
        print("INSTRUMENTATION ENABLED")
        print("=" * 64)
        instrument.plot_static_charge_profile(state.wave_field)


def compute_wave_motion(state):
    """Compute wave propagation, reflection, superposition and update tracker averages.
    The static pulse creates a natural equilibrium via Dirichlet BC reflections.
    """

    # DYNAMIC MAINTENANCE CHARGING ==================================
    # Soft landing: ramp up to target, then maintain with minimal intervention
    # Static pulse provides ~90% energy, dynamic chargers do the final 10%
    # envelope = ewave.compute_charge_envelope(state.charge_level)
    # if envelope > 0.001:  # Small threshold to avoid zero-amplitude calls
    #     # Wall charging - isotropic energy injection from all 6 boundaries
    #     spacing = max(int(state.wave_field.ewave_res // 2), 1)
    #     sources = max(state.wave_field.min_grid_size // spacing, 10)
    #     effective_boost = state.DYNAMIC_BOOST * envelope
    #     ewave.charge_oscillator_wall(
    #         state.wave_field, state.elapsed_t_rs, sources, effective_boost
    #     )
    # state.charging = envelope > 0.001  # Track charging state for UI display

    # DYNAMIC MAINTENANCE CHARGING (oscillator during simulation) =============================
    # Runs BEFORE propagation to inject energy to maintain stabilization
    # if state.charging and state.frame > 2000:  # hold off initial transient
    #     # ewave.charge_oscillator_sphere(state.wave_field, state.elapsed_t_rs, 0.10, 3.0)
    #     # NOTE: (too-light) ewave.charge_oscillator_falloff(state.wave_field, state.elapsed_t_rs)
    #     ewave.charge_oscillator_wall(state.wave_field, state.elapsed_t_rs, 6, 10)

    # WAVE PROPAGATION =======================================
    ewave.propagate_wave(
        state.wave_field,
        state.trackers,
        state.c_amrs,
        state.dt_rs,
        state.elapsed_t_rs,
        state.SIM_SPEED,
    )

    # PROPORTIONAL DAMPING ==================================
    # Only damp above target to catch overshoots, preserve natural equilibrium below
    # damping_factor = ewave.compute_damping_factor(state.charge_level)
    # if damping_factor < 1.0:  # Only apply if damping is active
    #     ewave.damp_full(state.wave_field, damping_factor)
    # state.damping = damping_factor < 0.9999  # Track damping state for UI display

    # DYNAMIC DAMPING methods (energy sink during simulation) =============================
    # Runs AFTER propagation to reduce energy to maintain stabilization
    # if state.damping:
    #     ewave.damp_full(state.wave_field, 0.9999)
    #     # NOTE: (too-light) ewave.damp_sphere(state.wave_field, 0.99)

    # IN-FRAME DATA SAMPLING & ANALYTICS ==================================
    # Frame skip reduces GPU->CPU transfer overhead
    if state.frame % 60 == 0:
        ewave.sample_avg_trackers(state.wave_field, state.trackers)
    state.rms_ampL = state.trackers.rms_ampL_am[None] * constants.ATTOMETER  # in m
    state.rms_ampT = state.trackers.rms_ampT_am[None] * constants.ATTOMETER  # in m
    state.avg_freq = state.trackers.avg_freq_rHz[None] / constants.RONTOSECOND
    state.avg_wavelength = constants.EWAVE_SPEED / (state.avg_freq or 1)  # prevents 0 div
    state.total_energy = (
        constants.MEDIUM_DENSITY
        * state.wave_field.universe_volume
        * state.avg_freq**2
        * (state.rms_ampL**2 + state.rms_ampT**2)
    )
    state.charge_level = state.total_energy / state.wave_field.nominal_energy
    state.charging = state.charge_level < 0.80  # stop charging, seeks energy stabilization
    state.damping = state.charge_level > 1.20  # start damping, seeks energy stabilization

    if state.INSTRUMENTATION:
        instrument.log_timestep_data(
            state.frame, state.charge_level, state.wave_field, state.trackers
        )
        if state.frame == 500:
            instrument.plot_probe_wave_profile(state.wave_field)


def render_elements(state):
    """Render grid, edges, flux mesh and test particles."""
    if state.SHOW_GRID:
        render.scene.lines(state.wave_field.grid_lines, width=1, color=colormap.COLOR_MEDIUM[1])

    if state.SHOW_EDGES:
        render.scene.lines(state.wave_field.edge_lines, width=1, color=colormap.COLOR_MEDIUM[1])

    if state.SHOW_FLUX_MESH > 0:
        ewave.update_flux_mesh_values(
            state.wave_field,
            state.trackers,
            state.COLOR_PALETTE,
            state.WARP_MESH,
        )
        flux_mesh.render_flux_mesh(render.scene, state.wave_field, state.SHOW_FLUX_MESH)


# ================================================================
# MAIN LOOP
# ================================================================


def main():
    """Main entry point for xperiment launcher."""
    selected_xperiment_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Initialize Taichi
    ti.init(arch=ti.gpu, log_level=ti.WARN)  # GPU preferred, suppress info logs

    # Initialize xperiment manager and state
    xperiment_mgr = XperimentManager()
    state = SimulationState()

    # Load xperiment from CLI argument or default
    default_xperiment = selected_xperiment_arg or "the_king"
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

            # os.execv replaces current process (macOS may show harmless warning)
            os.execv(sys.executable, [sys.executable, __file__, new_xperiment])

        if not state.PAUSED:
            # Run simulation step and update time
            state.compute_timestep()
            compute_wave_motion(state)
            state.elapsed_t_rs += state.dt_rs  # Accumulate simulation time
            state.frame += 1

        # Render scene elements
        render_elements(state)

        # Display additional UI elements and scene
        display_wave_menu(state)
        display_data_dashboard(state)
        display_level_specs(state, level_bar_vertices)
        render.show_scene()

        # Capture frame for video export (stops after VIDEO_FRAMES)
        if state.EXPORT_VIDEO:
            video.export_frame(state.frame, state.VIDEO_FRAMES)

    if state.INSTRUMENTATION:
        instrument.generate_plots()


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()
