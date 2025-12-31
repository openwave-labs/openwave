"""
XPERIMENT INSTRUMENTATION (data collection)

This provides zero-overhead data collection that can be toggled on/off per xperiment.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from openwave.common import colormap, constants


# ================================================================
# Module-level Parameters
# ================================================================

base_amplitude = constants.EWAVE_AMPLITUDE  # in meters
base_amplitude_am = constants.EWAVE_AMPLITUDE / constants.ATTOMETER  # in attometers
base_wavelength = constants.EWAVE_LENGTH  # in meters
base_wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETER  # in attometers
base_frequency = constants.EWAVE_FREQUENCY  # in Hz
base_frequency_rHz = constants.EWAVE_FREQUENCY * constants.RONTOSECOND  # in rHz (1/rontosecond)

# Compute angular frequency (ω = 2πf) for temporal phase variation
omega_rs = 2.0 * np.pi * base_frequency_rHz  # angular frequency (rad/rs)

# Compute angular wave number (k = 2π/λ) for spatial phase variation
k_am = 2.0 * np.pi / base_wavelength_am  # radians / am

_MODULE_DIR = Path(__file__).parent
PLOT_DIR = _MODULE_DIR / "_plots"

# Plot mode toggles
START_PAUSED = False  # True = start paused, False = start animating
START_FRAME = 0  # starting frame (0 to ANIMATION_FRAMES-1), useful when paused
SAVE_GIF = False  # True = save animation as GIF

# Animation parameters
ANIMATION_FRAMES = 100  # frames per cycle
ANIMATION_INTERVAL = 50  # ms between frames (20 FPS)
PERIOD_RS = 2 * np.pi / omega_rs  # one full wave period in rontoseconds

# ================================================================
# Displacement Functions
# ================================================================

wc_spacing = 4 * base_wavelength_am  # am, spacing between wave centers

x_am = np.linspace(-1.5 * wc_spacing, +1.5 * wc_spacing, 1000)

wc1_x_am = -wc_spacing / 2  # am, center of wave source 1
wc2_x_am = +wc_spacing / 2  # am, center of wave source 2
wc1_phase_rad = -np.pi / 2  # radians, phase offset for source 1
wc2_phase_rad = np.pi / 2  # radians, phase offset for source 2


def compute_radius_am(x_am: np.ndarray = x_am, source: int = 1) -> np.ndarray:
    """Compute modular radial distance from wave source center.

    Args:
        x_am: X coordinates in attometers
        source: Wave source (1 = wc1, 2 = wc2)
    """
    if source == 1:
        radius_am = np.abs(x_am - wc1_x_am)
    elif source == 2:
        radius_am = np.abs(x_am - wc2_x_am)
    else:
        raise ValueError(f"Invalid source: {source}. Use 1 or 2.")
    return radius_am


def get_source_offset(source: int = 1) -> float:
    """Get phase offset for a wave source."""
    if source == 1:
        return wc1_phase_rad
    elif source == 2:
        return wc2_phase_rad
    else:
        raise ValueError(f"Invalid source: {source}. Use 1 or 2.")


def compute_wave_flat(radius_am: np.ndarray, t_rs: float = 0.0) -> np.ndarray:
    """Fundamental Flat Wave - uniform oscillation throughout space (in attometers).

    ψ(t) = A·cos(ωt) - Spatially uniform pure temporal harmonic oscillation.

    Models a background oscillation, the fundamental energy source of spacetime.
    No spatial dependence - every point in space oscillates in phase with
    constant amplitude. Represents the superposition of energy waves reflected
    from all matter in the universe, creating a uniform flat wave field.

    Args:
        radius_am: Radial distance array (used only for output shape, not computation)
        t_rs: Time in rontoseconds (default 0.0 for static plot)
    """
    psi_am = base_amplitude_am * np.cos(omega_rs * t_rs)  # temporal phase only (rad)
    return np.full_like(radius_am, psi_am)  # uniform value at every spatial point


def compute_wave_standing(radius_am: np.ndarray, t_rs: float = 0.0) -> np.ndarray:
    """Fundamental Standing Wave - uniform oscillation throughout space (in attometers).

    ψ(t) = A·cos(ωt)cos(kr) - Spatially uniform pure temporal harmonic oscillation.

    Models a background oscillation, the fundamental energy source of spacetime.
    No spatial dependence - every point in space oscillates in phase with
    constant amplitude. Represents the superposition of energy waves reflected
    from all matter in the universe, creating a uniform wave field.

    Args:
        radius_am: Radial distance array (used only for output shape, not computation)
        t_rs: Time in rontoseconds (default 0.0 for static plot)
    """
    psi_am = (
        base_amplitude_am
        * np.cos(omega_rs * t_rs)  # temporal phases (rad)
        * np.sin(k_am * x_am)  # spatial phase, φ = k·r (rad)
    )
    return psi_am


def compute_wave_fullamp(
    radius_am: np.ndarray, t_rs: float = 0.0, direction: int = 1, source: int = 1
) -> np.ndarray:
    """Wave Displacement with full amplitude (in attometers).

    ψ(x,t) = A·cos(ωt ± kr + φ) - Spherical wave.

    Args:
        radius_am: Radial distance from wave source (attometers)
        t_rs: Time in rontoseconds (default 0.0 for static plot)
        direction: +1 for incoming wave, -1 for outgoing wave
        source: Wave source (1 = wc1, 2 = wc2) for phase offset
    """
    source_offset = get_source_offset(source)
    psi_am = base_amplitude_am * np.cos(  # full amplitude (am)  # oscillator cosine function
        omega_rs * t_rs  # temporal phase (rad)
        + direction * k_am * radius_am  # spatial phase, spherical wave fronts, φ = ±kr (rad)
        + source_offset  # direction only affects the spatial phase, not the source offset, (rad)
    )
    return psi_am


def compute_wave_ampfalloff(
    radius_am: np.ndarray, t_rs: float = 0.0, direction: int = 1, source: int = 1
) -> np.ndarray:
    """Wave Displacement with amplitude falloff (in attometers).

    ψ(x,t) = A(r)·cos(ωt ± kr + φ) - Spherical wave.

    Args:
        radius_am: Radial distance from wave source (attometers)
        t_rs: Time in rontoseconds (default 0.0 for static plot)
        direction: +1 for incoming wave, -1 for outgoing wave
        source: Wave source (1 = wc1, 2 = wc2) for phase offset
    """
    source_offset = get_source_offset(source)
    # λ as reference radius → amplitude = A₀ at r = λ
    # Clamp r to avoid singularity at r = 0 (min r = λ / 2π)
    r_safe_am = np.maximum(radius_am, base_wavelength_am / (2 * np.pi))
    amplitude_at_r_am = base_amplitude_am * base_wavelength_am / r_safe_am
    psi_am = (
        amplitude_at_r_am  # amplitude falloff for spherical wave: A(r) = A₀ · (λ / r) (am)
        * np.cos(  # oscillator cosine function
            omega_rs * t_rs  # temporal phase (rad)
            + direction * k_am * radius_am  # spatial phase, spherical wave fronts, φ = ±kr (rad)
            + source_offset  # direction only affects the spatial phase, not the source offset, (rad)
        )
    )
    return psi_am


def compute_wave_Wolff(radius_am: np.ndarray, t_rs: float = 0.0, source: int = 1) -> np.ndarray:
    """Milo Wolff Standing Longitudinal Wave Displacement (in attometers).

    ψ(r,t) = A·cos(ωt + φ₀)·sin(kr)/r - Standing wave around wave center.

    Standing wave converging around wave center from all directions.

    Args:
        radius_am: Radial distance from wave source (attometers)
        t_rs: Time in rontoseconds (default 0.0 for static plot)
        source: Wave source (1 = wc1, 2 = wc2) for phase offset
    """
    source_offset = get_source_offset(source)
    psi_am = (
        base_amplitude_am  # full amplitude (am)
        * np.cos(omega_rs * t_rs + source_offset)  # temporal phase with source offset (rad)
        * np.sin(k_am * radius_am)  # spatial phase, φ = k·r (rad)
        / radius_am  # amplitude falloff for spherical wave: A(r) = A₀/r (am)
        * base_wavelength_am  # normalize by wavelength to maintain units (am)
    )
    return psi_am


def compute_wave_LaFreniere(
    radius_am: np.ndarray, t_rs: float = 0.0, source: int = 1
) -> np.ndarray:
    """Gabriel LaFreniere Standing Longitudinal Wave Displacement (in attometers).

    ψ(r,t) = A·[sin(ωt + φ₀ - kr) - sin(ωt + φ₀)] / (kr) - Standing wave.

    Standing wave converging around wave center from all directions.

    Args:
        radius_am: Radial distance from wave source (attometers)
        t_rs: Time in rontoseconds (default 0.0 for static plot)
        source: Wave source (1 = wc1, 2 = wc2) for phase offset
    """
    source_offset = get_source_offset(source)
    kr = k_am * radius_am
    # When kr < π, apply smooth transition to avoid singularity at r=0
    kr = np.where(
        kr < np.pi,
        kr + (np.pi / 2) * (1 - kr / np.pi) ** 2,
        kr,
    )
    psi_am = (
        base_amplitude_am  # full amplitude (am)
        * (
            np.sin(omega_rs * t_rs - kr + source_offset) - np.sin(omega_rs * t_rs + source_offset)
        )  # both terms share source offset (same oscillator)
        / kr
    )
    return psi_am


def compute_wave_LaFreniere2(
    radius_am: np.ndarray, t_rs: float = 0.0, source: int = 1
) -> np.ndarray:
    """Gabriel LaFreniere Standing Longitudinal Wave Displacement (in attometers).

    ψ(r,t) = A·sin(-ωt + kr + φ₀) / kr - Standing wave.

    Standing wave converging around wave center from all directions.

    When kr < π, a smooth transition is applied to avoid the singularity at r=0:
        kr' = kr + (π/2)·(1 - kr/π)²

    Args:
        radius_am: Radial distance from wave source (attometers)
        t_rs: Time in rontoseconds (default 0.0 for static plot)
        source: Wave source (1 = wc1, 2 = wc2) for phase offset
    """
    source_offset = get_source_offset(source)
    kr = k_am * radius_am
    # When kr < π, apply smooth transition to avoid singularity at r=0
    kr = np.where(
        kr < np.pi,
        kr + (np.pi / 2) * (1 - kr / np.pi) ** 2,
        kr,
    )
    psi_am = (
        base_amplitude_am  # full amplitude (am)
        * (
            np.sin(-omega_rs * t_rs + kr + source_offset)
        )  # both terms share source offset (same oscillator)
        / kr
    )
    return psi_am


def compute_wave_LF(
    radius_am: np.ndarray, t_rs: float = 0.0, direction: int = 1, source: int = 1
) -> np.ndarray:
    """Gabriel LaFreniere Standing Longitudinal Wave Displacement (in attometers).

    ψ(r,t) = A·sin(-ωt + kr + φ₀) / kr - Standing wave.

    Standing wave converging around wave center from all directions.

    When kr < π, a smooth transition is applied to avoid the singularity at r=0:
        kr' = kr + (π/2)·(1 - kr/π)²

    Args:
        radius_am: Radial distance from wave source (attometers)
        t_rs: Time in rontoseconds (default 0.0 for static plot)
        source: Wave source (1 = wc1, 2 = wc2) for phase offset
    """
    source_offset = get_source_offset(source)
    ot = omega_rs * t_rs
    kr = k_am * radius_am
    # When kr < π, apply smooth transition to avoid singularity at r=0
    kr = np.where(
        kr < np.pi,
        kr + (np.pi / 2) * (1 - kr / np.pi) ** 2,
        kr,
    )
    psi_am = (
        base_amplitude_am  # full amplitude (am)
        * (
            np.cos(ot + source_offset) * np.sin(kr)
            + direction * np.sin(ot + source_offset) * (1 - np.cos(kr))
        )  # both terms share source offset (same oscillator)
        / kr
    )
    return psi_am


# ================================================================
# Plot Configuration
# ================================================================

# Each plot config: func, direction, source, ylim, height_ratio, label
# - func: string or list of strings (for sum)
# - direction: +1 incoming, -1 outgoing (or list for each func)
# - source: 1 = wc1, 2 = wc2 (or list for each func)
PLOT_CONFIGS0 = [  # 1 WC: flat + ampfalloff
    {
        "func": "flat",  # sum of both sources
        "direction": 1,
        "source": 1,  # WC1
        "ylim": (-1.5, 1.5),
        "height_ratio": 1,
        "label": "INCOMING Psi (am)",
    },
    {
        "func": "ampfalloff",
        "direction": -1,
        "source": 1,  # WC1
        "ylim": (-1.5, 6.5),
        "height_ratio": 2,
        "label": "OUTGOING Psi (am)",
    },
    {
        "func": ["flat", "ampfalloff"],  # sum of both sources
        "direction": [1, -1],
        "source": [1, 1],  # WC1 + WC1
        "ylim": (-3.5, 7.5),
        "height_ratio": 2,
        "label": "TOTAL Psi (am)",
    },
]

PLOT_CONFIGS0 = [  # 2 WC: flat + ampfalloff
    {
        "func": ["flat", "flat"],  # sum of both sources
        "direction": [1, 1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 1.5),
        "height_ratio": 1,
        "label": "INCOMING Psi (am)",
    },
    {
        "func": ["ampfalloff", "ampfalloff"],  # sum of both sources
        "direction": [-1, -1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 6.5),
        "height_ratio": 2,
        "label": "OUTGOING Psi (am)",
    },
    {
        "func": ["flat", "ampfalloff", "flat", "ampfalloff"],  # sum of both sources
        "direction": [1, -1, 1, -1],
        "source": [1, 1, 2, 2],  # WC1 + WC2
        "ylim": (-3.5, 7.5),
        "height_ratio": 2,
        "label": "TOTAL Psi (am)",
    },
]

PLOT_CONFIGS0 = [  # 2 WC: standing + ampfalloff
    {
        "func": ["standing", "standing"],  # sum of both sources
        "direction": [1, 1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 1.5),
        "height_ratio": 1,
        "label": "INCOMING Psi (am)",
    },
    {
        "func": ["ampfalloff", "ampfalloff"],  # sum of both sources
        "direction": [-1, -1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 6.5),
        "height_ratio": 2,
        "label": "OUTGOING Psi (am)",
    },
    {
        "func": ["standing", "ampfalloff", "standing", "ampfalloff"],  # sum of both sources
        "direction": [1, -1, 1, -1],
        "source": [1, 1, 2, 2],  # WC1 + WC2
        "ylim": (-3.5, 7.5),
        "height_ratio": 2,
        "label": "TOTAL Psi (am)",
    },
]


PLOT_CONFIGS0 = [  # 1 WC: fullamp + ampfalloff
    {
        "func": ["fullamp"],  # sum of both sources
        "direction": [1],
        "source": [1],  # WC1
        "ylim": (-1.5, 1.5),
        "height_ratio": 1,
        "label": "INCOMING Psi (am)",
    },
    {
        "func": ["ampfalloff"],  # sum of both sources
        "direction": [-1],
        "source": [1],  # WC1
        "ylim": (-1.5, 6.5),
        "height_ratio": 2,
        "label": "OUTGOING Psi (am)",
    },
    {
        "func": ["fullamp", "ampfalloff"],  # sum of both sources
        "direction": [1, -1],
        "source": [1, 1],  # WC1
        "ylim": (-3.5, 7.5),
        "height_ratio": 2,
        "label": "TOTAL Psi (am)",
    },
]

PLOT_CONFIG0 = [  # 2 WC: fullamp + ampfalloff
    {
        "func": ["fullamp", "fullamp"],  # sum of both sources
        "direction": [1, 1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 1.5),
        "height_ratio": 1,
        "label": "INCOMING Psi (am)",
    },
    {
        "func": ["ampfalloff", "ampfalloff"],  # sum of both sources
        "direction": [-1, -1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 6.5),
        "height_ratio": 2,
        "label": "OUTGOING Psi (am)",
    },
    {
        "func": ["fullamp", "ampfalloff", "fullamp", "ampfalloff"],  # sum of both sources
        "direction": [1, -1, 1, -1],
        "source": [1, 1, 2, 2],  # WC1 + WC2
        "ylim": (-3.5, 7.5),
        "height_ratio": 2,
        "label": "TOTAL Psi (am)",
    },
]

PLOT_CONFIGS0 = [  # 1 WC: wolff & lafreniere
    {
        "func": ["wolff"],  # sum of both sources
        "source": [1],  # WC1
        "ylim": (-1.5, 6.5),
        "height_ratio": 1,
        "label": "WOLFF STANDING WAVE",
    },
    {
        "func": ["lafreniere"],  # sum of both sources
        "source": [1],  # WC1
        "ylim": (-0.5, 1.5),
        "height_ratio": 1,
        "label": "LAFRENIERE STANDING WAVE",
    },
]

PLOT_CONFIGS0 = [  # 2 WC: wolff & lafreniere
    {
        "func": ["wolff", "wolff"],  # sum of both sources
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 6.5),
        "height_ratio": 1,
        "label": "WOLFF STANDING WAVE",
    },
    {
        "func": ["lafreniere", "lafreniere"],  # sum of both sources
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-0.5, 1.5),
        "height_ratio": 1,
        "label": "LAFRENIERE STANDING WAVE",
    },
]

PLOT_CONFIGS0 = [  # 1 WC: LFa + LFb
    {
        "func": "LF",  # sum of both sources
        "direction": 1,
        "source": 1,  # WC1
        "ylim": (-1.0, 1.0),
        "height_ratio": 1,
        "label": "INCOMING Psi (am)",
    },
    {
        "func": "LF",
        "direction": -1,
        "source": 1,  # WC1
        "ylim": (-1.0, 1.0),
        "height_ratio": 1,
        "label": "OUTGOING Psi (am)",
    },
    {
        "func": ["LF", "LF"],  # sum of both sources
        "direction": [1, -1],
        "source": [1, 1],  # WC1 + WC1
        "ylim": (-1.5, 1.5),
        "height_ratio": 1,
        "label": "TOTAL Psi (am)",
    },
]

PLOT_CONFIGS = [  # 2 WC: LFa + LFb
    {
        "func": ["LF", "LF"],  # sum of both sources
        "direction": [1, 1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.0, 1.0),
        "height_ratio": 1,
        "label": "INCOMING Psi (am)",
    },
    {
        "func": ["LF", "LF"],
        "direction": [-1, -1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.0, 1.0),
        "height_ratio": 1,
        "label": "OUTGOING Psi (am)",
    },
    {
        "func": ["LF", "LF", "LF", "LF"],  # sum of both sources
        "direction": [1, -1, 1, -1],
        "source": [1, 1, 2, 2],  # WC1 + WC1 + WC2 + WC2
        "ylim": (-1.5, 1.5),
        "height_ratio": 1,
        "label": "TOTAL Psi (am)",
    },
]
# Function registry - maps string names to actual functions
WAVE_FUNCTIONS = {
    "flat": compute_wave_flat,
    "standing": compute_wave_standing,
    "fullamp": compute_wave_fullamp,
    "ampfalloff": compute_wave_ampfalloff,
    "wolff": compute_wave_Wolff,
    "lafreniere": compute_wave_LaFreniere,
    "LF": compute_wave_LF,
}


def compute_plot_value(config: dict, t_rs: float) -> np.ndarray:
    """Compute wave value for a plot config (handles single func or sum of func).

    Args:
        config: Plot configuration dict with func, direction, source
        t_rs: Time in rontoseconds
    """
    func_spec = config["func"]
    direction_spec = config.get("direction", 1)
    source_spec = config.get("source", 1)

    if isinstance(func_spec, list):
        # Sum of multiple functions
        directions = (
            direction_spec
            if isinstance(direction_spec, list)
            else [direction_spec] * len(func_spec)
        )
        sources = source_spec if isinstance(source_spec, list) else [source_spec] * len(func_spec)
        result = np.zeros_like(x_am)
        for func_name, direction, source in zip(func_spec, directions, sources):
            func = WAVE_FUNCTIONS[func_name]
            radius_am = compute_radius_am(x_am, source=source)
            # Check if function accepts direction/source parameters
            if func_name in ("fullamp", "ampfalloff", "LF"):
                result += func(radius_am, t_rs, direction=direction, source=source)
            elif func_name in ("wolff", "lafreniere"):
                result += func(radius_am, t_rs, source=source)
            else:
                result += func(radius_am, t_rs)
        return result
    else:
        # Single function
        source = source_spec if not isinstance(source_spec, list) else source_spec[0]
        radius_am = compute_radius_am(x_am, source=source)
        func = WAVE_FUNCTIONS[func_spec]
        if func_spec in ("fullamp", "ampfalloff", "LF"):
            return func(radius_am, t_rs, direction=direction_spec, source=source)
        elif func_spec in ("wolff", "lafreniere"):
            return func(radius_am, t_rs, source=source)
        else:
            return func(radius_am, t_rs)


# ================================================================
# Plot Functions
# ================================================================


def plot_waves(start_paused: bool = False, start_frame: int = 0, save_gif: bool = False):
    """Animate the wave displacement over time with pause/resume support.

    Uses PLOT_CONFIGS to determine which wave functions to plot.
    Each plot shows the wave displacement and RMS amplitude (√⟨ψ²⟩ per x).

    RMS Tracking:
        Per-x RMS amplitude tracked via EMA on ψ² (squared displacement).
        RMS = √⟨ψ²⟩ represents the energy-equivalent amplitude because Energy ∝ ψ².
        Used for: energy calculation, force gradients, visualization scaling.

    Controls:
        SPACE: Pause/Resume animation
        LEFT/RIGHT arrows: Step frame when paused
        R: Reset RMS trackers to zero

    Args:
        start_paused: If True, start with animation paused
        start_frame: Initial frame to display (0 to ANIMATION_FRAMES-1)
        save_gif: If True, save animation as GIF file
    """
    num_plots = len(PLOT_CONFIGS)

    # Animation state (mutable container for nested function access)
    state = {"paused": start_paused, "frame": start_frame % ANIMATION_FRAMES}

    # Extract config values
    height_ratios = [cfg["height_ratio"] for cfg in PLOT_CONFIGS]

    # Detect which wave sources are used in PLOT_CONFIGS
    active_sources = set()
    for cfg in PLOT_CONFIGS:
        src = cfg.get("source", 1)
        if isinstance(src, list):
            active_sources.update(src)
        else:
            active_sources.add(src)
    use_wc1 = 1 in active_sources
    use_wc2 = 2 in active_sources

    # Create subplots based on config
    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        num_plots,
        1,
        figsize=(16, 9),
        facecolor=colormap.DARK_GRAY[1],
        gridspec_kw={"height_ratios": height_ratios},
    )
    # Handle single plot case (axes not a list)
    if num_plots == 1:
        axes = [axes]
    fig.suptitle("OPENWAVE Analytics", fontsize=20, family="Monospace")

    # Initialize line objects for each plot (wave + RMS amplitude)
    lines = []
    rms_lines = []  # per-x RMS amplitude lines
    rms_texts = []  # 3 RMS text objects per plot: [left, center, right]
    gradient_texts = []  # 2 gradient direction indicators per plot: [left_grad, right_grad]

    # Per-x RMS amplitude tracking via cycle averaging
    # Accumulate ψ² over one full cycle, then compute fixed average
    # After first cycle completes, RMS values become constant (periodic wave)
    rms_x_am2_sum = [np.zeros_like(x_am) for _ in PLOT_CONFIGS]  # Σψ² per x
    rms_x_am2_avg = [None for _ in PLOT_CONFIGS]  # final ⟨ψ²⟩ after cycle (None until complete)
    cycle_complete = [False for _ in PLOT_CONFIGS]  # track if cycle completed per plot

    # Section masks for regional RMS calculation
    # Masks depend on which wave centers are active
    if use_wc1 and use_wc2:
        # Both WCs: Left of WC1, between WCs, right of WC2
        mask_left = x_am < wc1_x_am
        mask_center = (x_am >= wc1_x_am) & (x_am <= wc2_x_am)
        mask_right = x_am > wc2_x_am
    elif use_wc1:
        # Only WC1: Left of WC1, at WC1 (±λ), right of WC1
        mask_left = x_am < wc1_x_am
        mask_center = np.zeros_like(x_am, dtype=bool)  # no center region
        mask_right = x_am > wc1_x_am
    elif use_wc2:
        # Only WC2: Left of WC2, at WC2 (±λ), right of WC2
        mask_left = x_am < wc2_x_am
        mask_center = np.zeros_like(x_am, dtype=bool)  # no center region
        mask_right = x_am > wc2_x_am
    else:
        # No WCs (shouldn't happen)
        mask_left = np.ones_like(x_am, dtype=bool)
        mask_center = np.zeros_like(x_am, dtype=bool)
        mask_right = np.zeros_like(x_am, dtype=bool)

    for ax, cfg in zip(axes, PLOT_CONFIGS):
        # Wave displacement line
        (line,) = ax.plot(
            [],
            [],
            color=colormap.viridis_palette[2][1],
            linewidth=2,
            label=cfg["label"],
        )
        lines.append(line)

        # Per-x RMS amplitude line (√⟨ψ²⟩)
        (rms_line,) = ax.plot(
            [],
            [],
            color=colormap.viridis_palette[4][1],
            linewidth=1.5,
            linestyle="-",
            alpha=0.8,
            label="RMS (√⟨ψ²⟩)",
        )
        rms_lines.append(rms_line)

    # Time display text (placed on first axes)
    time_text = axes[0].text(
        0.02,
        0.95,
        "",
        transform=axes[0].transAxes,
        fontsize=10,
        family="Monospace",
        verticalalignment="top",
    )

    # Configure all axes with static elements from config
    for ax, cfg in zip(axes, PLOT_CONFIGS):
        ylim_min, ylim_max = cfg["ylim"]
        ax.set_xlim(x_am.min(), x_am.max())
        ax.set_ylim(ylim_min * base_amplitude_am, ylim_max * base_amplitude_am)
        ax.axhline(
            y=base_amplitude_am,
            color=colormap.viridis_palette[2][1],
            linestyle="--",
            alpha=0.5,
            label="eWAVE AMP (am)",
        )
        ax.axhline(y=0, color="w", linestyle="--", alpha=0.3)
        # WC1 markers (only if WC1 is used)
        if use_wc1:
            ax.axvline(x=wc1_x_am, color="orange", linestyle="--", alpha=0.5, label="WC1")
            ax.axvline(
                x=wc1_x_am + base_wavelength_am,
                color="yellow",
                linestyle="--",
                alpha=0.3,
                label="WC1 ± λ",
            )
            ax.axvline(
                x=wc1_x_am - base_wavelength_am,
                color="yellow",
                linestyle="--",
                alpha=0.3,
            )
        # WC2 markers (only if WC2 is used)
        if use_wc2:
            ax.axvline(x=wc2_x_am, color="cyan", linestyle="--", alpha=0.5, label="WC2")
            ax.axvline(
                x=wc2_x_am + base_wavelength_am,
                color="magenta",
                linestyle="--",
                alpha=0.3,
                label="WC2 ± λ",
            )
            ax.axvline(
                x=wc2_x_am - base_wavelength_am,
                color="magenta",
                linestyle="--",
                alpha=0.3,
            )
        ax.set_xlabel("X (am)", family="Monospace")
        ax.set_ylabel("Displacement (am)", family="Monospace")

        # RMS text objects for 3 sections (left, center, right)
        # Each displays regional RMS: left of WC1, between WCs, right of WC2
        rms_text_props = dict(
            transform=ax.transAxes,
            fontsize=12,
            family="Monospace",
            color=colormap.viridis_palette[4][1],
            verticalalignment="bottom",
            animated=True,
            bbox=dict(facecolor=colormap.DARK_GRAY[1], edgecolor="none", pad=2),
        )
        # Left section: positioned at left
        rms_text_left = ax.text(0.02, 1.02, "", horizontalalignment="left", **rms_text_props)
        # Center section: positioned at center
        rms_text_center = ax.text(0.5, 1.02, "", horizontalalignment="center", **rms_text_props)
        # Right section: positioned at right
        rms_text_right = ax.text(0.98, 1.02, "", horizontalalignment="right", **rms_text_props)
        rms_texts.append((rms_text_left, rms_text_center, rms_text_right))

        # Gradient direction indicators (only when both WCs are used)
        # Positioned between RMS texts: left_grad between left-center, right_grad between center-right
        # Blue = attraction (gradient toward center), Red = repulsion (gradient away from center)
        if use_wc1 and use_wc2:
            grad_text_props = dict(
                transform=ax.transAxes,
                fontsize=10,
                family="Monospace",
                verticalalignment="bottom",
                animated=True,
                bbox=dict(facecolor=colormap.DARK_GRAY[1], edgecolor="none", pad=2),
            )
            grad_text_left = ax.text(
                0.26, 1.02, "", horizontalalignment="center", **grad_text_props
            )
            grad_text_right = ax.text(
                0.74, 1.02, "", horizontalalignment="center", **grad_text_props
            )
            gradient_texts.append((grad_text_left, grad_text_right))
        else:
            gradient_texts.append((None, None))

        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    plt.tight_layout()
    # Adjust spacing: top for suptitle + RMS texts, hspace for RMS texts between plots
    plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.35)

    def update_plot(frame):
        """Update plot data for a given frame."""
        t_rs = (frame / ANIMATION_FRAMES) * PERIOD_RS

        # Compute and update each plot from config
        for i, (line, rms_line, rms_text_tuple, grad_text_tuple, cfg) in enumerate(
            zip(lines, rms_lines, rms_texts, gradient_texts, PLOT_CONFIGS)
        ):
            psi = compute_plot_value(cfg, t_rs)
            line.set_data(x_am, psi)

            # Update per-x RMS amplitude via cycle averaging
            # Accumulate ψ² during first cycle, then use fixed average
            # RMS = √⟨ψ²⟩ represents energy-equivalent amplitude (Energy ∝ ψ²)
            psi2 = psi**2  # squared displacement

            if not cycle_complete[i]:
                # First cycle: accumulate ψ²
                rms_x_am2_sum[i] += psi2
                # Check if cycle complete (frame wraps back to 0)
                if frame == ANIMATION_FRAMES - 1:
                    rms_x_am2_avg[i] = rms_x_am2_sum[i] / ANIMATION_FRAMES
                    cycle_complete[i] = True

            # Use current average (running during first cycle, fixed after)
            if cycle_complete[i]:
                rms_x_am2 = rms_x_am2_avg[i]
            else:
                # During first cycle, show running average
                frames_so_far = frame + 1
                rms_x_am2 = rms_x_am2_sum[i] / frames_so_far

            rms_x_am = np.sqrt(rms_x_am2)  # per-x RMS amplitude
            rms_line.set_data(x_am, rms_x_am)

            # Calculate regional RMS for 3 sections: left, center, right
            # Regions depend on active WCs (center only exists with both WCs)
            rms_text_left, rms_text_center, rms_text_right = rms_text_tuple
            rms_left = np.sqrt(np.mean(rms_x_am2[mask_left])) if mask_left.any() else 0.0
            rms_right = np.sqrt(np.mean(rms_x_am2[mask_right])) if mask_right.any() else 0.0
            rms_text_left.set_text(f"(RMS = {rms_left:.2f}am)")
            rms_text_right.set_text(f"(RMS = {rms_right:.2f}am)")
            # Center region only exists when both WCs are used
            if mask_center.any():
                rms_center = np.sqrt(np.mean(rms_x_am2[mask_center]))
                rms_text_center.set_text(f"(RMS = {rms_center:.2f}am)")

                # Calculate and display amplitude gradient direction indicators (only with both WCs)
                #
                # Simplified Discrete Gradient: ∇A ≈ ΔA / Δx
                #   - ΔA = amplitude difference between adjacent regions (am)
                #   - Δx = wc_spacing as characteristic length scale (am)
                #   - Units: amplitude gradient (am/am = dimensionless, or 1/am if normalized)
                #
                # This is a discrete approximation of the continuous spatial derivative.
                # For accurate force calculation: F ∝ -∇E, where E ∝ A² (wave energy).
                # Here we use amplitude gradient as proxy for energy gradient direction.
                #
                # Force Direction (from -∇E principle):
                #   - Positive gradient (center > side) = force pushes outward = repulsion (red)
                #   - Negative gradient (center < side) = force pushes inward = attraction (blue)
                #   - Near zero gradient = neutral equilibrium (gray)
                #
                grad_text_left, grad_text_right = grad_text_tuple
                if grad_text_left is not None and grad_text_right is not None:
                    # Discrete gradient: ΔA / Δx (amplitude change per unit distance)
                    left_gradient = (rms_center - rms_left) / wc_spacing  # am / am
                    right_gradient = (rms_center - rms_right) / wc_spacing  # am / am
                    neutral_threshold = 0.01 * base_amplitude_am / wc_spacing  # 1% gradient

                    # Left gradient indicator (between left RMS and center RMS)
                    # Positive gradient = center higher = force pushes WC1 left (away) = repulsion
                    # Negative gradient = center lower = force pushes WC1 right (toward) = attraction
                    if abs(left_gradient) < neutral_threshold:
                        grad_text_left.set_text("(Neutral)")
                        grad_text_left.set_color("#aaaaaa")  # light gray
                    elif left_gradient > 0:
                        grad_text_left.set_text("<<<<< Repulsion")
                        grad_text_left.set_color("#ff4444")  # red
                    else:
                        grad_text_left.set_text("Attraction >>>>>")
                        grad_text_left.set_color("#4488ff")  # blue

                    # Right gradient indicator (between center RMS and right RMS)
                    # Positive gradient = center higher = force pushes WC2 right (away) = repulsion
                    # Negative gradient = center lower = force pushes WC2 left (toward) = attraction
                    if abs(right_gradient) < neutral_threshold:
                        grad_text_right.set_text("(Neutral)")
                        grad_text_right.set_color("#aaaaaa")  # light gray
                    elif right_gradient > 0:
                        grad_text_right.set_text("Repulsion >>>>>")
                        grad_text_right.set_color("#ff4444")  # red
                    else:
                        grad_text_right.set_text("<<<<< Attraction")
                        grad_text_right.set_color("#4488ff")  # blue
            else:
                rms_text_center.set_text("")  # hide center text when not applicable

        # Update time display with pause indicator
        pause_str = " [PAUSED]" if state["paused"] else ""
        time_text.set_text(f"t = {t_rs:.4f} rs  (frame {frame}/{ANIMATION_FRAMES}){pause_str}")

        # Flatten rms_texts and gradient_texts tuples for return
        all_rms_texts = [t for tup in rms_texts for t in tup]
        all_grad_texts = [t for tup in gradient_texts for t in tup if t is not None]
        return (
            tuple(lines)
            + tuple(rms_lines)
            + tuple(all_rms_texts)
            + tuple(all_grad_texts)
            + (time_text,)
        )

    def init():
        """Initialize animation."""
        return update_plot(state["frame"])

    def animate(frame):
        """Update animation for each frame."""
        if state["paused"]:
            return update_plot(state["frame"])
        state["frame"] = frame
        return update_plot(frame)

    def on_key(event):
        """Handle keyboard events for pause/resume and frame stepping.

        Controls:
            SPACE: Pause/Resume
            LEFT/RIGHT: Step frame when paused
            R: Reset RMS trackers to zero
        """
        if event.key == " ":
            state["paused"] = not state["paused"]
            update_plot(state["frame"])
            fig.canvas.draw_idle()
        elif event.key == "right" and state["paused"]:
            state["frame"] = (state["frame"] + 1) % ANIMATION_FRAMES
            update_plot(state["frame"])
            fig.canvas.draw_idle()
        elif event.key == "left" and state["paused"]:
            state["frame"] = (state["frame"] - 1) % ANIMATION_FRAMES
            update_plot(state["frame"])
            fig.canvas.draw_idle()
        elif event.key == "r":
            # Reset RMS trackers to zero and restart cycle averaging
            for i in range(len(rms_x_am2_sum)):
                rms_x_am2_sum[i] = np.zeros_like(x_am)
                rms_x_am2_avg[i] = None
                cycle_complete[i] = False
            state["frame"] = 0  # restart from frame 0 for clean cycle
            update_plot(state["frame"])
            fig.canvas.draw_idle()

    # Store lines in state for potential external access
    state["lines"] = lines

    # Connect keyboard handler
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Create animation
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=ANIMATION_FRAMES,
        interval=ANIMATION_INTERVAL,
        blit=True,
    )

    # Save as GIF if requested
    if save_gif:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = PLOT_DIR / "wave_animation.gif"
        print(f"\nSaving animation to {save_path}...")
        anim.save(save_path, writer="pillow", fps=20)
        print("Animation saved!\n")

    return anim


# Plot 4: Outgoing Transverse Displacement (bottom)
# plt.subplot(3, 1, 3)
# plt.plot(
#     compute_radius_am(),
#     psi_outTrans_am,
#     color=colormap.ironbow_palette[2][1],
#     linewidth=2,
#     label="Psi Outgoing Transverse (am)",
# )
# plt.axhline(
#     y=constants.EWAVE_AMPLITUDE / constants.ATTOMETER,
#     color=colormap.ironbow_palette[2][1],
#     linestyle="--",
#     alpha=0.5,
#     label="eWAVE AMP (am)",
# )
# plt.axhline(y=0, color="w", linestyle="--", alpha=0.3)
# plt.xlabel("Radius (am)", family="Monospace")
# plt.ylabel("Displacement (am)", family="Monospace")
# plt.grid(True, alpha=0.3)
# plt.legend()


if __name__ == "__main__":
    # Single plot function with pause support
    # Controls: SPACE = pause/resume, LEFT/RIGHT = step frames when paused
    anim = plot_waves(
        start_paused=START_PAUSED,
        start_frame=START_FRAME,
        save_gif=SAVE_GIF,
    )
    plt.show()
