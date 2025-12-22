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

# Compute angular frequency (ω = 2πf) for temporal term variation
omega_rs = 2.0 * np.pi * base_frequency_rHz  # angular frequency (rad/rs)

# Compute angular wave number (k = 2π/λ) for spatial term variation
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

x_am = np.linspace(-10 * base_wavelength_am, 10 * base_wavelength_am, 1000)

wc1_x_am = -4 * base_wavelength_am  # am, center of wave source 1
wc2_x_am = +4 * base_wavelength_am  # am, center of wave source 2
wc1_phase_rad = 0.0  # radians, phase offset for source 1
wc2_phase_rad = np.pi  # radians, phase offset for source 2


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


def get_source_phase(source: int = 1) -> float:
    """Get phase offset for a wave source."""
    if source == 1:
        return wc1_phase_rad
    elif source == 2:
        return wc2_phase_rad
    else:
        raise ValueError(f"Invalid source: {source}. Use 1 or 2.")


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
    source_phase = get_source_phase(source)
    psi_am = base_amplitude_am * np.cos(  # full amplitude (am)  # oscillator cosine function
        omega_rs * t_rs  # temporal term (rad)
        + direction * k_am * radius_am  # spatial term, spherical wave fronts, φ = ±kr (rad)
        + source_phase  # direction only affects the spatial term, not the source phase, (rad)
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
    source_phase = get_source_phase(source)
    # λ as reference radius → amplitude = A₀ at r = λ
    # Clamp r to avoid singularity at r = 0 (min r = λ / 2π)
    r_safe_am = np.maximum(radius_am, base_wavelength_am / (2 * np.pi))
    amplitude_at_r_am = base_amplitude_am * base_wavelength_am / r_safe_am
    psi_am = (
        amplitude_at_r_am  # amplitude falloff for spherical wave: A(r) = A₀ · (λ / r) (am)
        * np.cos(  # oscillator cosine function
            omega_rs * t_rs  # temporal term (rad)
            + direction * k_am * radius_am  # spatial term, spherical wave fronts, φ = ±kr (rad)
            + source_phase  # direction only affects the spatial term, not the source phase, (rad)
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
    source_phase = get_source_phase(source)
    psi_am = (
        base_amplitude_am  # full amplitude (am)
        * np.cos(omega_rs * t_rs + source_phase)  # temporal term with source offset (rad)
        * np.sin(k_am * radius_am)  # spatial term, φ = k·r (rad)
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
    source_phase = get_source_phase(source)
    psi_am = (
        base_amplitude_am  # full amplitude (am)
        * (
            np.sin(omega_rs * t_rs + source_phase - k_am * radius_am)
            - np.sin(omega_rs * t_rs + source_phase)
        )  # both terms share source phase (same oscillator)
        / (k_am * radius_am)
        * 2
        * np.pi  # TODO: test remove 2π factor
    )
    return psi_am


# ================================================================
# Plot Configuration
# ================================================================

# Each plot config: func, direction, source, ylim, height_ratio, title, label
# - func: string or list of strings (for sum)
# - direction: +1 incoming, -1 outgoing (or list for each func)
# - source: 1 = wc1, 2 = wc2 (or list for each func)
PLOT_CONFIGS = [
    {
        "func": "fullamp",
        "direction": 1,
        "source": 1,  # WC1
        "ylim": (-1.5, 1.5),
        "height_ratio": 1,
        "title": "INCOMING LONGITUDINAL DISPLACEMENT",
        "label": "Psi In (am)",
    },
    {
        "func": "ampfalloff",
        "direction": -1,
        "source": 1,  # WC1
        "ylim": (-1.5, 6.5),
        "height_ratio": 2,
        "title": "OUTGOING LONGITUDINAL DISPLACEMENT",
        "label": "Psi Out (am)",
    },
    {
        "func": ["fullamp", "ampfalloff"],  # sum of both sources
        "direction": [1, -1],
        "source": [1, 1],  # WC1 + WC1
        "ylim": (-3.5, 7.5),
        "height_ratio": 2,
        "title": "TOTAL LONGITUDINAL DISPLACEMENT",
        "label": "Psi Total (am)",
    },
]

PLOT_CONFIGS2 = [
    {
        "func": ["fullamp", "fullamp"],  # sum of both sources
        "direction": [1, 1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 1.5),
        "height_ratio": 1,
        "title": "INCOMING LONGITUDINAL DISPLACEMENT",
        "label": "Psi In (am)",
    },
    {
        "func": ["ampfalloff", "ampfalloff"],  # sum of both sources
        "direction": [-1, -1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 6.5),
        "height_ratio": 2,
        "title": "OUTGOING LONGITUDINAL DISPLACEMENT",
        "label": "Psi Out (am)",
    },
    {
        "func": ["fullamp", "ampfalloff", "fullamp", "ampfalloff"],  # sum of both sources
        "direction": [1, -1, 1, -1],
        "source": [1, 1, 2, 2],  # WC1 + WC2
        "ylim": (-3.5, 7.5),
        "height_ratio": 2,
        "title": "TOTAL LONGITUDINAL DISPLACEMENT",
        "label": "Psi Total (am)",
    },
]

PLOT_CONFIGS3 = [
    {
        "func": ["fullamp", "fullamp"],  # sum of both sources
        "direction": [-1, -1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 1.5),
        "height_ratio": 1,
        "title": "INCOMING LONGITUDINAL DISPLACEMENT",
        "label": "Psi In (am)",
    },
    {
        "func": ["ampfalloff", "ampfalloff"],  # sum of both sources
        "direction": [1, 1],
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 6.5),
        "height_ratio": 2,
        "title": "OUTGOING LONGITUDINAL DISPLACEMENT",
        "label": "Psi Out (am)",
    },
    {
        "func": ["fullamp", "ampfalloff", "fullamp", "ampfalloff"],  # sum of both sources
        "direction": [1, -1, 1, -1],
        "source": [1, 1, 2, 2],  # WC1 + WC2
        "ylim": (-3.5, 7.5),
        "height_ratio": 2,
        "title": "TOTAL LONGITUDINAL DISPLACEMENT",
        "label": "Psi Total (am)",
    },
]

PLOT_CONFIGS4 = [
    {
        "func": ["wolff", "wolff"],  # sum of both sources
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 6.5),
        "height_ratio": 1,
        "title": "WOLFF STANDING WAVE",
        "label": "Psi Wolff",
    },
    {
        "func": ["lafreniere", "lafreniere"],  # sum of both sources
        "source": [1, 2],  # WC1 + WC2
        "ylim": (-1.5, 6.5),
        "height_ratio": 1,
        "title": "LAFRENIERE STANDING WAVE",
        "label": "Psi LaFreniere",
    },
]

# Function registry - maps string names to actual functions
WAVE_FUNCTIONS = {
    "fullamp": compute_wave_fullamp,
    "ampfalloff": compute_wave_ampfalloff,
    "wolff": compute_wave_Wolff,
    "lafreniere": compute_wave_LaFreniere,
}


def compute_plot_value(config: dict, t_rs: float) -> np.ndarray:
    """Compute wave value for a plot config (handles single func or sum of funcs).

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
            if func_name in ("fullamp", "ampfalloff"):
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
        if func_spec in ("fullamp", "ampfalloff"):
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
    Each plot shows the wave displacement and an amplitude envelope (peak values per x).

    Controls:
        SPACE: Pause/Resume animation
        LEFT/RIGHT arrows: Step frame when paused
        R: Reset envelope to zero

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

    # Initialize line objects for each plot (wave + envelope)
    lines = []
    envelope_lines = []
    envelopes = [np.zeros_like(x_am) for _ in PLOT_CONFIGS]  # track peak per x

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

        # Amplitude envelope line (positive peaks only)
        (env_line,) = ax.plot(
            [],
            [],
            color=colormap.viridis_palette[4][1],
            linewidth=1.5,
            linestyle="-",
            alpha=0.8,
            label="Envelope (peak)",
        )
        envelope_lines.append(env_line)

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
        # WC1 markers
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
        # WC2 markers
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
        ax.set_title(cfg["title"], family="Monospace")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    plt.tight_layout()

    def update_plot(frame):
        """Update plot data for a given frame."""
        t_rs = (frame / ANIMATION_FRAMES) * PERIOD_RS

        # Compute and update each plot from config
        for i, (line, env_line, cfg) in enumerate(zip(lines, envelope_lines, PLOT_CONFIGS)):
            psi = compute_plot_value(cfg, t_rs)
            line.set_data(x_am, psi)

            # Update envelope: track max modulus displacement per x, max(|psi|)
            envelopes[i] = np.maximum(envelopes[i], np.abs(psi))
            env_line.set_data(x_am, envelopes[i])

        # Update time display with pause indicator
        pause_str = " [PAUSED]" if state["paused"] else ""
        time_text.set_text(f"t = {t_rs:.4f} rs  (frame {frame}/{ANIMATION_FRAMES}){pause_str}")

        return tuple(lines) + tuple(envelope_lines) + (time_text,)

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
            R: Reset envelope to zero
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
            # Reset envelopes to zero
            for i in range(len(envelopes)):
                envelopes[i] = np.zeros_like(x_am)
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
# plt.title("OUTGOING TRANSVERSE DISPLACEMENT", family="Monospace")
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
