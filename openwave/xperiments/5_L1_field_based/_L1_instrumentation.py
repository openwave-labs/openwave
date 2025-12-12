"""
XPERIMENT INSTRUMENTATION (data collection)

This provides zero-overhead data collection that can be toggled on/off per xperiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

from openwave.common import colormap, constants


# ================================================================
# Module-level Constants (computed once at import)
# ================================================================

_MODULE_DIR = Path(__file__).parent
DATA_DIR = _MODULE_DIR / "_data"
PLOT_DIR = _MODULE_DIR / "_plots"

# Module-level state
_timestep_buffer = []
_timestep_log_initialized = False
_BUFFER_FLUSH_INTERVAL = 100  # Flush every N timesteps


# ================================================================
# Instrumentation Functions (Zero-Overhead)
# ================================================================


def plot_static_charge_profile(wave_field):
    """
    Plot the displacement profile along the x-axis through the center of the wave field.

    Args:
        wave_field: WaveField instance containing displacement data
    """

    # Get center indices
    center_j = wave_field.ny // 2
    center_k = wave_field.nz // 2

    # Extract displacement along x-axis at center (y, z)
    x_indices = np.arange(wave_field.nx)
    displacements_L = np.zeros(wave_field.nx)
    displacements_T = np.zeros(wave_field.nx)

    # Sample longitudinal displacement values
    for i in range(wave_field.nx):
        displacements_L[i] = wave_field.displacement_am[i, center_j, center_k]
        # displacements_L[i] = wave_field.displacement_old_am[i, center_j, center_k]
        displacements_T[i] = 0.0

    # Calculate distance from center in grid indices
    center_x = wave_field.nx // 2
    distances = x_indices - center_x

    # Create the plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 6), facecolor=colormap.DARK_GRAY[1])
    fig.suptitle("OPENWAVE Analytics", fontsize=20, family="Monospace")

    # Plot 1: Longitudinal Displacement vs distance from center
    plt.subplot(1, 2, 1)
    plt.plot(
        distances,
        displacements_L,
        color=colormap.viridis_palette[0][1],
        linewidth=4,
        label="LONGITUDINAL",
    )
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.3)
    plt.ylim(-1.5, 1.5)
    plt.xlabel("Distance from Center (grid indices)", family="Monospace")
    plt.ylabel("Displacement (attometers)", family="Monospace")
    plt.title("INITIAL CHARGE PROFILE", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Transverse Displacement vs distance from center
    plt.subplot(1, 2, 2)
    plt.plot(
        distances,
        displacements_T,
        color=colormap.viridis_palette[4][1],
        linewidth=4,
        label="TRANSVERSE",
    )
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.3)
    plt.ylim(-1.5, 1.5)
    plt.xlabel("Distance from Center (grid indices)", family="Monospace")
    plt.ylabel("Displacement (attometers)", family="Monospace")
    plt.title("INITIAL CHARGE PROFILE", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save to directory
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOT_DIR / "charge_profile.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("\nPlot charge_profile saved to:\n", save_path, "\n")


def log_timestep_data(timestep: int, charge_level: float, wave_field, trackers) -> None:
    """Record all timestep data to a buffer, flush periodically to reduce I/O overhead.

    Args:
        timestep: Current simulation timestep
        charge_level: Current charge level (0.0 to 1.0+)
        wave_field: WaveField instance
        trackers: Trackers instance
    """
    global _timestep_buffer, _timestep_log_initialized

    # Define probe position
    px, py, pz = wave_field.nx * 5 // 6, wave_field.ny * 5 // 6, wave_field.nz // 2

    # Capture probe values
    displacement_am = wave_field.displacement_am[px, py, pz] / wave_field.scale_factor
    amplitude_am = trackers.amplitudeL_am[px, py, pz] / wave_field.scale_factor
    frequency_rHz = trackers.frequency_rHz[px, py, pz] * wave_field.scale_factor

    # Add to buffer
    _timestep_buffer.append([timestep, charge_level, displacement_am, amplitude_am, frequency_rHz])

    # Flush buffer periodically
    if len(_timestep_buffer) >= _BUFFER_FLUSH_INTERVAL:
        _flush_timestep_buffer()


def _flush_timestep_buffer() -> None:
    """Write buffered timestep data to disk."""
    global _timestep_buffer, _timestep_log_initialized

    if not _timestep_buffer:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DATA_DIR / "timestep_data.csv"

    # Write header on first flush
    if not _timestep_log_initialized:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestep", "charge_level", "displacement_am", "amplitude_am", "frequency_rHz"]
            )
        _timestep_log_initialized = True

    # Append all buffered rows at once
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        for row in _timestep_buffer:
            writer.writerow(
                [row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}"]
            )

    _timestep_buffer = []


def _read_timestep_data():
    """Read timestep data from consolidated CSV file.

    Returns:
        dict: Dictionary with lists for each column, or None if file doesn't exist
    """
    log_path = DATA_DIR / "timestep_data.csv"
    if not log_path.exists():
        print("\nTimestep data log file does not exist.\n")
        return None

    data = {
        "timesteps": [],
        "charge_levels": [],
        "displacements": [],
        "amplitudes": [],
        "frequencies": [],
    }

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["timesteps"].append(int(row["timestep"]))
            data["charge_levels"].append(float(row["charge_level"]))
            data["displacements"].append(float(row["displacement_am"]))
            data["amplitudes"].append(float(row["amplitude_am"]))
            data["frequencies"].append(float(row["frequency_rHz"]))

    return data


def plot_charge_log():
    """Plot the logged charge level over time."""
    data = _read_timestep_data()
    if data is None:
        return

    # Create the plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 6), facecolor=colormap.DARK_GRAY[1])
    fig.suptitle("OPENWAVE Analytics", fontsize=20, family="Monospace")

    plt.plot(
        data["timesteps"],
        [cl * 100 for cl in data["charge_levels"]],
        color=colormap.viridis_palette[2][1],
        linewidth=3,
        label="CHARGE LEVEL",
    )
    plt.axhline(y=120, color=colormap.RED[1], linestyle="--", alpha=0.5, label="MAX CHARGE LEVEL")
    plt.axhline(
        y=100, color=colormap.GREEN[1], linestyle="--", alpha=0.5, label="OPTIMAL CHARGE LEVEL"
    )
    plt.axhline(
        y=80, color=colormap.ORANGE[1], linestyle="--", alpha=0.5, label="MIN CHARGE LEVEL"
    )

    plt.xlabel("Timestep", family="Monospace")
    plt.ylabel("Charge Level (%)", family="Monospace")
    plt.title("ENERGY CHARGING & STABILITY", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save to directory
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOT_DIR / "charge_levels.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("\nPlot charge_levels saved to:\n", save_path, "\n")


def plot_probe_log():
    """Plot the logged displacement, amplitude, and frequency over time."""
    data = _read_timestep_data()
    if data is None:
        return

    # Create the plot with 2 subplots (stacked vertically)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 8), facecolor=colormap.DARK_GRAY[1])
    fig.suptitle("OPENWAVE Analytics", fontsize=20, family="Monospace")

    # Plot 1: Displacement and Amplitude (top)
    plt.subplot(2, 1, 1)
    plt.plot(
        data["timesteps"],
        data["displacements"],
        color=colormap.ironbow_palette[2][1],
        linewidth=2,
        label="DISPLACEMENT (am)",
    )
    plt.plot(
        data["timesteps"],
        data["amplitudes"],
        color=colormap.ironbow_palette[3][1],
        linewidth=2,
        label="RMS AMPLITUDE (am)",
    )
    plt.axhline(
        y=constants.EWAVE_AMPLITUDE / constants.ATTOMETER,
        color=colormap.ironbow_palette[4][1],
        linestyle="--",
        alpha=0.5,
        label="eWAVE AMPLITUDE (am)",
    )
    plt.axhline(y=0, color="w", linestyle="--", alpha=0.3)
    plt.xlabel("Timestep", family="Monospace")
    plt.ylabel("Displacement / RMS Amplitude (am)", family="Monospace")
    plt.title("DISPLACEMENT & AMPLITUDE OVER TIME", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Frequency (bottom)
    plt.subplot(2, 1, 2)
    plt.plot(
        data["timesteps"],
        data["frequencies"],
        color=colormap.blueprint_palette[2][1],
        linewidth=2,
        label="FREQUENCY (rHz)",
    )
    plt.axhline(
        y=constants.EWAVE_FREQUENCY * constants.RONTOSECOND,
        color=colormap.blueprint_palette[1][1],
        linestyle="--",
        alpha=0.5,
        label="eWAVE FREQUENCY (rHz)",
    )
    plt.axhline(y=0, color="w", linestyle="--", alpha=0.3)
    plt.xlabel("Timestep", family="Monospace")
    plt.ylabel("Frequency (rHz)", family="Monospace")
    plt.title("FREQUENCY OVER TIME", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save to directory
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOT_DIR / "probe_values.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("\nPlot probe values saved to:\n", save_path, "\n")


def generate_plots():
    """Generate all instrumentation plots."""
    # Flush any remaining buffered data before plotting
    _flush_timestep_buffer()
    plot_charge_log()
    plot_probe_log()
