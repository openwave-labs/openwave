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
DATA_DIR = _MODULE_DIR / "data"
PLOT_DIR = _MODULE_DIR / "plots"

# Module-level state
_timestep_buffer = []
_timestep_log_initialized = False
_BUFFER_FLUSH_INTERVAL = 100  # Flush every N timesteps


# ================================================================
# Instrumentation Functions (Zero-Overhead)
# ================================================================


def plot_probe_wave_profile(wave_field):
    """
    Plot the displacement profile along the x-axis through the probe position.

    Args:
        wave_field: WaveField instance containing displacement data
    """

    # Define probe position
    px, py, pz = wave_field.nx // 2, wave_field.ny // 2, wave_field.nz // 2

    # Extract displacement along x-axis at center (y, z)
    x_indices = np.arange(wave_field.nx)
    displacements_L = np.zeros(wave_field.nx)
    displacements_T = np.zeros(wave_field.nx)

    # Sample longitudinal displacement values
    for i in range(wave_field.nx):
        displacements_L[i] = wave_field.psiL_am[i, py, pz]
        displacements_T[i] = wave_field.psiT_am[i, py, pz]

    # Calculate distance from center in grid indices
    distances = x_indices - px

    # Create the plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 6), facecolor=colormap.DARK_GRAY[1])
    fig.suptitle("OPENWAVE Analytics", fontsize=20, family="Monospace")

    # Plot 1: Longitudinal Displacement vs distance from center
    plt.subplot(1, 2, 1)
    plt.plot(
        distances,
        displacements_L,
        color=colormap.viridis_palette[2][1],
        linewidth=4,
        label="LONGITUDINAL",
    )
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.3)
    plt.ylim(-1.0, 6.0)
    plt.xlabel("Distance from Wave-Center (grid indices)", family="Monospace")
    plt.ylabel("Displacement (attometers)", family="Monospace")
    plt.title("WAVE PROFILE", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Transverse Displacement vs distance from center
    plt.subplot(1, 2, 2)
    plt.plot(
        distances,
        displacements_T,
        color=colormap.ironbow_palette[2][1],
        linewidth=4,
        label="TRANSVERSE",
    )
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.3)
    plt.ylim(-1.0, 6.0)
    plt.xlabel("Distance from Wave-Center (grid indices)", family="Monospace")
    plt.ylabel("Displacement (attometers)", family="Monospace")
    plt.title("WAVE PROFILE", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save to directory
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOT_DIR / "wave_profile.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("\nPlot wave_profile saved to:\n", save_path, "\n")


def log_timestep_data(timestep: int, wave_field, trackers) -> None:
    """Record all timestep data to a buffer, flush periodically to reduce I/O overhead.

    Args:
        timestep: Current simulation timestep
        wave_field: WaveField instance
        trackers: Trackers instance
    """
    global _timestep_buffer, _timestep_log_initialized

    # Define probe position
    px, py, pz = wave_field.nx // 2, wave_field.ny // 2, wave_field.nz // 2

    # Capture probe values
    psiL_am = wave_field.psiL_am[px, py, pz] / wave_field.scale_factor
    psiT_am = wave_field.psiT_am[px, py, pz] / wave_field.scale_factor
    ampL_am = trackers.ampL_am[px, py, pz] / wave_field.scale_factor
    ampT_am = trackers.ampT_am[px, py, pz] / wave_field.scale_factor
    freq_rHz = trackers.freq_rHz[px, py, pz] * wave_field.scale_factor

    # Add to buffer
    _timestep_buffer.append([timestep, psiL_am, psiT_am, ampL_am, ampT_am, freq_rHz])

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
                [
                    "timestep",
                    "psiL_am",
                    "psiT_am",
                    "ampL_am",
                    "ampT_am",
                    "freq_rHz",
                ]
            )
        _timestep_log_initialized = True

    # Append all buffered rows at once
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        for row in _timestep_buffer:
            writer.writerow(
                [
                    row[0],
                    f"{row[1]:.6f}",
                    f"{row[2]:.6f}",
                    f"{row[3]:.6f}",
                    f"{row[4]:.6f}",
                    f"{row[5]:.6f}",
                ]
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
        "displacements_L": [],
        "displacements_T": [],
        "amplitudes_L": [],
        "amplitudes_T": [],
        "frequencies": [],
    }

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["timesteps"].append(int(row["timestep"]))
            data["displacements_L"].append(float(row["psiL_am"]))
            data["displacements_T"].append(float(row["psiT_am"]))
            data["amplitudes_L"].append(float(row["ampL_am"]))
            data["amplitudes_T"].append(float(row["ampT_am"]))
            data["frequencies"].append(float(row["freq_rHz"]))

    return data


def plot_probe_values():
    """Plot the logged displacement, amplitude, and frequency over time."""
    data = _read_timestep_data()
    if data is None:
        return

    # Create the plot with 3 subplots (stacked vertically)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(9, 9), facecolor=colormap.DARK_GRAY[1])
    fig.suptitle("OPENWAVE Analytics", fontsize=20, family="Monospace")

    # Plot 1: Longitudinal Displacement and Amplitude (top)
    plt.subplot(3, 1, 1)
    plt.plot(
        data["timesteps"],
        data["displacements_L"],
        color=colormap.viridis_palette[2][1],
        linewidth=2,
        label="DISPLACEMENT (am)",
    )
    plt.plot(
        data["timesteps"],
        data["amplitudes_L"],
        color=colormap.viridis_palette[3][1],
        linewidth=2,
        label="RMS AMPLITUDE (am)",
    )
    plt.axhline(
        y=constants.EWAVE_AMPLITUDE / constants.ATTOMETER,
        color=colormap.viridis_palette[4][1],
        linestyle="--",
        alpha=0.5,
        label="eWAVE AMPLITUDE (am)",
    )
    plt.axhline(y=0, color="w", linestyle="--", alpha=0.3)
    plt.xlabel("Timestep", family="Monospace")
    plt.ylabel("Displacement / Amplitude (am)", family="Monospace")
    plt.title("(LONGITUDINAL) DISPLACEMENT & AMPLITUDE OVER TIME", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Transverse Displacement and Amplitude (middle)
    plt.subplot(3, 1, 2)
    plt.plot(
        data["timesteps"],
        data["displacements_T"],
        color=colormap.ironbow_palette[2][1],
        linewidth=2,
        label="DISPLACEMENT (am)",
    )
    plt.plot(
        data["timesteps"],
        data["amplitudes_T"],
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
    plt.ylabel("Displacement / Amplitude (am)", family="Monospace")
    plt.title("(TRANSVERSE) DISPLACEMENT & AMPLITUDE OVER TIME", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Frequency (bottom)
    plt.subplot(3, 1, 3)
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
    plot_probe_values()
    plt.show()


if __name__ == "__main__":
    plot_probe_values()
    plt.show()
