"""
XPERIMENT INSTRUMENTATION (data collection)

This provides zero-overhead data collection that can be toggled on/off per xperiment.
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

from openwave.common import colormap


# ================================================================
# Module-level Constants (computed once at import)
# ================================================================

_MODULE_DIR = Path(__file__).parent
DATA_DIR = _MODULE_DIR / "_data"
PLOT_DIR = _MODULE_DIR / "_plots"

# Module-level state
_log_charge_level_initialized = False
_log_displacement_initialized = False


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
    print("\n" + "=" * 64)
    print("INSTRUMENTATION ENABLED")
    print("=" * 64)
    print("\nPlot charge_profile saved to:\n", save_path, "\n")


def log_charge_level(timestep: int, charge_level: float) -> None:
    """Record charge level at the current timestep."""
    global _log_charge_level_initialized

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DATA_DIR / "charge_levels.csv"

    # Write header on first call
    if not _log_charge_level_initialized:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "charge_level"])
        _log_charge_level_initialized = True

    # Append charge level data
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestep, f"{charge_level:.6f}"])


def plot_charge_log():
    """Plot the logged charge level over time."""
    log_path = DATA_DIR / "charge_levels.csv"
    if not log_path.exists():
        print("Charge log file does not exist.")
        return

    timesteps = []
    charge_levels = []

    # Read logged data
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps.append(int(row["timestep"]))
            charge_levels.append(float(row["charge_level"]))

    # Create the plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 6), facecolor=colormap.DARK_GRAY[1])
    fig.suptitle("OPENWAVE Analytics", fontsize=20, family="Monospace")

    plt.plot(
        timesteps,
        [cl * 100 for cl in charge_levels],
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
    plt.title("ENERGY CHARGING & STABILIZATION", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save to directory
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOT_DIR / "charge_levels.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("\nPlot charge_levels saved to:\n", save_path, "\n")


def log_displacement(timestep: int, displacement_am: float) -> None:
    """Record displacement at the current timestep and voxel."""
    global _log_displacement_initialized

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DATA_DIR / "displacements.csv"

    # Write header on first call
    if not _log_displacement_initialized:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "displacement_am"])
        _log_displacement_initialized = True

    # Append displacement data
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestep, f"{displacement_am:.6f}"])


def plot_displacement_log():
    """Plot the logged displacement over time."""
    log_path = DATA_DIR / "displacements.csv"
    if not log_path.exists():
        print("Displacement log file does not exist.")
        return

    timesteps = []
    displacements = []

    # Read logged data
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps.append(int(row["timestep"]))
            displacements.append(float(row["displacement_am"]))

    # Create the plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 6), facecolor=colormap.DARK_GRAY[1])
    fig.suptitle("OPENWAVE Analytics", fontsize=20, family="Monospace")

    plt.plot(
        timesteps,
        displacements,
        color=colormap.viridis_palette[2][1],
        linewidth=3,
        label="DISPLACEMENT (am)",
    )
    # plt.axhline(y=120, color=colormap.RED[1], linestyle="--", alpha=0.5, label="MAX CHARGE LEVEL")
    # plt.axhline(
    #     y=100, color=colormap.GREEN[1], linestyle="--", alpha=0.5, label="OPTIMAL CHARGE LEVEL"
    # )
    # plt.axhline(
    #     y=80, color=colormap.ORANGE[1], linestyle="--", alpha=0.5, label="MIN CHARGE LEVEL"
    # )

    plt.xlabel("Timestep", family="Monospace")
    plt.ylabel("Displacement (am)", family="Monospace")
    plt.title("DISPLACEMENT OVER TIME", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save to directory
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOT_DIR / "displacements.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("\nPlot displacements saved to:\n", save_path, "\n")


def generate_plots():
    """Generate all instrumentation plots."""
    plot_charge_log()
    plot_displacement_log()
