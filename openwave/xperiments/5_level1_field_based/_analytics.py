"""
DATA ANALYTICS

This provides zero-overhead analytics that can be toggled on/off per experiment.
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
_log_initialized = False


# ================================================================
# Analytics Functions (Zero-Overhead)
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
    print("DATA ANALYTICS ENABLED")
    print("=" * 64)
    print(f"\nPlot saved to: {save_path}")
    # plt.show()


def log_charge_level(frame: int, charge_level: float) -> None:
    """Record charge level at the current frame."""
    global _log_initialized

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DATA_DIR / "charge_level.csv"

    # Write header on first call
    if not _log_initialized:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "charge_level"])
        _log_initialized = True

    # Append charge level data
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([frame, f"{charge_level:.6f}"])


def plot_charge_log():
    """Plot the logged charge level over time."""
    log_path = DATA_DIR / "charge_level.csv"
    if not log_path.exists():
        print("Charge log file does not exist.")
        return

    frames = []
    charge_levels = []

    # Read logged data
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            charge_levels.append(float(row["charge_level"]))

    # Create the plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 6), facecolor=colormap.DARK_GRAY[1])
    fig.suptitle("OPENWAVE Analytics", fontsize=20, family="Monospace")

    plt.plot(
        frames,
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

    plt.xlabel("Frame", family="Monospace")
    plt.ylabel("Charge Level (%)", family="Monospace")
    plt.title("ENERGY CHARGING & STABILIZATION", family="Monospace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save to directory
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOT_DIR / "charge_levels.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("\nCharge level plot saved to:", save_path)
    print(f"")
    # plt.show()


if __name__ == "__main__":
    plot_charge_log()
