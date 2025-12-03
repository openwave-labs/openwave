"""
DATA ANALYTICS

This provides zero-overhead analytics that can be toggled on/off per experiment.
"""

from openwave.common import colormap


# ================================================================
# Analytics Functions (Zero-Overhead)
# ================================================================


def plot_static_charge_profile(wave_field):
    """
    Plot the displacement profile along the x-axis through the center of the wave field.

    Args:
        wave_field: WaveField instance containing displacement data
    """
    import matplotlib.pyplot as plt
    import numpy as np

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

    # Save to research directory
    from pathlib import Path

    save_path = Path(__file__).parent / "_plots" / "charge_profile.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("\n" + "=" * 64)
    print("DATA ANALYTICS ENABLED")
    print("=" * 64)
    print(f"\nPlot saved to: {save_path}")
    # plt.show()


def log_charge_level(frame, charge_level):
    """
    Record charge level at the current frame.

    Args:
        frame: Current simulation frame
        charge_level: Current charge level
    """
    from pathlib import Path

    log_path = Path(__file__).parent / "_data" / "charge_level.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as log_file:
        log_file.write(f"{frame}\t{charge_level:.6f}\n")
