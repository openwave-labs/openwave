"""
AMPLITUDE vs RADIUS PLOT
Visualizes amplitude falloff behavior for spherical energy waves

Shows:
- Amplitude cap (A ≤ r constraint)
- 1/r far-field amplitude reduction
- Near-field vs far-field regions
- EWT neutrino boundary at r = 1λ

Notation (from 00_wave_properties.md):
- ρ (rho) = medium density
- c = wave speed (speed of light)
- λ (lambda) = wavelength
- A = wave amplitude
- f = frequency (c / λ)
- ω (omega) = angular frequency (2πf)
- k = wave number (2π/λ)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import actual OpenWave constants
from openwave.common import colormap, constants

# ================================================================
# Energy Wave Parameters (from constants.py)
# ================================================================

# Energy-wave constants (EWT specifications)
A0 = constants.EWAVE_AMPLITUDE  # m, energy-wave amplitude (A, equilibrium-to-peak)
wavelength = constants.EWAVE_LENGTH  # m, energy-wave length (λ)
k = 2 * np.pi / wavelength  # wave number (k = 2π/λ)

# Convert to attometers for plotting
A0_am = A0 / constants.ATTOMETER  # attometers
wavelength_am = wavelength / constants.ATTOMETER  # attometers

# ================================================================
# Amplitude Functions
# ================================================================


def sine_stand_wolff(r, A0_val=A0):
    """
    Far-field amplitude falloff: A(r) = A₀ · (λ/r)

    Energy conservation for spherical waves requires A ∝ 1/r
    Valid in far-field region (r > 2λ from wave source)

    Args:
        r: Distance from wave source (meters)
        A0_val: Base amplitude A₀ (meters)

    Returns:
        Amplitude at distance r (meters)
    """

    sine = A0_val * np.sin(k * r) / r * wavelength  # * λ for x axis in λ units

    return sine


def sine_stand_lafreniere(r, A0_val=A0):
    """
    Far-field amplitude falloff: A(r) = A₀ · (λ/r)

    Energy conservation for spherical waves requires A ∝ 1/r
    Valid in far-field region (r > 2λ from wave source)

    Args:
        r: Distance from wave source (meters)
        A0_val: Base amplitude A₀ (meters)

    Returns:
        Amplitude at distance r (meters)
    """

    sine = A0_val * np.sin(k * r) / (k * r)  # * λ for x axis in λ units
    # sine = A0_val * (1 - np.cos(k * r)) / (k * r)  # * λ for x axis in λ units

    return sine  # * 2 * np.pi  # Adjusted for proper scaling


def amp_1_over_lafreniere(r, A0_val=A0):
    """
    Far-field amplitude falloff: A(r) = A₀ · (λ/r)

    Energy conservation for spherical waves requires A ∝ 1/r
    Valid in far-field region (r > 2λ from wave source)

    Args:
        r: Distance from wave source (meters)
        A0_val: Base amplitude A₀ (meters)

    Returns:
        Amplitude at distance r (meters)
    """

    amp = (
        A0_val
        / ((k * r + (np.pi / 2) * (1 - k * r / np.pi) ** 2) / k)
        * wavelength  # * λ for x axis in λ units
    )

    return amp


def amp_1_over_r(r, A0_val=A0):
    """
    Far-field amplitude falloff: A(r) = A₀ · (λ/r)

    Energy conservation for spherical waves requires A ∝ 1/r
    Valid in far-field region (r > 2λ from wave source)

    Args:
        r: Distance from wave source (meters)
        A0_val: Base amplitude A₀ (meters)

    Returns:
        Amplitude at distance r (meters)
    """

    amp = A0_val / r * wavelength  # * λ for x axis in λ units

    return amp


def amp_with_safe(r, A0_val=A0):
    """
    Far-field amplitude falloff: A(r) = A₀ · (λ/r)

    Energy conservation for spherical waves requires A ∝ 1/r
    Valid in far-field region (r > 2λ from wave source)

    Args:
        r: Distance from wave source (meters)
        A0_val: Base amplitude A₀ (meters)

    Returns:
        Amplitude at distance r (meters)
    """
    # Prevent division by zero
    r_reference = wavelength  # Reference radius = 1λ for near-field handling
    r_safe = np.maximum(np.abs(r), r_reference)  # in meters
    amp_falloff = r_reference / r_safe  # * λ for x axis in λ units
    amp = A0_val * amp_falloff

    return amp


def amp_with_cap(r, A0_val=A0):
    """
    Amplitude with physical cap: A(r) ≤ r

    Implements the constraint from energy_wave_level0.py:229
    Prevents granules from crossing through the wave source.

    For longitudinal waves, displacement cannot exceed distance to source:
        |x - x_eq| ≤ |x_eq - x_source|
        A ≤ r

    Args:
        r: Distance from wave source (meters)
        A0_val: Base amplitude A₀ (meters)

    Returns:
        Capped amplitude at distance r (meters)
    """
    # Calculate 1/r amplitude
    A_uncapped = amp_1_over_r(r, A0_val)

    # Apply cap: A ≤ r
    amp = np.minimum(np.abs(A_uncapped), np.abs(r))

    return amp


# ================================================================
# Create Plot
# ================================================================

# Distance range: 0 to r_max from wave source
r_max_lambda = 5  # Maximum distance in wavelengths (adjustable)
r_max = r_max_lambda * wavelength
r = np.linspace(-r_max, r_max, 1000)

# Create figure
fig, ax = plt.subplots(figsize=(16, 9))

# ================================================================
# Plot Amplitude Curves
# ================================================================

# Convert to attometers for y-axis
sine_stand_wolff_am = sine_stand_wolff(r) / constants.ATTOMETER
sine_stand_lafreniere_am = sine_stand_lafreniere(r) / constants.ATTOMETER
amp_1_over_lafreniere_am = amp_1_over_lafreniere(r) / constants.ATTOMETER
amp_with_safe_am = amp_with_safe(r) / constants.ATTOMETER
amp_capped_am = amp_with_cap(r) / constants.ATTOMETER
r_am = r / constants.ATTOMETER

# Sine sine_stand_wolff
ax.plot(
    r / wavelength,
    sine_stand_wolff_am,
    "orange",
    linewidth=2.5,
    alpha=0.8,
    label="sine: stand_wolff",
)

# Sine sine_stand_lafreniere
ax.plot(
    r / wavelength,
    sine_stand_lafreniere_am,
    "cyan",
    linewidth=2.5,
    alpha=0.8,
    label="sine: stand_lafreniere",
)

# 1/lafreniere amplitude (without cap)
ax.plot(
    r / wavelength,
    amp_1_over_lafreniere_am,
    "g--",
    linewidth=2.5,
    alpha=0.8,
    label="amp: 1/lafreniere",
)

# 1/r amplitude (without cap)
ax.plot(
    r / wavelength,
    amp_with_safe_am,
    "b-",
    linewidth=2.5,
    alpha=0.8,
    label="amp: safe at r_reference",
)

# Capped amplitude (actual implementation)
ax.plot(r / wavelength, amp_capped_am, "r--", linewidth=3, label="amp: capped at A ≤ r")


plt.axhline(y=0, color=colormap.BLACK[1], linestyle="-", alpha=1)

# ================================================================
# Mark Near-Field / Far-Field Regions
# ================================================================

# Near-field region (r < λ from wave source)
# Source region where waves are forming
ax.axvspan(-1.0, 1.0, alpha=0.15, color="red")

# Transition zone (λ < r < 2λ)
# Wave fronts organizing into spherical geometry
ax.axvspan(-2.0, 2.0, alpha=0.15, color="yellow")

# Far-field region (r > 2λ)
# Fully formed spherical waves, clean 1/r falloff
ax.axvspan(-r_max / wavelength, r_max / wavelength, alpha=0.1, color="green")


# ================================================================
# Annotations
# ================================================================

# Amplitude at r = 1λ
A_at_1lambda = amp_with_cap(wavelength)
A_at_1lambda_am = A_at_1lambda / constants.ATTOMETER
ax.plot(1.0, A_at_1lambda_am, "ro", markersize=10, label=f"A(1λ) = {A_at_1lambda/A0:.1f}A₀")

# Amplitude at r = 2λ
A_at_2lambda = amp_with_cap(2 * wavelength)
A_at_2lambda_am = A_at_2lambda / constants.ATTOMETER
ax.plot(2.0, A_at_2lambda_am, "go", markersize=10, label=f"A(2λ) = {A_at_2lambda/A0:.1f}A₀")

# Annotate regions (adjusted for 2 am y-axis scale)
ax.text(
    0.5,
    5.5,
    "Near\nField",
    ha="center",
    va="center",
    fontsize=11,
    weight="bold",
    color="darkred",
)
ax.text(
    1.5,
    5.5,
    "Transition\nZone",
    ha="center",
    va="center",
    fontsize=11,
    weight="bold",
    color="orange",
)
# Far field label position: centered between 2λ and r_max
far_field_center = (2.0 + r_max_lambda) / 2
ax.text(
    2.8,
    5.5,
    "Far Field\n(Fully Formed Waves)",
    ha="center",
    va="center",
    fontsize=11,
    weight="bold",
    color="darkgreen",
)

# Annotate cap constraint (adjusted for attometer scale)
# The uncapped line (blue) diverges from capped line (red) at r ≈ 0.18λ
# Point to where you can see the blue line going above the red line
# This is around r = 0.15λ where A_uncapped starts exceeding r
ax.annotate(
    "Cap becomes active\n(A > r prevented)",
    xy=(0.12, 1.1),
    xytext=(0.4, 1.4),
    arrowprops=dict(arrowstyle="->", lw=2, color="red"),
    fontsize=10,
    color="red",
    weight="bold",
)

# Annotate near field constraint
# Point to r = 1λ on the x-axis
# This is where the EWT neutrino boundary is located
#
ax.annotate(
    "Near field A₀ becomes active\n(also prevents singularity of 1/r)",
    xy=(1.1, 0.95),
    xytext=(1.1, 1.1),
    arrowprops=dict(arrowstyle="->", lw=2, color="red"),
    fontsize=10,
    color="red",
    weight="bold",
)

# ================================================================
# Labels and Formatting
# ================================================================

ax.set_xlabel("Distance from Wave Source (r/λ)", fontsize=13, weight="bold")
ax.set_ylabel("Amplitude (attometers)", fontsize=13, weight="bold")
ax.set_title(
    "Spherical Wave Amplitude Falloff, f(radius)\nEnergy Conservation & Physical Constraints",
    fontsize=15,
    weight="bold",
    pad=20,
)

# Set axis limits
ax.set_xlim(-r_max / wavelength, r_max / wavelength)
ax.set_ylim(-1.5, 6.0)  # 6 attometers max

# Set x-axis tick marks at 1λ spacing
ax.set_xticks(np.arange(-r_max / wavelength, r_max / wavelength + 1, 1.0))

# Grid
ax.grid(True, alpha=0.3, linestyle="--")

# Legend
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

# ================================================================
# Add Physics Equations as Text Box
# ================================================================

physics_text = (
    "Energy Conservation:\n"
    "  E_total = ρV(c/λ × A)²\n"
    "  A ∝ 1/r for spherical waves\n\n"
    "Implementation:\n"
    "  A(r) = A₀ · (λ/r) · boost\n"
    "  A_final = min(A(r), r)\n\n"
    "Constraints:\n"
    "  • r_min = 1λ (EWT neutrino boundary)\n"
    "  • A ≤ r (prevents source crossing)\n"
    "  • Far-field: r > 2λ"
)

props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(
    0.99,
    0.78,
    physics_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=props,
    family="monospace",
)

# ================================================================
# Save and Display
# ================================================================

plt.tight_layout()

# Use relative path from script location
script_dir = Path(__file__).parent
output_path = script_dir / "plots" / "wave_amplitude_vs_radius.png"

plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n✓ Plot saved: {output_path}")
print(f"✓ Using A₀ = {A0:.3e} m")
print(f"✓ Using λ = {wavelength:.3e} m\n")

plt.show()
