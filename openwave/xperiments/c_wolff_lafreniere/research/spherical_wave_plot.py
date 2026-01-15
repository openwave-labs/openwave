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
k_am = 2 * np.pi / wavelength_am  # wave number in attometers

# ================================================================
# Amplitude Functions
# ================================================================


def sine_stand_wolff(r_am, A0_am=A0_am):
    """
    Far-field amplitude falloff: A(r) = A₀ · (λ/r)

    Energy conservation for spherical waves requires A ∝ 1/r
    Valid in far-field region (r > 2λ from wave source)

    Args:
        r_am: Distance from wave source (attometers)
        A0_am: Base amplitude A₀ (attometers)

    Returns:
        Amplitude at distance r (attometers)
    """

    sine = A0_am * np.abs(np.sin(k_am * r_am) / r_am)

    return sine


def sine_stand_lafreniere(r_am, A0_am=A0_am):
    """
    Far-field amplitude falloff: A(r) = A₀ · (λ/r)

    Energy conservation for spherical waves requires A ∝ 1/r
    Valid in far-field region (r > 2λ from wave source)

    Args:
        r_am: Distance from wave source (attometers)
        A0_am: Base amplitude A₀ (attometers)

    Returns:
        Amplitude at distance r (attometers)
    """

    sine = A0_am * np.sin(k_am * r_am) / (k_am * r_am)
    # sine = A0_am * (1 - np.cos(k_am * r_am)) / (k_am * r_am)  # quadrature version
    return sine


def sine_lafreniere_near(r_am, A0_am=A0_am):
    """
    Far-field amplitude falloff: A(r) = A₀ · (λ/r)

    Energy conservation for spherical waves requires A ∝ 1/r
    Valid in far-field region (r > 2λ from wave source)

    Args:
        r_am: Distance from wave source (attometers)
        A0_am: Base amplitude A₀ (attometers)

    Returns:
        Amplitude at distance r (attometers)
    """

    kr_am = k_am * r_am + (np.pi / 2) * (1 - k_am * r_am / np.pi) ** 2
    sine = A0_am * np.sin(kr_am) / kr_am

    return sine


def amp_with_safe(r_am, A0_am=A0_am):
    """
    Far-field amplitude falloff: A(r) = A₀ · (λ/r)

    Energy conservation for spherical waves requires A ∝ 1/r
    Valid in far-field region (r > 2λ from wave source)

    Args:
        r_am: Distance from wave source (attometers)
        A0_am: Base amplitude A₀ (attometers)

    Returns:
        Amplitude at distance r (attometers)
    """
    # Prevent division by zero
    r_reference = wavelength_am  # Reference radius = 1λ for near-field handling
    r_safe = np.maximum(np.abs(r_am), r_reference)  # in attometers
    amp_falloff = r_reference / r_safe
    amp = A0_am * amp_falloff

    return amp


def amp_1_over_r(r_am, A0_am=A0_am):
    """
    Far-field amplitude falloff: A(r) = A₀ · (λ/r)

    Energy conservation for spherical waves requires A ∝ 1/r
    Valid in far-field region (r > 2λ from wave source)

    Args:
        r_am: Distance from wave source (attometers)
        A0_am: Base amplitude A₀ (attometers)

    Returns:
        Amplitude at distance r (attometers)
    """

    amp = A0_am / r_am

    return amp


def amp_with_cap(r_am, A0_am=A0_am):
    """
    Amplitude with physical cap: A(r) ≤ r

    Implements the constraint from energy_wave_level0.py:229
    Prevents granules from crossing through the wave source.

    For longitudinal waves, displacement cannot exceed distance to source:
        |x - x_eq| ≤ |x_eq - x_source|
        A ≤ r

    Args:
        r_am: Distance from wave source (attometers)
        A0_am: Base amplitude A₀ (attometers)

    Returns:
        Capped amplitude at distance r (attometers)
    """
    # Calculate 1/r amplitude
    A_uncapped = amp_1_over_r(r_am, A0_am)

    # Apply cap: A ≤ r
    amp = np.minimum(np.abs(A_uncapped), np.abs(r_am))
    return amp


# ================================================================
# Create Plot
# ================================================================

# Distance range: 0 to r_max from wave source
r_max_lambda = 5  # Maximum distance in wavelengths (adjustable)
r_max_am = 120  # in attometers
y_max_am = 1.0  # in attometers
y_min_am = -0.25  # in attometers
r_am = np.linspace(-r_max_am, r_max_am, 1000)

# Create figure
fig, ax = plt.subplots(figsize=(16, 9))

# ================================================================
# Plot Amplitude Curves
# ================================================================
# Sine sine_stand_wolff
ax.plot(
    r_am,
    sine_stand_wolff(r_am),
    "orange",
    linewidth=2.5,
    alpha=0.8,
    label="sine: stand_wolff",
)

# Sine sine_stand_lafreniere
ax.plot(
    r_am,
    sine_stand_lafreniere(r_am),
    "cyan",
    linewidth=2.5,
    alpha=0.8,
    label="sine: stand_lafreniere",
)

# 1/lafreniere amplitude (without cap)
ax.plot(
    r_am,
    sine_lafreniere_near(r_am),
    "g--",
    linewidth=2.5,
    alpha=0.8,
    label="sine: lafreniere_near",
)

# 1/r amplitude (without cap)
ax.plot(
    r_am,
    amp_with_safe(r_am),
    "b-",
    linewidth=2.5,
    alpha=0.8,
    label="amp: safe at r_reference",
)

# Capped amplitude (actual implementation)
ax.plot(r_am, amp_with_cap(r_am), "r--", linewidth=3, label="amp: capped at A ≤ r")


plt.axhline(y=0, color=colormap.BLACK[1], linestyle="-", alpha=1)

# ================================================================
# Mark Near-Field / Far-Field Regions
# ================================================================

# Near-field region (r < λ from wave source)
# Source region where waves are forming
ax.axvspan(-1 * wavelength_am, 1.0 * wavelength_am, alpha=0.15, color="red")

# Transition zone (λ < r < 2λ)
# Wave fronts organizing into spherical geometry
ax.axvspan(-2.0 * wavelength_am, 2.0 * wavelength_am, alpha=0.15, color="yellow")
# Far-field region (r > 2λ)
# Fully formed spherical waves, clean 1/r falloff
ax.axvspan(-r_max_am, r_max_am, alpha=0.1, color="green")


# ================================================================
# Annotations
# ================================================================

# Amplitude at r = 1λ
A_at_1lambda_am = amp_with_cap(wavelength_am)
ax.plot(
    1 * wavelength_am,
    A_at_1lambda_am,
    "ro",
    markersize=10,
    label=f"A(1λ) = {A_at_1lambda_am/A0_am:.1f}A₀",
)

# Amplitude at r = 2λ
A_at_2lambda_am = amp_with_cap(2 * wavelength_am)
ax.plot(
    2.0 * wavelength_am,
    A_at_2lambda_am,
    "go",
    markersize=10,
    label=f"A(2λ) = {A_at_2lambda_am/A0_am:.1f}A₀",
)
# Annotate regions (adjusted for 2 am y-axis scale)
ax.text(
    0,
    y_max_am * 0.96,
    "Near\nField",
    ha="center",
    va="center",
    fontsize=11,
    weight="bold",
    color="darkred",
)
ax.text(
    1.5 * wavelength_am,
    y_max_am * 0.96,
    "Transition\nZone",
    ha="center",
    va="center",
    fontsize=11,
    weight="bold",
    color="orange",
)
ax.text(
    -1.5 * wavelength_am,
    y_max_am * 0.96,
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
    2.8 * wavelength_am,
    y_max_am * 0.96,
    "Far Field\n(Fully Formed Waves)",
    ha="center",
    va="center",
    fontsize=11,
    weight="bold",
    color="darkgreen",
)
far_field_center = (2.0 + r_max_lambda) / 2
ax.text(
    -2.8 * wavelength_am,
    y_max_am * 0.96,
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
    xy=(A0_am, A0_am),
    xytext=(A0_am * 2, A0_am - 0.1),
    arrowprops=dict(arrowstyle="->", lw=2, color="red"),
    fontsize=10,
    color="red",
)

# Annotate near field constraint
# Point to r = 1λ on the x-axis
# This is where the EWT neutrino boundary is located
#
ax.annotate(
    "Near field A₀ becomes active\n(also prevents singularity of 1/r)",
    xy=(1.1 * wavelength_am, A0_am),
    xytext=(1.2 * wavelength_am, A0_am - 0.1),
    arrowprops=dict(arrowstyle="->", lw=2, color="red"),
    fontsize=10,
    color="red",
)

# ================================================================
# Labels and Formatting
# ================================================================

ax.set_xlabel("Distance from Wave Source r_am (attometers)", fontsize=13, weight="bold")
ax.set_ylabel("Amplitude (attometers)", fontsize=13, weight="bold")
ax.set_title(
    "Spherical Wave Amplitude Falloff, f(radius)\nEnergy Conservation & Physical Constraints",
    fontsize=15,
    weight="bold",
    pad=20,
)

# Set axis limits
ax.set_xlim(-r_max_am, r_max_am)
ax.set_ylim(y_min_am, y_max_am)

# Set x-axis tick marks
ax.set_xticks(np.arange(-r_max_am, r_max_am, 10.0))
# Grid
ax.grid(True, alpha=0.3, linestyle="--")

# Legend
ax.legend(fontsize=10, framealpha=0.95)

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
    0.01,
    0.88,
    physics_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="left",
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
print(f"✓ Amplitude A₀ = {A0_am:.2f} am")
print(f"✓ Wavelength λ = {wavelength_am:.2f} am")
print(f"✓ Wave Number k = {k_am:.2f} rad/am\n")

plt.show()
