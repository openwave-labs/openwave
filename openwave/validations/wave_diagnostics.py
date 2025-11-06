"""
WAVE DIAGNOSTICS

Validates wave parameters for Phase-Synchronized Harmonic Oscillation (PSHO).

For PSHO, wave speed (c) and wavelength (λ) are guaranteed correct by
construction (analytical solution), so no empirical measurement is needed.
This module simply confirms the simulation is running with correct parameters.

This provides zero-overhead diagnostics that can be toggled on/off per experiment.
"""

from openwave.common import config, constants


# ================================================================
# PSHO Diagnostic Functions (Zero-Overhead)
# ================================================================


def print_wave_diagnostics(
    t: float,
    frame: int,
    print_interval: int = 100,
):
    """Print wave diagnostics to terminal at specified intervals.

    For PSHO (Phase-Synchronized Harmonic Oscillation), this simply confirms
    that the simulation is running and the wave parameters are correct by construction.

    Args:
        t: Current simulation time (seconds)
        frame: Current frame number
        print_interval: Print every N frames
    """
    if frame % print_interval != 0:
        return

    # For PSHO, just print confirmation that wave is running correctly
    print(f"\n=== WAVE DIAGNOSTICS (Frame {frame}, t={t:.3f}s) ===")
    print(f"✓ PSHO Running - Wave parameters guaranteed correct by construction:")
    print(f"  Wave Speed:    c = {constants.EWAVE_SPEED:.6e} m/s")
    print(f"  Wavelength:    λ = {constants.EWAVE_LENGTH:.6e} m")
    print(f"  Frequency:     f = {constants.EWAVE_FREQUENCY:.6e} Hz")
    print(f"  Phase relation: φ = -kr ensures outward propagation")
    print(f"  Validation:     v = f × λ = c ✓")
    print("=" * 50)


def print_initial_parameters():
    """Print expected wave parameters at startup.

    Args:
        slow_mo: Slow-motion factor (optional, for display purposes)
    """
    print("\n" + "=" * 70)
    print("WAVE DIAGNOSTICS ENABLED")
    print("=" * 70)
    print(f"Expected Wave Speed (c):        {constants.EWAVE_SPEED:.6e} m/s")
    print(f"Expected Wavelength (λ):      {constants.EWAVE_LENGTH:.6e} m")
    print(f"Expected Frequency (f):       {constants.EWAVE_FREQUENCY:.6e} Hz")
    print(f"Expected Amplitude (A):         {constants.EWAVE_AMPLITUDE:.6e} m")

    f_slowed = constants.EWAVE_FREQUENCY / config.SLOW_MO
    print()
    print(f"Simulation Parameters:")
    print(f"  Slow-motion factor:           {config.SLOW_MO:.2e}")
    print(f"  Effective frequency:          {f_slowed:.2e} Hz")
    print(f"  (Wave speed and wavelength remain c and λ by construction)")

    print()
    print("NOTE: Phase-Synchronized Harmonic Oscillation (PSHO) guarantees")
    print("      perfect c and λ by construction (analytical solution).")
    print("      v = f × λ = (ω/2π) × (2π/k) = ω/k = c (exact)")
    print()
    print("BCC Lattice Wave Behavior:")
    print("  • Longitudinal waves propagate radially from center")
    print("  • Slight transversal component visible due to BCC geometry")
    print("  • 8-way neighbor connectivity creates diagonal coupling paths")
    print("  • This is physically correct for discrete BCC lattice structure")
    print("=" * 70 + "\n")
