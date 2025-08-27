# =====================
# TODO: create Quantum-Wave class
# =====================

LENGTH = lambda_l = 2.854096501e-17  # m, quantum-wave length
AMPLITUDE = A_l = 9.215405708e-19  # m, quantum-wave amplitude (equilibrium-to-peak)
SPEED = c = 299792458  # m / s, quantum-wave velocity (speed of light)
DENSITY = rho = 3.859764540e22  # kg / m^3, quantum-wave medium density (aether)


def energy_wave_equation(volume, amplitude=None, wavelength=None):
    """
    Energy Wave Equation: E = ρV(c/λ_l * A)²

    The fundamental equation from which all EWT equations are derived.

    Args:
        volume (float): Volume V in m³
        amplitude (float, optional): Amplitude A in m. Defaults to QWAVE_AMPLITUDE
        wavelength (float, optional): Wavelength λ_l in m. Defaults to QWAVE_LENGTH

    Returns:
        float: Energy E in Joules

    Raises:
        ValueError: If volume is not positive, or if amplitude/wavelength are not positive
        TypeError: If inputs are not numeric
    """
    if not isinstance(volume, (int, float)):
        raise TypeError("Volume must be numeric")
    if volume <= 0:
        raise ValueError("Volume must be positive")

    if amplitude is None:
        amplitude = AMPLITUDE
    elif not isinstance(amplitude, (int, float)) or amplitude <= 0:
        raise ValueError("Amplitude must be positive numeric value")

    if wavelength is None:
        wavelength = LENGTH
    elif not isinstance(wavelength, (int, float)) or wavelength <= 0:
        raise ValueError("Wavelength must be positive numeric value")

    return DENSITY * volume * (SPEED / wavelength * amplitude) ** 2


if __name__ == "__main__":
    # Example usage
    print(f"Energy for 1 m³ volume: {energy_wave_equation(1.0):.2e} J")
