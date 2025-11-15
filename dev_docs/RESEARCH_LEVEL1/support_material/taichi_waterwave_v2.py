"""
2D Wave Propagation Simulation with Displacement-Based Color Visualization

This simulation implements the classical 2D wave equation: ψ'' = c² ∇²ψ
where:
    ψ = displacement field (wave height/amplitude)
    ψ'' = ∂²ψ/∂t² (acceleration, second time derivative)
    c = wave propagation speed
    ∇²ψ = Laplacian (spatial curvature, second spatial derivative)

This is the fundamental equation for wave propagation in elastic media, including:
- Waves on strings and membranes
- Sound waves in fluids
- Electromagnetic waves (in vacuum)
- Shallow water waves (approximation)

Physical Behavior:
- Wave equation is linear and energy-conserving (with damping=0)
- Circular waves naturally exhibit amplitude decay ~ 1/√r due to geometric spreading
  (energy conservation: total energy spreads over expanding circumference 2πr)
- Additional amplitude decay occurs from numerical dissipation in discrete simulation
- Dirichlet boundary conditions: fixed ψ=0 at domain edges (rigid wall reflection)

Visualization:
- Blue: positive displacement (wave peaks)
- Red: negative displacement (wave troughs)
- Black: zero displacement (wave nodes)
- Power curve (exponent 0.3) enhances visibility of low-amplitude regions
- Fixed amplitude reference for color normalization causes older waves to appear dimmer

Alternative visualization approach (Fresnel-like shading):
- Calculate gradient magnitude to identify wave fronts (steep regions)
- brightness = (1 - cos_i)² where cos_i = 1/√(1 + |∇ψ|²)
- Makes wave fronts appear bright regardless of displacement sign
- Trade-off: loses direct displacement information but highlights wave structure

References:
- https://en.wikipedia.org/wiki/Wave_equation
- https://en.wikipedia.org/wiki/Laplace_operator
"""

import taichi as ti

ti.init(arch=ti.gpu)

# Simulation parameters
light_color = [1.0, 1.0, 1.0]  # unused in current visualization (legacy from Fresnel mode)
background = [0.0, 0.0, 0.0]  # background color (black)
wave_speed = 5.0  # wave propagation speed (c² in wave equation ψ'' = c² ∇²ψ)
amplitude = 50.0  # initial wave amplitude for source AND color normalization reference
damping = 0.0  # velocity damping coefficient (0 = no energy loss, >0 = exponential decay)
dx = 0.02  # spatial grid spacing
dt = 0.01  # time step for integration
shape = 800, 800  # simulation grid size (pixels)

# Field definitions
pixels = ti.Vector.field(3, dtype=float, shape=shape)  # RGB color output
displacement = ti.field(dtype=float, shape=shape)  # wave displacement field ψ
velocity = ti.field(dtype=float, shape=shape)  # time derivative of displacement ψ'


@ti.kernel
def reset():
    """Reset all wave fields to zero (clear the simulation)."""
    for i, j in displacement:
        displacement[i, j] = 0
        velocity[i, j] = 0


@ti.kernel
def charge_wave(amplitude: ti.f32, x: ti.f32, y: ti.f32):  # type: ignore
    """
    Create a wave source at position (x, y) with given amplitude.

    Uses a Gaussian spatial profile: ψ(r) = A * exp(-0.02 * r²)
    This creates a localized positive displacement (bump) that propagates outward.

    Note: Only creates positive initial displacement, so the negative phase
    (trough) develops naturally through wave equation dynamics, creating the
    characteristic red ring that follows the blue peak.

    Args:
        amplitude: Peak displacement magnitude at the source center
        x, y: Source position in grid coordinates
    """
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        r2 = (i - x) ** 2 + (j - y) ** 2
        displacement[i, j] += amplitude * ti.exp(-0.02 * r2)


@ti.func
def laplacian(i, j):
    """
    Compute the discrete Laplacian ∇²ψ at grid point (i, j).

    Uses 5-point stencil (4 neighbors + center):
    ∇²ψ ≈ (ψ_left + ψ_right + ψ_up + ψ_down - 4*ψ_center) / (4*dx²)

    This operator measures the local curvature of the displacement field.
    Positive curvature → positive acceleration (upward)
    Negative curvature → negative acceleration (downward)

    Boundary interaction:
    - Called only for interior points (excludes boundaries at i=0, i=max, j=0, j=max)
    - Reads neighbor values at boundaries (which are always 0)
    - This implements Dirichlet boundary conditions (fixed ψ=0 at edges)

    Note: Does NOT explicitly account for geometric spreading (1/√r decay).
    Amplitude decay emerges naturally from energy conservation as waves expand.

    Returns:
        Laplacian value (spatial curvature) at point (i, j)
    """
    return (
        displacement[i, j - 1]
        + displacement[i, j + 1]
        + displacement[i + 1, j]
        + displacement[i - 1, j]
        - 4 * displacement[i, j]
    ) / (4 * dx**2)


@ti.kernel
def propagate():
    """
    Propagate the wave field forward in time by one step dt.

    Implements the 2D wave equation: ψ'' = c² ∇²ψ - γ ψ'
    where:
        ψ'' = acceleration (second-order derivative of displacement with respect to time)
        c² = wave_speed (controls propagation speed)
        ∇²ψ = laplacian(i, j) (spatial curvature)
        γ = damping coefficient (energy dissipation)

    Integration scheme (Euler Integration):
        1. Compute acceleration from spatial curvature and damping
        2. Update velocity: v(t+dt) = v(t) + a(t)*dt
        3. Update displacement: ψ(t+dt) = ψ(t) + v(t+dt)*dt

    Boundary conditions (Dirichlet):
    - Propagation loop: ti.ndrange((1, shape[0]-1), (1, shape[1]-1))
    - Only updates interior points (excludes boundaries at i=0, i=max, j=0, j=max)
    - Boundaries remain fixed at ψ=0 (never updated)
    - This creates rigid walls where waves reflect back into the domain
    - Interior points compute Laplacian using boundary values (always 0)

    Physical interpretation:
    - Regions with positive curvature accelerate upward (like stretched string)
    - Regions with negative curvature accelerate downward (like compressed string)
    - Wave energy naturally spreads outward in expanding circles
    - Waves reflect at boundaries due to fixed zero displacement constraint
    """
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        acceleration = wave_speed * laplacian(i, j) - damping * velocity[i, j]
        velocity[i, j] += acceleration * dt
        displacement[i, j] += velocity[i, j] * dt


@ti.func
def gradient(i, j):
    """
    Compute the spatial gradient ∇ψ at grid point (i, j).

    Uses centered finite differences:
    ∂ψ/∂x ≈ (ψ(i+1,j) - ψ(i-1,j)) / (2*dx)
    ∂ψ/∂y ≈ (ψ(i,j+1) - ψ(i,j-1)) / (2*dx)

    The gradient magnitude |∇ψ| indicates wave steepness:
    - High gradient = steep wave front (rapid spatial change)
    - Low gradient = flat region (slow spatial change)

    Boundary handling:
    - Can be called on any point, including boundaries
    - At boundaries, uses neighbor values (including fixed ψ=0 at edges)
    - No special boundary conditionals needed

    Used in Fresnel-like visualization to identify and highlight wave fronts.
    Not used in current displacement-based visualization.

    Returns:
        2D vector [∂ψ/∂x, ∂ψ/∂y]
    """
    return ti.Vector(
        [
            (displacement[i + 1, j]) - (displacement[i - 1, j]),
            (displacement[i, j + 1]) - (displacement[i, j - 1]),
        ]
    ) * (0.5 / dx)


@ti.kernel
def visualize_wave():
    """
    Render wave field using displacement-based RGB color mapping.

    Color scheme:
    - Blue (RGB: 0.3, 0.5, 1.0): positive displacement (peaks)
    - Red (RGB: 1.0, 0.4, 0.3): negative displacement (troughs)
    - Black (RGB: 0, 0, 0): zero displacement (nodes)

    Visualization pipeline:
    1. Normalize displacement by reference amplitude (line 37)
    2. Apply power curve (exponent 0.3) to enhance visibility of low values
       - Makes dim waves more visible without saturating bright ones
       - Exponent < 1 boosts low values more than high values
    3. Map to RGB: multiply normalized intensity by color vector

    Why waves appear dimmer over time:
    - Actual amplitude decays due to geometric spreading (energy over 2πr)
    - Fixed amplitude reference (50.0) means decayed waves map to lower intensity
    - Fresh waves start at full amplitude and appear brightest
    - This is physically correct behavior, not a bug!

    Alternative approach (gradient-based/Fresnel shading):
    - Use gradient magnitude to compute brightness: b = (1 - cos_i)²
    - Highlights wave fronts regardless of displacement sign
    - Loses displacement information but shows wave structure clearly
    - See taichi_waterwave_v1.py for reference implementation
    """
    for i, j in pixels:
        d = displacement[i, j]

        # Normalize displacement to [-1, 1] range using fixed reference amplitude
        normalized_disp = ti.math.clamp(d / amplitude, -1.0, 1.0)

        # Apply power curve to enhance visibility while preserving sign
        sign = 1.0 if normalized_disp >= 0 else -1.0
        abs_disp = ti.abs(normalized_disp)
        intensity = ti.pow(abs_disp, 0.3) * sign  # power < 1 boosts low values

        # Define color palette
        blue_color = ti.Vector([0.3, 0.5, 1.0])  # positive displacement
        red_color = ti.Vector([1.0, 0.4, 0.3])  # negative displacement

        # Map intensity to color based on displacement sign
        if intensity >= 0:
            pixels[i, j] = intensity * blue_color
        else:
            pixels[i, j] = (-intensity) * red_color


def main():
    """
    Main simulation loop with interactive GUI.

    Controls:
    - Left mouse button: Create wave at cursor position
    - 'r' key: Reset simulation (clear all waves)
    - ESC or close window: Exit

    Loop structure:
    1. Handle user input events
    2. Propagate wave physics (update displacement and velocity fields)
    3. Render visualization (map displacement to RGB colors)
    4. Display frame to GUI window
    """
    print("\n================================================================")
    print("Taichi Water Wave Simulation v2")
    print("Controls:")
    print("  - Click to create waves")
    print("  - Press 'r' to reset")
    print("  - ESC to exit")
    print("================================================================\n")

    reset()
    gui = ti.GUI("Water Wave", shape)
    while gui.running:
        # Handle user input
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                gui.running = False
            elif e.key == "r":
                reset()
            elif e.key == ti.GUI.LMB:
                x, y = e.pos
                charge_wave(amplitude, x * shape[0], y * shape[1])

        # Update physics and render
        propagate()
        visualize_wave()
        gui.set_image(pixels)
        gui.show()


if __name__ == "__main__":
    main()
