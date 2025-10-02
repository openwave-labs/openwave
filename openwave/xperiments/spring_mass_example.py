"""
3D Cloth Simulation with Angled Initial Position

This script simulates a piece of cloth draped at an angle using a mass-spring system.
Key features:
- Cloth starts at a 45-degree angle (not horizontal)
- Mass-spring model with 8-way connectivity (including diagonals)
- Collision detection with a spherical obstacle
- Simplified physics focusing on essential cloth behavior
- Real-time visualization using Taichi's GGUI

The simulation demonstrates how cloth naturally drapes and collides with obstacles,
starting from an angled initial configuration for more interesting dynamics.

added comprehensive inline comments and docstrings to the second 3D cloth simulation script. The documentation now explains:

  1. Module-level docstring: Describes the cloth simulation with angled initial position
  2. Variable comments: Explains all simulation parameters and their effects
  3. Function docstrings: Documents each function's purpose and implementation
  4. Inline comments: Clarifies the physics calculations and algorithms
  5. Key differences from the first cloth simulation:
    - Angled initial position (45-degree angle)
    - Simplified collision handling (stops particles rather than projection)
    - Different spring connectivity pattern explanation
    - Clear phase separation in the physics step

  Key improvements include:
  - Explained the angled initial configuration and its purpose
  - Documented the 8-way spring connectivity pattern
  - Clarified Hooke's law implementation for spring forces
  - Detailed the three-phase physics simulation approach
  - Explained boundary clamping for neighbor indices
  - Added context for the simplified collision response
  - Documented the rendering pipeline and camera setup
"""

import taichi as ti

# Initialize Taichi with GPU acceleration
ti.init(arch=ti.gpu)  # Alternatively, ti.init(arch=ti.cpu)

# Grid and simulation parameters
N = 128  # Grid resolution (N x N particles)
cell_size = 1.0 / N  # Distance between adjacent particles
gravity = 1  # Gravity strength (simplified units)
stiffness = 1200  # Spring stiffness coefficient
damping = 2  # Velocity damping factor for stability
dt = 6e-4  # Time step for numerical integration

# Obstacle parameters
ball_radius = 0.2  # Radius of the spherical obstacle
ball_center = ti.Vector.field(3, float, (1,))  # Position of sphere center

# Particle state fields
x = ti.Vector.field(3, float, (N, N))  # 3D position of each particle
v = ti.Vector.field(3, float, (N, N))  # 3D velocity of each particle

# Mesh rendering data
num_triangles = (N - 1) * (N - 1) * 2  # Two triangles per grid cell
indices = ti.field(int, num_triangles * 3)  # Triangle vertex indices (flattened)
vertices = ti.Vector.field(3, float, N * N)  # Vertex buffer for rendering


def init_scene():
    """
    Initialize the cloth in an angled position and set obstacle location.

    The cloth starts at a 45-degree angle:
    - Extends along X axis (horizontal)
    - Y and Z coordinates create the angle (Y increases, Z decreases with j)
    - This creates a more dynamic initial configuration than a flat horizontal cloth
    """
    for i, j in ti.ndrange(N, N):
        # Position particles at 45-degree angle using 1/sqrt(2) for equal Y and Z components
        x[i, j] = ti.Vector(
            [
                i * cell_size,  # X: horizontal position
                j * cell_size / ti.sqrt(2),  # Y: rising with j (upward component)
                (N - j) * cell_size / ti.sqrt(2),  # Z: decreasing with j (forward component)
            ]
        )
    # Place obstacle below and slightly behind the cloth's initial position
    ball_center[0] = ti.Vector([0.5, -0.5, -0.0])


@ti.kernel
def set_indices():
    """
    Create triangle mesh topology for rendering.

    Each grid cell is split into two triangles:
    - Triangle 1: (i,j), (i+1,j), (i,j+1)
    - Triangle 2: (i+1,j+1), (i,j+1), (i+1,j)

    This creates a consistent triangulation of the cloth surface.
    """
    for i, j in ti.ndrange(N, N):
        if i < N - 1 and j < N - 1:
            square_id = (i * (N - 1)) + j  # Unique ID for this grid cell

            # First triangle (lower-left of the square)
            indices[square_id * 6 + 0] = i * N + j
            indices[square_id * 6 + 1] = (i + 1) * N + j
            indices[square_id * 6 + 2] = i * N + (j + 1)

            # Second triangle (upper-right of the square)
            indices[square_id * 6 + 3] = (i + 1) * N + j + 1
            indices[square_id * 6 + 4] = i * N + (j + 1)
            indices[square_id * 6 + 5] = (i + 1) * N + j


# Define spring connectivity pattern (8-way: cardinal + diagonal neighbors)
# Each particle connects to up to 8 neighbors:
# - 4 cardinal directions: left, right, up, down
# - 4 diagonal directions: corners
links = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]
links = [ti.Vector(v) for v in links]  # Convert to Taichi vectors for kernel use


@ti.kernel
def step():
    """
    Perform one physics simulation step.

    The simulation proceeds in three phases:
    1. Apply gravity to all particles
    2. Calculate and apply spring forces between connected particles
    3. Apply damping, handle collisions, and update positions
    """
    # Phase 1: Apply gravity (only affects Y component)
    for i in ti.grouped(x):
        v[i].y -= gravity * dt

    # Phase 2: Calculate spring forces
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])  # Force accumulator

        # Loop through all connected neighbors
        for d in ti.static(links):
            # Calculate neighbor index with boundary clamping
            j = min(max(i + d, 0), [N - 1, N - 1])  # Clamp to grid bounds

            # Spring force calculation
            relative_pos = x[j] - x[i]  # Vector from current to neighbor
            current_length = relative_pos.norm()  # Current spring length

            # Rest length based on grid topology
            original_length = cell_size * float(i - j).norm()

            if original_length != 0:
                # Hooke's law: F = k * (extension) * direction
                # Extension ratio: (current - rest) / rest
                force += (
                    stiffness
                    * relative_pos.normalized()  # Direction
                    * (current_length - original_length)  # Extension
                    / original_length  # Normalize by rest length
                )

        # Apply spring forces
        v[i] += force * dt

    # Phase 3: Damping, collision, and position update
    for i in ti.grouped(x):
        # Apply exponential velocity damping for stability
        v[i] *= ti.exp(-damping * dt)

        # Sphere collision: stop particles inside the ball
        if (x[i] - ball_center[0]).norm() <= ball_radius:
            v[i] = ti.Vector([0.0, 0.0, 0.0])  # Zero velocity on collision

        # Update position using forward Euler integration
        x[i] += dt * v[i]


@ti.kernel
def set_vertices():
    """
    Copy particle positions to the rendering vertex buffer.

    Converts 2D grid indices to 1D array for mesh rendering.
    """
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = x[i, j]


# Initialize simulation
init_scene()  # Set initial particle positions
set_indices()  # Create mesh topology

# Create visualization window and components
window = ti.ui.Window("Cloth", (800, 800), vsync=True)  # 800x800 window with VSync
canvas = window.get_canvas()  # Canvas for rendering
scene = ti.ui.Scene()  # 3D scene container
camera = ti.ui.make_camera()  # Camera for 3D view

# Main simulation loop
# Calculate actual simulation parameters
substeps = 30
time_per_frame = dt * substeps  # Total simulated time per rendered frame
target_fps = 60  # Assumed target FPS (VSync typically targets 60 Hz)
time_scale = time_per_frame * target_fps  # How much faster/slower than real-time

print(f"Target frame-rate: {target_fps} FPS (VSync enabled)")
print(f"Simulated time per frame: {time_per_frame:.4f} seconds")
print(f"Time scale: {time_scale:.2f}x real-time")

while window.running:
    # Run multiple physics substeps per frame for stability
    # substeps provides good balance between accuracy and performance
    for i in range(substeps):
        step()

    # Update rendering data
    set_vertices()

    # Configure camera to view the draped cloth
    camera.position(0.5, -0.5, 2)  # Camera position: centered, slightly below, back
    camera.lookat(0.5, -0.5, 0)  # Look at center of scene
    scene.set_camera(camera)

    # Set up lighting
    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))  # Main light from above

    # Render cloth mesh (gray, two-sided for visibility from all angles)
    scene.mesh(vertices, indices=indices, color=(0.5, 0.5, 0.5), two_sided=True)

    # Render collision sphere (red)
    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0, 0))

    # Display the rendered scene
    canvas.scene(scene)
    window.show()
