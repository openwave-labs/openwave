import time
import taichi as ti

ti.init(arch=ti.gpu)

# Create axis field (6 points = 3 axes)
axis_field = ti.Vector.field(3, dtype=ti.f32, shape=6)


@ti.kernel
def populate_axis():
    axis_field[0] = ti.Vector([0.0, 0.0, 0.0])
    axis_field[1] = ti.Vector([2.0, 0.0, 0.0])
    axis_field[2] = ti.Vector([0.0, 0.0, 0.0])
    axis_field[3] = ti.Vector([0.0, 2.0, 0.0])
    axis_field[4] = ti.Vector([0.0, 0.0, 0.0])
    axis_field[5] = ti.Vector([0.0, 0.0, 2.0])


populate_axis()


# Use dense particle sampling along axes instead
points_per_axis = 100
axis_field_particles = ti.Vector.field(3, dtype=ti.f32, shape=3 * points_per_axis)


@ti.kernel
def populate_axis_particles():
    for i in range(points_per_axis):
        t = i / (points_per_axis - 1)
        axis_field_particles[i] = ti.Vector([t * 2.0, 0.0, 0.0])  # X-axis
        axis_field_particles[points_per_axis + i] = ti.Vector([0.0, t * 2.0, 0.0])  # Y-axis
        axis_field_particles[2 * points_per_axis + i] = ti.Vector([0.0, 0.0, t * 2.0])  # Z-axis


populate_axis_particles()

# Simulate workload: 1.2M particles + physics + per-vertex colors
num_granules = 1200000  # Large granule count
granule_positions = ti.Vector.field(3, dtype=ti.f32, shape=num_granules)
granule_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_granules)
equilibrium = ti.Vector.field(3, dtype=ti.f32, shape=num_granules)


@ti.kernel
def init_granules():
    for i in range(num_granules):
        pos = ti.Vector([ti.random() * 1.0, ti.random() * 1.0, ti.random() * 1.0])
        granule_positions[i] = pos
        equilibrium[i] = pos
        granule_colors[i] = ti.Vector([0.1, 0.6, 0.9])


@ti.kernel
def simulate_physics(t: ti.f32):  # type: ignore
    """Simulate physics update for granules."""
    for i in range(num_granules):
        # Wave oscillation
        offset = ti.sin(t * 5.0 + equilibrium[i][0] * 10.0) * 0.02
        granule_positions[i] = equilibrium[i] + ti.Vector([0.0, 0.0, offset])
        # Per-vertex color (ironbow effect)
        displacement = ti.abs(offset)
        granule_colors[i] = ti.Vector([0.1 + displacement * 10.0, 0.6, 0.9])


init_granules()

window = ti.ui.Window("Lines vs Particles Test", (1200, 900), vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()  # GUI manager for overlay UI elements
scene = window.get_scene()  # Create scene once, reuse every frames
camera = ti.ui.Camera()

particles = False  # Start with lines
frames = 0  # frames counter for diagnostics
last_time = time.time()
elapsed_time = 0.0  # Time accumulator for diagnostics

# Set background once
canvas.set_background_color((0.1, 0.1, 0.15))  # Dark background

# Render loop
while window.running:
    # Calculate actual elapsed time (real-time tracking)
    current_time = time.time()
    dt_real = current_time - last_time
    last_time = current_time
    elapsed_time += dt_real  # Use real elapsed time instead of fixed DT
    fps = 0 if elapsed_time == 0 else frames / elapsed_time

    # Camera setup
    camera.position(2, 2, 2)
    camera.lookat(0, 0, 0)
    camera.up(0, 0, 1)
    scene.set_camera(camera)

    # Lighting (must be set each frames in GGUI)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

    # Physics computation
    simulate_physics(elapsed_time)

    # Render with per-vertex colors
    scene.particles(granule_positions, radius=0.002, per_vertex_color=granule_colors)

    with gui.sub_window("CONTROL", 0.78, 0.00, 0.22, 0.25) as sub:
        if sub.checkbox("Use Particles", particles):
            particles = True
        else:
            particles = False
        if sub.checkbox("Use Lines", not particles):
            particles = False
        else:
            particles = True
        sub.text(f"Frames Rendered: {frames}")
        sub.text(f"Elapsed Time: {elapsed_time:.2f}s")
        sub.text(f"Avg Frame-Rate: {fps:.0f} FPS")

    # Toggle between lines and particles to compare performance
    if particles:
        scene.particles(axis_field_particles, radius=0.002, color=(1, 1, 1))
    else:
        scene.lines(axis_field, color=(1, 1, 1), width=2)

    canvas.scene(scene)
    window.show()
    frames += 1  # Increment frames counter
