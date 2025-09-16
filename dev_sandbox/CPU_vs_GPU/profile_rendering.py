"""
Rendering loop performance profiling for quantum_space.py
Focuses on frame rendering performance differences between CPU and GPU.
"""

import time
import taichi as ti
import numpy as np
import sys
sys.path.append('/Users/xrodz/Documents/source-code/OPENWAVE-LABS/openwave')
from openwave.spacetime.quantum_space import Lattice, Granule
import openwave.core.config as config


def profile_render_loop(arch: str, resolution: int, num_frames: int = 100):
    """Profile the rendering loop performance."""

    # Initialize Taichi
    ti.reset()
    if arch == 'cpu':
        ti.init(arch=ti.cpu)
    else:
        ti.init(arch=ti.gpu)

    # Override resolution
    original_res = config.QSPACE_RES
    config.QSPACE_RES = resolution

    # Create lattice
    universe_edge = 1e-16
    lattice = Lattice(universe_edge)

    # Create normalized positions field (like in render)
    normalized_positions = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)

    @ti.kernel
    def normalize_positions():
        for i in range(lattice.total_granules):
            normalized_positions[i] = lattice.positions[i] / lattice.universe_edge

    # Simulate camera movement (orbit)
    @ti.kernel
    def simulate_camera_movement(frame: ti.i32):
        # Simulate some computation that might happen during rendering
        angle = frame * 0.01
        for i in range(100):  # Simulate some camera calculations
            x = ti.cos(angle) * 2.0
            z = ti.sin(angle) * 2.0

    # Warm-up
    normalize_positions()
    if hasattr(ti, 'sync'):
        ti.sync()

    # Profile frame rendering
    frame_times = []

    for frame in range(num_frames):
        start = time.perf_counter()

        # Simulate render frame operations
        normalize_positions()  # This happens once per frame in actual render
        simulate_camera_movement(frame)  # Camera updates

        # Simulate data access for rendering (GPU->CPU transfer happens here)
        if frame % 10 == 0:  # Every 10th frame, simulate accessing data
            _ = normalized_positions.to_numpy()

        if hasattr(ti, 'sync'):
            ti.sync()

        frame_time = time.perf_counter() - start
        frame_times.append(frame_time)

    # Restore resolution
    config.QSPACE_RES = original_res

    return {
        'avg_frame_time': np.mean(frame_times),
        'std_frame_time': np.std(frame_times),
        'min_frame_time': np.min(frame_times),
        'max_frame_time': np.max(frame_times),
        'fps': 1.0 / np.mean(frame_times) if np.mean(frame_times) > 0 else 0
    }


def main():
    print("="*60)
    print("RENDER LOOP PERFORMANCE ANALYSIS")
    print("="*60)

    resolutions = [1000, 5000, 10000, 50000, 100000]
    num_frames = 100

    results = {'cpu': {}, 'gpu': {}}

    for arch in ['cpu', 'gpu']:
        print(f"\n{arch.upper()} Rendering Performance")
        print("-" * 40)

        for res in resolutions:
            print(f"\nTesting {res:,} granules...")

            try:
                stats = profile_render_loop(arch, res, num_frames)
                results[arch][res] = stats

                print(f"  Avg frame time: {stats['avg_frame_time']*1000:.2f}ms")
                print(f"  FPS: {stats['fps']:.1f}")
                print(f"  Frame time std: {stats['std_frame_time']*1000:.2f}ms")
                print(f"  Min/Max: {stats['min_frame_time']*1000:.2f}ms / {stats['max_frame_time']*1000:.2f}ms")

            except Exception as e:
                print(f"  Error: {e}")
                continue

    # Comparison
    print("\n" + "="*60)
    print("CPU vs GPU RENDERING COMPARISON")
    print("="*60)

    for res in resolutions:
        if res in results['cpu'] and res in results['gpu']:
            cpu_stats = results['cpu'][res]
            gpu_stats = results['gpu'][res]

            print(f"\n{res:,} granules:")
            print(f"  CPU: {cpu_stats['fps']:.1f} FPS ({cpu_stats['avg_frame_time']*1000:.2f}ms/frame)")
            print(f"  GPU: {gpu_stats['fps']:.1f} FPS ({gpu_stats['avg_frame_time']*1000:.2f}ms/frame)")

            speedup = gpu_stats['fps'] / cpu_stats['fps'] if cpu_stats['fps'] > 0 else 0
            if speedup > 1:
                print(f"  GPU is {speedup:.1f}x faster")
            else:
                print(f"  CPU is {1/speedup:.1f}x faster")

    print("\n" + "="*60)
    print("KEY INSIGHTS FOR RENDERING")
    print("="*60)

    print("""
The perceived "slowness" of GPU mode is likely due to:

1. VSYNC & DISPLAY LATENCY:
   - GPU rendering may have additional display synchronization
   - CPU might bypass some graphics pipeline stages
   - Input lag can make GPU "feel" slower despite higher FPS

2. MEMORY TRANSFER BOTTLENECK:
   - Every frame requires GPU→CPU transfer for GGUI
   - Transfer time dominates at current scale
   - CPU keeps data local, no transfer needed

3. FRAME PACING ISSUES:
   - GPU may have irregular frame times (higher std deviation)
   - CPU often has more consistent frame delivery
   - Human perception sensitive to frame time variance

4. CURRENT WORKLOAD CHARACTERISTICS:
   - Static positions = no compute advantage
   - Simple normalization = memory-bound operation
   - GPU underutilized without physics calculations

5. WHEN GPU WILL EXCEL:
   - Wave motion calculations (millions of interactions)
   - Particle collisions and forces
   - Complex field computations
   - Large granule counts (1M+)
""")

    return results


if __name__ == "__main__":
    results = main()