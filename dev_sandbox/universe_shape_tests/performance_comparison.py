"""
Performance comparison between spherical and cubic quantum space implementations.
"""

import time
import psutil
import numpy as np
import taichi as ti
from memory_profiler import memory_usage

# Import both implementations
from dev_sandbox.spacetime_tests.qspace_sphere import Lattice as SphericalLattice
from openwave.spacetime.quantum_space import Lattice as CubicLattice
import openwave.core.constants as constants


def measure_initialization(lattice_class, size_param, name):
    """Measure initialization time and memory for lattice creation."""
    print(f"\n{'='*60}")
    print(f"Testing {name} Implementation")
    print(f"{'='*60}")

    # Measure memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Measure initialization time
    start_time = time.perf_counter()
    lattice = lattice_class(size_param)
    end_time = time.perf_counter()

    init_time = end_time - start_time

    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before

    # Get statistics
    stats = lattice.get_stats()

    return {
        "name": name,
        "init_time": init_time,
        "memory_used": mem_used,
        "stats": stats,
        "lattice": lattice,
    }


def measure_operations(lattice, name):
    """Measure performance of common operations."""
    print(f"\nOperation Performance for {name}:")

    results = {}

    # Test position update
    start = time.perf_counter()
    for _ in range(100):
        lattice.update_positions(0.001)
    end = time.perf_counter()
    results["position_update_100x"] = end - start
    print(f"  Position update (100 iterations): {results['position_update_100x']:.4f} seconds")

    # Test data access
    start = time.perf_counter()
    positions = lattice.positions.to_numpy()
    end = time.perf_counter()
    results["numpy_conversion"] = end - start
    print(f"  NumPy conversion: {results['numpy_conversion']:.4f} seconds")

    # Test zone access (if available)
    if hasattr(lattice, "zone_type"):
        start = time.perf_counter()
        zones = lattice.zone_type.to_numpy()
        end = time.perf_counter()
        results["zone_access"] = end - start
        print(f"  Zone data access: {results['zone_access']:.4f} seconds")

    return results


def compare_memory_efficiency(spherical_stats, cubic_stats):
    """Compare memory efficiency between implementations."""
    print("\n" + "=" * 60)
    print("Memory Efficiency Comparison")
    print("=" * 60)

    spherical_granules = spherical_stats["total_granules"]
    cubic_granules = cubic_stats["total_granules"]

    # For cubic, corner and center granules are provided
    if "corner_granules" in cubic_stats:
        cubic_total = cubic_stats["corner_granules"] + cubic_stats["center_granules"]
        print(f"Cubic total potential: {cubic_total:,}")

    print(f"Spherical granules: {spherical_granules:,}")
    print(f"Cubic granules: {cubic_granules:,}")
    print(
        f"Spherical advantage: {(1 - spherical_granules/cubic_granules) * 100:.1f}% fewer granules"
    )

    # Active vs total efficiency
    if "active_granules" in spherical_stats:
        active_ratio = spherical_stats["active_granules"] / spherical_granules
        print(f"\nSpherical zone distribution:")
        print(f"  Active zone: {spherical_stats['active_granules']:,} ({active_ratio*100:.1f}%)")
        print(
            f"  Buffer zone: {spherical_stats['buffer_granules']:,} ({(1-active_ratio)*100:.1f}%)"
        )


def main():
    """Run performance comparison."""
    print("=" * 60)
    print("QUANTUM SPACE PERFORMANCE COMPARISON")
    print("Spherical vs Cubic Implementation")
    print("=" * 60)

    # Test parameters
    universe_radius = 0.5e-15  # 0.5 femtometer
    universe_edge = 1e-15  # 1 femtometer cube (2x radius for comparison)

    # Initialize Taichi
    ti.init(arch=ti.gpu, kernel_profiler=True)

    # Test spherical implementation
    spherical_results = measure_initialization(SphericalLattice, universe_radius, "SPHERICAL")

    # Test cubic implementation
    cubic_results = measure_initialization(CubicLattice, universe_edge, "CUBIC")

    # Print initialization comparison
    print("\n" + "=" * 60)
    print("Initialization Performance Summary")
    print("=" * 60)
    print(f"Spherical initialization: {spherical_results['init_time']:.4f} seconds")
    print(f"Cubic initialization: {cubic_results['init_time']:.4f} seconds")
    print(f"Speedup: {cubic_results['init_time']/spherical_results['init_time']:.2f}x")

    print(f"\nMemory usage:")
    print(f"Spherical: {spherical_results['memory_used']:.2f} MB")
    print(f"Cubic: {cubic_results['memory_used']:.2f} MB")
    print(
        f"Memory saved: {cubic_results['memory_used'] - spherical_results['memory_used']:.2f} MB"
    )

    # Test operations
    spherical_ops = measure_operations(spherical_results["lattice"], "Spherical")
    cubic_ops = measure_operations(cubic_results["lattice"], "Cubic")

    # Compare memory efficiency
    compare_memory_efficiency(spherical_results["stats"], cubic_results["stats"])

    # Detailed statistics
    print("\n" + "=" * 60)
    print("Detailed Statistics")
    print("=" * 60)

    print("\nSpherical Implementation:")
    print(f"  Total granules: {spherical_results['stats']['total_granules']:,}")
    print(f"  Unit cell edge: {spherical_results['stats']['unit_cell_edge']:.2e} m")
    print(f"  Universe radius: {spherical_results['stats']['universe_radius']:.2e} m")
    if "efficiency" in spherical_results["stats"]:
        print(f"  Space efficiency: {spherical_results['stats']['efficiency']:.1f}%")

    print("\nCubic Implementation:")
    print(f"  Total granules: {cubic_results['stats']['total_granules']:,}")
    print(f"  Unit cell edge: {cubic_results['stats']['unit_cell_edge']:.2e} m")
    print(f"  Universe edge: {cubic_results['stats']['universe_edge']:.2e} m")

    # Performance ratio summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    granule_ratio = (
        spherical_results["stats"]["total_granules"] / cubic_results["stats"]["total_granules"]
    )
    print(f"Granule count ratio (spherical/cubic): {granule_ratio:.3f}")
    print(f"Computational savings: {(1-granule_ratio)*100:.1f}%")

    if "position_update_100x" in spherical_ops and "position_update_100x" in cubic_ops:
        update_ratio = spherical_ops["position_update_100x"] / cubic_ops["position_update_100x"]
        print(f"Position update speed ratio: {update_ratio:.3f}")
        print(f"Update performance gain: {(1-update_ratio)*100:.1f}%")

    # Kernel profiler results
    print("\n" + "=" * 60)
    print("Taichi Kernel Profiler Results")
    print("=" * 60)
    ti.profiler.print_kernel_profiler_info()

    return spherical_results, cubic_results


if __name__ == "__main__":
    spherical, cubic = main()
