"""
Performance profiling script for quantum_space.py
Compares CPU vs GPU performance across different stages of execution.
"""

import time
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import gc
import psutil
import os

# Import the quantum space modules
import sys

sys.path.append("/Users/xrodz/Documents/source-code/OPENWAVE-LABS/openwave")
from openwave.spacetime.quantum_space import Lattice, Granule
import openwave.core.config as config
import openwave.core.constants as constants


class PerformanceProfiler:
    """Profile performance of quantum space simulation on CPU vs GPU."""

    def __init__(self, universe_sizes: List[float] = None, resolutions: List[int] = None):
        """
        Initialize profiler with test parameters.

        Args:
            universe_sizes: List of universe edge sizes in meters
            resolutions: List of resolution values (total granules)
        """
        self.universe_sizes = universe_sizes or [1e-16, 5e-17, 1e-17]
        self.resolutions = resolutions or [10000, 50000, 100000, 500000]
        self.results = {}

    def profile_stage(self, func, *args, **kwargs) -> Tuple[float, any]:
        """
        Profile a single function execution.

        Returns:
            Tuple of (execution_time, result)
        """
        # Force garbage collection before timing
        gc.collect()

        # Warm-up run (for GPU especially)
        if hasattr(ti, "sync"):
            ti.sync()

        # Time the actual execution
        start = time.perf_counter()
        result = func(*args, **kwargs)

        # Ensure GPU operations complete
        if hasattr(ti, "sync"):
            ti.sync()

        end = time.perf_counter()

        return end - start, result

    def profile_initialization(
        self, arch: str, universe_edge: float, resolution: int
    ) -> Dict[str, float]:
        """
        Profile the initialization phase (Taichi init + Lattice creation).
        """
        times = {}

        # Time Taichi initialization
        ti.reset()
        start = time.perf_counter()
        if arch == "cpu":
            ti.init(arch=ti.cpu)
        else:
            ti.init(arch=ti.gpu)
        end = time.perf_counter()
        times["taichi_init"] = end - start

        # Override config resolution for testing
        original_res = config.QSPACE_RES
        config.QSPACE_RES = resolution

        # Time lattice creation (includes population)
        start = time.perf_counter()
        lattice = Lattice(universe_edge)
        if hasattr(ti, "sync"):
            ti.sync()
        end = time.perf_counter()
        times["lattice_creation"] = end - start

        # Restore original resolution
        config.QSPACE_RES = original_res

        return times, lattice

    def profile_normalization(self, lattice: Lattice) -> float:
        """
        Profile position normalization kernel.
        """
        # Create normalized positions field
        normalized_positions = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)

        @ti.kernel
        def normalize_positions():
            for i in range(lattice.total_granules):
                normalized_positions[i] = lattice.positions[i] / lattice.universe_edge

        # Profile normalization
        time_taken, _ = self.profile_stage(normalize_positions)
        return time_taken

    def profile_position_update(self, lattice: Lattice, iterations: int = 100) -> float:
        """
        Profile position update kernel (simulating wave motion).
        """

        # Initialize some random velocities for testing
        @ti.kernel
        def init_velocities():
            for i in lattice.velocities:
                # Small random velocities
                lattice.velocities[i] = ti.Vector(
                    [ti.random() * 0.1 - 0.05, ti.random() * 0.1 - 0.05, ti.random() * 0.1 - 0.05]
                )

        init_velocities()
        if hasattr(ti, "sync"):
            ti.sync()

        # Profile multiple position updates
        dt = 0.001
        start = time.perf_counter()
        for _ in range(iterations):
            lattice.update_positions(dt)
        if hasattr(ti, "sync"):
            ti.sync()
        end = time.perf_counter()

        return (end - start) / iterations  # Average time per update

    def profile_memory_transfer(self, lattice: Lattice) -> Dict[str, float]:
        """
        Profile memory transfer operations (GPU<->CPU).
        """
        times = {}

        # Profile GPU to CPU transfer
        start = time.perf_counter()
        positions_numpy = lattice.positions.to_numpy()
        end = time.perf_counter()
        times["gpu_to_cpu"] = end - start

        # Profile CPU to GPU transfer (if we modify and write back)
        positions_numpy += 0.001  # Small modification
        start = time.perf_counter()
        lattice.positions.from_numpy(positions_numpy)
        if hasattr(ti, "sync"):
            ti.sync()
        end = time.perf_counter()
        times["cpu_to_gpu"] = end - start

        return times

    def run_comprehensive_profile(self) -> Dict:
        """
        Run comprehensive profiling across different configurations.
        """
        results = {"cpu": {}, "gpu": {}}

        # Test different resolutions with fixed universe size
        universe_edge = 1e-16

        for arch in ["cpu", "gpu"]:
            print(f"\n{'='*50}")
            print(f"Profiling {arch.upper()} Performance")
            print(f"{'='*50}")

            arch_results = {
                "resolutions": [],
                "init_times": [],
                "lattice_times": [],
                "normalize_times": [],
                "update_times": [],
                "memory_transfer": [],
            }

            for resolution in self.resolutions:
                print(f"\nTesting resolution: {resolution:,} granules")

                try:
                    # Profile initialization
                    init_times, lattice = self.profile_initialization(
                        arch, universe_edge, resolution
                    )

                    # Profile normalization
                    norm_time = self.profile_normalization(lattice)

                    # Profile position updates
                    update_time = self.profile_position_update(lattice)

                    # Profile memory transfers (only relevant for GPU)
                    if arch == "gpu":
                        transfer_times = self.profile_memory_transfer(lattice)
                    else:
                        transfer_times = {"gpu_to_cpu": 0, "cpu_to_gpu": 0}

                    # Store results
                    arch_results["resolutions"].append(resolution)
                    arch_results["init_times"].append(init_times["taichi_init"])
                    arch_results["lattice_times"].append(init_times["lattice_creation"])
                    arch_results["normalize_times"].append(norm_time)
                    arch_results["update_times"].append(update_time)
                    arch_results["memory_transfer"].append(transfer_times)

                    # Print summary
                    print(f"  Taichi init: {init_times['taichi_init']:.4f}s")
                    print(f"  Lattice creation: {init_times['lattice_creation']:.4f}s")
                    print(f"  Normalization: {norm_time:.6f}s")
                    print(f"  Position update (avg): {update_time:.6f}s")
                    if arch == "gpu":
                        print(f"  GPU→CPU transfer: {transfer_times['gpu_to_cpu']:.6f}s")
                        print(f"  CPU→GPU transfer: {transfer_times['cpu_to_gpu']:.6f}s")

                    # Get memory usage
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    print(f"  Memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")

                except Exception as e:
                    print(f"  Error testing resolution {resolution}: {e}")
                    continue

                # Reset Taichi for next iteration
                ti.reset()

            results[arch] = arch_results

        self.results = results
        return results

    def analyze_performance(self) -> Dict:
        """
        Analyze performance differences between CPU and GPU.
        """
        if not self.results:
            print("No results to analyze. Run profiling first.")
            return {}

        analysis = {}

        cpu_results = self.results.get("cpu", {})
        gpu_results = self.results.get("gpu", {})

        if not cpu_results or not gpu_results:
            print("Incomplete results. Need both CPU and GPU profiles.")
            return {}

        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Compare at each resolution
        for i, resolution in enumerate(cpu_results["resolutions"]):
            if i >= len(gpu_results["resolutions"]):
                break

            print(f"\nResolution: {resolution:,} granules")
            print("-" * 40)

            # Calculate speedups
            lattice_speedup = cpu_results["lattice_times"][i] / gpu_results["lattice_times"][i]
            norm_speedup = cpu_results["normalize_times"][i] / gpu_results["normalize_times"][i]
            update_speedup = cpu_results["update_times"][i] / gpu_results["update_times"][i]

            print(f"Lattice creation:")
            print(f"  CPU: {cpu_results['lattice_times'][i]:.4f}s")
            print(f"  GPU: {gpu_results['lattice_times'][i]:.4f}s")
            print(f"  Speedup: {lattice_speedup:.2f}x")

            print(f"Normalization:")
            print(f"  CPU: {cpu_results['normalize_times'][i]:.6f}s")
            print(f"  GPU: {gpu_results['normalize_times'][i]:.6f}s")
            print(f"  Speedup: {norm_speedup:.2f}x")

            print(f"Position update:")
            print(f"  CPU: {cpu_results['update_times'][i]:.6f}s")
            print(f"  GPU: {gpu_results['update_times'][i]:.6f}s")
            print(f"  Speedup: {update_speedup:.2f}x")

            # Memory transfer overhead (GPU only)
            if "memory_transfer" in gpu_results:
                transfer = gpu_results["memory_transfer"][i]
                print(f"Memory transfer overhead (GPU):")
                print(f"  GPU→CPU: {transfer['gpu_to_cpu']:.6f}s")
                print(f"  CPU→GPU: {transfer['cpu_to_gpu']:.6f}s")

            analysis[resolution] = {
                "lattice_speedup": lattice_speedup,
                "norm_speedup": norm_speedup,
                "update_speedup": update_speedup,
            }

        return analysis

    def plot_results(self):
        """
        Create visualization plots of the profiling results.
        """
        if not self.results:
            print("No results to plot. Run profiling first.")
            return

        cpu_results = self.results.get("cpu", {})
        gpu_results = self.results.get("gpu", {})

        if not cpu_results or not gpu_results:
            print("Incomplete results. Need both CPU and GPU profiles.")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("CPU vs GPU Performance Comparison", fontsize=16)

        resolutions = cpu_results["resolutions"]

        # Plot 1: Lattice Creation Time
        ax1 = axes[0, 0]
        ax1.plot(resolutions, cpu_results["lattice_times"], "b-o", label="CPU")
        ax1.plot(resolutions, gpu_results["lattice_times"], "r-o", label="GPU")
        ax1.set_xlabel("Number of Granules")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_title("Lattice Creation Time")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Normalization Time
        ax2 = axes[0, 1]
        ax2.plot(resolutions, cpu_results["normalize_times"], "b-o", label="CPU")
        ax2.plot(resolutions, gpu_results["normalize_times"], "r-o", label="GPU")
        ax2.set_xlabel("Number of Granules")
        ax2.set_ylabel("Time (seconds)")
        ax2.set_title("Normalization Kernel Time")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Position Update Time
        ax3 = axes[1, 0]
        ax3.plot(resolutions, cpu_results["update_times"], "b-o", label="CPU")
        ax3.plot(resolutions, gpu_results["update_times"], "r-o", label="GPU")
        ax3.set_xlabel("Number of Granules")
        ax3.set_ylabel("Time (seconds)")
        ax3.set_title("Position Update Time (per iteration)")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 4: Speedup Ratios
        ax4 = axes[1, 1]
        lattice_speedups = [
            cpu_results["lattice_times"][i] / gpu_results["lattice_times"][i]
            for i in range(len(resolutions))
        ]
        norm_speedups = [
            cpu_results["normalize_times"][i] / gpu_results["normalize_times"][i]
            for i in range(len(resolutions))
        ]
        update_speedups = [
            cpu_results["update_times"][i] / gpu_results["update_times"][i]
            for i in range(len(resolutions))
        ]

        ax4.plot(resolutions, lattice_speedups, "g-o", label="Lattice Creation")
        ax4.plot(resolutions, norm_speedups, "m-o", label="Normalization")
        ax4.plot(resolutions, update_speedups, "c-o", label="Position Update")
        ax4.axhline(y=1, color="k", linestyle="--", alpha=0.5)
        ax4.set_xlabel("Number of Granules")
        ax4.set_ylabel("GPU Speedup (CPU time / GPU time)")
        ax4.set_title("GPU Speedup vs CPU")
        ax4.set_xscale("log")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        plt.savefig("quantum_space_performance.png", dpi=150)
        plt.show()

        print("\nPerformance plots saved to 'quantum_space_performance.png'")


def main():
    """
    Main profiling execution.
    """
    print("=" * 60)
    print("QUANTUM SPACE PERFORMANCE PROFILER")
    print("=" * 60)

    # Create profiler with test configurations
    profiler = PerformanceProfiler(
        resolutions=[100000, 500000, 1000000, 5000000, 10000000]  # Different granule counts
    )

    # Run comprehensive profiling
    results = profiler.run_comprehensive_profile()

    # Analyze results
    analysis = profiler.analyze_performance()

    # Create visualization plots
    profiler.plot_results()

    # Print key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    print(
        """
Why CPU might feel faster than GPU for current implementation:

1. **Small Problem Size**: With only 10,000-100,000 granules, the problem may be
   too small to fully utilize GPU parallelization. GPUs excel with millions of elements.

2. **Memory Transfer Overhead**: For visualization, data must transfer from GPU to CPU
   every frame. This overhead can dominate for small datasets.

3. **Kernel Launch Overhead**: GPU kernel launches have fixed overhead (~10-100 microseconds).
   For simple operations, this overhead may exceed computation time.

4. **Low Arithmetic Intensity**: Current kernels (population, normalization) are memory-bound
   with simple operations. GPUs prefer compute-intensive workloads.

5. **Static Scene**: Without wave motion computation, there's minimal parallel work.
   The GPU is underutilized rendering static positions.

Recommendations for GPU optimization:
- Increase granule count (millions for GPU efficiency)
- Implement wave physics calculations (high arithmetic intensity)
- Batch multiple operations in single kernels
- Minimize CPU-GPU memory transfers
- Use GPU for entire render pipeline if possible
"""
    )

    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
