import taichi as ti
import time

ti.init(arch=ti.metal)


@ti.kernel
def benchmark_kernel(field: ti.template()):  # type: ignore
    for i, j in ti.ndrange(10000, 10000):
        field[i, j] = ti.sqrt(ti.cast(i * j, ti.f32))


field = ti.field(dtype=ti.f32, shape=(10000, 10000))

# Warm up
for _ in range(10):
    benchmark_kernel(field)
ti.sync()

# Benchmark
start = time.time()
for _ in range(100):
    benchmark_kernel(field)
ti.sync()
end = time.time()

print("\n===============================")
print("PERFORMANCE TEST")
print("===============================")

print(f"System Performance: {100 / (end - start):.2f} iterations/sec")
print(f"Time per iteration: {(end - start) / 100 * 1000:.2f} ms")
