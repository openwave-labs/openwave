# System Upgrade Options for OpenWave

Last Updated: November 2025

## Executive Summary

This document evaluates hardware options for scaling OpenWave physics simulations beyond the current 350M voxel limit.

### Current State

| Metric | Value |
|--------|-------|
| System | M4 Max MacBook Pro 48GB |
| Tested Max | ~350M voxels (90% GPU, 56% memory, fan active) |
| Limitation | Wave propagation fails above 350M (cause unknown - needs testing on other hardware) |
| Framework | Taichi Lang (single-GPU only, no native multi-GPU support) |

### Key Constraints

1. **Taichi single-GPU limit**: Multi-GPU requires MPI refactoring (2-3 months effort)
2. **350M voxel barrier**: Cause unknown - could be hardware saturation, thermal throttling, or backend limits
3. **Memory scaling**: 1B voxels needs ~44GB total, 10B needs ~336GB

### Primary Recommendation

#### Wait for Apple M5 Ultra (expected 2026)

| Factor | Benefit |
|--------|---------|
| Compute | ~50-60 TFLOPS (3x current M4 Max) |
| Memory | Up to 512GB unified (enables 10B voxels) |
| Bandwidth | ~1000 GB/s (1.8x current) |
| Ecosystem | No code changes needed (stays on Metal) |
| Price | ~$5,500 base (similar to M3 Ultra 80-core) |

**Why wait?** M4 Ultra appears skipped. M5 Ultra will offer ~40-60% more compute than M3 Ultra at similar price. Current M4 Max is adequate for development.

### Alternative: If 350M Limit is Hardware-Specific

If testing on beefier hardware (M3 Ultra, RTX 5090) shows the 350M limit is specific to M4 Max 48GB thermal/compute saturation, then:

- **M3 Ultra 80-core (512GB)** becomes viable now for 1B+ voxels
- **RTX 5090 Linux system** could test CUDA backend (but only 32GB VRAM)

**Action needed**: Test OpenWave on different hardware to isolate the 350M voxel limit cause.

---

## OpenWave Memory Requirements

Based on analysis of `L1_field_grid.py` and `L1_wave_engine.py`, OpenWave requires the following ti.fields per voxel:

| Field | Type | Bytes | Purpose |
|-------|------|-------|---------|
| `psiL_new_am` | f32 | 4 | Wave displacement at t+dt |
| `psiL_am` | f32 | 4 | Wave displacement at t |
| `psiL_old_am` | f32 | 4 | Wave displacement at t-dt |
| `ampL_local_rms_am` | f32 | 4 | Longitudinal amplitude tracker |
| `last_crossing` | f32 | 4 | Zero-crossing detection |
| `freq_local_cross_rHz` | f32 | 4 | Local frequency tracker |
| **Total** | | **24 bytes/voxel** | |

### Memory Requirements by Voxel Count

| Voxels | Field Data | Overhead* | Total Needed | Recommended RAM |
|--------|------------|-----------|--------------|-----------------|
| **350M** (current max) | 7.8 GB | 15.5 GB | 23.3 GB | **48 GB** |
| 500M | 11.2 GB | 17.0 GB | 28.2 GB | **64 GB** |
| 700M | 15.6 GB | 19.0 GB | 34.7 GB | **64 GB** |
| **1B** (near-term goal) | 22.4 GB | 22.0 GB | 44.4 GB | **64-96 GB** |
| 3B | 67.1 GB | 42.2 GB | 109.2 GB | **128-256 GB** |
| **10B** (future goal) | 223.5 GB | 112.6 GB | 336.1 GB | **512 GB** |

*Overhead includes: Taichi runtime + JIT (~2GB), Metal driver overhead (~40% of field data), macOS + background apps (~10GB).

### Current Practical Limits (Tested on M4 Max 48GB)

**Observed metrics at different voxel counts:**

| Experiment | Voxels | GPU Util | Memory Pressure | Behavior |
|------------|--------|----------|-----------------|----------|
| our_queen | 100M | ~70% | ~44% | Correct wave propagation |
| max_res | 350M | ~90% | ~56% | Correct wave propagation, fan active |
| (tested) | 400M+ | ? | ? | **Wave scatters incorrectly** |

**Key Finding**: Above ~350M voxels, wave propagation becomes incorrect (energy scatters to unexpected locations, simulation no longer physically accurate).

**Cause: Unknown.** Possible factors (not yet tested):

1. **GPU compute saturation** - 90%+ utilization may cause race conditions
2. **Thermal throttling** - GPU reducing clocks under sustained load
3. **Memory pressure/leaks** - Accumulating issues at higher loads
4. **Metal backend limits** - Possible buffer or dispatch limits (unconfirmed)
5. **Numerical precision** - FP32 accumulation errors (unconfirmed)
6. **Hardware-specific** - May work on beefier systems

**To determine the actual cause**: Test on a different system (M3 Ultra, M4 Max 128GB, or NVIDIA with CUDA) to isolate whether this is a hardware limit or software/backend issue.

### Minimum System Requirements

| Use Case | Min RAM | Min TFLOPS | Bottleneck |
|----------|---------|------------|------------|
| Development (current) | 48 GB | 18 TFLOPS | Compute (90% GPU) |
| Near-term (1B voxels) | 96 GB | ~35 TFLOPS | Both |
| Production (3B voxels) | 256 GB | ~50 TFLOPS | Both |
| Future (10B voxels) | 512 GB | ~60+ TFLOPS | Memory first |

---

## Current System: M4 Max MacBook Pro

- **CPU**: 16-core (12 performance + 4 efficiency)
- **GPU**: 40-core Apple Silicon GPU
- **FP32 Compute**: 18.4 TFLOPS
- **Memory**: 48GB unified @ 546 GB/s bandwidth
- **Max Voxels**: ~350M tested (limited by Metal overhead + macOS apps)
- **Status**: Owned (baseline for comparison)

---

## Critical Limitation: Taichi Single-GPU Constraint

Per [GitHub issue #7664](https://github.com/taichi-dev/taichi/issues/7664), Taichi Lang does not support multi-GPU execution. Users attempting `CUDA_VISIBLE_DEVICES` or `TI_VISIBLE_DEVICE` found only one GPU is utilized.

**Implications:**

- Multi-GPU systems (2x RTX 5090, DGX Station) cannot be utilized without code refactoring
- Must choose single-GPU systems OR invest in MPI/domain decomposition rewrite
- Taichi focuses on maximizing single-GPU performance (demonstrated with 1 billion particle MPM simulation)

---

## Hardware Options Comparison

### Apple Silicon (macOS - Native Metal Backend)

| System | CPU | GPU Cores | FP32 TFLOPS | Memory | Bandwidth | Price | Max Voxels |
|--------|-----|-----------|-------------|--------|-----------|-------|------------|
| **M4 Max 40-core** (owned) | 16-core | 40 | 18.4 | 48GB | 546 GB/s | Owned | ~350M |
| M4 Max 64GB | 16-core | 40 | 18.4 | 64GB | 546 GB/s | ~$3,600 | ~500M |
| M4 Max 128GB | 16-core | 40 | 18.4 | 128GB | 546 GB/s | ~$4,000 | ~1.5B |
| **M3 Ultra 60-core** | 28-core | 60 | ~27 | 96-512GB | 819 GB/s | $3,999 | ~700M-10B |
| **M3 Ultra 80-core** | 32-core | 80 | ~36 | 96-512GB | 819 GB/s | $5,499+ | ~700M-10B |
| **M5 Ultra** (est. 2026) | ~36-core | ~100 | ~50-60 | 128-512GB+ | ~1000 GB/s | ~$5,000-6,000 | ~1.5B-10B+ |

#### Mac Studio M3 Ultra 60-Core ($3,999)

- **FP32**: ~27 TFLOPS (1.5x M4 Max)
- **Memory**: 96GB standard, up to 512GB
- **Bandwidth**: 819 GB/s (1.5x your current)
- **Max Voxels**: ~10 billion with 512GB config
- **Best for**: Large universe simulations requiring >48GB memory
- **Pros**: No code changes, stays in Metal ecosystem, massive memory capacity
- **Cons**: Only ~1.5x compute improvement, M3 generation (not latest)

#### Mac Studio M3 Ultra 80-Core ($5,499+)

- **FP32**: ~36 TFLOPS (2x M4 Max)
- **Memory**: 96GB standard, up to 512GB @ 819 GB/s
- **Max Voxels**: ~10 billion with 512GB config
- **Best for**: Maximum current Apple Silicon performance
- **Configuration options**:
  - 256GB + 4TB SSD: ~$8,099
  - 512GB + 16TB SSD: ~$14,099 (fully maxed)
- **Pros**: 2x GPU cores vs M4 Max, can run 600B+ parameter LLMs
- **Cons**: Diminishing returns on price/performance, M3 generation

#### Apple M5 Ultra (Expected 2026) - RECOMMENDED WAIT

- **FP32**: ~50-60 TFLOPS (estimated, ~2.7-3.3x M4 Max)
- **Memory**: Expected 128-512GB+ @ ~1000 GB/s
- **Max Voxels**: ~10B+ with high-memory config
- **Expected Price**: ~$5,000-6,000 base
- **Status**: Per [Bloomberg's Mark Gurman](https://www.macrumors.com/2024/11/26/when-to-expect-m4-macbook-air-mac-studio-mac-pro-2/), M5 Ultra Mac Studio is on Apple's 2026 release schedule
- **Why Wait**:
  - M4 Ultra appears to be skipped (March 2025 Mac Studio shipped with M3 Ultra, not M4 Ultra)
  - M5 Ultra will combine two M5 Max dies via UltraFusion
  - Expected ~40% improvement over M4 Max per core, doubled via Ultra fusion
  - Better price/performance than buying M3 Ultra now
- **Pros**: Latest architecture, best compute + memory combination, no code changes
- **Cons**: ~1 year wait, specifications not confirmed

**M5 Ultra Estimated Specifications** (extrapolated from M-series trends):

| Spec | M4 Max | M3 Ultra 80-core | M5 Ultra (est.) |
|------|--------|------------------|-----------------|
| CPU Cores | 16 | 32 | ~36-40 |
| GPU Cores | 40 | 80 | ~100-120 |
| FP32 TFLOPS | 18.4 | ~36 | ~50-60 |
| Max Memory | 128GB | 512GB | 512GB+ |
| Bandwidth | 546 GB/s | 819 GB/s | ~1000 GB/s |
| Process | 3nm | 3nm | 3nm (2nd gen) |

### NVIDIA Desktop (Linux/Windows - CUDA Backend)

| System | FP32 TFLOPS | Memory | Bandwidth | TDP | Price | Max Voxels |
|--------|-------------|--------|-----------|-----|-------|------------|
| RTX 4090 | 82.6 | 24GB | 1,008 GB/s | 450W | ~$1,599 | ~350M |
| **RTX 5090** | 104.8 | 32GB | 1,792 GB/s | 575W | ~$1,999 | ~500M |
| RTX 5090 system | 104.8 | 32GB | 1,792 GB/s | 575W | ~$4,000-5,000 | ~500M |

#### RTX 5090 Custom System (~$4,000-5,000)

- **FP32**: 104.8 TFLOPS (5.7x your M4 Max)
- **Memory**: 32GB GDDR7 @ 1,792 GB/s (3.3x your bandwidth)
- **Max Voxels**: ~500 million (limited by 32GB VRAM)
- **Architecture**: NVIDIA Blackwell, 21,760 CUDA cores
- **Best for**: Maximum single-GPU compute performance for smaller simulations
- **Pros**: Massive compute advantage, excellent bandwidth, CUDA ecosystem
- **Cons**: Requires Linux/Windows, **32GB VRAM is limiting** (less than your 48GB unified), 575W TDP
- **Verdict**: Not recommended for OpenWave's 1B+ voxel goals due to memory constraint

### NVIDIA Compact/Edge Systems

| System | FP32 TFLOPS | Memory | Bandwidth | TDP | Price | Max Voxels |
|--------|-------------|--------|-----------|-----|-------|------------|
| **DGX Spark GB10** | 31 | 128GB | 273 GB/s | ~100W | $3,999 | ~3B |
| Jetson AGX Thor | 7.8 | 128GB | 273 GB/s | 40-130W | $2,799-3,499 | ~3B |

#### NVIDIA DGX Spark ($3,999) - NOT RECOMMENDED

- **FP32**: 31 TFLOPS (1.7x your M4 Max)
- **Memory**: 128GB unified @ 273 GB/s
- **Max Voxels**: ~3 billion
- **AI Performance**: 1 PFLOP sparse FP4 (for inference)
- **Form Factor**: 150 × 150 × 50.5 mm (desktop)
- **Networking**: 200 Gbps ConnectX-7 (two can be linked)
- **OS**: NVIDIA DGX OS (Linux)
- **Requires**: Switching OpenWave from Metal to CUDA backend (code changes, not MPI)
- **Verdict**: **Not recommended** - M5 Ultra will offer ~2x compute, ~4x bandwidth, more memory, and no code changes at similar price. Only consider if you urgently need >48GB memory before 2026.

#### NVIDIA Jetson AGX Thor ($2,799-3,499)

- **FP32**: 7.8 TFLOPS (SLOWER than M4 Max!)
- **Memory**: 128GB LPDDR5X @ 273 GB/s
- **AI Performance**: 2070 TFLOPS FP4 sparse (for AI inference only)
- **Target**: Robotics, edge AI, humanoid robots
- **Verdict**: **NOT RECOMMENDED for OpenWave** - the impressive 2070 TFLOPS is FP4 for neural network inference, not FP32 physics simulation

### AMD Alternatives (ROCm Backend)

| System | FP32 TFLOPS | Memory | Bandwidth | TDP | Price | Max Voxels |
|--------|-------------|--------|-----------|-----|-------|------------|
| **Instinct MI300X** | 163.4 | 192GB | 5,300 GB/s | 750W | ~$15,000+ | ~5B |
| Radeon RX 7900 XTX | 61.4 | 24GB | 960 GB/s | 355W | ~$900 | ~350M |

#### AMD Instinct MI300X (Data Center)

- **FP32**: 163.4 TFLOPS (8.9x your M4 Max)
- **FP64**: 81.7 TFLOPS (excellent for scientific computing)
- **Memory**: 192GB HBM3 @ 5,300 GB/s (9.7x your bandwidth!)
- **Max Voxels**: ~5 billion
- **Architecture**: CDNA 3, 153 billion transistors
- **Best for**: HPC, scientific simulation, quantum chemistry
- **Taichi Support**: AMD validates Taichi on MI250X/MI300X via [ROCm](https://rocm.blogs.amd.com/artificial-intelligence/taichi/README.html)
- **Pros**: Exceptional FP64 for scientific computing, massive memory + bandwidth
- **Cons**: Data center hardware, 750W TDP, ~$15,000+ cost, enterprise deployment

#### AMD Radeon RX 7900 XTX (Consumer)

- **FP32**: 61.4 TFLOPS (3.3x your M4 Max)
- **Memory**: 24GB GDDR6 @ 960 GB/s
- **Max Voxels**: ~350 million (limited by 24GB VRAM)
- **Price**: ~$900
- **Taichi Support**: Via Vulkan backend (not ROCm for consumer cards)
- **Pros**: Good price/performance ratio
- **Cons**: Less mature Taichi support than CUDA, **24GB VRAM is very limiting**

---

## Performance Comparison Summary

| System | FP32 TFLOPS | vs M4 Max | Memory | Bandwidth | Max Voxels | Price | Taichi Backend |
|--------|-------------|-----------|--------|-----------|------------|-------|----------------|
| **M4 Max 48GB** (yours) | 18.4 | baseline | 48GB | 546 GB/s | ~350M | Owned | Metal |
| M4 Max 128GB | 18.4 | 1x | 128GB | 546 GB/s | ~1.5B | ~$4,000 | Metal |
| M3 Ultra 60-core | ~27 | 1.5x | 96-512GB | 819 GB/s | ~700M-10B | $3,999 | Metal |
| M3 Ultra 80-core | ~36 | 2x | 96-512GB | 819 GB/s | ~700M-10B | $5,499 | Metal |
| **M5 Ultra** (est.) | ~50-60 | 2.7-3.3x | 128-512GB+ | ~1000 GB/s | ~1.5B-10B+ | ~$5,500 | Metal |
| DGX Spark GB10 | 31 | 1.7x | 128GB | 273 GB/s | ~1.5B | $3,999 | CUDA |
| RTX 5090 system | 104.8 | 5.7x | 32GB | 1,792 GB/s | ~350M | ~$5,000 | CUDA |
| Instinct MI300X | 163.4 | 8.9x | 192GB | 5,300 GB/s | ~3B | $15,000+ | ROCm |
| Jetson AGX Thor | 7.8 | 0.4x | 128GB | 273 GB/s | ~1.5B | $2,799 | CUDA |

---

## Recommendations

### Primary Recommendation: Wait for M5 Ultra (2026)

For OpenWave's goal of 1-10 billion voxel simulations, the **Apple M5 Ultra** offers the best balance:

| Factor | M5 Ultra Advantage |
|--------|-------------------|
| Compute | ~50-60 TFLOPS FP32 (2.7-3.3x current M4 Max) |
| Memory | Up to 512GB+ unified (10B+ voxels) |
| Bandwidth | ~1000 GB/s (1.8x current) |
| Ecosystem | No code changes (Metal backend) |
| Price | ~$5,500 base (competitive with M3 Ultra 80-core) |
| Timing | ~1 year wait (2026 release schedule) |

**Why not buy M3 Ultra now?**

- M5 Ultra will offer ~40-60% more compute at similar price
- Better to invest in latest architecture when making $5,000+ purchase
- Current M4 Max handles development phase adequately

### If You Can't Wait

| Need | Recommendation | Price | Tradeoff |
|------|----------------|-------|----------|
| More memory now | M3 Ultra 80-core (256GB) | ~$8,099 | Older generation, ~2x compute only |
| More compute now | RTX 5090 Linux system | ~$5,000 | Only 32GB VRAM, different OS |
| Both memory + compute | 2x DGX Spark (linked) | $7,998 | Requires MPI refactoring |
| Research-grade | AMD Instinct MI300X | $15,000+ | Enterprise deployment |

### NOT Recommended

| System | Reason |
|--------|--------|
| Jetson AGX Thor | FP32 is 7.8 TFLOPS (slower than M4 Max); designed for AI inference |
| RTX 5090 alone | Only 32GB VRAM limits to ~500M voxels; OpenWave needs 1B+ |
| RTX 4090 | Only 24GB VRAM; even more memory-limited |
| DGX Spark | Only 1.7x compute, lower bandwidth (273 vs 546 GB/s), requires CUDA backend switch; M5 Ultra is better in every metric |
| Multi-GPU setups | Taichi doesn't support multi-GPU without MPI refactoring |
| M4 Ultra | Likely skipped; Apple shipped M3 Ultra in March 2025 Mac Studio |

---

## Voxel Count Scaling Guide

| Voxels | Total RAM Needed | Suitable Systems |
|--------|------------------|------------------|
| 350M | ~23 GB | M4 Max 48GB (current), RTX 5090 32GB |
| 500M | ~28 GB | M4 Max 64GB |
| 1B | ~44 GB | M4 Max 128GB, M3 Ultra 96GB+, DGX Spark |
| 3B | ~109 GB | M3 Ultra 256GB+, M5 Ultra 256GB+ |
| 10B | ~336 GB | M3 Ultra 512GB, M5 Ultra 512GB |

---

## Future: Multi-GPU Path

If OpenWave needs to scale beyond single-GPU limits, the required changes are:

1. **Domain Decomposition**: Split universe into spatial regions
2. **MPI Communication**: For inter-GPU/inter-node coordination
3. **Ghost Zones**: Boundary data exchange between domains
4. **Estimated Effort**: 2-3 months engineering time

See `distributed_architecture_plan.md` for detailed implementation strategy.

**Recommendation**: Maximize single-GPU performance first (M5 Ultra with 512GB) before investing in multi-GPU refactoring. The engineering effort for MPI is substantial and should only be undertaken when single-GPU limits are truly reached.

### Multi-GPU Options (Requires Code Refactoring)

| System | Total FP32 | Memory | Price | Notes |
|--------|------------|--------|-------|-------|
| 2x RTX 5090 | 210 TFLOPS | 64GB | ~$10,000 | Requires MPI + domain decomposition |
| 2x DGX Spark (linked) | 62 TFLOPS | 256GB | $7,998 | 200 Gbps ConnectX-7 link |
| 4x RTX 4090 | 330 TFLOPS | 96GB | ~$12,000 | Requires NVLink/MPI setup |

---

## Additional Considerations

### Physics Simulation & HPC Best Practices

Based on [Puget Systems recommendations](https://www.pugetsystems.com/solutions/ai-and-hpc-workstations/scientific-computing/hardware-recommendations/):

- **Memory**: Plan 4-8GB per CPU core for mesh-based solvers
- **Memory Channels**: Dual AMD EPYC provides 24 memory channels for bandwidth-bound work
- **Interconnects**: NVLink, UCX, and SHMEM are critical for multi-GPU scaling
- **CPU Options**: Intel Xeon or AMD Threadripper PRO/EPYC for HPC workloads

### Taichi Backend Compatibility

| Backend | Platform | GPU Support | Multi-GPU |
|---------|----------|-------------|-----------|
| Metal | macOS | Apple Silicon | No |
| CUDA | Linux/Windows | NVIDIA | No (single GPU) |
| Vulkan | Cross-platform | NVIDIA, AMD, Intel | No |
| ROCm | Linux | AMD Instinct | No (under development) |
| OpenGL | Cross-platform | Limited | No |
| CPU | All | N/A | Multi-core only |

### Power & Thermal Considerations

| System | TDP | Notes |
|--------|-----|-------|
| M4 Max MacBook Pro | ~90W | Efficient, laptop form factor |
| Mac Studio M3 Ultra | ~200W | Desktop, quiet cooling |
| Mac Studio M5 Ultra (est.) | ~200W | Desktop, quiet cooling |
| RTX 5090 | 575W | Requires 1000W+ PSU, significant cooling |
| DGX Spark | ~100W | Compact, efficient |
| Instinct MI300X | 750W | Data center cooling required |

---

## Conclusion

### Timeline Recommendation

| Phase | Action | System |
|-------|--------|--------|
| Now - 2026 | Development & testing | M4 Max 48GB (owned) |
| 2026 | Production upgrade | M5 Ultra 256-512GB (~$8,000-10,000) |
| Future | Multi-GPU scaling | MPI refactoring + multi-node |

### Key Decision Points

1. **Can M4 Max 48GB handle 1B voxels?** Unknown. Current tested limit is ~350M voxels. The cause is unclear - could be hardware saturation (90% GPU), thermal throttling, or software limits. Testing on beefier hardware is needed.

2. **When to upgrade?** When you need:
   - More than ~350M voxels (current limit)
   - Faster iteration on large simulations (compute limit)
   - Production-ready 1B-10B voxel capability

3. **Best upgrade path?** Wait for M5 Ultra (2026) for best compute + memory combination at reasonable price. If testing shows 350M limit is hardware-specific, M3 Ultra becomes viable sooner.

4. **When to consider multi-GPU?** Only after exhausting single-GPU options (M5 Ultra 512GB) and confirming the 2-3 month MPI refactoring investment is worthwhile.

### Avoid

- **Jetson AGX Thor**: Despite impressive AI specs, FP32 performance is worse than your M4 Max
- **RTX 5090 for OpenWave**: 32GB VRAM is too limiting for 1B+ voxel goals
- **M4 Ultra**: Likely skipped by Apple; M5 Ultra is the next viable upgrade
- **Premature multi-GPU investment**: Maximize single-GPU first

---

## Sources

- [Taichi Lang Multi-GPU Issue #7664](https://github.com/taichi-dev/taichi/issues/7664)
- [NVIDIA DGX Spark Specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [NVIDIA RTX 5090 Specifications](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216)
- [Jetson AGX Thor Technical Analysis](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Thor:_Powering_the_Future_of_Physical_AI)
- [Apple Mac Studio M3 Ultra Specifications](https://www.apple.com/mac-studio/specs/)
- [Apple M3 Ultra Announcement](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/)
- [AMD Instinct MI300X Data Sheet](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [Taichi ROCm Compatibility](https://rocm.blogs.amd.com/artificial-intelligence/taichi/README.html)
- [Puget Systems Scientific Computing Recommendations](https://www.pugetsystems.com/solutions/ai-and-hpc-workstations/scientific-computing/hardware-recommendations/)
- [Apple Silicon HPC Performance Analysis](https://arxiv.org/html/2502.05317v1)
- [M5 Ultra Mac Studio 2026 Report (Bloomberg)](https://www.macrumors.com/2024/11/26/when-to-expect-m4-macbook-air-mac-studio-mac-pro-2/)
- [Apple M4 Ultra Status](https://www.macworld.com/article/2320613/new-mac-pro-ultra-release-date-specs-price-m4-m5.html)
