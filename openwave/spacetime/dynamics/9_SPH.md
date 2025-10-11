# SPH

Reviewing paper - "Particle-Based Fluid Simulation for Interactive Applications" (the original SPH fluids paper by Müller et al., 2003). This is a classic foundational paper on SPH for computer graphics.

Let me analyze what's relevant to your XPBD implementation for the quantum lattice:

Key Analysis: Why This Paper is NOT Directly Applicable

This paper is about SPH (Smoothed Particle Hydrodynamics) for FLUIDS, which is fundamentally different from your use case:

SPH vs XPBD for Your Lattice

| Aspect           | SPH (This Paper)                     | XPBD (Your Need)                |
|------------------|--------------------------------------|---------------------------------|
| Connectivity     | Dynamic neighbors (kernel radius)    | Fixed neighbors (BCC lattice)   |
| Material Type    | Fluids (changing topology)           | Elastic solid (fixed structure) |
| Key Forces       | Pressure, viscosity, surface tension | Distance constraints            |
| Neighbor Finding | Spatial hash grid (expensive)        | Pre-computed links (free)       |
| Density          | Varies spatially (incompressibility) | Constant (lattice spacing)      |
| Best For         | Water, gases, liquids                | Cloth, springs, solids          |

Why You Should NOT Use SPH

Your quantum lattice has:

- ✅ Fixed 8 neighbors per granule (BCC connectivity)
- ✅ Constant spacing (rest length L₀)
- ✅ Elastic behavior (spring-like)
- ✅ No fluid flow or topology change

SPH requires:

- ❌ Dynamic neighbor search every frame
- ❌ Kernel smoothing over neighbors
- ❌ Density computation (ρ = Σ mⱼW(r))
- ❌ Pressure forces from density gradients
- ❌ Viscosity from velocity smoothing

Verdict: Using SPH would add massive unnecessary complexity!
