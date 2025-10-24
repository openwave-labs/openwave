# Dimensional Analysis for Physical Modeling

- Similitude Theory and Prototype Scaling in Fluid Dynamics

## Overview

Similitude theory (also called dimensional analysis) is the fundamental framework for creating scaled physical models that accurately represent full-scale prototype behavior. This is essential for testing designs in controlled environments like wave tanks or wind tunnels before building expensive full-scale prototypes.

## Core Principle

Dynamic similarity between model and prototype requires matching dimensionless numbers that govern the physics of the system. Simply scaling geometric dimensions is insufficient - the flow physics must be preserved through careful consideration of fluid properties and operating conditions.

## Key Dimensionless Numbers

### 1. Reynolds Number (Re)

**Governs:** Viscous effects, turbulence, boundary layer behavior  
**Formula:** Re = ρVL/μ  
**Variables:**

- ρ = fluid density [kg/m³]
- V = characteristic velocity [m/s]
- L = characteristic length [m]
- μ = dynamic viscosity [Pa·s]

**Physical meaning:** Ratio of inertial forces to viscous forces

### 2. Froude Number (Fr)

**Governs:** Free-surface effects, wave patterns, gravity-driven flows  
**Formula:** Fr = V/√(gL)  
**Variables:**

- V = characteristic velocity [m/s]
- g = gravitational acceleration [9.81 m/s²]
- L = characteristic length [m]

**Physical meaning:** Ratio of inertial forces to gravitational forces

### 3. Mach Number (Ma)

**Governs:** Compressibility effects, shock waves, density variations  
**Formula:** Ma = V/a  
**Variables:**

- V = flow velocity [m/s]
- a = speed of sound in fluid [m/s]
- For ideal gas: a = √(γRT) where γ = specific heat ratio, R = gas constant, T = temperature

**Physical meaning:** Ratio of flow velocity to speed of sound

## Incompressible Fluids (Water, Low-Speed Liquids)

### Wave Energy Converter Example

For marine applications like wave energy converters (WECs) in wave tanks:

**Primary concern:** Froude number similarity (gravity waves dominate)

**Scaling relationships using Froude similarity:**

- Length: Lₘ = Lₚ/λ (where λ = scale ratio)
- Time: tₘ = tₚ/√λ
- Velocity: Vₘ = Vₚ/√λ  
- Force: Fₘ = Fₚ/λ³
- Power: Pₘ = Pₚ/λ^(7/2)

**The fundamental challenge:**

- Froude scaling: V ∝ √L
- Reynolds scaling: V ∝ 1/L (for same fluid)
- **Cannot satisfy both simultaneously in the same fluid**

**Practical approach:**

1. Prioritize Froude similarity (waves/gravity dominant)
2. Ensure Reynolds number is "high enough" (Re > 10⁴) to maintain turbulent flow regime
3. Apply correction factors for viscous effects if needed
4. Typical scale ratios: 1:10 to 1:100

### Variables to Control

- Model geometry (scaled by λ)
- Wave period and height (scaled by √λ)
- Water depth (scaled by λ)
- Cannot easily change water properties (ρ, μ) in practice

## Compressible Fluids (Air, Gases)

### Additional Complexity

For aerodynamic testing and gas flows, compressibility adds significant complications:

**Must consider three dimensionless numbers:**

1. Reynolds Number (viscous effects)
2. Mach Number (compressibility effects)
3. Froude Number (only if gravity matters - rare in pure aerodynamics)

### Key Differences from Incompressible Flow

**Fluid properties become variable:**

- Density (ρ) varies with pressure and temperature: ρ = p/(RT)
- Viscosity (μ) varies with temperature: μ ∝ T^0.7 (approximately)
- Speed of sound varies: a = √(γRT)

### Scaling Conflicts

For a 1:10 scale aircraft model to maintain similarity:

- Matching Mach: need same V/a ratio → same velocity if same gas
- Matching Reynolds: need 10× higher ρV/μ ratio
- **Impossible to satisfy both in standard atmospheric conditions**

### Engineering Solutions

1. **Pressurized Wind Tunnels**
   - Increase pressure (up to 10 atm) to raise density and Reynolds number
   - Maintains Mach number at same velocity
   - Example: European Transonic Wind-tunnel (ETW)

2. **Cryogenic Wind Tunnels**
   - Lower temperature (down to -160°C using liquid nitrogen)
   - Reduces viscosity, increases Reynolds number
   - Allows independent control of Re and Ma
   - Example: NASA National Transonic Facility

3. **Alternative Test Gases**
   - Heavy gases (SF₆): Higher density, different speed of sound
   - Light gases (Helium): Lower viscosity, higher speed of sound
   - Allows manipulation of dimensionless parameters

4. **Variable Density Tunnels**
   - Adjust both pressure and temperature
   - Provides two degrees of freedom for matching similarity parameters

## Practical Scaling Methodology

### Step-by-Step Process

1. **Identify dominant physics**
   - Free surface flows → Froude number
   - High-speed flows → Mach number
   - Viscous/turbulent flows → Reynolds number

2. **Determine scale ratio (λ)**
   - Based on facility constraints
   - Typical range: 1:10 to 1:100

3. **Calculate dimensionless numbers for prototype**
   - Reₚ, Frₚ, Maₚ as applicable

4. **Design model test conditions**
   - Match most critical dimensionless number(s)
   - Ensure other numbers are in acceptable range
   - Document which similarities are compromised

5. **Apply correction factors**
   - Scale-up results using appropriate relationships
   - Account for mismatched dimensionless numbers
   - Validate with computational methods if possible

## Common Pitfalls and Limitations

1. **Scale Effects**
   - Some phenomena don't scale linearly (e.g., surface tension, cavitation)
   - Boundary layer transition may occur at different locations

2. **Facility Constraints**
   - Tank/tunnel size limits maximum model scale
   - Power limitations for flow generation
   - Measurement accuracy at small scales

3. **Cost vs. Accuracy Trade-offs**
   - Perfect similarity often prohibitively expensive
   - Must balance test objectives with practical constraints

## Summary Table: Scaling Requirements by Application

| Application | Primary Number | Secondary | Typical Scale | Key Challenge |
|------------|---------------|-----------|---------------|---------------|
| Ship hulls | Froude | Reynolds | 1:20-1:100 | Viscous drag scaling |
| Wave energy | Froude | Reynolds | 1:10-1:50 | Power take-off scaling |
| Subsonic aircraft | Reynolds | Mach | 1:10-1:50 | Achieving high Re |
| Transonic aircraft | Mach | Reynolds | 1:10-1:30 | Matching both Ma and Re |
| Wind turbines | Reynolds | Froude | 1:50-1:500 | Blade boundary layers |
| Submarines | Reynolds | - | 1:20-1:50 | Turbulence modeling |

## References for Further Reading

- Similarity and Dimensional Methods in Mechanics - L.I. Sedov
- Theory and Application of Similarity - H. Schlichting
- ITTC Guidelines for Wave Tank Testing (International Towing Tank Conference)
- AIAA Ground Testing Standards for Aerodynamic Scaling
