# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenWave is an open-source quantum physics simulator implementing Energy Wave Theory (EWT) to model the formation of matter and energy from quantum wave interactions. The project simulates phenomena from spacetime emergence through particle formation to complex matter behavior.

### Project Goals

To develop OpenWave, an open-source computer simulator with objectives described in the text below, based on the papers attached as 9 files, with special attention to the file `Relationship of the Speed of Light to Aether Density` where there is a Planck mass correction from previous papers (affecting granule mass), using the `a6. Constants and Equations - Waves.pdf` as constants reference, built in phases. Simulation physics, constants, and equations will be drawn from the EWT research papers located at the `/requirements` folder. For performance on the granular physics simulations we'll be using the Taichi Lang python library.

### What is OpenWave?

- Refer to `README.md` for a detailed descriptions and scope of OpenWave.

### Known Challenges & Limitations

#### Blender Limitation Lessons

- Previous QSCOPE experiments showed that Blender’s physics engine was not suitable for Planck-scale simulation (wave modifier limits, animation-focused, partial physics, limited parallel processing).
- This project will use a dedicated physics computational backend, independent of 3D modeling software.

#### Granularity vs. Performance

- Full Planck-scale fidelity may be computationally prohibitive; require user-tunable resolution.

## Installation

- Refer to `README.md` for installation guidance of OpenWave.

## Project Architecture

### Modules Structure and Objects Map

- Refer to `OBJECTS.md` file for the Modules Structure, Objects Map and System Architecture.

## CLI Usage (Work in Progress)

- Refer to `README.md` for CLI usage instructions.

## Scientific Documentation & Requirements

### Project Requirements

The `requirements/requirements_source/` directory contains simulation specification documents:

1. Simulating a Fundamental Particle - EWT.pdf
2. Simulating Standalone Particles - EWT.pdf  
3. Simulating Composite Particles - EWT.pdf
4. Simulating Atoms - EWT.pdf
5. Simulating Molecules - EWT.pdf

### Scientific Source Materials

The `requirements/scientific_source/` directory contains foundational EWT research papers:

- a1. The Geometry of Spacetime and the Unification of Forces v2.3.pdf
- a2. The Geometry of Particles and the Explanation of Their Creation and Decay v2.pdf
- a3. The Physics of SubAtomic Particles.pdf
- a4. Relationship of the Speed of Light to Aether Density.docx (contains Planck mass correction)
- a5. The Relationship of Planck Constants and Wave Constants v2.pdf
- a6. Constants and Equations - Waves.pdf (primary constants reference)
- a7. Constants and Equations - Classical.pdf
- a8. Geometry - EWT.pdf
- a9. Mechanics - EWT.pdf

**Key Reference**: File `a6. Constants and Equations - Waves.pdf` serves as the primary constants reference for the simulation, and `a4. Relationship of the Speed of Light to Aether Density` contains important Planck mass corrections that affect granule mass calculations.

## Physics Context

This project implements Energy Wave Theory concepts:

- Quantum Waves as fundamental building blocks
- Wave interactions forming particles and matter
- Simulation from Planck scale to macroscopic phenomena
- Validation against experimental observations
